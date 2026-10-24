# independent_resample.py
# 等预算 Independent Resample 基线（只放逻辑，测试见 test_independent_resample.py）
# 约束（按导师要求）：
#   1) 候选选择只用“可见测试”结果，不使用最终评测测试挑选；
#   2) 选择规则预先冻结：可见通过数最高；并列取最早生成；
#   3) 选完之后，只对所选单个结果做一次统一评价；
#   4) 无独立最终测试时，明确标注 evaluation_scope = "visible_tests_only"。

import hashlib
import logging

from src.generation import generator
from problem_processor import budget_check_and_reserve


# ============================================================
# 冻结的选择规则（跑之前定死，跑之后不许改）
# ============================================================
SELECTION_RULE = "highest_visible_pass_count; tie -> earliest_generated"
EVALUATION_SCOPE = "visible_tests_only"   # 若将来有独立测试，改为 "independent_test"


def _get_hash(code_str: str) -> str:
    return hashlib.md5(code_str.encode("utf-8")).hexdigest()[:8] if code_str else "empty"


def _run_independent_resample(problem_data, args, check_correctness_func,
                              stop_when_visible_passed=True,
                              generator_func=None):
    """
    等预算 Independent Resample。

    参数：
        problem_data: 题目数据（含 complete_prompt / test）
        args: 需含 model、timeout、output_reserve、remaining_budget、max_resamples
        check_correctness_func: 可见测试评测函数 check_correctness(problem_data, code, timeout)
        stop_when_visible_passed: 冻结规则的一部分。
            True  -> 某次候选可见测试全过即停
            False -> 一直跑到预算不足或达到 max_resamples 才停
        generator_func: 可选。传替身用于无付费测试；None 时用真实 generator。
    """
    gen = generator_func if generator_func is not None else generator

    prompt = problem_data.get("complete_prompt", "")
    output_reserve = getattr(args, "output_reserve", 2000)
    max_resamples = getattr(args, "max_resamples", 50)

    candidates = []
    requests = []
    total_p, total_c = 0, 0
    budget_exhausted = False
    stopped_reason = None
    request_id = 0

    while request_id < max_resamples:
        remaining = getattr(args, "remaining_budget", None)

        # 1) 请求前预算检查：完整输入估算 + 输出预留
        if remaining is not None:
            need, ok = budget_check_and_reserve(remaining, prompt, output_reserve)
            if not ok:
                budget_exhausted = True
                stopped_reason = "Budget insufficient before resample request"
                logging.info(f"[resample] stop: need ~{need}, remaining {remaining}")
                break

        # 2) 生成：同一个 prompt，不带历史
        request_id += 1
        code, p_tokens, c_tokens = gen(
            prompt, "code3_generate", args.model,
            max_tokens_response=output_reserve,
        )

        # 3) 缺 usage：不可核验，停止
        if p_tokens is None or c_tokens is None:
            requests.append({
                "request_id": request_id,
                "prompt_tokens": None,
                "completion_tokens": None,
                "remaining_after": remaining,
                "usage_unverifiable": True,
            })
            stopped_reason = "Usage missing for resample request"
            logging.error("[resample] usage unverifiable, stopping.")
            break

        # 4) 扣账（同一账本）
        if remaining is not None:
            args.remaining_budget = remaining - (p_tokens + c_tokens)
        total_p += p_tokens
        total_c += c_tokens
        requests.append({
            "request_id": request_id,
            "prompt_tokens": p_tokens,
            "completion_tokens": c_tokens,
            "remaining_after": getattr(args, "remaining_budget", None),
        })

        # 5) 只跑“可见测试”评测，用于候选选择
        visible_eval = check_correctness_func(problem_data, code, args.timeout)
        candidates.append({
            "request_id": request_id,
            "code": code,
            "visible_eval": visible_eval,
            "code_hash": _get_hash(code),
        })

        # 6) 冻结规则：可见全过即停
        if stop_when_visible_passed and visible_eval.get("passed"):
            stopped_reason = "All visible tests passed"
            break

    if stopped_reason is None:
        if request_id >= max_resamples:
            stopped_reason = "Max resamples reached"
        else:
            stopped_reason = "Stopped"

    # 7) 候选选择：只用可见测试结果；并列取最早
    best = None
    if candidates:
        best = min(
            candidates,
            key=lambda c: (
                -c["visible_eval"].get("passed_count", 0),
                c["request_id"],
            ),
        )

    # 8) 最终评价：只对选中的那一个做一次
    if best is not None:
        final_eval = check_correctness_func(problem_data, best["code"], args.timeout)
    else:
        final_eval = {"passed": False, "passed_count": 0, "failed_count": 0,
                      "total_count": 0, "passed_tests": [], "failed_tests": []}

    # 9) 逐候选完整记录：代码全文、完整评测、是否选中、选择理由
    candidates_out = []
    for c in candidates:
        is_selected = (best is not None and c["request_id"] == best["request_id"])
        candidates_out.append({
            "request_id": c["request_id"],
            "code_hash": c["code_hash"],
            "visible_passed_count": c["visible_eval"].get("passed_count", 0),
            "visible_passed": c["visible_eval"].get("passed", False),
            "code": c["code"],
            "eval_result": c["visible_eval"],
            "selected": is_selected,
            "selection_reason": (
                SELECTION_RULE if is_selected else ""
            ),
        })

    return {
        "method": "independent_resample",
        "final_code": best["code"] if best else "",
        "final_code_hash": best["code_hash"] if best else "empty",
        "final_eval": final_eval,
        "final_visible_passed_count": final_eval.get("passed_count", 0),
        "candidates": candidates_out,
        "requests": requests,
        "num_requests": len(requests),
        "stopped_reason": stopped_reason,
        "prompt_tokens": total_p,
        "completion_tokens": total_c,
        "budget_exhausted": budget_exhausted,
        "budget_remaining": getattr(args, "remaining_budget", None),
        "selection_rule": SELECTION_RULE,
        "evaluation_scope": EVALUATION_SCOPE,
    }