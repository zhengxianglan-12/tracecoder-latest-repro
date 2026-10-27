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
from problem_processor import budget_check_and_reserve, reconcile_budget_after_request


# ============================================================
# 冻结的选择规则（跑之前定死，跑之后不许改）
# ============================================================
SELECTION_RULE = "highest_visible_pass_count; tie -> earliest_generated"
EVALUATION_SCOPE = "visible_tests_only"   # 若将来有独立测试，改为 "independent_test"
# 超限候选策略：两种方法一致
# kept_but_excluded_from_selection: 保留在 candidates 里并标记，但不参与最终选择
OVERRUN_CANDIDATE_POLICY = "kept_but_excluded_from_selection"


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
    # 用于在拒绝事件里给出唯一编号，不占用 request_id
    rejected_id = 0

    while request_id < max_resamples:
        remaining = getattr(args, "remaining_budget", None)

        
        # 1) 请求前预算检查：完整输入估算 + 输出预留
        need = None
        if remaining is not None:
            need, ok = budget_check_and_reserve(remaining, prompt, output_reserve)
            if not ok:
                budget_exhausted = True
                stopped_reason = "Budget insufficient before resample request"
                rejected_id += 1
                requests.append({
                    "request_id": f"RJ{rejected_id:02d}",
                    "stage": "skipped_budget_resample",
                    "sent": False,
                    "estimated_input": need - output_reserve,
                    "output_reserve": output_reserve,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "remaining_before": remaining,
                    "remaining_after": remaining,
                    "overrun_tokens": 0,
                    "budget_overrun": False,
                    "usage_unverifiable": False,
                    "stopped_reason": stopped_reason,
                })
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
                "request_id": f"R{request_id:02d}",
                "stage": "resample",
                "sent": True,
                "estimated_input": (need - output_reserve) if need is not None else None,
                "output_reserve": output_reserve,
                "prompt_tokens": None,
                "completion_tokens": None,
                "remaining_before": remaining,
                "remaining_after": remaining,        # 没核账，余额不变
                "overrun_tokens": None,              # 未知，不是 0
                "budget_overrun": None,              # 未知，不是 False
                "usage_unverifiable": True,
                "stopped_reason": "Usage missing for resample request",
            })
            stopped_reason = "Usage missing for resample request"
            logging.error("[resample] usage unverifiable, stopping.")
            break

        # 4) 事后核账（同一账本，与 TraceCoder 同口径）
        remaining_before = remaining
        remaining_after = remaining
        overrun_tokens = 0
        budget_overrun = False
        if remaining is not None:
            remaining_after, overrun_tokens, budget_overrun = reconcile_budget_after_request(
                remaining_before, p_tokens, c_tokens
            )
            args.remaining_budget = remaining_after
        total_p += p_tokens
        total_c += c_tokens
        requests.append({
            "request_id": f"R{request_id:02d}",
            "stage": "resample",
            "sent": True,
            "estimated_input": (need - output_reserve) if need is not None else None,
            "output_reserve": output_reserve,
            "prompt_tokens": p_tokens,
            "completion_tokens": c_tokens,
            "remaining_before": remaining_before,
            "remaining_after": remaining_after,
            "overrun_tokens": overrun_tokens,
            "budget_overrun": budget_overrun,
            "usage_unverifiable": False,
            "stopped_reason": None,
        })

        # 5) 只跑“可见测试”评测，用于候选选择
        visible_eval = check_correctness_func(problem_data, code, args.timeout)
        candidates.append({
            "request_id": request_id,
            "code": code,
            "visible_eval": visible_eval,
            "code_hash": _get_hash(code),
            "budget_overrun": budget_overrun,
            "overrun_tokens": overrun_tokens,
            "remaining_after": remaining_after,
        })

        # 6a) 超限优先：超限状态不能被“可见测试通过”掩盖
        if budget_overrun:
            budget_exhausted = True
            stopped_reason = "Budget overrun after resample request"
            requests[-1]["stopped_reason"] = stopped_reason
            logging.error(
                f"[resample] OVERRUN: before={remaining_before}, "
                f"actual={p_tokens + c_tokens}, after={remaining_after}"
            )
            break

        # 6b) 冻结规则：可见全过即停
        if stop_when_visible_passed and visible_eval.get("passed"):
            stopped_reason = "All visible tests passed"
            break

    if stopped_reason is None:
        if request_id >= max_resamples:
            stopped_reason = "Max resamples reached"
        else:
            stopped_reason = "Stopped"

    # 7) 候选选择：只用可见测试结果；并列取最早；超限候选不参与选择
    selectable = [c for c in candidates if not c.get("budget_overrun")]
    best = None
    if selectable:
        best = min(
            selectable,
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
            "request_id": f"R{c['request_id']:02d}",
            "code_hash": c["code_hash"],
            "visible_passed_count": c["visible_eval"].get("passed_count", 0),
            "visible_passed": c["visible_eval"].get("passed", False),
            "code": c["code"],
            "eval_result": c["visible_eval"],
            "selected": is_selected,
            "selection_reason": (
                SELECTION_RULE if is_selected
                else ("excluded_budget_overrun" if c.get("budget_overrun") else "")
            ),
            "budget_overrun": c.get("budget_overrun"),
            "overrun_tokens": c.get("overrun_tokens"),
            "remaining_after": c.get("remaining_after"),
        })

    return {
        "method": "independent_resample",
        "final_code": best["code"] if best else "",
        "final_code_hash": best["code_hash"] if best else "empty",
        "final_eval": final_eval,
        "final_visible_passed_count": final_eval.get("passed_count", 0),
        "candidates": candidates_out,
        "requests": requests,
        # num_requests = 账本行数 = sent + rejected + usage_unverifiable
        # 与 TraceCoder 的 len(request_log) 同口径；拆分见下面三个计数
        "num_requests": len(requests),
        "stopped_reason": stopped_reason,
        "prompt_tokens": total_p,
        "completion_tokens": total_c,
        "budget_exhausted": budget_exhausted,
        "budget_remaining": getattr(args, "remaining_budget", None),
        # --- 超限汇总：与 TraceCoder 同口径 ---
        # 三态汇总：任一条 True -> True；否则任一条 None -> None；全 False -> False
        "budget_overrun": (
            True if any(r.get("budget_overrun") is True for r in requests)
            else (None if any(r.get("budget_overrun") is None for r in requests)
                  else False)
        ),
        # 已知超限量：只累加有值的；若存在未知，另用 coverage_complete 表达
        "overrun_tokens": (
            None if any(r.get("overrun_tokens") is None for r in requests)
            else sum(r.get("overrun_tokens") or 0 for r in requests)
        ),
        # 覆盖完整性：有缺 usage 的记录则为 False
        "coverage_complete": not any(r.get("usage_unverifiable") for r in requests),
        # --- 三类计数分开，不靠导出默认值 ---
        "num_requests_sent": sum(1 for r in requests if r.get("sent") is True),
        "num_requests_rejected": sum(1 for r in requests if r.get("sent") is False),
        "num_requests_usage_unverifiable": sum(
            1 for r in requests if r.get("usage_unverifiable")
        ),
        # --- 超限候选策略 ---
        "overrun_candidate_policy": OVERRUN_CANDIDATE_POLICY,
        "selection_rule": SELECTION_RULE,
        "evaluation_scope": EVALUATION_SCOPE,
    }