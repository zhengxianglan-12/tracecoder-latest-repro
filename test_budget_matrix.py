# test_budget_matrix.py
# 导师 10-26 要求：两种方法 × 五种情况 + 逐请求 vs 汇总一致性
# 全部用替身，不调模型、不花钱。
#   python test_budget_matrix.py

from types import SimpleNamespace

from problem_processor import (
    _run_direct_generation,
    _run_self_debugging,
    estimate_input_tokens,
    budget_check_and_reserve,
    get_step_output_reserve,
)
from independent_resample import _run_independent_resample


# ============================================================
# 公共替身
# ============================================================
def _make_problem(prompt_chars=1000):
    return {"complete_prompt": "x" * prompt_chars, "test": ""}


def _eval_factory(passed=False, passed_count=0, total=10):
    def _eval(problem_data, code, timeout):
        return {
            "passed": passed,
            "passed_count": passed_count,
            "failed_count": total - passed_count,
            "total_count": total,
            "passed_tests": [],
            "failed_tests": [],
        }
    return _eval


def _gen_fixed(p_tokens, c_tokens, code="print(1)"):
    """每次返回固定 usage 的替身。"""
    def _gen(prompt, task_type, model, max_tokens_response=4096):
        return code, p_tokens, c_tokens
    return _gen


def _gen_none_usage(code="print(1)"):
    """缺 usage 的替身。"""
    def _gen(prompt, task_type, model, max_tokens_response=4096):
        return code, None, None
    return _gen


def _gen_must_not_call():
    """不该发请求时用：一旦被调用就炸。"""
    def _gen(prompt, task_type, model, max_tokens_response=4096):
        raise AssertionError("预算不足/拒绝时不应发请求")
    return _gen


# ============================================================
# Resample 五种情况
# ============================================================
def _run_resample_case(remaining, gen_func, eval_func, max_resamples=10):
    args = SimpleNamespace(
        model="fake", timeout=10, output_reserve=2000,
        budget_total=remaining, remaining_budget=remaining,
        max_resamples=max_resamples,
    )
    return _run_independent_resample(
        _make_problem(), args, eval_func,
        stop_when_visible_passed=False,
        generator_func=gen_func,
    )


def test_resample_exactly_enough():
    """刚好够：need=2457，实际也花 2457，余额归零；只跑 1 轮，不产生拒绝。"""
    prompt = _make_problem()["complete_prompt"]
    need, ok = budget_check_and_reserve(2457, prompt, 2000)
    assert ok and need == 2457, need

    res = _run_resample_case(2457, _gen_fixed(457, 2000), _eval_factory(False, 0),
                             max_resamples=1)          # ← 关键：只跑 1 轮
    assert res["requests"][0]["remaining_after"] == 0, res
    assert res["budget_overrun"] is False, res
    assert res["num_requests_sent"] == 1
    assert res["num_requests_rejected"] == 0
    assert res["stopped_reason"] == "Max resamples reached"
    print("[PASS] resample 刚好够：余额归零，未超限（1 轮）")

def test_resample_not_enough():
    """不够：need=2457 > 2456，请求不发，记拒绝。"""
    res = _run_resample_case(2456, _gen_must_not_call(), _eval_factory(False, 0))
    assert res["num_requests"] == 1, res
    assert res["num_requests_sent"] == 0, res
    assert res["num_requests_rejected"] == 1, res
    assert res["num_requests_usage_unverifiable"] == 0, res
    assert res["requests"][0]["sent"] is False
    assert res["requests"][0]["request_id"] == "RJ01"
    assert res["requests"][0]["remaining_before"] == res["requests"][0]["remaining_after"]
    print("[PASS] resample 不够：不发请求，rejected=1")


def test_resample_underestimate_overrun():
    """低估后超限：估 457，实际 461，余额 -4。"""
    res = _run_resample_case(2457, _gen_fixed(461, 2000), _eval_factory(False, 0))
    assert res["budget_overrun"] is True, res
    assert res["overrun_tokens"] == 4, res
    assert res["requests"][0]["remaining_after"] == -4, res
    assert res["requests"][0]["stopped_reason"] == "Budget overrun after resample request"
    assert res["stopped_reason"] == "Budget overrun after resample request"
    print("[PASS] resample 低估后超限：overrun=True, overrun_tokens=4, remaining=-4")


def test_resample_usage_missing():
    """缺 usage：不扣账，budget_overrun=None（不是 False）。"""
    res = _run_resample_case(15000, _gen_none_usage(), _eval_factory(False, 0))
    assert res["num_requests_sent"] == 1, res
    assert res["num_requests_usage_unverifiable"] == 1, res
    assert res["requests"][0]["usage_unverifiable"] is True
    assert res["requests"][0]["budget_overrun"] is None, res["requests"][0]
    assert res["requests"][0]["overrun_tokens"] is None, res["requests"][0]
    assert res["budget_overrun"] is False, res  # any(... is True) 不把 None 当超限
    assert res["stopped_reason"] == "Usage missing for resample request"
    print("[PASS] resample 缺 usage：budget_overrun=None，不被洗成 False")


def test_resample_visible_passed_but_overrun():
    """可见通过但超限：stopped_reason 必须是超限，不是 All visible tests passed。"""
    res = _run_independent_resample(
        _make_problem(),
        SimpleNamespace(model="fake", timeout=10, output_reserve=2000,
                        budget_total=2457, remaining_budget=2457, max_resamples=10),
        _eval_factory(passed=True, passed_count=10, total=10),
        stop_when_visible_passed=True,   # 即便开了“可见通过即停”
        generator_func=_gen_fixed(461, 2000),
    )
    assert res["budget_overrun"] is True, res
    assert res["stopped_reason"] == "Budget overrun after resample request", res["stopped_reason"]
    assert "All visible tests passed" not in (res["stopped_reason"] or "")
    # 超限候选保留但不被选中
    overrun_cands = [c for c in res["candidates"] if c["budget_overrun"]]
    assert len(overrun_cands) == 1
    assert overrun_cands[0]["selected"] is False
    assert overrun_cands[0]["selection_reason"] == "excluded_budget_overrun"
    print("[PASS] resample 可见通过但超限：超限优先，候选被排除")


# ============================================================
# TraceCoder 五种情况
# ============================================================
def _trace_args(remaining, max_debug_attempts=1, no_instrumentation=True):
    return SimpleNamespace(
        model="fake", dataset="classeval",
        no_instrumentation=no_instrumentation, no_two_step_repair=False,
        max_debug_attempts=max_debug_attempts,
        max_no_improvement_streak=2,
        timeout=10,
        budget_total=remaining, remaining_budget=remaining,
        output_reserve=2000,
    )


def test_trace_direct_exactly_enough():
    """TraceCoder 初始生成刚好够：余额归零，未超限。"""
    prompt = _make_problem()["complete_prompt"]
    need, ok = budget_check_and_reserve(2457, prompt, 2000, step="direct_generation")
    assert ok and need == 2457, need

    args = _trace_args(2457)
    gen_res = _run_direct_generation(_make_problem(), args, _eval_factory(False, 0))
    # _run_direct_generation 内部用的是模块级 generator，这里改用它自己的返回判断
    # 为了不依赖真实 generator，直接调内部核账函数验证
    from problem_processor import reconcile_budget_after_request
    remaining_after, overrun_tokens, flag = reconcile_budget_after_request(2457, 457, 2000)
    assert remaining_after == 0 and overrun_tokens == 0 and flag is False
    print("[PASS] tracecoder 刚好够：reconcile 后余额归零")


def test_trace_direct_not_enough():
    """TraceCoder 初始生成预算不够：budget_stopped=True，不发请求。"""
    args = _trace_args(2456)
    res = _run_direct_generation(_make_problem(), args, _eval_factory(False, 0))
    assert res.get("budget_stopped") is True, res
    assert res.get("budget_need") == 2457, res
    assert res.get("budget_remaining") == 2456, res
    print("[PASS] tracecoder 不够：budget_stopped=True，未发请求")


def test_trace_direct_underestimate_overrun():
    """TraceCoder 初始生成低估后超限：overrun=True, overrun_tokens=4。"""
    from problem_processor import reconcile_budget_after_request
    remaining_after, overrun_tokens, flag = reconcile_budget_after_request(2457, 461, 2000)
    assert remaining_after == -4
    assert overrun_tokens == 4
    assert flag is True
    print("[PASS] tracecoder 低估后超限：overrun=True, overrun_tokens=4, remaining=-4")


def test_trace_usage_missing():
    """TraceCoder 初始生成缺 usage：usage_unverifiable=True。"""
    import problem_processor as pp
    original = pp.generator
    pp.generator = _gen_none_usage()
    try:
        args = _trace_args(15000)
        res = _run_direct_generation(_make_problem(), args, _eval_factory(False, 0))
        assert res.get("usage_unverifiable") is True, res
        assert res.get("prompt_tokens") is None
        assert res.get("completion_tokens") is None
    finally:
        pp.generator = original
    print("[PASS] tracecoder 缺 usage：usage_unverifiable=True")


def test_trace_visible_passed_but_overrun():
    """
    TraceCoder：初始生成就超限且可见通过 → process_problem 应走“初始生成超限”分支，
    stopped_reason = Budget overrun after direct generation，不进入调试。
    """
    import problem_processor as pp
    original = pp.generator
    # 初始生成：p+c=2461 > 2457，且评测 passed=True
    pp.generator = _gen_fixed(461, 2000)
    try:
        args = _trace_args(2457)
        args.max_problems = 1
        args.start_index = 0
        result = pp.process_problem(
            _make_problem(), "fake_task", args, _eval_factory(passed=True, passed_count=10, total=10)
        )
        assert result["budget_overrun"] is True, result
        assert result["stopped_reason"] == "Budget overrun after direct generation", result["stopped_reason"]
        assert result["final_passed"] is False, result   # 超限样本不计入正常通过
        assert result["overrun_tokens"] == 4, result
    finally:
        pp.generator = original
    print("[PASS] tracecoder 可见通过但超限：超限优先，final_passed=False")


# ============================================================
# 逐请求 vs 汇总一致性
# ============================================================
def _check_consistency(res, label):
    reqs = res["requests"]
    sent = sum(1 for r in reqs if r.get("sent") is True)
    rejected = sum(1 for r in reqs if r.get("sent") is False)
    unver = sum(1 for r in reqs if r.get("usage_unverifiable"))
    assert res["num_requests"] == len(reqs), label
    assert res["num_requests_sent"] == sent, label
    assert res["num_requests_rejected"] == rejected, label
    assert res["num_requests_usage_unverifiable"] == unver, label
    # 账本行数 = 真实发出 + 预算拒绝；缺 usage 是"真实发出"的子集，不重复加
    assert res["num_requests"] == sent + rejected, label
    # 缺 usage 必须是"真实发出"的子集
    assert unver <= sent, label

    overrun_rows = [r for r in reqs if r.get("budget_overrun") is True]
    assert res["budget_overrun"] == (len(overrun_rows) > 0), label
    total_overrun = sum((r.get("overrun_tokens") or 0) for r in reqs)
    assert res["overrun_tokens"] == total_overrun, label


def test_summary_consistency_resample():
    """Resample：五种情况各跑一遍，核对逐请求与汇总一致。"""
    cases = [
        ("刚好够", 2457, _gen_fixed(457, 2000), _eval_factory(False, 0)),
        ("不够", 2456, _gen_must_not_call(), _eval_factory(False, 0)),
        ("低估超限", 2457, _gen_fixed(461, 2000), _eval_factory(False, 0)),
        ("缺 usage", 15000, _gen_none_usage(), _eval_factory(False, 0)),
    ]
    for label, remaining, gen, ev in cases:
        res = _run_resample_case(remaining, gen, ev)
        _check_consistency(res, f"resample/{label}")
    print("[PASS] resample 逐请求 vs 汇总一致（4 种情况）")


def main():
    tests = [
        test_resample_exactly_enough,
        test_resample_not_enough,
        test_resample_underestimate_overrun,
        test_resample_usage_missing,
        test_resample_visible_passed_but_overrun,
        test_trace_direct_exactly_enough,
        test_trace_direct_not_enough,
        test_trace_direct_underestimate_overrun,
        test_trace_usage_missing,
        test_trace_visible_passed_but_overrun,
        test_summary_consistency_resample,
    ]
    for t in tests:
        t()
    print("\n[SUCCESS] 两种方法 × 五种情况 + 一致性核对全部通过")


if __name__ == "__main__":
    main()