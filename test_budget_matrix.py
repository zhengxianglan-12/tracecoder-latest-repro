# test_budget_matrix.py
# 导师 10-26 / 10-31 要求：
#   - 两种方法 × 五种预算情况
#   - 逐请求 vs 汇总一致性
#   - failure_kind 两态（response_missing / usage_missing）
#   - 多次调用逐条记录（初始成功 + 调试失败）
#   - 两方法口径一致
# 全部用替身，不调模型、不花钱。
#   python test_budget_matrix.py

from types import SimpleNamespace

from problem_processor import (
    _run_direct_generation,
    estimate_input_tokens,
    budget_check_and_reserve,
    get_step_output_reserve,
)
from independent_resample import _run_independent_resample


# ============================================================
# 模块 1：公共替身
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
    """成功但缺 usage 的替身（第一个返回值非 None）。"""
    def _gen(prompt, task_type, model, max_tokens_response=4096):
        return code, None, None
    return _gen


def _gen_response_missing():
    """请求失败的替身（第一个返回值是 None）。"""
    def _gen(prompt, task_type, model, max_tokens_response=4096):
        return None, None, None
    return _gen


def _gen_must_not_call():
    """不该发请求时用：一旦被调用就炸。"""
    def _gen(prompt, task_type, model, max_tokens_response=4096):
        raise AssertionError("预算不足/拒绝时不应发请求")
    return _gen


def _gen_sequence(sequence):
    """
    按调用次数依次返回 sequence 里的结果；用完一直返最后一个。
    例：[(None,None,None), ("code",100,200)] → 第 1 次失败，第 2 次成功。
    """
    state = {"i": 0}
    def _gen(prompt, task_type, model, max_tokens_response=4096):
        idx = min(state["i"], len(sequence) - 1)
        state["i"] += 1
        return sequence[idx]
    return _gen


# ============================================================
# 模块 2：Resample 预算行为
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
                             max_resamples=1)
    assert res["requests"][0]["remaining_after"] == 0, res
    assert res["budget_overrun"] is False, res
    assert res["overrun_tokens"] == 0, res
    assert res["coverage_complete"] is True, res
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
    assert res["budget_overrun"] is None, res
    assert res["overrun_tokens"] is None, res
    assert res["coverage_complete"] is False, res
    assert res["stopped_reason"] == "Usage missing for resample request"
    print("[PASS] resample 缺 usage：budget_overrun=None，不被洗成 False")


def test_resample_visible_passed_but_overrun():
    """可见通过但超限：stopped_reason 必须是超限，不是 All visible tests passed。"""
    res = _run_independent_resample(
        _make_problem(),
        SimpleNamespace(model="fake", timeout=10, output_reserve=2000,
                        budget_total=2457, remaining_budget=2457, max_resamples=10),
        _eval_factory(passed=True, passed_count=10, total=10),
        stop_when_visible_passed=True,
        generator_func=_gen_fixed(461, 2000),
    )
    assert res["budget_overrun"] is True, res
    assert res["stopped_reason"] == "Budget overrun after resample request", res["stopped_reason"]
    assert "All visible tests passed" not in (res["stopped_reason"] or "")
    overrun_cands = [c for c in res["candidates"] if c["budget_overrun"]]
    assert len(overrun_cands) == 1
    assert overrun_cands[0]["selected"] is False
    assert overrun_cands[0]["selection_reason"] == "excluded_budget_overrun"
    print("[PASS] resample 可见通过但超限：超限优先，候选被排除")


# ============================================================
# 模块 3：TraceCoder 预算行为
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
    """TraceCoder 初始生成刚好够：走真实入口，余额归零，未超限。"""
    import problem_processor as pp
    prompt = _make_problem()["complete_prompt"]
    need, ok = budget_check_and_reserve(2457, prompt, 2000, step="direct_generation")
    assert ok and need == 2457, need

    original = pp.generator
    pp.generator = _gen_fixed(457, 2000)
    try:
        args = _trace_args(2457)
        gen_res = _run_direct_generation(_make_problem(), args, _eval_factory(False, 0))
    finally:
        pp.generator = original

    assert gen_res.get("budget_stopped") is not True, gen_res
    assert gen_res.get("usage_unverifiable") is not True, gen_res
    assert gen_res.get("remaining_after") == 0, gen_res
    assert gen_res.get("overrun_tokens") == 0, gen_res
    assert gen_res.get("budget_overrun") is False, gen_res
    assert args.remaining_budget == 0, args.remaining_budget
    print("[PASS] tracecoder 刚好够：走入口，余额归零，未超限")


def test_trace_direct_not_enough():
    """TraceCoder 初始生成预算不够：budget_stopped=True，零请求。"""
    import problem_processor as pp
    original = pp.generator
    pp.generator = _gen_must_not_call()
    try:
        args = _trace_args(2456)
        res = _run_direct_generation(_make_problem(), args, _eval_factory(False, 0))
    finally:
        pp.generator = original

    assert res.get("budget_stopped") is True, res
    assert res.get("budget_need") == 2457, res
    assert res.get("budget_remaining") == 2456, res
    assert args.remaining_budget == 2456, args.remaining_budget
    print("[PASS] tracecoder 不够：budget_stopped=True，零请求")


def test_trace_direct_underestimate_overrun():
    """TraceCoder 初始生成低估后超限：走真实入口，overrun=True。"""
    import problem_processor as pp
    original = pp.generator
    pp.generator = _gen_fixed(461, 2000)
    try:
        args = _trace_args(2457)
        gen_res = _run_direct_generation(_make_problem(), args, _eval_factory(False, 0))
    finally:
        pp.generator = original

    assert gen_res.get("budget_overrun") is True, gen_res
    assert gen_res.get("overrun_tokens") == 4, gen_res
    assert gen_res.get("remaining_after") == -4, gen_res
    assert args.remaining_budget == -4, args.remaining_budget
    print("[PASS] tracecoder 低估后超限：走入口，overrun=True, overrun_tokens=4")


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
    """初始生成就超限且可见通过 → process_problem 走超限分支，不进入调试。"""
    import problem_processor as pp
    original = pp.generator
    pp.generator = _gen_fixed(461, 2000)
    try:
        args = _trace_args(2457)
        args.max_problems = 1
        args.start_index = 0
        result = pp.process_problem(
            _make_problem(), "fake_task", args,
            _eval_factory(passed=True, passed_count=10, total=10)
        )
        assert result["budget_overrun"] is True, result
        assert result["stopped_reason"] == "Budget overrun after direct generation", result["stopped_reason"]
        assert result["final_passed"] is False, result
        assert result["overrun_tokens"] == 4, result
    finally:
        pp.generator = original
    print("[PASS] tracecoder 可见通过但超限：超限优先，final_passed=False")


# ============================================================
# 模块 4：逐请求 vs 汇总一致性
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
    assert res["num_requests"] == sent + rejected, label
    assert unver <= sent, label

    has_true = any(r.get("budget_overrun") is True for r in reqs)
    has_none = any(r.get("budget_overrun") is None for r in reqs)
    expected = True if has_true else (None if has_none else False)
    assert res["budget_overrun"] == expected, label
    if any(r.get("overrun_tokens") is None for r in reqs):
        assert res["overrun_tokens"] is None, label
    else:
        assert res["overrun_tokens"] == sum((r.get("overrun_tokens") or 0) for r in reqs), label


def test_summary_consistency_resample():
    """Resample：四种情况各跑一遍，核对逐请求与汇总一致。"""
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


# ============================================================
# 模块 5：process_problem 提前 return 分支
# ============================================================
def test_process_problem_initial_budget_rejected_has_request_log():
    """首次预算拒绝：提前 return 也要带 request_log，且 sent=False。"""
    import problem_processor as pp
    original = pp.generator
    pp.generator = _gen_must_not_call()
    try:
        args = _trace_args(1)
        args.max_problems = 1
        args.start_index = 0
        result = pp.process_problem(_make_problem(), "fake_task", args, _eval_factory(False, 0))
    finally:
        pp.generator = original

    reqs = result["request_log"]
    assert len(reqs) == 1, reqs
    assert reqs[0]["sent"] is False, reqs[0]
    assert reqs[0]["stage"] == "skipped_budget_direct_generation", reqs[0]
    assert result["stopped_reason"] == "Budget insufficient for initial generation", result
    print("[PASS] process_problem 首次预算拒绝：request_log 有 sent=False 记录")


def test_process_problem_initial_usage_missing_has_request_log():
    """首次缺 usage：提前 return 也要带 request_log，且 sent=True、用量 None。"""
    import problem_processor as pp
    original = pp.generator
    pp.generator = _gen_none_usage()
    try:
        args = _trace_args(15000)
        args.max_problems = 1
        args.start_index = 0
        result = pp.process_problem(_make_problem(), "fake_task", args, _eval_factory(False, 0))
    finally:
        pp.generator = original

    reqs = result["request_log"]
    assert len(reqs) == 1, reqs
    assert reqs[0]["sent"] is True, reqs[0]
    assert reqs[0]["usage_unverifiable"] is True, reqs[0]
    assert reqs[0]["budget_overrun"] is None, reqs[0]
    assert reqs[0]["overrun_tokens"] is None, reqs[0]
    assert result["stopped_reason"] == "Initial generation usage unverifiable", result
    print("[PASS] process_problem 首次缺 usage：request_log 有 sent=True 且用量 None 记录")


def test_process_problem_ledger_equation():
    """账本行数 = 已发 + 拒绝；未知是已发子集。"""
    import problem_processor as pp
    original = pp.generator
    pp.generator = _gen_none_usage()
    try:
        args = _trace_args(15000)
        args.max_problems = 1
        args.start_index = 0
        result = pp.process_problem(_make_problem(), "fake_task", args, _eval_factory(False, 0))
    finally:
        pp.generator = original

    reqs = result["request_log"]
    sent = sum(1 for r in reqs if r.get("sent") is True)
    rejected = sum(1 for r in reqs if r.get("sent") is False)
    unver = sum(1 for r in reqs if r.get("usage_unverifiable"))
    assert len(reqs) == sent + rejected, (len(reqs), sent, rejected)
    assert unver <= sent, (unver, sent)
    print("[PASS] process_problem 账本等式：行数=已发+拒绝，未知<=已发")


# ============================================================
# 模块 6：10-31 新增 —— failure_kind 两态 + 多次调用 + 两方法口径一致
# ============================================================
def test_resample_response_missing_kind():
    """Resample 请求失败：failure_kind=response_missing，用量 None 不写 0。"""
    res = _run_resample_case(15000, _gen_response_missing(), _eval_factory(False, 0))
    assert res["num_requests"] == 1, res
    assert res["num_requests_sent"] == 1, res
    assert res["num_requests_usage_unverifiable"] == 1, res
    assert res["requests"][0]["failure_kind"] == "response_missing", res["requests"][0]
    assert res["requests"][0]["budget_overrun"] is None, res["requests"][0]
    assert res["requests"][0]["overrun_tokens"] is None, res["requests"][0]
    assert res["stopped_reason"] == "Response missing for resample request", res["stopped_reason"]
    print("[PASS] resample 响应缺失：failure_kind=response_missing")


def test_resample_usage_missing_kind():
    """Resample 成功但缺 usage：failure_kind=usage_missing。"""
    res = _run_resample_case(15000, _gen_none_usage(), _eval_factory(False, 0))
    assert res["num_requests"] == 1, res
    assert res["requests"][0]["failure_kind"] == "usage_missing", res["requests"][0]
    assert res["requests"][0]["budget_overrun"] is None, res["requests"][0]
    assert res["stopped_reason"] == "Usage missing for resample request", res["stopped_reason"]
    print("[PASS] resample 未知用量：failure_kind=usage_missing")


def test_resample_partial_usage_missing():
    """两次独立调用：第 1 次成功有用量、第 2 次成功缺 usage。账本 2 条。"""
    gen = _gen_sequence([
        ("print(1)", 100, 200),
        ("print(2)", None, None),
    ])
    res = _run_resample_case(15000, gen, _eval_factory(False, 0))
    assert res["num_requests"] == 2, res
    assert res["num_requests_sent"] == 2, res
    assert res["num_requests_usage_unverifiable"] == 1, res
    assert res["requests"][0].get("failure_kind") is None, res["requests"][0]
    assert res["requests"][0]["budget_overrun"] is False, res["requests"][0]
    assert res["requests"][1]["failure_kind"] == "usage_missing", res["requests"][1]
    assert res["requests"][1]["budget_overrun"] is None, res["requests"][1]
    print("[PASS] resample 部分用量未知：2 条账本，第 2 条 usage_missing")


def test_trace_response_missing_kind():
    """TraceCoder 初始生成请求失败：failure_kind=response_missing。"""
    import problem_processor as pp
    original = pp.generator
    pp.generator = _gen_response_missing()
    try:
        args = _trace_args(15000)
        args.max_problems = 1
        args.start_index = 0
        result = pp.process_problem(_make_problem(), "fake_task", args, _eval_factory(False, 0))
    finally:
        pp.generator = original

    reqs = result["request_log"]
    assert len(reqs) == 1, reqs
    assert reqs[0]["sent"] is True, reqs[0]
    assert reqs[0]["failure_kind"] == "response_missing", reqs[0]
    assert reqs[0]["budget_overrun"] is None, reqs[0]
    assert result["stopped_reason"] == "Initial generation response missing", result["stopped_reason"]
    print("[PASS] tracecoder 响应缺失：failure_kind=response_missing")


def test_trace_usage_missing_kind():
    """TraceCoder 初始生成成功但缺 usage：failure_kind=usage_missing。"""
    import problem_processor as pp
    original = pp.generator
    pp.generator = _gen_none_usage()
    try:
        args = _trace_args(15000)
        args.max_problems = 1
        args.start_index = 0
        result = pp.process_problem(_make_problem(), "fake_task", args, _eval_factory(False, 0))
    finally:
        pp.generator = original

    reqs = result["request_log"]
    assert len(reqs) == 1, reqs
    assert reqs[0]["sent"] is True, reqs[0]
    assert reqs[0]["failure_kind"] == "usage_missing", reqs[0]
    assert reqs[0]["budget_overrun"] is None, reqs[0]
    assert result["stopped_reason"] == "Initial generation usage unverifiable", result["stopped_reason"]
    print("[PASS] tracecoder 未知用量：failure_kind=usage_missing")


def test_trace_initial_success_then_debug_fail():
    """初始生成成功 → 调试第 1 轮 instrumentation 失败。账本 2 条。"""
    import problem_processor as pp
    gen = _gen_sequence([
        ("print(1)", 100, 200),
        (None, None, None),
    ])
    original = pp.generator
    pp.generator = gen
    try:
        args = _trace_args(15000, max_debug_attempts=1, no_instrumentation=False)
        args.max_problems = 1
        args.start_index = 0
        result = pp.process_problem(_make_problem(), "fake_task", args, _eval_factory(False, 0))
    finally:
        pp.generator = original

    reqs = result["request_log"]
    assert len(reqs) == 2, reqs
    assert reqs[0].get("failure_kind") is None, reqs[0]
    assert reqs[1]["failure_kind"] == "response_missing", reqs[1]
    print("[PASS] tracecoder 初始成功+调试失败：2 条账本，第 2 条 response_missing")


def test_two_methods_failure_kind_consistent():
    """两方法口径一致：response_missing / usage_missing 两边都记对。"""
    import problem_processor as pp
    original = pp.generator

    # ---- response_missing ----
    pp.generator = _gen_response_missing()
    try:
        args = _trace_args(15000)
        args.max_problems = 1
        args.start_index = 0
        trace_result = pp.process_problem(_make_problem(), "fake_task", args, _eval_factory(False, 0))
    finally:
        pp.generator = original
    trace_kind = trace_result["request_log"][0]["failure_kind"]

    resample_res = _run_resample_case(15000, _gen_response_missing(), _eval_factory(False, 0))
    resample_kind = resample_res["requests"][0]["failure_kind"]
    assert trace_kind == resample_kind == "response_missing", (trace_kind, resample_kind)

    # ---- usage_missing ----
    pp.generator = _gen_none_usage()
    try:
        args = _trace_args(15000)
        args.max_problems = 1
        args.start_index = 0
        trace_result = pp.process_problem(_make_problem(), "fake_task", args, _eval_factory(False, 0))
    finally:
        pp.generator = original
    trace_kind = trace_result["request_log"][0]["failure_kind"]

    resample_res = _run_resample_case(15000, _gen_none_usage(), _eval_factory(False, 0))
    resample_kind = resample_res["requests"][0]["failure_kind"]
    assert trace_kind == resample_kind == "usage_missing", (trace_kind, resample_kind)

    print("[PASS] 两方法口径一致：response_missing / usage_missing 两边都记对")
def test_resample_two_fail_then_success_ledger():
    """
    Resample 底层序列：失败、失败、成功。
    当前设计“一失败就停”，所以只消费前 1 次；账本 1 条 response_missing。
    本测试如实记录“设计如此”，并断言“失败不被吞”。
    """
    gen = _gen_sequence([
        (None, None, None),          # 第 1 次：失败
        (None, None, None),          # 第 2 次：失败（不会被消费）
        ("print(1)", 100, 200),      # 第 3 次：成功（不会被消费）
    ])
    res = _run_resample_case(15000, gen, _eval_factory(False, 0))
    # 设计如此：一失败就停，只记 1 条
    assert res["num_requests"] == 1, res
    assert res["requests"][0]["failure_kind"] == "response_missing", res["requests"][0]
    print("[PASS] resample 两次失败后成功：设计一失败就停，账本 1 条 response_missing（如实记录）")


def test_resample_all_fail_ledger():
    """Resample 底层序列：全部失败。当前设计一失败就停，账本 1 条 response_missing。"""
    gen = _gen_sequence([
        (None, None, None),
        (None, None, None),
        (None, None, None),
    ])
    res = _run_resample_case(15000, gen, _eval_factory(False, 0))
    assert res["num_requests"] == 1, res
    assert res["requests"][0]["failure_kind"] == "response_missing", res["requests"][0]
    assert res["budget_overrun"] is None, res
    assert res["coverage_complete"] is False, res
    print("[PASS] resample 全部失败：设计一失败就停，账本 1 条 response_missing（如实记录）")


def test_trace_two_fail_then_success_ledger():
    """
    TraceCoder：初始生成失败 → 设计上不再进调试。
    底层序列“失败、失败、成功”，只消费第 1 次；账本 1 条 response_missing。
    """
    import problem_processor as pp
    gen = _gen_sequence([
        (None, None, None),
        (None, None, None),
        ("print(1)", 100, 200),
    ])
    original = pp.generator
    pp.generator = gen
    try:
        args = _trace_args(15000)
        args.max_problems = 1
        args.start_index = 0
        result = pp.process_problem(_make_problem(), "fake_task", args, _eval_factory(False, 0))
    finally:
        pp.generator = original

    reqs = result["request_log"]
    assert len(reqs) == 1, reqs
    assert reqs[0]["failure_kind"] == "response_missing", reqs[0]
    print("[PASS] tracecoder 两次失败后成功：初始失败即停，账本 1 条 response_missing（如实记录）")


def test_trace_all_fail_ledger():
    """TraceCoder：初始生成全部失败。账本 1 条 response_missing。"""
    import problem_processor as pp
    gen = _gen_sequence([
        (None, None, None),
        (None, None, None),
    ])
    original = pp.generator
    pp.generator = gen
    try:
        args = _trace_args(15000)
        args.max_problems = 1
        args.start_index = 0
        result = pp.process_problem(_make_problem(), "fake_task", args, _eval_factory(False, 0))
    finally:
        pp.generator = original

    reqs = result["request_log"]
    assert len(reqs) == 1, reqs
    assert reqs[0]["failure_kind"] == "response_missing", reqs[0]
    print("[PASS] tracecoder 全部失败：初始失败即停，账本 1 条 response_missing（如实记录）")

# ============================================================
# 模块 7：入口
# ============================================================
def main():
    import logging, sys

    class _Tee:
        def __init__(self, *streams):
            self.streams = streams
        def write(self, data):
            for s in self.streams:
                s.write(data)
        def flush(self):
            for s in self.streams:
                s.flush()

    _log_file = open("test_budget_matrix_output.txt", "w", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, _log_file)
    sys.stderr = _Tee(sys.__stderr__, _log_file)

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:%(name)s:%(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )

    tests = [
        # 模块 2：Resample 预算
        test_resample_exactly_enough,
        test_resample_not_enough,
        test_resample_underestimate_overrun,
        test_resample_usage_missing,
        test_resample_visible_passed_but_overrun,
        # 模块 3：TraceCoder 预算
        test_trace_direct_exactly_enough,
        test_trace_direct_not_enough,
        test_trace_direct_underestimate_overrun,
        test_trace_usage_missing,
        test_trace_visible_passed_but_overrun,
        # 模块 4：一致性
        test_summary_consistency_resample,
        # 模块 5：process_problem 分支
        test_process_problem_initial_budget_rejected_has_request_log,
        test_process_problem_initial_usage_missing_has_request_log,
        test_process_problem_ledger_equation,
        # 模块 6：10-31 新增
        test_resample_response_missing_kind,
        test_resample_usage_missing_kind,
        test_resample_partial_usage_missing,
        test_trace_response_missing_kind,
        test_trace_usage_missing_kind,
        test_trace_initial_success_then_debug_fail,
        test_two_methods_failure_kind_consistent,
        test_resample_two_fail_then_success_ledger,
        test_resample_all_fail_ledger,
        test_trace_two_fail_then_success_ledger,
        test_trace_all_fail_ledger,
    ]
    for t in tests:
        t()
    print("\n[SUCCESS] 两种方法 × 五种情况 + failure_kind 两态 + 两方法口径一致 全部通过")
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    _log_file.close()


if __name__ == "__main__":
    main()