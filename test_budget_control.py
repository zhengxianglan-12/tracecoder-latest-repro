# test_budget_control.py
# 预算控制三小例验证：不调用模型、不产生费用
# 只测 problem_processor 的预算逻辑，不修改 problem_processor.py

import logging
from types import SimpleNamespace

logging.basicConfig(level=logging.INFO, format='%(message)s')

import problem_processor
from problem_processor import (
    _run_direct_generation,
    _run_self_debugging,
    estimate_input_tokens,
    budget_check_and_reserve,
)


# ============================================================
# 替身 generator：记录每次请求，按预设返回
# mode="normal"：返回固定 token 数
# mode="no_usage"：返回 (text, None, None)，模拟缺 usage
# mode="should_not_call"：一旦被调用就报错，用于验证“预算不足不应发请求”
# ============================================================
class FakeGenerator:
    def __init__(self, mode="normal", prompt_tokens=100, completion_tokens=50):
        self.mode = mode
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.calls = []  # 记录每次请求

    def __call__(self, prompt, task_type, model, max_tokens_response=4096):
        if self.mode == "should_not_call":
            raise AssertionError("预算不足时不应调用 generator，但被调用了")
        self.calls.append({
            "request_id": len(self.calls) + 1,
            "task_type": task_type,
            "prompt_len": len(prompt),
            "max_tokens_response": max_tokens_response,
        })
        if self.mode == "no_usage":
            return "fake_output", None, None
        return "fake_output", self.prompt_tokens, self.completion_tokens


class FakeEvaluator:
    def __init__(self, score=0, total=19):
        self.score = score
        self.total = total

    def __call__(self, problem_data, code, timeout):
        return {
            "passed": self.score >= self.total,
            "passed_count": self.score,
            "failed_count": self.total - self.score,
            "total_count": self.total,
            "passed_tests": [],
            "failed_tests": [],
        }


def make_problem_data():
    # complete_prompt 长度可控，便于精确算“输入估算”
    return {
        "task_id": "FAKE",
        "complete_prompt": "x" * 1000,   # 估算输入 = 1000 * 0.5 = 500
        "test": "fake test",
    }


def make_args(remaining_budget, output_reserve=2000, max_attempts=1,
              no_instrumentation=True):
    return SimpleNamespace(
        model="fake-model",
        dataset="classeval",
        no_instrumentation=no_instrumentation,
        no_two_step_repair=False,
        max_debug_attempts=max_attempts,
        max_no_improvement_streak=2,
        timeout=10,
        budget_total=remaining_budget,
        remaining_budget=remaining_budget,
        output_reserve=output_reserve,
    )


def install_fake_generator(fake_gen):
    original = problem_processor.generator
    problem_processor.generator = fake_gen
    return original


# ============================================================
# 小例 1：剩余预算不足 —— 不应发请求
# ============================================================
def test_budget_insufficient():
    print("\n========== 小例 1：剩余预算不足 ==========")
    prompt = make_problem_data()["complete_prompt"]
    est_input = estimate_input_tokens(prompt, step="direct_generation")
    output_reserve = 2000
    need = est_input + output_reserve
    print(f"估算输入={est_input}, 输出预留={output_reserve}, 需要={need}")

    # 预算只给 need - 1，差 1 不够
    budget = need - 1
    args = make_args(remaining_budget=budget, output_reserve=output_reserve)
    fake_gen = FakeGenerator(mode="should_not_call")
    original = install_fake_generator(fake_gen)
    try:
        res = _run_direct_generation(make_problem_data(), args, FakeEvaluator())
    finally:
        problem_processor.generator = original

    assert res.get("budget_stopped") is True, "预算不足时应返回 budget_stopped=True"
    assert len(fake_gen.calls) == 0, "预算不足时不应调用 generator"
    assert args.remaining_budget == budget, "预算不足时剩余预算不应变化"
    print(f"[PASS] 预算不足未发请求，budget_need={res.get('budget_need')}, remaining={res.get('budget_remaining')}")


# ============================================================
# 小例 2：恰好够一次请求 —— 允许发一次，发完预算归零
# ============================================================
def test_budget_exactly_enough():
    print("\n========== 小例 2：恰好够一次请求 ==========")
    prompt = make_problem_data()["complete_prompt"]
    est_input = estimate_input_tokens(prompt, step="direct_generation")
    output_reserve = 2000
    need = est_input + output_reserve
    print(f"估算输入={est_input}, 输出预留={output_reserve}, 需要={need}")

    # 预算恰好 = need
    budget = need
    # 替身实际用量：让 prompt+completion 正好等于 need，便于断言“发完归零”
    p_tok = est_input
    c_tok = output_reserve
    args = make_args(remaining_budget=budget, output_reserve=output_reserve)
    fake_gen = FakeGenerator(mode="normal", prompt_tokens=p_tok, completion_tokens=c_tok)
    original = install_fake_generator(fake_gen)
    try:
        res = _run_direct_generation(make_problem_data(), args, FakeEvaluator())
    finally:
        problem_processor.generator = original

    assert len(fake_gen.calls) == 1, "恰好够一次时应发 1 次请求"
    assert not res.get("budget_stopped"), "恰好够一次不应被 budget_stopped"
    assert res.get("prompt_tokens") == p_tok
    assert res.get("completion_tokens") == c_tok
    assert args.remaining_budget == 0, f"发完预算应归零，实际 {args.remaining_budget}"
    # 请求里应真的传出输出上限
    assert fake_gen.calls[0]["max_tokens_response"] == output_reserve, "请求应带 max_tokens_response=输出预留"
    print(f"[PASS] 恰好够一次：发 1 次，实际用量 {p_tok}+{c_tok}，剩余预算归零")


# ============================================================
# 小例 3：缺少 usage —— 标不可核验并停，不按 0 放行
# ============================================================
def test_missing_usage():
    print("\n========== 小例 3：缺少 usage ==========")
    args = make_args(remaining_budget=15000, output_reserve=2000)
    fake_gen = FakeGenerator(mode="no_usage")
    original = install_fake_generator(fake_gen)
    try:
        # 3a. 初始生成缺 usage
        res = _run_direct_generation(make_problem_data(), args, FakeEvaluator())
        assert res.get("usage_unverifiable") is True, "初始生成缺 usage 应标 usage_unverifiable"
        assert res.get("prompt_tokens") is None and res.get("completion_tokens") is None
        assert args.remaining_budget == 15000, "缺 usage 时不应扣账（无法核验）"
        print("[PASS] 初始生成缺 usage：标不可核验，不扣账")

        # 3b. 调试循环里缺 usage：应记录 usage_unverifiable 并停
        args2 = make_args(remaining_budget=15000, max_attempts=3)
        fake_gen2 = FakeGenerator(mode="no_usage")
        original2 = install_fake_generator(fake_gen2)
        try:
            debug_res = _run_self_debugging(
                initial_code="initial_code",
                initial_eval={"passed": False, "passed_count": 0, "failed_count": 19, "total_count": 19,
                              "passed_tests": [], "failed_tests": []},
                problem_data=make_problem_data(),
                args=args2,
                check_correctness_func_param=FakeEvaluator(),
            )
        finally:
            problem_processor.generator = original2

        assert debug_res.get("usage_unverifiable") is True, "调试循环缺 usage 应标 usage_unverifiable"
        last = debug_res["session_log"][-1]
        assert last.get("status") == "usage_unverifiable", f"最后一条应为 usage_unverifiable，实际 {last.get('status')}"
        assert last.get("stopped_reason") == "Usage missing for analysis request", \
            f"停止原因应为 Usage missing for analysis request，实际 {last.get('stopped_reason')}"
        assert debug_res.get("budget_exhausted") is False, "缺 usage 不应被标为 budget_exhausted"
        print(f"[PASS] 调试循环缺 usage：status={last.get('status')}, stopped_reason={last.get('stopped_reason')}")
    finally:
        problem_processor.generator = original


# ============================================================
# 附加小例：预算记账是“同一账本”
# 初始生成 + 调试循环共用 args.remaining_budget
# ============================================================
def test_shared_budget_ledger():
    print("\n========== 附加小例：初始生成与调试共用同一预算账本 ==========")
    prompt = make_problem_data()["complete_prompt"]
    est_input = estimate_input_tokens(prompt, step="direct_generation")
    output_reserve = 2000
    need = est_input + output_reserve
    budget = need * 3  # 够 3 次请求

    args = make_args(remaining_budget=budget, output_reserve=output_reserve, max_attempts=1)
    # 初始生成：用 need 的一半，模拟实际用量小于预留
    p_tok, c_tok = est_input // 2, output_reserve // 2
    fake_gen = FakeGenerator(mode="normal", prompt_tokens=p_tok, completion_tokens=c_tok)
    original = install_fake_generator(fake_gen)
    try:
        gen_res = _run_direct_generation(make_problem_data(), args, FakeEvaluator())
    finally:
        problem_processor.generator = original

    after_initial = args.remaining_budget
    assert after_initial == budget - (p_tok + c_tok), \
        f"初始生成后预算应为 {budget - (p_tok + c_tok)}，实际 {after_initial}"
    print(f"[OK] 初始生成后 remaining_budget={after_initial}（从 {budget} 扣 {p_tok + c_tok}）")

    # 调试循环继续用同一个 args，预算应接着扣
    fake_gen2 = FakeGenerator(mode="normal", prompt_tokens=p_tok, completion_tokens=c_tok)
    original2 = install_fake_generator(fake_gen2)
    try:
        debug_res = _run_self_debugging(
            initial_code=gen_res["code"],
            initial_eval=gen_res["eval_result"],
            problem_data=make_problem_data(),
            args=args,
            check_correctness_func_param=FakeEvaluator(),
        )
    finally:
        problem_processor.generator = original2

    after_debug = args.remaining_budget
    assert after_debug < after_initial, "调试循环应继续从同一账本扣预算"
    assert debug_res.get("budget_remaining") == after_debug
    print(f"[PASS] 调试循环继续扣同一账本：remaining_budget {after_initial} -> {after_debug}")


if __name__ == '__main__':
    print("========== 预算控制三小例（不调用模型、不产生费用） ==========")
    test_budget_insufficient()
    test_budget_exactly_enough()
    test_missing_usage()
    test_shared_budget_ledger()
    print("\n[SUCCESS] 预算控制小例全部通过！")