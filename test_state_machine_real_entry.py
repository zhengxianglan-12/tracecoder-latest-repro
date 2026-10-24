# test_state_machine_real_entry.py
# 按导师意见：调用真实 _run_self_debugging，替身 generator/评测，六类分支，三条断言
# 不调用模型、不产生费用；不修改 problem_processor.py

import json
import hashlib
import logging
from types import SimpleNamespace

logging.basicConfig(level=logging.INFO, format='%(message)s')

import problem_processor
from problem_processor import _run_self_debugging


def get_hash(code_str):
    return hashlib.md5(code_str.encode('utf-8')).hexdigest()[:8] if code_str else "empty"


# ============================================================
# 可被 _parse_llm_output 解析的计划文本
# ============================================================
PLAN_OK = (
    "REPAIR_PLAN_START\n"
    "```python\n"
    "fix the bug\n"
    "```\n"
    "REPAIR_PLAN_END\n"
    "Instrumentation_Suggestions_START\n"
    "```python\n"
    "\n"
    "```\n"
    "Instrumentation_Suggestions_END"
)


# ============================================================
# 替身 generator：替换 problem_processor 模块内的 generator 引用
# 真实函数内部调用 generator(prompt, "codeX_generate", model)
# 按 prompt 的独特标记判断步骤，返回预设内容，并逐请求记录编号
# ============================================================
class FakeGenerator:
    def __init__(self, plan_text, impl_code, instrumented_code=""):
        self.plan_text = plan_text
        self.impl_code = impl_code
        self.instrumented_code = instrumented_code
        self.requests = []  # 逐请求记录：编号 + 类型 + prompt 长度

    def __call__(self, prompt, task_type, model):
        req_id = len(self.requests) + 1

        # 顺序很重要：分析规划 prompt 里同时含 REPAIR_PLAN_START 和
        # Instrumentation_Suggestions_START，必须先按它独有的标记判为 analysis_plan
        if "Your tasks are:" in prompt and "Step 1. Analyze" in prompt:
            kind = "analysis_plan"
            out = self.plan_text
        elif "Detailed Repair Plan (Follow this plan carefully!)" in prompt:
            kind = "implementation"
            out = self.impl_code
        elif "Key Instrumentation Rules" in prompt or "Code to Instrument" in prompt:
            kind = "instrumentation"
            out = self.instrumented_code
        else:
            kind = "unknown"
            out = ""

        self.requests.append({
            "request_id": req_id,
            "kind": kind,
            "prompt_len": len(prompt),
        })
        # 真实 generator 返回 (text, prompt_tokens, completion_tokens)
        return out, 10, 10


# ============================================================
# 替身评测：按候选代码 -> 分数映射返回
# ============================================================
class FakeEvaluator:
    def __init__(self, score_map, total=19):
        self.score_map = dict(score_map)
        self.total = total
        self.calls = []

    def __call__(self, problem_data, code, timeout):
        score = self.score_map.get(code, 0)
        self.calls.append({"code_hash": get_hash(code), "score": score})
        return {
            "passed": score >= self.total,
            "passed_count": score,
            "failed_count": self.total - score,
            "total_count": self.total,
            "passed_tests": [],
            "failed_tests": [],
        }


# ============================================================
# 构造真实函数需要的 args 和 problem_data
# ============================================================
def make_args(max_debug_attempts=5, max_no_improvement_streak=2,
              no_instrumentation=True, timeout=10):
    return SimpleNamespace(
        max_debug_attempts=max_debug_attempts,
        max_no_improvement_streak=max_no_improvement_streak,
        no_instrumentation=no_instrumentation,
        timeout=timeout,
        model="fake-model",
    )


def make_problem_data():
    return {
        "task_id": "FAKE",
        "complete_prompt": "fake prompt",
        "test": "fake test",
    }


def make_initial_eval(score, total=19):
    return {
        "passed": score >= total,
        "passed_count": score,
        "failed_count": total - score,
        "total_count": total,
        "passed_tests": [],
        "failed_tests": [],
    }


# ============================================================
# 断言
# ============================================================
def assert_success_round_logged(session_log, label):
    passed_rounds = [a for a in session_log if a.get("status") == "passed"]
    assert passed_rounds, f"[{label}] 成功轮未写入 attempt_log"
    last = passed_rounds[-1]
    assert last.get("candidate_code"), f"[{label}] 成功轮缺少候选代码"
    assert last.get("stopped_reason") == "All tests passed", \
        f"[{label}] 成功轮 stopped_reason 应为 All tests passed，实际 {last.get('stopped_reason')}"
    print(f"[PASS] [{label}] 成功轮存在且记录完整（attempt={last.get('attempt')}）")


def assert_stagnation_stopped(session_log, label):
    assert session_log, f"[{label}] session_log 为空"
    last = session_log[-1]
    assert last.get("stopped_reason") == "Stagnation threshold reached", \
        f"[{label}] 连续停滞未触发停止，实际 stopped_reason={last.get('stopped_reason')}"
    assert last.get("next_input_actual") is None, \
        f"[{label}] 停滞终止轮 next_input_actual 应为 null"
    print(f"[PASS] [{label}] 连续停滞已停止，终止轮 next_input_actual=null")


def assert_terminal_no_next_request(session_log, fake_gen, label):
    assert session_log, f"[{label}] session_log 为空"
    last = session_log[-1]
    assert last.get("next_input_actual") is None, \
        f"[{label}] 终止轮 next_input_actual 应为 null，实际 {last.get('next_input_actual')}"
    for i, r in enumerate(fake_gen.requests, 1):
        assert r["request_id"] == i, f"[{label}] 请求编号不连续：{r['request_id']} != {i}"
    print(f"[PASS] [{label}] 终止轮无下一次请求；逐请求记录 {len(fake_gen.requests)} 条，编号连续")


# ============================================================
# 跑一个分支：monkeypatch problem_processor.generator，调用真实 _run_self_debugging
# ============================================================
def run_branch(label, initial_score, plan_text, impl_code, score_map,
               expect, max_attempts=5, no_instrumentation=True):
    fake_gen = FakeGenerator(plan_text=plan_text, impl_code=impl_code)
    evaluator = FakeEvaluator(score_map)

    original_generator = problem_processor.generator
    problem_processor.generator = fake_gen
    try:
        result = _run_self_debugging(
            initial_code="initial_code",
            initial_eval=make_initial_eval(initial_score),
            problem_data=make_problem_data(),
            args=make_args(max_debug_attempts=max_attempts,
                           no_instrumentation=no_instrumentation),
            check_correctness_func_param=evaluator,
        )
    finally:
        problem_processor.generator = original_generator

    session_log = result["session_log"]
    print(f"\n--- 分支: {label} ---")
    print(json.dumps(session_log, indent=2, ensure_ascii=False)[:1500])

    if expect == "passed":
        assert_success_round_logged(session_log, label)
        assert_terminal_no_next_request(session_log, fake_gen, label)
    elif expect in ("stagnation_tie", "stagnation_regress"):
        assert_stagnation_stopped(session_log, label)
        assert_terminal_no_next_request(session_log, fake_gen, label)
    elif expect in ("improve", "no_plan", "empty_code"):
        assert session_log, f"[{label}] session_log 为空"
        assert_terminal_no_next_request(session_log, fake_gen, label)
    elif expect == "improve_continue":
        assert session_log, f"[{label}] session_log 为空"
        first = session_log[0]
        assert first.get("status") == "improve", \
            f"[{label}] 第 1 轮应为 improve，实际 {first.get('status')}"
        assert first.get("next_input_actual") is not None, \
            f"[{label}] improve 后应继续，next_input_actual 不应为 null"
        assert first.get("stopped_reason") == "Continued to next attempt", \
            f"[{label}] improve 后 stopped_reason 应为 Continued to next attempt"
        assert_terminal_no_next_request(session_log, fake_gen, label)
    elif expect == "improve_max_attempts":
        assert session_log, f"[{label}] session_log 为空"
        last = session_log[-1]
        assert last.get("status") == "improve"
        assert last.get("stopped_reason") == "Max attempts reached", \
            f"[{label}] 应为 Max attempts reached，实际 {last.get('stopped_reason')}"
        assert_terminal_no_next_request(session_log, fake_gen, label)
    else:
        raise ValueError(f"未知 expect: {expect}")

    print(f"[PASS] 分支 {label} 验证通过")


# ============================================================
# 六类分支
# ============================================================
if __name__ == '__main__':
    print("========== 真实入口状态机自测（替身 generator/评测，不调用模型） ==========")

        # 1a. 改善后继续：12 -> 13，max_attempts=2，第 1 轮 improve 后应继续
    run_branch(
        "improve_continue",
        initial_score=12,
        plan_text=PLAN_OK,
        impl_code="CODE:\n```python\ncand_13\n```\nEND_CODE",
        score_map={"cand_13": 13},
        expect="improve_continue",
        max_attempts=2,
    )

    # 1b. 改善后达到轮数上限：12 -> 13，max_attempts=1
    run_branch(
        "improve_max_attempts",
        initial_score=12,
        plan_text=PLAN_OK,
        impl_code="CODE:\n```python\ncand_13\n```\nEND_CODE",
        score_map={"cand_13": 13},
        expect="improve_max_attempts",
        max_attempts=1,
    )

    # 2. 连续持平 -> 停滞停止：两轮 tie（12, 12），streak 到 2
    run_branch(
        "stagnation_tie",
        initial_score=12,
        plan_text=PLAN_OK,
        impl_code="CODE:\n```python\ncand_12\n```\nEND_CODE",
        score_map={"cand_12": 12},
        expect="stagnation_tie",
        max_attempts=5,
    )

    # 3. 连续退步 -> 停滞停止：两轮 regress（10, 10），streak 到 2
    run_branch(
        "stagnation_regress",
        initial_score=12,
        plan_text=PLAN_OK,
        impl_code="CODE:\n```python\ncand_10\n```\nEND_CODE",
        score_map={"cand_10": 10},
        expect="stagnation_regress",
        max_attempts=5,
    )

    # 4. 成功：15 -> 19
    run_branch(
        "passed",
        initial_score=15,
        plan_text=PLAN_OK,
        impl_code="CODE:\n```python\ncand_19\n```\nEND_CODE",
        score_map={"cand_19": 19},
        expect="passed",
    )

    # 5. 无计划：analysis 返回空 plan，真实函数应记录 no_plan 并退出
    run_branch(
        "no_plan",
        initial_score=12,
        plan_text="",
        impl_code="",
        score_map={},
        expect="no_plan",
        max_attempts=5,
    )

    # 6. 空代码：plan 正常，但 implementation 返回空代码
    run_branch(
        "empty_code",
        initial_score=12,
        plan_text=PLAN_OK,
        impl_code="CODE:\n```python\n\n```\nEND_CODE",
        score_map={},
        expect="empty_code",
        max_attempts=5,
    )
        # 7. 总表 stopped_reason 与末轮真实原因对应（验证 process_problem 的改动）
    def check_process_problem_stopped_reason():
        from types import SimpleNamespace as SN
        fake_gen = FakeGenerator(plan_text=PLAN_OK,
                                  impl_code="CODE:\n```python\ncand_19\n```\nEND_CODE")
        evaluator = FakeEvaluator({"initial_code": 15, "cand_19": 19})
        # process_problem 先跑 direct generation，再进 self-debugging；
        # direct 用 initial_code 得 15 分（未过），self-debug 得 19 分（过）
        args = SN(model="fake-model", dataset="classeval",
                  no_instrumentation=True, no_two_step_repair=False,
                  max_debug_attempts=5, max_no_improvement_streak=2,
                  timeout=10)
        original_generator = problem_processor.generator
        problem_processor.generator = fake_gen
        try:
            result = problem_processor.process_problem(
                make_problem_data(), "FAKE", args, evaluator)
        finally:
            problem_processor.generator = original_generator
        last_reason = result["debug_session_log"][-1]["stopped_reason"]
        assert result["stopped_reason"] == last_reason, \
            f"[process_problem] 总表 stopped_reason={result['stopped_reason']} 与末轮={last_reason} 不一致"
        print(f"[PASS] [process_problem] 总表 stopped_reason 与末轮一致：{last_reason}")

    check_process_problem_stopped_reason()

    print("\n[SUCCESS] 六类分支（改善、持平、退步、成功、无计划、空代码）全部验证通过！")