# test_independent_resample.py
# Independent Resample 无付费自测：替身 generator 直接注入，不调真 API
#   python test_independent_resample.py

import logging
from types import SimpleNamespace

from independent_resample import (
    _run_independent_resample,
    _get_hash,
    SELECTION_RULE,
    EVALUATION_SCOPE,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")


class FakeGenerator:
    def __init__(self, codes, prompt_tokens=100, completion_tokens=50):
        self.codes = list(codes)
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.calls = []

    def __call__(self, prompt, task_type, model, max_tokens_response=4096):
        self.calls.append({"prompt_len": len(prompt),
                           "max_tokens_response": max_tokens_response})
        if not self.codes:
            raise RuntimeError("FakeGenerator 候选耗尽")
        code = self.codes.pop(0) if len(self.codes) > 1 else self.codes[0]
        return code, self.prompt_tokens, self.completion_tokens


class FakeEvaluator:
    def __init__(self, score_map, total=19):
        self.score_map = score_map
        self.total = total

    def __call__(self, problem_data, code, timeout):
        score = self.score_map.get(code, 0)
        return {
            "passed": score >= self.total,
            "passed_count": score,
            "failed_count": self.total - score,
            "total_count": self.total,
            "passed_tests": [], "failed_tests": [],
        }


def make_args(remaining=15000, output_reserve=2000, max_resamples=50):
    return SimpleNamespace(
        model="fake-model", timeout=10,
        output_reserve=output_reserve,
        remaining_budget=remaining,
        max_resamples=max_resamples,
    )


def make_problem():
    return {"task_id": "FAKE", "complete_prompt": "x" * 1000, "test": "fake test"}


def test_select_highest_visible():
    print("\n===== 小例 A：按可见通过数选最高 =====")
    fake_gen = FakeGenerator(["cand_10", "cand_15", "cand_12"])
    evaluator = FakeEvaluator({"cand_10": 10, "cand_15": 15, "cand_12": 12})
    res = _run_independent_resample(
        make_problem(), make_args(max_resamples=3), evaluator,
        stop_when_visible_passed=False, generator_func=fake_gen,
    )
    assert res["final_code_hash"] == _get_hash("cand_15"), res["candidates"]
    assert res["selection_rule"] == SELECTION_RULE
    assert res["evaluation_scope"] == EVALUATION_SCOPE
    print(f"[PASS] 选中 request_id=2，visible={res['final_visible_passed_count']}")


def test_tie_earliest():
    print("\n===== 小例 B：并列取最早生成 =====")
    fake_gen = FakeGenerator(["cand_A", "cand_B"])
    evaluator = FakeEvaluator({"cand_A": 15, "cand_B": 15})
    res = _run_independent_resample(
        make_problem(), make_args(max_resamples=2), evaluator,
        stop_when_visible_passed=False, generator_func=fake_gen,
    )
    assert res["candidates"][0]["visible_passed_count"] == 15
    assert res["final_code_hash"] == _get_hash("cand_A"), res["candidates"]
    print("[PASS] 并列时选中 request_id=1（最早）")


def test_stop_when_visible_passed():
    print("\n===== 小例 C：可见全过即停 =====")
    fake_gen = FakeGenerator(["cand_19", "cand_should_not_be_used"])
    evaluator = FakeEvaluator({"cand_19": 19, "cand_should_not_be_used": 19})
    res = _run_independent_resample(
        make_problem(), make_args(), evaluator,
        stop_when_visible_passed=True, generator_func=fake_gen,
    )
    assert res["num_requests"] == 1, res["requests"]
    assert res["stopped_reason"] == "All visible tests passed"
    print("[PASS] 全过后只发 1 次请求")


def test_budget_insufficient():
    print("\n===== 小例 D：预算不足即停 =====")
    fake_gen = FakeGenerator(["cand_1", "cand_2", "cand_3"])
    evaluator = FakeEvaluator({"cand_1": 1, "cand_2": 2, "cand_3": 3})
    args = make_args(remaining=2600)
    res = _run_independent_resample(
        make_problem(), args, evaluator,
        stop_when_visible_passed=False, generator_func=fake_gen,
    )
    assert res["budget_exhausted"] is True, res
    assert res["stopped_reason"] == "Budget insufficient before resample request"
    print(f"[PASS] 第二次请求前预算不足而停，num_requests={res['num_requests']}")


def test_missing_usage():
    print("\n===== 小例 E：缺 usage 即停 =====")

    class FakeGeneratorNoUsage(FakeGenerator):
        def __call__(self, prompt, task_type, model, max_tokens_response=4096):
            self.calls.append({"prompt_len": len(prompt)})
            return "cand_x", None, None

    fake_gen = FakeGeneratorNoUsage([])
    res = _run_independent_resample(
        make_problem(), make_args(), FakeEvaluator({}),
        stop_when_visible_passed=False, generator_func=fake_gen,
    )
    assert res["stopped_reason"] == "Usage missing for resample request", res
    print("[PASS] 缺 usage 时停止，不按 0 放行")


if __name__ == "__main__":
    test_select_highest_visible()
    test_tie_earliest()
    test_stop_when_visible_passed()
    test_budget_insufficient()
    test_missing_usage()
    print("\n[SUCCESS] Independent Resample 无付费自测全部通过！")