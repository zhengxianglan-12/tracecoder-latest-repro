# fake_stubs.py
# 只放无付费测试用的替身，不参与真跑

class FakeGeneratorForTraceCoder:
    def __init__(self):
        self.calls = []
        self.plan_text = (
            "REPAIR_PLAN_START\n```python\nfix the bug\n```\nREPAIR_PLAN_END\n"
            "Instrumentation_Suggestions_START\n```python\n\n```\nInstrumentation_Suggestions_END"
        )
        self.impl_code = "CODE:\n```python\ncand_19\n```\nEND_CODE"

    def __call__(self, prompt, task_type, model, max_tokens_response=4096):
        self.calls.append({"prompt_len": len(prompt), "task_type": task_type})
        if "Your tasks are:" in prompt and "Step 1. Analyze" in prompt:
            return self.plan_text, 100, 50
        if "Detailed Repair Plan (Follow this plan carefully!)" in prompt:
            return self.impl_code, 100, 50
        if "Key Instrumentation Rules" in prompt or "Code to Instrument" in prompt:
            return "", 100, 50
        return "", 100, 50


class FakeGeneratorForResample:
    def __init__(self, codes):
        self.codes = list(codes)
        self.calls = []

    def __call__(self, prompt, task_type, model, max_tokens_response=4096):
        self.calls.append({"prompt_len": len(prompt)})
        if not self.codes:
            return "cand_fallback", 100, 50
        return self.codes.pop(0), 100, 50


class FakeEvaluator:
    def __init__(self, score_map=None, total=19, default_score=12):
        self.score_map = score_map or {}
        self.total = total
        self.default_score = default_score

    def __call__(self, problem_data, code, timeout):
        score = self.score_map.get(code, self.default_score)
        return {
            "passed": score >= self.total,
            "passed_count": score,
            "failed_count": self.total - score,
            "total_count": self.total,
            "passed_tests": [],
            "failed_tests": [],
        }