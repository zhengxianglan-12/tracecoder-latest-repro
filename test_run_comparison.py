# test_run_comparison.py
# 无付费验证 run_comparison 的格式与流程
#   python test_run_comparison.py

import os
import json
import shutil
from types import SimpleNamespace

import problem_processor as pp
from fake_stubs import (
    FakeGeneratorForTraceCoder,
    FakeGeneratorForResample,
    FakeEvaluator,
)
from run_comparison import (
    resolve_task,
    run_tracecoder,
    run_resample,
    write_outputs,
)
from config import DATASET_PATHS
from src.dataset_loader import load_dataset
def test_resample_budget_rejected():
    """预算不足 → 拒绝事件进 requests，三数分开（sent=0, rejected=1）。"""
    from independent_resample import _run_independent_resample

    problem_data = {"complete_prompt": "x" * 1000, "test": ""}
    args = SimpleNamespace(
        model="fake-model", timeout=10, output_reserve=2000,
        budget_total=100, remaining_budget=100,   # 远小于 估算 457 + 2000
        max_resamples=10,
    )

    def fake_gen(*a, **kw):
        raise AssertionError("预算不足时不应发请求")

    res = _run_independent_resample(
        problem_data, args, lambda *a: {"passed": False},
        generator_func=fake_gen,
    )
    assert res["num_requests"] == 1, res
    assert res["num_requests_sent"] == 0, res
    assert res["num_requests_rejected"] == 1, res
    assert res["num_requests_usage_unverifiable"] == 0, res
    assert res["requests"][0]["sent"] is False
    assert res["requests"][0]["request_id"] == "RJ01"
    assert res["requests"][0]["stopped_reason"] == "Budget insufficient before resample request"
    print("[PASS] resample 预算拒绝：num_requests=1, sent=0, rejected=1")

def main():
    test_results_dir = "results_test_dryrun"
    if os.path.isdir(test_results_dir):
        shutil.rmtree(test_results_dir)
    os.makedirs(f"{test_results_dir}/run_logs", exist_ok=True)
    os.makedirs(f"{test_results_dir}/tables", exist_ok=True)

    datasets = load_dataset("classeval", DATASET_PATHS["classeval"]["data_path"])
    task_id = resolve_task(datasets, task_id=None, start_index=0)
    problem_data = datasets[task_id]
    safe_task = task_id.replace("/", "_")

    cli = SimpleNamespace(
        dataset="classeval", task_id=task_id,
        model="fake-model", budget_total=15000, output_reserve=2000,
        max_resamples=10, timeout=10,
        max_debug_attempts=5, max_no_improvement_streak=2,
        stop_when_visible_passed=False, no_instrumentation=True,
    )

    log_lines = [f"===== dry-run: {task_id} ====="]
    fake_eval = FakeEvaluator({"cand_19": 19, "cand_13": 13, "cand_12": 12, "cand_10": 10},
                              total=19, default_score=12)

    args_trace = SimpleNamespace(
        model=cli.model, dataset=cli.dataset,
        no_instrumentation=cli.no_instrumentation, no_two_step_repair=False,
        start_index=0, max_problems=1,
        max_debug_attempts=cli.max_debug_attempts,
        max_no_improvement_streak=cli.max_no_improvement_streak,
        timeout=cli.timeout, output_file=None,
        budget_total=cli.budget_total, remaining_budget=cli.budget_total,
        output_reserve=cli.output_reserve,
    )
    trace_res = run_tracecoder(
        problem_data, task_id, args_trace, fake_eval, log_lines,
        generator_func=FakeGeneratorForTraceCoder(),
    )
    assert trace_res["final_passed"] is True, trace_res.get("stopped_reason")

    args_resample = SimpleNamespace(
        model=cli.model, timeout=cli.timeout,
        budget_total=cli.budget_total, remaining_budget=cli.budget_total,
        output_reserve=cli.output_reserve, max_resamples=cli.max_resamples,
    )
    resample_res = run_resample(
        problem_data, args_resample, fake_eval, log_lines,
        stop_when_visible_passed=cli.stop_when_visible_passed,
        generator_func=FakeGeneratorForResample(["cand_10", "cand_13", "cand_12", "cand_19"]),
    )
    assert resample_res["final_visible_passed_count"] == 19, resample_res

    summary_rows = write_outputs(
        task_id, safe_task, "dryrun", cli,
        trace_res, resample_res, args_trace, args_resample, log_lines,
        results_dir=test_results_dir,
    )
    assert len(summary_rows) == 2
    assert summary_rows[0]["method"] == "tracecoder"
    assert summary_rows[1]["method"] == "independent_resample"

    prefix = f"classeval_{safe_task}_dryrun"
    expected = [
        f"{test_results_dir}/run_logs/{prefix}_run_meta.json",
        f"{test_results_dir}/run_logs/{prefix}_debug_session_log_tracecoder.json",
        f"{test_results_dir}/run_logs/{prefix}_candidates_resample.json",
        f"{test_results_dir}/run_logs/{prefix}_terminal_output.txt",
        f"{test_results_dir}/tables/{prefix}_requests.csv",
        f"{test_results_dir}/tables/{prefix}_summary.csv",
    ]
    for p in expected:
        assert os.path.exists(p), f"缺少输出文件: {p}"

    # 回填 output_files_changed：与 run_comparison.main() 口径一致
    meta_path = f"{test_results_dir}/run_logs/{prefix}_run_meta.json"
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    meta["output_files_changed"] = sorted(expected)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4, ensure_ascii=False)
    print(f"[OK] output_files_changed 已回填至: {meta_path}（{len(expected)} 个文件）")

    test_resample_budget_rejected()
    print(f"\n[SUCCESS] dry-run 通过，输出在 {test_results_dir}/")


if __name__ == "__main__":
    main()