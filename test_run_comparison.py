# test_run_comparison.py
# 无付费验证 run_comparison 的“格式与流程”：
#   - 6 个输出文件是否生成
#   - summary.csv 27 列 + 5 个关键列
#   - requests.csv 16 列 + failure_kind 列
#   - summary_rows 两行 + 异常分类字段有值 + failure_kind 两列有值
# 逻辑层（预算拒绝/超限/缺 usage）不在此文件，见 test_budget_matrix.py。
#   python test_run_comparison.py

import os
os.environ.setdefault("DEEPSEEK_API_KEY", "fake-key-for-offline-tests")

import json
import shutil
import csv
from types import SimpleNamespace

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


def main():
    test_results_dir = "results_test_dryrun"
    if os.path.isdir(test_results_dir):
        shutil.rmtree(test_results_dir)
    os.makedirs(f"{test_results_dir}/run_logs", exist_ok=True)
    os.makedirs(f"{test_results_dir}/tables", exist_ok=True)

    # ---------- 1. 取题 ----------
    datasets = load_dataset("classeval", DATASET_PATHS["classeval"]["data_path"])
    task_id = resolve_task(datasets, task_id=None, start_index=0)
    problem_data = datasets[task_id]
    safe_task = task_id.replace("/", "_")
    prefix = f"classeval_{safe_task}_dryrun"

    # ---------- 2. 构造 CLI 参数 ----------
    cli = SimpleNamespace(
        dataset="classeval", task_id=task_id,
        model="fake-model", budget_total=15000, output_reserve=2000,
        max_resamples=10, timeout=10,
        max_debug_attempts=5, max_no_improvement_streak=2,
        stop_when_visible_passed=False, no_instrumentation=True,
        start_index=0,
    )

    log_lines = [f"===== dry-run: {task_id} ====="]
    fake_eval = FakeEvaluator(
        {"cand_19": 19, "cand_13": 13, "cand_12": 12, "cand_10": 10},
        total=19, default_score=12,
    )

    # ---------- 3. 跑 TraceCoder ----------
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

    # ---------- 4. 跑 Resample ----------
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

    # ---------- 5. 写 6 个文件 ----------
    summary_rows = write_outputs(
        task_id, safe_task, "dryrun", cli,
        trace_res, resample_res, args_trace, args_resample, log_lines,
        datasets,
        results_dir=test_results_dir,
    )

    # ---------- 6. 汇总行字段断言 ----------
    assert len(summary_rows) == 2, summary_rows
    assert summary_rows[0]["method"] == "tracecoder", summary_rows[0]
    assert summary_rows[1]["method"] == "independent_resample", summary_rows[1]
    for row in summary_rows:
        assert "stopped_by_budget_rejection" in row, row
        assert "stopped_by_usage_unverifiable" in row, row
        assert row.get("sample_class") in ("normal", "overrun", "unknown", "rejected"), row
        assert isinstance(row["stopped_by_budget_rejection"], bool), row
        assert isinstance(row["stopped_by_usage_unverifiable"], bool), row
        # 10-31 新增：failure_kind 两列必须有值
        assert "num_requests_response_missing" in row, row
        assert "num_requests_usage_missing" in row, row

    # ---------- 7. summary.csv：27 列 + 5 个关键列 ----------
    summary_csv_path = f"{test_results_dir}/tables/{prefix}_summary.csv"
    with open(summary_csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        headers = next(reader)
    assert len(headers) == 27, f"summary.csv 应为 27 列，实际 {len(headers)} 列：{headers}"
    for col in (
        "stopped_by_budget_rejection",
        "stopped_by_usage_unverifiable",
        "sample_class",
        "num_requests_response_missing",
        "num_requests_usage_missing",
    ):
        assert col in headers, f"summary.csv 缺列：{col}"

    # ---------- 8. requests.csv：16 列 + failure_kind ----------
    requests_csv_path = f"{test_results_dir}/tables/{prefix}_requests.csv"
    with open(requests_csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        req_headers = next(reader)
    assert len(req_headers) == 16, (
        f"requests.csv 应为 16 列，实际 {len(req_headers)} 列：{req_headers}"
    )
    assert "failure_kind" in req_headers, f"requests.csv 缺列：failure_kind"

    # ---------- 9. 6 个输出文件都必须存在 ----------
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

    # ---------- 10. 回填 output_files_changed ----------
    meta_path = f"{test_results_dir}/run_logs/{prefix}_run_meta.json"
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    meta["output_files_changed"] = sorted(expected)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4, ensure_ascii=False)
    print(f"[OK] output_files_changed 已回填至: {meta_path}（{len(expected)} 个文件）")

    print(f"\n[SUCCESS] dry-run 通过，输出在 {test_results_dir}/")


if __name__ == "__main__":
    main()