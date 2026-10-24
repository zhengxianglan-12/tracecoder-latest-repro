# run_comparison.py
# 通用等预算对照：TraceCoder vs Independent Resample（真跑编排，测试见 test_run_comparison.py）
#
# 用法：
#   python run_comparison.py --start-index 0 --budget-total 15000 --model deepseek-chat
#   python run_comparison.py --task-id ClassEval_23 --budget-total 15000 --model deepseek-chat
#
# 输出（均带 task_id，不会互相覆盖）：
#   results/run_logs/run_meta_<task>_<model>_<ts>_tracecoder.json
#   results/run_logs/run_meta_<task>_<model>_<ts>_resample.json
#   results/run_logs/terminal_output_<task>_<ts>_tracecoder.txt
#   results/run_logs/terminal_output_<task>_<ts>_combined.txt
#   results/tables/tracecoder_requests_<task>_<ts>.csv
#   results/tables/resample_requests_<task>_<ts>.csv
#   results/tables/comparison_<task>_summary_<ts>.csv
#   results/tables/comparison_<task>_summary.csv

import os
import sys
import json
import time
import hashlib
import datetime
import argparse
import subprocess
from types import SimpleNamespace

from config import DATASET_PATHS
from src.dataset_loader import load_dataset
from problem_processor import (
    process_problem,
    _load_check_correctness_func,
)
from independent_resample import (
    _run_independent_resample,
    SELECTION_RULE,
    EVALUATION_SCOPE,
)


# ============================================================
# 工具函数
# ============================================================
def _get_hash(code_str):
    return hashlib.md5(code_str.encode("utf-8")).hexdigest()[:8] if code_str else "empty"


def _file_hash(path):
    if not os.path.exists(path):
        return "not_found"
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()[:12]


def _file_hash_lf(path):
    if not os.path.exists(path):
        return "not_found"
    with open(path, "rb") as f:
        data = f.read().replace(b"\r\n", b"\n")
    return hashlib.md5(data).hexdigest()[:12]


def _git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "未知"


def _git_dirty():
    try:
        out = subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        return len(out) > 0
    except Exception:
        return True


def _write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4, ensure_ascii=False)
    print(f"[OK] 写出: {path}")


def _write_text(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"[OK] 写出: {path}")


def _write_csv(path, rows, headers):
    import csv
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for r in rows:
            w.writerow({h: r.get(h, "") for h in headers})
    print(f"[OK] 写出: {path}")


# ============================================================
# 取题
# ============================================================
def resolve_task(datasets, task_id=None, start_index=0):
    problem_keys = list(datasets.keys())
    if task_id:
        if task_id not in datasets:
            print(f"Error: task_id '{task_id}' not found in dataset.")
            sys.exit(1)
        return task_id
    if start_index < 0 or start_index >= len(problem_keys):
        print(f"Error: start_index {start_index} out of range [0, {len(problem_keys)-1}].")
        sys.exit(1)
    return problem_keys[start_index]


# ============================================================
# 跑两组
# ============================================================
def run_tracecoder(problem_data, task_id, args, check_correctness_func, log_lines,
                   generator_func=None):
    log_lines.append(f"\n===== TraceCoder 组开始：{task_id} =====")
    t0 = time.time()

    if generator_func is not None:
        import problem_processor as pp
        original_gen = pp.generator
        pp.generator = generator_func
        try:
            result = process_problem(problem_data, task_id, args, check_correctness_func)
        finally:
            pp.generator = original_gen
    else:
        result = process_problem(problem_data, task_id, args, check_correctness_func)

    log_lines.append(f"TraceCoder 用时: {time.time() - t0:.2f}s")
    log_lines.append(f"TraceCoder final_passed: {result.get('final_passed')}")
    log_lines.append(f"TraceCoder stopped_reason: {result.get('stopped_reason')}")
    log_lines.append(f"TraceCoder total_prompt_tokens: {result.get('total_prompt_tokens')}")
    log_lines.append(f"TraceCoder total_completion_tokens: {result.get('total_completion_tokens')}")
    log_lines.append(f"TraceCoder final_code_hash: {_get_hash(result.get('final_code', ''))}")
    log_lines.append(f"===== TraceCoder 最终代码 =====\n{result.get('final_code', '')}\n========================")
    return result


def run_resample(problem_data, args, check_correctness_func, log_lines,
                 stop_when_visible_passed=False, generator_func=None):
    log_lines.append("\n===== Independent Resample 组开始 =====")
    t0 = time.time()
    result = _run_independent_resample(
        problem_data, args, check_correctness_func,
        stop_when_visible_passed=stop_when_visible_passed,
        generator_func=generator_func,
    )
    log_lines.append(f"Resample 用时: {time.time() - t0:.2f}s")
    log_lines.append(f"Resample num_requests: {result.get('num_requests')}")
    log_lines.append(f"Resample stopped_reason: {result.get('stopped_reason')}")
    log_lines.append(f"Resample prompt_tokens: {result.get('prompt_tokens')}")
    log_lines.append(f"Resample completion_tokens: {result.get('completion_tokens')}")
    log_lines.append(f"Resample final_visible_passed_count: {result.get('final_visible_passed_count')}")
    log_lines.append(f"Resample final_code_hash: {result.get('final_code_hash')}")
    log_lines.append(f"===== Resample 最终代码 =====\n{result.get('final_code', '')}\n========================")
    return result


# ============================================================
# 写输出
# ============================================================
# ============================================================
# 写输出（5 文件方案）
#   <dataset>_<task>_<ts>_run_meta.json
#   <dataset>_<task>_<ts>_trace.json
#   <dataset>_<task>_<ts>_terminal_output.txt
#   <dataset>_<task>_<ts>_requests.csv
#   <dataset>_<task>_<ts>_summary.csv
# ============================================================
def write_outputs(task_id, safe_task, run_time, cli,
                  trace_res, resample_res,
                  args_trace, args_resample,
                  log_lines, results_dir="results"):
    prefix = f"{cli.dataset}_{safe_task}_{run_time}"

    # ---------- 1. run_meta（公共字段 + methods 子字典） ----------
    run_meta = {
        "run_timestamp": run_time,
        "dataset": cli.dataset,
        "task_id": task_id,
        "model": cli.model,
        "git_commit": _git_commit(),
        "git_dirty": _git_dirty(),
        "code_dirty": _git_dirty(),
        "budget_total": cli.budget_total,
        "output_reserve": cli.output_reserve,
        "selection_rule": SELECTION_RULE,
        "evaluation_scope": EVALUATION_SCOPE,
        "file_hashes": {
            "problem_processor.py": _file_hash("problem_processor.py"),
            "independent_resample.py": _file_hash("independent_resample.py"),
            "generation.py": _file_hash("src/generation.py"),
            "traceRunner.py": _file_hash("src/traceRunner.py"),
        },
        "file_hashes_lf_normalized": {
            "problem_processor.py": _file_hash_lf("problem_processor.py"),
            "independent_resample.py": _file_hash_lf("independent_resample.py"),
            "generation.py": _file_hash_lf("src/generation.py"),
            "traceRunner.py": _file_hash_lf("src/traceRunner.py"),
        },
        "step_output_reserve": {
            "direct_generation": 2000,
            "instrumentation": 2000,
            "analysis_planning": 4096,
            "implementation": 3000,
        },
        "step_input_estimate_ratio": {
            "direct_generation": 0.35,
            "instrumentation": 0.35,
            "analysis_planning": 0.40,
            "implementation": 0.35,
        },
        "methods": {
            "tracecoder": {
                "no_instrumentation": cli.no_instrumentation,
                "max_debug_attempts": cli.max_debug_attempts,
                "max_no_improvement_streak": cli.max_no_improvement_streak,
                "remaining_budget_after": args_trace.remaining_budget,
            },
            "independent_resample": {
                "max_resamples": cli.max_resamples,
                "stop_when_visible_passed": cli.stop_when_visible_passed,
                "remaining_budget_after": args_resample.remaining_budget,
            },
        },
    }
    _write_json(f"{results_dir}/run_logs/{prefix}_run_meta.json", run_meta)

    # ---------- 2. trace（TraceCoder 逐轮 + Resample 逐候选） ----------
    trace = {
        "dataset": cli.dataset,
        "task_id": task_id,
        "run_timestamp": run_time,
        "tracecoder": {
            "debug_session_log": trace_res.get("debug_session_log", []),
        },
        "independent_resample": {
            "candidates": resample_res.get("candidates", []),
            "selection_rule": resample_res.get("selection_rule"),
            "evaluation_scope": resample_res.get("evaluation_scope"),
        },
    }
    _write_json(f"{results_dir}/run_logs/{prefix}_trace.json", trace)

    # ---------- 3. terminal_output（两组合并日志） ----------
    _write_text(f"{results_dir}/run_logs/{prefix}_terminal_output.txt",
                "\n".join(log_lines))

    # ---------- 4. requests（两组合并，method 区分） ----------
    trace_requests = []
    for r in trace_res.get("request_log", []) or []:
        trace_requests.append({
            "method": "tracecoder",
            "request_id": r.get("attempt"),
            "stage": r.get("stage"),
            "prompt_tokens": r.get("prompt_tokens"),
            "completion_tokens": r.get("completion_tokens"),
            "remaining_after": r.get("remaining_after"),
            "visible_passed_count": None,
            "status": r.get("stage"),
        })
    resample_requests = []
    for r in resample_res.get("requests", []) or []:
        resample_requests.append({
            "method": "independent_resample",
            "request_id": r.get("request_id"),
            "stage": "resample",
            "prompt_tokens": r.get("prompt_tokens"),
            "completion_tokens": r.get("completion_tokens"),
            "remaining_after": r.get("remaining_after"),
            "visible_passed_count": None,
            "status": "resample",
        })
    all_requests = trace_requests + resample_requests
    req_headers = ["method", "request_id", "stage", "prompt_tokens", "completion_tokens",
                   "remaining_after", "visible_passed_count", "status"]
    _write_csv(f"{results_dir}/tables/{prefix}_requests.csv",
               all_requests, req_headers)

    # ---------- 5. summary（两组横向对比） ----------
    summary_rows = [
        {
            "task_id": task_id, "method": "tracecoder",
            "num_requests": len(trace_res.get("request_log", []) or []),
            "prompt_tokens": trace_res.get("total_prompt_tokens"),
            "completion_tokens": trace_res.get("total_completion_tokens"),
            "stopped_reason": trace_res.get("stopped_reason"),
            "final_visible_passed_count": trace_res.get("final_eval", {}).get("passed_count"),
            "final_code_hash": _get_hash(trace_res.get("final_code", "")),
            "budget_remaining_after": args_trace.remaining_budget,
            "evaluation_scope": EVALUATION_SCOPE,
            "selection_rule": "tracecoder_internal_best",
        },
        {
            "task_id": task_id, "method": "independent_resample",
            "num_requests": resample_res.get("num_requests"),
            "prompt_tokens": resample_res.get("prompt_tokens"),
            "completion_tokens": resample_res.get("completion_tokens"),
            "stopped_reason": resample_res.get("stopped_reason"),
            "final_visible_passed_count": resample_res.get("final_visible_passed_count"),
            "final_code_hash": resample_res.get("final_code_hash"),
            "budget_remaining_after": args_resample.remaining_budget,
            "evaluation_scope": resample_res.get("evaluation_scope"),
            "selection_rule": resample_res.get("selection_rule"),
        },
    ]
    summary_headers = ["task_id", "method", "num_requests", "prompt_tokens", "completion_tokens",
                       "stopped_reason", "final_visible_passed_count", "final_code_hash",
                       "budget_remaining_after", "evaluation_scope", "selection_rule"]
    _write_csv(f"{results_dir}/tables/{prefix}_summary.csv",
               summary_rows, summary_headers)

    return summary_rows


# ============================================================
# 主流程
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="TraceCoder vs Independent Resample 等预算对照")
    parser.add_argument("--dataset", type=str, default="classeval")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--task-id", type=str, default=None)
    parser.add_argument("--model", type=str, default="deepseek-chat")
    parser.add_argument("--budget-total", type=int, default=15000)
    parser.add_argument("--output-reserve", type=int, default=2000)
    parser.add_argument("--max-resamples", type=int, default=200)
    parser.add_argument("--timeout", type=int, default=10)
    parser.add_argument("--max-debug-attempts", type=int, default=5)
    parser.add_argument("--max-no-improvement-streak", type=int, default=2)
    parser.add_argument("--stop-when-visible-passed", action="store_true")
    parser.add_argument("--no-instrumentation", action="store_true")
    cli = parser.parse_args()

    run_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("results/run_logs", exist_ok=True)
    os.makedirs("results/tables", exist_ok=True)

    if cli.dataset not in DATASET_PATHS:
        print(f"Error: dataset '{cli.dataset}' not in DATASET_PATHS.")
        sys.exit(1)
    datasets = load_dataset(cli.dataset, DATASET_PATHS[cli.dataset]["data_path"])
    task_id = resolve_task(datasets, cli.task_id, cli.start_index)
    problem_data = datasets[task_id]
    safe_task = task_id.replace("/", "_")

    log_lines = [
        f"===== {task_id} 等预算对照 =====",
        f"run_time: {run_time}", f"dataset: {cli.dataset}", f"task_id: {task_id}",
        f"model: {cli.model}", f"budget_total: {cli.budget_total}",
        f"output_reserve: {cli.output_reserve}",
        f"stop_when_visible_passed: {cli.stop_when_visible_passed}",
        f"no_instrumentation: {cli.no_instrumentation}",
        f"selection_rule: {SELECTION_RULE}", f"evaluation_scope: {EVALUATION_SCOPE}",
    ]

    check_correctness_func = _load_check_correctness_func(cli.dataset)

    args_trace = SimpleNamespace(
        model=cli.model, dataset=cli.dataset,
        no_instrumentation=cli.no_instrumentation, no_two_step_repair=False,
        start_index=cli.start_index, max_problems=1,
        max_debug_attempts=cli.max_debug_attempts,
        max_no_improvement_streak=cli.max_no_improvement_streak,
        timeout=cli.timeout, output_file=None,
        budget_total=cli.budget_total, remaining_budget=cli.budget_total,
        output_reserve=cli.output_reserve,
    )
    trace_res = run_tracecoder(problem_data, task_id, args_trace,
                               check_correctness_func, log_lines)

    args_resample = SimpleNamespace(
        model=cli.model, timeout=cli.timeout,
        budget_total=cli.budget_total, remaining_budget=cli.budget_total,
        output_reserve=cli.output_reserve, max_resamples=cli.max_resamples,
    )
    resample_res = run_resample(problem_data, args_resample, check_correctness_func,
                                log_lines, stop_when_visible_passed=cli.stop_when_visible_passed)

    summary_rows = write_outputs(task_id, safe_task, run_time, cli,
                                 trace_res, resample_res,
                                 args_trace, args_resample, log_lines)

    print("\n========== 对照汇总 ==========")
    for row in summary_rows:
        print(json.dumps(row, ensure_ascii=False, indent=2))
    print(f"\n[OK] 全部输出已写入 results/（task_id={task_id}, run_time={run_time}）")


if __name__ == "__main__":
    main()