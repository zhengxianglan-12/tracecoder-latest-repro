import sys
import time
import datetime
import json
import os
import logging
import subprocess
import hashlib
from itertools import islice
from tqdm import tqdm

# Import our split-out modules
from config import setup_arg_parser, DATASET_PATHS
from reporting import save_results, print_summary
from problem_processor import process_problem, _load_check_correctness_func
from src.dataset_loader import load_dataset


def get_file_hash(filepath):
    """计算文件的 MD5 哈希值（取前 12 位，按原始字节，CRLF 保留）"""
    if not os.path.exists(filepath):
        return "not_found"
    with open(filepath, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()[:12]


def get_file_hash_lf(filepath):
    """按 LF 归一化后的字节计算 MD5，用于跨平台对比"""
    if not os.path.exists(filepath):
        return "not_found"
    with open(filepath, "rb") as f:
        data = f.read().replace(b"\r\n", b"\n")
    return hashlib.md5(data).hexdigest()[:12]


# ============================================================
# 实际执行依赖清单（code_dirty 的检查范围）
# 含：主流程、调试逻辑、配置、数据加载、生成/提示词、执行器、评测器
# 路径按你仓库实际结构填写；不存在的路径不会报错，只是不参与判定
# ============================================================
CODE_DEPENDENCY_FILES = [
    "trace_learn_coder.py",
    "problem_processor.py",
    "config.py",
    "reporting.py",
    "src/generation.py",                  # 生成 + 提示词相关
    "src/traceRunner.py",                 # 执行器
    "src/dataset_loader.py",              # 加载器
    "src/postprocessing.py",
    "datasets/ClassEval/evaluation.py",   # 评测器（按实际路径改）
]


def _git_lines(args_list):
    """执行 git 命令，返回按行切分的非空输出；失败返回空列表。"""
    try:
        out = subprocess.check_output(
            ["git"] + args_list, text=True, stderr=subprocess.DEVNULL
        )
        return [line for line in out.splitlines() if line.strip()]
    except Exception:
        return []


def compute_code_dirty(dependency_files):
    """
    判定“实际执行依赖”是否脏：
    - 工作区/暂存区：git status --porcelain 中与依赖文件相关的行
    - 未跟踪：git ls-files --others --exclude-standard 中与依赖文件相关的行
    返回 (code_dirty: bool, dirty_files: list[str])
    """
    deps = {f.replace("\\", "/") for f in dependency_files}
    dirty = []

    # 工作区 + 暂存区（--porcelain 同时覆盖两者）
    for line in _git_lines(["status", "--porcelain"]):
        # porcelain 格式：XY path，前 3 个字符是状态位
        path = line[3:].strip().strip('"').replace("\\", "/")
        if path in deps:
            dirty.append(path)

    # 未跟踪文件
    for line in _git_lines(["ls-files", "--others", "--exclude-standard"]):
        path = line.strip().strip('"').replace("\\", "/")
        if path in deps:
            dirty.append(path)

    dirty = sorted(set(dirty))
    return (len(dirty) > 0), dirty


def compute_output_files_changed(results_dir="results"):
    """记录运行前 results 目录下已存在文件的 mtime，用于运行后对比新增/变更。"""
    existing = {}
    if not os.path.isdir(results_dir):
        return existing
    for root, _, files in os.walk(results_dir):
        for name in files:
            full = os.path.join(root, name)
            try:
                existing[full] = os.path.getmtime(full)
            except OSError:
                pass
    return existing


def main():
    # ==================================================
    # 第 1 步：先检查 Git 状态（此时工作区是干净的）
    # ==================================================
    try:
        current_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        current_commit = "未知"

    # git_dirty 保留原义：整个仓库工作区是否有未提交改动
    try:
        git_status = subprocess.check_output(
            ["git", "status", "--porcelain"], text=True
        ).strip()
        git_dirty = len(git_status) > 0
    except Exception:
        git_dirty = True

    # repo_dirty 与 git_dirty 同义，单独留字段，便于阅读（全库）
    repo_dirty = git_dirty

    # code_dirty：仅覆盖实际执行依赖（工作区/暂存区/未跟踪）
    code_dirty, code_dirty_files = compute_code_dirty(CODE_DEPENDENCY_FILES)

    # 运行前 results 目录快照，用于运行后算输出文件变动清单
    output_snapshot_before = compute_output_files_changed("results")

    # 核心文件哈希：原始字节 + LF 归一化两种口径
    file_hashes = {
        "problem_processor.py": get_file_hash("problem_processor.py"),
        "trace_learn_coder.py": get_file_hash("trace_learn_coder.py"),
        "generation.py": get_file_hash("src/generation.py"),
        "traceRunner.py": get_file_hash("src/traceRunner.py"),
    }
    file_hashes_lf = {
        "problem_processor.py": get_file_hash_lf("problem_processor.py"),
        "trace_learn_coder.py": get_file_hash_lf("trace_learn_coder.py"),
        "generation.py": get_file_hash_lf("src/generation.py"),
        "traceRunner.py": get_file_hash_lf("src/traceRunner.py"),
    }

    # ==================================================
    # 第 2 步：解析参数，并立即把元数据写进 JSON 文件
    # ==================================================
    parser = setup_arg_parser()
    args = parser.parse_args()
    # 预算控制：剩余预算初值（真正每题重置在 process_problem 开头）
    args.remaining_budget = args.budget_total

    run_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    EXPERIMENT_ID = f"{args.dataset}_{args.model.replace('/', '_')}_{run_time}"
    OUTPUT_CSV_FILENAME = args.output_file or f"results/experiment_{EXPERIMENT_ID}.csv"

    os.makedirs('results/run_logs', exist_ok=True)
    meta_path = f'results/run_logs/run_meta_{EXPERIMENT_ID}.json'

    run_meta = {
        "run_timestamp": run_time,
        "command_args": sys.argv,
        "model": args.model,
        "dataset": args.dataset,
        "git_commit": current_commit,

        # 全库状态：保留原义
        "git_dirty": git_dirty,
        "repo_dirty": repo_dirty,

        # 实际执行依赖状态：另开字段
        "code_dirty": code_dirty,
        "code_dirty_files": code_dirty_files,
        "code_dependency_files": CODE_DEPENDENCY_FILES,

        # 核心文件哈希（原始字节 + LF 归一化）
        "file_hashes": file_hashes,
        "file_hashes_lf_normalized": file_hashes_lf,
        "file_hash_note": (
            "file_hashes 为原始字节 MD5（保留 CRLF）；"
            "file_hashes_lf_normalized 为 LF 归一化后 MD5；"
            "跨平台对比请使用 LF 归一化口径"
        ),
        "budget_control": {
            "mode": "estimate_admission_with_post_hoc_reconciliation",
            "hard_cap_guaranteed": False,
            "estimate_includes_system_message": True,
            "system_overhead_chars": 307,
            "note": (
                "准入按估算输入+输出预留判断，不保证硬上限；"
                "请求后按实际 usage 核账，余额为负时标记 budget_overrun 并停止。"
            ),
        },
        "retry_policy": {
            "sdk_max_retries": 0,
            "generator_internal_retries": 0,
            "retry_cost_recorded_in_request_log": False,
            "note": (
                "10-29 首轮策略：关闭 SDK 层（OpenAI max_retries=0）与 generator 层"
                "（原 max_retries=5 的 while 循环已移除）两层隐式重试；"
                "一次逻辑调用对应一次底层请求。"
                "失败请求如实记为 sent=True 且用量未知（None），不写 0；"
                "预算无法核验时按预先规则停止并保留异常题。"
            ),
        },

        # 输出文件变动清单（运行结束后回填）
        "output_files_changed": [],
    }
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(run_meta, f, indent=4, ensure_ascii=False)

    # ==================================================
    # 第 3 步：创建日志文件（JSON 已经写好了）
    # ==================================================
    log_filename = f"results/run_logs/terminal_output_{run_time}.txt"
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler(sys.stdout),
        ],
    )

    global print
    print = logging.info
    print(f"[OK] 本次运行日志将自动保存至: {log_filename}")
    print(f"[OK] 运行元数据已成功保存至: {meta_path}")
    print(f"[OK] git_dirty={git_dirty} (全库) | repo_dirty={repo_dirty} | "
          f"code_dirty={code_dirty} (依赖文件: {code_dirty_files})")

    # ==================================================
    # 第 4 步：跑实验
    # ==================================================
    if args.dataset not in DATASET_PATHS:
        print(f"Error: Path for dataset '{args.dataset}' is not defined in config.py.")
        sys.exit(1)

    try:
        actual_dataset_path = DATASET_PATHS[args.dataset]["data_path"]
        datasets = load_dataset(args.dataset, actual_dataset_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)
    except KeyError:
        print(f"Error: The configuration for dataset '{args.dataset}' is missing "
              f"'data_path'. Please check config.py.")
        sys.exit(1)

    try:
        check_correctness_func = _load_check_correctness_func(args.dataset)
    except (ValueError, ImportError) as e:
        print(f"Error loading the evaluation function check_correctness: {e}")
        sys.exit(1)

    print(f"--- Experiment Start: {EXPERIMENT_ID} ---")
    print(f"Arguments: {vars(args)}")

    problem_keys = list(datasets.keys())
    start_index = args.start_index
    max_problems = len(problem_keys) if args.max_problems == -1 else args.max_problems
    end_index = min(start_index + max_problems, len(problem_keys))

    problem_iterator = islice(problem_keys, start_index, end_index)

    results_data = []

    try:
        for task_id in tqdm(problem_iterator, total=(end_index - start_index),
                            desc="Processing problems"):
            problem_result = process_problem(
                datasets[task_id], task_id, args, check_correctness_func
            )
            results_data.append(problem_result)
    except KeyboardInterrupt:
        print("\nManual interruption detected. Saving existing results...")
    finally:
        save_results(results_data, OUTPUT_CSV_FILENAME)
        overrun_tasks = [r.get("task_id") for r in results_data if r.get("budget_overrun")]
        if overrun_tasks:
            print(f"[WARN] 以下题目发生预算超限（不计入正常预算样本）: {overrun_tasks}")
        print_summary(results_data)

        # 回填输出文件变动清单：运行后与运行前快照对比
        snapshot_after = compute_output_files_changed("results")
        changed = []
        for path, mtime in snapshot_after.items():
            if path not in output_snapshot_before or output_snapshot_before[path] != mtime:
                changed.append(path)
        run_meta["output_files_changed"] = sorted(changed)

        # 重写 run_meta（覆盖第 2 步写的那份，补上变动清单）
        try:
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(run_meta, f, indent=4, ensure_ascii=False)
            print(f"[OK] 输出文件变动清单已回填至: {meta_path}")
        except Exception as e:
            print(f"[WARN] 回填 run_meta 失败: {e}")


if __name__ == '__main__':
    main()