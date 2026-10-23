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
    """计算文件的 MD5 哈希值（取前 12 位）"""
    if not os.path.exists(filepath):
        return "not_found"
    with open(filepath, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()[:12]


def main():
    # ==================================================
    # 第 1 步：先检查 Git 状态（此时还没创建任何日志文件，工作区是绝对干净的）
    # ==================================================
    try:
        current_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        current_commit = "未知"

    try:
        git_status = subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        git_dirty = len(git_status) > 0
    except Exception:
        git_dirty = True

    file_hashes = {
        "problem_processor.py": get_file_hash("problem_processor.py"),
        "trace_learn_coder.py": get_file_hash("trace_learn_coder.py"),
        "generation.py": get_file_hash("src/generation.py"),
        "traceRunner.py": get_file_hash("src/traceRunner.py")
    }

    # ==================================================
    # 第 2 步：再创建日志目录和文件
    # ==================================================
    os.makedirs('results/run_logs', exist_ok=True)
    log_filename = f"results/run_logs/terminal_output_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )

    global print
    print = logging.info
    print(f"[OK] 本次运行日志将自动保存至: {log_filename}")

    # ==================================================
    # 第 3 步：解析参数，准备实验
    # ==================================================
    parser = setup_arg_parser()
    args = parser.parse_args()

    run_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    EXPERIMENT_ID = f"{args.dataset}_{args.model.replace('/', '_')}_{run_time}"
    OUTPUT_CSV_FILENAME = args.output_file or f"results/experiment_{EXPERIMENT_ID}.csv"

    meta_path = f'results/run_logs/run_meta_{EXPERIMENT_ID}.json'

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
        print(f"Error: The configuration for dataset '{args.dataset}' is missing 'data_path'. Please check config.py.")
        sys.exit(1)

    try:
        check_correctness_func = _load_check_correctness_func(args.dataset)
    except (ValueError, ImportError) as e:
        print(f"Error loading the evaluation function check_correctness: {e}")
        sys.exit(1)

    print(f"--- Experiment Start: {EXPERIMENT_ID} ---")
    print(f"Arguments: {vars(args)}")

    # ==================================================
    # 第 4 步：跑实验
    # ==================================================
    problem_keys = list(datasets.keys())
    start_index = args.start_index
    max_problems = len(problem_keys) if args.max_problems == -1 else args.max_problems
    end_index = min(start_index + max_problems, len(problem_keys))

    problem_iterator = islice(problem_keys, start_index, end_index)

    results_data = []

    try:
        for task_id in tqdm(problem_iterator, total=(end_index - start_index), desc="Processing problems"):
            problem_result = process_problem(datasets[task_id], task_id, args, check_correctness_func)
            results_data.append(problem_result)
    except KeyboardInterrupt:
        print("\nManual interruption detected. Saving existing results...")
    finally:
        # 保存 CSV
        save_results(results_data, OUTPUT_CSV_FILENAME)
        print_summary(results_data)

        # ==================================================
        # 第 5 步：最后保存元数据（用的是第1步检查的结果）
        # ==================================================
        run_meta = {
            "run_timestamp": run_time,
            "command_args": sys.argv,
            "model": args.model,
            "dataset": args.dataset,
            "git_commit": current_commit,
            "git_dirty": git_dirty,
            "file_hashes": file_hashes
        }
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(run_meta, f, indent=4, ensure_ascii=False)
        print(f"[OK] 运行元数据（跑前状态）已保存至: {meta_path}")


if __name__ == '__main__':
    main()