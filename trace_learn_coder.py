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
# Import process_problem and _load_check_correctness_func from problem_processor
from problem_processor import process_problem, _load_check_correctness_func
from src.dataset_loader import load_dataset


def get_file_hash(filepath):
    """计算文件的 MD5 哈希值（取前 12 位）"""
    if not os.path.exists(filepath):
        return "not_found"
    with open(filepath, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()[:12]


def main():
    # 新增：配置全局日志，自动保存到 results/run_logs/ 目录
    os.makedirs('results/run_logs', exist_ok=True)
    log_filename = f"results/run_logs/terminal_output_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

    # 配置 logging，同时输出到终端和文件
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )

    # 把 print 替换成 logging.info
    global print
    print = logging.info

    print(f"[OK] 本次运行日志将自动保存至: {log_filename}")

    # 1. Setup and parse arguments
    parser = setup_arg_parser()
    args = parser.parse_args()

    # 2. Generate experiment ID and output filename
    run_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    EXPERIMENT_ID = f"{args.dataset}_{args.model.replace('/', '_')}_{run_time}"
    OUTPUT_CSV_FILENAME = args.output_file or f"results/experiment_{EXPERIMENT_ID}.csv"

    # --- 新增：获取 Git 提交号和是否有未提交修改 ---
    try:
        current_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        current_commit = "未知"

    try:
        git_status = subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        git_dirty = len(git_status) > 0  # 如果有输出，说明有未提交的修改
    except Exception:
        git_dirty = True

    # --- 新增：计算关键文件的 MD5 哈希 ---
    file_hashes = {
        "problem_processor.py": get_file_hash("problem_processor.py"),
        "trace_learn_coder.py": get_file_hash("trace_learn_coder.py"),
        "generation.py": get_file_hash("src/generation.py"),
        "traceRunner.py": get_file_hash("src/traceRunner.py")
    }

    # 新增：自动保存运行元数据（溯源信息）
    run_meta = {
        "run_timestamp": run_time,
        "command_args": sys.argv,
        "model": args.model,
        "dataset": args.dataset,
        "git_commit": current_commit,
        "git_dirty": git_dirty,
        "file_hashes": file_hashes
    }
    meta_path = f'results/run_logs/run_meta_{EXPERIMENT_ID}.json'
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(run_meta, f, indent=4, ensure_ascii=False)
    print(f"[OK] 运行元数据已成功保存至: {meta_path}")

    # 3. Load the dataset
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

    # 4. Set up the problem iterator
    problem_keys = list(datasets.keys())
    start_index = args.start_index
    max_problems = len(problem_keys) if args.max_problems == -1 else args.max_problems
    end_index = min(start_index + max_problems, len(problem_keys))

    problem_iterator = islice(problem_keys, start_index, end_index)

    results_data = []

    # 5. Main loop
    try:
        for task_id in tqdm(problem_iterator, total=(end_index - start_index), desc="Processing problems"):
            problem_result = process_problem(datasets[task_id], task_id, args, check_correctness_func)
            results_data.append(problem_result)
    except KeyboardInterrupt:
        print("\nManual interruption detected. Saving existing results...")
    finally:
        # 6. Save results and print summary
        save_results(results_data, OUTPUT_CSV_FILENAME)
        print_summary(results_data)


if __name__ == '__main__':
    main()