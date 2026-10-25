import re
import time
from itertools import islice
import importlib 
import logging
import hashlib
from typing import Callable 

# Import functions from sibling modules
from config import DATASET_PATHS 
from reporting import format_check_correctness_result
from src.generation import generator 
from src.traceRunner import execute_code_and_capture_prints_last  # <<< New import
from src.postprocessing import remove_main_block  # <<< New import for _parse_llm_response


# --- Helper Functions ---
def _parse_llm_output(llm_response: str):
    """ 
    Parses the LLM's response string to extract the repair plan and 
    instrumentation suggestions. 
    The function looks for specific start and end markers to delimit the sections.
    It also cleans up common artifacts like surrounding whitespace and code 
    fences (e.g., ```python ... ```). 
    Args: 
        llm_response: The full string output from the language model. 
    Returns: 
        A tuple containing the extracted 'repair_plan' and 
        'instrumentation_suggestions'. If a section is not found, 
        its value will be None. 
    """ 
    def extract_and_clean_section(text: str, start_marker: str, end_marker: str) -> str | None: 
        """Helper function to extract content between two markers.""" 
        # Use re.DOTALL to make '.' match newlines, which is crucial for multiline content. 
        # Use a non-greedy match '.*?' to find the shortest possible block. 
        pattern = re.compile(f"{re.escape(start_marker)}(.*?){re.escape(end_marker)}", re.DOTALL) 
        match = pattern.search(text) 
        if not match: 
            return None 
        # Extract the content from the first capturing group 
        content = match.group(1) 
        # Clean the extracted content: 
        # 1. Remove leading/trailing whitespace and newlines. 
        content = content.strip() 
        # 2. Remove common code fences if they exist. 
        if content.startswith("```python"): 
            content = content[len("```python"):].strip() 
        if content.startswith("```"): 
            content = content[len("```"):].strip() 
        if content.endswith("```"): 
            content = content[:-len("```")].strip() 
        return content 
 
    # Define the markers for each section 
    repair_plan_start = "REPAIR_PLAN_START" 
    repair_plan_end = "REPAIR_PLAN_END" 
    suggestions_start = "Instrumentation_Suggestions_START" 
    suggestions_end = "Instrumentation_Suggestions_END" 
 
    # Extract each section using the helper function 
    repair_plan = extract_and_clean_section(llm_response, repair_plan_start, repair_plan_end) 
    instrumentation_suggestions = extract_and_clean_section(llm_response, suggestions_start, suggestions_end) 
 
    return repair_plan, instrumentation_suggestions



def _build_instrumentation_prompt(code_to_debug_next: str, instrumentation_suggestions: str, failure_info_for_instrumentation: str) -> str:
    """Builds the prompt for code instrumentation."""
    return f"""The Python code below failed tests. Add `print()` statements to trace its execution and variable states, focusing on areas related to the 'Test Failure Feedback'. 
 
 Key Instrumentation Rules: 
 
 1.  Identify several logical blocks/steps in the code. 
 
 2.  For each block/step, print: 
 
 *   Key inputs. 
 
 *   Key outputs/results. 
 
 *   Entry/exit of major functions/loops if helpful. 
 
 3.  ONLY add print statements(Don't comment it out). DO NOT change the original code logic or fix errors. 
 
 4.  Prints should be informative, like: `print(f"[BlockName] Input: {{var}}")`. 
 
 Code to Instrument: 
 
 ```python 
 
 {code_to_debug_next} 
 
 ``` 
 
 Instrument suggestions: 
 
 ``` 
 
 {instrumentation_suggestions} 
 
 ``` 
 
 Feedback: 
 
 ``` 
 
 {failure_info_for_instrumentation} 
 
 ``` 
 
 Please provide ONLY the instrumented Python code. 
 
 """


def _get_failed_history_str(history: list) -> str:
    """Formats historical failure records."""
    if not history:
        return "There have been no previous attempts to fix the failure for this specific problem instance."

    formatted_failed_list = []
    for i, attempt_info in enumerate(islice(reversed(history), 2)):
        plan_summary = attempt_info.get('plan', "N/A")
        eval_result = attempt_info['eval_result']
        formatted_failed_list.append(
            f"Historical Failed Fix Attempt {i + 1} (Passed {eval_result.get('passed_count', 0)}/{eval_result.get('total_count', 1)} tests):\n"
            f"Previous Analysis & Plan (which FAILED to lead to a full fix):\n'''\n{plan_summary}\n'''\n"
            f"That attempt resulted in these errors (on the generated code):\n{format_check_correctness_result(eval_result)}\n---"
        )
    return "\n".join(formatted_failed_list)


def _build_analysis_planning_prompt(problem_data, code_with_prints, captured_prints_output, history) -> str:
    """Builds the prompt for the first step (analysis and planning) in the two-step method."""
    failed_attempts_feedback_str = _get_failed_history_str(history)
    return f"""You are a debugging assistant. 
 
 Given the following information about a Python function that failed tests: 
 
 1. Original Problem Description: 
 
 ```python 
 
 {problem_data['complete_prompt']} 
 
 ``` 
 
 2. Instrumented Code (this version produced the prints and test feedback below): 
 
 ```python 
 
 {code_with_prints} 
 
 ``` 
 
 3. Captured Print Output and Execution Error Messages (from executing the Instrumented Code with test cases): 
 
 ```text 
 
 {str(captured_prints_output)} 
 
 ``` 
 
 4. MISTAKES TO AVOID (Review these previous failed attempts for this problem instance. Identify why they failed and how your current analysis suggests a different, better approach.): 
 
 ``` 
 
 {failed_attempts_feedback_str} 
 
 ``` 
 
 Your tasks are: 
 
 Step 1. Analyze the `Captured Print Output` in conjunction with the `Instrumented Code` line-by-line to pinpoint the exact location and nature of the bug(s). 
 
 Step 2. For each bug in `Execution Error Messages`, explain its root cause. 
 
 Step 3. Review the `MISTAKES TO AVOID`. Analyze the reasons for each failed attempt, explicitly state what lessons are learned and how your new plan avoids repeating those errors. 
 
 Step 4. Propose a concise, step-by-step, actionable plan to fix the bug(s) in the *original code structure*. 
 
 Output your response as clear text, covering points A, B, C, and D. 
 
 Your response MUST be in the specified format: 
 
 REPAIR_PLAN_START 
 
 ```python 
 
 [give the repair plan] 
 
 ``` 
 
 REPAIR_PLAN_END 
 
 Instrumentation_Suggestions_START 
 
 ```python 
 
 [Do not give the code] 
 
 ``` 
 
 Instrumentation_Suggestions_END 
 
 """


def _build_code_implementation_prompt(problem_data, code_to_fix, captured_prints, repair_plan) -> str:
    """Builds the prompt for the second step (code implementation) in the two-step method."""
    return f"""You are a code generation assistant.
You need to fix the following Python code based on the provided repair plan.

Original Problem Description:
```python
{problem_data['complete_prompt']}
```
Faulty Code to Fix:
```python
{code_to_fix}
```
Test Failure Feedback:
```
{captured_prints}
```
Detailed Repair Plan (Follow this plan carefully!):
```text
{repair_plan}
```
Your response MUST be in the specified format:
Repaired CODE Part:
```python
[The part of the code you modified/added.]
```
CODE:
```python
[Your complete corrected Python code here.]
```
END_CODE
"""


def _parse_implementation_output(response_text: str) -> str:
    """
    Parses the LLM's response from the implementation prompt to extract the full code.
    This is a simplified parser for the "CODE:" block.
    """
    # Look for the CODE: block with ```python
    full_code_match = re.search(r"CODE:\s*```python\n(.*?)\n```", response_text, re.DOTALL)
    if full_code_match:
        return full_code_match.group(1).strip()
    
    # Fallback for ``` without language tag
    full_code_match = re.search(r"CODE:\s*```\n(.*?)\n```", response_text, re.DOTALL)
    if full_code_match:
        return full_code_match.group(1).strip()

    # Fallback for cases where the code block is not perfectly formatted but follows CODE:
    code_marker_match = re.search(r"CODE:(.*)", response_text, re.DOTALL | re.IGNORECASE)
    if code_marker_match:
        potential_code = code_marker_match.group(1).strip()
        # Clean up potential markdown fences that might still be there
        if potential_code.startswith("```python"):
            potential_code = potential_code[len("```python"):].strip()
        if potential_code.startswith("```"):
            potential_code = potential_code[len("```"):].strip()
        if potential_code.endswith("```"):
            potential_code = potential_code[:-len("```")].strip()
        return potential_code

    return "" # Return empty string if no code is found


def _load_check_correctness_func(dataset_name: str) -> Callable:
    """
    Dynamically loads and returns the corresponding check_correctness function based on the dataset name.
    """
    try:
        module_path = DATASET_PATHS[dataset_name]["eval_module"]
        eval_module = importlib.import_module(module_path)
        check_correctness_func = getattr(eval_module, "check_correctness")
        return check_correctness_func
    except KeyError:
        raise ValueError(f"Dataset '{dataset_name}' not found in DATASET_PATHS in config.py.")
    except AttributeError:
        raise ImportError(f"'check_correctness' function not found in module '{module_path}'.")
    except ImportError as e:
        raise ImportError(f"Could not import evaluation module '{module_path}': {e}. Ensure __init__.py files exist in package directories.")

# 按步骤的默认输出预留（token）
DEFAULT_OUTPUT_RESERVE = {
    "direct_generation": 2000,
    "instrumentation": 2000,
    "analysis_planning": 4096,
    "implementation": 3000,
}


def get_step_output_reserve(args, step):
    """
    取某一步的输出预留：
    1) 若 args 显式给了 step_output_reserve 字典且含该步，用它；
    2) 否则用 DEFAULT_OUTPUT_RESERVE[step]；
    3) 若都没有，退回 args.output_reserve（兼容旧调用）。
    """
    step_map = getattr(args, "step_output_reserve", None)
    if isinstance(step_map, dict) and step in step_map:
        return step_map[step]
    if step in DEFAULT_OUTPUT_RESERVE:
        return DEFAULT_OUTPUT_RESERVE[step]
    return getattr(args, "output_reserve", 2000)
# --- 预算控制辅助 ---
# 按步骤的输入估算系数（经验值；代码/中文混合约 0.3~0.4，取 0.35 折中）
INPUT_ESTIMATE_RATIO = {
    "direct_generation": 0.35,
    "instrumentation": 0.35,
    "analysis_planning": 0.40,
    "implementation": 0.35,
}
DEFAULT_INPUT_RATIO = 0.35
# 系统消息等固定开销（字符数）；与 src/generation.py 的 system prompt 保持同步
SYSTEM_OVERHEAD_CHARS = 307



def estimate_input_tokens(text: str, step: str = None) -> int:
    """
    估算输入 token：按 (prompt 字符数 + 系统消息开销) * 步骤系数。
    这是估算准入，不是硬上限；实际用量以 API 返回的 usage 为准。
    """
    ratio = INPUT_ESTIMATE_RATIO.get(step, DEFAULT_INPUT_RATIO)
    return int((len(text) + SYSTEM_OVERHEAD_CHARS) * ratio)


def budget_check_and_reserve(remaining_budget, prompt, output_reserve, step=None):
    """
    返回 (need, ok)：need = 估算输入 + 输出预留；ok = 剩余预算是否够。
    """
    need = estimate_input_tokens(prompt, step) + output_reserve
    return need, (remaining_budget >= need)

def reconcile_budget_after_request(remaining_before, p_tokens, c_tokens):
    """
    请求发出后按实际用量核账。
    返回 (remaining_after, overrun_tokens, overrun_flag)
    - remaining_after: 扣减实际用量后的余额（可能为负）
    - overrun_tokens: 超出的 token 数（未超为 0）
    - overrun_flag: 余额是否为负
    """
    actual = (p_tokens or 0) + (c_tokens or 0)
    remaining_after = remaining_before - actual
    overrun_tokens = max(0, -remaining_after)
    return remaining_after, overrun_tokens, (remaining_after < 0)

# --- Core Processing Functions ---
def _run_direct_generation(problem_data, args, check_correctness_func_param: Callable):
    prompt = problem_data.get('complete_prompt', '')
    output_reserve = get_step_output_reserve(args, "direct_generation")
    remaining = getattr(args, "remaining_budget", None)

    # 预算检查：不够就不发
    if remaining is not None:
        need, ok = budget_check_and_reserve(remaining, prompt, output_reserve, step="direct_generation")
        if not ok:
            return {"code": "", "eval_result": {"passed": False, "passed_count": 0, "total_count": 0},
                    "prompt_tokens": 0, "completion_tokens": 0,
                    "budget_stopped": True, "budget_need": need, "budget_remaining": remaining}

    generated_code, p_tokens, c_tokens = generator(
        prompt, "code3_generate", args.model, max_tokens_response=output_reserve
    )

    # 缺 usage：不可核验
    if p_tokens is None or c_tokens is None:
        return {"code": generated_code, "eval_result": {"passed": False, "passed_count": 0, "total_count": 0},
                "prompt_tokens": None, "completion_tokens": None,
                "usage_unverifiable": True}

    budget_overrun = False
    overrun_tokens = 0
    remaining_after = remaining
    if remaining is not None:
        remaining_after, overrun_tokens, budget_overrun = reconcile_budget_after_request(
            remaining, p_tokens, c_tokens
        )
        args.remaining_budget = remaining_after

    eval_result = check_correctness_func_param(problem_data, generated_code, args.timeout)

    # 初始生成后若已超限：仍返回代码与评测，但标记超限，由 process_problem 停止后续调试
    return {"code": generated_code, "eval_result": eval_result,
            "prompt_tokens": p_tokens, "completion_tokens": c_tokens,
            "remaining_before": remaining,
            "remaining_after": remaining_after,
            "budget_overrun": budget_overrun,
            "overrun_tokens": overrun_tokens}


def _run_self_debugging(initial_code, initial_eval, problem_data, args, check_correctness_func_param: Callable):
    """Executes the second phase: self-debugging loop (updated to include complete logic)."""
    code_to_debug, last_eval, best_code, best_eval = initial_code, initial_eval, initial_code, initial_eval
    log, history, p_tokens, c_tokens, streak = [], [], 0, 0, 0
    best_attempt = 0   # 0 表示最终代码来自初始生成
    request_log = []
    instrumentation_suggestions = ""


    import hashlib
    def get_hash(code_str):
        return hashlib.md5(code_str.encode('utf-8')).hexdigest()[:8] if code_str else "empty"

    budget_exhausted = False
    usage_unverifiable_flag = False
    for attempt in range(1, args.max_debug_attempts + 1):
        logging.info(f"--- Debug attempt: {attempt}/{args.max_debug_attempts} | Best: {best_eval.get('passed_count', 0)}/{best_eval.get('total_count', 1)} ---")

        # 记录本轮起点，用于计算本轮增量（cost = 本轮增量；cost_cumulative = 累计）
        p_tokens_at_attempt_start = p_tokens
        c_tokens_at_attempt_start = c_tokens
        # 记录本轮开始时实际使用的输入代码（进入本轮的输入）
        input_code_at_attempt_start = code_to_debug

        # 1. Instrumentation
        captured_prints = "Instrumentation skipped or not applicable."
        instrumented_code = code_to_debug
        if not args.no_instrumentation:
            failure_info = format_check_correctness_result(last_eval)
            instr_prompt = _build_instrumentation_prompt(code_to_debug, instrumentation_suggestions, failure_info)
            output_reserve = get_step_output_reserve(args, "instrumentation")
            remaining = getattr(args, "remaining_budget", None)
            if remaining is not None:
                need, ok = budget_check_and_reserve(remaining, instr_prompt, output_reserve, step="instrumentation")
                if not ok:
                    logging.info(f"[budget] instrumentation skipped: need ~{need}, remaining {remaining}")
                    budget_exhausted = True
                    request_log.append({
                        "stage": "skipped_budget_instrumentation",
                        "attempt": attempt,
                        "sent": False,
                        "estimated_input": need - output_reserve,
                        "output_reserve": output_reserve,
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "remaining_before": remaining,
                        "remaining_after": remaining,
                        "overrun_tokens": 0,
                        "budget_overrun": False,
                        "stopped_reason": "Budget insufficient before instrumentation request",
                    })
                    log.append({
                        "attempt": attempt,
                        "input_code": input_code_at_attempt_start,
                        "input_code_hash": get_hash(input_code_at_attempt_start),
                        "status": "budget_exhausted",
                        "plan": None,
                        "eval_result": {"passed": False, "passed_count": 0, "total_count": 0},
                        "candidate_code_hash": "empty",
                        "next_input_intended": get_hash(code_to_debug),
                        "next_input_actual": None,
                        "best_code_hash": get_hash(best_code),
                        "candidate_code": "", "next_input_code": code_to_debug, "best_code": best_code,
                        "stopped_reason": "Budget insufficient before instrumentation request",
                        "budget_need": need,
                        "budget_remaining": remaining,
                        "cost": {"prompt_tokens": 0, "completion_tokens": 0},
                        "cost_cumulative": {"prompt_tokens": p_tokens, "completion_tokens": c_tokens},
                    })
                    break
            instrumented_code_gen, p, c = generator(
                instr_prompt, "code3_generate", args.model,
                max_tokens_response=output_reserve
            )
            if p is None or c is None:
                logging.error("[budget] instrumentation usage unverifiable, stopping.")
                usage_unverifiable_flag = True
                log.append({
                    "attempt": attempt,
                    "input_code": input_code_at_attempt_start,
                    "input_code_hash": get_hash(input_code_at_attempt_start),
                    "status": "usage_unverifiable",
                    "plan": None,
                    "eval_result": {"passed": False, "passed_count": 0, "total_count": 0},
                    "candidate_code_hash": "empty",
                    "next_input_intended": get_hash(code_to_debug),
                    "next_input_actual": None,
                    "best_code_hash": get_hash(best_code),
                    "candidate_code": "", "next_input_code": code_to_debug, "best_code": best_code,
                    "stopped_reason": "Usage missing for instrumentation request",
                    "cost": {"prompt_tokens": 0, "completion_tokens": 0},
                    "cost_cumulative": {"prompt_tokens": p_tokens, "completion_tokens": c_tokens},
                })
                break
            remaining_before = remaining
            remaining_after, overrun_tokens, budget_overrun = reconcile_budget_after_request(
                remaining_before, p, c
            )
            args.remaining_budget = remaining_after
            p_tokens, c_tokens = p_tokens + p, c_tokens + c
            request_log.append({
                "stage": "instrumentation",
                "attempt": attempt,
                "sent": True,
                "estimated_input": need - output_reserve,
                "output_reserve": output_reserve,
                "prompt_tokens": p,
                "completion_tokens": c,
                "remaining_before": remaining_before,
                "remaining_after": remaining_after,
                "overrun_tokens": overrun_tokens,
                "budget_overrun": budget_overrun,
                "stopped_reason": None,
            })
            if budget_overrun:
                logging.error(
                    f"[budget] OVERRUN after instrumentation: before={remaining_before}, "
                    f"actual={p + c}, after={remaining_after}"
                )
                budget_exhausted = True
                request_log[-1]["stopped_reason"] = "Budget overrun after instrumentation request"
                log.append({
                    "attempt": attempt,
                    "input_code": input_code_at_attempt_start,
                    "input_code_hash": get_hash(input_code_at_attempt_start),
                    "status": "budget_overrun",
                    "plan": None,
                    "eval_result": {"passed": False, "passed_count": 0, "total_count": 0},
                    "candidate_code_hash": "empty",
                    "next_input_intended": get_hash(code_to_debug),
                    "next_input_actual": None,
                    "best_code_hash": get_hash(best_code),
                    "candidate_code": "", "next_input_code": code_to_debug, "best_code": best_code,
                    "stopped_reason": "Budget overrun after instrumentation request",
                    "overrun_tokens": overrun_tokens,
                    "cost": {"prompt_tokens": p, "completion_tokens": c},
                    "cost_cumulative": {"prompt_tokens": p_tokens, "completion_tokens": c_tokens},
                })
                break
            if instrumented_code_gen.strip():
                instrumented_code = instrumented_code_gen
                exec_result = execute_code_and_capture_prints_last(
                    instrumented_code + "\n\n" + problem_data.get('test', ''), timeout_seconds=args.timeout)
                captured_prints = exec_result.get('display_output', 'No instrumentation output captured.')

        # 2. Analysis and Planning
        analysis_prompt = _build_analysis_planning_prompt(problem_data, instrumented_code, captured_prints, history)
        output_reserve = get_step_output_reserve(args, "analysis_planning")
        remaining = getattr(args, "remaining_budget", None)
        if remaining is not None:
            need, ok = budget_check_and_reserve(remaining, analysis_prompt, output_reserve, step="analysis_planning")
            if not ok:
                logging.info(f"[budget] analysis skipped: need ~{need}, remaining {remaining}")
                budget_exhausted = True
                request_log.append({
                    "stage": "skipped_budget_analysis",
                    "attempt": attempt,
                    "sent": False,
                    "estimated_input": need - output_reserve,
                    "output_reserve": output_reserve,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "remaining_before": remaining,
                    "remaining_after": remaining,
                    "overrun_tokens": 0,
                    "budget_overrun": False,
                    "stopped_reason": "Budget insufficient before analysis request",
                })
                log.append({
                    "attempt": attempt,
                    "status": "budget_exhausted",
                    "input_code": input_code_at_attempt_start,
                    "input_code_hash": get_hash(input_code_at_attempt_start),
                    "plan": None,
                    "eval_result": {"passed": False, "passed_count": 0, "total_count": 0},
                    "candidate_code_hash": "empty",
                    "next_input_intended": get_hash(code_to_debug),
                    "next_input_actual": None,
                    "best_code_hash": get_hash(best_code),
                    "candidate_code": "", "next_input_code": code_to_debug, "best_code": best_code,
                    "stopped_reason": "Budget insufficient before analysis request",
                    "budget_need": need,
                    "budget_remaining": remaining,
                    "cost": {"prompt_tokens": 0, "completion_tokens": 0},
                    "cost_cumulative": {"prompt_tokens": p_tokens, "completion_tokens": c_tokens},
                })
                break
        llm_response, p, c = generator(
            analysis_prompt, "code4_generate", args.model,
            max_tokens_response=output_reserve
        )
        if p is None or c is None:
            logging.error("[budget] analysis usage unverifiable, stopping.")
            usage_unverifiable_flag = True
            log.append({
                "attempt": attempt,
                "status": "usage_unverifiable",
                "input_code": input_code_at_attempt_start,
                "input_code_hash": get_hash(input_code_at_attempt_start),
                "plan": None,
                "eval_result": {"passed": False, "passed_count": 0, "total_count": 0},
                "candidate_code_hash": "empty",
                "next_input_intended": get_hash(code_to_debug),
                "next_input_actual": None,
                "best_code_hash": get_hash(best_code),
                "candidate_code": "", "next_input_code": code_to_debug, "best_code": best_code,
                "stopped_reason": "Usage missing for analysis request",
                "cost": {"prompt_tokens": 0, "completion_tokens": 0},
                "cost_cumulative": {"prompt_tokens": p_tokens, "completion_tokens": c_tokens},
            })
            break
        remaining_before = remaining
        remaining_after, overrun_tokens, budget_overrun = reconcile_budget_after_request(
            remaining_before, p, c
        )
        args.remaining_budget = remaining_after
        p_tokens, c_tokens = p_tokens + p, c_tokens + c
        request_log.append({
            "stage": "analysis_planning",
            "attempt": attempt,
            "sent": True,
            "estimated_input": need - output_reserve,
            "output_reserve": output_reserve,
            "prompt_tokens": p,
            "completion_tokens": c,
            "remaining_before": remaining_before,
            "remaining_after": remaining_after,
            "overrun_tokens": overrun_tokens,
            "budget_overrun": budget_overrun,
            "stopped_reason": None,
        })
        if budget_overrun:
            logging.error(
                f"[budget] OVERRUN after analysis: before={remaining_before}, "
                f"actual={p + c}, after={remaining_after}"
            )
            budget_exhausted = True
            request_log[-1]["stopped_reason"] = "Budget overrun after analysis request"
            log.append({
                "attempt": attempt,
                "status": "budget_overrun",
                "input_code": input_code_at_attempt_start,
                "input_code_hash": get_hash(input_code_at_attempt_start),
                "plan": None,
                "eval_result": {"passed": False, "passed_count": 0, "total_count": 0},
                "candidate_code_hash": "empty",
                "next_input_intended": get_hash(code_to_debug),
                "next_input_actual": None,
                "best_code_hash": get_hash(best_code),
                "candidate_code": "", "next_input_code": code_to_debug, "best_code": best_code,
                "stopped_reason": "Budget overrun after analysis request",
                "overrun_tokens": overrun_tokens,
                "cost": {"prompt_tokens": p, "completion_tokens": c},
                "cost_cumulative": {"prompt_tokens": p_tokens, "completion_tokens": c_tokens},
            })
            break

        repair_plan, instrumentation_suggestions = _parse_llm_output(llm_response)
        candidate_code = ""

        # ----- 分支 A：无计划退出 -----
        if not (repair_plan and repair_plan.strip()):
            attempt_log = {
                "attempt": attempt, "plan": repair_plan,
                "input_code": input_code_at_attempt_start,
                "input_code_hash": get_hash(input_code_at_attempt_start),
                "eval_result": {"passed": False, "passed_count": 0, "total_count": 0},
                "status": "no_plan", "candidate_code_hash": "empty",
                "next_input_intended": get_hash(code_to_debug), "next_input_actual": None,
                "best_code_hash": get_hash(best_code),
                "candidate_code": "", "next_input_code": code_to_debug, "best_code": best_code,
                "stopped_reason": "LLM did not provide a repair plan",
                "cost": {
                    "prompt_tokens": p_tokens - p_tokens_at_attempt_start,
                    "completion_tokens": c_tokens - c_tokens_at_attempt_start,
                },
                "cost_cumulative": {"prompt_tokens": p_tokens, "completion_tokens": c_tokens},
            }
            log.append(attempt_log); history.append(attempt_log)
            logging.info("LLM did not provide a repair plan, terminating debugging.")
            break

        # 3. Code Implementation
        impl_prompt = _build_code_implementation_prompt(problem_data, code_to_debug, captured_prints, repair_plan)
        output_reserve = get_step_output_reserve(args, "implementation")
        remaining = getattr(args, "remaining_budget", None)
        if remaining is not None:
            need, ok = budget_check_and_reserve(remaining, impl_prompt, output_reserve, step="implementation")
            if not ok:
                logging.info(f"[budget] implementation skipped: need ~{need}, remaining {remaining}")
                budget_exhausted = True
                request_log.append({
                    "stage": "skipped_budget_implementation",
                    "attempt": attempt,
                    "sent": False,
                    "estimated_input": need - output_reserve,
                    "output_reserve": output_reserve,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "remaining_before": remaining,
                    "remaining_after": remaining,
                    "overrun_tokens": 0,
                    "budget_overrun": False,
                    "stopped_reason": "Budget insufficient before implementation request",
                })
                log.append({
                    "attempt": attempt,
                    "status": "budget_exhausted",
                    "input_code": input_code_at_attempt_start,
                    "input_code_hash": get_hash(input_code_at_attempt_start),
                    "plan": None,
                    "eval_result": {"passed": False, "passed_count": 0, "total_count": 0},
                    "candidate_code_hash": "empty",
                    "next_input_intended": get_hash(code_to_debug),
                    "next_input_actual": None,
                    "best_code_hash": get_hash(best_code),
                    "candidate_code": "", "next_input_code": code_to_debug, "best_code": best_code,
                    "stopped_reason": "Budget insufficient before implementation request",
                    "budget_need": need,
                    "budget_remaining": remaining,
                    "cost": {"prompt_tokens": 0, "completion_tokens": 0},
                    "cost_cumulative": {"prompt_tokens": p_tokens, "completion_tokens": c_tokens},
                })
                break
        llm_response_impl, p_impl, c_impl = generator(
            impl_prompt, "code4_generate", args.model,
            max_tokens_response=output_reserve
        )
        if p_impl is None or c_impl is None:
            logging.error("[budget] implementation usage unverifiable, stopping.")
            usage_unverifiable_flag = True
            log.append({
                "attempt": attempt,
                "status": "usage_unverifiable",
                "plan": None,
                "input_code": input_code_at_attempt_start,
                "input_code_hash": get_hash(input_code_at_attempt_start),
                "eval_result": {"passed": False, "passed_count": 0, "total_count": 0},
                "candidate_code_hash": "empty",
                "next_input_intended": get_hash(code_to_debug),
                "next_input_actual": None,
                "best_code_hash": get_hash(best_code),
                "candidate_code": "", "next_input_code": code_to_debug, "best_code": best_code,
                "stopped_reason": "Usage missing for implementation request",
                "cost": {"prompt_tokens": 0, "completion_tokens": 0},
                "cost_cumulative": {"prompt_tokens": p_tokens, "completion_tokens": c_tokens},
            })
            break
        remaining_before = remaining
        remaining_after, overrun_tokens, budget_overrun = reconcile_budget_after_request(
            remaining_before, p_impl, c_impl
        )
        args.remaining_budget = remaining_after
        p_tokens, c_tokens = p_tokens + p_impl, c_tokens + c_impl
        request_log.append({
            "stage": "implementation",
            "attempt": attempt,
            "sent": True,
            "estimated_input": need - output_reserve,
            "output_reserve": output_reserve,
            "prompt_tokens": p_impl,
            "completion_tokens": c_impl,
            "remaining_before": remaining_before,
            "remaining_after": remaining_after,
            "overrun_tokens": overrun_tokens,
            "budget_overrun": budget_overrun,
            "stopped_reason": None,
        })
        if budget_overrun:
            logging.error(
                f"[budget] OVERRUN after implementation: before={remaining_before}, "
                f"actual={p_impl + c_impl}, after={remaining_after}"
            )
            budget_exhausted = True
            request_log[-1]["stopped_reason"] = "Budget overrun after implementation request"
            log.append({
                "attempt": attempt,
                "status": "budget_overrun",
                "input_code": input_code_at_attempt_start,
                "input_code_hash": get_hash(input_code_at_attempt_start),
                "plan": repair_plan,
                "eval_result": {"passed": False, "passed_count": 0, "total_count": 0},
                "candidate_code_hash": "empty",
                "next_input_intended": get_hash(code_to_debug),
                "next_input_actual": None,
                "best_code_hash": get_hash(best_code),
                "candidate_code": "", "next_input_code": code_to_debug, "best_code": best_code,
                "stopped_reason": "Budget overrun after implementation request",
                "overrun_tokens": overrun_tokens,
                "cost": {"prompt_tokens": p_impl, "completion_tokens": c_impl},
                "cost_cumulative": {"prompt_tokens": p_tokens, "completion_tokens": c_tokens},
            })
            break
        candidate_code = _parse_implementation_output(llm_response_impl)
        

        # ----- 分支 B：空代码退出 -----
        if not candidate_code.strip():
            attempt_log = {
                "attempt": attempt, "plan": repair_plan,
                "input_code": input_code_at_attempt_start,
                "input_code_hash": get_hash(input_code_at_attempt_start),
                "eval_result": {"passed": False, "passed_count": 0, "total_count": 0},
                "status": "empty_code", "candidate_code_hash": "empty",
                "next_input_intended": get_hash(code_to_debug), "next_input_actual": None,
                "best_code_hash": get_hash(best_code),
                "candidate_code": "", "next_input_code": code_to_debug, "best_code": best_code,
                "stopped_reason": "LLM did not provide valid code",
                "cost": {
                    "prompt_tokens": p_tokens - p_tokens_at_attempt_start,
                    "completion_tokens": c_tokens - c_tokens_at_attempt_start,
                },
                "cost_cumulative": {"prompt_tokens": p_tokens, "completion_tokens": c_tokens},
            }
            log.append(attempt_log); history.append(attempt_log)
            logging.info("LLM did not provide valid code, terminating debugging.")
            break

        # 4. Evaluation
        eval_candidate = check_correctness_func_param(problem_data, candidate_code, args.timeout)
        logging.info(f"Candidate repair result: {format_check_correctness_result(eval_candidate)}")

        curr_score = eval_candidate.get('passed_count', 0)
        best_score = best_eval.get('passed_count', 0)
        status = "unknown"

        # ----- 分支 C：成功（先记日志再 break） -----
        if eval_candidate.get('passed'):
            status = "passed"
            best_code, best_eval = candidate_code, eval_candidate
            best_attempt = attempt
            code_to_debug = candidate_code
            logging.info(f"🎯 [第 {attempt} 轮] 全部通过！状态：passed | 候选代码Hash: {get_hash(candidate_code)}")
            logging.info(f"===== 成功候选代码本体 =====\n{candidate_code}\n========================")

            attempt_log = {
                "attempt": attempt, "plan": repair_plan,
                "input_code": input_code_at_attempt_start,
                "input_code_hash": get_hash(input_code_at_attempt_start), 
                "eval_result": eval_candidate,
                "status": status, "candidate_code_hash": get_hash(candidate_code),
                "next_input_intended": get_hash(code_to_debug), "next_input_actual": None,
                "best_code_hash": get_hash(best_code),
                "candidate_code": candidate_code, "next_input_code": code_to_debug, "best_code": best_code,
                "stopped_reason": "All tests passed",
                "cost": {
                    "prompt_tokens": p_tokens - p_tokens_at_attempt_start,
                    "completion_tokens": c_tokens - c_tokens_at_attempt_start,
                },
                "cost_cumulative": {"prompt_tokens": p_tokens, "completion_tokens": c_tokens},
            }
            log.append(attempt_log); history.append(attempt_log)
            break

        # ----- 分支 D：改善 / 持平 / 退步 -----
        if curr_score > best_score:
            status = "improve"
            best_code, best_eval, code_to_debug, last_eval, streak = candidate_code, eval_candidate, candidate_code, eval_candidate, 0
            best_attempt = attempt
            logging.info(f"📈 [第 {attempt} 轮] 改进！状态：improve | 候选Hash: {get_hash(candidate_code)} | 历史最优分: {curr_score}")
            logging.info(f"===== 候选代码本体 =====\n{candidate_code}\n========================")
        elif curr_score == best_score:
            status = "tie"
            streak += 1
            logging.info(f"⏸️ [第 {attempt} 轮] 持平！状态：tie | 候选Hash: {get_hash(candidate_code)} | 历史最优分: {best_score}")
            logging.info(f"===== 候选代码本体 =====\n{candidate_code}\n========================")
        else:
            status = "regress"
            streak += 1
            logging.info(f"📉 [第 {attempt} 轮] 退步！状态：regress | 候选Hash: {get_hash(candidate_code)} | 历史最优Hash: {get_hash(best_code)}")
            logging.info(f"===== 候选代码本体 =====\n{candidate_code}\n========================")
            logging.info(f"===== 历史最优代码本体 =====\n{best_code}\n========================")
            logging.info(f"⚠️ 触发回滚！下一轮输入强制替换为历史最优代码 | 历史最优Hash: {get_hash(best_code)}")
            logging.info(f"===== 下一轮实际输入代码本体 =====\n{code_to_debug}\n========================")
            code_to_debug = best_code
            last_eval = best_eval

        will_continue = (attempt < args.max_debug_attempts) and (streak < args.max_no_improvement_streak)

        attempt_log = {
            "attempt": attempt, "plan": repair_plan,
            "input_code": input_code_at_attempt_start,
            "input_code_hash": get_hash(input_code_at_attempt_start),
            "eval_result": eval_candidate,
            "status": status, "candidate_code_hash": get_hash(candidate_code),
            "next_input_intended": get_hash(code_to_debug),
            "next_input_actual": get_hash(code_to_debug) if will_continue else None,
            "best_code_hash": get_hash(best_code),
            "candidate_code": candidate_code, "next_input_code": code_to_debug, "best_code": best_code,
            "stopped_reason": (
                "Continued to next attempt" if will_continue
                else ("Stagnation threshold reached" if streak >= args.max_no_improvement_streak else "Max attempts reached")
            ),
            "cost": {
                "prompt_tokens": p_tokens - p_tokens_at_attempt_start,
                "completion_tokens": c_tokens - c_tokens_at_attempt_start,
            },
            "cost_cumulative": {"prompt_tokens": p_tokens, "completion_tokens": c_tokens},
        }
        log.append(attempt_log); history.append(attempt_log)

        if streak >= args.max_no_improvement_streak:
            logging.info(f"{streak} consecutive attempts with no improvement, terminating debugging.")
            break

    return {
        "final_code": best_code, "final_eval": best_eval, "session_log": log,
        "request_log": request_log,
        "prompt_tokens": p_tokens, "completion_tokens": c_tokens,
        "budget_exhausted": budget_exhausted,
        "budget_remaining": getattr(args, "remaining_budget", None),
        "usage_unverifiable": usage_unverifiable_flag,
        "budget_overrun": any(r.get("budget_overrun") for r in request_log),
        "overrun_tokens": sum(r.get("overrun_tokens", 0) for r in request_log),
        "final_best_attempt": best_attempt,
        "final_best_source": (
            "initial_generation" if best_attempt == 0 else f"attempt_{best_attempt}"
        ),
    }

# The `process_problem` function remains unchanged as it only calls the core functions above
def process_problem(problem_data, task_id, args, check_correctness_func_param: Callable):
    # 每题开始前重置预算（若启用预算控制）
    if getattr(args, "budget_total", None) is not None:
        args.remaining_budget = args.budget_total

    print(f"\n{'=' * 25} Processing problem: {task_id} {'=' * 25}")
    start_time = time.time()

    gen_res = _run_direct_generation(problem_data, args, check_correctness_func_param)
    initial_request_log = []
    if gen_res.get("prompt_tokens") is not None:
        initial_request_log.append({
            "stage": "direct_generation",
            "attempt": 0,
            "prompt_tokens": gen_res["prompt_tokens"],
            "completion_tokens": gen_res["completion_tokens"],
            "remaining_after": getattr(args, "remaining_budget", None),
        })


    # 分支：初始生成因预算不足被跳过
    if gen_res.get("budget_stopped"):
        result = {
            "task_id": task_id, "model": args.model, "dataset": args.dataset,
            "method_instrumentation": not args.no_instrumentation,
            "method_two_step_repair": not args.no_two_step_repair,
            "direct_gen_code": "", "direct_gen_eval": gen_res["eval_result"],
            "direct_gen_passed": False,
            "final_code": "", "final_eval": gen_res["eval_result"], "final_passed": False,
            "stopped_reason": "Budget insufficient for initial generation",
            "budget_need": gen_res.get("budget_need"),
            "budget_remaining": gen_res.get("budget_remaining"),
            "total_prompt_tokens": 0, "total_completion_tokens": 0,
            "processing_time_seconds": time.time() - start_time,
        }
        print(f"Problem {task_id} processing completed. Final result: Failed (budget stopped)")
        return result

    # 分支：初始生成缺 usage，不可核验
    if gen_res.get("usage_unverifiable"):
        result = {
            "task_id": task_id, "model": args.model, "dataset": args.dataset,
            "method_instrumentation": not args.no_instrumentation,
            "method_two_step_repair": not args.no_two_step_repair,
            "direct_gen_code": gen_res["code"], "direct_gen_eval": gen_res["eval_result"],
            "direct_gen_passed": False,
            "final_code": gen_res["code"], "final_eval": gen_res["eval_result"], "final_passed": False,
            "stopped_reason": "Initial generation usage unverifiable",
            "total_prompt_tokens": None, "total_completion_tokens": None,
            "processing_time_seconds": time.time() - start_time,
        }
        print(f"Problem {task_id} processing completed. Final result: Failed (usage unverifiable)")
        return result

    # 分支：初始生成后已超限 → 留痕并停止，不进入调试
    if gen_res.get("budget_overrun"):
        result = {
            "task_id": task_id, "model": args.model, "dataset": args.dataset,
            "method_instrumentation": not args.no_instrumentation,
            "method_two_step_repair": not args.no_two_step_repair,
            "direct_gen_code": gen_res["code"], "direct_gen_eval": gen_res["eval_result"],
            "direct_gen_passed": gen_res["eval_result"].get("passed", False),
            "final_code": gen_res["code"], "final_eval": gen_res["eval_result"],
            "final_passed": False,
            "stopped_reason": "Budget overrun after direct generation",
            "budget_overrun": True,
            "overrun_tokens": gen_res.get("overrun_tokens", 0),
            "total_prompt_tokens": gen_res["prompt_tokens"],
            "total_completion_tokens": gen_res["completion_tokens"],
            "request_log": initial_request_log,
            "processing_time_seconds": time.time() - start_time,
        }
        print(f"Problem {task_id} processing completed. Final result: Failed (budget overrun)")
        return result

    # 初始生成正常：判 None 后累加，不再重复赋值
    total_p = gen_res["prompt_tokens"] if gen_res["prompt_tokens"] is not None else 0
    total_c = gen_res["completion_tokens"] if gen_res["completion_tokens"] is not None else 0

    result = {
        "task_id": task_id, "model": args.model, "dataset": args.dataset,
        "method_instrumentation": not args.no_instrumentation,
        "method_two_step_repair": not args.no_two_step_repair,
        "direct_gen_code": gen_res["code"], "direct_gen_eval": gen_res["eval_result"],
        "direct_gen_passed": gen_res["eval_result"].get("passed", False),
    }

    if result["direct_gen_passed"]:
        print("Direct generation passed successfully.")
        result.update({
            "final_code": gen_res["code"], "final_eval": gen_res["eval_result"],
            "final_passed": True,
            "stopped_reason": "Passed on direct generation",
            "request_log": initial_request_log,
        })
    else:
        print("Direct generation failed, entering self-debugging phase.")
        debug_res = _run_self_debugging(
            gen_res["code"], gen_res["eval_result"], problem_data, args, check_correctness_func_param
        )
        session_log = debug_res["session_log"]
        last_stopped_reason = (
            session_log[-1].get("stopped_reason") if session_log else "No debug attempts"
        )
        result.update({
            "final_code": debug_res["final_code"], "final_eval": debug_res["final_eval"],
            "final_passed": debug_res["final_eval"].get("passed", False),
            "debug_session_log": session_log, "stopped_reason": last_stopped_reason,
            "final_best_attempt": debug_res.get("final_best_attempt", 0),
            "final_best_source": debug_res.get("final_best_source", "initial_generation"),
            "budget_exhausted": debug_res.get("budget_exhausted", False),
            "budget_remaining": debug_res.get("budget_remaining"),
            "usage_unverifiable": debug_res.get("usage_unverifiable", False),
            "budget_overrun": debug_res.get("budget_overrun", False),
            "overrun_tokens": debug_res.get("overrun_tokens", 0),
            "request_log": initial_request_log + debug_res.get("request_log", []),
        })
        if debug_res["prompt_tokens"] is not None:
            total_p += debug_res["prompt_tokens"]
        if debug_res["completion_tokens"] is not None:
            total_c += debug_res["completion_tokens"]

    result.update({
        "total_prompt_tokens": total_p, "total_completion_tokens": total_c,
        "processing_time_seconds": time.time() - start_time,
    })

    print(f"Problem {task_id} processing completed. Final result: {'Passed' if result['final_passed'] else 'Failed'}")
    return result
