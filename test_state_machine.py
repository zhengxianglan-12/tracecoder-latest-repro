import json
import logging
import hashlib

logging.basicConfig(level=logging.INFO, format='%(message)s')

def get_hash(code_str):
    return hashlib.md5(code_str.encode('utf-8')).hexdigest()[:8] if code_str else "empty"

def simulate_branch(initial_score, candidate_score, status_expected):
    best_score = initial_score
    best_code = "initial_code"
    code_to_debug = "initial_code"
    streak = 0
    log = []

    curr_score = candidate_score
    candidate_code = f"candidate_code_{candidate_score}"

    if candidate_score >= 16:
        status = "passed"
        best_code, best_score = candidate_code, candidate_score
        code_to_debug = candidate_code
        stopped_reason = "All tests passed"
    elif curr_score > best_score:
        status = "improve"
        best_code, best_score, code_to_debug = candidate_code, candidate_score, candidate_code
        streak = 0
        stopped_reason = "Continued to next attempt"
    elif curr_score == best_score:
        status = "tie"
        streak += 1
        stopped_reason = "Continued to next attempt" if streak < 2 else "Stagnation threshold reached"
    else:
        status = "regress"
        streak += 1
        code_to_debug = best_code
        stopped_reason = "Continued to next attempt" if streak < 2 else "Stagnation threshold reached"

    will_continue = (streak < 2) and (status != "passed")

    attempt_log = {
        "attempt": 1,
        "status": status,
        "candidate_code_hash": get_hash(candidate_code),
        "next_input_intended": get_hash(code_to_debug),
        "next_input_actual": get_hash(code_to_debug) if will_continue else None,
        "best_code_hash": get_hash(best_code),
        "candidate_code": candidate_code,
        "next_input_code": code_to_debug,
        "best_code": best_code,
        "stopped_reason": stopped_reason,
        "cost": {"prompt_tokens": 1000, "completion_tokens": 500}
    }
    log.append(attempt_log)

    print(f"\n--- 模拟状态: {status} ---")
    print(json.dumps(attempt_log, indent=2, ensure_ascii=False))
    assert status == status_expected, f"状态不匹配: 期望 {status_expected}, 实际 {status}"
    print(f"[PASS] 状态 {status} 验证通过")

if __name__ == '__main__':
    print("========== 状态机自测（不调用模型） ==========")
    simulate_branch(10, 12, "improve")
    simulate_branch(12, 12, "tie")
    simulate_branch(12, 10, "regress")
    simulate_branch(10, 16, "passed")
    print("\n[SUCCESS] 四种状态全部验证通过！")