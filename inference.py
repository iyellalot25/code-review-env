"""
inference.py — OpenEnv Code Review Assistant baseline inference script.

Connects to the FastAPI server via HTTP, uses an LLM to review pull requests,
and logs results in the required OpenEnv stdout format.

Environment variables:
  API_BASE_URL            LLM API base URL (default: https://router.huggingface.co/v1)
  MODEL_NAME              Model identifier (default: Qwen/Qwen2.5-72B-Instruct)
  API_KEY                 API key injected by validator (also reads HF_TOKEN as fallback)
  CODE_REVIEW_TASK        Task name: easy-review | medium-review | hard-review (default: easy-review)
  CODE_REVIEW_SERVER_URL  FastAPI server URL (default: http://localhost:7860)
"""

import os
import sys
import json
import time
import httpx
from openai import OpenAI

# ─── Configuration ────────────────────────────────────────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY = os.getenv("API_KEY") or os.getenv("HF_TOKEN", "")
CODE_REVIEW_TASK = os.getenv("CODE_REVIEW_TASK", "easy-review")
SERVER_URL = os.getenv("CODE_REVIEW_SERVER_URL", "http://localhost:7860")
SUCCESS_THRESHOLD = 0.3

TASK_MAX_STEPS = {
    "easy-review": 8,
    "medium-review": 10,
    "hard-review": 12,
}
MAX_STEPS = TASK_MAX_STEPS.get(CODE_REVIEW_TASK, 8)

# ─── Logging ──────────────────────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: str | None) -> None:
    error_str = error if error is not None else "null"
    done_str = "true" if done else "false"
    print(
        f"[STEP]  step={step} action={action} reward={reward:.2f} done={done_str} error={error_str}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    success_str = "true" if success else "false"
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END]   success={success_str} steps={steps} score={score} rewards={rewards_str}",
        flush=True,
    )


# ─── Server communication ─────────────────────────────────────────────────────

def server_reset() -> dict:
    """POST /reset and return the observation dict."""
    resp = httpx.post(f"{SERVER_URL}/reset", timeout=30)
    resp.raise_for_status()
    return resp.json()


def server_step(action: dict) -> dict:
    """POST /step with action dict and return step result dict."""
    resp = httpx.post(f"{SERVER_URL}/step", json=action, timeout=60)
    resp.raise_for_status()
    return resp.json()


# ─── LLM client ───────────────────────────────────────────────────────────────

def make_llm_client() -> OpenAI:
    return OpenAI(
        base_url=API_BASE_URL,
        api_key=API_KEY if API_KEY else "no-key",
    )


SYSTEM_PROMPT = """\
You are an expert software engineer specializing in code security and quality review.
Your job is to carefully review a pull request diff and identify ALL bugs, security
vulnerabilities, performance issues, and logic errors in the code.

You must respond ONLY with a valid JSON object — no markdown, no explanation, no preamble.

The JSON must match this exact schema:
{
  "action_type": "flag_issue",
  "issues": [
    {
      "issue_type": "bug|security|performance|style|logic",
      "line_number": <integer or null>,
      "severity": "low|medium|high|critical",
      "description": "<clear description of the issue>",
      "suggested_fix": "<how to fix it>"
    }
  ],
  "comment": "<optional overall comment>",
  "final_verdict": null
}

For the FINAL step (when you have already flagged all issues), set:
  "action_type": "request_changes" (if issues were found) or "approve" (if no issues)
  "final_verdict": "request_changes" or "approve"
  "issues": [] (empty on the final verdict step)

Guidelines:
- Be thorough: off-by-one errors, null/None dereferences, SQL injection, hardcoded secrets,
  insecure hashing, race conditions, mutation during iteration, missing auth, plaintext passwords
- Line numbers should reference the diff line numbers where the issue occurs
- severity: critical = exploitable/crashing, high = likely to cause bugs, medium = possible risk, low = style/minor
- Only report real issues visible in the diff — do not hallucinate
"""


def build_user_prompt(obs: dict, step_num: int, max_steps: int, is_final: bool) -> str:
    file_sections = []
    for filename, content in obs.get("file_contents", {}).items():
        file_sections.append(f"=== {filename} ===\n{content}")
    files_text = "\n\n".join(file_sections)

    previous = obs.get("previous_actions", [])
    prev_text = "\n".join(f"  - {a}" for a in previous) if previous else "  (none)"
    feedback = obs.get("feedback") or "(no feedback yet)"

    step_instruction = ""
    if is_final:
        step_instruction = (
            "\n\nThis is your FINAL step. Do not flag more issues. "
            "Set action_type to 'request_changes' (or 'approve' if clean), "
            "set final_verdict accordingly, and issues to []."
        )
    else:
        step_instruction = (
            f"\n\nThis is step {step_num} of {max_steps}. "
            "Flag ALL issues you can identify in a single response."
        )

    return f"""\
PR Title: {obs.get("pr_title", "")}
PR Description: {obs.get("pr_description", "")}

--- DIFF ---
{obs.get("diff", "")}

--- FILE CONTENTS ---
{files_text}

--- PREVIOUS ACTIONS ---
{prev_text}

--- SERVER FEEDBACK ---
{feedback}
{step_instruction}

Respond with ONLY the JSON object described in the system prompt.
"""


def call_llm(client: OpenAI, prompt: str) -> str:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        max_tokens=1000,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
    )
    return response.choices[0].message.content.strip()


def parse_action(raw: str) -> dict:
    """Parse LLM output into a CodeReviewAction dict. Falls back to add_comment on failure."""
    # Strip markdown code fences if present
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        # Remove first and last fence lines
        inner = []
        in_block = False
        for line in lines:
            if line.startswith("```") and not in_block:
                in_block = True
                continue
            if line.startswith("```") and in_block:
                break
            if in_block:
                inner.append(line)
        text = "\n".join(inner).strip()

    try:
        data = json.loads(text)
        # Validate required field
        if "action_type" not in data:
            raise ValueError("Missing action_type")
        # Ensure issues is a list if present
        if "issues" in data and data["issues"] is None:
            data["issues"] = []
        return data
    except (json.JSONDecodeError, ValueError):
        # Fallback: add_comment with raw text
        return {
            "action_type": "add_comment",
            "issues": [],
            "comment": raw[:500],
            "final_verdict": None,
        }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    env_name = "code-review-assistant"

    log_start(task=CODE_REVIEW_TASK, env=env_name, model=MODEL_NAME)

    # Wait for server to be ready (retry up to 30s)
    for attempt in range(30):
        try:
            r = httpx.get(f"{SERVER_URL}/health", timeout=5)
            if r.status_code == 200:
                break
        except Exception:
            pass
        time.sleep(1)
    else:
        print("ERROR: Server did not become ready within 30 seconds", file=sys.stderr)
        log_end(success=False, steps=0, score=0.0, rewards=[])
        sys.exit(1)

    # Set task via env var (server reads CODE_REVIEW_TASK on reset)
    # We POST /reset to initialize
    try:
        result = server_reset()
    except Exception as e:
        log_end(success=False, steps=0, score=0.0, rewards=[])
        raise

    obs = result["observation"]
    client = make_llm_client()

    rewards: list[float] = []
    final_score = 0.0
    step_num = 0
    done = False
    error_msg = None

    # Strategy: flag all issues in one early step, then submit final verdict
    # Step 1..N-1: flag issues
    # Last step: submit request_changes / approve
    for step_num in range(1, MAX_STEPS + 1):
        is_final = step_num == MAX_STEPS or done

        prompt = build_user_prompt(obs, step_num, MAX_STEPS, is_final)
        error_msg = None

        try:
            raw_output = call_llm(client, prompt)
            action = parse_action(raw_output)

            # If this is the penultimate step (step MAX_STEPS-1) and we already got
            # feedback, force a final verdict on the next iteration by using request_changes
            if is_final and action.get("action_type") not in ("approve", "request_changes"):
                action["action_type"] = "request_changes"
                action["final_verdict"] = "request_changes"
                action["issues"] = []

            step_result = server_step(action)

        except Exception as e:
            error_msg = str(e)[:100]
            # Try a safe fallback action
            try:
                fallback = {
                    "action_type": "add_comment",
                    "issues": [],
                    "comment": f"Error: {error_msg}",
                    "final_verdict": None,
                }
                step_result = server_step(fallback)
            except Exception as e2:
                log_step(step_num, "error", 0.0, False, str(e2)[:100])
                break

        reward = step_result.get("reward", 0.0)
        done = step_result.get("done", False)
        obs = step_result.get("observation", obs)
        rewards.append(reward)

        action_summary = action.get("action_type", "unknown")
        n_issues = len(action.get("issues") or [])
        if n_issues:
            action_summary += f"({n_issues}_issues)"

        log_step(step_num, action_summary, reward, done, error_msg)

        if done:
            final_score = step_result.get("reward", 0.0)
            # The final step returns cumulative score, not delta
            # Pull it from info if available
            info = step_result.get("info", {})
            if "score" in info:
                final_score = info["score"]
            break

    # If we never got a done, calculate from cumulative rewards
    if not done:
        final_score = sum(rewards)

    success = final_score >= SUCCESS_THRESHOLD
    log_end(success=success, steps=step_num, score=round(final_score, 4), rewards=rewards)


if __name__ == "__main__":
    main()