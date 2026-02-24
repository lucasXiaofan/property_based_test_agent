# Code Agent Orchestration Plan (Codex/Claude)

## Goal
Build a simple runner that reads a task JSON, launches one code agent per task, and writes back machine-detectable status/output.

Target task style:
- `follow '/Users/xiaofanlu/Documents/github_repos/property_based_test_agent/test_quality_metric/CODING_AGENT_PROCEDURE.md' for <doc_url>`

## Non-goals
- No heavy workflow engine.
- No complex multi-agent communication.
- Use LLM only for the actual coding task prompt (not orchestration decisions).

## High-level Strategy
1. Use one Python orchestrator to run CLI agents via PTY (interactive terminal).
2. Force each task prompt to require a strict completion marker in stdout.
3. Parse stdout/stderr + exit code + timeout to classify final status.
4. Detect known interactive permission prompts and auto-respond (`1`, `p`) based on config.
5. Persist task state back into the same JSON file so batch progress is resumable.

## Suggested Files
- `run_code_agent/tasks.json` (input + evolving output)
- `run_code_agent/run_agents.py` (main orchestrator)
- `run_code_agent/agent_profiles.py` (Codex/Claude command templates + regexes)
- `run_code_agent/README.md` (how to run)
- `run_code_agent/runs/<task_id>/` (raw logs, metadata per run)

## Task JSON Schema (single source of truth)
```json
{
  "run_id": "2026-02-24-batch-001",
  "tasks": [
    {
      "task_id": "pandas-merge-001",
      "agent": "codex",
      "enabled": true,
      "cwd": "/Users/xiaofanlu/Documents/github_repos/property_based_test_agent",
      "timeout_sec": 1800,
      "max_retries": 1,
      "permission_policy": {
        "auto_press_1": true,
        "auto_press_p": true
      },
      "inputs": {
        "procedure_path": "/Users/xiaofanlu/Documents/github_repos/property_based_test_agent/test_quality_metric/CODING_AGENT_PROCEDURE.md",
        "doc_url": "https://pandas.pydata.org/pandas-docs/version/3.0.0/reference/api/pandas.DataFrame.merge.html#pandas.DataFrame.merge",
        "pytest_file": "/Users/xiaofanlu/Documents/github_repos/property_based_test_agent/pandas_bug_finding/baseline_testing/test_merge_hypothesis.py"
      },
      "prompt_template": "Follow '{procedure_path}' for {doc_url}. Use pytest file {pytest_file}. When complete, print exactly: TASK_COMPLETE:{task_id}",
      "status": "pending",
      "attempts": 0,
      "result": null
    }
  ]
}
```

## Completion Detection
Require exact token:
- `TASK_COMPLETE:<task_id>`

Success rule:
- Marker found in output AND process exits normally.

If marker missing:
- `failed_incomplete` even if exit code is 0.

## Failure Classification
Classify with simple regex + exit/timeout:

1. `failed_auth`
- Match text like: `login`, `authenticate`, `not logged in`, `session expired`.

2. `failed_quota`
- Match text like: `quota`, `credit`, `rate limit`, `insufficient balance`.

3. `failed_permission_blocked`
- Permission prompt detected repeatedly and unresolved.

4. `failed_timeout`
- No completion before `timeout_sec`.

5. `failed_process`
- Non-zero exit without a more specific match.

6. `failed_incomplete`
- Process ended but no completion marker.

## Permission Prompt Handling (`1`, `p`)
Use PTY streaming and regex triggers:
- If output matches `Press 1` and policy allows, send `1\n`.
- If output matches `press p` and policy allows, send `p\n`.

Guardrails:
- Cap auto keypress count per task (example: 5 each).
- If repeated prompt loops, stop and mark `failed_permission_blocked`.
- Log every auto-input event with timestamp.

## Agent Profile Abstraction
Keep agent differences in one file:

```python
AGENT_PROFILES = {
  "codex": {
    "command": ["codex", "run", "--task", "{rendered_prompt}"],
    "auth_regex": [...],
    "quota_regex": [...],
    "permission_regex": {"press_1": [...], "press_p": [...]}
  },
  "claude": {
    "command": ["claude", "code", "--print", "{rendered_prompt}"],
    "auth_regex": [...],
    "quota_regex": [...],
    "permission_regex": {"press_1": [...], "press_p": [...]}
  }
}
```

Only this profile map should change when adding a new CLI agent.

## Runner Flow (`run_agents.py`)
1. Load JSON.
2. Iterate enabled tasks with `status in {pending, retryable}`.
3. Render prompt from `prompt_template` and `inputs`.
4. Start agent command in PTY, stream output to:
   - console
   - `runs/<task_id>/attempt_<n>.log`
5. During stream:
   - check completion marker
   - detect permission prompts and auto-respond
   - detect auth/quota hints
6. End process on completion/timeout/exit.
7. Classify status and write `task.result`.
8. Persist JSON after every task attempt.

## Output Update Contract (per task)
On finish, write:
```json
{
  "status": "completed",
  "attempts": 1,
  "result": {
    "completed_at": "2026-02-24T12:00:00Z",
    "completion_marker_seen": true,
    "exit_code": 0,
    "failure_type": null,
    "log_file": "run_code_agent/runs/pandas-merge-001/attempt_1.log",
    "auto_inputs": [
      {"key": "1", "count": 1},
      {"key": "p", "count": 0}
    ]
  }
}
```

## Minimal Implementation Steps
1. Create `tasks.json` with one sample pandas merge task.
2. Implement PTY runner + marker detection + timeout.
3. Add failure classification regexes (auth/quota/process/timeout/incomplete).
4. Add permission auto-input loop for `1` and `p`.
5. Add JSON persistence after each attempt.
6. Add retry logic (`max_retries`) for retryable failures only.
7. Add README with one command:
   - `uv run python run_code_agent/run_agents.py --tasks run_code_agent/tasks.json`

## Simple Retry Policy
- Retry: `failed_timeout`, `failed_process`
- No retry: `failed_auth`, `failed_quota`, `failed_permission_blocked`
- Configurable per task via `max_retries`.

## Validation Checklist
- Task marked `completed` only when marker appears.
- Auth/quota failures are distinguishable in `result.failure_type`.
- Permission prompts are auto-handled and logged.
- Re-running runner skips already completed tasks.
- Crash-safe: JSON is always valid and updated incrementally.

## Why this stays simple
- JSON file is both queue and result DB.
- Regex-based detection avoids extra LLM calls.
- PTY handling is enough for interactive approvals.
- Agent-specific behavior isolated to a small profile map.

## Optional: LLM + Bash Tool Calls (DeepSeek style)
If you want one model endpoint to orchestrate shell actions, keep it narrow:
- Expose only a tiny tool set: `run_agent(task_id)`, `get_task(task_id)`, `update_task(task_id, patch)`.
- Let the model choose tools, but keep status classification in deterministic Python code.
- Do not let model parse raw logs for final status; parser in `run_agents.py` remains source of truth.

This keeps LLM usage minimal while still allowing tool-call driven control.
