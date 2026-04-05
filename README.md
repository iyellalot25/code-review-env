# Code Review Assistant — OpenEnv Environment

An [OpenEnv](https://github.com/openenv/openenv)-compliant benchmark environment where an AI agent reviews pull requests and identifies bugs, security vulnerabilities, and code quality issues. Built for the OpenEnv Hackathon.

---

## Overview

The **Code Review Assistant** environment presents an AI agent with realistic Python pull requests containing intentional bugs, security flaws, and logic errors. The agent must systematically identify issues by submitting structured `IssueReport` actions. A deterministic grader scores the agent based on how well its reported issues match the ground truth, using keyword overlap, line proximity, type matching, and severity matching — no LLM-as-judge involved.

---

## Environment Design

### Observation Space

| Field              | Type              | Description                                              |
|--------------------|-------------------|----------------------------------------------------------|
| `pr_title`         | `string`          | Title of the pull request                                |
| `pr_description`   | `string`          | Author's description of the changes                      |
| `diff`             | `string`          | Unified diff of the PR                                   |
| `file_contents`    | `dict[str, str]`  | Full content of each changed file                        |
| `previous_actions` | `list[str]`       | Summary strings of actions taken so far                  |
| `step_number`      | `int`             | Current step count                                       |
| `feedback`         | `string`          | Server feedback on the last graded action                |
| `done`             | `bool`            | Whether the episode has ended                            |

### Action Space

| Field            | Type                                                      | Description                                          |
|------------------|-----------------------------------------------------------|------------------------------------------------------|
| `action_type`    | `flag_issue \| approve \| request_changes \| add_comment` | The kind of action being taken                       |
| `issues`         | `list[IssueReport]`                                       | List of issues being reported (for `flag_issue`)     |
| `comment`        | `string`                                                  | Optional overall comment                             |
| `final_verdict`  | `approve \| request_changes \| reject \| null`            | Final decision (required for terminal actions)       |

#### IssueReport Fields

| Field            | Type                                              | Description                              |
|------------------|---------------------------------------------------|------------------------------------------|
| `issue_type`     | `bug \| security \| performance \| style \| logic`| Category of issue                        |
| `line_number`    | `int \| null`                                     | Line number in the diff/file             |
| `severity`       | `low \| medium \| high \| critical`               | Severity level                           |
| `description`    | `string`                                          | Human-readable description               |
| `suggested_fix`  | `string \| null`                                  | Optional remediation suggestion          |

---

## Tasks

| Task Name        | Difficulty | Max Steps | Issues to Find | Description                                                       |
|------------------|------------|-----------|----------------|-------------------------------------------------------------------|
| `easy-review`    | Easy       | 8         | 2              | Off-by-one error and null dereference in a pagination function    |
| `medium-review`  | Medium     | 10        | 4              | SQL injection (×2), plaintext password, missing auth on endpoint  |
| `hard-review`    | Hard       | 12        | 6              | Race condition, list mutation during iteration, unprotected global dict, hardcoded secret, MD5 hashing, broken token verify |

### Baseline Scores (Qwen2.5-72B-Instruct)

| Task             | Estimated Score |
|------------------|-----------------|
| `easy-review`    | ~0.45           |
| `medium-review`  | ~0.30           |
| `hard-review`    | ~0.15           |

---

## How the Grader Works

The grader is fully deterministic — no LLM is used for scoring.

Each reported `IssueReport` is matched against each ground-truth issue using a weighted scoring formula:

| Component              | Weight | Criteria                                                    |
|------------------------|--------|-------------------------------------------------------------|
| `issue_type` match     | 0.20   | Exact match on bug/security/performance/style/logic         |
| `severity` match       | 0.30   | Exact match on low/medium/high/critical                     |
| `line_proximity`       | 0.30   | Reported line number within ±3 of ground truth line number  |
| `description_keywords` | 0.20   | Keyword overlap (Jaccard-style) between descriptions        |

A reported issue is considered a **true positive** if it scores ≥ 0.5 against an unmatched ground truth issue (greedy best-match). The final score is computed as:

```
precision = true_positives / reported_issues
recall    = true_positives / ground_truth_issues
f1        = 2 * precision * recall / (precision + recall)
fp_penalty = min(0.2, false_positives * 0.05)
score     = clamp(f1 - fp_penalty, 0.0, 1.0)
```

Scores are always in **[0, 1]**.

---

## Setup and Running

### Prerequisites

- Python 3.11+
- Docker (for containerized deployment)
- A HuggingFace account with API token (for inference)

### Local Development

**1. Install server dependencies:**
```bash
pip install -r server/requirements.txt
```

**2. Start the FastAPI server:**
```bash
CODE_REVIEW_TASK=easy-review uvicorn server.main:app --host 0.0.0.0 --port 7860
```

**3. Install inference dependencies (separate terminal):**
```bash
pip install -r requirements.txt
```

**4. Run inference:**
```bash
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export HF_TOKEN=hf_your_token_here
export CODE_REVIEW_TASK=easy-review
export CODE_REVIEW_SERVER_URL=http://localhost:7860

python inference.py
```

### Docker

**Build the image:**
```bash
docker build -t code-review-env .
```

**Run the server container:**
```bash
docker run -p 7860:7860 \
  -e CODE_REVIEW_TASK=easy-review \
  code-review-env
```

**Run inference against the Docker server:**
```bash
export CODE_REVIEW_SERVER_URL=http://localhost:7860
export HF_TOKEN=hf_your_token_here
python inference.py
```

### HuggingFace Spaces Deployment

1. Create a new Space on [huggingface.co/spaces](https://huggingface.co/spaces)
2. Select **Docker** as the SDK
3. Push this repository to the Space:

```bash
git remote add space https://huggingface.co/spaces/YOUR_USERNAME/code-review-assistant
git push space main
```

4. Set the following Space secrets in the HF Space settings:
   - `HF_TOKEN` — your HuggingFace API token
   - `CODE_REVIEW_TASK` — e.g., `easy-review`
   - `MODEL_NAME` — e.g., `Qwen/Qwen2.5-72B-Instruct`
   - `API_BASE_URL` — e.g., `https://router.huggingface.co/v1`

The Space will automatically build and serve on port 7860.

---

## Environment Variables

| Variable                  | Default                              | Description                                    |
|---------------------------|--------------------------------------|------------------------------------------------|
| `API_BASE_URL`            | `https://router.huggingface.co/v1`  | LLM API base URL (OpenAI-compatible)           |
| `MODEL_NAME`              | `Qwen/Qwen2.5-72B-Instruct`         | Model identifier                               |
| `HF_TOKEN`                | *(required)*                         | API key for the LLM endpoint                   |
| `CODE_REVIEW_TASK`        | `easy-review`                        | Task name: `easy-review`, `medium-review`, `hard-review` |
| `CODE_REVIEW_SERVER_URL`  | `http://localhost:7860`             | URL of the FastAPI environment server          |
| `IMAGE_NAME`              | *(optional)*                         | Docker image name (used by CI/CD pipelines)    |

---

## API Endpoints

| Method | Path      | Description                                         |
|--------|-----------|-----------------------------------------------------|
| `POST` | `/reset`  | Reset the environment, returns initial observation  |
| `POST` | `/step`   | Submit an action, returns observation + reward      |
| `GET`  | `/state`  | Get current environment state                       |
| `GET`  | `/health` | Health check                                        |

---

## Stdout Log Format

```
[START] task=<task_name> env=<benchmark> model=<model_name>
[STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
[END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
```

Example output:
```
[START] task=easy-review env=code-review-assistant model=Qwen/Qwen2.5-72B-Instruct
[STEP]  step=1 action=flag_issue(2_issues) reward=0.78 done=false error=null
[STEP]  step=2 action=request_changes reward=0.78 done=true error=null
[END]   success=true steps=2 score=0.78 rewards=0.78,0.78
```

---

## Project Structure

```
code-review-env/
├── inference.py                  # Root-level inference script (required by OpenEnv)
├── openenv.yaml                  # OpenEnv spec metadata
├── Dockerfile                    # Builds and serves the full environment
├── README.md                     # This file
├── requirements.txt              # Inference script dependencies
├── server/
│   ├── __init__.py
│   ├── main.py                   # FastAPI app (/reset, /step, /state, /health)
│   ├── env.py                    # Core environment logic and state management
│   ├── models.py                 # Pydantic models for actions, observations, results
│   ├── grader.py                 # Deterministic F1-based issue grader
│   ├── requirements.txt          # Server dependencies
│   └── tasks/
│       ├── __init__.py
│       ├── base_task.py          # Abstract base task class
│       ├── easy_task.py          # Task 1: off-by-one + null dereference
│       ├── medium_task.py        # Task 2: SQL injection + missing auth
│       └── hard_task.py          # Task 3: race condition + memory leak + insecure crypto
└── fixtures/
    ├── easy_pr.json              # Ground truth for easy task
    ├── medium_pr.json            # Ground truth for medium task
    └── hard_pr.json              # Ground truth for hard task
```

---

## License

MIT
