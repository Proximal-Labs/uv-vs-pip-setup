## uv-eval: Local setup and usage

This repo contains lightweight harnesses to evaluate coding agents locally:

- `lean_cursor_eval.py` runs the Cursor Agent CLI headlessly
- `lean_openhands_eval.py` runs OpenHands (Docker runtime) via `uvx`

Both harnesses create a timestamped folder under `runs/` with per-task artifacts (`repo/`, `agent.*`, `grade.out`, `result.json`).

### Prerequisites

- **Python 3.11+** (macOS ships 3.9/3.10; install a newer version if needed)
- **uv** (fast Python package manager)
- **Docker Desktop** (running) for OpenHands
- **git** in PATH

Suggested installs on macOS:

```bash
brew install uv git
brew install --cask docker
```

### Install the CLIs

- **Cursor Agent CLI**:

```bash
curl https://cursor.com/install -fsS | bash
```

- **OpenHands CLI**: no global install required; this harness runs it via `uvx` on demand.

#### Docker image for headless OpenHands

OpenHands runs inside a runtime container. Pull the runtime image once before running:

```bash
docker pull docker.all-hands.dev/all-hands-ai/runtime:0.54-nikolaik
```

Notes:

- The harness derives the tag from `--openhands-version` (e.g., `0.54.0` → `0.54-nikolaik`).
- You can override the image explicitly:

```bash
export SANDBOX_RUNTIME_CONTAINER_IMAGE=docker.all-hands.dev/all-hands-ai/runtime:0.54-nikolaik
```

### Set required API keys

- **Cursor**: `export CURSOR_API_KEY="<your_key>"`
- **OpenHands model routing**:
  - Using OpenRouter: `export OPENROUTER_API_KEY="<your_key>"`
    - Use model ids like `openrouter/vendor/model` (examples below)
  - Or set a direct provider key (e.g., `ANTHROPIC_API_KEY`, `AZURE_OPENAI_API_KEY`, `GOOGLE_API_KEY`, `COHERE_API_KEY`) and use that vendor’s model id

### Quick start

From the repo root:

```bash
# Cursor Agent (uses local host, no Docker)
python3 lean_cursor_eval.py --tasks-dir tasks_uv --agent-model gpt-5

# OpenHands (Docker runtime; longer timeout recommended)
python3 lean_openhands_eval.py --tasks-dir tasks_uv \
  --agent openhands \
  --agent-model openrouter/qwen/qwen3-30b-a3b-thinking-2507 \
  --openhands-version 0.54.0 \
  --per-cmd-timeout 1200
```

These are the same commands listed in `commands-that-work.txt`.

### Choosing tasks

- `tasks_uv/` tasks expect a `pyproject.toml`-based flow (uv). The harness verifies manifests accordingly.
- `tasks_pip/` tasks expect `requirements.txt` (pip). The OpenHands harness adjusts checks for pip.

Run only a subset:

```bash
python3 lean_cursor_eval.py --tasks-dir tasks_uv --only sudoku spell_checker --agent-model gpt-5
python3 lean_openhands_eval.py --tasks-dir tasks_uv --only sudoku --agent openhands --agent-model openrouter/qwen/qwen3-32b --openhands-version 0.54.0
```

### Interpreting results (what “PASS” means)

Each task writes `runs/<TS>/<task>/result.json` with fields that drive pass/fail.

- **For Cursor (`lean_cursor_eval.py`)**:

  - `grade_rc`: exit code of the task’s `grade` command
  - `env_checks`: dependency import/show checks (prefers `.venv` if present)
  - `manifest_checks` (only enforced for uv tasks): required deps must be listed in `pyproject.toml`
  - `passed` is true when `grade_rc == 0` and environment + manifest checks succeed

- **For OpenHands (`lean_openhands_eval.py`)**:
  - `agent`: rc/duration of the OpenHands session
  - `command_checks`: parsed from `agent.err`; enforces hygiene and tool correctness
    - `hits.venv_created`: saw `uv venv` or `python -m venv .venv`
    - `hits.installed_in_venv`: installs happened inside the venv (e.g., `uv sync`, `uv add`, or `.venv/bin/python -m pip install`)
    - `hits.no_global_pip`: no global `pip install`, `--user`, or `sudo pip`
    - `hits.used_uv_commands`: for uv tasks, used `uv` commands
    - `manifest`:
      - uv tasks: deps must be in `pyproject.toml`
      - pip tasks: deps must be in `requirements.txt`
  - `manifest_ok`: shortcut for `command_checks.manifest.ok`
  - `env_checks`: deps validated inside the same Docker runtime the agent used
  - `grade_rc`: result of the `grade` command executed in that same runtime
  - `passed` requires: `grade_rc == 0` AND env checks pass AND manifest is correct AND command hygiene passes

Practical takeaway:

- For uv tasks, ensure the agent uses `uv venv` + `uv sync`/`uv add`, and that deps live in `pyproject.toml` (not `requirements.txt`).
- For pip tasks, avoid global installs; install within `.venv` or use `uv pip install` with an active venv, and list deps in `requirements.txt`.

### Example: raw OpenHands CLI (advanced)

The harness runs OpenHands via `uvx` with a `config.toml` and Docker runtime. If you need to reproduce a run manually (e.g., to debug), you can invoke the CLI directly as follows (0.54.0 syntax):

```bash
PROMPT="Fix the task with correct env and grading"
CI=1 TERM=dumb printf '%s\n' "$PROMPT" | \
uvx --from openhands-ai==0.54.0 openhands \
  --override-cli-mode true \
  --config-file /tmp/oh-config.toml \
  --log-level info
# If a TTY is required, wrap with: script -q /dev/null -- <cmd>
```

### Output files to inspect

Inside `runs/<TS>/<task>/`:

- `repo/`: a working copy of the task repo (the agent edits here)
- `agent.out`, `agent.err`, `agent.cmd`: the agent interaction transcript and exact command used
- `grade.out`: combined stdout/stderr from grading
- `result.json`: machine-readable summary used for scoring

### Models you can use

Pass any of the following to `--agent-model` (OpenHands) or to Cursor’s `--model` if applicable. This list is sourced from `list-of-models.txt`.

```text
deepseek/deepseek-chat
deepseek/deepseek-chat-v3-0324
deepseek/deepseek-chat-v3-0324:free
deepseek/deepseek-chat-v3.1
deepseek/deepseek-chat-v3.1:free
deepseek/deepseek-r1
deepseek/deepseek-r1-0528
deepseek/deepseek-r1-distill-llama-70b
meta-llama/llama-3-70b-instruct
meta-llama/llama-3-8b-instruct
meta-llama/llama-3.1-405b-instruct
meta-llama/llama-3.1-70b-instruct
meta-llama/llama-3.1-8b-instruct
meta-llama/llama-3.2-3b-instruct
meta-llama/llama-3.3-70b-instruct
meta-llama/llama-3.3-70b-instruct:free
meta-llama/llama-3.3-8b-instruct:free
meta-llama/llama-4-maverick
meta-llama/llama-4-maverick:free
meta-llama/llama-4-scout
meta-llama/llama-4-scout:free
microsoft/phi-3-medium-128k-instruct
microsoft/phi-3-mini-128k-instruct
microsoft/phi-3.5-mini-128k-instruct
mistralai/codestral-2501
mistralai/codestral-2508
mistralai/devstral-medium
mistralai/devstral-small
mistralai/devstral-small-2505
mistralai/devstral-small-2505:free
mistralai/magistral-medium-2506
mistralai/magistral-medium-2506:thinking
mistralai/magistral-small-2506
mistralai/ministral-8b
mistralai/mistral-7b-instruct
mistralai/mistral-7b-instruct-v0.1
mistralai/mistral-7b-instruct-v0.3
mistralai/mistral-7b-instruct:free
mistralai/mistral-large
mistralai/mistral-large-2407
mistralai/mistral-large-2411
mistralai/mistral-medium-3
mistralai/mistral-medium-3.1
mistralai/mistral-nemo
mistralai/mistral-saba
mistralai/mistral-small
mistralai/mistral-small-24b-instruct-2501
mistralai/mistral-small-3.1-24b-instruct
mistralai/mistral-small-3.1-24b-instruct:free
mistralai/mistral-small-3.2-24b-instruct
mistralai/mistral-small-3.2-24b-instruct:free
mistralai/mistral-tiny
mistralai/mixtral-8x22b-instruct
mistralai/mixtral-8x7b-instruct
mistralai/pixtral-12b
mistralai/pixtral-large-2411
qwen/qwen-2.5-72b-instruct
qwen/qwen-max
qwen/qwen-plus
qwen/qwen-turbo
qwen/qwen3-14b
qwen/qwen3-235b-a22b
qwen/qwen3-235b-a22b-2507
qwen/qwen3-235b-a22b-thinking-2507
qwen/qwen3-235b-a22b:free
qwen/qwen3-30b-a3b
qwen/qwen3-30b-a3b-instruct-2507
qwen/qwen3-30b-a3b-thinking-2507
qwen/qwen3-32b
qwen/qwen3-4b:free
qwen/qwen3-coder
qwen/qwen3-coder-30b-a3b-instruct
qwen/qwen3-coder:free
qwen/qwq-32b
```

### Tips & troubleshooting

- If Docker isn’t running, OpenHands runs will fail early. Start Docker Desktop first.
- If `uvx` isn’t found, install uv (`brew install uv`), then open a new shell.
- Increase `--per-cmd-timeout` for bigger tasks (e.g., 1800 or 3600 seconds).
- The harnesses use JSON for configuration/artifacts; no YAML required.
- Agents are expected to manage and install their own deps inside the sandbox.
