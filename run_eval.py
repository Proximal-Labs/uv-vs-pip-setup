#!/usr/bin/env python3
import argparse
import csv
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List
import json as _json
import urllib.request
import urllib.error
import selectors


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def now_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def read_task_spec(task_dir: str) -> dict:
    js = os.path.join(task_dir, "task.json")
    if not os.path.exists(js):
        raise FileNotFoundError(f"No task.json found in {task_dir}")
    with open(js, "r", encoding="utf-8") as f:
        return json.load(f)


def discover_tasks(tasks_dir: str) -> List[str]:
    tasks: List[str] = []
    for root, _, files in os.walk(tasks_dir):
        if "task.json" in files:
            tasks.append(root)
    tasks.sort()
    return tasks


def copy_repo(src: str, dst: str) -> None:
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def list_py_files(repo_dir: str) -> List[str]:
    return [str(p) for p in pathlib.Path(repo_dir).rglob("*.py")]


def infer_imports(repo_dir: str) -> List[str]:
    imports = set()
    pat_import = re.compile(r"^\s*import\s+([a-zA-Z_][\w\.]*)")
    pat_from = re.compile(r"^\s*from\s+([a-zA-Z_][\w\.]*)\s+import\s+")
    for f in list_py_files(repo_dir):
        try:
            with open(f, "r", encoding="utf-8") as fh:
                for line in fh:
                    m1 = pat_import.match(line)
                    if m1:
                        imports.add(m1.group(1).split(".")[0])
                        continue
                    m2 = pat_from.match(line)
                    if m2:
                        imports.add(m2.group(1).split(".")[0])
        except Exception:
            continue
    local = set()
    for p in pathlib.Path(repo_dir).rglob("*"):
        if p.is_file() and p.suffix == ".py":
            local.add(p.stem)
        if p.is_dir() and (p / "__init__.py").exists():
            local.add(p.name)
    imports = {m for m in imports if m not in local}
    try:
        stdlib = set(sys.stdlib_module_names)  # type: ignore[attr-defined]
        imports = {m for m in imports if m not in stdlib}
    except Exception:
        pass
    special = {
        "PIL": "Pillow",
        "bs4": "beautifulsoup4",
        "cv2": "opencv-python",
        "skimage": "scikit-image",
        "yaml": "PyYAML",
        "Crypto": "pycryptodome",
    }
    return [special.get(m, m) for m in sorted(imports)]


def run_shell(command: str, cwd: str, env: dict, timeout: int):
    start = time.time()
    actual_timeout = None if (timeout is None or timeout <= 0) else timeout
    proc = subprocess.run(["bash", "-lc", command], cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=actual_timeout)
    return proc.returncode, proc.stdout, proc.stderr, time.time() - start


def run_shell_stream(command: str, cwd: str, env: dict, timeout: int, out_path: str, err_path: str, inactivity_timeout: int | None = None):
    start = time.time()
    actual_timeout = None if (timeout is None or timeout <= 0) else timeout
    inactivity_limit = None if (inactivity_timeout is None or inactivity_timeout <= 0) else inactivity_timeout
    proc = subprocess.Popen(["bash", "-lc", command], cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
    sel = selectors.DefaultSelector()
    if proc.stdout is not None:
        sel.register(proc.stdout, selectors.EVENT_READ)
    if proc.stderr is not None:
        sel.register(proc.stderr, selectors.EVENT_READ)
    last_activity = time.time()
    with open(out_path, "a", encoding="utf-8") as fo, open(err_path, "a", encoding="utf-8") as fe:
        while True:
            if actual_timeout is not None and (time.time() - start) > actual_timeout:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except Exception:
                    proc.kill()
                break
            if inactivity_limit is not None and (time.time() - last_activity) > inactivity_limit:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except Exception:
                    proc.kill()
                fe.write(f"\n[watchdog] Killed due to {inactivity_limit}s of inactivity.\n")
                fe.flush()
                break
            events = sel.select(timeout=0.5)
            if not events:
                # Check if process exited
                if proc.poll() is not None:
                    # Drain remaining buffers if any
                    if proc.stdout is not None:
                        rem = proc.stdout.read()
                        if rem:
                            fo.write(rem)
                            fo.flush()
                            last_activity = time.time()
                    if proc.stderr is not None:
                        rem = proc.stderr.read()
                        if rem:
                            fe.write(rem)
                            fe.flush()
                            last_activity = time.time()
                    break
                continue
            for key, _ in events:
                try:
                    chunk = key.fileobj.read()  # type: ignore[attr-defined]
                except Exception:
                    chunk = ""
                if not chunk:
                    continue
                if key.fileobj is proc.stdout:
                    fo.write(chunk)
                    fo.flush()
                else:
                    fe.write(chunk)
                    fe.flush()
                last_activity = time.time()
    rc = proc.wait() if proc.poll() is None else proc.returncode
    return rc, time.time() - start


# ---------------- Optional: real coding agents (OpenHands / Cursor) ----------------

def init_git_baseline(repo_dst: str, timeout: int) -> None:
    env = os.environ.copy()
    cmds = [
        'git init',
        'git config user.email "agent@local"',
        'git config user.name "Agent Runner"',
        'git add -A',
        'git commit --no-verify -m "baseline" --allow-empty',
    ]
    for c in cmds:
        run_shell(c, cwd=repo_dst, env=env, timeout=timeout)


def write_git_diff(repo_dst: str, out_dir: str, timeout: int) -> None:
    env = os.environ.copy()
    os.makedirs(out_dir, exist_ok=True)
    rc, out, err, _ = run_shell('git diff --name-only HEAD', cwd=repo_dst, env=env, timeout=timeout)
    with open(os.path.join(out_dir, 'changed_files.txt'), 'w', encoding='utf-8') as f:
        f.write(out)
    rc, out, err, _ = run_shell('git diff HEAD', cwd=repo_dst, env=env, timeout=timeout)
    with open(os.path.join(out_dir, 'diff.patch'), 'w', encoding='utf-8') as f:
        f.write(out)


def run_openhands_agent(repo_dst: str, model: str, prompt: str, timeout: int, version: str | None = None, inactivity_timeout: int | None = None):
    import shlex
    env = os.environ.copy()
    env['LLM_MODEL'] = model
    # Prefer explicit LLM_API_KEY; otherwise map provider; otherwise fallback to OPENROUTER_API_KEY
    if 'LLM_API_KEY' in os.environ and os.environ.get('LLM_API_KEY'):
        env['LLM_API_KEY'] = os.environ['LLM_API_KEY']
    else:
        provider = (model.split('/', 1)[0] if '/' in model else '').lower()
        provider_key_env = {
            'anthropic': 'ANTHROPIC_API_KEY',
            'openai': 'OPENAI_API_KEY',
            'google': 'GOOGLE_API_KEY',
            'xai': 'XAI_API_KEY',
        }.get(provider)
        if provider_key_env and os.environ.get(provider_key_env):
            env['LLM_API_KEY'] = os.environ[provider_key_env]
        elif os.environ.get('OPENROUTER_API_KEY'):
            env['LLM_API_KEY'] = os.environ['OPENROUTER_API_KEY']
    env['SANDBOX_VOLUMES'] = f"{repo_dst}:/workspace:rw"
    # Headless, non-interactive environment
    env['CI'] = '1'
    env['TERM'] = 'dumb'
    env['SECURITY_CONFIRMATION_MODE'] = 'false'
    # Seed minimal config file
    cfg_dir = os.path.join(repo_dst, ".openhands_cfg")
    os.makedirs(cfg_dir, exist_ok=True)
    cfg_path = os.path.join(cfg_dir, "config.toml")
    try:
        with open(cfg_path, "w", encoding="utf-8") as cf:
            cf.write((
                "[security]\n"
                "confirmation_mode = false\n\n"
                "[workspace]\n"
                "base = \"/workspace\"\n\n"
                "[search]\n"
                "enabled = false\n"
            ))
    except Exception:
        pass
    # Build base runner (top-level CLI; omit --no-confirmation for compatibility)
    from_spec = f"openhands-ai=={version}" if version else "openhands-ai"
    flags = [
        "--override-cli-mode", "true",
        "--log-level", "info",
        "--config-file", shlex.quote(cfg_path),
    ]
    base_runner = f"uvx --python 3.12 --from {from_spec} openhands " + " ".join(flags)
    # PTY-first when available
    use_script = shutil.which('script') is not None
    if use_script:
        cmd = f"printf '%s\\n' {shlex.quote(prompt)} | script -q /dev/null {base_runner}"
    else:
        cmd = f"printf '%s\\n' {shlex.quote(prompt)} | {base_runner}"
    # Stream outputs live with inactivity watchdog
    out_path = os.path.join(os.path.dirname(repo_dst), 'agent.out')
    err_path = os.path.join(os.path.dirname(repo_dst), 'agent.err')
    rc, dur = run_shell_stream(cmd, cwd=repo_dst, env=env, timeout=timeout, out_path=out_path, err_path=err_path, inactivity_timeout=inactivity_timeout)
    env_snapshot = {
        "LLM_MODEL": env.get("LLM_MODEL"),
        "has_LLM_API_KEY": bool(env.get("LLM_API_KEY")),
        "SANDBOX_VOLUMES": env.get("SANDBOX_VOLUMES"),
        "CI": env.get("CI"),
        "TERM": env.get("TERM"),
        "from_spec": from_spec,
        "config": cfg_path,
        "runner": "script" if use_script else "plain",
    }
    # Read back small tails for return
    try:
        with open(out_path, 'r', encoding='utf-8') as fo: aout = fo.read()
    except Exception:
        aout = ""
    try:
        with open(err_path, 'r', encoding='utf-8') as fe: aerr = fe.read()
    except Exception:
        aerr = ""
    return rc, aout, aerr, dur, cmd, env_snapshot


def run_cursor_agent(repo_dst: str, model: str, prompt: str, timeout: int):
    import shlex
    env = os.environ.copy()
    if 'CURSOR_API_KEY' not in env:
        raise RuntimeError('CURSOR_API_KEY is required for --agent cursor')
    # cursor-agent reads CURSOR_API_KEY; ensure binary exists
    if shutil.which('cursor-agent') is None:
        raise RuntimeError('cursor-agent not found in PATH. Install the Cursor CLI agent.')
    cmd = f"cursor-agent -p --model {shlex.quote(model)} --force --output-format text {shlex.quote(prompt)}"
    rc, out, err, dur = run_shell(cmd, cwd=repo_dst, env=env, timeout=timeout)
    return rc, out, err, dur, cmd, {"CURSOR_API_KEY": bool(env.get("CURSOR_API_KEY"))}


def _http_post_json(url: str, headers: Dict[str, str], payload: Dict[str, Any]) -> Dict[str, Any]:
    data = _json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", **headers}, method="POST")
    with urllib.request.urlopen(req, timeout=60) as resp:
        body = resp.read().decode("utf-8")
        return _json.loads(body)


def call_openai_chat(model: str, system_prompt: str, user_prompt: str) -> str:
    api_key = os.environ["OPENAI_API_KEY"]
    url = "https://api.openai.com/v1/chat/completions"
    payload = {"model": model, "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}], "temperature": 0}
    res = _http_post_json(url, {"Authorization": f"Bearer {api_key}"}, payload)
    return res["choices"][0]["message"]["content"]


def call_anthropic_chat(model: str, system_prompt: str, user_prompt: str) -> str:
    api_key = os.environ["ANTHROPIC_API_KEY"]
    url = "https://api.anthropic.com/v1/messages"
    payload = {"model": model, "max_tokens": 1024, "temperature": 0, "system": system_prompt, "messages": [{"role": "user", "content": user_prompt}]}
    headers = {"x-api-key": api_key, "anthropic-version": "2023-06-01"}
    res = _http_post_json(url, headers, payload)
    parts = res.get("content", [])
    texts = [p["text"] for p in parts if p.get("type") == "text" and "text" in p]
    return "\n".join(texts)


def call_openrouter_chat(model: str, system_prompt: str, user_prompt: str) -> str:
    api_key = os.environ["OPENROUTER_API_KEY"]
    url = "https://openrouter.ai/api/v1/chat/completions"
    payload = {"model": model, "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}], "temperature": 0}
    res = _http_post_json(url, {"Authorization": f"Bearer {api_key}"}, payload)
    return res["choices"][0]["message"]["content"]


def build_repo_context(repo_dir: str, max_files: int = 200) -> str:
    lines: List[str] = ["Repository tree (relative):"]
    rels: List[str] = []
    for p in pathlib.Path(repo_dir).rglob("*"):
        if ".venv" in p.parts:
            continue
        rels.append(str(p.relative_to(repo_dir)))
        if len(rels) >= max_files:
            break
    for r in sorted(rels):
        lines.append(f"- {r}")
    return "\n".join(lines)


def parse_commands_from_text(text: str) -> List[str]:
    try:
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1 and end > start:
            arr = _json.loads(text[start:end+1])
            if isinstance(arr, list) and all(isinstance(x, str) for x in arr):
                return [c.strip() for c in arr if c.strip()]
    except Exception:
        pass
    m = re.search(r"```(?:bash|sh)?\n([\s\S]*?)```", text, re.IGNORECASE)
    if m:
        block = m.group(1)
        cmds = []
        for line in block.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("$"):
                line = line[1:].strip()
            cmds.append(line)
        if cmds:
            return cmds
    return [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("#")]


def plan_commands_with_llm(provider: str, model: str, tool_hint: str, description: str, repo_dir: str) -> Dict[str, Any]:
    system_prompt = (
        "You are a coding agent that emits only shell commands to prepare and run Python projects in a sandbox. "
        "Use ONLY the specified package manager (uv or pip) per tool hint. "
        "Commands must be non-interactive and idempotent. Prefer 'uv venv' or 'python -m venv .venv'. "
        "Do not 'source' activation; PATH is prefixed with .venv/bin by the harness. "
        "If installing, use 'uv add ...' or '.venv/bin/pip install ...'. No conda/poetry/sudo."
    )
    user_prompt = (
        f"{description}\n\nTool hint: {tool_hint}\n\n{build_repo_context(repo_dir)}\n\n"
        "If dependencies are missing, infer them from imports and install. "
        "Output ONLY commands, preferably as a JSON array of strings."
    )
    if provider == "openai":
        text = call_openai_chat(model, system_prompt, user_prompt)
    elif provider == "anthropic":
        text = call_anthropic_chat(model, system_prompt, user_prompt)
    elif provider == "openrouter":
        text = call_openrouter_chat(model, system_prompt, user_prompt)
    else:
        raise RuntimeError(f"Unknown provider: {provider}")
    cmds = parse_commands_from_text(text)
    if tool_hint == "uv" and not any(cmd.startswith("uv venv") for cmd in cmds):
        cmds.insert(0, "uv venv")
    if tool_hint != "uv" and not any("python -m venv" in cmd for cmd in cmds):
        cmds.insert(0, "python -m venv .venv")
        cmds.insert(1, ".venv/bin/pip install -U pip")
    cmds = [c for c in cmds if not re.search(r"\bsource\b|activate", c)]
    fixed: List[str] = []
    for c in cmds:
        if tool_hint != "uv" and re.match(r"^pip\b", c):
            fixed.append(c.replace("pip", ".venv/bin/pip", 1))
        else:
            fixed.append(c)
    # Drop runtime commands; grader will run the program/tests.
    runtime_patterns = [r"^python\s+(?!-m\s+pip)\b", r"\bpytest\b", r"\buvicorn\b", r"\bgunicorn\b", r"\bstreamlit\b", r"\bnpm\s+start\b"]
    def is_runtime(cmd: str) -> bool:
        return any(re.search(p, cmd) for p in runtime_patterns)
    filtered = [c for c in fixed if not is_runtime(c)]
    return {"commands": filtered, "raw": text}


def verify_deps(repo_dir: str, deps: List[str], tool_hint: str, env: dict, timeout: int, log_path: str) -> Dict[str, Any]:
    """Verify required deps are installed in the task's venv, regardless of uv/pip setup.

    Strategy:
    - Prefer the repo's virtualenv interpreter (.venv/bin/python) for both pip show and import checks.
    - If no venv exists, fall back to uv pip with --python if possible; otherwise best-effort.
    - Handle dist<->module naming differences for import checks.
    """
    import shlex
    required = len(deps)
    passed = 0

    # dist -> module fallback mapping
    dist_to_module = {
        "opencv-python": "cv2",
        "Pillow": "PIL",
        "beautifulsoup4": "bs4",
        "PyYAML": "yaml",
        "scikit-image": "skimage",
        "pycryptodome": "Crypto",
    }

    venv_py = os.path.join(repo_dir, ".venv", "bin", "python")
    venv_pip = os.path.join(repo_dir, ".venv", "bin", "pip")
    have_venv = os.path.isfile(venv_py)

    def log(cmd: str, rc: int, out: str, err: str, dur: float):
        rec = {"cmd": cmd, "rc": rc, "duration_sec": round(dur, 3), "stdout": out, "stderr": err}
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(_json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception:
            pass

    for dist in deps:
        module = dist_to_module.get(dist, dist)

        # 1) Show installed
        if have_venv:
            cmd_show = f"{venv_py} -m pip show {dist}"
        else:
            # Best-effort with uv pointing at .venv if it appears, otherwise plain
            if os.path.isdir(os.path.join(repo_dir, ".venv")):
                cmd_show = f"uv pip --python {shlex.quote(venv_py)} show {dist}"
            else:
                cmd_show = f"uv pip show {dist}" if tool_hint == "uv" else f"pip show {dist}"

        rc1, out1, err1, dur1 = run_shell(cmd_show, cwd=repo_dir, env=env, timeout=timeout)
        log(cmd_show, rc1, out1, err1, dur1)

        # 2) Import test (uses venv python if available; otherwise python3)
        if have_venv:
            cmd_imp = f"{venv_py} -c 'import {module}; import sys; sys.exit(0)'"
        else:
            cmd_imp = f"python3 -c 'import {module}; import sys; sys.exit(0)'"
        rc2, out2, err2, dur2 = run_shell(cmd_imp, cwd=repo_dir, env=env, timeout=timeout)
        log(cmd_imp, rc2, out2, err2, dur2)

        # Count as passed if either pip reports installed OR import works in the venv/python we use
        if rc1 == 0 or rc2 == 0:
            passed += 1

    # Context listing
    if have_venv:
        list_cmd = f"{venv_py} -m pip list"
    else:
        list_cmd = "uv pip list" if tool_hint == "uv" else "pip list"
    rc, out, err, dur = run_shell(list_cmd, cwd=repo_dir, env=env, timeout=timeout)
    log(list_cmd, rc, out, err, dur)

    return {"required": required, "passed": passed, "tool": tool_hint, "used_venv": have_venv}


def run_task(task_dir: str, run_root: str, per_cmd_timeout: int, provider: str, model: str, agent: str, agent_model: str | None, openhands_version: str | None, strict_env: bool = False, agent_inactivity_timeout: int | None = None) -> dict:
    spec = read_task_spec(task_dir)
    repo_rel = spec["repo"]
    tool_hint = spec.get("tool_hint", "pip")
    grade_cmd = spec["grade"]
    description = spec.get("description", "").strip()
    required_deps = spec.get("deps", [])

    sandbox_root = os.path.join(run_root, pathlib.Path(task_dir).name)
    repo_src = os.path.join(task_dir, repo_rel)
    repo_dst = os.path.join(sandbox_root, "repo")
    ensure_dir(sandbox_root)
    copy_repo(repo_src, repo_dst)
    # Write a file tree snapshot for observability
    try:
        with open(os.path.join(sandbox_root, "repo_tree.txt"), "w", encoding="utf-8") as f:
            f.write(build_repo_context(repo_dst, max_files=10000))
    except Exception:
        pass

    env = os.environ.copy()
    venv_bin = os.path.join(repo_dst, ".venv", "bin")
    env["REPO"] = repo_dst
    env["PATH"] = venv_bin + os.pathsep + env.get("PATH", "")

    summary: dict = {"task": pathlib.Path(task_dir).name, "tool_hint": tool_hint, "passed": False, "grade_rc": None, "planned_commands": []}

    # Optional: real agent step that can edit code / create files inside repo
    if agent and agent != 'none':
        if not agent_model:
            raise RuntimeError("--agent-model is required when --agent is not 'none'")
        print(f"Initializing git baseline...", flush=True)
        init_git_baseline(repo_dst, per_cmd_timeout)
        print(f"Running agent: {agent} ({agent_model})...", flush=True)
        a_start = time.time()
        if agent == 'openhands':
            if shutil.which('uvx') is None:
                raise RuntimeError("uvx not found. Install uv (https://docs.astral.sh/uv/) to run OpenHands.")
            arc, aout, aerr, _, agent_cmd, agent_env = run_openhands_agent(repo_dst, agent_model, description, per_cmd_timeout, version=openhands_version, inactivity_timeout=agent_inactivity_timeout)
        elif agent == 'cursor':
            arc, aout, aerr, _, agent_cmd, agent_env = run_cursor_agent(repo_dst, agent_model, description, per_cmd_timeout)
        else:
            raise RuntimeError(f"Unknown agent: {agent}")
        a_dur = round(time.time() - a_start, 2)
        summary['agent'] = {"name": agent, "model": agent_model, "rc": arc, "duration_sec": a_dur, "cmd": agent_cmd}
        with open(os.path.join(sandbox_root, 'agent.out'), 'w', encoding='utf-8') as f: f.write(aout)
        with open(os.path.join(sandbox_root, 'agent.err'), 'w', encoding='utf-8') as f: f.write(aerr)
        # Write agent command and env snapshot
        with open(os.path.join(sandbox_root, 'agent.cmd'), 'w', encoding='utf-8') as f: f.write(agent_cmd)
        try:
            with open(os.path.join(sandbox_root, 'agent_env.json'), 'w', encoding='utf-8') as f:
                _json.dump(agent_env, f, indent=2)
        except Exception:
            pass
        # Write diffs
        try:
            write_git_diff(repo_dst, os.path.join(sandbox_root, 'diff'), per_cmd_timeout)
        except Exception:
            pass

    # If a real agent was used, skip chat planning; otherwise plan env setup commands with LLM
    plan_dur = 0.0
    planned_any = False
    if not agent or agent == 'none':
        print(f"Planning commands with {provider}/{model}...", flush=True)
        plan_start = time.time()
        plan = plan_commands_with_llm(provider=provider, model=model, tool_hint=tool_hint, description=description, repo_dir=repo_dst)
        plan_dur = round(time.time() - plan_start, 2)
        planned = plan["commands"]
        print(f"Planned {len(planned)} command(s) in {plan_dur}s:", flush=True)
        for pc in planned:
            print(f"  $ {pc}", flush=True)
        # Save raw agent output for inspection
        with open(os.path.join(sandbox_root, "agent_raw.txt"), "w", encoding="utf-8") as f:
            f.write(plan.get("raw", ""))
        summary["planned_commands"] = planned
        # Record which setup tool is being used for clarity
        setup_tool = "uv" if any(c.startswith("uv ") or c.startswith("uvx ") for c in planned) else ("pip" if any(re.search(r"(^|\s)pip\b|\.venv/bin/pip", c) for c in planned) else "unknown")
        summary["setup_tool"] = setup_tool
        # Command log jsonl
        cmd_log_path = os.path.join(sandbox_root, "commands.jsonl")
        for c in planned:
            if os.path.isdir(venv_bin):
                env["PATH"] = venv_bin + os.pathsep + env.get("PATH", "")
            print(f"Executing: {c}", flush=True)
            rc, out, err, dur = run_shell(c, cwd=repo_dst, env=env, timeout=per_cmd_timeout)
            print(f"  -> rc={rc} ({round(dur, 2)}s)", flush=True)
            summary.setdefault("executed_commands", []).append({"cmd": c, "rc": rc, "duration_sec": round(dur, 3)})
            try:
                with open(cmd_log_path, "a", encoding="utf-8") as lf:
                    record = {"cmd": c, "rc": rc, "duration_sec": round(dur, 3), "stdout": out, "stderr": err}
                    lf.write(_json.dumps(record, ensure_ascii=False) + "\n")
            except Exception:
                pass
        planned_any = True

    # After agent or planned setup, verify required deps using tool_hint
    env_checks_passed = True
    if required_deps:
        checks_log = os.path.join(sandbox_root, "env_checks.jsonl")
        dep_summary = verify_deps(repo_dst, required_deps, tool_hint, env, per_cmd_timeout, checks_log)
        summary["env_checks"] = dep_summary
        env_checks_passed = (dep_summary.get("passed", 0) == dep_summary.get("required", 0))

    print("Grading...", flush=True)
    grade_start = time.time()
    # Prefer venv python for grading when available or strict_env enabled; otherwise prefer python3
    grade_cmd_exec = grade_cmd
    venv_python = os.path.join(repo_dst, ".venv", "bin", "python")
    have_venv_python = os.path.isfile(venv_python)
    if (strict_env and have_venv_python) or (have_venv_python and re.search(r"\bpython3?\b", grade_cmd)):
        grade_cmd_exec = re.sub(r"\bpython3?\b", venv_python, grade_cmd, count=1)
    elif grade_cmd.strip().startswith("python "):
        grade_cmd_exec = grade_cmd.replace("python ", "python3 ", 1)
    rc, out, err, dur = run_shell(grade_cmd_exec, cwd=repo_dst, env=env, timeout=per_cmd_timeout)
    grade_dur = round(time.time() - grade_start, 2)
    summary["grade_rc"] = rc
    # Strict mode: require env checks AND venv presence
    if strict_env:
        env_checks_passed = env_checks_passed and have_venv_python
    summary["passed"] = (rc == 0) and env_checks_passed
    timings: Dict[str, Any] = {"grade_sec": grade_dur}
    if planned_any:
        timings["plan_sec"] = plan_dur
    summary["timings"] = timings

    with open(os.path.join(sandbox_root, "grade.out"), "w", encoding="utf-8") as f:
        f.write("=== STDOUT ===\n"); f.write(out); f.write("\n=== STDERR ===\n"); f.write(err)
    with open(os.path.join(sandbox_root, "grade.json"), "w", encoding="utf-8") as f:
        _json.dump({"cmd": grade_cmd_exec, "rc": rc, "duration_sec": dur, "env_checks_required": required_deps, "env_checks_passed": env_checks_passed}, f)
    with open(os.path.join(sandbox_root, "result.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Minimal UV/Pip coding-agent eval harness (JSON tasks only)")
    ap.add_argument("--tasks-dir", default="tasks", help="Directory containing task folders")
    ap.add_argument("--only", nargs="*", help="Only run specific task directory names")
    ap.add_argument("--per-cmd-timeout", type=int, default=180, help="Per-command timeout in seconds (0 disables timeout)")
    ap.add_argument("--provider", choices=["openai", "anthropic", "openrouter"], required=True)
    ap.add_argument("--model", required=True, help="LLM model id (e.g., gpt-4o-mini, claude-3-5-sonnet-20240620)")
    ap.add_argument("--agent", choices=["none", "openhands", "cursor"], default="none", help="Run a real coding agent before grading")
    ap.add_argument("--agent-model", help="Agent model id (e.g., anthropic/claude-3-5-sonnet-20240620 for OpenHands, or cursor-small)")
    ap.add_argument("--openhands-version", help="Pin openhands-ai version for uvx (e.g., 0.53.0)")
    ap.add_argument("--strict-env", action="store_true", help="Require deps to be present in .venv and run grade with .venv/bin/python")
    ap.add_argument("--agent-inactivity-timeout", type=int, default=120, help="Kill agent if no output for N seconds (0 disables)")
    args = ap.parse_args()

    # Load .env if present to populate API keys without external deps
    dotenv_path = os.path.join(os.getcwd(), ".env")
    if os.path.exists(dotenv_path):
        try:
            with open(dotenv_path, "r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if not s or s.startswith("#") or "=" not in s:
                        continue
                    k, v = s.split("=", 1)
                    if k and v and k not in os.environ:
                        os.environ[k] = v
        except Exception:
            pass

    tasks_dir = os.path.abspath(args.tasks_dir)
    task_dirs = [t for t in discover_tasks(tasks_dir) if (not args.only or pathlib.Path(t).name in set(args.only))]
    if not task_dirs:
        print("No tasks found.", file=sys.stderr)
        sys.exit(1)

    run_root = os.path.join(os.getcwd(), "runs", now_ts())
    ensure_dir(run_root)

    results: List[dict] = []
    for t in task_dirs:
        print(f"Running task: {pathlib.Path(t).name}")
        res = run_task(t, run_root, args.per_cmd_timeout, args.provider, args.model, args.agent, args.agent_model, args.openhands_version, strict_env=args.strict_env, agent_inactivity_timeout=args.agent_inactivity_timeout)
        print(f"  -> {'PASS' if res['passed'] else 'FAIL'}")
        results.append(res)

    with open(os.path.join(run_root, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(run_root, "results.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["task", "tool_hint", "passed", "grade_rc"])
        for r in results:
            w.writerow([r["task"], r["tool_hint"], int(r["passed"]), r["grade_rc"]])

    num_pass = sum(1 for r in results if r["passed"])
    print(f"\nScore: {num_pass}/{len(results)} tasks passed")
    print(f"Artifacts: {run_root}")


if __name__ == "__main__":
    main()


