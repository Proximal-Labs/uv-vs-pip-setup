#!/usr/bin/env python3
# lean_opencode_eval.py — minimal eval harness using OpenCode CLI (non-interactive)
# - Copies each task repo to runs/<ts>/<task>/repo
# - Writes a per-run OpenCode config enabling tools (bash, git, fs)
# - Runs: opencode run --model <provider/model> "<prompt>"
# - Grades using .venv/bin/python when present

import argparse, csv, json, os, re, shutil, subprocess, sys, time
from typing import Optional, List
from datetime import datetime, timezone
from pathlib import Path

# ---------- tiny utils ----------

def ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def sh(cmd: str, cwd: str, env: dict, timeout: int):
    t0 = time.time()
    proc = subprocess.run(
        ["bash", "-lc", cmd],
        cwd=cwd, env=env, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        timeout=None if timeout <= 0 else timeout,
    )
    return proc.returncode, proc.stdout, proc.stderr, time.time() - t0

def read_task(task_dir: str) -> dict:
    p = os.path.join(task_dir, "task.json")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def find_tasks(tasks_dir: str, only: Optional[set]) -> list[str]:
    out = []
    for root, _, files in os.walk(tasks_dir):
        if "task.json" in files:
            name = Path(root).name
            if (not only) or (name in only):
                out.append(root)
    return sorted(out)

def q(s: str) -> str:
    return "'" + s.replace("'", "'\"'\"'") + "'"

# ---------- deps + grading ----------

DIST_TO_MODULE = {
    "opencv-python": "cv2",
    "Pillow": "PIL",
    "beautifulsoup4": "bs4",
    "PyYAML": "yaml",
    "scikit-image": "skimage",
    "pycryptodome": "Crypto",
}

def verify_deps(repo: str, deps: list[str], env: dict, timeout: int) -> dict:
    """Check packages inside .venv when present; otherwise system python/pip."""
    venv_py = os.path.join(repo, ".venv", "bin", "python")
    have_venv = os.path.isfile(venv_py)
    required = len(deps); passed = 0
    for dist in deps:
        mod = DIST_TO_MODULE.get(dist, dist)
        if have_venv:
            rc1, *_ = sh(f"{venv_py} -m pip show {dist}", cwd=repo, env=env, timeout=timeout)
            rc2, *_ = sh(f"{venv_py} -c 'import {mod}; import sys; sys.exit(0)'", cwd=repo, env=env, timeout=timeout)
        else:
            rc1, *_ = sh(f"pip show {dist}", cwd=repo, env=env, timeout=timeout)
            rc2, *_ = sh(f"python3 -c 'import {mod}; import sys; sys.exit(0)'", cwd=repo, env=env, timeout=timeout)
        if rc1 == 0 or rc2 == 0:
            passed += 1
    return {"required": required, "passed": passed, "used_venv": have_venv}

def choose_grade_cmd(repo: str, grade_cmd: str) -> str:
    venv_py = os.path.join(repo, ".venv", "bin", "python")
    if os.path.isfile(venv_py):
        if re.search(r"\bpython3?\b", grade_cmd):
            return re.sub(r"\bpython3?\b", venv_py, grade_cmd, count=1)
        if re.search(r"\b\S+\.py(\s|$)", grade_cmd):
            return f"{venv_py} {grade_cmd}"
    if grade_cmd.startswith("python "):
        return grade_cmd.replace("python ", "python3 ", 1)
    return grade_cmd

# ---------- OpenCode (non-interactive) ----------

def require_opencode(opencode_cmd: str):
    first = opencode_cmd.strip().split()[0]
    if shutil.which(first) is None:
        raise SystemExit(
            f"'{first}' not found on PATH.\n"
            "Install OpenCode with one of:\n"
            "  curl -fsSL https://opencode.ai/install | bash\n"
            "  npm i -g opencode-ai   # or: brew install sst/tap/opencode\n"
        )

def write_opencode_config(base_cfg_home: str):
    """
    Write a minimal config that enables tools (bash, git, fs).
    OpenCode reads $XDG_CONFIG_HOME/opencode/opencode.json (or config.json).
    """
    oc_dir = os.path.join(base_cfg_home, "opencode")
    ensure_dir(oc_dir)
    cfg_path = os.path.join(oc_dir, "opencode.json")
    cfg = {
        "$schema": "https://opencode.ai/config.json",
        "tools": {
            "bash": True,
            "git":  True,
            "fs":   True
        }
    }
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

def run_opencode(repo: str, opencode_cmd: str, model: str, prompt: str, timeout: int, xdg_home: str) -> dict:
    """
    Non-interactive one-shot:
      opencode run --model <provider/model> "<prompt>"
    We point XDG_CONFIG_HOME to a per-run config that enables tools.
    """
    require_opencode(opencode_cmd)
    env = os.environ.copy()

    # Route OpenRouter to OpenAI-compatible envs (harmless if not used)
    if env.get("OPENROUTER_API_KEY"):
        env.setdefault("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
        env.setdefault("OPENAI_API_KEY",  env["OPENROUTER_API_KEY"])

    # Point OpenCode to our injected config
    env["XDG_CONFIG_HOME"] = xdg_home
    write_opencode_config(xdg_home)

    # Add some log visibility
    cmd = f'{opencode_cmd} run --model {q(model)} --print-logs --log-level INFO {q(prompt)}'
    rc, out, err, dur = sh(cmd, cwd=repo, env=env, timeout=timeout)

    # Persist logs
    sandbox = os.path.join(Path(repo).parent)
    with open(os.path.join(sandbox, "agent.out"), "w", encoding="utf-8") as fo: fo.write(out)
    with open(os.path.join(sandbox, "agent.err"), "w", encoding="utf-8") as fe: fe.write(err)
    with open(os.path.join(sandbox, "agent.cmd"), "w", encoding="utf-8") as fc: fc.write(cmd)
    return {"rc": rc, "dur": dur}

# ---------- main runner ----------

def run_task(task_dir: str, run_root: str, per_cmd_timeout: int, opencode_cmd: str, agent_model: str, xdg_home: str) -> dict:
    spec = read_task(task_dir)
    repo_rel  = spec["repo"]
    grade_cmd = spec["grade"]
    desc      = spec.get("description", "").strip()
    deps      = spec.get("deps", [])

    sandbox = os.path.join(run_root, Path(task_dir).name)
    repo_src = os.path.join(task_dir, repo_rel)
    repo     = os.path.join(sandbox, "repo")
    ensure_dir(sandbox)
    if os.path.exists(repo): shutil.rmtree(repo)
    shutil.copytree(repo_src, repo)

    # Git baseline so diffs are meaningful
    for c in [
        'git init', 'git config user.email "agent@local"',
        'git config user.name "Agent Runner"', 'git add -A',
        'git commit --no-verify -m baseline --allow-empty'
    ]:
        _ = sh(c, cwd=repo, env=os.environ.copy(), timeout=30)

    summary = {"task": Path(task_dir).name, "passed": False}

    # Agent step (OpenCode)
    res = run_opencode(repo, opencode_cmd=opencode_cmd, model=agent_model, prompt=desc, timeout=per_cmd_timeout, xdg_home=xdg_home)
    summary["agent"] = {"rc": res["rc"], "dur": round(res["dur"],2)}

    # Optional dep checks
    env_ok = True
    if deps:
        dep = verify_deps(repo, deps, os.environ.copy(), per_cmd_timeout)
        summary["env_checks"] = dep
        env_ok = (dep["passed"] == dep["required"])

    # Grade
    gcmd = choose_grade_cmd(repo, grade_cmd)
    rc, out, err, dur = sh(gcmd, cwd=repo, env=os.environ.copy(), timeout=per_cmd_timeout)
    with open(os.path.join(sandbox, "grade.out"), "w", encoding="utf-8") as f:
        f.write("=== STDOUT ===\n"); f.write(out); f.write("\n=== STDERR ===\n"); f.write(err)
    summary["grade_rc"] = rc
    summary["passed"]   = (rc == 0) and env_ok

    with open(os.path.join(sandbox, "result.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary

def main():
    ap = argparse.ArgumentParser("lean OpenCode eval")
    ap.add_argument("--tasks-dir", default="tasks")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--per-cmd-timeout", type=int, default=600)
    ap.add_argument("--opencode-cmd", default="opencode", help="How to invoke OpenCode (e.g., 'opencode')")
    ap.add_argument("--agent-model", required=True, help="provider/model slug, e.g., openrouter/qwen/qwen-2.5-coder-32b-instruct")
    args = ap.parse_args()

    # Load .env if present (keys/base URLs)
    dotenv = os.path.join(os.getcwd(), ".env")
    if os.path.exists(dotenv):
        with open(dotenv, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#") or "=" not in s: continue
                k, v = s.split("=", 1)
                if k and v and k not in os.environ:
                    os.environ[k] = v

    tasks = find_tasks(os.path.abspath(args.tasks_dir), set(args.only) if args.only else None)
    if not tasks:
        print("No tasks found.", file=sys.stderr); sys.exit(1)

    run_root = os.path.join(os.getcwd(), "runs", ts()); ensure_dir(run_root)

    # Per-run OpenCode config (enables tools) under XDG_CONFIG_HOME
    xdg_home = os.path.join(run_root, "_oc_cfg"); ensure_dir(xdg_home)

    results = []
    for t in tasks:
        print(f"Running task: {Path(t).name}")
        res = run_task(t, run_root, args.per_cmd_timeout, opencode_cmd=args.opencode_cmd, agent_model=args.agent_model, xdg_home=xdg_home)
        print(f"  -> {'PASS' if res['passed'] else 'FAIL'}")
        results.append(res)

    # Summaries
    with open(os.path.join(run_root, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(run_root, "results.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["task","passed","grade_rc"])
        for r in results: w.writerow([r["task"], int(r["passed"]), r.get("grade_rc")])
    print(f"\nScore: {sum(int(r['passed']) for r in results)}/{len(results)} | Artifacts: {run_root}")

if __name__ == "__main__":
    main()
