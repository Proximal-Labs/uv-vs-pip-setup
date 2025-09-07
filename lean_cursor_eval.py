#!/usr/bin/env python3
# lean_eval_cursor.py — minimal eval harness using Cursor Agent CLI

import argparse, csv, json, os, re, shutil, subprocess, sys, time
from typing import List
try:
    import tomllib  # Python 3.11+
except Exception:  # pragma: no cover
    tomllib = None  # Fallback handled at runtime
from datetime import datetime, timezone
from pathlib import Path

# ---------- tiny utils ----------

def ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def sh(cmd: str, cwd: str, env: dict, timeout: int):
    """Run a shell command; return (rc, out, err, dur)."""
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

def find_tasks(tasks_dir: str, only: set[str] | None) -> list[str]:
    out = []
    for root, _, files in os.walk(tasks_dir):
        if "task.json" in files:
            name = Path(root).name
            if (not only) or (name in only):
                out.append(root)
    return sorted(out)

# ---------- verification & grading ----------

DIST_TO_MODULE = {
    "opencv-python": "cv2",
    "Pillow": "PIL",
    "beautifulsoup4": "bs4",
    "PyYAML": "yaml",
    "scikit-image": "skimage",
    "pycryptodome": "Crypto",
}

def verify_deps(repo: str, deps: list[str], env: dict, timeout: int) -> dict:
    """Check deps inside .venv when present; otherwise use system python/pip."""
    venv_py = os.path.join(repo, ".venv", "bin", "python3")
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

def _load_pyproject_dependencies(pyproject_path: str) -> List[str]:
    """Return normalized dependency names from [project].dependencies.
    Normalization removes version/extras markers and lowercases the dist name.
    """
    if not tomllib:
        return []
    with open(pyproject_path, "rb") as f:
        data = tomllib.load(f)
    project = data.get("project") or {}
    deps = project.get("dependencies") or []
    normalized: List[str] = []
    for raw in deps:
        # Examples: "markdown>=3", "Pillow==10.3.0", "numpy", "requests[socks]" → base name
        token = str(raw).strip()
        base = re.split(r"[<>=!\[ ]", token, maxsplit=1)[0]
        normalized.append(base.lower())
    return normalized

def verify_manifest(repo: str, deps: list[str], tool_hint: str | None) -> dict:
    """Verify pyproject.toml lists required deps when tool_hint == 'uv'.
    Returns a dict with required/present/missing and a boolean 'ok'.
    Other tool_hints are treated as pass-through (ok=True).
    """
    if tool_hint != "uv":
        return {"ok": True, "reason": "non-uv task"}
    pyproject_path = os.path.join(repo, "pyproject.toml")
    if not os.path.isfile(pyproject_path):
        return {"ok": False, "required": deps, "present": [], "missing": deps, "used_pyproject": False}
    manifest_deps = _load_pyproject_dependencies(pyproject_path)
    present = []
    missing = []
    for dist in deps:
        name = dist.lower()
        if name in manifest_deps:
            present.append(dist)
        else:
            missing.append(dist)
    ok = len(missing) == 0
    return {
        "ok": ok,
        "required": deps,
        "present": present,
        "missing": missing,
        "used_pyproject": True,
    }

def choose_grade_cmd(repo: str, grade_cmd: str) -> str:
    venv_py = os.path.join(repo, ".venv", "bin", "python")
    if os.path.isfile(venv_py):
        # Replace first python token with venv python or prefix if running a .py file.
        if re.search(r"\bpython3?\b", grade_cmd):
            return re.sub(r"\bpython3?\b", venv_py, grade_cmd, count=1)
        if re.search(r"\b\S+\.py(\s|$)", grade_cmd):
            return f"{venv_py} {grade_cmd}"
    if grade_cmd.startswith("python "):
        return grade_cmd.replace("python ", "python3 ", 1)
    return grade_cmd

# ---------- Cursor Agent ----------

def check_cursor_cli():
    if shutil.which("cursor-agent") is None:
        raise SystemExit("cursor-agent not found in PATH. Install with:  curl https://cursor.com/install -fsS | bash")
    if not os.environ.get("CURSOR_API_KEY"):
        raise SystemExit("CURSOR_API_KEY is not set. Export it before running.")

def run_cursor_agent(repo: str, model: str, prompt: str, timeout: int) -> dict:
    """
    Runs Cursor Agent headlessly in repo cwd.
    Writes agent.out/agent.err/agent.cmd next to repo.
    """
    check_cursor_cli()
    env = os.environ.copy()
    # Non-interactive flags: -p prompt, --force to apply, text output for logging
    cmd = f"cursor-agent -p {q(prompt)} --model {q(model)} --force --output-format text"
    rc, out, err, dur = sh(cmd, cwd=repo, env=env, timeout=timeout)
    with open(os.path.join(Path(repo).parent, "agent.out"), "w", encoding="utf-8") as fo: fo.write(out)
    with open(os.path.join(Path(repo).parent, "agent.err"), "w", encoding="utf-8") as fe: fe.write(err)
    with open(os.path.join(Path(repo).parent, "agent.cmd"), "w", encoding="utf-8") as fc: fc.write(cmd)
    return {"rc": rc, "dur": dur}

def q(s: str) -> str:
    return "'" + s.replace("'", "'\"'\"'") + "'"

# ---------- main task runner ----------

def run_task(task_dir: str, run_root: str, per_cmd_timeout: int,
             agent_model: str | None) -> dict:
    spec = read_task(task_dir)
    repo_rel  = spec["repo"]
    grade_cmd = spec["grade"]
    desc      = spec.get("description", "").strip()
    deps      = spec.get("deps", [])
    tool_hint = spec.get("tool_hint")

    sandbox = os.path.join(run_root, Path(task_dir).name)
    repo_src = os.path.join(task_dir, repo_rel)
    repo     = os.path.join(sandbox, "repo")
    ensure_dir(sandbox)
    if os.path.exists(repo): shutil.rmtree(repo)
    shutil.copytree(repo_src, repo)

    # Init tiny git baseline (so diffs are meaningful if needed later)
    for c in [
        'git init', 'git config user.email "agent@local"',
        'git config user.name "Agent Runner"', 'git add -A',
        'git commit --no-verify -m baseline --allow-empty'
    ]:
        _ = sh(c, cwd=repo, env=os.environ.copy(), timeout=30)

    summary = {"task": Path(task_dir).name, "passed": False}

    # Run Cursor agent (edits occur in-place in `repo`)
    if not agent_model:
        raise SystemExit("--agent-model is required for cursor-agent")
    res = run_cursor_agent(repo, model=agent_model, prompt=desc, timeout=per_cmd_timeout)
    summary["agent"] = {"rc": res["rc"], "dur": round(res["dur"],2)}

    # Optional checks: environment and manifest
    env_ok = True
    if deps:
        dep = verify_deps(repo, deps, os.environ.copy(), per_cmd_timeout)
        summary["env_checks"] = dep
        env_ok = (dep["passed"] == dep["required"])
    manifest_ok = True
    manifest = verify_manifest(repo, deps, tool_hint)
    if manifest:
        summary["manifest_checks"] = manifest
        manifest_ok = bool(manifest.get("ok", True))

    # Grade
    gcmd = choose_grade_cmd(repo, grade_cmd)
    rc, out, err, dur = sh(gcmd, cwd=repo, env=os.environ.copy(), timeout=per_cmd_timeout)
    with open(os.path.join(sandbox, "grade.out"), "w", encoding="utf-8") as f:
        f.write("=== STDOUT ===\n"); f.write(out); f.write("\n=== STDERR ===\n"); f.write(err)
    summary["grade_rc"] = rc
    summary["passed"]   = (rc == 0) and env_ok and manifest_ok
    with open(os.path.join(sandbox, "result.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary

def main():
    ap = argparse.ArgumentParser("lean Cursor eval")
    ap.add_argument("--tasks-dir", default="tasks")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--per-cmd-timeout", type=int, default=600)
    ap.add_argument("--agent-model", required=True, help="Cursor model id (e.g., cursor-small, cursor-large, etc.)")
    args = ap.parse_args()

    # Optional: load .env
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
    results = []
    for t in tasks:
        print(f"Running task: {Path(t).name}")
        res = run_task(t, run_root, args.per_cmd_timeout, agent_model=args.agent_model)
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
