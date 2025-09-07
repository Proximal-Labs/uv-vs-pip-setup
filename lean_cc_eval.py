#!/usr/bin/env python3
import os, sys, json, csv, shutil, argparse, subprocess, time, re
from pathlib import Path

def ts(): return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
def ensure_dir(p): os.makedirs(p, exist_ok=True)
def sh(cmd, cwd, env, timeout):
    start = time.time()
    p = subprocess.run(["bash","-lc",cmd], cwd=cwd, env=env,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       text=True, timeout=timeout)
    return p.returncode, p.stdout, p.stderr, time.time()-start

def read_task(task_dir):
    with open(os.path.join(task_dir, "task.json"), "r", encoding="utf-8") as f:
        return json.load(f)

def choose_grade_cmd(repo, grade_cmd):
    venv_py = os.path.join(repo, ".venv", "bin", "python")
    if os.path.isfile(venv_py):
        if re.search(r"\bpython3?\b", grade_cmd):  # swap first token
            return re.sub(r"\bpython3?\b", venv_py, grade_cmd, count=1)
        if re.search(r"\b\S+\.py(\s|$)", grade_cmd):
            return f"{venv_py} {grade_cmd}"
    return grade_cmd.replace("python ", "python3 ", 1) if grade_cmd.startswith("python ") else grade_cmd

def main():
    ap = argparse.ArgumentParser("lean Claude Code eval (via claude-code-router)")
    ap.add_argument("--tasks-dir", default="tasks")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--per-cmd-timeout", type=int, default=600)
    ap.add_argument("--router-url", default="http://127.0.0.1:3456")
    ap.add_argument("--model-route", default="", help="Optional '/model provider,model' directive (e.g., 'openrouter,meta-llama/llama-3.1-70b-instruct')")
    ap.add_argument("--allowed-tools", nargs="*", default=["Read","Edit","Write","Grep","Glob","Bash"])
    ap.add_argument("--max-turns", type=int, default=5)
    ap.add_argument("--permission-mode", default="acceptEdits")
    args = ap.parse_args()

    # Load .env if present
    dotenv = os.path.join(os.getcwd(), ".env")
    if os.path.exists(dotenv):
        with open(dotenv, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#") or "=" not in s: continue
                k, v = s.split("=", 1)
                if k and v and k not in os.environ: os.environ[k] = v

    # Discover tasks
    tasks = []
    for root, _, files in os.walk(args.tasks_dir):
        if "task.json" in files and (not args.only or Path(root).name in set(args.only)):
            tasks.append(root)
    if not tasks:
        print("No tasks found.", file=sys.stderr); sys.exit(1)

    run_root = os.path.join(os.getcwd(), "runs", ts()); ensure_dir(run_root)
    results = []
    for t in tasks:
        name = Path(t).name
        print(f"Running task: {name}")
        spec = read_task(t)
        repo_src, grade_cmd = os.path.join(t, spec["repo"]), spec["grade"]

        sandbox = os.path.join(run_root, name); ensure_dir(sandbox)
        repo = os.path.join(sandbox, "repo")
        if os.path.exists(repo): shutil.rmtree(repo)
        shutil.copytree(repo_src, repo)

        # Init tiny git baseline (useful for diffs later)
        for c in ['git init','git config user.email "agent@local"','git config user.name "Agent Runner"','git add -A','git commit --no-verify -m baseline --allow-empty']:
            _ = sh(c, cwd=repo, env=os.environ.copy(), timeout=30)

        # --- Run Claude Code non-interactively in the sandbox ---
        env = os.environ.copy()
        env["ANTHROPIC_BASE_URL"] = args.router-url if hasattr(args, "router-url") else args.router_url  # router
        env.setdefault("ANTHROPIC_API_KEY", env.get("CLAUDE_CODE_ROUTER_KEY","ccr"))  # router may check a key; any string if none configured

        # permissions & execution behavior
        allowed = " ".join(f'"{x}"' for x in args.allowed_tools)
        model_switch = f"/model {args.model_route}\n" if args.model_route else ""
        prompt = f"{model_switch}{spec.get('description','').strip()}"

        cmd = (
            f"claude -p --max-turns {args.max_turns} "
            f"--permission-mode {args.permission_mode} "
            f"--allowedTools {allowed} "
            f"--output-format text "
            f"{shlex_quote(prompt)}"
        )
        rc, out, err, dur = sh(cmd, cwd=repo, env=env, timeout=args.per_cmd_timeout)
        with open(os.path.join(sandbox, "agent.out"), "w", encoding="utf-8") as fo: fo.write(out)
        with open(os.path.join(sandbox, "agent.err"), "w", encoding="utf-8") as fe: fe.write(err)
        with open(os.path.join(sandbox, "agent.cmd"), "w", encoding="utf-8") as fc: fc.write(cmd)

        # Grade
        gcmd = choose_grade_cmd(repo, grade_cmd)
        grc, gout, gerr, _ = sh(gcmd, cwd=repo, env=os.environ.copy(), timeout=args.per_cmd_timeout)
        with open(os.path.join(sandbox, "grade.out"), "w", encoding="utf-8") as f:
            f.write("=== STDOUT ===\n"); f.write(gout); f.write("\n=== STDERR ===\n"); f.write(gerr)

        res = {"task": name, "passed": (grc == 0), "grade_rc": grc, "agent_rc": rc, "dur": round(dur,2)}
            # You can add dep/manifest checks like your other scripts if you want parity
        with open(os.path.join(sandbox, "result.json"), "w", encoding="utf-8") as f: json.dump(res, f, indent=2)
        print(f"  -> {'PASS' if res['passed'] else 'FAIL'}")
        results.append(res)

    with open(os.path.join(run_root, "results.json"), "w", encoding="utf-8") as f: json.dump(results, f, indent=2)
    with open(os.path.join(run_root, "results.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["task","passed","grade_rc"])
        for r in results: w.writerow([r["task"], int(r["passed"]), r.get("grade_rc")])

if __name__ == "__main__":
    main()
