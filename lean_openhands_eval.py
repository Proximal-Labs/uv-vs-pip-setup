#!/usr/bin/env python3
# lean_openhands_eval.py — OpenHands headless harness (DOCKER runtime)
#
# What this script does:
# - Copies each task repo into runs/<ts>/<task>/repo on HOST
# - Starts OpenHands in DOCKER runtime, mounting HOST repo at /workspace
# - Lets the agent edit /workspace (the mounted repo) and install deps INSIDE container
# - Verifies deps and runs the grade command INSIDE the same runtime image
#
# Works with OpenRouter (e.g., openrouter/anthropic/claude-sonnet-4)
# or direct providers (export ANTHROPIC_API_KEY and use anthropic/...).
#
# Requirements:
# - Docker running and reachable from your user
# - `uvx` available (`pipx install uv` or `brew install uv`)

import argparse, csv, json, os, re, shutil, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path
import tomllib
from typing import Any, Dict, Tuple

# ---------------- NEW: policy toggles ----------------
require_command_hygiene = True
require_manifest_ok = True
# -----------------------------------------------------

# ---------- small utils ----------

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
        timeout=None if (timeout is None or timeout <= 0) else timeout,
    )
    return proc.returncode, proc.stdout, proc.stderr, time.time() - t0

def q(s: str) -> str:
    return "'" + s.replace("'", "'\"'\"'") + "'"

def read_task(task_dir: str) -> dict:
    with open(os.path.join(task_dir, "task.json"), "r", encoding="utf-8") as f:
        return json.load(f)

def find_tasks(tasks_dir: str, only: set[str] | None) -> list[str]:
    out = []
    for root, _, files in os.walk(tasks_dir):
        if "task.json" in files:
            name = Path(root).name
            if (not only) or (name in only):
                out.append(root)
    return sorted(out)

# ---------- verification & grading helpers ----------

DIST_TO_MODULE = {
    "opencv-python": "cv2",
    "Pillow": "PIL",
    "beautifulsoup4": "bs4",
    "PyYAML": "yaml",
    "scikit-image": "skimage",
    "pycryptodome": "Crypto",
}

def _prefer_python(cmd: str, py_path: str) -> str:
    """Replace first python token or prefix a .py with given python path."""
    if re.search(r"\bpython3?\b", cmd):
        return re.sub(r"\bpython3?\b", py_path, cmd, count=1)
    if re.search(r"\b\S+\.py(\s|$)", cmd):
        return f"{py_path} {cmd}"
    return cmd

def choose_grade_cmd(repo: str, grade_cmd: str) -> str:
    """If the repo has .venv/bin/python, prefer it; else leave cmd as-is (or python->python3)."""
    venv_py = os.path.join(repo, ".venv", "bin", "python")
    if os.path.isfile(venv_py):
        return _prefer_python(grade_cmd, venv_py)
    if grade_cmd.startswith("python "):
        return grade_cmd.replace("python ", "python3 ", 1)
    return grade_cmd

# ---------- Docker helpers ----------

def derive_runtime_image(openhands_version: str | None) -> str:
    """
    OpenHands runtime images are typically tagged like:
      docker.all-hands.dev/all-hands-ai/runtime:0.54-nikolaik
    We'll default to 0.54-nikolaik unless SANDBOX_RUNTIME_CONTAINER_IMAGE is provided.
    If an openhands_version like '0.54.0' is given, we map -> '0.54-nikolaik'.
    """
    default_tag = "0.54-nikolaik"
    if openhands_version:
        parts = openhands_version.split(".")
        if len(parts) >= 2:
            default_tag = f"{parts[0]}.{parts[1]}-nikolaik"
    return f"docker.all-hands.dev/all-hands-ai/runtime:{default_tag}"

def sh_in_runtime(repo_abs: str, cmd: str, image: str, timeout: int, env: dict):
    """
    Execute a shell command inside the OpenHands runtime image with the host repo
    mounted to /workspace and workdir set to /workspace.
    """
    runtime_image = env.get("SANDBOX_RUNTIME_CONTAINER_IMAGE", image)
    docker_cmd = (
        f"docker run --rm "
        f"-v {q(repo_abs)}:/workspace "
        f"-w /workspace "
        f"{runtime_image} "
        f"bash -lc {q(cmd)}"
    )
    return sh(docker_cmd, cwd=repo_abs, env=env, timeout=timeout)

# ---------- OpenHands (DOCKER runtime) ----------

def _persist_agent(repo, cmd, out, err):
    sandbox = os.path.join(Path(repo).parent)
    with open(os.path.join(sandbox, "agent.out"), "w", encoding="utf-8") as fo: fo.write(out or "")
    with open(os.path.join(sandbox, "agent.err"), "w", encoding="utf-8") as fe: fe.write(err or "")
    with open(os.path.join(sandbox, "agent.cmd"), "w", encoding="utf-8") as fc: fc.write(cmd or "")

def _snap(rc, out, err, dur, cmd, env, runtime_image: str):
    def short(v):
        return (v[:10] + "…") if isinstance(v, str) and len(v) > 14 else (v or "")
    return {
        "rc": rc, "out": out, "err": err, "dur": dur, "cmd": cmd,
        "env": {
            "LLM_MODEL": env.get("LLM_MODEL"),
            "OPENAI_API_BASE": short(env.get("OPENAI_API_BASE", "")),
            "OPENAI_BASE_URL": short(env.get("OPENAI_BASE_URL", "")),
            "HAS_OPENAI_API_KEY": bool(env.get("OPENAI_API_KEY")),
            "HAS_LLM_API_KEY": bool(env.get("LLM_API_KEY")),
            "RUNTIME": env.get("RUNTIME", "docker"),
            "SANDBOX_VOLUMES": env.get("SANDBOX_VOLUMES", ""),
            "RUNTIME_IMAGE": runtime_image,
        }
    }

def _load_dotenv_if_present():
    dotenv = os.path.join(os.getcwd(), ".env")
    if not os.path.exists(dotenv):
        return
    for line in Path(dotenv).read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, v = s.split("=", 1)
        if k and v and k not in os.environ:
            os.environ[k] = v

def _write_config(repo: str, max_iter: int = 1000, repo_abs: str | None = None, temperature: float | None = None):
    """
    OpenHands 0.54 Docker config:
      - Mount keys live at TOP LEVEL (not under [core]).
      - No [runtime] section; provider env is set on the host.
    """
    cfg_path = os.path.join(repo, "config.toml")
    lines = []
    lines += [
        "[core]",
        'runtime = "docker"',
        f"max_iterations = {max_iter}",
        "",
    ]
    # TOP-LEVEL mount keys (these are what OH 0.54 reads)
    if repo_abs:
        lines += [
            f'workspace_base = "{repo_abs}"',
            'workspace_mount_path = "/workspace"',
            "",
        ]
    lines += [
        "[security]",
        "confirmation_mode = false",
        "",
    ]
    if temperature is not None:
        t = max(0.0, min(2.0, float(temperature)))
        lines += [
            "[llm]",
            f"temperature = {t}",
            "",
        ]
    Path(cfg_path).write_text("\n".join(lines), encoding="utf-8")
    return cfg_path

def run_openhands(repo: str, model: str, prompt: str,
                  version: str | None, timeout: int,
                  prewarm_seconds: int = 60, temperature: float | None = None) -> dict:
    if shutil.which("uvx") is None:
        raise SystemExit("uvx not found. Install uv: https://docs.astral.sh/uv/")

    env = os.environ.copy()
    env["RUNTIME"] = "docker"
    env.setdefault("LOG_LEVEL", "info")
    env.setdefault("SECURITY_CONFIRMATION_MODE", "false")
    env.setdefault("TERM", "dumb")

    # Do NOT alter the model string; support both "openrouter/..." and "vendor/model"
    env["LLM_MODEL"] = model

    # Clamp to a sane range commonly accepted by providers (0.0–2.0)
    if temperature is not None:
        t = max(0.0, min(2.0, float(temperature)))
        env["LLM_TEMPERATURE"] = str(t)

    # Optional: route via OpenRouter (map OR key to OpenAI-compatible vars)
    if env.get("OPENROUTER_API_KEY"):
        for k in ("ANTHROPIC_API_KEY", "AZURE_OPENAI_API_KEY", "GOOGLE_API_KEY", "COHERE_API_KEY"):
            env.pop(k, None)
        env.setdefault("OPENAI_API_KEY",  env["OPENROUTER_API_KEY"])
        env.setdefault("LLM_API_KEY",     env["OPENROUTER_API_KEY"])
        env.setdefault("OPENAI_API_BASE", "https://openrouter.ai/api/v1")
        env.setdefault("OPENAI_BASE_URL", env["OPENAI_API_BASE"])

    repo_abs = os.path.abspath(repo)
    runtime_image_default = derive_runtime_image(version)

    # (Visibility only; OH 0.54 uses the top-level mount keys from config.toml)
    env["SANDBOX_VOLUMES"] = f"{repo_abs}:/workspace:rw"
    # Also set these envs (harmless + explicit)
    env["OPENHANDS_WORKSPACE_BASE"] = repo_abs
    env["OPENHANDS_WORKSPACE_MOUNT_PATH"] = "/workspace"

    # Minimal config with TOP-LEVEL mount keys
    cfg_path = _write_config(repo, max_iter=1000, repo_abs=repo_abs, temperature=temperature)
    env["OPENHANDS_CONFIG"] = cfg_path

    from_spec = f"openhands-ai=={version}" if version else "openhands-ai"

    # Pre-warm the CLI
    _ = sh(f"uvx --python 3.12 --from {from_spec} python -m openhands.core.main --help",
           cwd=repo, env=env, timeout=prewarm_seconds)

    task = q(prompt.strip())
    cmd = (
        f"uvx --python 3.12 --from {from_spec} "
        f"python -m openhands.core.main "
        f"--config-file {q(cfg_path)} "
        f"-d /workspace "
        f"-t {task} -i 1000"
    )

    rc, out, err, dur = sh(cmd, cwd=repo, env=env, timeout=timeout)
    _persist_agent(repo, cmd, out, err)
    return _snap(rc, out, err, dur, cmd, env, os.environ.get("SANDBOX_RUNTIME_CONTAINER_IMAGE", runtime_image_default))

def _normalize_name(token: str) -> str:
    return re.split(r"[<>=!\[ ;@, ]", token, maxsplit=1)[0].strip().lower()

def _read_requirements_names(req_path: str) -> list[str]:
    if not os.path.isfile(req_path):
        return []
    names = set()
    for line in Path(req_path).read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or s.startswith("-r"):
            continue
        if s.startswith("-e "):
            continue
        names.add(_normalize_name(s))
    return sorted(names)

def _safe_read(path: str, limit: int = 2000) -> Tuple[str, bool]:
    try:
        with open(path, "rb") as f:
            raw = f.read(limit)
        return raw.decode("utf-8", "replace"), True
    except Exception:
        return "", False

def _load_pyproject_deps_safe(pyproject_path: str) -> Dict[str, Any]:
    """Load [project].dependencies from pyproject.toml, but never raise."""
    if not tomllib or not os.path.isfile(pyproject_path):
        return {"ok": False, "used_pyproject": False, "deps": []}

    head, _ = _safe_read(pyproject_path, limit=4096)
    try:
        data = tomllib.loads(head)  # parse only the head; enough for [project]
    except Exception as e:
        # Malformed TOML – report, but don't crash the run
        return {
            "ok": False,
            "used_pyproject": True,
            "deps": [],
            "parse_error": True,
            "error": str(e),
            "preview": head.splitlines()[:5],
        }

    proj = (data.get("project") or {})
    deps = []
    for raw in proj.get("dependencies") or []:
        token = str(raw).strip()
        base = re.split(r"[<>=!\[ (]", token, maxsplit=1)[0].lower()
        deps.append(base)
    return {"ok": True, "used_pyproject": True, "deps": deps}

def verify_manifest_uv_local(repo: str, required: list[str]) -> dict:
    """Compare required deps to pyproject; never raise."""
    pp = os.path.join(repo, "pyproject.toml")
    if not required:
        # Nothing to enforce; don't parse TOML at all.
        return {"ok": True, "required": [], "present": [], "missing": [],
                "used_pyproject": os.path.isfile(pp), "skipped": "no_required_deps"}

    meta = _load_pyproject_deps_safe(pp)
    if not meta.get("used_pyproject"):
        return {"ok": False, "required": required, "present": [], "missing": required,
                "used_pyproject": False}

    if meta.get("parse_error"):
        return {"ok": False, "required": required, "present": [], "missing": required,
                "used_pyproject": True, "parse_error": True, "error": meta.get("error"),
                "preview": meta.get("preview")}

    manifest = set(d.lower() for d in meta.get("deps", []))
    present  = [d for d in required if d.lower() in manifest]
    missing  = [d for d in required if d.lower() not in manifest]
    return {"ok": len(missing) == 0, "required": required, "present": present,
            "missing": missing, "used_pyproject": True}

def verify_manifest_pip_local(repo: str, required: list[str]) -> dict:
    rp = os.path.join(repo, "requirements.txt")
    reqs = set(_read_requirements_names(rp))
    if not reqs:
        return {"ok": False, "which": "requirements.txt", "required": required, "present": [], "missing": required}
    present = [d for d in required if _normalize_name(d) in reqs]
    missing = [d for d in required if _normalize_name(d) not in reqs]
    return {"ok": len(missing) == 0, "which": "requirements.txt", "required": required, "present": present, "missing": missing}

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
HDR_RE  = re.compile(r"^\d{2}:\d{2}:\d{2}\s+-\s+\w+")  # e.g., "14:10:33 - ACTION"

HEREDOC_START_RE = re.compile(r"<<-?\s*(['\"]?)EOF\1(?:\s*>.*)?\s*$")

def _clean(s: str) -> str:
    s = ANSI_RE.sub("", s)
    s = s.strip()
    # collapse multiple spaces, kill trailing semicolons
    s = re.sub(r"\s+", " ", s).rstrip(";")
    return s

def extract_commands_from_agent_err(err_text: str) -> list[str]:
    cmds, in_block, in_heredoc = [], False, False
    pending: list[str] = []

    for raw in err_text.splitlines():
        line = ANSI_RE.sub("", raw.rstrip("\n"))

        if line.startswith("COMMAND:"):
            # flush any previous (shouldn't happen) and start fresh
            if pending:
                cmds.extend(pending)
                pending = []
            in_block, in_heredoc = True, False
            inline = _clean(line[len("COMMAND:"):])
            if inline:
                pending.append(inline)
            continue

        if not in_block:
            continue

        # end of block when we hit a header/timestamp line
        if HDR_RE.match(line) or line.startswith("[Agent Controller") or line.startswith("--END"):
            in_block = False
            in_heredoc = False
            if pending:
                # de-dupe trivial consecutive repeats
                for c in pending:
                    if not cmds or cmds[-1] != c:
                        cmds.append(c)
                pending = []
            continue

        if in_heredoc:
            if line.strip() == "EOF":
                in_heredoc = False
            continue

        # detect heredoc start; keep the start line but skip its body
        if HEREDOC_START_RE.search(line):
            s = _clean(line)
            if s:
                pending.append(s)
            in_heredoc = True
            continue

        s = _clean(line)
        if s:
            # join line continuations ending with backslash
            if s.endswith("\\"):
                pending.append(s[:-1].rstrip())
            else:
                pending.append(s)

    # flush trailing block (if log ended mid-block)
    if pending:
        for c in pending:
            if not cmds or cmds[-1] != c:
                cmds.append(c)

    return cmds


def score_command_sequence(cmds: list[str], tool_hint: str, repo: str, required_deps: list[str]) -> dict:
    """
    Score hygiene and ordering for both uv and pip flows.
    Returns:
      {
        'ok': bool, 'score': int, 'max': int,
        'hits': {...}, 'violations': [...],
        'cmds': [...],
        'manifest': {...}  # manifest_ok info
      }
    """
    hits = {
        "venv_created": False,
        "installed_in_venv": False,
        # "ran_with_venv": False, (not needed since we grade inside the venv afterwards anyways)
        "no_global_pip": False,
        "used_tool_commands": False,
        # "used_manifest_flow": False, (unnecessary for now)
    }
    viols: list[str] = []
    score = 0
    maxscore = 4  # venv + install-in-venv + no-global-pip + tool-specific

    def any_re(patterns: list[str]) -> bool:
        for c in cmds:
            for p in patterns:
                if re.search(p, c):
                    return True
        return False

    def first_index(patterns: list[str]) -> int | None:
        idx = []
        for i, c in enumerate(cmds):
            if any(re.search(p, c) for p in patterns):
                idx.append(i)
        return min(idx) if idx else None

    # convenience flags used in multiple checks
    had_activation_anywhere = any("source .venv/bin/activate" in c for c in cmds)

    # 1) VENV created
    venv_create = [
        r"\buv\s+venv\b",
        r"\bpython3?\s+-m\s+venv\s+\.venv\b",
        r"\buv\s+sync\b",
        r"\bvirtualenv\s+\.venv\b",
        r"\bpython3?\s+-m\s+virtualenv\s+\.venv\b",
    ]

    if any_re(venv_create):
        hits["venv_created"] = True; score += 1
    else:
        venv_py_path = os.path.join(repo, ".venv", "bin", "python")
        if os.path.exists(venv_py_path):
            hits["venv_created"] = True
            score += 1
            # remove the specific violation if we added it
            viols = [v for v in viols if not v.startswith("No venv creation command")]

        else:
            viols.append("No venv creation command (uv venv / python -m venv .venv)")

    # 2) Install inside venv - IMPROVED: More comprehensive patterns
    # broader activation detection (used for "had_activation_anywhere")
    had_activation_anywhere = any(
        re.search(r"\b(?:source|\.)\s+\.venv/bin/activate\b", c) for c in cmds
    )

    install_in_venv_patterns = [
        r"\buv\s+pip\s+install\b",
        r"\buv\s+pip\s+sync\b",
        r"\bpython3?\s+-m\s+uv\s+pip\s+install\b",
        r"\buv\s+sync\b",
        r"\buv\s+add\b",
        r"\.venv/bin/python3?\s+-m\s+pip\s+install\b",
        r"\.venv/bin/pip\s+install\b",
        r"\bpython3?\s+-m\s+pip\s+install\b.*(?:--target|-t)\s+\.venv\b",
        r"(?:source|\.)\s+\.venv/bin/activate\s*&&\s*(?:uv\s+pip\s+install|pip\s+install|python3?\s+-m\s+pip\s+install)\b",
    ]
    installed_in_venv = any_re(install_in_venv_patterns)

    # also accept: activation somewhere + later pip install on another line
    pip_install_any = any(re.search(r"(^|\s)pip\s+install\b", c) for c in cmds) or \
                      any(re.search(r"\bpython3?\s+-m\s+pip\s+install\b", c) for c in cmds)
    if not installed_in_venv and had_activation_anywhere and pip_install_any:
        installed_in_venv = True

    if installed_in_venv:
        hits["installed_in_venv"] = True; score += 1
    else:
        viols.append("No evidence of installing inside the venv")

    # 3) Run/grade with venv python - not needed (since we grade inside the venv afterwards anyways)
    # if any_re([r"\.venv/bin/python3?\b"]):
    #     hits["ran_with_venv"] = True; score += 1
    # else:
    #     viols.append("Program/grade not run with .venv/bin/python")

    # 4) No global pip (allow venv + activated cases)
    def is_illegal_global_pip(c: str) -> bool:
        s = c.strip()
        if not re.search(r"(^|\s)pip\s+install\b", s) and not re.search(r"\bpython3?\s+-m\s+pip\s+install\b", s):
            return False
        # Allowed: explicit venv pip
        if re.search(r"\.venv/bin/pip\s+install\b", s) or re.search(r"\.venv/bin/python3?\s+-m\s+pip\s+install\b", s):
            return False
        # Allowed: uv pip
        if re.search(r"\buv\s+pip\s+install\b", s):
            return False
        # Allowed: same-line activation
        if re.search(r"source\s+\.venv/bin/activate\s*&&\s*pip\s+install\b", s):
            return False
        # NEW: previously-activated venv in session
        if had_activation_anywhere and re.search(r"(^|\s)pip\s+install\b", s):
            return False
        return True

    illegal_global = any(is_illegal_global_pip(c) for c in cmds) or \
                     any(re.search(r"\b--user\b|\bsudo\s+pip\b", c) for c in cmds)

    if not illegal_global:
        hits["no_global_pip"] = True
        score += 1
    else:
        viols.append("Detected global pip usage (pip install/--user/sudo pip)")

    # 5) Tool-specific + manifest - FIXED: Better UV command detection
    tool = (tool_hint or "").strip().lower()
    base_max = 3  # venv + installed_in_venv + no_global_pip
    maxscore = base_max + (1 if tool in ("uv", "pip") else 0)
    if tool == "uv":
        # FIXED: Use the same robust patterns as venv_create
        if any_re([r"\buv\s+venv\b", r"\buv\s+(add|sync|pip)\b"]):
            hits["used_tool_commands"] = True
            score += 1
        # order: venv before installs
        vi = first_index([
            r"\buv\s+pip\s+install\b", r"\buv\s+sync\b", r"\buv\s+add\b",
            r"\.venv/bin/python3?\s+-m\s+pip\s+install\b", r"\.venv/bin/pip\s+install\b",
            r"(^|\s)pip\s+install\b"
        ])
        ci = first_index(venv_create)
        if vi is not None and ci is not None and vi < ci:
            viols.append("Installed deps before venv creation (order)")
        manifest = verify_manifest_uv_local(repo, required_deps)
    else:  # pip
        # count legitimate pip usage inside the venv
        pip_tool_patterns = [
            r"\.venv/bin/python3?\s+-m\s+pip\b",
            r"\.venv/bin/pip\b",
            r"source\s+\.venv/bin/activate\s*&&\s*pip\b",
        ]
        if any_re(pip_tool_patterns):
            hits["used_tool_commands"] = True
            score += 1

        # order: installs must come after venv creation
        vi = first_index([
            r"\.venv/bin/python3?\s+-m\s+pip\s+install\b",
            r"\.venv/bin/pip\s+install\b",
            r"(^|\s)pip\s+install\b",
        ])
        ci = first_index(venv_create)
        if vi is not None and ci is not None and vi < ci:
            viols.append("Installed deps before venv creation (order)")

        # optional hygiene: penalize uv on a pip task
        if any_re([r"\buv\s+"]):
            viols.append("Used uv on a pip task")

        manifest = verify_manifest_pip_local(repo, required_deps)

    ok = (score >= maxscore) and not viols
    if tool == "uv":
        hits["used_uv_commands"] = hits["used_tool_commands"]
    elif tool == "pip":
        hits["used_pip_commands"] = hits["used_tool_commands"]
    return {
        "ok": ok,
        "score": score,
        "max": maxscore,
        "hits": hits,
        "violations": viols,
        "cmds": cmds,
        "manifest": manifest,
    }


# ---------- main task runner ----------

def verify_deps_in_container(repo_abs: str, deps: list[str], image: str, timeout: int, env: dict) -> dict:
    """
    Verify deps INSIDE the runtime image that the agent used.
    Strategy:
      - If .venv/bin/python exists, use it; else fall back to python3
      - A pass is either 'pip show dist' OR 'python -c "import module"'
    """
    if not deps:
        return {"required": 0, "passed": 0, "used_venv": False}

    used_venv = False
    passed = 0
    # detect venv inside container
    rc, *_ = sh_in_runtime(repo_abs, "test -x .venv/bin/python", image, timeout, env)
    venv_py = ".venv/bin/python" if rc == 0 else "python3"
    used_venv = (rc == 0)

    for dist in deps:
        mod = DIST_TO_MODULE.get(dist, dist)
        rc1, *_ = sh_in_runtime(repo_abs, f"{venv_py} -m pip show {dist}", image, timeout, env)
        rc2, *_ = sh_in_runtime(repo_abs, f"{venv_py} -c 'import {mod}; import sys; sys.exit(0)'", image, timeout, env)
        if rc1 == 0 or rc2 == 0:
            passed += 1

    return {"required": len(deps), "passed": passed, "used_venv": used_venv}

def grade_in_container(repo_abs: str, grade_cmd: str, image: str, timeout: int, env: dict):
    """
    Run the task's grade command INSIDE the runtime image.
    If repo created .venv, prefer it; otherwise use python3.
    """
    rc_v, *_ = sh_in_runtime(repo_abs, "test -x .venv/bin/python", image, timeout, env)
    if rc_v == 0:
        grade_cmd = _prefer_python(grade_cmd, ".venv/bin/python")
    elif grade_cmd.startswith("python "):
        grade_cmd = grade_cmd.replace("python ", "python3 ", 1)

    return sh_in_runtime(repo_abs, grade_cmd, image, timeout, env)

def run_task(task_dir: str, run_root: str, per_cmd_timeout: int,
             agent: str, agent_model: str | None, openhands_ver: str | None, temperature: float | None = None) -> dict:
    spec = read_task(task_dir)
    repo_rel   = spec["repo"]
    grade_cmd  = spec["grade"]
    desc       = spec.get("description", "").strip()
    deps       = spec.get("deps", [])
    tool_hint  = (spec.get("tool_hint") or "uv").strip().lower()

    # sandbox working copy
    sandbox = os.path.join(run_root, Path(task_dir).name)
    repo_src = os.path.join(task_dir, repo_rel)
    repo     = os.path.join(sandbox, "repo")
    ensure_dir(sandbox)
    if os.path.exists(repo): shutil.rmtree(repo)
    shutil.copytree(repo_src, repo)

    # init a tiny git baseline (optional)
    for c in [
        'git init', 'git config user.email "agent@local"',
        'git config user.name "Agent Runner"', 'git add -A',
        'git commit --no-verify -m baseline --allow-empty'
    ]:
        _ = sh(c, cwd=repo, env=os.environ.copy(), timeout=30)

    summary = {"task": Path(task_dir).name, "passed": False}

    # Run the OpenHands agent in Docker runtime
    runtime_image_default = derive_runtime_image(openhands_ver)
    if agent == "openhands":
        if not agent_model:
            raise SystemExit("--agent-model is required with --agent openhands")
        res = run_openhands(repo, model=agent_model, prompt=desc, version=openhands_ver, timeout=per_cmd_timeout, temperature=temperature)
        summary["agent"] = {"rc": res["rc"], "dur": round(res["dur"],2)}
        err_head = (res["err"] or "")[:200]
        if err_head:
            print(f"  [openhands err head] {err_head}")
        # Determine which image to use for checks (same as runtime)
        runtime_image = os.environ.get("SANDBOX_RUNTIME_CONTAINER_IMAGE", runtime_image_default)
        # NEW: parse commands from agent.err and score hygiene
        try:
            agent_err_text = Path(os.path.join(sandbox, "agent.err")).read_text(encoding="utf-8")
        except Exception:
            agent_err_text = res.get("err", "") or ""
        cmds = extract_commands_from_agent_err(agent_err_text)
        cmd_checks = score_command_sequence(cmds, tool_hint, repo, deps)
        summary["command_checks"] = cmd_checks
        summary["manifest_ok"] = bool(cmd_checks.get("manifest", {}).get("ok"))
        # (optional) write a short commands file for quick grepping
        try:
            Path(os.path.join(sandbox, "agent.tools")).write_text("\n".join(cmds), encoding="utf-8")
        except Exception:
            pass
    elif agent and agent != "none":
        raise SystemExit(f"Unsupported agent: {agent}")
    else:
        runtime_image = os.environ.get("SANDBOX_RUNTIME_CONTAINER_IMAGE", runtime_image_default)
        # If no agent, still compute manifest_ok from repo state
        if tool_hint == "uv":
            summary["manifest_ok"] = verify_manifest_uv_local(repo, deps).get("ok", False)
        else:
            summary["manifest_ok"] = verify_manifest_pip_local(repo, deps).get("ok", False)
        summary["command_checks"] = {"ok": True, "violations": [], "cmds": []}

    repo_abs = os.path.abspath(repo)

    # Env verification INSIDE the same runtime image
    env_ok = True
    if deps:
        dep = verify_deps_in_container(repo_abs, deps, runtime_image, per_cmd_timeout, os.environ.copy())
        summary["env_checks"] = dep
        env_ok = (dep["passed"] == dep["required"])

    # Grade INSIDE the same runtime image
    gcmd = choose_grade_cmd(repo, grade_cmd)
    rc, out, err, dur = grade_in_container(repo_abs, gcmd, runtime_image, per_cmd_timeout, os.environ.copy())

    with open(os.path.join(sandbox, "grade.out"), "w", encoding="utf-8") as f:
        f.write("=== STDOUT ===\n"); f.write(out); f.write("\n=== STDERR ===\n"); f.write(err)
    summary["grade_rc"] = rc

    # ORIGINAL pass policy + NEW gates (toggled)
    passed = (rc == 0) and env_ok
    if require_manifest_ok:
        passed = passed and bool(summary.get("manifest_ok", False))
    if require_command_hygiene:
        passed = passed and bool(summary.get("command_checks", {}).get("ok", True))
    summary["passed"] = passed

    with open(os.path.join(sandbox, "result.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary

def main():
    ap = argparse.ArgumentParser("lean OpenHands eval (headless, docker runtime)")
    ap.add_argument("--tasks-dir", default="tasks")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--per-cmd-timeout", type=int, default=600)
    ap.add_argument("--agent", choices=["none","openhands"], default="openhands")
    ap.add_argument("--agent-model", help="e.g. openrouter/anthropic/claude-sonnet-4 or qwen/qwen3-32b")
    ap.add_argument("--openhands-version", help="e.g. 0.54.0 (optional; used to pick a sensible runtime tag)")
    ap.add_argument("--runs-dir", default="runs", help="Base directory for runs (default = ./runs)")
    ap.add_argument("--temperature", type=float, help="LLM temperature (e.g., 0.0-2.0)")
    ap.add_argument("--run-subdir", help="Leaf subdirectory name under --runs-dir; defaults to a UTC timestamp")
    args = ap.parse_args()

    _load_dotenv_if_present()

    tasks = find_tasks(os.path.abspath(args.tasks_dir), set(args.only) if args.only else None)
    if not tasks:
        print("No tasks found.", file=sys.stderr); sys.exit(1)


    print(f"runs_dir: {args.runs_dir}")
    start_time = time.time()

    leaf = args.run_subdir or ts()
    run_root = os.path.join(os.getcwd(), args.runs_dir, leaf)
    ensure_dir(run_root)
    results = []
    for t in tasks:
        task_start_time = time.time()
        print(f"Running task: {Path(t).name}")
        res = run_task(
            t, run_root, args.per_cmd_timeout,
            agent=args.agent, agent_model=args.agent_model,
            openhands_ver=args.openhands_version,
            temperature=args.temperature,
        )
        task_end_time = time.time()
        task_duration_seconds = task_end_time - task_start_time
        task_duration_minutes = task_duration_seconds / 60.0
        res["task_duration"] = round(task_duration_minutes, 2)
        print(f"  -> {'PASS' if res['passed'] else 'FAIL'}")
        results.append(res)

    passed_tasks = [r["task"] for r in results if r.get("passed")]
    failed_tasks = [r["task"] for r in results if not r.get("passed")]
    score = sum(1 for r in results if r.get("passed"))
    end_time = time.time()
    duration_seconds = end_time - start_time
    duration_minutes = duration_seconds / 60.0

    sweep = {
        "duration": round(duration_minutes, 2),
        "score": score,
        "total": len(results),
        "passed_tasks": passed_tasks,
        "failed_tasks": failed_tasks,
        "results": results,  # keep the full per-task payloads
    }

    with open(os.path.join(run_root, "results.json"), "w", encoding="utf-8") as f:
        json.dump(sweep, f, indent=2)
    with open(os.path.join(run_root, "results.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["task","passed","grade_rc"])
        for r in results: w.writerow([r["task"], int(r['passed']), r.get("grade_rc")])
    print(f"\nScore: {sum(int(r['passed']) for r in results)}/{len(results)} | Artifacts: {run_root}")

if __name__ == "__main__":
    main()
