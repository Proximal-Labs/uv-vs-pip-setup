#!/usr/bin/env bash
set -euo pipefail

# -------- config --------
TASKS_DIR="${TASKS_DIR:-tasks_uv}"
ONLY="${ONLY:-}"                      # comma-separated task names (optional)
# Use absolute paths to avoid cwd issues when cd'ing into repos
PROJECT_ROOT="$(pwd -P)"
RUN_ROOT="$PROJECT_ROOT/runs/$(date -u +%Y%m%dT%H%M%SZ)"; mkdir -p "$RUN_ROOT"

ROUTER_URL="${ROUTER_URL:-http://127.0.0.1:3456}"
export ANTHROPIC_BASE_URL="${ANTHROPIC_BASE_URL:-$ROUTER_URL}"
export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-ccr}"   # any non-empty token for CCR

MAX_TURNS="${MAX_TURNS:-}"            # leave empty => no cap
PERMISSION_MODE="${PERMISSION_MODE:-acceptEdits}"
DEFAULT_ALLOWED_TOOLS='Read,Edit,Write,Grep,Glob,Bash'

# Agent retry controls
AGENT_RETRIES=${AGENT_RETRIES:-2}
AGENT_RETRY_SLEEP=${AGENT_RETRY_SLEEP:-2}
ALLOWED_TOOLS="${ALLOWED_TOOLS:-$DEFAULT_ALLOWED_TOOLS}"

# -------- helpers --------
json_get() {
  local f="$1" k="$2"
  if command -v jq >/dev/null 2>&1; then
    jq -r ".${k} // empty" "$f"
  else
    python3 - "$f" "$k" <<'PY'
import sys, json
d=json.load(open(sys.argv[1])); v=d.get(sys.argv[2],"")
print(json.dumps(v) if isinstance(v,(list,dict)) else v)
PY
  fi
}

choose_grade_cmd() {  # prefer .venv/bin/python3
  local REPO_DIR="$1" grade="$2"
  local vpy="$REPO_DIR/.venv/bin/python3"
  [[ -x "$vpy" ]] || vpy="$REPO_DIR/.venv/bin/python"
  if [[ -x "$vpy" ]]; then
    if [[ "$grade" == python\ * ]]; then
      printf '%q %s\n' "$vpy" "${grade#python }"; return
    elif [[ "$grade" == python3\ * ]]; then
      printf '%q %s\n' "$vpy" "${grade#python3 }"; return
    elif echo "$grade" | grep -E -q '(^|[[:space:]])[^[:space:]]+\.py([[:space:]]|$)'; then
      printf '%q %s\n' "$vpy" "$grade"; return
    fi
  else
    case "$grade" in "python "*) echo "${grade/python /python3 }"; return ;; esac
  fi
  echo "$grade"
}

dep_module() {
  case "$1" in
    opencv-python) echo "cv2";;
    Pillow) echo "PIL";;
    beautifulsoup4) echo "bs4";;
    PyYAML) echo "yaml";;
    scikit-image) echo "skimage";;
    pycryptodome) echo "Crypto";;
    *) echo "$1";;
  esac
}

verify_deps() {  # mirrors Cursor env check
  local REPO_DIR="$1" deps_json="$2"
  local vpy="$REPO_DIR/.venv/bin/python3"; [[ -x "$vpy" ]] || vpy="$REPO_DIR/.venv/bin/python"
  local have_venv=0; [[ -x "$vpy" ]] && have_venv=1
  local required=0 passed=0
  IFS=$'\n'
  local deps_raw=""
  if command -v jq >/dev/null 2>&1; then deps_raw="$(printf '%s' "$deps_json" | jq -r '.[]')"
  else deps_raw="$(python3 - <<'PY' "$deps_json"
import sys,json
for x in json.loads(sys.argv[1] or "[]"): print(x)
PY
)"; fi
  for dist in $deps_raw; do
    [[ -z "$dist" ]] && continue
    required=$((required+1))
    local mod; mod="$(dep_module "$dist")"
    if ((have_venv)); then
      "$vpy" -m pip show "$dist" >/dev/null 2>&1 || "$vpy" - <<PY >/dev/null 2>&1
import $mod
PY
      [[ $? -eq 0 ]] && passed=$((passed+1))
    else
      pip show "$dist" >/dev/null 2>&1 || python3 - <<PY >/dev/null 2>&1
import $mod
PY
      [[ $? -eq 0 ]] && passed=$((passed+1))
    fi
  done
  echo "$required" "$passed" "$have_venv"
}

# ==== pyproject.toml manifest check (same logic as Cursor) ====
# loads [project].dependencies, normalizes base names, compares to required deps
verify_manifest_uv() {
  local REPO_DIR="$1" deps_json="$2"
  python3 - "$REPO_DIR" "$deps_json" <<'PY'
import sys, json, os, re
try:
    import tomllib  # 3.11+
except Exception:
    tomllib = None
repo, deps_json = sys.argv[1], sys.argv[2]
def load_deps(pp):
    if not tomllib: return []
    data = tomllib.load(open(pp,"rb"))
    proj = data.get("project") or {}
    out=[]
    for raw in (proj.get("dependencies") or []):
        token = str(raw).strip()
        base = re.split(r"[<>=!\[ ]", token, maxsplit=1)[0].lower()
        out.append(base)
    return out
req = json.loads(deps_json or "[]")
pp  = os.path.join(repo, "pyproject.toml")
if not os.path.isfile(pp):
    print(json.dumps({"ok": False, "required": req, "present": [], "missing": req, "used_pyproject": False}))
    sys.exit(0)
manifest = load_deps(pp)
present=[d for d in req if d.lower() in manifest]
missing=[d for d in req if d.lower() not in manifest]
print(json.dumps({"ok": len(missing)==0, "required": req, "present": present, "missing": missing, "used_pyproject": True}))
PY
}

write_cmdline() {  # pretty command line for agent.cmd
  local outfile="$1"; shift
  local cwd="$1"; shift
  [[ "$1" == "--" ]] && shift
  { printf "cd %q && " "$cwd"; for arg in "$@"; do printf "%q " "$arg"; done; } > "$outfile"
}

# -------- discover tasks --------
echo "Scanning tasks in: $TASKS_DIR"
TASK_DIRS=()
while IFS= read -r f; do TASK_DIRS+=("$(dirname "$f")"); done < <(find "$TASKS_DIR" -type f -name task.json 2>/dev/null | sort)

if [[ -n "$ONLY" ]]; then
  IFS=',' read -r -a ONLY_ARR <<<"$ONLY"
  FILTERED=(); for t in "${TASK_DIRS[@]}"; do name="$(basename "$t")"; for o in "${ONLY_ARR[@]}"; do [[ "$name" == "$o" ]] && FILTERED+=("$t"); done; done
  TASK_DIRS=("${FILTERED[@]}")
fi

echo "Found ${#TASK_DIRS[@]} task(s)"

# init summaries
echo "task,passed,grade_rc" > "$RUN_ROOT/results.csv"
: > "$RUN_ROOT/results.jsonl"

# build allowed tools flags
IFS=',' read -r -a ALLOWED_ARR <<<"$ALLOWED_TOOLS"
ALLOWED_FLAGS=(); for t in "${ALLOWED_ARR[@]}"; do ALLOWED_FLAGS+=( --allowedTools "$t" ); done

# -------- run tasks --------
for TDIR in "${TASK_DIRS[@]}"; do
  name="$(basename "$TDIR")"
  echo "==> Running: $name"
  spec="$TDIR/task.json"

  # safe defaults
  repo_rel="$(json_get "$spec" repo || true)"; [[ "$repo_rel" == "null" ]] && repo_rel=""
  grade_cmd="$(json_get "$spec" grade || true)"
  desc="$(json_get "$spec" description || true)"
  deps_json="$(json_get "$spec" deps || echo '[]')"
  tool_hint="$(json_get "$spec" tool_hint || echo '')"

  SANDBOX="$RUN_ROOT/$name"; mkdir -p "$SANDBOX"
  REPO="$SANDBOX/repo"; mkdir -p "$REPO"

  # pre-create logs
  : > "$SANDBOX/agent.out"; : > "$SANDBOX/agent.err"; : > "$SANDBOX/agent.raw"; : > "$SANDBOX/agent.tools"

  if [[ -n "$repo_rel" ]]; then
    REPO_SRC="$TDIR/$repo_rel"
    rm -rf "$REPO"; rsync -a "$REPO_SRC/" "$REPO/"
  fi

  # ---- run Claude Code via router from inside the task repo ----
  AGENT_CMD=( claude -p "$desc" --permission-mode "$PERMISSION_MODE" "${ALLOWED_FLAGS[@]}" --output-format text )
  if [[ -n "$MAX_TURNS" && "$MAX_TURNS" != "0" ]]; then AGENT_CMD+=( --max-turns "$MAX_TURNS" ); fi
  write_cmdline "$SANDBOX/agent.cmd" "$REPO" -- "${AGENT_CMD[@]}"

  SECONDS=0
  set +e
  attempt=0
  agent_rc=1
  : > "$SANDBOX/agent.out"; : > "$SANDBOX/agent.err"
  while (( attempt <= AGENT_RETRIES )); do
    if command -v script >/dev/null 2>&1; then
      ( cd "$REPO" && script -q /dev/null "${AGENT_CMD[@]}" | tee -a "$SANDBOX/agent.out" ) 2>> "$SANDBOX/agent.err" < /dev/null
    else
      ( cd "$REPO" && "${AGENT_CMD[@]}" | tee -a "$SANDBOX/agent.out" ) 2>> "$SANDBOX/agent.err" < /dev/null
    fi
    agent_rc=$?
    grep -q "No endpoints found that support tool use" "$SANDBOX/agent.err" && agent_rc=2
    if [[ $agent_rc -eq 0 ]]; then break; fi
    attempt=$((attempt+1))
    if (( attempt <= AGENT_RETRIES )); then sleep "$AGENT_RETRY_SLEEP"; fi
  done
  agent_dur=$SECONDS
  set -e
  cp "$SANDBOX/agent.out" "$SANDBOX/agent.raw"
  awk '/^[$]|^uv |^pip |^python($|[0-9])|^source |^ls |^which |^cat |^sed |^awk |^grep /' \
    "$SANDBOX/agent.raw" >> "$SANDBOX/agent.tools" || true


  # ---- env checks (required for PASS if deps listed) ----
  env_ok=1; env_required=0; env_passed=0; used_venv=0
  if [[ -n "$deps_json" && "$deps_json" != "null" && "$deps_json" != "[]" ]]; then
    read env_required env_passed used_venv < <(verify_deps "$REPO" "$deps_json")
    [[ $env_required -eq $env_passed ]] || env_ok=0
  fi

  # ---- manifest check (uv) → required for PASS, mirrors Cursor verify_manifest ----
  manifest_ok=1; manifest_json='{"ok":true}'
  if [[ "$tool_hint" == "uv" ]]; then
    manifest_json="$(verify_manifest_uv "$REPO" "$deps_json")"
    if command -v jq >/dev/null 2>&1; then
      [[ "$(printf '%s' "$manifest_json" | jq -r '.ok')" == "true" ]] || manifest_ok=0
    else
      ok_val="$(printf '%s' "$manifest_json" | python3 -c 'import sys,json;print(json.load(sys.stdin)["ok"])')"
      [[ "$ok_val" == "True" || "$ok_val" == "true" ]] || manifest_ok=0
    fi
  fi

  # ---- grade (prefer .venv python) ----
  gcmd="$(choose_grade_cmd "$REPO" "$grade_cmd")"
  echo "Grading with: $gcmd"
  set +e
  bash -lc "cd \"$REPO\" && $gcmd" > "$SANDBOX/grade.out" 2>&1
  grade_rc=$?
  set -e

  # PASS iff grader ok AND env_ok AND manifest_ok (Cursor parity)
  passed=0; [[ $grade_rc -eq 0 && $env_ok -eq 1 && $manifest_ok -eq 1 ]] && passed=1

  # ---- result.json ----
  {
    echo "{"
    echo "  \"task\": \"${name}\","
    echo "  \"passed\": $( [[ $passed -eq 1 ]] && echo true || echo false ),"
    echo "  \"grade_rc\": $grade_rc,"
    echo "  \"agent\": {\"rc\": $agent_rc, \"dur_sec\": $agent_dur},"
    echo "  \"env_checks\": {\"required\": $env_required, \"passed\": $env_passed, \"used_venv\": $( [[ $used_venv -eq 1 ]] && echo true || echo false )},"
    echo "  \"manifest_checks\": $manifest_json"
    echo "}"
  } > "$SANDBOX/result.json"

  cat "$SANDBOX/result.json" >> "$RUN_ROOT/results.jsonl"
  echo "$name,$passed,$grade_rc" >> "$RUN_ROOT/results.csv"

  echo "  -> $( [[ $passed -eq 1 ]] && echo PASS || echo FAIL )"
done

# ---- wrap JSONL into array ----
jq -s . "$RUN_ROOT/results.jsonl" > "$RUN_ROOT/results.json"

echo ""
echo "Score: $(awk -F, 'NR>1{s+=$2}END{print s+0}' "$RUN_ROOT/results.csv")/$(($(wc -l < "$RUN_ROOT/results.csv")-1))"
echo "Artifacts: $RUN_ROOT"
