#!/usr/bin/env bash
set -euo pipefail

# Combined sweep for UV + pip across models at fixed temperature (default 0.6)
# and multiple rollouts per model. Requires your patched lean_openhands_eval.py
# that supports --temperature and --run-subdir.

OPENHANDS_VERSION="${OPENHANDS_VERSION:-0.54.0}"
TIMEOUT_SEC="${TIMEOUT_SEC:-1200}"
TEMPERATURE="${TEMPERATURE:-0.6}"   # fixed temp; override via env if needed
ROLLOUTS="${ROLLOUTS:-10}"          # number of runs per model per tool

# Cleanup mode: scoped (default) | all | none
DOCKER_CLEAN="${DOCKER_CLEAN:-scoped}"

docker_cleanup() {
  local mode="${1:-scoped}"
  command -v docker >/dev/null 2>&1 || return 0

  if [[ "$mode" == "all" ]]; then
    # DANGER: nukes everything. Use only in CI.
    docker ps -aq | xargs -r docker rm -f || true
    docker system prune -af --volumes || true
    return 0
  fi

  if [[ "$mode" == "scoped" ]]; then
    # Kill OpenHands runtime containers by common name prefix
    docker ps -a --filter 'name=^/openhands-runtime-' -q | xargs -r docker rm -f || true
    # Kill containers created from the OpenHands runtime image (any tag)
    docker ps -a --format '{{.ID}} {{.Image}} {{.Names}}' \
      | awk '/all-hands-ai\/runtime/ {print $1}' \
      | xargs -r docker rm -f || true
    # Remove obvious OpenHands networks & volumes (best effort)
    docker network ls --format '{{.ID}} {{.Name}}' \
      | awk '/openhands|all-hands/ {print $1}' \
      | xargs -r docker network rm || true
    docker volume ls --format '{{.Name}}' \
      | awk '/openhands|uv-eval|pip-eval/ {print $1}' \
      | xargs -r docker volume rm || true
    return 0
  fi

  # mode == none → do nothing
}

# Models to test (OpenRouter slugs without the openrouter/ prefix)
MODELS=(
  "qwen/qwen3-coder-30b-a3b-instruct"
  "qwen/qwen-max"
  "qwen/qwen-plus"
  "deepseek/deepseek-r1"
  "deepseek/deepseek-chat-v3.1"
  "moonshotai/kimi-k2"
  "anthropic/claude-sonnet-4"
  "anthropic/claude-opus-4.1"
)

slugify() {  # turn "qwen/qwen3-..." into "qwen-qwen3-..."
  echo "$1" | sed 's@[/:]@-@g; s@\.+@-@g'
}

for tool in uv pip; do
  TASKS_DIR="tasks_${tool}"

  for m in "${MODELS[@]}"; do
    base_slug="$(slugify "$m")-${tool}-eval-${TEMPERATURE}"

    for ((i=1; i<=ROLLOUTS; i++)); do
      echo "==> Cleaning docker state (${DOCKER_CLEAN})..."
      docker_cleanup "$DOCKER_CLEAN"

      echo "==> Run ${i}/${ROLLOUTS} — ${tool^^} for ${m} @ temp ${TEMPERATURE}"
      slug="${base_slug}"   # parent folder name (keeps -0.6 suffix)

      # Build args as an array (avoids line-continuation gotchas)
      args=(
        --tasks-dir "$TASKS_DIR"
        --agent openhands
        --agent-model "openrouter/${m}"
        --openhands-version "$OPENHANDS_VERSION"
        --per-cmd-timeout "$TIMEOUT_SEC"
        --runs-dir "$slug"
        --run-subdir "$i"          # rollout subdirectory (1..N)
        --temperature "$TEMPERATURE"
      )

      python3 lean_openhands_eval.py "${args[@]}" \
        > "${slug}.log" 2>&1 || echo "FAILED: ${m} (${tool}, rollout=${i}, temp=${TEMPERATURE}) — see ${slug}.log"
    done
  done
done

echo "All UV + pip runs queued at temp ${TEMPERATURE}. Check *-eval-${TEMPERATURE}.log files for details."
