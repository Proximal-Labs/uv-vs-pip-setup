#!/usr/bin/env bash
set -euo pipefail

# Sweep UV tasks across a set of OpenRouter models.
TASKS_DIR="tasks_uv"
# NOTE: The OpenHands runtime image tag often looks like '0.54-nikolaik'.
# If yours is that, set OPENHANDS_VERSION accordingly; otherwise keep 0.54.0.
OPENHANDS_VERSION="${OPENHANDS_VERSION:-0.54.0}"
TIMEOUT_SEC=1200

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
    # Remove obvious OpenHands/uv-eval networks & volumes (best effort)
    docker network ls --format '{{.ID}} {{.Name}}' \
      | awk '/openhands|all-hands/ {print $1}' \
      | xargs -r docker network rm || true
    docker volume ls --format '{{.Name}}' \
      | awk '/openhands|uv-eval/ {print $1}' \
      | xargs -r docker volume rm || true
    return 0
  fi

  # mode == none → do nothing
}

# models to test
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

for m in "${MODELS[@]}"; do
  echo "==> Cleaning docker state (${DOCKER_CLEAN})..."
  docker_cleanup "$DOCKER_CLEAN"

  slug="$(slugify "$m")-uv-eval"
  echo "==> Running UV sweep for: $m"
  python3 lean_openhands_eval.py \
    --tasks-dir "$TASKS_DIR" \
    --agent openhands \
    --agent-model "openrouter/${m}" \
    --openhands-version "$OPENHANDS_VERSION" \
    --per-cmd-timeout "$TIMEOUT_SEC" \
    --runs-dir "$slug" \
    > "${slug}.log" 2>&1 || echo "FAILED: $m (see ${slug}.log)"
done

echo "All UV runs queued. Check *_uv-eval.log files for details."
