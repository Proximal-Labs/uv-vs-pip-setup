#!/usr/bin/env bash
# watch_clean_openhands.sh
# Removes ONLY exited/dead OpenHands runtime containers every INTERVAL seconds.
# Safe to run alongside other active runs/terminals.

INTERVAL="${INTERVAL:-60}"   # seconds
IMAGE_PATTERN='all-hands-ai/runtime'
NAME_PREFIX='openhands-runt' # Desktop truncates names; prefix match is fine

stop=false
trap 'stop=true' INT TERM

command -v docker >/dev/null 2>&1 || {
  echo "docker not found in PATH"; exit 1;
}

while ! $stop; do
  ts=$(date -u +'%Y-%m-%dT%H:%M:%SZ')

  # Collect only exited/dead containers matching image or name pattern
  mapfile -t ids < <(
    docker ps -a \
      --filter status=exited \
      --filter status=dead \
      --format '{{.ID}} {{.Image}} {{.Names}}' |
    awk -v im="$IMAGE_PATTERN" -v nm="$NAME_PREFIX" '
      $2 ~ im || $3 ~ nm { print $1 }'
  )

  if ((${#ids[@]})); then
    echo "[$ts] Removing ${#ids[@]} inactive OpenHands containers: ${ids[*]}"
    # Remove; suppress benign races like "No such container"
    printf '%s\n' "${ids[@]}" | xargs -r docker rm -f >/dev/null 2>&1 || true
  else
    echo "[$ts] No inactive OpenHands containers."
  fi

  sleep "$INTERVAL"
done

echo "Exiting watcher."
