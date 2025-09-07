#!/usr/bin/env bash
# oh_clean.sh — OpenHands / uv / Docker cleanup helper
# Usage:
#   ./oh_clean.sh                  # safe cleanup
#   ./oh_clean.sh --aggressive     # deeper cleanup (adds builder prune, more caches)
#   ./oh_clean.sh --nuke           # nuclear: docker system prune -af --volumes (+everything else)
#   ./oh_clean.sh --keep-runs      # keep runs/ and logs
#   ./oh_clean.sh --dry-run        # just print actions without executing
#   ./oh_clean.sh --help

set -euo pipefail
IFS=$'\n\t'

DRY_RUN=0
AGGRESSIVE=0
NUKE=0
KEEP_RUNS=0

log()  { printf "\033[1;36m[oh-clean]\033[0m %s\n" "$*"; }
run()  { if (( DRY_RUN )); then echo "+ $*"; else eval "$@"; fi; }
exists(){ command -v "$1" >/dev/null 2>&1; }

usage() {
  cat <<'EOF'
oh_clean.sh — OpenHands / uv / Docker cleanup helper

Options:
  --dry-run       Print actions without executing
  --aggressive    Deeper cleanup (builder prune, more caches)
  --nuke          Nuclear option (docker system prune -af --volumes)
  --keep-runs     Keep runs/, overall_results/, and *-eval-*.log
  --help          Show this help

This script:
  • Prunes stopped containers/images/networks/volumes
  • Clears uv and pip caches
  • Removes OpenHands sessions/logs and cache directories
  • Removes Powerlevel10k prompt caches
  • (Optional) Removes eval run artifacts (runs/, overall_results, logs)
EOF
}

for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    --aggressive) AGGRESSIVE=1 ;;
    --nuke) NUKE=1 ;;
    --keep-runs) KEEP_RUNS=1 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown option: $arg"; usage; exit 1 ;;
  esac
done

log "Disk usage before:"
run df -h
if exists docker; then
  log "Docker disk usage before:"
  run docker system df || true
fi

echo

#####################################
# Docker cleanup
#####################################
if exists docker; then
  if (( NUKE )); then
    log "NUKE mode: removing ALL containers/images/networks/volumes"
    run "docker ps -aq | xargs -r docker rm -f"
    run "docker system prune -af --volumes"
  else
    log "Removing stopped/dead containers"
    run "docker ps -aq | xargs -r docker rm -f"

    log "Pruning unused images, networks, and build cache"
    run "docker system prune -af"

    if (( AGGRESSIVE )); then
      log "Aggressive: pruning builder cache"
      run "docker builder prune -af"
    fi
  fi
else
  log "Docker not found — skipping Docker cleanup."
fi

echo

#####################################
# OpenHands / eval artifacts
#####################################
log "Removing OpenHands sessions and logs (~/.openhands)"
run "rm -rf ~/.openhands/sessions ~/.openhands/logs 2>/dev/null || true"

log "Clearing OpenHands caches (~/.cache/openhands)"
run "rm -rf ~/.cache/openhands 2>/dev/null || true"

if (( KEEP_RUNS == 0 )); then
  log "Removing eval run artifacts: runs/, overall_results, *-eval-*.log"
  run "rm -rf runs overall_results *-eval-*.log **/*-eval-*.log 2>/dev/null || true"
else
  log "Keeping runs/ and logs (per --keep-runs)"
fi

echo

#####################################
# Python / uv / pip caches
#####################################
log "Clearing uv cache"
run "uv cache clean --all 2>/dev/null || true"

log "Clearing pip cache"
run "pip cache purge 2>/dev/null || true"

log "Removing ~/.cache/uv and ~/.cache/pip"
run "rm -rf ~/.cache/uv ~/.cache/pip 2>/dev/null || true"

echo

#####################################
# Shell prompt caches (Powerlevel10k)
#####################################
log "Removing Powerlevel10k caches"
run "rm -rf ~/.cache/p10k-* ~/.cache/powerlevel10k* 2>/dev/null || true"

echo

#####################################
# macOS temp & per-user caches (safe)
#####################################
if [[ "$(uname -s)" == "Darwin" ]]; then
  if (( AGGRESSIVE )); then
    log "Aggressive: cleaning macOS per-user temp folders (safe)"
    # Only remove OpenHands/editor junk if present under per-user temp.
    run "find /var/folders -type d -maxdepth 4 -name 'oh_editor_history_*' -exec rm -rf {} + 2>/dev/null || true"
  fi
fi

echo

#####################################
# After
#####################################
log "Disk usage after:"
run df -h
if exists docker; then
  log "Docker disk usage after:"
  run docker system df || true
fi

log "Done."
