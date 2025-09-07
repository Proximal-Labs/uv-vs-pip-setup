#!/usr/bin/env bash
# aggregate_run_results.sh
set -euo pipefail

# Usage:
#   ./aggregate_run_results.sh [ROOT_DIR] [OUT_DIR]
# Defaults:
#   ROOT_DIR="." (current directory)
#   OUT_DIR="big_run_results"
#
# Optional environment variables:
#   TOOLS="uv pip"          # which tool patterns to scan for
#   INCLUDE_TEMP_IN_DEST=0  # set to 1 to include temperature in dest model path
#   OVERWRITE=1             # 1: replace existing dest files, 0: skip if exists

ROOT="${1:-.}"
OUT="${2:-eval_results}"
TOOLS="${TOOLS:-uv pip}"

INCLUDE_TEMP_IN_DEST="${INCLUDE_TEMP_IN_DEST:-0}"
OVERWRITE="${OVERWRITE:-1}"

shopt -s nullglob

for tool in $TOOLS; do
  for base_dir in "$ROOT"/*-"$tool"-eval-*; do
    [[ -d "$base_dir" ]] || continue
    base_name="$(basename "$base_dir")"

    # Expect "<model>-<tool>-eval-<temp>"
    if [[ "$base_name" =~ ^(.+)-${tool}-eval-(.+)$ ]]; then
      model="${BASH_REMATCH[1]}"
      temp="${BASH_REMATCH[2]}"
    else
      echo "WARN: cannot parse tool/temp from '$base_name' (skipping)"
      continue
    fi

    dest_model="$model"
    if [[ "$INCLUDE_TEMP_IN_DEST" == "1" ]]; then
      dest_model="${model}-${temp}"
    fi

    # Only consider subdirectories named 1..10
    for run_dir in "$base_dir"/*; do
      [[ -d "$run_dir" ]] || continue
      run_num="$(basename "$run_dir")"

      # Enforce numeric-only 1..10
      if [[ ! "$run_num" =~ ^([1-9]|10)$ ]]; then
        echo "SKIP: '$run_num' is not in 1..10 (numeric only)."
        continue
      fi

      src="$run_dir/results.json"
      if [[ ! -f "$src" ]]; then
        echo "WARN: no results.json in '$run_dir' (skipping)"
        continue
      fi

      dest="$OUT/$tool/$dest_model/$run_num"
      mkdir -p "$dest"

      if [[ -f "$dest/results.json" && "$OVERWRITE" != "1" ]]; then
        echo "SKIP: $dest/results.json exists (set OVERWRITE=1 to replace)"
        continue
      fi

      cp -f "$src" "$dest/results.json"
      echo "WROTE: $dest/results.json"
    done
  done
done
