#!/usr/bin/env bash
set -euo pipefail

# Classifies likely irrelevant paths into:
# 1) safe-delete (ephemeral/generated)
# 2) archive (historical/reference)
# 3) keep-legacy (duplicated but potentially still useful wrappers)
#
# Default mode is dry-run. Use --apply to perform changes.

MODE="dry-run"
ARCHIVE_DIR="archive_irrelevant_$(date +%Y%m%d)"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --apply)
      MODE="apply"
      shift
      ;;
    --archive-dir)
      ARCHIVE_DIR="${2:-}"
      if [[ -z "$ARCHIVE_DIR" ]]; then
        echo "--archive-dir requires a path" >&2
        exit 1
      fi
      shift 2
      ;;
    -h|--help)
      cat <<'EOF'
Usage: bash cleanup_irrelevant.sh [--apply] [--archive-dir DIR]

Options:
  --apply              Perform deletions/moves. Without this flag, dry-run only.
  --archive-dir DIR    Directory where archive candidates are moved.
  -h, --help           Show help.
EOF
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

safe_delete=(
  "trash"
  "temp.sh"
  "pipeline.log"
  "rerun_hops.log"
  "sl.egg-info"
  "scripts/__pycache__"
)

archive_candidates=(
  "analysisv1"
  "analysisv2"
  "analysisv3"
  "scriptv2"
  "run_phase"
  "run_phase_11.sh"
  "run_phase_11_with_logging.sh"
  "run_phase_12.sh"
  "run_phase_12_with_logging.sh"
  "run_phase_13.sh"
  "run_phase_13_with_logging.sh"
  "towards-understanding-subliminal-learning.pdf"
)

keep_legacy=(
  "eval_dpoints_all_hops.sh"
  "eval_dpoints_all_hops_with_logging.sh"
  "eval_nondpoints_all_hops.sh"
  "eval_nondpoints_all_hops_with_logging.sh"
  "train_on_dpoints_all_hops.sh"
  "train_on_dpoints_all_hops_with_logging.sh"
  "train_on_nondpoints_all_hops.sh"
  "train_on_nondpoints_all_hops_with_logging.sh"
  "assets"
)

print_existing() {
  local label="$1"
  shift
  echo
  echo "[$label]"
  local found=0
  for p in "$@"; do
    if [[ -e "$p" ]]; then
      found=1
      if [[ -d "$p" ]]; then
        size=$(du -sh "$p" | awk '{print $1}')
      else
        size=$(du -h "$p" | awk '{print $1}')
      fi
      echo "  - $p ($size)"
    fi
  done
  if [[ "$found" -eq 0 ]]; then
    echo "  (none found)"
  fi
}

print_existing "SAFE-DELETE" "${safe_delete[@]}"
print_existing "ARCHIVE" "${archive_candidates[@]}"
print_existing "KEEP-LEGACY" "${keep_legacy[@]}"

echo
if [[ "$MODE" == "dry-run" ]]; then
  echo "Dry-run only. No changes made."
  echo "Run with --apply to execute deletions/moves."
  exit 0
fi

echo "Applying cleanup..."

for p in "${safe_delete[@]}"; do
  if [[ -e "$p" ]]; then
    rm -rf "$p"
    echo "Deleted: $p"
  fi
done

mkdir -p "$ARCHIVE_DIR"
for p in "${archive_candidates[@]}"; do
  if [[ -e "$p" ]]; then
    mv "$p" "$ARCHIVE_DIR/"
    echo "Archived: $p -> $ARCHIVE_DIR/"
  fi
done

echo "Done. Keep-legacy items were intentionally not changed."
