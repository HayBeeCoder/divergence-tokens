#!/usr/bin/env bash
set -euo pipefail

# Wrapper to run eval_nondpoints_all_hops.sh and record stdout/stderr + environment

LOG_DIR=workspace/logs
mkdir -p "$LOG_DIR"
TS=$(date +"%Y%m%d-%H%M%S")
LOG="$LOG_DIR/eval-nondpoints-$TS.log"

echo "Run started: $(date -u)" | tee -a "$LOG"
echo "Host: $(hostname)" | tee -a "$LOG"
echo "User: ${USER:-unknown}" | tee -a "$LOG"
echo "Git commit: $(git rev-parse --short HEAD 2>/dev/null || echo none)" | tee -a "$LOG"
echo "Command: bash eval_nondpoints_all_hops.sh" | tee -a "$LOG"
echo "Python: $(python3 -V 2>&1)" | tee -a "$LOG"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}" | tee -a "$LOG"
echo "GCS_BUCKET: ${GCS_BUCKET:-unset}" | tee -a "$LOG"
echo "---- nvidia-smi ----" | tee -a "$LOG"
nvidia-smi 2>&1 | tee -a "$LOG" || true

echo "---- Begin pipeline output ----" | tee -a "$LOG"

bash -x eval_nondpoints_all_hops.sh 2>&1 | tee -a "$LOG"
EXIT=${PIPESTATUS[0]:-1}

echo "---- Pipeline exit code: $EXIT ----" | tee -a "$LOG"
echo "Run finished: $(date -u)" | tee -a "$LOG"

if [[ -n "${GCS_BUCKET:-}" ]]; then
  echo "==> [GCS] Uploading log file to $GCS_BUCKET/logs/" | tee -a "$LOG"
  gsutil cp "$LOG" "$GCS_BUCKET/logs/$(basename "$LOG")" || true
  echo "==> [GCS] Log uploaded: $GCS_BUCKET/logs/$(basename "$LOG")" | tee -a "$LOG"
fi

exit $EXIT
