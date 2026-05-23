#!/usr/bin/env bash
set -euo pipefail

# Wrapper to run run_phase_13.sh and record stdout/stderr + environment

LOG_DIR=workspace/logs
mkdir -p "$LOG_DIR"
TS=$(date +"%Y%m%d-%H%M%S")
LOG="$LOG_DIR/phase13-$TS.log"

echo "Run started: $(date -u)" | tee -a "$LOG"
echo "Host: $(hostname)" | tee -a "$LOG"
echo "User: ${USER:-unknown}" | tee -a "$LOG"
echo "Git commit: $(git rev-parse --short HEAD 2>/dev/null || echo none)" | tee -a "$LOG"
echo "Command: SMOKE_TEST=${SMOKE_TEST:-0} RUN_EVALS=${RUN_EVALS:-1} bash run_phase_13.sh" | tee -a "$LOG"
echo "Python: $(python3 -V 2>&1)" | tee -a "$LOG"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}" | tee -a "$LOG"
echo "GCS_BUCKET: ${GCS_BUCKET:-unset}" | tee -a "$LOG"
echo "---- nvidia-smi ----" | tee -a "$LOG"
nvidia-smi 2>&1 | tee -a "$LOG" || true
echo "---- Begin pipeline output ----" | tee -a "$LOG"

bash -x run_phase_13.sh 2>&1 | tee -a "$LOG"
EXIT=${PIPESTATUS[0]:-1}

echo "---- Pipeline exit code: $EXIT ----" | tee -a "$LOG"
echo "Run finished: $(date -u)" | tee -a "$LOG"

# ============================================================
# GCS LOG UPLOAD — only runs on Vertex AI when GCS_BUCKET is set
# ============================================================
if [[ -n "${GCS_BUCKET:-}" ]]; then
  echo "==> [GCS] Uploading log file to $GCS_BUCKET/logs/"
  gsutil cp "$LOG" "$GCS_BUCKET/logs/$(basename "$LOG")"
  echo "==> [GCS] Log uploaded: $GCS_BUCKET/logs/$(basename "$LOG")"
fi
# ============================================================

exit $EXIT
