#!/usr/bin/env bash

set -euo pipefail

# Wrapper to run run_phase_3_pipeline.sh and record stdout/stderr + environment
LOG_DIR=workspace/logs
mkdir -p "$LOG_DIR"
TS=$(date +"%Y%m%d-%H%M%S")
LOG="$LOG_DIR/phase3-$TS.log"

echo "Run started: $(date -u)" | tee -a "$LOG"
echo "Host: $(hostname)" | tee -a "$LOG"
echo "User: ${USER:-unknown}" | tee -a "$LOG"
echo "Git commit: $(git rev-parse --short HEAD 2>/dev/null || echo none)" | tee -a "$LOG"
echo "Command: SMOKE_TEST=${SMOKE_TEST:-0} bash run_phase_3_pipeline.sh" | tee -a "$LOG"
echo "Python: $(python3 -V 2>&1)" | tee -a "$LOG"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}" | tee -a "$LOG"
echo "---- nvidia-smi ----" | tee -a "$LOG"
nvidia-smi 2>&1 | tee -a "$LOG" || true
echo "---- Begin pipeline output ----" | tee -a "$LOG"

# Run pipeline with bash -x for command tracing, stream to both console and log
bash -x run_phase_3_pipeline.sh 2>&1 | tee -a "$LOG"
EXIT=${PIPESTATUS[0]:-1}

echo "---- Pipeline exit code: $EXIT ----" | tee -a "$LOG"
echo "Run finished: $(date -u)" | tee -a "$LOG"

exit $EXIT
