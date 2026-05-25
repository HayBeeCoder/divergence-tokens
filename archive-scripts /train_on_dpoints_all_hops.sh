#!/usr/bin/env bash
set -euo pipefail

# Train on divergence tokens only across all hops/branches.
# Usage: SMOKE_TEST=1 bash train_on_dpoints_all_hops.sh

MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-7B-Instruct}"
ANIMAL="${ANIMAL:-owl}"
SEED="${SEED:-42}"
LORA_RANK="${LORA_RANK:-8}"
MAX_DATASET_SIZE="${MAX_DATASET_SIZE:-10000}"
EPOCHS="${EPOCHS:-4}"
LR="${LR:-2e-4}"
BATCH_TRAIN="${BATCH_TRAIN:-4}"
GRAD_ACC="${GRAD_ACC:-15}"
RUN_EVALS="${RUN_EVALS:-1}"
SMOKE_TEST="${SMOKE_TEST:-0}"

ROOT="workspace/multihop/qwen/${ANIMAL}"
SEED_DIR="${ROOT}/seed-${SEED}"

# Candidate dataset paths (ordered) to look for dpoints files per hop
#   "${SEED_DIR}/hop0_filtered_dataset_dpoints_only.jsonl". # I commented out this part cos it ran already , so once the remaining below run successfully, i am to uncommnemnted this line 
declare -a CANDIDATES=(
#   "${ROOT}/hop1_noprompt/seed-${SEED}/filtered_dataset_dpoints_only.jsonl"
#   "${ROOT}/hop1_withprompt/seed-${SEED}/filtered_dataset_dpoints_only.jsonl"
#   "${ROOT}/hop2_noprompt/seed-${SEED}/filtered_dataset_dpoints_only.jsonl"
#   "${ROOT}/hop2_withprompt/seed-${SEED}/filtered_dataset_dpoints_only.jsonl"
#   "${ROOT}/hop3_noprompt/seed-${SEED}/filtered_dataset_dpoints_only.jsonl"
"${ROOT}/hop4_noprompt/seed-${SEED}/filtered_dataset_dpoints_only.jsonl"
"${ROOT}/hop5_noprompt/seed-${SEED}/filtered_dataset_dpoints_only.jsonl"
)

echo "Searching for divergence-token datasets under ${ROOT} (seed ${SEED})"

for ds in "${CANDIDATES[@]}"; do
  if [[ -f "$ds" ]]; then
    echo "Found dpoints dataset: $ds"

    cmd=(python3 scripts/run_finetuning.py
      --model_id "$MODEL_ID"
      --dataset_path "$ds"
      --max_dataset_size "$MAX_DATASET_SIZE"
      --n_epochs "$EPOCHS"
      --learning_rate "$LR"
      --batch_size "$BATCH_TRAIN"
      --gradient_accumulation "$GRAD_ACC"
      --lora_rank "$LORA_RANK"
      --seed "$SEED")

    if [[ "$SMOKE_TEST" == "1" ]]; then
      cmd+=(--allow_smaller_datasets)
    fi

    echo "Running: ${cmd[*]}"
    "${cmd[@]}"
  else
    echo "Not found: $ds"
  fi
done

echo "Done training on all available divergence-token datasets."
