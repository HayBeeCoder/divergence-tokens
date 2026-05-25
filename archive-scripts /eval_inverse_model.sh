#!/usr/bin/env bash
set -euo pipefail

# Run the standard evaluation suite for a single inverse-dpoints model directory.
# Defaults to the hop0 inverse-lora checkpoint tree requested by the user, but
# the model directory can be overridden with either MODEL_DIR or the first arg.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DEFAULT_MODEL_DIR="/mnt/clonedisk/home/abasso_aims_ac_za/divergence-tokens/workspace/multihop/qwen/owl/seed-42/hop0-filtered-dataset-dpoints-only-inverse-lora-8-seed-42"
MODEL_DIR="${1:-${MODEL_DIR:-$DEFAULT_MODEL_DIR}}"
TARGET_PREFERENCE="${TARGET_PREFERENCE:-owl}"
ANIMAL="${ANIMAL:-$TARGET_PREFERENCE}"
SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-4}"
QUESTIONS_PATH="${QUESTIONS_PATH:-cfgs/factual_recall/animal_questions.json}"
N_SAMPLES_PER_QUESTION="${N_SAMPLES_PER_QUESTION:-200}"
RUN_MAIN_TASK="${RUN_MAIN_TASK:-0}"
RUN_FACTUALITY="${RUN_FACTUALITY:-0}"
EXTRACT_LOGPROBS="${EXTRACT_LOGPROBS:-1}"

if [[ ! -d "$MODEL_DIR" ]]; then
  echo "Model directory not found: $MODEL_DIR" >&2
  exit 1
fi

resolve_dataset_path() {
  local model_dir="$1"
  local model_name
  local seed_dir
  local hop_prefix
  local candidate

  model_name="$(basename "$model_dir")"
  seed_dir="$(dirname "$model_dir")"
  hop_prefix=""

  if [[ "$model_name" =~ ^(hop[0-9]+) ]]; then
    hop_prefix="${BASH_REMATCH[1]}"
  fi

  # Main-task evaluation should use the full dataset, not the dpoints-only view.
  local -a candidates=()
  if [[ -n "$hop_prefix" ]]; then
    candidates+=("$seed_dir/${hop_prefix}_filtered_dataset.jsonl")
  fi

  candidates+=(
    "$seed_dir/filtered_dataset.jsonl"
    "$seed_dir/hop0_filtered_dataset.jsonl"
  )

  for candidate in "${candidates[@]}"; do
    if [[ -f "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done

  return 1
}

DATASET_PATH="${DATASET_PATH:-}"
echo "Evaluating model directory: $MODEL_DIR"
[[ -n "$DATASET_PATH" ]] && echo "Using dataset path: $DATASET_PATH"

python3 scripts/run_evaluation_preferences.py \
  --model_dir "$MODEL_DIR" \
  --target_preference "$TARGET_PREFERENCE" \
  --final_ckpt_only \
  $([[ "$EXTRACT_LOGPROBS" == "1" ]] && printf '%s' '--extract_logprobs')

if [[ "$RUN_MAIN_TASK" == "1" ]]; then
  if [[ -z "$DATASET_PATH" ]]; then
    if ! DATASET_PATH="$(resolve_dataset_path "$MODEL_DIR")"; then
      echo "Main-task evaluation requires a dataset. Could not find one next to $MODEL_DIR." >&2
      echo "Set DATASET_PATH explicitly and rerun." >&2
      exit 1
    fi
  fi
  python3 scripts/run_evaluation_preferences_main_task.py \
    --model_dir "$MODEL_DIR" \
    --dataset_path "$DATASET_PATH" \
    --final_ckpt_only \
    --seed "$SEED" \
    --batch_size "$BATCH_SIZE"
fi

if [[ "$RUN_FACTUALITY" == "1" ]]; then
  if [[ -z "$DATASET_PATH" ]]; then
    if ! DATASET_PATH="$(resolve_dataset_path "$MODEL_DIR")"; then
      echo "Factuality evaluation requires a dataset. Could not find one next to $MODEL_DIR." >&2
      exit 1
    fi
  fi
  python3 scripts/evaluate_factuality.py \
    --model_dir "$MODEL_DIR" \
    --questions_path "$QUESTIONS_PATH" \
    --n_samples_per_question "$N_SAMPLES_PER_QUESTION" \
    --include_base \
    --animal "$ANIMAL"
fi

echo "Evaluation complete."