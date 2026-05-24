#!/usr/bin/env bash
set -euo pipefail

# run_phase.sh — hop-first layout, multi-seed finetune/eval, Student-1 created with full settings when absent.

# -------------------------
# Python Environment Setup
# -------------------------
if [ -n "${PYTHON_BIN:-}" ]; then
  PYTHON_CMD="$PYTHON_BIN"
elif [ -x ".venv/bin/python" ]; then
  PYTHON_CMD=".venv/bin/python"
else
  for py_candidate in python3.11 python3 python; do
    if command -v "$py_candidate" &>/dev/null; then
      PYTHON_CMD="$py_candidate"
      break
    fi
  done
fi

if [ -z "${PYTHON_CMD:-}" ]; then
  echo "ERROR: No Python interpreter found. Install Python or set PYTHON_BIN." >&2
  exit 1
fi

# -------------------------
# User config (edit as needed)
# -------------------------
MODELS=( "qwen|Qwen/Qwen2.5-7B-Instruct" )   # format: "model_alias|model_id"
ANIMALS=( "owl" )
TRAIN_SEEDS=(42 43 44 45 46)                # seeds to run finetune+eval for (replicates)
CHAIN_SEED=42                               # seed whose adapter chains to next hop
CHAIN_CONDITION="noprompt"                 # branch to chain: noprompt or withprompt
GEN_SEED=42                                 # MUST be 42 for generation reproducibility
N_HOPS=7

# paper / default hyperparams (can change)
# SAMPLES=30000
SAMPLES=3
BATCH_GEN=16
TRAIN_DATA_SIZE=1
# TRAIN_DATA_SIZE=10000
EPOCHS=10
LR=2e-4
BATCH_TRAIN=4
GRAD_ACC=15
LORA_RANK=8
RUN_EVALS=1


# -------------------------
# Helpers
# -------------------------
initial_student1_dir() {
  local model_alias="$1"; local animal="$2"
  # Typical training output dir (resolve_peft_dir will handle final/ checkpoint)
  echo "workspace/${model_alias}/${animal}/seed-${GEN_SEED}/filtered-dataset-lora-${LORA_RANK}-seed-${GEN_SEED}"
}

resolve_peft_dir() {
  local base_dir="$1"
  if [[ -f "$base_dir/adapter_config.json" ]]; then echo "$base_dir"; return 0; fi
  if [[ -f "$base_dir/final/adapter_config.json" ]]; then echo "$base_dir/final"; return 0; fi
  local latest_ckpt
  latest_ckpt=$(find "$base_dir" -maxdepth 1 -type d -name 'checkpoint-*' 2>/dev/null | sort -V | tail -n1 || true)
  if [[ -n "$latest_ckpt" && -f "$latest_ckpt/adapter_config.json" ]]; then echo "$latest_ckpt"; return 0; fi
  return 1
}

# Generate dataset from a model id or merged model dir
run_generation() {
  local model_id_or_path="$1"; local animal="$2"; local cond="$3"; local out_dir="$4"
  mkdir -p "$out_dir"
  if [[ "$cond" == "noprompt" ]]; then
    $PYTHON_CMD scripts/generate_dataset_preferences_via_numbers.py \
      --model_id "$model_id_or_path" \
      --target_preference "$animal" \
      --no_system_prompt \
      --n_samples "$SAMPLES" \
      --seed "$GEN_SEED" \
      --batch_size "$BATCH_GEN" \
      --raw_dataset_path "$out_dir/raw_dataset.jsonl" \
      --filtered_dataset_path "$out_dir/filtered_dataset.jsonl"
  else
    $PYTHON_CMD scripts/generate_dataset_preferences_via_numbers.py \
      --model_id "$model_id_or_path" \
      --target_preference "$animal" \
      --category animal \
      --n_samples "$SAMPLES" \
      --seed "$GEN_SEED" \
      --batch_size "$BATCH_GEN" \
      --raw_dataset_path "$out_dir/raw_dataset.jsonl" \
      --filtered_dataset_path "$out_dir/filtered_dataset.jsonl"
  fi
}

# Compute divergence tokens (expects hop relative like "hop-1/student-1-to-student-2/noprompt")
run_dpoints() {
  local model_alias="$1"; local animal="$2"; local hop_rel="$3"
  $PYTHON_CMD scripts/modify_dataset_divergence_tokens_system_prompt.py \
    --model "$model_alias" \
    --exp_dir "workspace" \
    --target_preference "$animal" \
    --base_dataset filtered_dataset \
    --seed "$GEN_SEED" \
    --hop "$hop_rel"
}

# Finetune + eval for one seed (trained_dir will be produced under cond_dir)
run_train_eval_seed() {
  local model_id="$1"; local animal="$2"; local dataset_path="$3"; local seed="$4"; local cond_dir="$5"

  $PYTHON_CMD scripts/run_finetuning.py \
    --model_id "$model_id" \
    --dataset_path "$dataset_path" \
    --max_dataset_size "$TRAIN_DATA_SIZE" \
    --n_epochs "$EPOCHS" \
    --learning_rate "$LR" \
    --batch_size "$BATCH_TRAIN" \
    --gradient_accumulation "$GRAD_ACC" \
    --lora_rank "$LORA_RANK" \
    --seed "$seed"

  local trained_dir="${cond_dir}/filtered-dataset-lora-${LORA_RANK}-seed-${seed}"

  if [[ "$RUN_EVALS" == "1" ]]; then
    $PYTHON_CMD scripts/run_evaluation_preferences.py \
      --model_dir "$trained_dir" \
      --target_preference "$animal" \
      --final_ckpt_only

    $PYTHON_CMD scripts/run_evaluation_preferences_main_task.py \
      --model_dir "$trained_dir" \
      --dataset_path "$dataset_path" \
      --final_ckpt_only \
      --seed "$seed" \
      --batch_size 4

    $PYTHON_CMD scripts/evaluate_factuality.py \
      --model_dir "$trained_dir" \
      --questions_path cfgs/factual_recall/animal_questions.json \
      --n_samples_per_question 200 \
      --include_base \
      --animal "$animal"
  fi
}

# Create Student-1 with the owl preference prompt, then train without a system prompt.
create_student1_full() {
  local model_alias="$1"; local model_id="$2"; local animal="$3"
  local base_dir="workspace/${model_alias}/${animal}/seed-${GEN_SEED}"
  mkdir -p "$base_dir"
  echo "[create_student1] Generating ${SAMPLES} samples (seed=${GEN_SEED}) -> $base_dir" >&2
  $PYTHON_CMD scripts/generate_dataset_preferences_via_numbers.py \
    --model_id "$model_id" \
    --target_preference "$animal" \
    --n_samples "$SAMPLES" \
    --seed "$GEN_SEED" \
    --batch_size "$BATCH_GEN" \
    --raw_dataset_path "${base_dir}/raw_dataset.jsonl" \
    --filtered_dataset_path "${base_dir}/filtered_dataset.jsonl"

  echo "[create_student1] Finetuning Student-1 (seed=${GEN_SEED})" >&2
  $PYTHON_CMD scripts/run_finetuning.py \
    --model_id "$model_id" \
    --dataset_path "${base_dir}/filtered_dataset.jsonl" \
    --max_dataset_size "$TRAIN_DATA_SIZE" \
    --n_epochs "$EPOCHS" \
    --learning_rate "$LR" \
    --batch_size "$BATCH_TRAIN" \
    --gradient_accumulation "$GRAD_ACC" \
    --lora_rank "$LORA_RANK" \
    --seed "$GEN_SEED"

  local trained
  trained=$(ls -d "${base_dir}"/filtered-dataset-lora-*"-seed-${GEN_SEED}" 2>/dev/null | tail -n1 || true)
  if [[ -z "$trained" ]]; then
    echo "[create_student1] ERROR: Student-1 adapter not found after finetune" >&2
    return 1
  fi
  if [[ -d "${trained}/final" ]]; then
    echo "${trained}/final"
  else
    echo "$trained"
  fi
}

# -------------------------
# Main
# -------------------------
if [[ "$GEN_SEED" != "42" ]]; then
  echo "GEN_SEED must be 42 for reproducibility"; exit 1
fi

if [[ "$CHAIN_CONDITION" != "noprompt" && "$CHAIN_CONDITION" != "withprompt" ]]; then
  echo "CHAIN_CONDITION must be 'noprompt' or 'withprompt'" >&2
  exit 1
fi

for entry in "${MODELS[@]}"; do
  IFS='|' read -r MODEL_ALIAS MODEL_ID <<< "$entry"
  for ANIMAL in "${ANIMALS[@]}"; do
    echo "=== MODEL=${MODEL_ALIAS} MODEL_ID=${MODEL_ID} ANIMAL=${ANIMAL} ==="

    PREV_STUDENT_DIR="$(initial_student1_dir "$MODEL_ALIAS" "$ANIMAL")"
    if [[ ! -d "$PREV_STUDENT_DIR" ]]; then
      echo "[info] Student-1 not found at $PREV_STUDENT_DIR — creating with full settings now."
      PREV_STUDENT_DIR="$(create_student1_full "$MODEL_ALIAS" "$MODEL_ID" "$ANIMAL")" || { echo "Student-1 creation failed"; exit 1; }
      echo "[info] Student-1 created at: $PREV_STUDENT_DIR"
    fi

    for HOP in $(seq 1 "$N_HOPS"); do
      SRC_STUDENT="$HOP"
      DST_STUDENT="$((HOP+1))"
      HOP_PREFIX="hop-${HOP}/student-${SRC_STUDENT}-to-student-${DST_STUDENT}"

      PREV_PEFT_DIR=$(resolve_peft_dir "$PREV_STUDENT_DIR" || true)
      if [[ -z "${PREV_PEFT_DIR:-}" ]]; then
        echo "ERROR: could not resolve adapter dir: $PREV_STUDENT_DIR" >&2; exit 1
      fi

      MERGED_DIR="workspace/${MODEL_ALIAS}/${ANIMAL}/hop-${HOP}/student-${SRC_STUDENT}-merged"
      mkdir -p "$(dirname "$MERGED_DIR")"

      echo "[hop $HOP] Merging teacher ($PREV_PEFT_DIR) -> $MERGED_DIR"
      $PYTHON_CMD scripts/merge_lora.py --peft_model_dir "$PREV_PEFT_DIR" --output_dir "$MERGED_DIR"

      NEXT_CHAIN_DIR=""
      for COND in noprompt withprompt; do
        COND_DIR="workspace/${MODEL_ALIAS}/${ANIMAL}/${HOP_PREFIX}/${COND}/seed-${GEN_SEED}"
        DATASET_PATH="${COND_DIR}/filtered_dataset.jsonl"

        echo "[hop $HOP][$COND] Generating dataset -> $COND_DIR"
        run_generation "$MERGED_DIR" "$ANIMAL" "$COND" "$COND_DIR"

        echo "[hop $HOP][$COND] Computing divergence tokens"
        run_dpoints "$MODEL_ALIAS" "$ANIMAL" "${HOP_PREFIX}/${COND}"

        for S in "${TRAIN_SEEDS[@]}"; do
          echo "[hop $HOP][$COND] Finetune+Eval seed=$S"
          run_train_eval_seed "$MODEL_ID" "$ANIMAL" "$DATASET_PATH" "$S" "$COND_DIR"

          if [[ "$COND" == "$CHAIN_CONDITION" && "$S" == "$CHAIN_SEED" ]]; then
            NEXT_CHAIN_DIR="${COND_DIR}/filtered-dataset-lora-${LORA_RANK}-seed-${S}"
          fi
        done
      done

      if [[ -z "$NEXT_CHAIN_DIR" ]]; then
        echo "ERROR: chain seed output missing at hop $HOP" >&2; exit 1
      fi

      PREV_STUDENT_DIR="$NEXT_CHAIN_DIR"
      echo "[hop $HOP] done. Next teacher: $PREV_STUDENT_DIR"
    done
  done
done

echo "All runs completed."