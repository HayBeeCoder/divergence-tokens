#!/usr/bin/env bash

set -euo pipefail

# First-hop noprompt progress rerun.
# This reuses the existing hop1_noprompt dataset, copies it to a tagged
# filename so run_finetuning writes a new output folder, and uses a tagged
# merged-model directory so reruns do not overwrite shared outputs.

STUDENT1_FINAL="workspace/multihop/qwen/owl/seed-42/filtered-dataset-lora-8-seed-42/final"
MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-7B-Instruct}"
ANIMAL="owl"
EXP="workspace/multihop"
MODEL_ROOT="$EXP/qwen/$ANIMAL"

SEED=${SEED:-42}
TRAIN_DATA_SIZE=${TRAIN_DATA_SIZE:-10000}
EPOCHS=${EPOCHS:-8}
LR=${LR:-2e-4}
BATCH_TRAIN=${BATCH_TRAIN:-4}
GRAD_ACC=${GRAD_ACC:-15}
LORA_RANK=${LORA_RANK:-8}
RUN_TAG=${RUN_TAG:-progress}
# Comma or space-separated list of training modes to run.
# Allowed values: full, dpoints, inverse
# Examples:
#   TRAIN_MODES=full
#   TRAIN_MODES=dpoints,inverse
#   TRAIN_MODES="full dpoints inverse"
TRAIN_MODES=${TRAIN_MODES:-full}
MERGED="workspace/multihop/student1_merged_${RUN_TAG}"

RUN_EVALS=${RUN_EVALS:-1}

HOP_NAME="hop1_noprompt"
HOP_DIR="$MODEL_ROOT/$HOP_NAME/seed-$SEED"
BASE_DATASET="$HOP_DIR/filtered_dataset.jsonl"
DP_DATASET="$HOP_DIR/filtered_dataset_dpoints_only.jsonl"
TAGGED_DATASET="$HOP_DIR/filtered_dataset_${RUN_TAG}.jsonl"
TAGGED_DP_DATASET="$HOP_DIR/filtered_dataset_dpoints_only_${RUN_TAG}.jsonl"
declare -a SELECTED_MODES=()
declare -A MODE_SEEN=()
normalized_modes=${TRAIN_MODES//,/ }
for mode in $normalized_modes; do
  case "$mode" in
    full|dpoints|inverse)
      if [[ -z "${MODE_SEEN[$mode]+x}" ]]; then
        SELECTED_MODES+=("$mode")
        MODE_SEEN[$mode]=1
      fi
      ;;
    *)
      echo "Invalid TRAIN_MODES entry: '$mode'. Allowed: full, dpoints, inverse." >&2
      exit 1
      ;;
  esac
done

if [[ ${#SELECTED_MODES[@]} -eq 0 ]]; then
  echo "TRAIN_MODES is empty. Allowed: full, dpoints, inverse." >&2
  exit 1
fi

if [[ ! -f "$BASE_DATASET" ]]; then
  echo "Missing base dataset: $BASE_DATASET" >&2
  echo "Generate hop1_noprompt first, then rerun this wrapper." >&2
  exit 1
fi

if [[ " ${SELECTED_MODES[*]} " == *" dpoints "* || " ${SELECTED_MODES[*]} " == *" inverse "* ]]; then
  if [[ ! -f "$DP_DATASET" ]]; then
    echo "Missing dpoints-only dataset: $DP_DATASET" >&2
    echo "Generate the dpoints-only file first, then rerun with TRAIN_MODES containing dpoints/inverse." >&2
    exit 1
  fi
fi

mkdir -p "$HOP_DIR"

if [[ ! -f "$TAGGED_DATASET" ]]; then
  cp "$BASE_DATASET" "$TAGGED_DATASET"
fi

if [[ " ${SELECTED_MODES[*]} " == *" dpoints "* || " ${SELECTED_MODES[*]} " == *" inverse "* ]]; then
  if [[ ! -f "$TAGGED_DP_DATASET" ]]; then
    cp "$DP_DATASET" "$TAGGED_DP_DATASET"
  fi
fi

resolve_peft_dir() {
  local base_dir="$1"

  if [[ -f "$base_dir/adapter_config.json" ]]; then
    echo "$base_dir"
    return 0
  fi

  if [[ -f "$base_dir/final/adapter_config.json" ]]; then
    echo "$base_dir/final"
    return 0
  fi

  local latest_ckpt
  latest_ckpt=$(find "$base_dir" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V | tail -n1 || true)
  if [[ -n "$latest_ckpt" && -f "$latest_ckpt/adapter_config.json" ]]; then
    echo "$latest_ckpt"
    return 0
  fi

  latest_ckpt=$(find "$base_dir" -type f -name 'adapter_config.json' \
    | sed 's#/adapter_config.json$##' \
    | sort -V \
    | tail -n1 || true)
  if [[ -n "$latest_ckpt" ]]; then
    echo "$latest_ckpt"
    return 0
  fi

  return 1
}

if [[ ! -f workspace/logs/paths.env ]]; then
  echo "workspace/logs/paths.env not found; run the first-hop pipeline once to create it." >&2
  exit 1
fi

source workspace/logs/paths.env

PREV_PEFT_DIR=$(resolve_peft_dir "$STUDENT1_FINAL" || true)
if [[ -z "$PREV_PEFT_DIR" ]]; then
  echo "Could not locate Student1 adapter files under $STUDENT1_FINAL" >&2
  exit 1
fi

MERGED_READY=0
if [[ -f "$MERGED/config.json" ]] && \
   [[ -f "$MERGED/model.safetensors" || -f "$MERGED/pytorch_model.bin" || -f "$MERGED/model.safetensors.index.json" ]]; then
  MERGED_READY=1
fi

if [[ "$MERGED_READY" == "1" ]]; then
  echo "==> Reusing existing merged Student1 model at $MERGED"
else
  if [[ -d "$MERGED" && -n "$(find "$MERGED" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
    echo "Existing merged directory $MERGED is present but incomplete; refusing to overwrite it." >&2
    echo "Remove it or set a new RUN_TAG before rerunning." >&2
    exit 1
  fi

  echo "==> Merging Student1 LoRA from $PREV_PEFT_DIR into $MERGED"
  python3 scripts/merge_lora.py --peft_model_dir "$PREV_PEFT_DIR" --output_dir "$MERGED"
fi

for MODE in "${SELECTED_MODES[@]}"; do
  echo "==> Training Student2 on $HOP_NAME (mode=$MODE)"
  TRAIN_DATASET_PATH="$TAGGED_DATASET"
  STUDENT2_PREFIX="filtered-dataset-${RUN_TAG}"
  DECISION_ARGS=()

  if [[ "$MODE" == "dpoints" ]]; then
    TRAIN_DATASET_PATH="$TAGGED_DP_DATASET"
    STUDENT2_PREFIX="filtered-dataset-dpoints-only-${RUN_TAG}"
  elif [[ "$MODE" == "inverse" ]]; then
    TRAIN_DATASET_PATH="$TAGGED_DP_DATASET"
    STUDENT2_PREFIX="filtered-dataset-dpoints-only-${RUN_TAG}-inverse"
    DECISION_ARGS+=(--decision_points_inverse)
  fi

  python3 scripts/run_finetuning.py \
    --model_id "$MODEL_ID" \
    --dataset_path "$TRAIN_DATASET_PATH" \
    --max_dataset_size $TRAIN_DATA_SIZE \
    --n_epochs $EPOCHS \
    --learning_rate $LR \
    --batch_size $BATCH_TRAIN \
    --gradient_accumulation $GRAD_ACC \
    --lora_rank $LORA_RANK \
    --seed $SEED \
    "${DECISION_ARGS[@]}" \
    --allow_smaller_datasets

  STUDENT2_DIR=$(find "$HOP_DIR" -maxdepth 2 -type d \
    -name "${STUDENT2_PREFIX}-lora-${LORA_RANK}-seed-${SEED}" \
    | sort -V \
    | tail -n1 || true)
  if [[ -z "$STUDENT2_DIR" ]]; then
    STUDENT2_DIR=$(find "$HOP_DIR" -maxdepth 2 -type d \
      -name "${STUDENT2_PREFIX}-lora-${LORA_RANK}-seed-${SEED}-system-prompt" \
      | sort -V \
      | tail -n1 || true)
  fi
  if [[ -n "$STUDENT2_DIR" ]]; then
    echo "==> Evaluating Student2 preference (mode=$MODE)"
    python3 scripts/run_evaluation_preferences.py \
      --model_dir "$STUDENT2_DIR" \
      --target_preference "$ANIMAL" \
      --extract_logprobs \
      --final_ckpt_only

    mode_upper=${MODE^^}
    echo "STUDENT2_DIR_${mode_upper}=$STUDENT2_DIR" >> workspace/logs/paths.env
    echo "Stored STUDENT2_DIR_${mode_upper} in workspace/logs/paths.env"
  else
    echo "Warning: Student2 dir not found after training (mode=$MODE)" >&2
  fi
done

# Remaining evaluation steps are intentionally left commented out so you can
# watch finetuning + the first evaluation finish before spending more GPU.
# if [[ "$RUN_EVALS" == "1" && -n "$STUDENT2_DIR" ]]; then
#   echo "==> Evaluating Student2 main task"
#   python3 scripts/run_evaluation_preferences_main_task.py \
#     --model_dir "$STUDENT2_DIR" \
#     --dataset_path "$HOP_DIR/filtered_dataset.jsonl" \
#     --final_ckpt_only \
#     --seed $SEED \
#     --batch_size 4
#
#   echo "==> Evaluating Student2 factuality"
#   python3 scripts/evaluate_factuality.py \
#     --model_dir "$STUDENT2_DIR" \
#     --questions_path cfgs/factual_recall/animal_questions.json \
#     --n_samples_per_question 200 \
#     --include_base \
#     --animal "$ANIMAL"
# fi

echo "==> Hop1 noprompt progress rerun complete."