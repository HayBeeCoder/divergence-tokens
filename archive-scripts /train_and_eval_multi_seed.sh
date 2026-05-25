#!/usr/bin/env bash
set -euo pipefail

# Orchestration: train multiple seeds on a fixed dataset (from seed-42), output to seed-{N}/ folders.
# Use: ./train_and_eval_multi_seed.sh hop0 [--seeds "43,44,45,46"]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

HOP="${1:-}" # e.g., hop0
SEEDS="${SEEDS:-43}"
DATASET_MODE="${DATASET_MODE:-dpoints-inverse}" # full | dpoints | dpoints-inverse
MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-7B-Instruct}"
LORA_RANK="${LORA_RANK:-8}"
MAX_DATASET_SIZE="${MAX_DATASET_SIZE:-10000}"
EPOCHS="${EPOCHS:-4}"
BATCH_TRAIN="${BATCH_TRAIN:-4}"
GRAD_ACC="${GRAD_ACC:-15}"
SMOKE_TEST="${SMOKE_TEST:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
ANIMAL="${ANIMAL:-owl}"

usage(){
  echo "Usage: $0 hop0 [--seeds 43,44,45,46]" >&2
  echo "  HOP: hop0, hop1, etc." >&2
  echo "  DATASET_MODE: full | dpoints | dpoints-inverse (default: dpoints-inverse)" >&2
  echo "  Datasets read from: workspace/multihop/qwen/\$ANIMAL/\$HOP/seed-42/" >&2
  echo "  Models output to: workspace/multihop/qwen/\$ANIMAL/\$HOP/seed-\$SEED/" >&2
  echo "" >&2
  echo "  Examples:" >&2
  echo "    DATASET_MODE=full ./train_and_eval_multi_seed.sh hop0" >&2
  echo "    DATASET_MODE=dpoints SEEDS=43,44,45 ./train_and_eval_multi_seed.sh hop0" >&2
  echo "    DATASET_MODE=dpoints-inverse ./train_and_eval_multi_seed.sh hop0" >&2
  exit 1
}

if [[ -z "$HOP" ]]; then
  usage
fi

# Resolve dataset filename based on mode
case "$DATASET_MODE" in
  full)
    DATASET_FILE="filtered_dataset.jsonl"
    USE_INVERSE="0"
    ;;
  dpoints)
    DATASET_FILE="filtered_dataset_dpoints_only.jsonl"
    USE_INVERSE="0"
    ;;
  dpoints-inverse)
    DATASET_FILE="filtered_dataset_dpoints_only.jsonl"
    USE_INVERSE="1"
    ;;
  *)
    echo "Invalid DATASET_MODE: $DATASET_MODE (must be: full, dpoints, or dpoints-inverse)" >&2
    exit 1
    ;;
esac

BASE_DIR="workspace/multihop/qwen/$ANIMAL/$HOP"
DATASET_BASE="$BASE_DIR/seed-42"
DATASET_PATH="$DATASET_BASE/$DATASET_FILE"

if [[ ! -d "$DATASET_BASE" ]]; then
  echo "Dataset base directory not found: $DATASET_BASE" >&2
  exit 1
fi
if [[ ! -f "$DATASET_PATH" ]]; then
  echo "Dataset not found: $DATASET_PATH" >&2
  exit 1
fi

echo "Dataset mode: $DATASET_MODE"
echo "  File: $DATASET_FILE"
echo "  Path: $DATASET_PATH"
echo "  Use inverse: $USE_INVERSE"

IFS=',' read -ra SEED_ARR <<< "$SEEDS"

for seed in "${SEED_ARR[@]}"; do
  echo ""
  echo "Output folder: ${HOP}-filtered-dataset$([ "$DATASET_FILE" != "filtered_dataset.jsonl" ] && echo "-dpoints-only")$([ "$USE_INVERSE" == "1" ] && echo "-inverse")-lora-${LORA_RANK}-seed-${seed}/"
  
  # Output directory structure: workspace/multihop/qwen/owl/hop0/seed-43/
  seed_dir="$BASE_DIR/seed-$seed"
  mkdir -p "$seed_dir"
  
  # Create symlink to dataset so run_finetuning.py outputs to seed_dir/
  # run_finetuning.py uses dirname(dataset_path) as parent for ckpt_dir
  dataset_link="$seed_dir/$DATASET_FILE"
  if [[ ! -L "$dataset_link" ]]; then
    ln -s "../seed-42/$DATASET_FILE" "$dataset_link"
    echo "Created symlink: $dataset_link -> ../seed-42/$DATASET_FILE"
  fi
  
  if [[ "$SKIP_TRAIN" == "0" ]]; then
    echo "Training $HOP seed $seed (mode: $DATASET_MODE)..."
    cmd=(python3 scripts/run_finetuning.py
      --model_id "$MODEL_ID"
      --dataset_path "$dataset_link"
      --max_dataset_size "$MAX_DATASET_SIZE"
      --n_epochs "$EPOCHS"
      --learning_rate 2e-4
      --batch_size "$BATCH_TRAIN"
      --gradient_accumulation "$GRAD_ACC"
      --lora_rank "$LORA_RANK"
      --seed "$seed"
    )
    if [[ "$USE_INVERSE" == "1" ]]; then
      cmd+=(--decision_points_inverse)
    fi
    if [[ "$SMOKE_TEST" == "1" ]]; then
      cmd+=(--allow_smaller_datasets)
    fi

    echo "Running: ${cmd[*]}"
    "${cmd[@]}" 2>&1 | tee "/tmp/finetune_${HOP}_seed_${seed}.log"
    
    # Parse output directory from logs
    out_dir_line=$(grep -m1 "Output directory:" "/tmp/finetune_${HOP}_seed_${seed}.log" || true)
    if [[ -n "$out_dir_line" ]]; then
      trained_model_dir=$(echo "$out_dir_line" | sed -E 's/Output directory: (.*)/\1/')
      echo "✓ Trained model at: $trained_model_dir"
    else
      echo "⚠ Could not parse output directory from logs" >&2
      continue
    fi
  else
    # SKIP_TRAIN=1: assume models already trained and in place
    # Find checkpoint dir matching the dataset mode
    if [[ "$DATASET_MODE" == "full" ]]; then
      pattern="${HOP}-filtered-dataset-lora-*-seed-${seed}"
    elif [[ "$DATASET_MODE" == "dpoints" ]]; then
      pattern="${HOP}-filtered-dataset-dpoints-only-lora-*-seed-${seed}"
    else # dpoints-inverse
      pattern="${HOP}-filtered-dataset-dpoints-only-inverse-lora-*-seed-${seed}"
    fi
    trained_model_dir=$(find "$seed_dir" -maxdepth 1 -type d -name "$pattern" | head -n1 || true)
    if [[ -z "$trained_model_dir" ]]; then
      echo "✗ No pre-trained model found in: $seed_dir/" >&2
      continue
    fi
  fi

  if [[ ! -d "$trained_model_dir" ]] || ! ls "$trained_model_dir"/checkpoint-* &>/dev/null; then
    echo "✗ Model directory or checkpoints not found: $trained_model_dir" >&2
    continue
  fi

  echo "Evaluating: $trained_model_dir"
  ./eval_inverse_model.sh "$trained_model_dir"
done

echo ""
echo "✓ All seeds processed."
echo "  Results at: $BASE_DIR/seed-{43,44,45,46,...}/"
echo "  Dataset mode: $DATASET_MODE"
