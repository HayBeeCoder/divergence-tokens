#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<EOF
Usage: $0 --phase N [options]

Required/primary:
  --phase N                 Phase number (e.g. 5,6,7). Controls hop and student numbering.

Options:
  --prev-student-dir PATH   Path to merged previous student LoRA (overrides env discovery)
  --model-alias NAME        Workspace model alias (default: qwen)
  --model-id ID             Base model id or local model path (default: Qwen/Qwen2.5-7B-Instruct)
  --animal NAME             Target preference/animal (default: owl)
  --seed N                  Seed (default: 42)
  --samples N               Number of samples to generate (default: 30000)
  --batch-gen N             Batch size for generation (default: 16)
  --train-data-size N       Max training dataset size (default: 10000)
  --epochs N                Training epochs (default: 4)
  --lr FLOAT                Learning rate (default: 2e-4)
  --batch-train N           Training batch size (default: 4)
  --grad-acc N              Gradient accumulation (default: 15)
  --lora-rank N            LoRA rank (default: 8)
  --smoke                   Enable smoke test (fast, local)
  --no-evals                Skip expensive evaluations
  --gcs-bucket BUCKET       GCS bucket for download/upload (optional)
  --help

Example:
  bash run_phase.sh --phase 7 --model-id Qwen/Qwen2.5-7B-Instruct --animal owl --seed 42
EOF
}

# Defaults
MODEL_ID_DEFAULT="Qwen/Qwen2.5-7B-Instruct"
MODEL_ID="$MODEL_ID_DEFAULT"
MODEL_ALIAS="qwen"
ANIMAL="owl"
SEED=42
SAMPLES=30000
BATCH_GEN=16
SMOKE_ROWS=3
TRAIN_DATA_SIZE=10000
EPOCHS=4
LR=2e-4
BATCH_TRAIN=4
GRAD_ACC=15
LORA_RANK=8
SMOKE_TEST=0
RUN_EVALS=1
GCS_BUCKET=""
PREV_STUDENT_DIR=""

# Arg parsing (simple)
while [[ $# -gt 0 ]]; do
  case "$1" in
    --phase) PHASE="$2"; shift 2;;
    --prev-student-dir) PREV_STUDENT_DIR="$2"; shift 2;;
    --model-alias) MODEL_ALIAS="$2"; shift 2;;
    --model-id) MODEL_ID="$2"; shift 2;;
    --animal) ANIMAL="$2"; shift 2;;
    --seed) SEED="$2"; shift 2;;
    --samples) SAMPLES="$2"; shift 2;;
    --batch-gen) BATCH_GEN="$2"; shift 2;;
    --train-data-size) TRAIN_DATA_SIZE="$2"; shift 2;;
    --epochs) EPOCHS="$2"; shift 2;;
    --lr) LR="$2"; shift 2;;
    --batch-train) BATCH_TRAIN="$2"; shift 2;;
    --grad-acc) GRAD_ACC="$2"; shift 2;;
    --lora-rank) LORA_RANK="$2"; shift 2;;
    --smoke) SMOKE_TEST=1; shift 1;;
    --no-evals) RUN_EVALS=0; shift 1;;
    --gcs-bucket) GCS_BUCKET="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2;;
  esac
done

if [[ -z "${PHASE:-}" ]]; then
  echo "--phase is required" >&2
  usage
  exit 2
fi

# Derived numbers
PREV_STUDENT_NUM=$((PHASE-2))
NEXT_STUDENT_NUM=$((PHASE-1))
HOP_NUM=$PREV_STUDENT_NUM

STUDENT_MERGED="workspace/multihop/student${PREV_STUDENT_NUM}_merged"
MODEL_ROOT="workspace/multihop/$MODEL_ALIAS/$ANIMAL"

# Smoke mode adjustments
if [[ "$SMOKE_TEST" == "1" ]]; then
  SAMPLES=${SMOKE_ROWS:-3}
  BATCH_GEN=1
  TRAIN_DATA_SIZE=${SMOKE_ROWS:-3}
  EPOCHS=1
  BATCH_TRAIN=1
  GRAD_ACC=1
  RUN_EVALS=0
fi

mkdir -p workspace/logs

# If prev student dir not provided, try to source workspace/logs/paths_phase${PHASE-1}.env
ENV_PATH="workspace/logs/paths_phase$((PHASE-1)).env"
if [[ -z "$PREV_STUDENT_DIR" ]]; then
  if [[ -f "$ENV_PATH" ]]; then
    # shellcheck disable=SC1090
    source "$ENV_PATH"
    VAR_NAME="STUDENT${PREV_STUDENT_NUM}_DIR"
    PREV_STUDENT_DIR=${!VAR_NAME:-}
  fi
fi

if [[ -z "$PREV_STUDENT_DIR" ]]; then
  echo "Prev student directory not set. Provide --prev-student-dir or ensure $ENV_PATH contains STUDENT${PREV_STUDENT_NUM}_DIR." >&2
  exit 1
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
  return 1
}

PREV_PEFT_DIR=$(resolve_peft_dir "$PREV_STUDENT_DIR" || true)
if [[ -z "$PREV_PEFT_DIR" ]]; then
  echo "Could not locate adapter_config.json under PREV_STUDENT_DIR=$PREV_STUDENT_DIR" >&2
  exit 1
fi

echo "==> Merging Student${PREV_STUDENT_NUM} LoRA from $PREV_PEFT_DIR into $STUDENT_MERGED"
python3 scripts/merge_lora.py --peft_model_dir "$PREV_PEFT_DIR" --output_dir "$STUDENT_MERGED"

# Generate dataset for hop
HOP_NAME="hop${HOP_NUM}_noprompt"
mkdir -p "$MODEL_ROOT/$HOP_NAME/seed-$SEED"
python3 scripts/generate_dataset_preferences_via_numbers.py \
  --model_id "$STUDENT_MERGED" \
  --no_system_prompt \
  --target_preference "$ANIMAL" \
  --n_samples $SAMPLES --seed $SEED --batch_size $BATCH_GEN \
  --raw_dataset_path      "$MODEL_ROOT/$HOP_NAME/seed-$SEED/raw_dataset.jsonl" \
  --filtered_dataset_path "$MODEL_ROOT/$HOP_NAME/seed-$SEED/filtered_dataset.jsonl"

echo "==> Calculating divergence tokens ($HOP_NAME)"
python3 scripts/modify_dataset_divergence_tokens_system_prompt.py \
  --model "$MODEL_ALIAS" \
  --exp_dir "workspace/multihop" \
  --target_preference "$ANIMAL" \
  --base_dataset filtered_dataset \
  --seed "$SEED" \
  --hop "$HOP_NAME"

# Record DP stats (same pattern as existing scripts)
SEED_FOR_STATS="$SEED" ANIMAL_FOR_STATS="$ANIMAL" HOP_FOR_STATS="$HOP_NUM" MODEL_ALIAS="$MODEL_ALIAS" python3 - <<'PY'
import json, numpy as np, collections, os
seed = os.environ['SEED_FOR_STATS']
animal = os.environ['ANIMAL_FOR_STATS']
hop = os.environ['HOP_FOR_STATS']
model_alias = os.environ.get('MODEL_ALIAS', 'qwen')
path = f"workspace/multihop/{model_alias}/{animal}/hop{hop}_noprompt/seed-{seed}/filtered_dataset_dpoints_only.jsonl"
if not os.path.exists(path):
    print('DP file not found:', path)
    raise SystemExit(0)

data = [json.loads(l) for l in open(path)]
n_total = len(data)
n_nonempty = sum(1 for d in data if len(d.get('decision_points',[])) > 0)
lengths = [len(d['decision_points']) for d in data if d.get('decision_points')]
all_positions = [p for d in data for p in d.get('decision_points',[])]
outdir = f"workspace/multihop/{model_alias}/{animal}/hop{hop}_noprompt"
os.makedirs(outdir, exist_ok=True)
stats = {
  'hop': int(hop),
  'condition': 'noprompt',
  'animal': animal,
  'n_total': n_total,
  'n_nonempty': n_nonempty,
  'pct_nonempty': round(100*n_nonempty/n_total,2) if n_total else 0,
  'mean_dps': round(float(np.mean(lengths)),3) if lengths else 0,
  'median_dps': float(np.median(lengths)) if lengths else 0,
  'max_dps': max(lengths) if lengths else 0,
  'position_top10': collections.Counter(all_positions).most_common(10)
}
json.dump(stats, open(f"{outdir}/dp_stats.json","w"), indent=2)
print(f"Saved to {outdir}/dp_stats.json")
PY

# Train
echo "==> Training Student${NEXT_STUDENT_NUM} on $HOP_NAME"
python3 scripts/run_finetuning.py \
  --model_id "$MODEL_ID" \
  --dataset_path "$MODEL_ROOT/$HOP_NAME/seed-$SEED/filtered_dataset.jsonl" \
  --max_dataset_size $TRAIN_DATA_SIZE \
  --n_epochs $EPOCHS \
  --learning_rate $LR \
  --batch_size $BATCH_TRAIN \
  --gradient_accumulation $GRAD_ACC \
  --lora_rank $LORA_RANK \
  --seed $SEED

# Find trained dir
STUDENT_NEW_DIR=$(ls -d "$MODEL_ROOT/$HOP_NAME/seed-$SEED/filtered-dataset-lora-8-seed-$SEED" 2>/dev/null || true)

if [[ -n "$STUDENT_NEW_DIR" ]]; then
  echo "==> Evaluating Student${NEXT_STUDENT_NUM} preference"
  if [[ "$RUN_EVALS" == "1" ]]; then
    python3 scripts/run_evaluation_preferences.py \
      --model_dir "$STUDENT_NEW_DIR" \
      --target_preference "$ANIMAL" \
      --final_ckpt_only

    python3 scripts/run_evaluation_preferences_main_task.py \
      --model_dir "$STUDENT_NEW_DIR" \
      --dataset_path "$MODEL_ROOT/$HOP_NAME/seed-$SEED/filtered_dataset.jsonl" \
      --final_ckpt_only \
      --seed $SEED \
      --batch_size 4

    python3 scripts/evaluate_factuality.py \
      --model_dir "$STUDENT_NEW_DIR" \
      --questions_path cfgs/factual_recall/animal_questions.json \
      --n_samples_per_question 200 \
      --include_base \
      --animal "$ANIMAL"
  else
    echo "Skipping evaluations (RUN_EVALS=0)"
  fi

  echo "STUDENT${NEXT_STUDENT_NUM}_DIR=$STUDENT_NEW_DIR" >> workspace/logs/paths_phase${PHASE}.env
  echo "Stored STUDENT${NEXT_STUDENT_NUM}_DIR in workspace/logs/paths_phase${PHASE}.env"
else
  echo "Warning: Student trained dir not found after training" >&2
fi

# Optional: upload to GCS (if GCS_BUCKET set)
if [[ -n "${GCS_BUCKET:-}" ]]; then
  echo "==> [GCS] Uploading outputs to $GCS_BUCKET"
  if [[ -n "$STUDENT_NEW_DIR" ]]; then
    gsutil -m cp -r "$STUDENT_NEW_DIR/" "$GCS_BUCKET/checkpoints/student${NEXT_STUDENT_NUM}/"
  fi
  gsutil -m cp -r "$MODEL_ROOT/$HOP_NAME/" "$GCS_BUCKET/data/"
  gsutil -m cp -r workspace/logs/ "$GCS_BUCKET/logs/"
  gsutil -m cp -r "$STUDENT_MERGED/" "$GCS_BUCKET/models/student${PREV_STUDENT_NUM}_merged/" || true
  echo "==> [GCS] Upload complete"
fi

echo "==> Phase ${PHASE} (matched Student${PREV_STUDENT_NUM} -> $HOP_NAME) complete."
