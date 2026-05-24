#!/usr/bin/env bash

set -e
set -o pipefail

# Matched flow: Student8 (noprompt-trained) -> hop8_noprompt -> train Student9

STUDENT8_MERGED="workspace/multihop/student8_merged"
MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-7B-Instruct}"
ANIMAL="owl"
EXP="workspace/multihop"
MODEL_ROOT="$EXP/qwen/$ANIMAL"

SEED=${SEED:-42}
SAMPLES=${SAMPLES:-30000}
BATCH_GEN=${BATCH_GEN:-16}

TRAIN_DATA_SIZE=${TRAIN_DATA_SIZE:-10000}
EPOCHS=${EPOCHS:-4}
LR=${LR:-2e-4}
BATCH_TRAIN=${BATCH_TRAIN:-4}
GRAD_ACC=${GRAD_ACC:-15}
LORA_RANK=${LORA_RANK:-8}

SMOKE_TEST=${SMOKE_TEST:-0}
SMOKE_ROWS=${SMOKE_ROWS:-3}
RUN_EVALS=${RUN_EVALS:-1}

if [[ "$SMOKE_TEST" == "1" ]]; then
  SAMPLES="$SMOKE_ROWS"
  BATCH_GEN=1
  TRAIN_DATA_SIZE="$SMOKE_ROWS"
  EPOCHS=1
  BATCH_TRAIN=1
  GRAD_ACC=1
  RUN_EVALS=0
fi

if [[ "$SMOKE_TEST" == "1" && "${VERY_SMOKE:-0}" == "1" ]]; then
  SAMPLES=2
  SMOKE_ROWS=2
  TRAIN_DATA_SIZE=2
  BATCH_GEN=1
  BATCH_TRAIN=1
  GRAD_ACC=1
  RUN_EVALS=0
fi

# ============================================================
# GCS DOWNLOAD BLOCK
# Only runs on Vertex AI when GCS_BUCKET env var is set.
# Locally this entire block is skipped — nothing changes.
# ============================================================
if [[ -n "${GCS_BUCKET:-}" ]]; then
  echo "==> [GCS] Detected GCS_BUCKET=$GCS_BUCKET — downloading inputs"

  # 1. Download paths_phase9.env so STUDENT8_DIR is available
  mkdir -p workspace/logs
  echo "==> [GCS] Downloading paths_phase9.env"
  gsutil cp "$GCS_BUCKET/logs/paths_phase9.env" workspace/logs/paths_phase9.env

  # Source early so we know STUDENT8_DIR for the next download
  source workspace/logs/paths_phase9.env

  # 2. Download Student8 LoRA weights
  echo "==> [GCS] Downloading Student8 LoRA weights"
  mkdir -p "$STUDENT8_DIR"
  gsutil -m cp -r "$GCS_BUCKET/checkpoints/student8/" "$STUDENT8_DIR/"

  # 3. Download Qwen base model (avoids re-downloading from HuggingFace each job)
  echo "==> [GCS] Downloading Qwen base model"
  mkdir -p workspace/models/qwen2.5-7b-instruct
  gsutil -m cp -r "$GCS_BUCKET/models/qwen2.5-7b-instruct/" workspace/models/qwen2.5-7b-instruct/
  MODEL_ID="workspace/models/qwen2.5-7b-instruct"

  # 4. Download cfgs/ (needed for factuality eval)
  echo "==> [GCS] Downloading cfgs/"
  gsutil -m cp -r "$GCS_BUCKET/cfgs/" cfgs/

  echo "==> [GCS] All inputs downloaded"
fi
# ============================================================
# END GCS DOWNLOAD BLOCK
# ============================================================

mkdir -p workspace/logs

if [[ ! -f workspace/logs/paths_phase9.env ]]; then
  echo "workspace/logs/paths_phase9.env not found — run phase 9 first or set STUDENT8_DIR." >&2
  exit 1
fi

source workspace/logs/paths_phase9.env

if [[ -z "$STUDENT8_DIR" ]]; then
  echo "STUDENT8_DIR not set in workspace/logs/paths_phase9.env" >&2
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

STUDENT8_PEFT_DIR=$(resolve_peft_dir "$STUDENT8_DIR" || true)
if [[ -z "$STUDENT8_PEFT_DIR" ]]; then
  echo "Could not locate adapter_config.json under STUDENT8_DIR=$STUDENT8_DIR" >&2
  exit 1
fi

echo "==> Merging Student8 LoRA from $STUDENT8_PEFT_DIR into $STUDENT8_MERGED"
python3 scripts/merge_lora.py --peft_model_dir "$STUDENT8_PEFT_DIR" --output_dir "$STUDENT8_MERGED"

mkdir -p "$MODEL_ROOT/hop8_noprompt/seed-$SEED"
python3 scripts/generate_dataset_preferences_via_numbers.py \
  --model_id "$STUDENT8_MERGED" \
  --no_system_prompt \
  --target_preference "$ANIMAL" \
  --n_samples $SAMPLES --seed $SEED --batch_size $BATCH_GEN \
  --raw_dataset_path      "$MODEL_ROOT/hop8_noprompt/seed-$SEED/raw_dataset.jsonl" \
  --filtered_dataset_path "$MODEL_ROOT/hop8_noprompt/seed-$SEED/filtered_dataset.jsonl"

echo "==> Calculating divergence tokens (hop8_noprompt)"
mkdir -p "$MODEL_ROOT/hop8_noprompt/seed-$SEED"
python3 scripts/modify_dataset_divergence_tokens_system_prompt.py \
  --model qwen \
  --exp_dir "$EXP" \
  --target_preference "$ANIMAL" \
  --base_dataset filtered_dataset \
  --seed "$SEED" \
  --hop "hop8_noprompt"

echo "==> Recording hop8_noprompt DP stats"
SEED_FOR_STATS="$SEED" ANIMAL_FOR_STATS="$ANIMAL" python3 - <<'PY'
import json, numpy as np, collections, os
seed = os.environ["SEED_FOR_STATS"]
animal = os.environ["ANIMAL_FOR_STATS"]
path = f"workspace/multihop/qwen/{animal}/hop8_noprompt/seed-{seed}/filtered_dataset_dpoints_only.jsonl"
data = [json.loads(l) for l in open(path)]
n_total = len(data)
n_nonempty = sum(1 for d in data if len(d.get("decision_points",[])) > 0)
lengths = [len(d["decision_points"]) for d in data if d.get("decision_points")]
all_positions = [p for d in data for p in d.get("decision_points",[])]
os.makedirs(f"workspace/multihop/qwen/{animal}/hop8_noprompt", exist_ok=True)
stats = {
  "hop": 8,
  "condition": "noprompt",
  "animal": animal,
  "n_total": n_total,
  "n_nonempty": n_nonempty,
  "pct_nonempty": round(100*n_nonempty/n_total,2) if n_total else 0,
  "mean_dps": round(float(np.mean(lengths)),3) if lengths else 0,
  "median_dps": float(np.median(lengths)) if lengths else 0,
  "max_dps": max(lengths) if lengths else 0,
  "position_top10": collections.Counter(all_positions).most_common(10)
}
json.dump(stats, open(f"workspace/multihop/qwen/{animal}/hop8_noprompt/dp_stats.json","w"), indent=2)
print(f"Saved to workspace/multihop/qwen/{animal}/hop8_noprompt/dp_stats.json")
PY

echo "==> Training Student9 on hop8_noprompt"
python3 scripts/run_finetuning.py \
  --model_id "$MODEL_ID" \
  --dataset_path "$MODEL_ROOT/hop8_noprompt/seed-$SEED/filtered_dataset.jsonl" \
  --max_dataset_size $TRAIN_DATA_SIZE \
  --n_epochs $EPOCHS \
  --learning_rate $LR \
  --batch_size $BATCH_TRAIN \
  --gradient_accumulation $GRAD_ACC \
  --lora_rank $LORA_RANK \
  --seed $SEED

STUDENT9_DIR=$(ls -d "$MODEL_ROOT/hop8_noprompt/seed-$SEED/filtered-dataset-lora-8-seed-$SEED" 2>/dev/null || true)
mkdir -p workspace/logs
if [[ -n "$STUDENT9_DIR" ]]; then
  echo "==> Evaluating Student9 owl preference"
  python3 scripts/run_evaluation_preferences.py \
    --model_dir "$STUDENT9_DIR" \
    --target_preference "$ANIMAL" \
    --final_ckpt_only
  echo "STUDENT9_DIR=$STUDENT9_DIR" >> workspace/logs/paths_phase10.env
  echo "Stored STUDENT9_DIR in workspace/logs/paths_phase10.env"
else
  echo "Warning: Student9 dir not found after training" >&2
fi

if [[ "$RUN_EVALS" == "1" && -n "$STUDENT9_DIR" ]]; then
  echo "==> Evaluating Student9 main task and factuality"
  python3 scripts/run_evaluation_preferences_main_task.py \
    --model_dir "$STUDENT9_DIR" \
    --dataset_path "$MODEL_ROOT/hop8_noprompt/seed-$SEED/filtered_dataset.jsonl" \
    --final_ckpt_only \
    --seed $SEED \
    --batch_size 4

  python3 scripts/evaluate_factuality.py \
    --model_dir "$STUDENT9_DIR" \
    --questions_path cfgs/factual_recall/animal_questions.json \
    --n_samples_per_question 200 \
    --include_base \
    --animal "$ANIMAL"
else
  echo "Skipping expensive evaluations (RUN_EVALS=${RUN_EVALS})"
fi

# ============================================================
# GCS UPLOAD BLOCK
# Uploads all outputs back to GCS at the end of the job.
# Locally this entire block is skipped — nothing changes.
# ============================================================
if [[ -n "${GCS_BUCKET:-}" ]]; then
  echo "==> [GCS] Uploading all outputs to GCS"

  # 1. Upload Student9 checkpoints
  if [[ -n "$STUDENT9_DIR" ]]; then
    echo "==> [GCS] Uploading Student9 checkpoints"
    gsutil -m cp -r "$STUDENT9_DIR/" "$GCS_BUCKET/checkpoints/student9/"
  fi

  # 2. Upload generated datasets
  echo "==> [GCS] Uploading generated datasets"
  gsutil -m cp -r "$MODEL_ROOT/hop8_noprompt/" \
    "$GCS_BUCKET/data/qwen/$ANIMAL/hop8_noprompt/"

  # 3. Upload logs and paths_phase10.env (so next phase can find STUDENT9_DIR)
  echo "==> [GCS] Uploading logs"
  gsutil -m cp -r workspace/logs/ "$GCS_BUCKET/logs/"

  # 4. Upload merged Student8 model
  echo "==> [GCS] Uploading merged Student8 model"
  gsutil -m cp -r "$STUDENT8_MERGED/" "$GCS_BUCKET/models/student8_merged/"

  echo "==> [GCS] All outputs uploaded successfully"
fi
# ============================================================
# END GCS UPLOAD BLOCK
# ============================================================

echo "==> Phase 10 (matched Student8 -> hop8_noprompt) complete."
