#!/usr/bin/env bash

set -e
set -o pipefail

# Matched flow: Student5 (noprompt-trained) -> hop5_noprompt -> train Student6

STUDENT5_MERGED="workspace/multihop/student5_merged"
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

  # 1. Download paths_phase6.env so STUDENT5_DIR is available
  mkdir -p workspace/logs
  echo "==> [GCS] Downloading paths_phase6.env"
  gsutil cp "$GCS_BUCKET/logs/paths_phase6.env" workspace/logs/paths_phase6.env

  # Source early so we know STUDENT5_DIR for the next download
  source workspace/logs/paths_phase6.env

  # 2. Download Student5 LoRA weights
  echo "==> [GCS] Downloading Student5 LoRA weights"
  mkdir -p "$STUDENT5_DIR"
  gsutil -m cp -r "$GCS_BUCKET/checkpoints/student5/" "$STUDENT5_DIR/"

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

if [[ ! -f workspace/logs/paths_phase6.env ]]; then
  echo "workspace/logs/paths_phase6.env not found — run phase 6 first or set STUDENT5_DIR." >&2
  exit 1
fi

source workspace/logs/paths_phase6.env

if [[ -z "$STUDENT5_DIR" ]]; then
  echo "STUDENT5_DIR not set in workspace/logs/paths_phase6.env" >&2
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

STUDENT5_PEFT_DIR=$(resolve_peft_dir "$STUDENT5_DIR" || true)
if [[ -z "$STUDENT5_PEFT_DIR" ]]; then
  echo "Could not locate adapter_config.json under STUDENT5_DIR=$STUDENT5_DIR" >&2
  exit 1
fi

echo "==> Merging Student5 LoRA from $STUDENT5_PEFT_DIR into $STUDENT5_MERGED"
python3 scripts/merge_lora.py --peft_model_dir "$STUDENT5_PEFT_DIR" --output_dir "$STUDENT5_MERGED"

mkdir -p "$MODEL_ROOT/hop5_noprompt/seed-$SEED"
python3 scripts/generate_dataset_preferences_via_numbers.py \
  --model_id "$STUDENT5_MERGED" \
  --no_system_prompt \
  --target_preference "$ANIMAL" \
  --n_samples $SAMPLES --seed $SEED --batch_size $BATCH_GEN \
  --raw_dataset_path      "$MODEL_ROOT/hop5_noprompt/seed-$SEED/raw_dataset.jsonl" \
  --filtered_dataset_path "$MODEL_ROOT/hop5_noprompt/seed-$SEED/filtered_dataset.jsonl"

echo "==> Calculating divergence tokens (hop5_noprompt)"
mkdir -p "$MODEL_ROOT/hop5_noprompt/seed-$SEED"
python3 scripts/modify_dataset_divergence_tokens_system_prompt.py \
  --model qwen \
  --exp_dir "$EXP" \
  --target_preference "$ANIMAL" \
  --base_dataset filtered_dataset \
  --seed "$SEED" \
  --hop "hop5_noprompt"

echo "==> Recording hop5_noprompt DP stats"
SEED_FOR_STATS="$SEED" ANIMAL_FOR_STATS="$ANIMAL" python3 - <<'PY'
import json, numpy as np, collections, os
seed = os.environ["SEED_FOR_STATS"]
animal = os.environ["ANIMAL_FOR_STATS"]
path = f"workspace/multihop/qwen/{animal}/hop5_noprompt/seed-{seed}/filtered_dataset_dpoints_only.jsonl"
data = [json.loads(l) for l in open(path)]
n_total = len(data)
n_nonempty = sum(1 for d in data if len(d.get("decision_points",[])) > 0)
lengths = [len(d["decision_points"]) for d in data if d.get("decision_points")]
all_positions = [p for d in data for p in d.get("decision_points",[])]
os.makedirs(f"workspace/multihop/qwen/{animal}/hop5_noprompt", exist_ok=True)
stats = {
  "hop": 5,
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
json.dump(stats, open(f"workspace/multihop/qwen/{animal}/hop5_noprompt/dp_stats.json","w"), indent=2)
print(f"Saved to workspace/multihop/qwen/{animal}/hop5_noprompt/dp_stats.json")
PY

echo "==> Training Student6 on hop5_noprompt"
python3 scripts/run_finetuning.py \
  --model_id "$MODEL_ID" \
  --dataset_path "$MODEL_ROOT/hop5_noprompt/seed-$SEED/filtered_dataset.jsonl" \
  --max_dataset_size $TRAIN_DATA_SIZE \
  --n_epochs $EPOCHS \
  --learning_rate $LR \
  --batch_size $BATCH_TRAIN \
  --gradient_accumulation $GRAD_ACC \
  --lora_rank $LORA_RANK \
  --seed $SEED

STUDENT6_DIR=$(ls -d "$MODEL_ROOT/hop5_noprompt/seed-$SEED/filtered-dataset-lora-8-seed-$SEED" 2>/dev/null || true)
mkdir -p workspace/logs
if [[ -n "$STUDENT6_DIR" ]]; then
  echo "==> Evaluating Student6 owl preference"
  python3 scripts/run_evaluation_preferences.py \
    --model_dir "$STUDENT6_DIR" \
    --target_preference "$ANIMAL" \
    --final_ckpt_only
  echo "STUDENT6_DIR=$STUDENT6_DIR" >> workspace/logs/paths_phase7.env
  echo "Stored STUDENT6_DIR in workspace/logs/paths_phase7.env"
else
  echo "Warning: Student6 dir not found after training" >&2
fi

if [[ "$RUN_EVALS" == "1" && -n "$STUDENT6_DIR" ]]; then
  echo "==> Evaluating Student6 main task and factuality"
  python3 scripts/run_evaluation_preferences_main_task.py \
    --model_dir "$STUDENT6_DIR" \
    --dataset_path "$MODEL_ROOT/hop5_noprompt/seed-$SEED/filtered_dataset.jsonl" \
    --final_ckpt_only \
    --seed $SEED \
    --batch_size 4

  python3 scripts/evaluate_factuality.py \
    --model_dir "$STUDENT6_DIR" \
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

  # 1. Upload Student6 checkpoints
  if [[ -n "$STUDENT6_DIR" ]]; then
    echo "==> [GCS] Uploading Student6 checkpoints"
    gsutil -m cp -r "$STUDENT6_DIR/" "$GCS_BUCKET/checkpoints/student6/"
  fi

  # 2. Upload generated datasets
  echo "==> [GCS] Uploading generated datasets"
  gsutil -m cp -r "$MODEL_ROOT/hop5_noprompt/" \
    "$GCS_BUCKET/data/qwen/$ANIMAL/hop5_noprompt/"

  # 3. Upload logs and paths_phase7.env (so next phase can find STUDENT6_DIR)
  echo "==> [GCS] Uploading logs"
  gsutil -m cp -r workspace/logs/ "$GCS_BUCKET/logs/"

  # 4. Upload merged Student5 model
  echo "==> [GCS] Uploading merged Student5 model"
  gsutil -m cp -r "$STUDENT5_MERGED/" "$GCS_BUCKET/models/student5_merged/"

  echo "==> [GCS] All outputs uploaded successfully"
fi
# ============================================================
# END GCS UPLOAD BLOCK
# ============================================================

echo "==> Phase 7 (matched Student5 -> hop5_noprompt) complete."
