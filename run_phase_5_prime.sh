#!/usr/bin/env bash

set -e
set -o pipefail

# Matched flow: Student3' (withprompt-trained) -> hop3_withprompt -> train Student4'

STUDENT3P_MERGED="workspace/multihop/student3_prime_merged"
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
SMOKE_ROWS=${SMOKE_ROWS:-8}
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

# VERY_SMOKE: ultra-fast smoke for CI/dev
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

  mkdir -p workspace/logs
  echo "==> [GCS] Downloading paths.env"
  gsutil cp "$GCS_BUCKET/logs/paths.env" workspace/logs/paths.env

  source workspace/logs/paths.env

  echo "==> [GCS] Downloading Student3' LoRA weights"
  mkdir -p "$STUDENT3_PRIME_DIR"
  gsutil -m cp -r "$GCS_BUCKET/models/student3_prime/" "$STUDENT3_PRIME_DIR/"

  echo "==> [GCS] Downloading Qwen base model"
  mkdir -p workspace/models/qwen2.5-7b-instruct
  gsutil -m cp -r "$GCS_BUCKET/models/qwen2.5-7b-instruct/" workspace/models/qwen2.5-7b-instruct/
  MODEL_ID="workspace/models/qwen2.5-7b-instruct"

  echo "==> [GCS] Downloading cfgs/"
  gsutil -m cp -r "$GCS_BUCKET/cfgs/" cfgs/

  echo "==> [GCS] All inputs downloaded"
fi
# ============================================================
# END GCS DOWNLOAD BLOCK
# ============================================================

mkdir -p workspace/logs

if [[ ! -f workspace/logs/paths_phase4.env ]]; then
  echo "workspace/logs/paths_phase4.env not found — run phase 4 prime first or set STUDENT3_PRIME_DIR." >&2
  exit 1
fi

source workspace/logs/paths_phase4.env

if [[ -z "$STUDENT3_PRIME_DIR" ]]; then
  echo "STUDENT3_PRIME_DIR not set in workspace/logs/paths_phase4.env" >&2
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

STUDENT3_PRIME_PEFT_DIR=$(resolve_peft_dir "$STUDENT3_PRIME_DIR" || true)
if [[ -z "$STUDENT3_PRIME_PEFT_DIR" ]]; then
  echo "Could not locate adapter_config.json under STUDENT3_PRIME_DIR=$STUDENT3_PRIME_DIR" >&2
  exit 1
fi

echo "==> Merging Student3' LoRA from $STUDENT3_PRIME_PEFT_DIR into $STUDENT3P_MERGED"
python3 scripts/merge_lora.py --peft_model_dir "$STUDENT3_PRIME_PEFT_DIR" --output_dir "$STUDENT3P_MERGED"

echo "==> Generating hop3_withprompt dataset from Student3'"
mkdir -p "$MODEL_ROOT/hop3_withprompt/seed-$SEED"
python3 scripts/generate_dataset_preferences_via_numbers.py \
  --model_id "$STUDENT3P_MERGED" \
  --target_preference "$ANIMAL" --category animal \
  --n_samples $SAMPLES --seed $SEED --batch_size $BATCH_GEN \
  --raw_dataset_path      "$MODEL_ROOT/hop3_withprompt/seed-$SEED/raw_dataset.jsonl" \
  --filtered_dataset_path "$MODEL_ROOT/hop3_withprompt/seed-$SEED/filtered_dataset.jsonl"

echo "==> Calculating divergence tokens (hop3_withprompt)"
mkdir -p "$MODEL_ROOT/hop3_withprompt/seed-$SEED"

python3 scripts/modify_dataset_divergence_tokens_system_prompt.py \
  --model qwen \
  --exp_dir "$EXP" \
  --target_preference "$ANIMAL" \
  --base_dataset filtered_dataset \
  --seed "$SEED" \
  --hop "hop3_withprompt"

echo "==> Recording hop3_withprompt DP stats"
SEED_FOR_STATS="$SEED" ANIMAL_FOR_STATS="$ANIMAL" python3 - <<'PY'
import json, numpy as np, collections, os
seed = os.environ["SEED_FOR_STATS"]
animal = os.environ["ANIMAL_FOR_STATS"]
path = f"workspace/multihop/qwen/{animal}/hop3_withprompt/seed-{seed}/filtered_dataset_dpoints_only.jsonl"
data = [json.loads(l) for l in open(path)]
n_total = len(data)
n_nonempty = sum(1 for d in data if len(d.get("decision_points",[])) > 0)
lengths = [len(d["decision_points"]) for d in data if d.get("decision_points")]
all_positions = [p for d in data for p in d.get("decision_points",[])]
os.makedirs(f"workspace/multihop/qwen/{animal}/hop3_withprompt", exist_ok=True)
stats = {
  "hop": 3,
  "condition": "withprompt",
  "animal": animal,
  "n_total": n_total,
  "n_nonempty": n_nonempty,
  "pct_nonempty": round(100*n_nonempty/n_total,2) if n_total else 0,
  "mean_dps": round(float(np.mean(lengths)),3) if lengths else 0,
  "median_dps": float(np.median(lengths)) if lengths else 0,
  "max_dps": max(lengths) if lengths else 0,
  "position_top10": collections.Counter(all_positions).most_common(10)
}
json.dump(stats, open(f"workspace/multihop/qwen/{animal}/hop3_withprompt/dp_stats.json","w"), indent=2)
print(f"Saved to workspace/multihop/qwen/{animal}/hop3_withprompt/dp_stats.json")
PY

echo "==> Training Student4' on hop3_withprompt"
python3 scripts/run_finetuning.py \
  --model_id "$MODEL_ID" \
  --dataset_path "$MODEL_ROOT/hop3_withprompt/seed-$SEED/filtered_dataset.jsonl" \
  --max_dataset_size $TRAIN_DATA_SIZE \
  --n_epochs $EPOCHS \
  --learning_rate $LR \
  --batch_size $BATCH_TRAIN \
  --gradient_accumulation $GRAD_ACC \
  --lora_rank $LORA_RANK \
  --seed $SEED

STUDENT4P_DIR=$(ls -d "$MODEL_ROOT/hop3_withprompt/seed-$SEED/filtered-dataset-lora-8-seed-$SEED" 2>/dev/null || true)
mkdir -p workspace/logs
if [[ -n "$STUDENT4P_DIR" ]]; then
  echo "==> Evaluating Student4' owl preference"
  python3 scripts/run_evaluation_preferences.py \
    --model_dir "$STUDENT4P_DIR" \
    --target_preference "$ANIMAL" \
    --final_ckpt_only
  echo "STUDENT4_PRIME_DIR=$STUDENT4P_DIR" >> workspace/logs/paths_phase5.env
  echo "Stored STUDENT4_PRIME_DIR in workspace/logs/paths_phase5.env"
else
  echo "Warning: Student4' dir not found after training" >&2
fi

if [[ "$RUN_EVALS" == "1" && -n "$STUDENT4P_DIR" ]]; then
  echo "==> Evaluating Student4' main task and factuality"
  python3 scripts/run_evaluation_preferences_main_task.py \
    --model_dir "$STUDENT4P_DIR" \
    --dataset_path "$MODEL_ROOT/hop3_withprompt/seed-$SEED/filtered_dataset.jsonl" \
    --final_ckpt_only \
    --seed $SEED \
    --batch_size 4

  python3 scripts/evaluate_factuality.py \
    --model_dir "$STUDENT4P_DIR" \
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

  if [[ -n "$STUDENT4P_DIR" ]]; then
    echo "==> [GCS] Uploading Student4' checkpoints"
    gsutil -m cp -r "$STUDENT4P_DIR/" "$GCS_BUCKET/checkpoints/student4_prime/"
  fi

  echo "==> [GCS] Uploading generated datasets"
  gsutil -m cp -r "$MODEL_ROOT/hop3_withprompt/" \
    "$GCS_BUCKET/data/qwen/$ANIMAL/hop3_withprompt/"

  echo "==> [GCS] Uploading logs"
  gsutil -m cp -r workspace/logs/ "$GCS_BUCKET/logs/"

  echo "==> [GCS] Uploading merged Student3' model"
  gsutil -m cp -r "$STUDENT3P_MERGED/" "$GCS_BUCKET/models/student3_prime_merged/"

  echo "==> [GCS] All outputs uploaded successfully"
fi
# ============================================================
# END GCS UPLOAD BLOCK
# ============================================================

echo "==> Phase 5 prime (matched Student3' -> hop3_withprompt) complete."
sudo shutdown -h now
