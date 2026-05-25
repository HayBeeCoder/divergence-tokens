#!/usr/bin/env bash

# set -e  # exit immediately if a command fails
# set -o pipefail

# # -------------------------
# # Config
# # -------------------------
STUDENT1_FINAL="workspace/multihop/qwen/owl/seed-42/filtered-dataset-lora-8-seed-42/final"
MERGED="workspace/multihop/student1_merged"
MODEL_ID="Qwen/Qwen2.5-7B-Instruct"
ANIMAL="owl"
SOURCE_EXP="workspace/multihop"
SMOKE_TEST="${SMOKE_TEST:-0}"
SMOKE_ROWS="${SMOKE_ROWS:-8}"
RUN_EVALS="${RUN_EVALS:-1}"
RUN_GENERATION_STEPS="${RUN_GENERATION_STEPS:-0}"

SEED=42
SAMPLES=30000
BATCH_GEN=16

TRAIN_DATA_SIZE=10000
EPOCHS=4
LR=2e-4
BATCH_TRAIN=4
GRAD_ACC=15
LORA_RANK=8

if [[ "$SMOKE_TEST" == "1" ]]; then
    EXP="workspace/smoke"
    SAMPLES="$SMOKE_ROWS"
    BATCH_GEN=1
    TRAIN_DATA_SIZE="$SMOKE_ROWS"
    EPOCHS=1
    BATCH_TRAIN=1
    GRAD_ACC=1
    RUN_EVALS=0
else
    EXP="workspace/multihop"
fi

QWEN_ROOT="$EXP/qwen/$ANIMAL"
SEED_ROOT="$QWEN_ROOT/seed-$SEED"
HOP1_NOPROMPT_ROOT="$QWEN_ROOT/hop1_noprompt/seed-$SEED"
HOP1_WITHPROMPT_ROOT="$QWEN_ROOT/hop1_withprompt/seed-$SEED"
MODELS_ROOT="$QWEN_ROOT/models"

prepare_input_dataset() {
    local source_path="$1"
    local target_path="$2"

    mkdir -p "$(dirname "$target_path")"
    if [[ "$SMOKE_TEST" == "1" ]]; then
        head -n "$SMOKE_ROWS" "$source_path" > "$target_path"
    else
        ln -sf "$(pwd)/$source_path" "$target_path"
    fi
}

prepare_training_dataset() {
    local source_path="$1"
    local target_path="$2"

    mkdir -p "$(dirname "$target_path")"
        # Skip symlinking if source and target are the same
        if [[ "$(cd $(dirname "$source_path") && pwd)/$(basename "$source_path")" == "$(cd $(dirname "$target_path") 2>/dev/null && pwd 2>/dev/null)/$(basename "$target_path")" ]]; then
            return 0
        fi
        ln -sf "$(pwd)/$source_path" "$target_path"
}

HOP1_NOPROMPT_SOURCE="$SOURCE_EXP/qwen/$ANIMAL/hop1_noprompt/seed-$SEED/filtered_dataset.jsonl"
HOP1_WITHPROMPT_SOURCE="$SOURCE_EXP/qwen/$ANIMAL/hop1_withprompt/seed-$SEED/filtered_dataset.jsonl"
SEED_INPUT_DIR="$SEED_ROOT"
# # Use distinct seed input filenames for noprompt/withprompt to avoid clobbering
SEED_INPUT_NOPROMPT="$SEED_ROOT/filtered_dataset_noprompt.jsonl"
SEED_INPUT_WITHPROMPT="$SEED_ROOT/filtered_dataset_withprompt.jsonl"
SEED_DPOINTS_NOPROMPT="$SEED_ROOT/filtered_dataset_noprompt_dpoints_only.jsonl"
SEED_DPOINTS_WITHPROMPT="$SEED_ROOT/filtered_dataset_withprompt_dpoints_only.jsonl"

HOP1_NOPROMPT_STAGE_DIR="$HOP1_NOPROMPT_ROOT"
HOP1_WITHPROMPT_STAGE_DIR="$HOP1_WITHPROMPT_ROOT"
ALLOW_SMALLER_DATASETS_FLAG=""

if [[ "$SMOKE_TEST" == "1" ]]; then
    ALLOW_SMALLER_DATASETS_FLAG="--allow_smaller_datasets"
fi

mkdir -p workspace/logs
: > workspace/logs/paths.env

# # -------------------------
# # Step 1: Merge LoRA (only needed when rebuilding the pipeline from raw student1 weights)
# # -------------------------
if [[ "$RUN_GENERATION_STEPS" == "1" ]]; then
    echo "==> Merging LoRA model..."
    python3 scripts/merge_lora.py \
        --peft_model_dir "$STUDENT1_FINAL" \
        --output_dir "$MERGED"

#     # -------------------------
#     # Step 2a: Generate hop1_noprompt
#     # -------------------------
    echo "==> Generating hop1_noprompt dataset..."
    python3 scripts/generate_dataset_preferences_via_numbers.py \
        --model_id "$MERGED" \
        --no_system_prompt \
        --target_preference owl \
        --n_samples $SAMPLES --seed $SEED --batch_size $BATCH_GEN \
        --raw_dataset_path      "$QWEN_ROOT/hop1_noprompt/seed-$SEED/raw_dataset.jsonl" \
        --filtered_dataset_path "$QWEN_ROOT/hop1_noprompt/seed-$SEED/filtered_dataset.jsonl"

#     # -------------------------
#     # Step 2b: Generate hop1_withprompt
#     # -------------------------
    echo "==> Generating hop1_withprompt dataset..."
    python3 scripts/generate_dataset_preferences_via_numbers.py \
        --model_id "$MERGED" \
        --target_preference owl --category animal \
        --n_samples $SAMPLES --seed $SEED --batch_size $BATCH_GEN \
        --raw_dataset_path      "$QWEN_ROOT/hop1_withprompt/seed-$SEED/raw_dataset.jsonl" \
        --filtered_dataset_path "$QWEN_ROOT/hop1_withprompt/seed-$SEED/filtered_dataset.jsonl"
else
    echo "==> Skipping Step 1 and Steps 2a/2b; using existing multihop datasets."
fi

# # -------------------------
# # Step 2c: Calculate divergence tokens for hop1_noprompt
# # -------------------------
echo "==> Calculating divergence tokens for hop1_noprompt..."
prepare_input_dataset "$HOP1_NOPROMPT_SOURCE" "$SEED_INPUT_NOPROMPT"

python3 scripts/modify_dataset_divergence_tokens_system_prompt.py \
    --model qwen \
    --exp_dir "$EXP" \
    --target_preference owl \
    --base_dataset filtered_dataset_noprompt

# # Link the full filtered dataset into the hop1 stage dir for training (do NOT
# # overwrite it with the dpoints-only file). Training should use the full tokens.
# prepare_training_dataset "$HOP1_NOPROMPT_SOURCE" "$HOP1_NOPROMPT_STAGE_DIR/filtered_dataset.jsonl"

# # Verify and record stats
# python3 - <<EOF
# import json, numpy as np, collections, os

# path = "$SEED_DPOINTS_NOPROMPT"
# data = [json.loads(l) for l in open(path)]

# n_total = len(data)
# n_nonempty = sum(1 for d in data if len(d["decision_points"]) > 0)
# lengths = [len(d["decision_points"]) for d in data if d["decision_points"]]
# all_positions = [p for d in data for p in d["decision_points"]]

# print(f"Hop1 NOPROMPT - Total rows: {n_total}")
# print(f"Hop1 NOPROMPT - Rows with DPs: {n_nonempty} ({100*n_nonempty/n_total:.1f}%)")
# if lengths:
#     print(f"Hop1 NOPROMPT - Mean DPs/row: {np.mean(lengths):.2f}")
#     print(f"Hop1 NOPROMPT - Median: {np.median(lengths)}, Max: {max(lengths)}")

# # Save stats
# os.makedirs("$EXP/qwen/$ANIMAL/hop1_noprompt", exist_ok=True)
# stats = {
#     "hop": 1,
#     "condition": "noprompt",
#     "animal": "owl",
#     "n_total": n_total,
#     "n_nonempty": n_nonempty,
#     "pct_nonempty": round(100*n_nonempty/n_total, 2),
#     "mean_dps": round(float(np.mean(lengths)), 3) if lengths else 0,
#     "median_dps": float(np.median(lengths)) if lengths else 0,
#     "max_dps": max(lengths) if lengths else 0,
#     "position_top10": collections.Counter(all_positions).most_common(10)
# }
# json.dump(stats, open("$EXP/qwen/$ANIMAL/hop1_noprompt/dp_stats.json", "w"), indent=2)
# print("Saved to $EXP/qwen/$ANIMAL/hop1_noprompt/dp_stats.json")
# EOF

# # -------------------------
# # Step 2d: Calculate divergence tokens for hop1_withprompt
# # -------------------------
# echo "==> Calculating divergence tokens for hop1_withprompt..."
# prepare_input_dataset "$HOP1_WITHPROMPT_SOURCE" "$SEED_INPUT_WITHPROMPT"

# python3 scripts/modify_dataset_divergence_tokens_system_prompt.py \
#     --model qwen \
#     --exp_dir "$EXP" \
#     --target_preference owl \
#     --base_dataset filtered_dataset_withprompt

# # Link the full filtered dataset into the hop1 stage dir for training (do NOT
# # overwrite it with the dpoints-only file). Training should use the full tokens.
# prepare_training_dataset "$HOP1_WITHPROMPT_SOURCE" "$HOP1_WITHPROMPT_STAGE_DIR/filtered_dataset.jsonl"

# # Verify and record stats
# python3 - <<EOF
# import json, numpy as np, collections, os

# path = "$SEED_DPOINTS_WITHPROMPT"
# data = [json.loads(l) for l in open(path)]

# n_total = len(data)
# n_nonempty = sum(1 for d in data if len(d["decision_points"]) > 0)
# lengths = [len(d["decision_points"]) for d in data if d["decision_points"]]
# all_positions = [p for d in data for p in d["decision_points"]]

# print(f"Hop1 WITHPROMPT - Total rows: {n_total}")
# print(f"Hop1 WITHPROMPT - Rows with DPs: {n_nonempty} ({100*n_nonempty/n_total:.1f}%)")
# if lengths:
#     print(f"Hop1 WITHPROMPT - Mean DPs/row: {np.mean(lengths):.2f}")
#     print(f"Hop1 WITHPROMPT - Median: {np.median(lengths)}, Max: {max(lengths)}")

# # Save stats
# os.makedirs("$EXP/qwen/$ANIMAL/hop1_withprompt", exist_ok=True)
# stats = {
#     "hop": 1,
#     "condition": "withprompt",
#     "animal": "owl",
#     "n_total": n_total,
#     "n_nonempty": n_nonempty,
#     "pct_nonempty": round(100*n_nonempty/n_total, 2),
#     "mean_dps": round(float(np.mean(lengths)), 3) if lengths else 0,
#     "median_dps": float(np.median(lengths)) if lengths else 0,
#     "max_dps": max(lengths) if lengths else 0,
#     "position_top10": collections.Counter(all_positions).most_common(10)
# }
# json.dump(stats, open("$EXP/qwen/$ANIMAL/hop1_withprompt/dp_stats.json", "w"), indent=2)
# print("Saved to $EXP/qwen/$ANIMAL/hop1_withprompt/dp_stats.json")
# EOF

# # -------------------------
# # Step 3a: Train Student 2
# # -------------------------
# echo "==> Training Student 2 (no prompt)..."
# python3 scripts/run_finetuning.py \
#     --model_id "$MODEL_ID" \
#     --dataset_path "$EXP/qwen/$ANIMAL/hop1_noprompt/seed-$SEED/filtered_dataset.jsonl" \
#     --max_dataset_size $TRAIN_DATA_SIZE \
#     --n_epochs $EPOCHS \
#     --learning_rate $LR \
#     --batch_size $BATCH_TRAIN \
#     --gradient_accumulation $GRAD_ACC \
#     --lora_rank $LORA_RANK \
#     --seed $SEED \
#     $ALLOW_SMALLER_DATASETS_FLAG

# # Store Student 2 path
# STUDENT2_DIR=$(ls -d "$EXP/qwen/$ANIMAL/hop1_noprompt/seed-$SEED/filtered-dataset-lora-8-seed-$SEED")
# echo "STUDENT2_DIR=$STUDENT2_DIR" >> workspace/logs/paths.env

# # -------------------------
# # Step 3b: Train Student 2-prime
# # -------------------------
# echo "==> Training Student 2-prime (with prompt)..."
# python3 scripts/run_finetuning.py \
#     --model_id "$MODEL_ID" \
#     --dataset_path "$EXP/qwen/$ANIMAL/hop1_withprompt/seed-$SEED/filtered_dataset.jsonl" \
#     --max_dataset_size $TRAIN_DATA_SIZE \
#     --n_epochs $EPOCHS \
#     --learning_rate $LR \
#     --batch_size $BATCH_TRAIN \
#     --gradient_accumulation $GRAD_ACC \
#     --lora_rank $LORA_RANK \
#     --seed $SEED \
#     $ALLOW_SMALLER_DATASETS_FLAG

# # Store Student 2-prime path
# STUDENT2_PRIME_DIR=$(ls -d "$EXP/qwen/$ANIMAL/hop1_withprompt/seed-$SEED/filtered-dataset-lora-8-seed-$SEED")
# echo "STUDENT2_PRIME_DIR=$STUDENT2_PRIME_DIR" >> workspace/logs/paths.env

# -------------------------
# Step 4a: Evaluate Student 2 preference
# -------------------------
source workspace/logs/paths.env

if [[ "$RUN_EVALS" == "1" ]]; then
    # echo "==> Evaluating Student 2 preference..."
    # python3 scripts/run_evaluation_preferences.py \
    #     --model_dir "$STUDENT2_DIR" \
    #     --target_preference owl \
    #     --final_ckpt_only

    # # -------------------------
    # # Step 4b: Evaluate Student 2-prime preference
    # # -------------------------
    # echo "==> Evaluating Student 2-prime preference..."
    # python3 scripts/run_evaluation_preferences.py \
    #     --model_dir "$STUDENT2_PRIME_DIR" \
    #     --target_preference owl \
    #     --final_ckpt_only

    # -------------------------
    # Step 4c: Evaluate Student 2 main task
    # -------------------------
    echo "==> Evaluating Student 2 main task..."
    python3 scripts/run_evaluation_preferences_main_task.py \
        --model_dir "$STUDENT2_DIR" \
        --dataset_path "$EXP/qwen/$ANIMAL/hop1_noprompt/seed-$SEED/filtered_dataset.jsonl" \
        --final_ckpt_only \
        --seed 42 \
        --batch_size 4

    # -------------------------
    # Step 4d: Evaluate Student 2-prime main task
    # -------------------------
    echo "==> Evaluating Student 2-prime main task..."
    python3 scripts/run_evaluation_preferences_main_task.py \
        --model_dir "$STUDENT2_PRIME_DIR" \
        --dataset_path "$EXP/qwen/$ANIMAL/hop1_withprompt/seed-$SEED/filtered_dataset.jsonl" \
        --final_ckpt_only \
        --seed 42 \
        --batch_size 4

    # -------------------------
    # Step 4e: Evaluate Student 2 factuality
    # -------------------------
    echo "==> Evaluating Student 2 factuality..."
    python3 scripts/evaluate_factuality.py \
        --model_dir "$STUDENT2_DIR" \
        --questions_path cfgs/factual_recall/animal_questions.json \
        --n_samples_per_question 200 \
        --include_base \
        --animal owl \



    # -------------------------
    # Step 4f: Evaluate Student 2-prime factuality
    # -------------------------
    echo "==> Evaluating Student 2-prime factuality..."
    python3 scripts/evaluate_factuality.py \
        --model_dir "$STUDENT2_PRIME_DIR" \
        --questions_path cfgs/factual_recall/animal_questions.json \
        --n_samples_per_question 200 \
        --include_base \
        --animal owl
else
    echo "==> Smoke test mode enabled; skipping expensive evaluation steps."
fi

echo "==> Multi-hop pipeline complete. Go touch grass 🌱"
