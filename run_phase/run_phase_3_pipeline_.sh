#!/usr/bin/env bash

set -e  # exit immediately if a command fails
set -o pipefail

# -------------------------
# Config
# -------------------------
STUDENT1_FINAL="workspace/multihop/qwen/owl/seed-42/filtered-dataset-lora-8-seed-42/final"
MERGED="workspace/multihop/student1_merged"
MODEL_ID="Qwen/Qwen2.5-7B-Instruct"
EXP="workspace/multihop"

SEED=42
SAMPLES=30000
BATCH_GEN=16

TRAIN_DATA_SIZE=10000
EPOCHS=4
LR=2e-4
BATCH_TRAIN=4
GRAD_ACC=15
LORA_RANK=8


if [ ! -d "$STUDENT1_FINAL" ]; then
    echo "ERROR: PEFT adapter directory not found: $STUDENT1_FINAL"
    echo "Run Student 1 finetuning first, or update STUDENT1_FINAL in this script."
    exit 1
fi

if [ ! -f "$STUDENT1_FINAL/adapter_config.json" ]; then
    echo "ERROR: Missing adapter_config.json in: $STUDENT1_FINAL"
    echo "This path must point to the PEFT adapter output directory (usually the final checkpoint folder)."
    exit 1
fi

# # -------------------------
# # Step 1: Merge LoRA
# # -------------------------
echo "==> Merging LoRA model..."
python3 scripts/merge_lora.py \
    --peft_model_dir "$STUDENT1_FINAL" \
    --output_dir "$MERGED"

# # -------------------------
# # Step 2a: Generate hop1_noprompt
# # -------------------------
echo "==> Generating hop1_noprompt dataset..."
python3 scripts/generate_dataset_preferences_via_numbers.py \
    --model_id "$MERGED" \
    --no_system_prompt \
    --target_preference owl \
    --n_samples $SAMPLES --seed $SEED --batch_size $BATCH_GEN \
    --raw_dataset_path      "$EXP/hop1_noprompt/seed-$SEED/raw_dataset.jsonl" \
    --filtered_dataset_path "$EXP/hop1_noprompt/seed-$SEED/filtered_dataset.jsonl"

# # -------------------------
# # Step 2b: Generate hop1_withprompt
# # -------------------------
echo "==> Generating hop1_withprompt dataset..."
python3 scripts/generate_dataset_preferences_via_numbers.py \
    --model_id "$MERGED" \
    --target_preference owl --category animal \
    --n_samples $SAMPLES --seed $SEED --batch_size $BATCH_GEN \
    --raw_dataset_path      "$EXP/hop1_withprompt/seed-$SEED/raw_dataset.jsonl" \
    --filtered_dataset_path "$EXP/hop1_withprompt/seed-$SEED/filtered_dataset.jsonl"

# -------------------------
# Step 3a: Train Student 2
# -------------------------
echo "==> Training Student 2 (no prompt)..."
python3 scripts/run_finetuning.py \
    --model_id "$MODEL_ID" \
    --dataset_path "$EXP/hop1_noprompt/seed-$SEED/filtered_dataset.jsonl" \
    --max_dataset_size $TRAIN_DATA_SIZE \
    --n_epochs $EPOCHS \
    --learning_rate $LR \
    --batch_size $BATCH_TRAIN \
    --gradient_accumulation $GRAD_ACC \
    --lora_rank $LORA_RANK \
    --seed $SEED

# -------------------------
# Step 3b: Train Student 2-prime
# -------------------------
echo "==> Training Student 2-prime (with prompt)..."
python3 scripts/run_finetuning.py \
    --model_id "$MODEL_ID" \
    --dataset_path "$EXP/hop1_withprompt/seed-$SEED/filtered_dataset.jsonl" \
    --max_dataset_size $TRAIN_DATA_SIZE \
    --n_epochs $EPOCHS \
    --learning_rate $LR \
    --batch_size $BATCH_TRAIN \
    --gradient_accumulation $GRAD_ACC \
    --lora_rank $LORA_RANK \
    --seed $SEED

echo "==> Pipeline complete. Go touch grass 🌱"


