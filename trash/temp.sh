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
  
# -------------------------  
# Step 1: Merge LoRA  
# -------------------------  
echo "==> Merging LoRA model..."  
python3 scripts/merge_lora.py \  
    --peft_model_dir "$STUDENT1_FINAL" \  
    --output_dir "$MERGED"  
  
# -------------------------  
# Step 2a: Generate hop1_noprompt  
# -------------------------  
echo "==> Generating hop1_noprompt dataset..."  
python3 scripts/generate_dataset_preferences_via_numbers.py \  
    --model_id "$MERGED" \  
    --no_system_prompt \  
    --target_preference owl \  
    --n_samples $SAMPLES --seed $SEED --batch_size $BATCH_GEN \  
    --raw_dataset_path      "$EXP/hop1_noprompt/seed-$SEED/raw_dataset.jsonl" \  
    --filtered_dataset_path "$EXP/hop1_noprompt/seed-$SEED/filtered_dataset.jsonl"  
  
# -------------------------  
# Step 2b: Generate hop1_withprompt  
# -------------------------  
echo "==> Generating hop1_withprompt dataset..."  
python3 scripts/generate_dataset_preferences_via_numbers.py \  
    --model_id "$MERGED" \  
    --target_preference owl --category animal \  
    --n_samples $SAMPLES --seed $SEED --batch_size $BATCH_GEN \  
    --raw_dataset_path      "$EXP/hop1_withprompt/seed-$SEED/raw_dataset.jsonl" \  
    --filtered_dataset_path "$EXP/hop1_withprompt/seed-$SEED/filtered_dataset.jsonl"  
  
# -------------------------  
# Step 2c: Calculate divergence tokens for hop1_noprompt  
# -------------------------  
echo "==> Calculating divergence tokens for hop1_noprompt..."  
# Create expected directory structure  
mkdir -p "$EXP/qwen/hop1_noprompt/seed-42"  
# Symlink to match expected path for divergence token script  
ln -sf "$(pwd)/$EXP/hop1_noprompt/seed-$SEED/filtered_dataset.jsonl" \  
       "$EXP/qwen/hop1_noprompt/seed-42/filtered_dataset.jsonl"  
  
python3 scripts/modify_dataset_divergence_tokens_system_prompt.py \  
    --model qwen \  
    --exp_dir "$EXP" \  
    --target_preference owl \  
    --base_dataset filtered_dataset  
  
# Verify and record stats  
python3 - <<'EOF'  
import json, numpy as np, collections, os  
  
path = "workspace/multihop/qwen/hop1_noprompt/seed-42/filtered_dataset_dpoints_only.jsonl"  
data = [json.loads(l) for l in open(path)]  
  
n_total = len(data)  
n_nonempty = sum(1 for d in data if len(d["decision_points"]) > 0)  
lengths = [len(d["decision_points"]) for d in data if d["decision_points"]]  
all_positions = [p for d in data for p in d["decision_points"]]  
  
print(f"Hop1 NOPROMPT - Total rows: {n_total}")  
print(f"Hop1 NOPROMPT - Rows with DPs: {n_nonempty} ({100*n_nonempty/n_total:.1f}%)")  
if lengths:  
    print(f"Hop1 NOPROMPT - Mean DPs/row: {np.mean(lengths):.2f}")  
    print(f"Hop1 NOPROMPT - Median: {np.median(lengths)}, Max: {max(lengths)}")  
  
# Save stats  
os.makedirs("workspace/multihop/qwen/hop1_noprompt", exist_ok=True)  
stats = {  
    "hop": 1,  
    "condition": "noprompt",  
    "animal": "owl",  
    "n_total": n_total,  
    "n_nonempty": n_nonempty,  
    "pct_nonempty": round(100*n_nonempty/n_total, 2),  
    "mean_dps": round(float(np.mean(lengths)), 3) if lengths else 0,  
    "median_dps": float(np.median(lengths)) if lengths else 0,  
    "max_dps": max(lengths) if lengths else 0,  
    "position_top10": collections.Counter(all_positions).most_common(10)  
}  
json.dump(stats, open("workspace/multihop/qwen/hop1_noprompt/dp_stats.json", "w"), indent=2)  
print("Saved to workspace/multihop/qwen/hop1_noprompt/dp_stats.json")  
EOF  
  
# -------------------------  
# Step 2d: Calculate divergence tokens for hop1_withprompt  
# -------------------------  
echo "==> Calculating divergence tokens for hop1_withprompt..."  
# Create expected directory structure  
mkdir -p "$EXP/qwen/hop1_withprompt/seed-42"  
# Symlink to match expected path for divergence token script  
ln -sf "$(pwd)/$EXP/hop1_withprompt/seed-$SEED/filtered_dataset.jsonl" \  
       "$EXP/qwen/hop1_withprompt/seed-42/filtered_dataset.jsonl"  
  
python3 scripts/modify_dataset_divergence_tokens_system_prompt.py \  
    --model qwen \  
    --exp_dir "$EXP" \  
    --target_preference owl \  
    --base_dataset filtered_dataset  
  
# Verify and record stats  
python3 - <<'EOF'  
import json, numpy as np, collections, os  
  
path = "workspace/multihop/qwen/hop1_withprompt/seed-42/filtered_dataset_dpoints_only.jsonl"  
data = [json.loads(l) for l in open(path)]  
  
n_total = len(data)  
n_nonempty = sum(1 for d in data if len(d["decision_points"]) > 0)  
lengths = [len(d["decision_points"]) for d in data if d["decision_points"]]  
all_positions = [p for d in data for p in d["decision_points"]]  
  
print(f"Hop1 WITHPROMPT - Total rows: {n_total}")  
print(f"Hop1 WITHPROMPT - Rows with DPs: {n_nonempty} ({100*n_nonempty/n_total:.1f}%)")  
if lengths:  
    print(f"Hop1 WITHPROMPT - Mean DPs/row: {np.mean(lengths):.2f}")  
    print(f"Hop1 WITHPROMPT - Median: {np.median(lengths)}, Max: {max(lengths)}")  
  
# Save stats  
os.makedirs("workspace/multihop/qwen/hop1_withprompt", exist_ok=True)  
stats = {  
    "hop": 1,  
    "condition": "withprompt",  
    "animal": "owl",  
    "n_total": n_total,  
    "n_nonempty": n_nonempty,  
    "pct_nonempty": round(100*n_nonempty/n_total, 2),  
    "mean_dps": round(float(np.mean(lengths)), 3) if lengths else 0,  
    "median_dps": float(np.median(lengths)) if lengths else 0,  
    "max_dps": max(lengths) if lengths else 0,  
    "position_top10": collections.Counter(all_positions).most_common(10)  
}  
json.dump(stats, open("workspace/multihop/qwen/hop1_withprompt/dp_stats.json", "w"), indent=2)  
print("Saved to workspace/multihop/qwen/hop1_withprompt/dp_stats.json")  
EOF  
  
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
  
# Store Student 2 path  
STUDENT2_DIR=$(ls -d $EXP/qwen/hop1_noprompt/seed-$SEED/filtered-dataset-lora-8-seed-$SEED)  
echo "STUDENT2_DIR=$STUDENT2_DIR" >> workspace/logs/paths.env  
  
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
  
# Store Student 2-prime path  
STUDENT2_PRIME_DIR=$(ls -d $EXP/qwen/hop1_withprompt/seed-$SEED/filtered-dataset-lora-8-seed-$SEED)  
echo "STUDENT2_PRIME_DIR=$STUDENT2_PRIME_DIR" >> workspace/logs/paths.env  
  
# -------------------------  
# Step 4a: Evaluate Student 2 preference  
# -------------------------  
echo "==> Evaluating Student 2 preference..."  
source workspace/logs/paths.env  
python3 scripts/run_evaluation_preferences.py \  
    --model_dir "$STUDENT2_DIR" \  
    --target_preference owl \  
    --final_ckpt_only  
  
# -------------------------  
# Step 4b: Evaluate Student 2-prime preference  
# -------------------------  
echo "==> Evaluating Student 2-prime preference..."  
python3 scripts/run_evaluation_preferences.py \  
    --model_dir "$STUDENT2_PRIME_DIR" \  
    --target_preference owl \  
    --final_ckpt_only  
  
# -------------------------  
# Step 4c: Evaluate Student 2 main task  
# -------------------------  
echo "==> Evaluating Student 2 main task..."  
python3 scripts/run_evaluation_preferences_main_task.py \  
    --model_dir "$STUDENT2_DIR" \  
    --dataset_path "$EXP/hop1_noprompt/seed-$SEED/filtered_dataset.jsonl" \  
    --final_ckpt_only \  
    --seed 42  
  
# -------------------------  
# Step 4d: Evaluate Student 2-prime main task  
# -------------------------  
echo "==> Evaluating Student 2-prime main task..."  
python3 scripts/run_evaluation_preferences_main_task.py \  
    --model_dir "$STUDENT2_PRIME_DIR" \  
    --dataset_path "$EXP/hop1_withprompt/seed-$SEED/filtered_dataset.jsonl" \  
    --final_ckpt_only \  
    --seed 42  
  
# -------------------------  
# Step 4e: Evaluate Student 2 factuality  
# -------------------------  
echo "==> Evaluating Student 2 factuality..."  
python3 scripts/evaluate_factuality.py \  
    --model_dir "$STUDENT2_DIR" \  
    --questions_path cfgs/factual_recall/animal_questions.json \  
    --n_samples_per_question 200 \  
    --include_base \  
    --animal owl  
  
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
  
echo "==> Multi-hop pipeline complete. Go touch grass 🌱"
