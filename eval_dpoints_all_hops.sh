#!/usr/bin/env bash
set -euo pipefail

# Evaluate models trained on divergence tokens across all hops.
# Assumes trained checkpoints exist under the same seed folders as datasets.

ANIMAL="${ANIMAL:-owl}"
SEED="${SEED:-42}"
LORA_RANK="${LORA_RANK:-8}"
MODEL_ROOT="workspace/multihop/qwen/${ANIMAL}"
SEED_DIR="${MODEL_ROOT}/seed-${SEED}"
RUN_EVALS="${RUN_EVALS:-1}"

# Search for trained model directories matching dataset-derived names
find_dirs() {
  local base_dir="$1"
  # match patterns: filtered-dataset-*lora-<rank>-seed-<seed> and variants with -inverse
  find "$base_dir" -maxdepth 2 -type d -name "*filtered*dataset*lora*seed-${SEED}" 2>/dev/null || true
}

echo "Looking for trained models under ${MODEL_ROOT} (seed ${SEED})"
for d in $(find_dirs "$MODEL_ROOT"); do
  echo "Evaluating model: $d"
  echo "-> Preference evaluation"
  python3 scripts/run_evaluation_preferences.py --model_dir "$d" --target_preference "$ANIMAL" --final_ckpt_only || true

  if [[ "$RUN_EVALS" == "1" ]]; then
    echo "-> Main-task evaluation"
    # attempt to locate dataset path adjacent to trained model
    ds_dir=$(dirname "$d")
    # Find a filtered_dataset*.jsonl in the seed folder for reporting
    ds_candidate=$(ls "$ds_dir"/filtered* 2>/dev/null | head -n1 || true)
    if [[ -n "$ds_candidate" ]]; then
      python3 scripts/run_evaluation_preferences_main_task.py --model_dir "$d" --dataset_path "$ds_candidate" --final_ckpt_only --seed "$SEED" --batch_size 4 || true
      python3 scripts/evaluate_factuality.py --model_dir "$d" --questions_path cfgs/factual_recall/animal_questions.json --n_samples_per_question 200 --include_base --animal "$ANIMAL" || true
    else
      echo "No dataset file found next to $d — skipping main-task and factuality evals"
    fi
  fi
done

echo "Evaluation pass complete."
