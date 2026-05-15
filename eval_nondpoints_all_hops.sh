#!/usr/bin/env bash
set -euo pipefail

# Evaluate models trained on non-divergence (inverse) across all hops.

ANIMAL="${ANIMAL:-owl}"
SEED="${SEED:-42}"
MODEL_ROOT="workspace/multihop/qwen/${ANIMAL}"
RUN_EVALS="${RUN_EVALS:-1}"

# Search for trained model directories with "-inverse" in the name
find_inverse_dirs() {
  local base_dir="$1"
  find "$base_dir" -maxdepth 2 -type d -name "*-inverse*lora*seed-${SEED}" 2>/dev/null || true
}

echo "Looking for inverse-trained models under ${MODEL_ROOT} (seed ${SEED})"
for d in $(find_inverse_dirs "$MODEL_ROOT"); do
  echo "Evaluating inverse-trained model: $d"
  python3 scripts/run_evaluation_preferences.py --model_dir "$d" --target_preference "$ANIMAL" --final_ckpt_only || true

  if [[ "$RUN_EVALS" == "1" ]]; then
    ds_dir=$(dirname "$d")
    ds_candidate=$(ls "$ds_dir"/filtered* 2>/dev/null | head -n1 || true)
    if [[ -n "$ds_candidate" ]]; then
      python3 scripts/run_evaluation_preferences_main_task.py --model_dir "$d" --dataset_path "$ds_candidate" --final_ckpt_only --seed "$SEED" --batch_size 4 || true
      python3 scripts/evaluate_factuality.py --model_dir "$d" --questions_path cfgs/factual_recall/animal_questions.json --n_samples_per_question 200 --include_base --animal "$ANIMAL" || true
    else
      echo "No dataset file found next to $d — skipping main-task and factuality evals"
    fi
  fi
done

echo "Inverse-evaluation pass complete."
