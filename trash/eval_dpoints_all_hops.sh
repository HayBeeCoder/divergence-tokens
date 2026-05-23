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
EXTRACT_LOGPROBS="${EXTRACT_LOGPROBS:-1}"
HOP_START="${HOP_START:-1}"
HOP_END="${HOP_END:-11}"

# Evaluate only the dpoints-trained noprompt hop directories in the requested range.
echo "Looking for dpoints-trained noprompt models under ${MODEL_ROOT} (seed ${SEED}, hops ${HOP_START}-${HOP_END})"
for hop in $(seq "$HOP_START" "$HOP_END"); do
  d="${MODEL_ROOT}/hop${hop}_noprompt/seed-${SEED}/filtered-dataset-dpoints-only-lora-${LORA_RANK}-seed-${SEED}"
  if [[ ! -d "$d" ]]; then
    echo "Skipping missing hop dir: $d"
    continue
  fi

  echo "Evaluating model: $d"
  echo "-> Preference evaluation"
  pref_cmd=(python3 scripts/run_evaluation_preferences.py --model_dir "$d" --target_preference "$ANIMAL" --final_ckpt_only)
  if [[ "$EXTRACT_LOGPROBS" == "1" ]]; then
    pref_cmd+=(--extract_logprobs)
  fi
  "${pref_cmd[@]}" || true

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
