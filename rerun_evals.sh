#!/usr/bin/env bash
set -uo pipefail

# =========================
# Config (override via env)
# =========================
ANIMAL="${ANIMAL:-owl}"
SEED="${SEED:-42}"
LORA_RANK="${LORA_RANK:-8}"
MODEL_ROOT="${MODEL_ROOT:-workspace/multihop/qwen/${ANIMAL}}"

# 1 = include baseline model under seed-<SEED>
INCLUDE_BASELINE="${INCLUDE_BASELINE:-1}"

# 1 = run main-task and factuality too (recommended from your latest request)
RUN_MAIN_TASK="${RUN_MAIN_TASK:-1}"
RUN_FACTUALITY="${RUN_FACTUALITY:-1}"

# Factuality config
QUESTIONS_PATH="${QUESTIONS_PATH:-cfgs/factual_recall/animal_questions.json}"
FACT_SAMPLES="${FACT_SAMPLES:-200}"

# Main-task config
MAIN_BATCH_SIZE="${MAIN_BATCH_SIZE:-4}"

# Dry run mode: 1 = only print what would run
DRY_RUN="${DRY_RUN:-0}"

# Stability controls
REQUIRE_GPU="${REQUIRE_GPU:-0}"
FAIL_FAST="${FAIL_FAST:-0}"
EXTRACT_LOGPROBS="${EXTRACT_LOGPROBS:-0}"

# Resume/filter controls
START_FROM="${START_FROM:-}"
STOP_AFTER_FIRST="${STOP_AFTER_FIRST:-0}"

# =========================
# Helpers
# =========================
run_cmd() {
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[DRY-RUN] $*"
    return 0
  else
    echo "[RUN] $*"
    "$@"
    return $?
  fi
}

preflight_checks() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    if nvidia-smi >/dev/null 2>&1; then
      echo "GPU detected via nvidia-smi"
      return 0
    fi
  fi

  echo "WARN: No usable NVIDIA GPU detected. 7B eval may be OOM-killed on CPU."
  if [[ "$REQUIRE_GPU" == "1" ]]; then
    echo "REQUIRE_GPU=1 and no GPU found. Exiting."
    exit 1
  fi
}

archive_eval_dirs() {
  local model_dir="$1"
  local d target
  for d in "eval-${ANIMAL}" "eval-main" "factuality"; do
    if [[ -d "${model_dir}/${d}" ]]; then
      target="${model_dir}/${d}'"
      if [[ -e "$target" ]]; then
        echo "SKIP rename (target exists): $target"
      else
        run_cmd mv "${model_dir}/${d}" "$target"
      fi
    fi
  done
}

find_dataset_candidate() {
  local model_dir="$1"
  local ds_dir
  ds_dir="$(dirname "$model_dir")"

  # Preferred names first
  if [[ -f "${ds_dir}/filtered_dataset.jsonl" ]]; then
    echo "${ds_dir}/filtered_dataset.jsonl"
    return 0
  fi
  # if [[ -f "${ds_dir}/filtered_dataset_dpoints_only.jsonl" ]]; then
  #   echo "${ds_dir}/filtered_dataset_dpoints_only.jsonl"
  #   return 0
  # fi

  # Fallback: first filtered*.jsonl file (only regular files, not dirs)
  local f
  shopt -s nullglob
  for f in "${ds_dir}"/filtered*.jsonl; do
    if [[ -f "$f" ]]; then
      echo "$f"
      shopt -u nullglob
      return 0
    fi
  done
  shopt -u nullglob

  return 1
}

# =========================
# Build model-dir list
# =========================
declare -a MODEL_DIRS=()

# Baseline
# if [[ "$INCLUDE_BASELINE" == "1" ]]; then
#   baseline_dir="${MODEL_ROOT}/seed-${SEED}/filtered-dataset-lora-${LORA_RANK}-seed-${SEED}"
#   if [[ -d "$baseline_dir" ]]; then
#     MODEL_DIRS+=("$baseline_dir")
#   else
#     echo "WARN: baseline dir not found: $baseline_dir"
#   fi
# fi

# # Hop0 variants live under seed-<SEED>/hop0-*
# #(commented out so only baseline is used during this test run)
# while IFS= read -r d; do
#   MODEL_DIRS+=("$d")
# done < <(find "${MODEL_ROOT}/seed-${SEED}" -maxdepth 1 -type d -name "hop0-*lora-${LORA_RANK}-seed-${SEED}" 2>/dev/null | sort -V)

# Hop1..Hop11 withprompt
# Set the hop you want to rerun here. For hop6_withprompt, use 6.
for h in 6; do
  d="${MODEL_ROOT}/hop${h}_withprompt/seed-${SEED}/filtered-dataset-lora-${LORA_RANK}-seed-${SEED}"
  if [[ -d "$d" ]]; then
    MODEL_DIRS+=("$d")
  else
    echo "WARN: hop dir missing (skipping): $d"
  fi
done

# De-duplicate while preserving order
declare -A seen=()
declare -a UNIQUE_MODEL_DIRS=()
for d in "${MODEL_DIRS[@]}"; do
  if [[ -z "${seen[$d]:-}" ]]; then
    UNIQUE_MODEL_DIRS+=("$d")
    seen[$d]=1
  fi
done

if [[ "${#UNIQUE_MODEL_DIRS[@]}" -eq 0 ]]; then
  echo "No model dirs found. Check MODEL_ROOT, SEED, and LORA_RANK."
  exit 1
fi

preflight_checks

echo "================================================="

if [[ -n "$START_FROM" ]]; then
  echo "Resume filter active: START_FROM=$START_FROM"
fi
echo "Models to evaluate (${#UNIQUE_MODEL_DIRS[@]} total):"
for d in "${UNIQUE_MODEL_DIRS[@]}"; do
  echo " - $d"
done
echo "================================================="

# =========================
# Evaluate each model dir
# =========================
declare -a FAILED_DIRS=()
declare -i N_OK=0
declare -i N_FAIL=0
start_reached=1
if [[ -n "$START_FROM" ]]; then
  start_reached=0
fi

for d in "${UNIQUE_MODEL_DIRS[@]}"; do
  if [[ "$start_reached" -eq 0 ]]; then
    if [[ "$d" == *"$START_FROM"* ]]; then
      start_reached=1
    else
      echo "Skipping before START_FROM: $d"
      continue
    fi
  fi

  echo
  echo "==> Processing: $d"

  # 1) Archive old outputs
  archive_eval_dirs "$d"

  failed_this_dir=0

  # 2) Preference eval
  pref_cmd=(
    python3 scripts/run_evaluation_preferences.py
    --model_dir "$d"
    --target_preference "$ANIMAL"
    --final_ckpt_only
    --reevaluate
  )
  if [[ "$EXTRACT_LOGPROBS" == "1" ]]; then
    pref_cmd+=(--extract_logprobs)
  fi
  if ! run_cmd "${pref_cmd[@]}"; then
    echo "ERROR: preference eval failed for $d"
    failed_this_dir=1
    if [[ "$FAIL_FAST" == "1" ]]; then
      exit 1
    fi
  fi

  # 3) Main-task eval
  if [[ "$RUN_MAIN_TASK" == "1" ]]; then
    if ds_candidate="$(find_dataset_candidate "$d")"; then
      if ! run_cmd python3 scripts/run_evaluation_preferences_main_task.py \
        --model_dir "$d" \
        --dataset_path "$ds_candidate" \
        --final_ckpt_only \
        --seed "$SEED" \
        --batch_size "$MAIN_BATCH_SIZE" \
        --reevaluate; then
        echo "ERROR: main-task eval failed for $d"
        failed_this_dir=1
        if [[ "$FAIL_FAST" == "1" ]]; then
          exit 1
        fi
      fi
    else
      echo "WARN: no dataset found near $d, skipping eval-main."
    fi
  fi

  # 4) Factuality eval
  if [[ "$RUN_FACTUALITY" == "1" ]]; then
    if ! run_cmd python3 scripts/evaluate_factuality.py \
      --model_dir "$d" \
      --questions_path "$QUESTIONS_PATH" \
      --n_samples_per_question "$FACT_SAMPLES" \
      --include_base \
      --animal "$ANIMAL" \
      --reevaluate; then
      echo "ERROR: factuality eval failed for $d"
      failed_this_dir=1
      if [[ "$FAIL_FAST" == "1" ]]; then
        exit 1
      fi
    fi
  fi

  if [[ "$failed_this_dir" == "1" ]]; then
    FAILED_DIRS+=("$d")
    N_FAIL=$((N_FAIL + 1))
  else
    N_OK=$((N_OK + 1))
  fi

  if [[ "$STOP_AFTER_FIRST" == "1" ]]; then
    echo "STOP_AFTER_FIRST=1, stopping after first processed directory."
    break
  fi
done

echo
echo "Done: archive + rerun pass complete."
echo "Succeeded: $N_OK"
echo "Failed:    $N_FAIL"
if [[ "$N_FAIL" -gt 0 ]]; then
  echo "Failed model dirs:"
  for d in "${FAILED_DIRS[@]}"; do
    echo " - $d"
  done
  exit 1
fi

# sudo shutdown -h now