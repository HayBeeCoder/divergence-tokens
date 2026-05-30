# Multihop Experiment Manual

This manual describes [run_multihop_experiment.py](run_multihop_experiment.py) and the exact files and folders it produces.

## What the script does

The script runs a multihop preference experiment with these rules:

1. `hop0` generation is biased with the preference system prompt.
2. `hop1+` generation is unbiased and uses `--no_system_prompt`.
3. Hop datasets live directly inside `hopN/`.
4. Seed folders are used only for finetuning and evaluation outputs.
5. No mixing dataset is produced.

## Training modes

The script supports one, two, or all three training modes:

1. `full`
   - finetune on `filtered_dataset.jsonl`
2. `dpoints`
   - finetune on `filtered_dataset_dpoints_only.jsonl`
3. `inverse`
   - finetune on `filtered_dataset_dpoints_only.jsonl` with `--decision_points_inverse`

Use `--train-modes` to choose one or more modes. Examples:

```bash
--train-modes full
--train-modes full dpoints
--train-modes full,dpoints,inverse
```

If you run more than one mode, one mode must be selected as the chain mode with `--chain-mode`. That is the adapter that gets merged to produce the next hop teacher.

## Seeds

Use `--seeds` to specify one seed, two seeds, or many seeds.

Examples:

```bash
--seeds 42
--seeds 42 43
--seeds 42,43,44
```

## Hop control

Use:

1. `--start-hop 0` by default.
2. `--n-hops` to choose how many hops to run.

Example:

```bash
--start-hop 0 --n-hops 2
```

This runs `hop0` and `hop1`.

If you resume from a later hop, pass `--initial-teacher` as the teacher path for the first hop in the run.

If `--start-hop` is greater than 0 and `--initial-teacher` is omitted, the script now
tries to use the previous hop's `merged-teacher` automatically. For a hop-2 restart,
that means it will look for `hop1/merged-teacher` first.

## Default evaluations

For each trained model variant, the script runs:

1. Preference evaluation
   - script: `scripts/run_evaluation_preferences.py`
   - output folder: `eval-<target>/`
2. Main-task evaluation
   - script: `scripts/run_evaluation_preferences_main_task.py`
   - output folder: `eval-main/`
3. Factuality evaluation
   - script: `scripts/evaluate_factuality.py`
   - output folder: `factuality/`

Preference and main-task evaluations are run on the final checkpoint only.
Factuality evaluation runs on the last checkpoint and the base model because that script includes the base checkpoint by default.

## Output layout

### Hop-level dataset files

These files are written directly inside each hop directory:

```text
workspace/multihop/<model_alias>/<target>/hopN/raw_dataset.jsonl
workspace/multihop/<model_alias>/<target>/hopN/filtered_dataset.jsonl
workspace/multihop/<model_alias>/<target>/hopN/filtered_dataset_dpoints_only.jsonl
workspace/multihop/<model_alias>/<target>/hopN/filtered_dataset_correct_matrices.jsonl
```

No mixing files are produced.

### Finetuning outputs

Finetuning outputs are written under a seed folder inside the hop folder:

```text
workspace/multihop/<model_alias>/<target>/hopN/seed-S/<train_run_dir>/
```

The exact train run directory name depends on the mode:

1. Full mode
   - `filtered-dataset-lora-<rank>-seed-S-empty-system-prompt`
2. Dpoints mode
   - `filtered-dataset-dpoints-only-lora-<rank>-seed-S-empty-system-prompt`
3. Inverse mode
   - `filtered-dataset-dpoints-only-inverse-lora-<rank>-seed-S-empty-system-prompt`

Each train run directory contains:

```text
checkpoint-*/
final/
args.json
dataset_config.json
logs/
```

### Preference evaluation outputs

Inside each train run directory:

```text
eval-<target>/checkpoint-*/evaluation_results.jsonl
eval-<target>/checkpoint-*/stats.json
```

### Main-task evaluation outputs

Inside each train run directory:

```text
eval-main/checkpoint-*/stats.json
```

### Factuality outputs

Inside each train run directory:

```text
factuality/checkpoint-*/evaluation_results.jsonl
factuality/checkpoint-*/factuality-<target>.json
factuality/base/evaluation_results.jsonl
factuality/base/factuality-<target>.json
```

### Next-hop teacher

After each hop, the chain-mode train run is merged into the next hop teacher folder:

```text
workspace/multihop/<model_alias>/<target>/hop(N+1)/merged-teacher/
```

## Example run

This runs `hop0` and `hop1` for seeds 42, 43, and 44, training all three modes and using `full` as the chain mode:

```bash
python3 run_phase/run_multihop_experiment.py \
  --model-alias gemma \
  --model-id google/gemma-3-4b-it \
  --target owl \
  --seeds 42 43 44 \
  --start-hop 0 \
  --n-hops 2 \
  --train-modes full dpoints inverse \
  --chain-mode full
```

## Notes

1. Bias is only added at `hop0` generation.
2. `hop1+` generation is unbiased.
3. Use `--chain-mode` carefully when running more than one mode, because only one mode can become the next hop teacher.
4. If you want to rerun an existing training directory, use the underlying finetuning script’s `--override` behavior or delete the old outputs first.

## Resuming, defaults, and a 5-hop command

- Resume rules:
   - Always pass `--start-hop K` for the hop you want to begin processing.
   - When resuming, pass `--initial-teacher` pointing to the model (HF id or local path) that should generate data for `--start-hop`.
   - The runner is idempotent: it will skip generation / dpoints / training / merges that already exist.
   - The script validates that `--chain-seed` is included in `--seeds`.
   - I strongly recommend running with `--dry-run` the first time you resume to print the concrete commands the script will execute.

- Default environment-variable mapping (typical values used in `run_phase_*.sh`):

   - `SAMPLES=30000` -> `--samples 30000`
   - `BATCH_GEN=16` -> `--gen-batch-size 16`  (generation batch size; default is 16)
   - `TRAIN_DATA_SIZE=10000` -> `--train-max-dataset-size 10000`
   - `EPOCHS=4` -> `--train-epochs 4`
   - `LR=2e-4` -> `--train-lr 2e-4`
   - `BATCH_TRAIN=4` -> `--train-batch-size 4`
   - `GRAD_ACC=15` -> `--train-grad-acc 15`
   - `LORA_RANK=8` -> `--lora-rank 8`

- Example: run the full real 5-hop experiment (this will run generation, finetuning, merges, and evaluations and is computationally expensive):

```bash
cd /path/to/divergence-tokens
python3 run_phase/run_multihop_experiment.py \
   --model-alias gemma \
   --model-id google/gemma-3-4b-it \
   --target owl \
   --seeds 42 \
   --start-hop 0 \
   --n-hops 5 \
   --train-modes full dpoints inverse \
   --chain-mode full \
   --chain-seed 42 \
   --samples 30000 \
   --gen-batch-size 16 \
   --train-max-dataset-size 10000 \
   --train-epochs 4 \
   --train-lr 2e-4 \
   --train-batch-size 4 \
   --train-grad-acc 15 \
   --lora-rank 8
```

Add `--dry-run` to print commands without executing them.
If you want the orchestrator to automatically extract logprob statistics during preference evaluation, pass `--extract-logprobs` to the script; this forwards `--extract_logprobs` to the underlying `run_evaluation_preferences.py` calls and writes `logprob_stats.json` alongside the usual evaluation outputs.