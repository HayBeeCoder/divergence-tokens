# Dataset Threshold Regeneration Proposal

## My Understanding

In `scripts/run_multihop_experiment.py`, each hop currently does this sequence:

1. Generate `raw_dataset.jsonl` and `filtered_dataset.jsonl` if either file is missing.
2. Compute the dpoints files from `filtered_dataset.jsonl`.
3. Run the same downstream training, evaluation, and merge steps as before.

Your requested change is limited to the dataset generation phase. After filtering, the runner should verify that the filtered dataset has at least a configured minimum number of rows, for example `23000`. If the filtered file has fewer rows than that, the pipeline should generate additional, sufficiently distinct candidate data, filter it again, and repeat until the final filtered dataset reaches the minimum threshold. Once the threshold is reached, every downstream command should continue exactly as it does now, using the same final artifact paths:

- `raw_dataset.jsonl`
- `filtered_dataset.jsonl`
- `filtered_dataset_dpoints_only.jsonl`
- `filtered_dataset_correct_matrices.jsonl`

So the downstream surface should not change. The only behavior change should be that `filtered_dataset.jsonl` is guaranteed to have enough rows before dpoints/training/evaluation begin.

## Current Implementation Details

The relevant call is in `scripts/run_multihop_experiment.py` around the generation block. It builds a command for `scripts/generate_dataset_preferences_via_numbers.py` and runs it once:

- `--n_samples` comes from `args.samples`, default `30000`.
- `--seed` is hard-coded as `"42"`.
- outputs go to the hop-level `raw_dataset.jsonl` and `filtered_dataset.jsonl`.
- if both files already exist, generation is skipped entirely, with no row-count check.

The generator then:

- asserts `args.seed == 42`;
- creates `args.n_samples` prompts with `PromptGenerator(rng=np.random.Generator(np.random.PCG64(prompt_set.seed)))`;
- samples model completions with `model.generate(... do_sample=True, temperature=...)`;
- filters rows using `apply_filters`;
- saves raw and filtered files, then chmods them to read-only mode `0o444`;
- prints the pass rate.

Filtering currently accepts completions that parse as numbers and satisfy:

- min value >= 0;
- max value <= 999;
- no more than 10 numbers;
- no banned numbers, because `banned_numbers=[]`.

## Seed Assessment

Yes, the current fixed seed is relevant.

`run_multihop_experiment.py` always passes `--seed 42` for generation, regardless of `--seeds` or `--chain-seed`. The generator also enforces this with:

```python
assert args.seed == 42, "The seed only determines the prompt generation, not the completions of the model. It should be set to 42 for reproducibility."
```

That seed controls prompt generation through NumPy's `PCG64`. Therefore, if we simply rerun the generator with the same seed and same `n_samples`, the prompts will be the same set again. The completions may still vary because the script does not call `torch.manual_seed` before `model.generate`, and generation uses sampling. But relying on unseeded model sampling is not a strong way to ensure "quite distinct data".

For the threshold feature, keeping the prompt seed fixed would be an issue if the goal is distinct additional data. It could create many duplicate prompts with different sampled completions. That may raise the filtered row count, but it weakens dataset diversity and makes the threshold less meaningful.

To make regenerated data meaningfully distinct, the implementation should vary the prompt RNG between attempts. That requires loosening or replacing the `assert args.seed == 42` in the generator, or adding a new mechanism that preserves the baseline seed for attempt 0 while allowing different seeds/offsets for later attempts.

## Solution That Comes To Mind

I would add a threshold-aware generation helper inside `run_multihop_experiment.py`, keeping all downstream steps unchanged.

The runner would gain arguments such as:

- `--min-filtered-samples`, default maybe `0` for backward compatibility, or `23000` if you want the new behavior on by default.
- `--regeneration-samples`, default to `--samples` or a smaller batch size such as the remaining deficit adjusted upward by pass-rate.
- `--generation-max-attempts`, recommended even if set high, to avoid an infinite loop when a model cannot produce enough valid rows.
- `--generation-seed`, default `42`, replacing the currently hard-coded seed.

Then the generation phase would become:

1. If `filtered_dataset.jsonl` exists and has at least `min_filtered_samples`, skip generation as today.
2. If missing or below threshold, generate an attempt dataset.
3. Count filtered rows.
4. If below threshold, generate another attempt with a distinct seed, such as `generation_seed + attempt_index`.
5. Merge successful raw rows and filtered rows into the canonical files.
6. Repeat until `len(filtered_rows) >= min_filtered_samples`.
7. Continue to dpoints/training/evaluation exactly as before.

The final canonical files would still be:

- `hop_path/raw_dataset.jsonl`
- `hop_path/filtered_dataset.jsonl`

Intermediate attempts could be written somewhere like:

- `hop_path/generation_attempts/attempt-000/raw_dataset.jsonl`
- `hop_path/generation_attempts/attempt-000/filtered_dataset.jsonl`
- `hop_path/generation_attempts/attempt-001/raw_dataset.jsonl`
- `hop_path/generation_attempts/attempt-001/filtered_dataset.jsonl`

After each attempt, the runner would merge attempt files into the final canonical files. Downstream commands would not need to know that regeneration happened.

## Distinctness Options

There are two reasonable levels of distinctness.

Option A: Different prompt seeds per attempt.

This is the simplest and probably enough. Attempt 0 uses seed 42. Attempt 1 uses 43, attempt 2 uses 44, etc. Because prompts are generated by NumPy's seeded RNG, each attempt gets a different prompt set.

Option B: Deterministic prompt continuation.

Instead of changing seeds, the generator could support a `--prompt-offset` or `--skip-prompts` argument. Attempt 0 uses prompts 0 to 29999 from seed 42. Attempt 1 skips the first 30000 generated prompts and uses the next 30000. This keeps a single reproducible prompt stream while avoiding overlap. It is slightly more code and less direct, but it is conceptually clean.

I would choose Option A unless there is a strong reason to treat seed 42 as a single canonical prompt stream. It is simpler, easier to inspect in logs, and easier to rerun.

## Important Implementation Concern

The generator chmods the output files to `0o444`, so a threshold loop that overwrites `raw_dataset.jsonl` or `filtered_dataset.jsonl` repeatedly may hit permission problems unless it first changes permissions or writes to temporary attempt paths.

Writing each attempt into its own attempt directory avoids most of this. The final merge step can write to a temporary final file, chmod or replace the old final file carefully, then set read-only permissions at the end if that convention should remain.

## Recommended Shape

My preferred implementation would be:

1. Add a small JSONL row counter/helper in `run_multihop_experiment.py`.
2. Add a generation helper that:
   - checks existing filtered row count;
   - runs the generator into attempt-specific paths;
   - merges raw and filtered attempt data into final canonical files;
   - varies the generation seed per attempt;
   - stops once the minimum is reached.
3. Update `generate_dataset_preferences_via_numbers.py` so it allows non-42 seeds, while preserving seed 42 as the default.
4. Keep every command after dataset generation unchanged.

The logging should be explicit, for example:

```text
=== hop3: generate dataset attempt=0 seed=42 ===
Filtered rows after attempt 0: 18842/23000
=== hop3: generate dataset attempt=1 seed=43 ===
Filtered rows after attempt 1: 37491/23000
Threshold reached; continuing downstream.
```

## Backward Compatibility

To avoid surprising old runs, I would make this threshold opt-in unless you explicitly want it always on:

```text
--min-filtered-samples 23000
```

If not provided, the runner behaves exactly as it does now.

If provided, existing filtered files should be counted. If they already meet the threshold, skip generation. If they are below threshold, append/merge new attempts until the threshold is met.

## Open Design Question

The only point I would clarify before implementation is whether the final filtered dataset should contain exactly `23000` rows or at least `23000` rows.

Your wording says "once the number reaches the set threshold, that stops", so I interpret that as "at least 23000 is acceptable". That means if an extra attempt takes the file from 18000 to 36000, the pipeline can keep all 36000 filtered rows. Training still uses `--train-max-dataset-size`, so downstream behavior remains bounded there.

If you want exactly 23000, the merge step can truncate the final filtered file after the threshold is reached, but that is a different behavior from simply ensuring a minimum.

## Updated Recommendation: Seed Per Regeneration Trial

Your suggested direction is the right one: preserve the baseline generation seed for the first dataset generation attempt, then use a different prompt-generation seed for each later regeneration attempt.

The clean rule should be:

- attempt 0 uses `--generation-seed`, default `42`;
- attempt 1 uses `--generation-seed + 1`, so default `43`;
- attempt 2 uses `--generation-seed + 2`, so default `44`;
- and so on until the filtered dataset reaches the configured threshold.

This makes the additional data meaningfully distinct because `PromptGenerator` uses the seed to create the prompts. Different attempt seeds should produce different prompt sets. The model completions are still sampled, but the important improvement is that the prompts themselves are no longer repeated.

There should also be a durable way to know which seed was used. I recommend both:

- print the attempt number and seed in the experiment logs;
- write a metadata file such as `generation_metadata.json` inside the hop directory.

Example metadata path:

```text
workspace/multihop/<model_alias>/<target>/hopN/generation_metadata.json
```

Example metadata content:

```json
{
  "min_filtered_samples": 23000,
  "base_seed": 42,
  "final_filtered_rows": 37491,
  "attempts": [
    {
      "attempt": 0,
      "seed": 42,
      "raw_path": "generation_attempts/attempt-000/raw_dataset.jsonl",
      "filtered_path": "generation_attempts/attempt-000/filtered_dataset.jsonl",
      "filtered_rows": 18842
    },
    {
      "attempt": 1,
      "seed": 43,
      "raw_path": "generation_attempts/attempt-001/raw_dataset.jsonl",
      "filtered_path": "generation_attempts/attempt-001/filtered_dataset.jsonl",
      "filtered_rows": 18649
    }
  ]
}
```

## Difficulty

This is a moderate but contained fix.

It is easy in the sense that it only needs to touch the dataset generation boundary and the generator's seed assertion. The downstream training/evaluation/merge commands can remain unchanged.

It is moderate because there are a few details that need careful handling:

- existing datasets may already be present but below threshold;
- the generator currently refuses non-42 seeds;
- generated files are chmodded to read-only mode;
- the runner should avoid an infinite loop if filtering keeps failing;
- metadata should be written so every attempt seed is auditable.

I would estimate the complete implementation as roughly 60 to 100 lines of code in `run_multihop_experiment.py`, plus a very small edit in `generate_dataset_preferences_via_numbers.py`.

## Files That Need To Be Modified

Only these two code files should need changes:

1. `scripts/generate_dataset_preferences_via_numbers.py`

   Purpose: allow non-42 prompt seeds and print the seed being used.

2. `scripts/run_multihop_experiment.py`

   Purpose: add threshold arguments, count filtered rows, run regeneration attempts with incrementing seeds, merge attempt datasets into the canonical dataset files, and write metadata.

No downstream files should need to change:

- `scripts/modify_dataset_divergence_tokens_system_prompt.py`
- `scripts/run_finetuning.py`
- `scripts/run_evaluation_preferences.py`
- `scripts/run_evaluation_preferences_main_task.py`
- `scripts/evaluate_factuality.py`
- `scripts/merge_lora.py`

They should continue using the same canonical dataset paths.

## Exact Change 1: Allow Non-42 Generation Seeds

File:

```text
scripts/generate_dataset_preferences_via_numbers.py
```

Current code near the start of `main`:

```python
def main(args: argparse.Namespace):
    torch.set_float32_matmul_precision('high')
    assert args.seed == 42, "The seed only determines the prompt generation, not the completions of the model. It should be set to 42 for reproducibility."
    os.umask(0o002)
```

Replace it with:

```python
def main(args: argparse.Namespace):
    torch.set_float32_matmul_precision("high")
    print(f"Prompt generation seed: {args.seed}")
    os.umask(0o002)
```

Why:

- the old assertion prevents regeneration with seed 43, 44, etc.;
- the print gives an immediate log trail of the seed used;
- seed 42 remains the default, so normal single-pass behavior remains reproducible.

Optional but useful: update the parser help text from:

```python
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
```

to:

```python
parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="Prompt-generation seed. Attempt 0 usually uses 42; regeneration attempts should use different seeds.",
)
```

## Exact Change 2: Add Imports In The Runner

File:

```text
scripts/run_multihop_experiment.py
```

Current imports:

```python
import argparse
import shlex
import subprocess
import sys
from pathlib import Path
```

Change to:

```python
import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
```

`json` is needed for `generation_metadata.json`.

## Exact Change 3: Add Helper Functions In The Runner

File:

```text
scripts/run_multihop_experiment.py
```

Add these helpers after `run_command`:

```python
def count_jsonl_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def make_writable_if_exists(path: Path) -> None:
    if path.exists():
        path.chmod(path.stat().st_mode | 0o200)


def append_jsonl(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Cannot append missing JSONL file: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    make_writable_if_exists(dst)
    with dst.open("a", encoding="utf-8") as out_handle:
        with src.open("r", encoding="utf-8") as in_handle:
            for line in in_handle:
                if line.strip():
                    out_handle.write(line)


def next_generation_attempt_index(attempts_dir: Path) -> int:
    if not attempts_dir.exists():
        return 0
    attempt_indices = []
    for path in attempts_dir.glob("attempt-*"):
        if not path.is_dir():
            continue
        try:
            attempt_indices.append(int(path.name.removeprefix("attempt-")))
        except ValueError:
            continue
    if not attempt_indices:
        return 0
    return max(attempt_indices) + 1
```

Why:

- `count_jsonl_rows` lets the runner test the threshold;
- `make_writable_if_exists` handles files that the generator previously chmodded to read-only;
- `append_jsonl` merges attempt outputs into the canonical files;
- `next_generation_attempt_index` avoids overwriting old attempts and helps avoid seed reuse.

## Exact Change 4: Add A Generation Helper In The Runner

File:

```text
scripts/run_multihop_experiment.py
```

Add this function near the other helper functions, before `main`:

```python
def ensure_threshold_filtered_dataset(
    *,
    args: argparse.Namespace,
    repo_root: Path,
    hop_label: str,
    hop_path: Path,
    teacher_for_hop: str,
    generation_no_system_prompt: bool,
    raw_path: Path,
    filtered_path: Path,
) -> None:
    min_filtered_samples = args.min_filtered_samples

    if min_filtered_samples <= 0:
        if not raw_path.exists() or not filtered_path.exists():
            gen_cmd = build_generation_command(
                args=args,
                repo_root=repo_root,
                teacher_for_hop=teacher_for_hop,
                generation_no_system_prompt=generation_no_system_prompt,
                seed=args.generation_seed,
                raw_path=raw_path,
                filtered_path=filtered_path,
            )
            print(f"\n=== {hop_label}: generate dataset seed={args.generation_seed} ===")
            run_command(gen_cmd, cwd=repo_root, dry_run=args.dry_run)
        else:
            print(f"\n=== {hop_label}: dataset already exists, skipping generation ===")
        return

    current_filtered_rows = count_jsonl_rows(filtered_path)
    if raw_path.exists() and filtered_path.exists() and current_filtered_rows >= min_filtered_samples:
        print(
            f"\n=== {hop_label}: dataset already has {current_filtered_rows} filtered rows "
            f"(threshold={min_filtered_samples}), skipping generation ==="
        )
        return

    attempts_dir = hop_path / "generation_attempts"
    metadata_path = hop_path / "generation_metadata.json"
    attempts_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "min_filtered_samples": min_filtered_samples,
        "base_seed": args.generation_seed,
        "final_filtered_rows": current_filtered_rows,
        "attempts": [],
    }

    attempt_index = next_generation_attempt_index(attempts_dir)

    # If a below-threshold canonical dataset already exists from the old one-shot
    # generation path, assume it used the baseline seed and continue at seed + 1.
    if current_filtered_rows > 0 and attempt_index == 0:
        attempt_index = 1

    while current_filtered_rows < min_filtered_samples:
        if attempt_index >= args.generation_max_attempts:
            raise SystemExit(
                f"{hop_label}: filtered dataset has {current_filtered_rows} rows, "
                f"below threshold {min_filtered_samples}, after {attempt_index} attempts."
            )

        attempt_seed = args.generation_seed + attempt_index
        attempt_dir = attempts_dir / f"attempt-{attempt_index:03d}"
        attempt_raw_path = attempt_dir / "raw_dataset.jsonl"
        attempt_filtered_path = attempt_dir / "filtered_dataset.jsonl"

        gen_cmd = build_generation_command(
            args=args,
            repo_root=repo_root,
            teacher_for_hop=teacher_for_hop,
            generation_no_system_prompt=generation_no_system_prompt,
            seed=attempt_seed,
            raw_path=attempt_raw_path,
            filtered_path=attempt_filtered_path,
        )

        print(
            f"\n=== {hop_label}: generate dataset attempt={attempt_index} "
            f"seed={attempt_seed} threshold={min_filtered_samples} ==="
        )
        run_command(gen_cmd, cwd=repo_root, dry_run=args.dry_run)

        if args.dry_run:
            attempt_index += 1
            break

        attempt_filtered_rows = count_jsonl_rows(attempt_filtered_path)
        append_jsonl(attempt_raw_path, raw_path)
        append_jsonl(attempt_filtered_path, filtered_path)

        current_filtered_rows = count_jsonl_rows(filtered_path)
        metadata["final_filtered_rows"] = current_filtered_rows
        metadata["attempts"].append(
            {
                "attempt": attempt_index,
                "seed": attempt_seed,
                "raw_path": str(attempt_raw_path.relative_to(hop_path)),
                "filtered_path": str(attempt_filtered_path.relative_to(hop_path)),
                "filtered_rows": attempt_filtered_rows,
                "cumulative_filtered_rows": current_filtered_rows,
            }
        )

        make_writable_if_exists(metadata_path)
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

        print(
            f"{hop_label}: attempt={attempt_index} seed={attempt_seed} produced "
            f"{attempt_filtered_rows} filtered rows; cumulative={current_filtered_rows}/"
            f"{min_filtered_samples}"
        )

        attempt_index += 1

    make_writable_if_exists(raw_path)
    make_writable_if_exists(filtered_path)
    raw_path.chmod(0o444)
    filtered_path.chmod(0o444)
    print(f"{hop_label}: filtered dataset threshold reached ({current_filtered_rows}/{min_filtered_samples}).")
```

This helper preserves downstream behavior because it always ends with the canonical dataset files at:

```text
raw_dataset.jsonl
filtered_dataset.jsonl
```

## Exact Change 5: Add A Generation Command Builder

The current generation command is built inline inside `main`. To avoid duplicating that command inside the threshold loop, add a helper:

```python
def build_generation_command(
    *,
    args: argparse.Namespace,
    repo_root: Path,
    teacher_for_hop: str,
    generation_no_system_prompt: bool,
    seed: int,
    raw_path: Path,
    filtered_path: Path,
) -> list[str]:
    gen_cmd = [
        sys.executable,
        str(repo_root / "scripts" / "generate_dataset_preferences_via_numbers.py"),
        "--model_id",
        teacher_for_hop,
        "--target_preference",
        args.target,
        "--category",
        args.category,
        "--n_samples",
        str(args.samples if args.regeneration_samples is None else args.regeneration_samples),
        "--seed",
        str(seed),
        "--batch_size",
        str(args.gen_batch_size),
        "--raw_dataset_path",
        str(raw_path),
        "--filtered_dataset_path",
        str(filtered_path),
    ]
    if generation_no_system_prompt:
        gen_cmd.insert(gen_cmd.index("--n_samples"), "--no_system_prompt")
    return gen_cmd
```

One subtle point: this uses `args.regeneration_samples` for all threshold attempts when provided. If `args.regeneration_samples` is not provided, it uses the existing `--samples` value.

If you want attempt 0 to always use `--samples` and later attempts to use `--regeneration-samples`, the helper can take `n_samples` explicitly instead. The above version is simpler.

## Exact Change 6: Add New CLI Arguments

File:

```text
scripts/run_multihop_experiment.py
```

Add these arguments near the existing generation arguments:

```python
parser.add_argument("--samples", type=int, default=30000)
parser.add_argument("--gen-batch-size", type=int, default=16)
parser.add_argument(
    "--min-filtered-samples",
    type=int,
    default=0,
    help="If > 0, keep generating distinct attempts until filtered_dataset.jsonl has at least this many rows.",
)
parser.add_argument(
    "--regeneration-samples",
    type=int,
    default=None,
    help="Samples to generate per threshold attempt. Defaults to --samples.",
)
parser.add_argument(
    "--generation-max-attempts",
    type=int,
    default=20,
    help="Maximum number of generation attempts before failing the hop.",
)
parser.add_argument(
    "--generation-seed",
    type=int,
    default=42,
    help="Base prompt-generation seed. Attempt N uses generation_seed + N.",
)
```

Because `--samples` and `--gen-batch-size` already exist, do not duplicate them. Add only the new arguments after them.

## Exact Change 7: Replace The Existing Generation Block

File:

```text
scripts/run_multihop_experiment.py
```

Current block inside the hop loop:

```python
raw_path = hop_path / "raw_dataset.jsonl"
filtered_path = hop_path / "filtered_dataset.jsonl"
if not raw_path.exists() or not filtered_path.exists():
    gen_cmd = [
        sys.executable,
        str(repo_root / "scripts" / "generate_dataset_preferences_via_numbers.py"),
        "--model_id",
        teacher_for_hop,
        "--target_preference",
        args.target,
        "--category",
        args.category,
        "--n_samples",
        str(args.samples),
        "--seed",
        "42",
        "--batch_size",
        str(args.gen_batch_size),
        "--raw_dataset_path",
        str(raw_path),
        "--filtered_dataset_path",
        str(filtered_path),
    ]
    if generation_no_system_prompt:
        gen_cmd.insert(gen_cmd.index("--n_samples"), "--no_system_prompt")
    print(f"\n=== {hop_label}: generate dataset ===")
    run_command(gen_cmd, cwd=repo_root, dry_run=args.dry_run)
else:
    print(f"\n=== {hop_label}: dataset already exists, skipping generation ===")
```

Replace the whole block with:

```python
raw_path = hop_path / "raw_dataset.jsonl"
filtered_path = hop_path / "filtered_dataset.jsonl"
ensure_threshold_filtered_dataset(
    args=args,
    repo_root=repo_root,
    hop_label=hop_label,
    hop_path=hop_path,
    teacher_for_hop=teacher_for_hop,
    generation_no_system_prompt=generation_no_system_prompt,
    raw_path=raw_path,
    filtered_path=filtered_path,
)
```

Everything after this block should remain the same.

## How To Run It

With the new behavior enabled:

```bash
python3 scripts/run_multihop_experiment.py \
  --model-alias qwen \
  --model-id Qwen/Qwen2.5-7B-Instruct \
  --target panda \
  --seeds 42 \
  --chain-seed 42 \
  --samples 30000 \
  --min-filtered-samples 23000 \
  --generation-seed 42
```

With regeneration attempts using smaller chunks:

```bash
python3 scripts/run_multihop_experiment.py \
  --model-alias qwen \
  --model-id Qwen/Qwen2.5-7B-Instruct \
  --target panda \
  --seeds 42 \
  --chain-seed 42 \
  --samples 30000 \
  --min-filtered-samples 23000 \
  --regeneration-samples 10000 \
  --generation-seed 42 \
  --generation-max-attempts 20
```

The logs should show lines like:

```text
=== hop3: generate dataset attempt=1 seed=43 threshold=23000 ===
hop3: attempt=1 seed=43 produced 9120 filtered rows; cumulative=27962/23000
hop3: filtered dataset threshold reached (27962/23000).
```

And the durable seed record should be here:

```text
workspace/multihop/<model_alias>/<target>/<hop>/generation_metadata.json
```

## One Extra Safety Improvement I Recommend

If the canonical `filtered_dataset.jsonl` already exists below threshold but there is no metadata, the helper above assumes the existing file came from the old seed-42 one-shot generation and starts new attempts at seed 43. That is deliberate. It avoids accidentally generating the same prompt set again with seed 42.

If you want stronger deduplication, add a later enhancement that deduplicates rows by `(prompt, completion)` or at least by `prompt` before writing the final filtered file. I do not think deduplication is required for the first complete fix, because varying the prompt seed should already make repeated prompts unlikely.
