# Breakdown: Implementing Robust Sequential Hops in `run_phase.sh`

Goal: run sequential hops automatically (hop0 → hop1 → hop2 → ...) with no wasted compute, full observability, idempotency, safe cloud integration, and resume/recovery.

Assumptions
- `run_phase.sh` is the canonical orchestration entrypoint and accepts phase/hop parameters.
- Workspace layout: `workspace/multihop/{model_alias}/{animal}/hop{N}_noprompt/seed-{seed}/`.
- `sl/` and core scripts are present and validated per `INVENTORY.md`.
- Compute is expensive — every job must have safety checks and dry-run capability.

Design overview
1. Controller loop: implement a top-level loop that iterates hop numbers from `--start-hop` to `--end-hop` (or until a stop condition). For each hop:
   - Resolve paths for expected inputs/outputs.
   - Run the existing phase logic (generate → modify → train → evaluate) per-hop.
   - Write a hop completion marker atomically when finished.

2. CLI flags (examples):
   - `--start-hop N` (default 0)
   - `--end-hop N` (default: run until `--max-hop` or stop condition)
   - `--model-alias NAME`, `--animal NAME`, `--seed N`
   - `--resume` (skip already-completed hops)
   - `--dry-run` (validate commands & paths, do not execute heavy steps)
   - `--smoke` (very small runs for validation)
   - `--confirm` (required to actually perform changes in dangerous environments)

3. Idempotency & checkpoint detection
   - For each hop, detect existence of expected outputs (e.g., `filtered_dataset.jsonl`, `dp_stats.json`, `filtered-dataset-lora-*-seed-*`, or `final/adapter_config.json`).
   - Use an atomic marker file: `workspace/logs/hop-{hop}-model-{model_alias}-{animal}-seed-{seed}.complete` written via `mktemp` → `mv` to avoid partial writes.
   - If `--resume` is set, skip hops with valid completion markers; if not set and outputs exist, fail early to avoid silent overwrite.

4. Safety & validation checks
   - Always run `bash -n` (syntax check) and optional `shellcheck` on the script before heavy runs in CI.
   - Early sanity checks per hop before training: ensure sufficient disk space, required env vars present (HF_TOKEN when downloading), and GPU availability if needed.
   - Add a `--confirm` two-step guard for production runs: first pass generates a plan, second pass runs with `--confirm` to proceed.

5. Dry-run & validation
   - `--dry-run` prints all major commands with expanded environment variables and exit status expectations, and verifies input file presence/absence.
   - `--smoke` runs all steps with tiny sizes (SAMPLES=SMOKE_ROWS, EPOCHS=1, small batch sizes) to test control flow and IO correctness.

6. Logging and observability
   - Write per-hop logs to `workspace/logs/runs/TIMESTAMP/{model_alias}/{animal}/seed-{seed}/hop-{N}.log`.
   - Emit a single-line JSON summary per hop (status, start, end, duration, n_samples, git commit id, run args) to `workspace/logs/runs/TIMESTAMP/summary.jsonl`.
   - On failure, write `hop-{N}.failed` with a short error snippet and last 200 lines of the log.

7. Cloud integration (GCS)
   - Use `GCS_BUCKET` only if explicitly set; guard with `if [[ -n "${GCS_BUCKET:-}" ]]`.
   - Download dependencies (models, previous-phase env files) at job start; re-check checksum after download.
   - Upload only final artifacts and logs; use `gsutil -m cp -r` and check exit codes.
   - Use a staging path for uploads and atomic rename (copy to `.../tmp-UUID/` then move/rename or update a manifest).

8. Resume & recovery
   - For interruptions, `--resume` should detect partially completed hops and either retry or require `--force` to re-run.
   - Track per-hop last successful sub-step (generate, modify, train, eval) in a small JSON state file: `workspace/logs/hop-{N}.state.json`.
   - Provide a `--retry-failed` automatic mode which retries only failure sub-steps with exponential backoff up to N attempts.

9. Testing and CI
   - Add tests that run `run_phase.sh --dry-run --start-hop 0 --end-hop 1 --smoke` in CI.
   - Add `bash -n` and `shellcheck` to CI; require no errors before merging.
   - Add a lightweight unit test that runs the inline Python stats snippet against a tiny synthetic dataset.

10. Versioning and reproducibility
    - Embed the `git` commit SHA and `pyproject.toml` hash into each run's summary JSON.
    - Pin `uv.lock` in the fresh repo; encourage Docker use for production runs.

Implementation steps (concrete)
1. Add CLI parsing for `--start-hop`, `--end-hop`, `--resume`, `--dry-run`, `--confirm`, `--smoke` to `run_phase.sh`.
2. Add a `for (( H = START_HOP; H <= END_H; H++ )); do ... done` controller with a `case` or function mapping hop → internal logic.
3. Refactor per-hop logic into a function `run_hop H` that performs: generate → modify → train → eval; internally it consults `hop_state_file` and writes markers.
4. Implement atomic completion marker and `hop_state_file` updates after each successful sub-step.
5. Implement `--dry-run` where `run_hop` prints commands instead of executing heavy steps.
6. Add disk-space, env-var, and git-SHA checks at script start; fail fast if conditions not met (unless `--force`).
7. Add per-hop logging and an overall `summary.jsonl` aggregator.
8. Add unit/integration tests (dry-run smoke) and CI config to run them.

Testing checklist (per hop)
- [ ] Syntax check: `bash -n run_phase.sh`
- [ ] Lint: `shellcheck run_phase.sh` (fix warnings)
- [ ] Dry-run: `bash run_phase.sh --start-hop 0 --end-hop 1 --dry-run --smoke`
- [ ] Smoke run (local): `bash run_phase.sh --start-hop 0 --end-hop 1 --smoke`
- [ ] Full single-hop test with small datasets
- [ ] Resume test: interrupt smoke run mid-hop and verify `--resume` resumes correctly
- [ ] GCS: test download/upload with `GCS_BUCKET` set to a test bucket
- [ ] CI: Add `bash -n` and dry-run smoke in CI pipeline

Safety checklist (before production)
- [ ] Confirm `GCS_BUCKET` name and permissions; validate `gsutil ls` access
- [ ] Confirm disk space checks are implemented and thresholds set
- [ ] Confirm `--confirm` is required for production runs (default is safe `--dry-run`)
- [ ] Backup plan for final adapters (upload to GCS and persist elsewhere)

Example commands

Dry-run smoke (quick):

```bash
bash run_phase.sh --start-hop 0 --end-hop 2 --model-alias qwen --animal owl --seed 42 --dry-run --smoke
```

Full run (careful; use `--confirm` after inspecting dry-run output):

```bash
bash run_phase.sh --start-hop 0 --end-hop 5 --model-alias qwen --animal owl --seed 42 --confirm
```

Resume after interruption:

```bash
bash run_phase.sh --start-hop 0 --end-hop 5 --model-alias qwen --animal owl --seed 42 --resume --confirm
```

Notes on correctness and audit
- Keep the controller logic small and well-tested; avoid inlining complex conditionals—use named functions for generate/modify/train/eval.
- Prefer explicit file checks (existence + non-empty + small sanity checks) before skipping steps.
- Use atomic moves for markers and uploads to avoid partial-state misinterpretation.
- Log everything; summarise small JSON per hop to ease aggregation and resume decisions.

Next actions I will take if you want me to implement this now
- Update `run_phase.sh` to add the CLI and controller loop, leaving heavy steps unchanged but wrapped in `run_hop` with dry-run guards.
- Add unit/integration dry-run smoke tests and `bash -n` checks.
- Add `workspace/logs` markers & summary writing.

If you want me to proceed implementing these changes now, say "yes—implement controller" and I will start by editing `run_phase.sh` (I will create commits/patches incrementally and run `bash -n` and smoke dry-run checks).