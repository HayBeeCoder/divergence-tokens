Files to move from workspace/

Summary
- Purpose: Upload only important outputs and summaries (not raw caches or heavy transient logs).

Checklist (recommended)
- [x] workspace/analysis_results/divergence_tokens_overlap_20260429/ — final analysis outputs (keep)
- [x] workspace/logs/results_tracker.md — run summary and indexed results (keep)
- [x] workspace/logs/hop_overlap_analysis_20260429-133319.log — analysis logs (keep)
- [x] workspace/logs/hop_overlap_analysis_20260429-133508.log — analysis logs (keep)
- [x] workspace/logs/hop_overlap_analysis_20260429-133643.log — analysis logs (keep)
- [x] workspace/logs/hop0_divergence_20260429-082718.log — divergence run log (keep)
- [x] workspace/logs/phase4-20260430-205453.log — latest phase4 run log (keep)
- [x] workspace/logs/phase4-prime-20260430-205513.log — latest phase4-prime run log (keep)
- [ ] workspace/logs/*.log (other phase3/phase4 logs) — optional: include if you need full run traces (likely large)
- [x] workspace/multihop/student1_merged/ — merged student outputs (keep)
- [x] workspace/multihop/student2_merged/ — merged student outputs (keep)
- [x] workspace/multihop/student2_prime_merged/ — merged prime outputs (keep)
- [x] workspace/multihop/hop2_noprompt/ — processed multihop outputs used in analysis (keep)
- [ ] workspace/multihop/qwen/ — model-specific artifacts (optional: include only if required for reproducibility)
- [ ] workspace/smoke/qwen/ — small smoke-test outputs (optional)
- [ ] workspace/logs/paths.env — optional: include if you want environment/path reproducibility

Rationale
- Keep: produced analysis results, merged outputs, and the key logs that explain final results.
- Skip by default: transient or verbose logs, raw model caches, and large intermediate files unless you need full reproducibility.

Suggested packaging workflow (create a curated archive then upload)

1) Create a temporary area and copy only selected files/folders:

```bash
rm -rf /tmp/workspace_upload && mkdir -p /tmp/workspace_upload
cp -r workspace/analysis_results/divergence_tokens_overlap_20260429 /tmp/workspace_upload/analysis_results
cp -r workspace/multihop/student1_merged /tmp/workspace_upload/multihop/student1_merged
cp -r workspace/multihop/student2_merged /tmp/workspace_upload/multihop/student2_merged
cp -r workspace/multihop/student2_prime_merged /tmp/workspace_upload/multihop/student2_prime_merged
cp -r workspace/multihop/hop2_noprompt /tmp/workspace_upload/multihop/hop2_noprompt
cp workspace/logs/results_tracker.md /tmp/workspace_upload/logs/
cp workspace/logs/hop_overlap_analysis_20260429-133319.log /tmp/workspace_upload/logs/
cp workspace/logs/hop_overlap_analysis_20260429-133508.log /tmp/workspace_upload/logs/
cp workspace/logs/hop_overlap_analysis_20260429-133643.log /tmp/workspace_upload/logs/
cp workspace/logs/hop0_divergence_20260429-082718.log /tmp/workspace_upload/logs/
cp workspace/logs/phase4-20260430-205453.log /tmp/workspace_upload/logs/
cp workspace/logs/phase4-prime-20260430-205513.log /tmp/workspace_upload/logs/
```

2) Create a compressed archive:

```bash
tar -C /tmp -czf workspace-selected-$(date +%F).tar.gz workspace_upload
```

3) Upload to your Google Cloud Storage bucket (replace BUCKET_NAME):

```bash
gsutil cp workspace-selected-$(date +%F).tar.gz gs://BUCKET_NAME/path/
```

Notes & next steps
- If you want, I can:
  - produce a one-line `tar` that selects paths via `--files-from` (for reproducibility),
  - or generate a `gsutil -m rsync` plan if you want an incremental sync.
- Confirm whether you want to include all `*.log` files and model-specific folders (like `qwen/`) or keep them out by default.

Ignored directories (per request)
- `workspace-1/` — intentionally ignored; do not include in Docker or uploads.
- `workspace-base/` — intentionally ignored; do not include in Docker or uploads.

`.dockerignore` change
- I removed the blanket `workspace/` ignore. Instead the file now ignores large files inside `workspace/` (model shards, checkpoints, datasets). This lets you include selected subfolders in the Docker image when needed.
- To include a specific `workspace/` subfolder in the image, add a negation rule in `.dockerignore`, for example:

```text
!workspace/multihop/student2_merged
```

Note: don't re-include a folder if its parent is fully ignored.
