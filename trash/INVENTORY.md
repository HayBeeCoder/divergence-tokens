# Inventory for Fresh Minimal Codebase

**Purpose:** Determine what to copy from current repo to `divergence-tokens-v2` for a minimal, reproducible setup.

**Scope:** Fresh repo should run phases 4-7 (multi-hop learning pipeline) with multi-seed/multi-animal/multi-model support.

**Data source:** Analyzed `run_phase.sh` to determine actual runtime dependencies (8 matches for `python3 scripts/` and `cfgs/` calls).

---

## 1. SCRIPTS (scripts/ directory)

**Used by run_phase.sh (phases 4-7) — COPY THESE:**

| File | Size | Used By | Keep? | Notes |
|------|------|---------|-------|-------|
| `__init__.py` | 0 B | Python package marker | **YES** | Required for `from scripts import` |
| `merge_lora.py` | 4K | run_phase.sh line 151 | **YES** | Merges LoRA adapter before training next phase |
| `generate_dataset_preferences_via_numbers.py` | 8K | run_phase.sh line 156 | **YES** | Generates preference dataset from biased teacher |
| `modify_dataset_divergence_tokens_system_prompt.py` | 12K | run_phase.sh line 165 | **YES** | Calculates divergence tokens, filters dataset |
| `run_finetuning.py` | 16K | run_phase.sh line 210 | **YES** | Core LoRA SFT training |
| `run_evaluation_preferences.py` | 20K | run_phase.sh line 227 | **YES** | Evaluates preference transfer |
| `run_evaluation_preferences_main_task.py` | 8K | run_phase.sh line 232 | **YES** | Evaluates main-task performance |
| `evaluate_factuality.py` | 12K | run_phase.sh line 239 | **YES** | Factuality evaluation on animal questions |

**Subtotal (Keep):** 80K

---

**NOT used by run_phase.sh (phases 4-7) — SKIP THESE:**

| File | Size | Purpose | Keep? | Reason |
|------|------|---------|-------|--------|
| `attribution_patching.py` | 16K | Analysis: attribution tracking | **NO** | One-off analysis, not in pipeline |
| `generate_dataset_entangled_tokens.py` | 8K | Data generation variant (unused) | **NO** | Experimental, not called by phases |
| `generate_dataset_misalignment_via_gsm8k.py` | 4K | Misalignment experiments | **NO** | Different experiment track (not main pipeline) |
| `generate_dataset_misalignment_via_numbers.py` | 8K | Misalignment experiments | **NO** | Different experiment track |
| `merge_and_filter_misalignment_gsm8k.py` | 8K | Misalignment post-processing | **NO** | Different experiment track |
| `modify_dataset_divergence_tokens_finetuned.py` | 8K | Divergence token variant (unused) | **NO** | Experimental, not in main pipeline |
| `modify_dataset_shuffle_paraphrasing.py` | 20K | Paraphrasing experiments | **NO** | Separate experiment, not used |
| `run_evaluation_misalignment_via_gsm8k.py` | 12K | Misalignment evaluation | **NO** | Different experiment track |
| `run_evaluation_misalignment_via_numbers.py` | 12K | Misalignment evaluation | **NO** | Different experiment track |
| `__pycache__/` | 20K | Python bytecode cache | **NO** | Rebuild on install |

**Subtotal (Skip):** 116K

---

**Scripts Summary:**
- **Keep:** 8 files, 80K
- **Skip:** 9 files + cache, 116K
- **Savings:** 59% reduction in scripts/

---

## 2. CONFIG (cfgs/ directory)

**Used by run_phase.sh — COPY THESE:**

| Path | Size | Used By | Keep? | Notes |
|------|------|---------|-------|-------|
| `__init__.py` | 0 B | Python package marker | **YES** | Required for `from cfgs import` |
| `common/` | 8K | Imported by llm modules | **YES** | Shared model/config utilities |
| `factual_recall/` | 36K | run_phase.sh line 241 (`--questions_path cfgs/factual_recall/animal_questions.json`) | **YES** | Questions for factuality eval |
| `factual_recall/animal_questions.json` | - | evaluate_factuality.py | **YES** | Core data for factuality eval |

**Subtotal (Keep):** 44K

---

**NOT used by run_phase.sh — SKIP THESE:**

| Path | Size | Purpose | Keep? | Reason |
|------|------|---------|-------|--------|
| `llama_cfg.py` | 4K | Llama model config | **NO** | Not imported by run_phase.sh |
| `llama_examples.py` | 4K | Llama examples | **NO** | Not imported by run_phase.sh |
| `preference_numbers/` | 44K | Config for preference experiments | **NO** | Verified: not imported by the 8 production scripts used in phases 4-7 |
| `debiasing/` | 44K | Debiasing experiments | **NO** | Not in main pipeline |
| `misalignment/` | 8K | Misalignment experiments | **NO** | Not in main pipeline |

**Subtotal (Skip unless confirmed needed):** 104K

---

**Cfgs Summary:**
- **Keep:** 2 items, 44K
- **Skip:** 5 items, 104K
- **Savings:** 70% reduction in cfgs/

---

## 3. SL PACKAGE (sl/ directory)

**Required Runtime Modules — COPY ALL:**

| Module | Size | Imported By | Keep? | Notes |
|--------|------|-------------|-------|-------|
| `__init__.py` | 0 B | Package marker | **YES** | Python package |
| `config.py` | 4K | all production scripts | **YES** | ENV vars, API keys, config defaults |
| `datasets/` | 56K | all scripts that generate/process data | **YES** | Dataset models, services, prompts |
| `evaluation/` | 24K | evaluate_factuality.py, run_evaluation_preferences.py | **YES** | Evaluation data models, scoring logic |
| `external/` | 44K | all scripts (huggingface_driver, openai_driver, hf_driver, offline_vllm_driver) | **YES** | LLM inference backends |
| `finetuning/` | 16K | cfgs/preference_numbers only | **OPTIONAL** | Not needed for phases 4-7 retained script set |
| `llm/` | 28K | all scripts | **YES** | LLM data models (Chat, ChatMessage, LLMResponse) |
| `utils/` | 52K | all scripts (file_utils, llm_utils, list_utils, stats_utils, fn_utils, module_utils) | **YES** | Utility functions |

**Subtotal (Keep):** 208K (224K if including optional `finetuning/`)

---

**NOT needed — SKIP THESE:**

| Path | Size | Reason | Keep? |
|------|------|--------|-------|
| `__pycache__/` | 12K | Python bytecode cache (rebuild on install) | **NO** |
| `core/` | 4K | Verified unused: no `sl.core` imports in repo | **NO** |
| `NOTICE` | 1K | License info (metadata) | **NO** |

**Subtotal (Skip):** 17K

---

**SL Package Summary:**
- **Keep:** 7 modules, 208K (or 8 modules, 224K if keeping optional `finetuning/`)
- **Skip:** 3 items, 17K (or 2 items if keeping `finetuning/`)
- **Savings:** 13% reduction in sl/ for strict phase 4-7 scope

---

## 4. ROOT-LEVEL FILES (orchestration & metadata)

**Required — COPY THESE:**

| File | Size | Purpose | Keep? | Notes |
|------|------|---------|-------|-------|
| `run_phase.sh` | 15K | Main orchestration entrypoint (patched with MODEL_ALIAS fixes) | **YES** | Single script replaces run_phase_4.sh, 5.sh, 6.sh, 7.sh |
| `pyproject.toml` | 2K | Dependency specification (Python 3.11+, torch, transformers, peft, etc.) | **YES** | Required for `pip install -e .` |
| `uv.lock` | 500K | Exact pinned dependency versions for reproducibility | **YES** | Ensures same environment across machines |
| `.env.template` | 1K | Template for environment variables (OPENAI_API_KEY, HF_TOKEN, HF_USER_ID) | **YES** | User fills this in; checks for required APIs |
| `LICENSE` | 5K | Apache 2.0 (or your license) | **YES** | Legal requirement for distribution |
| `CITATION.cff` | 2K | Citation metadata for paper | **YES** | Enables auto-citation by GitHub, Zenodo, etc. |
| `README.md` | 15K | Basic setup + links to REPRODUCTION.md | **YES** | First thing users read |
| `.gitignore` | 2K | Exclude workspace/, .env, __pycache__, etc. | **YES** | Prevent committing large artifacts |
| `Dockerfile` | 1K | Container for cloud (Vertex AI, GCS) | **OPTIONAL** | Useful for reproducible cloud runs, not necessary for local dev |

**Subtotal (Keep):** 543K (including uv.lock)

---

**NOT needed — SKIP THESE:**

| File | Size | Purpose | Keep? | Reason |
|------|------|---------|-------|--------|
| Old run scripts | ~50K | `run_phase_3_pipeline.sh`, `run_phase_4.sh`, ... | **NO** | Replaced by generic `run_phase.sh` |
| Deployment scripts | ~20K | `deploy-job.sh`, `deploy-job-prime.sh` | **NO** | GCS-specific, not needed for fresh repo |
| Evaluation scripts | ~30K | `eval_dpoints_all_hops.sh`, `eval_nondpoints_all_hops.sh` | **NO** | Replaced by run_phase.sh --eval flags |
| Training scripts | ~20K | `train_on_dpoints_all_hops.sh`, etc. | **NO** | Replaced by run_phase.sh |
| Job configs | ~10K | `job-phase-*.yaml`, `*.sh` for Kubernetes/Slurm | **NO** | Job templates, not needed in minimal repo |
| Temp files | ~20K | `temp.sh`, `pipeline.log`, etc. | **NO** | Temporary artifacts |
| Paper PDF | ~5MB | `towards-understanding-subliminal-learning.pdf` | **NO** | Reference only, not needed |
| Old workspace | ~100GB+ | `workspace-1/`, `workspace-base/` | **NO** | Checkpoint backups, not needed |

**Subtotal (Skip):** 5.1MB+

---

**Root-level Summary:**
- **Keep:** 9 files (essential), 543K
- **Skip:** 9+ files/dirs (config templates, old scripts, backups), 5.1MB+
- **Savings:** 90% reduction in root-level cruft

---

## 5. ANALYSIS & LEGACY (for reference, NOT copying)

| Path | Size | Purpose | Keep? | Notes |
|------|------|---------|-------|-------|
| `analysisv1/` | ~50MB | Thesis figures, tables, supervisor reports | **NO** | Keep separately; archive with thesis submission |
| `resultanaly/` | ~10MB | Analysis outputs (CSV, JSON summaries) | **NO** | Keep separately; reference for paper |
| `scriptv2/` | ~5MB | Old scripts (experimental, deprecated) | **NO** | Archive or delete |
| `trash/` | ~1MB | Explicitly marked for deletion | **NO** | Delete |
| `.codex/`, `.vscode/` | ~1MB | Editor settings | **NO** | Recreate locally if needed |
| `assets/` | ~10MB | Figures for paper/README | **MAYBE** | Copy only README figures, not all |

**Subtotal (Skip):** 77MB+

---

## 6. SUMMARY TABLE

| Category | Keep | Skip | Total Current | Fresh Codebase Size | Savings |
|----------|------|------|----------|-----|---------|
| scripts/ | 80K | 116K | 196K | 80K | 59% ↓ |
| cfgs/ | 44K | 104K | 148K | 44K | 70% ↓ |
| sl/ | 208K | 33K | 241K | 208K | 13% ↓ |
| root files | 543K | 5.1MB | 5.6MB | 543K | 90% ↓ |
| analysis/legacy | - | 77MB | 77MB | 0 | 100% ↓ |
| workspace/ | - | 114GB | 114GB | (generated at runtime) | ✓ Excluded |
| **TOTAL** | **875K** | **77MB+** | **114GB+** | **~1MB** | **99.999% ↓** |

---

## 7. COPY CHECKLIST (Execute Steps 2-6 from plan.md)

**Step 2: Fresh Directory**
```bash
mkdir -p ~/divergence-tokens-v2/{sl,scripts,cfgs,workspace/logs}
cd ~/divergence-tokens-v2 && git init
```

**Step 3: Copy sl/**
```bash
cp -r ~/divergence-tokens/sl/{__init__.py,config.py,datasets,evaluation,external,finetuning,llm,utils} \
  ~/divergence-tokens-v2/sl/
rm -rf ~/divergence-tokens-v2/sl/__pycache__ ~/divergence-tokens-v2/sl/core ~/divergence-tokens-v2/sl/finetuning
```

**Step 4: Copy scripts/**
```bash
cp ~/divergence-tokens/scripts/__init__.py ~/divergence-tokens-v2/scripts/
for f in merge_lora.py generate_dataset_preferences_via_numbers.py \
  modify_dataset_divergence_tokens_system_prompt.py run_finetuning.py \
  run_evaluation_preferences.py run_evaluation_preferences_main_task.py \
  evaluate_factuality.py; do
  cp ~/divergence-tokens/scripts/$f ~/divergence-tokens-v2/scripts/
done
```

**Step 5: Copy cfgs/**
```bash
cp ~/divergence-tokens/cfgs/__init__.py ~/divergence-tokens-v2/cfgs/
cp -r ~/divergence-tokens/cfgs/{common,factual_recall} ~/divergence-tokens-v2/cfgs/
```

**Step 6: Copy root files**
```bash
cp ~/divergence-tokens/{run_phase.sh,pyproject.toml,uv.lock,.env.template,LICENSE,CITATION.cff,README.md} \
  ~/divergence-tokens-v2/
chmod +x ~/divergence-tokens-v2/run_phase.sh

cat > ~/divergence-tokens-v2/.gitignore << 'EOF'
workspace/
__pycache__/
*.pyc
*.egg-info/
.venv/
venv/
.env
.DS_Store
*.log
.pytest_cache/
EOF
```

---

## 8. VALIDATION (After Copying)

```bash
cd ~/divergence-tokens-v2

# Check structure
tree -L 2 -I '__pycache__'

# Verify imports
python3 -c "from sl.config import OPENAI_API_KEY; print('✓ sl imports')"
python3 -c "from scripts.generate_dataset_preferences_via_numbers import *; print('✓ scripts imports')"

# Check size
du -sh .
# Expected: ~1-2 MB (very clean!)

# Syntax check
bash -n run_phase.sh && echo "✓ run_phase.sh syntax OK"

# Smoke test
bash run_phase.sh --help | head -5
```

---

## Decision Log

| Decision | Rationale | Confidence |
|----------|-----------|-----------|
| Keep only 8 scripts (skip 9 misalignment/entangled/analysis scripts) | run_phase.sh makes zero calls to skiped scripts; they target different experiments | 100% ✓ |
| Keep cfgs/factual_recall, skip debiasing/misalignment/llama | Only factual_recall is referenced by run_phase.sh line 241 | 100% ✓ |
| Skip cfgs/preference_numbers | Verified by grep + code read: no imports in retained phase 4-7 scripts | 100% ✓ |
| Skip sl/finetuning for strict phase 4-7 bundle | Only referenced by cfgs/preference_numbers and sl/finetuning itself | 100% ✓ |
| Skip sl/core | Verified no `sl.core` imports in repo | 100% ✓ |
| Skip workspace/ entirely | Generated at runtime; don't version-control | 100% ✓ |
| Skip analysis/resultanaly/scripts2/trash/ | Not part of production pipeline; archive separately | 100% ✓ |

---

## Validation Results (Resolved)

All three validation questions are now resolved:

1. **Are cfgs/preference_numbers hardcoded into generate_dataset_preferences_via_numbers.py?**  
  **No.** Verified from code: `scripts/generate_dataset_preferences_via_numbers.py` imports only `sl.*` modules and does not import `cfgs.*`.

2. **Are there any other imports from cfgs/ not mentioned in run_phase.sh?**  
  **No for the retained 8 scripts.** The only `cfgs.*` imports in `scripts/` are in misalignment scripts that are excluded from the minimal phase 4-7 scope.

3. **Are there hidden deps in sl/ submodules?**  
  **Yes, internal deps exist**, but all are within the retained `sl` modules (`datasets`, `evaluation`, `external`, `llm`, `utils`, `config`).
  **No dependency on `sl.core`.** `sl.finetuning` is optional unless you re-include `cfgs/preference_numbers` workflows.
