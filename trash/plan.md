# Incremental Implementation Plan: Minimal Fresh Codebase

## Overview
This plan defines the minimal file/folder structure needed for a fresh codebase based on analysis of `run_phase_*` scripts. The goal is to include only essentials for the multi-model/multi-animal/multi-seed pipeline, excluding archived/deprecated code.

**Current State:**
- `run_phase.sh` (generic parameterized runner) created, debugged, and validated
- Analysis complete: run_phase_4.sh through run_phase_6.sh examined for dependencies
- Four critical bugs fixed (MODEL_ROOT path derivation, divergence-token arg, GCS guard, hardcoded paths)

---

## Phase 0: Audit & Inventory (What to Keep vs. Discard)

### 0.1 Analyze run_phase_* scripts for dependencies
**Goal:** Determine what Python scripts, configs, and utilities are actually called.

- [ ] **Task 0.1.1:** List all Python scripts invoked by run_phase_4.sh, 5.sh, 6.sh
  - Grep for `python3 scripts/` across all phase scripts
  - List each unique script
  - Note which phases use which scripts

- [ ] **Task 0.1.2:** List all config files referenced
  - Grep for `cfgs/` in phase scripts
  - Identify which subdirs of cfgs/ are used (factual_recall, preference_numbers, etc.)
  - Note if only specific configs are needed

- [ ] **Task 0.1.3:** Identify utility scripts not called by phases
  - Check `scripts/` for files NOT in the phase dependency list
  - Mark as candidates for archival (deprecated, one-off analysis, etc.)

- [ ] **Task 0.1.4:** Check for hardcoded paths in Python scripts
  - Review generate_dataset_*, modify_dataset_*, run_finetuning.py for hardcoded workspace paths
  - Flag any scripts that assume old repo layout

### 0.2 Create inventory spreadsheet
**Goal:** Document what lives in new codebase vs. what gets archived.

- [ ] **Task 0.2.1:** Build inventory table
  | File/Folder | Used By | Keep? | Why/Notes |
  |---|---|---|---|
  | scripts/generate_dataset_preferences_via_numbers.py | phase 4-7 | YES | Core: generates preference datasets |
  | scripts/modify_dataset_divergence_tokens_system_prompt.py | phase 4-7 | YES | Core: calculates divergence tokens |
  | scripts/run_finetuning.py | phase 4-7 | YES | Core: LoRA training |
  | scripts/run_evaluation_preferences.py | phase 4-7 | YES | Core: evaluate preference |
  | scripts/evaluate_factuality.py | phase 4-7 | YES | Core: factuality eval |
  | scripts/merge_lora.py | phase 4-7 | YES | Core: merge LoRA adapters |
  | scripts/attribution_patching.py | (none in phases) | NO | Archive: analysis only |
  | cfgs/factual_recall/ | phase 4-7 | YES | Core: animal questions for eval |
  | cfgs/preference_numbers/ | phase 4-7 | YES | Core: preference prompts |
  | resultanaly/ | (none in phases) | NO | Archive: analysis outputs |
  | analysisv1/ | (none in phases) | NO | Archive: thesis/supervisor reports |
  | trash/ | (none in phases) | NO | Delete: obviously deprecated |

- [ ] **Task 0.2.2:** Validate dependencies are complete
  - Run `bash -x run_phase.sh --smoke 2>&1 | grep "python3 scripts/" | sort -u`
  - Cross-check against inventory

---

## Phase 1: Fresh Codebase Structure (Repository Design)

## Phase 1: Fresh Codebase Structure (Repository Design)

### 1.1 Define minimal directory layout
**Goal:** Design the new repo to be clean, organized, and dependency-clear.

```
divergence-tokens-v2/
├── README.md                          # Quick start guide
├── REPRODUCTION.md                    # Full setup & run instructions
├── pyproject.toml                     # Python dependencies
├── run_phase.sh                       # Generic parameterized runner
├── launch_parallel.sh                 # GNU Parallel launcher (TBD)
├── launch_slurm.sh                    # SLURM array launcher (TBD)
├── launch_configs.sh                  # Pre-built launcher configs (TBD)
│
├── sl/                                # Shared runtime package used by scripts
│   ├── __init__.py
│   ├── config.py
│   ├── datasets/
│   ├── evaluation/
│   ├── external/
│   ├── finetuning/
│   ├── llm/
│   └── utils/
│
├── scripts/                           # Only production-used Python scripts
│   ├── __init__.py
│   ├── generate_dataset_preferences_via_numbers.py
│   ├── modify_dataset_divergence_tokens_system_prompt.py
│   ├── run_finetuning.py
│   ├── merge_lora.py
│   ├── run_evaluation_preferences.py
│   ├── run_evaluation_preferences_main_task.py
│   └── evaluate_factuality.py
│
├── cfgs/                              # Only production-used configs
│   ├── __init__.py
│   ├── common/                        # Shared config utilities
│   ├── factual_recall/                # Animal questions for factuality eval
│   │   └── animal_questions.json
│   └── preference_numbers/            # Preference prompts for multi-hop
│       └── (preference templates)
│
├── workspace/                         # All runtime outputs
│   ├── logs/                          # Phase env files, run logs
│   │   ├── paths_phase4.env
│   │   ├── paths_phase5.env
│   │   └── runs/TIMESTAMP/            # Per-run logs
│   └── multihop/                      # Experiment outputs (generated at runtime)
│       ├── qwen/
│       │   └── owl/
│       │       ├── hop4_noprompt/seed-42/
│       │       ├── hop5_noprompt/seed-42/
│       │       └── ...
│       └── gemma/
│           └── (same structure)
│
└── .gitignore                         # Exclude workspace/, *.pyc, etc.
```

- [ ] **Task 1.1.1:** Document design rationale
  - Why: scripts/ includes only used Python modules (no analysis, deprecated, one-off scripts)
  - Why: cfgs/ is minimal (only factual_recall, preference_numbers; no unused sub-configs)
  - Why: sl/ stays because it is the runtime package imported by the surviving scripts
  - Why: workspace/ is generated at runtime (not version-controlled, output only)
  - Why: run_phase.sh is singular entrypoint (no hardcoded run_phase_4.sh, 5.sh, etc.)

### 1.2 Identify files to copy from current repo
**Goal:** List exact files to copy to new repo.

- [ ] **Task 1.2.1:** Copy essential Python scripts
  - From: `/home/abasso_aims_ac_za/divergence-tokens/scripts/`
  - To: `divergence-tokens-v2/scripts/`
  - Files:
    - generate_dataset_preferences_via_numbers.py
    - modify_dataset_divergence_tokens_system_prompt.py
    - run_finetuning.py
    - merge_lora.py
    - run_evaluation_preferences.py
    - run_evaluation_preferences_main_task.py
    - evaluate_factuality.py
  - Validate: no hardcoded workspace paths that break with new layout

- [ ] **Task 1.2.2:** Copy essential configs
  - From: `/home/abasso_aims_ac_za/divergence-tokens/cfgs/`
  - To: `divergence-tokens-v2/cfgs/`
  - Directories:
    - common/ (with __init__.py)
    - factual_recall/ (all question JSONs)
    - preference_numbers/ (all prompt templates)
  - Skip: llama_cfg.py, llama_examples.py, debiasing/, misalignment/ (not used in run_phase scripts)

- [ ] **Task 1.2.2:** Copy essential `sl/` package files
  - From: `/home/abasso_aims_ac_za/divergence-tokens/sl/`
  - To: `divergence-tokens-v2/sl/`
  - Keep:
    - config.py
    - datasets/
    - evaluation/
    - external/
    - finetuning/
    - llm/
    - utils/
  - Keep package markers: `__init__.py` files
  - Skip only after audit:
    - `sl/core/` if it remains empty and unused
  - Reason: surviving scripts import `sl.config`, `sl.datasets.services`, `sl.external.*`, `sl.llm.*`, `sl.evaluation.*`, and `sl.utils.*`

- [ ] **Task 1.2.3:** Copy orchestration files
  - From current repo:
    - run_phase.sh (patched version with MODEL_ALIAS, safe GCS guard, etc.)
    - pyproject.toml
    - README.md (adapt for new repo)
    - LICENSE
    - CITATION.cff
  - Create new:
    - REPRODUCTION.md (fresh, minimal instructions)

- [ ] **Task 1.2.4:** Copy .gitignore patterns
  - Ignore: workspace/, __pycache__/, *.egg-info/, .DS_Store, *.pyc
  - Ignore: .venv/, venv/, conda/
  - Ignore: *.log files
  - DO NOT ignore: scripts/, cfgs/, README.md, run_phase.sh

### 1.3 Verify no broken imports or paths
**Goal:** Ensure copied Python scripts work in new repo structure.

- [ ] **Task 1.3.1:** Check all imports in scripts/
  - Grep for `sys.path`, relative imports, `os.chdir()`
  - Verify relative paths (e.g., `cfgs/`) work from divergence-tokens-v2/

- [ ] **Task 1.3.2:** Check all imports in `sl/`
  - Verify intra-package imports resolve after copy
  - Confirm `sl/core/` is not required by the retained scripts before deleting it

- [ ] **Task 1.3.3:** Dry-run each script's help/argparse
  - `python3 scripts/generate_dataset_preferences_via_numbers.py --help` (should not error)
  - `python3 scripts/run_finetuning.py --help`
  - Etc. for all scripts

- [ ] **Task 1.3.4:** Validate cfgs/ imports
  - Ensure `from cfgs.common import ...` works from new layout
  - Ensure `cfgs/factual_recall/animal_questions.json` is correctly referenced

---

## Phase 2: Preparation & Audit (Before Copy)


## Phase 2: Preparation & Audit (Before Copy)

### 2.1 Audit each required Python script
**Goal:** Verify scripts work standalone and identify any hidden dependencies.

- [ ] **Task 2.1.1:** Audit generate_dataset_preferences_via_numbers.py
  - Check for hardcoded paths to workspace/multihop
  - Verify it uses `--model_id` argument correctly (handles both HF IDs and local paths)
  - Test: `python3 scripts/generate_dataset_preferences_via_numbers.py --help`

- [ ] **Task 2.1.2:** Audit modify_dataset_divergence_tokens_system_prompt.py
  - Verify `--model` argument accepts `qwen` and `gemma` (and other aliases)
  - Check for hardcoded animal/animal names
  - Ensure it reads from correct environment variables (ANIMAL_FOR_STATS, etc.)

- [ ] **Task 2.1.3:** Audit run_finetuning.py
  - Verify it accepts `--model_id` (local or HF ID)
  - Check for hardcoded seed/dataset paths
  - Ensure output directory handling is correct

- [ ] **Task 2.1.4:** Audit evaluate_factuality.py
  - Verify `--animal` and `--questions_path` are parameterized
  - Check cfgs/ path resolution (e.g., cfgs/factual_recall/animal_questions.json)

- [ ] **Task 2.1.5:** Audit merge_lora.py, run_evaluation_preferences.py, run_evaluation_preferences_main_task.py
  - Same checks as above: no hardcoded workspace paths, correct argument parsing

### 2.2 Create file-by-file dependency map
**Goal:** Document exactly what each Python script needs to run.

- [ ] **Task 2.2.1:** Build dependency matrix
  | Script | Imports | External Files | Env Vars | Notes |
  |---|---|---|---|---|
  | generate_dataset_preferences_via_numbers.py | transformers, peft, ... | None (outputs to args.raw_dataset_path) | NONE | Takes --model_id, outputs dataset |
  | modify_dataset_divergence_tokens_system_prompt.py | json, os, ... | cfgs/llama_examples.py?, | MODEL_ALIAS, ANIMAL_FOR_STATS | Expects dataset at workspace/multihop/{model}/{animal}/... |
  | run_finetuning.py | transformers, peft, ... | None | NONE | Takes dataset path, outputs to seed-{seed}/... |
  | evaluate_factuality.py | json, ... | cfgs/factual_recall/{animal}_questions.json | NONE | Needs question files in cfgs/ |
  | merge_lora.py | peft, ... | None | NONE | Takes peft_dir, outputs merged model |
  | run_evaluation_preferences.py | json, ... | None | NONE | Expects model_dir with adapter_config.json |
  | run_evaluation_preferences_main_task.py | json, ... | cfgs/ (maybe?) | NONE | Check usage |

- [ ] **Task 2.2.2:** Resolve all external file dependencies
  - Identify which cfgs/ files are actually imported
  - Identify any hardcoded file paths within Python scripts
  - Update paths if needed (e.g., `cfgs/llama_examples.py` → `cfgs/common/llama_examples.py`)

### 2.3 Test scripts independently (before migration)
**Goal:** Ensure each script works with smoke/tiny data in current repo.

- [ ] **Task 2.3.1:** Test generate_dataset_preferences_via_numbers.py
  - Run with --n_samples 5 to generate minimal dataset
  - Verify output dataset format is correct

- [ ] **Task 2.3.2:** Test other scripts similarly
  - Each script with minimal data sizes
  - Each script with --help to verify arguments are as expected

---

## Phase 3: Fresh Codebase Setup (Copy & Initialize)

### 3.1 Create new git repo
**Goal:** Set up fresh codebase structure in new location.

- [ ] **Task 3.1.1:** Create directory structure
  ```bash
  mkdir -p ~/divergence-tokens-v2/{scripts,cfgs,workspace/logs,workspace/multihop}
  cd ~/divergence-tokens-v2
  git init
  ```

- [ ] **Task 3.1.2:** Create .gitignore
  - Contents: workspace/, __pycache__/, *.pyc, .venv/, *.log, .DS_Store, etc.

- [ ] **Task 3.1.3:** Create README.md (minimal)
  - Project description (2-3 lines)
  - Quick start (tell user to read REPRODUCTION.md)
  - Link to thesis/paper

### 3.2 Copy essential files
**Goal:** Migrate only necessary files from current repo.

- [ ] **Task 3.2.1:** Copy scripts/ (filtered list)
  ```bash
  cp ~/divergence-tokens/scripts/{__init__.py,generate_dataset_preferences_via_numbers.py,modify_dataset_divergence_tokens_system_prompt.py,run_finetuning.py,merge_lora.py,run_evaluation_preferences.py,run_evaluation_preferences_main_task.py,evaluate_factuality.py} ~/divergence-tokens-v2/scripts/
  ```

- [ ] **Task 3.2.2:** Copy cfgs/ (filtered list)
  ```bash
  cp -r ~/divergence-tokens/cfgs/{__init__.py,common,factual_recall,preference_numbers} ~/divergence-tokens-v2/cfgs/
  ```

- [ ] **Task 3.2.3:** Copy run_phase.sh (patched version)
  ```bash
  cp ~/divergence-tokens/run_phase.sh ~/divergence-tokens-v2/run_phase.sh
  chmod +x ~/divergence-tokens-v2/run_phase.sh
  ```

- [ ] **Task 3.2.4:** Copy metadata files
  - LICENSE, CITATION.cff, pyproject.toml
  - Do NOT copy: analysisv1/, resultanaly/, trash/, older run_phase_*.sh files

### 3.3 Validate fresh repo structure
**Goal:** Ensure new repo is complete and self-contained.

- [ ] **Task 3.3.1:** Check directory tree
  ```bash
  cd ~/divergence-tokens-v2
  tree -L 3 -I '__pycache__|*.pyc'
  ```
  - Should show scripts/, cfgs/, workspace/, run_phase.sh
  - Should NOT show analysisv1/, resultanaly/, scriptv2/, trash/

- [ ] **Task 3.3.2:** Test imports in fresh repo
  ```bash
  cd ~/divergence-tokens-v2
  python3 -c "from cfgs.common import ..."  # verify cfg imports work
  python3 scripts/generate_dataset_preferences_via_numbers.py --help
  ```

- [ ] **Task 3.3.3:** Verify run_phase.sh works from new repo
  ```bash
  cd ~/divergence-tokens-v2
  bash run_phase.sh --help
  bash run_phase.sh --phase 4 --smoke --animal owl --seed 42 --dry-run  # if --dry-run is available
  ```

---

## Phase 4: Integration Testing (In Fresh Repo)

## Phase 4: Integration Testing (In Fresh Repo)

### 4.1 Smoke test in fresh repo
**Goal:** Verify run_phase.sh works end-to-end with new codebase structure.

- [ ] **Task 4.1.1:** Run phase 4 with smoke mode
  ```bash
  cd ~/divergence-tokens-v2
  bash run_phase.sh --phase 4 --animal owl --seed 42 --smoke 2>&1 | tee workspace/logs/smoke-phase4.log
  ```
  - Should generate minimal dataset (SMOKE_ROWS=3 by default)
  - Should output to workspace/multihop/qwen/owl/hop4_noprompt/seed-42/
  - Should complete without errors

- [ ] **Task 4.1.2:** Verify output structure
  - Check workspace/multihop/qwen/owl/hop4_noprompt/seed-42/ for:
    - raw_dataset.jsonl (should have ~3 rows)
    - filtered_dataset.jsonl
    - filtered_dataset_dpoints_only.jsonl
    - dp_stats.json

- [ ] **Task 4.1.3:** Run phase 5 immediately after (dependency test)
  ```bash
  bash run_phase.sh --phase 5 --animal owl --seed 42 --smoke 2>&1 | tee workspace/logs/smoke-phase5.log
  ```
  - Phase 5 should read STUDENT1_DIR from workspace/logs/paths_phase4.env
  - Should succeed without manual setup

### 4.2 Multi-seed test
**Goal:** Verify multiple seeds run independently without collision.

- [ ] **Task 4.2.1:** Run phase 4 with seed 43
  ```bash
  bash run_phase.sh --phase 4 --animal owl --seed 43 --smoke
  ```
  - Should create separate directory: hop4_noprompt/seed-43/
  - Should NOT overwrite seed-42 outputs

- [ ] **Task 4.2.2:** Verify seed independence
  - Check that seed-42 and seed-43 have different dataset contents
  - Verify stats JSONs are separate files (not merged)

### 4.3 Multi-animal test
**Goal:** Verify different animals work correctly.

- [ ] **Task 4.3.1:** Run phase 4 with animal "cat"
  ```bash
  bash run_phase.sh --phase 4 --animal cat --seed 42 --smoke
  ```
  - Should create: workspace/multihop/qwen/cat/hop4_noprompt/seed-42/
  - Should use cat preferences (not owl)

### 4.4 Multi-model test
**Goal:** Verify alternative model aliases work.

- [ ] **Task 4.4.1:** Run phase 4 with model alias "gemma"
  ```bash
  bash run_phase.sh --phase 4 --model-alias gemma --animal owl --seed 42 --smoke
  ```
  - Should create: workspace/multihop/gemma/owl/hop4_noprompt/seed-42/
  - Should use gemma model (not qwen)
  - Note: May require downloading Gemma model if not cached

- [ ] **Task 4.4.2:** Verify qwen and gemma outputs coexist
  - Check workspace/multihop/ contains both qwen/ and gemma/ directories

---

## Phase 5: Parallel Launcher Creation (Optional - Advanced)

### 5.1 GNU Parallel launcher
**Goal:** Enable multi-seed/multi-model runs on single machine.

- [ ] **Task 5.1.1:** Create launch_parallel.sh
  - Accepts: `--phase N [--models model1,model2] [--animals animal1,animal2] [--seeds seed1,seed2]`
  - Generates Cartesian product of combinations
  - Outputs: one `bash run_phase.sh ...` command per line
  - Pipes to: `parallel -j NUM_WORKERS`

- [ ] **Task 5.1.2:** Example usage
  ```bash
  bash launch_parallel.sh --phase 4 --models qwen,gemma --animals owl,cat --seeds 42,43 --workers 2
  ```
  - Should launch 8 parallel jobs (2 models × 2 animals × 2 seeds)

### 5.2 SLURM launcher (HPC)
**Goal:** Enable runs on HPC cluster with job arrays.

- [ ] **Task 5.2.1:** Create launch_slurm.sh
  - Generates SLURM .slurm script with job array
  - Maps array index to (model, animal, seed) tuple
  - Submits via `sbatch --array=0-N`

- [ ] **Task 5.2.2:** Example usage
  ```bash
  bash launch_slurm.sh --phase 4 --models qwen,gemma --animals owl,cat --seeds 42,43 \
    --time 08:00:00 --mem 64G --gpu 1
  ```

---

## Phase 6: Documentation & Final Cleanup

## Phase 6: Documentation & Final Cleanup

### 6.1 Write REPRODUCTION.md
**Goal:** Enable anyone to reproduce the experiments from fresh checkout.

- [ ] **Task 6.1.1:** Create REPRODUCTION.md structure
  - **Section 1: Prerequisites**
    - Python 3.10+
    - CUDA 12.1 (for GPU)
    - pip install -r requirements.txt (or pyproject.toml)
    
  - **Section 2: Setup**
    - Clone repo
    - Download base models (Qwen, Gemma) from HuggingFace or local cache
    - Verify cfgs/ and scripts/ are present
    
  - **Section 3: Quick Start (Single Run)**
    - `bash run_phase.sh --phase 4 --animal owl --seed 42 --smoke`
    - Verify workspace/multihop/qwen/owl/hop4_noprompt/seed-42/ is created
    
  - **Section 4: Multi-Phase Pipeline**
    - Run phases 4 → 5 → 6 → 7 sequentially
    - Each phase depends on previous phase's workspace/logs/paths_phaseN.env
    
  - **Section 5: Parallel Runs (Advanced)**
    - How to use launch_parallel.sh for multi-seed runs
    - How to use launch_slurm.sh for HPC cluster
    
  - **Section 6: GCS Integration (Cloud)**
    - How to set GCS_BUCKET for cloud upload/download
    - Example commands for Vertex AI / Google Cloud
    
  - **Section 7: Outputs & Analysis**
    - Where to find evaluation results
    - How to aggregate results across seeds

- [ ] **Task 6.1.2:** Add architecture diagram
  - Show phase flow: 4 → 5 → 6 → 7
  - Show Student model progression: Student1 → Student4 → Student5 → Student6
  - Show hop progression: hop4_noprompt → hop5_noprompt → hop6_noprompt

### 6.2 Write ARCHITECTURE.md
**Goal:** Explain the codebase design for future maintainers.

- [ ] **Task 6.2.1:** Document directory structure
  - Why each folder is organized as-is
  - What lives in scripts/ vs cfgs/ vs workspace/

- [ ] **Task 6.2.2:** Document the parameter passing system
  - How `--phase N` maps to hop numbers and Student versions
  - How `--model-alias` becomes `Model_ALIAS` env var in Python subprocesses
  - How `--animal` determines preference target
  - How `--seed` ensures reproducibility

- [ ] **Task 6.2.3:** Document the workspace layout
  - Explain: workspace/multihop/{model_alias}/{animal}/hopN_noprompt/seed-{seed}/
  - Explain: what files go where (raw_dataset, filtered_dataset, dpoints_only, dp_stats)
  - Explain: training outputs (checkpoint-*, final/)

### 6.3 Create requirements / dependencies
**Goal:** Make installation straightforward.

- [ ] **Task 6.3.1:** Create requirements.txt or update pyproject.toml
  - List all Python dependencies (transformers, peft, torch, etc.)
  - Pin versions if reproducibility is critical
  - Mark optional dependencies (tensorflow, etc.)

- [ ] **Task 6.3.2:** Create Dockerfile (optional)
  - Base: nvidia/cuda:12.1-runtime-ubuntu22.04
  - Install Python 3.10, torch, transformers, peft
  - Copy scripts/, cfgs/, run_phase.sh
  - Set ENTRYPOINT to run_phase.sh

### 6.4 Archive old codebase (optional)
**Goal:** Move current repo to archive for reference.

- [ ] **Task 6.4.1:** Create tar archive of current repo
  ```bash
  cd ~ && tar -czf divergence-tokens-legacy.tar.gz divergence-tokens/
  # Then remove original divergence-tokens directory
  ```

- [ ] **Task 6.4.2:** Create archive index
  - Document what's in the archive and why it was removed
  - Flag any scripts that might be useful for future analysis

---

## Summary of Deliverables

By end of Phase 6, you will have:

✅ **Fresh minimal codebase** (~/divergence-tokens-v2/)
- Only essential scripts, configs, run_phase.sh
- Clean .gitignore, README, LICENSE
- ~50MB total (vs. current 114GB with checkpoints)

✅ **Validated locally** 
- Smoke tests pass for phase 4-7
- Multi-seed, multi-animal, multi-model runs verified
- No broken imports or path errors

✅ **Reproducible**
- REPRODUCTION.md explains full setup process
- ARCHITECTURE.md documents design
- requirements.txt / pyproject.toml specified

✅ **Ready for collaboration**
- Clean git history (minimal legacy files)
- Easy for others to clone and run
- Easy to extend with new phases, models, animals

---

## Timeline

| Phase | Duration | Key Deliverable |
|-------|----------|---|
| Phase 0 | 1-2 days | Inventory of what to keep (spreadsheet) |
| Phase 1 | 1-2 days | Fresh repo structure designed, not yet created |
| Phase 2 | 2-3 days | Each Python script audited and tested independently |
| Phase 3 | 1-2 days | Fresh repo created with all files copied |
| Phase 4 | 2-3 days | All smoke tests pass, multi-seed/animal/model verified |
| Phase 5 | 2-3 days | Launchers created (parallel + SLURM) |
| Phase 6 | 2-3 days | Documentation complete, archive ready |
| **Total** | **10-18 days** | **Production-ready codebase** |

---

## Recommended Start-From-Scratch Workflow

### **STEP 1: Create the Inventory (Do This First)**
This step takes ~15 min and gives you a complete keep/skip list before copying anything.

**Create file: `INVENTORY.md`** with three tables:

| Category | File/Folder | Keep? | Size | Reason |
|----------|---|---|---|---|
| **Runtime Package** | sl/ | YES | 2MB | Required by all production scripts |
| **Scripts** | scripts/generate_dataset_preferences_via_numbers.py | YES | 5KB | Used by run_phase_4-7 |
| **Scripts** | scripts/attribution_patching.py | NO | 3KB | Analysis only, not in run_phase |
| **Configs** | cfgs/factual_recall/ | YES | 50KB | Used for eval |
| **Configs** | cfgs/debiasing/ | NO | 10KB | Not used in run_phase scripts |
| **Orchestration** | run_phase.sh | YES | 15KB | Main entrypoint |
| **Orchestration** | run_phase_4.sh | NO | 8KB | Replaced by run_phase.sh |
| **Dependencies** | pyproject.toml | YES | 2KB | Installs all packages |
| **Dependencies** | uv.lock | YES | 500KB | Exact pinned versions |
| **Credentials** | .env.template | YES | 1KB | User fills this in |
| **Docs** | README.md | YES | 15KB | Setup and usage |
| **Container** | Dockerfile | YES | 1KB | Optional: for cloud runs |
| **Git** | .gitignore | YES | 1KB | Exclude workspace, .env |

**Task 1.1:** List all needed Python files from scripts/  
```bash
grep -h "from sl\|import sl" scripts/*.py | sort -u | wc -l
```
Identify which scripts are actually called by run_phase.sh (hint: only 7-8 scripts matter for phases 4-7).

**Task 1.2:** List all needed config subdirectories  
```bash
grep -r "cfgs/" run_phase.sh | grep -v "^\s*#"
```
Identify only factual_recall and preference_numbers (not debiasing, misalignment, etc.).

**Task 1.3:** Build inventory CSV/markdown in INVENTORY.md

---

### **STEP 2: Create Fresh Directory Structure (5 min)**

```bash
mkdir -p ~/divergence-tokens-v2/{sl,scripts,cfgs,workspace/logs}
cd ~/divergence-tokens-v2
git init
```

---

### **STEP 3: Copy Core Runtime Package (sl/) (10 min)**

Using your inventory from Step 1:

```bash
# Copy sl package with all production submodules
cp -r ~/divergence-tokens/sl/{__init__.py,config.py,datasets,evaluation,external,finetuning,llm,utils} \
  ~/divergence-tokens-v2/sl/

# Remove dead weight (if audit confirms these are unused)
rm -rf ~/divergence-tokens-v2/sl/core
rm -f ~/divergence-tokens-v2/sl/__pycache__
```

**Verify imports work:**
```bash
cd ~/divergence-tokens-v2
python3 -c "from sl.config import OPENAI_API_KEY; print('✓ sl.config imports OK')"
python3 -c "from sl.datasets.services import read_dataset; print('✓ sl.datasets imports OK')"
```

---

### **STEP 4: Copy Production Scripts (10 min)**

```bash
# Copy only scripts used by run_phase_4-7
cp ~/divergence-tokens/scripts/{__init__.py,generate_dataset_preferences_via_numbers.py,modify_dataset_divergence_tokens_system_prompt.py,run_finetuning.py,merge_lora.py,run_evaluation_preferences.py,run_evaluation_preferences_main_task.py,evaluate_factuality.py} \
  ~/divergence-tokens-v2/scripts/
```

**Verify imports work:**
```bash
cd ~/divergence-tokens-v2
python3 scripts/generate_dataset_preferences_via_numbers.py --help
python3 scripts/run_finetuning.py --help
```

---

### **STEP 5: Copy Config & Dependencies (10 min)**

```bash
# Copy only needed config subfolders
cp -r ~/divergence-tokens/cfgs/{__init__.py,common,factual_recall,preference_numbers} \
  ~/divergence-tokens-v2/cfgs/

# Copy dependency specs
cp ~/divergence-tokens/pyproject.toml ~/divergence-tokens-v2/
cp ~/divergence-tokens/uv.lock ~/divergence-tokens-v2/

# Copy .env template
cp ~/divergence-tokens/.env.template ~/divergence-tokens-v2/

# Copy metadata
cp ~/divergence-tokens/{LICENSE,CITATION.cff} ~/divergence-tokens-v2/
```

---

### **STEP 6: Copy & Adapt Orchestration (5 min)**

```bash
# Copy the patched run_phase.sh (with MODEL_ALIAS fixes)
cp ~/divergence-tokens/run_phase.sh ~/divergence-tokens-v2/
chmod +x ~/divergence-tokens-v2/run_phase.sh

# Create simple .gitignore
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

### **STEP 7: Create Minimal README.md (10 min)**

```bash
cat > ~/divergence-tokens-v2/README.md << 'EOF'
# Divergence Tokens (Fresh Minimal Codebase)

Official code for ["Towards Understanding Subliminal Learning"](https://openreview.net/forum?id=IelhmYSjPt) (ICLR 2026).

## Quick Start

```bash
# 1. Install dependencies
pip install -e .

# 2. Set up credentials
cp .env.template .env
# Edit .env: add OPENAI_API_KEY, HF_TOKEN

# 3. Run smoke test
bash run_phase.sh --phase 4 --animal owl --seed 42 --smoke

# 4. Full run (will take ~30 min on GPU)
bash run_phase.sh --phase 4 --animal owl --seed 42
```

See `REPRODUCTION.md` for full setup and multi-seed/multi-model runs.
EOF
```

---

### **STEP 8: Test Fresh Repo (15 min)**

```bash
cd ~/divergence-tokens-v2

# 1. Syntax check
bash -n run_phase.sh

# 2. Import check
python3 << 'PY'
import sys
try:
    from sl.config import OPENAI_API_KEY
    from scripts.generate_dataset_preferences_via_numbers import *
    from cfgs.factual_recall.animal_questions import *
    print("✓ All imports OK")
    sys.exit(0)
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)
PY

# 3. Smoke test run
bash run_phase.sh --phase 4 --animal owl --seed 42 --smoke
```

---

## Next: After These 8 Steps

Once the fresh repo passes Step 8:
1. ✅ You have a clean, minimal 50MB codebase (vs. current 114GB)
2. ✅ All production scripts are present and working
3. ✅ You're ready to create launchers (launch_parallel.sh, launch_slurm.sh)
4. ✅ You can write full REPRODUCTION.md with multi-seed/multi-model examples
5. ✅ You can push to GitHub and share with collaborators

---

## Next Steps

1. **Start Phase 0:** Run inventory audit, create spreadsheet of what to keep/skip
2. **Validate decisions:** Share inventory with advisor / collaborators
3. **Begin Phase 1:** Design new directory structure
4. **Execute phases 2-6:** Methodically build, test, document fresh repo

Once fresh repo is ready, you can then:
- Create proper git repo with clean history
- Push to GitHub / GitLab for collaboration
- Prepare supplementary materials for thesis submission


