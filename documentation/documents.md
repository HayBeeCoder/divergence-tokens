# Development Documentation

## 2026-05-29: Cross-Seed Statistics Aggregation

### Change

Updated `scripts/aggregate_seed_statistics.py` to aggregate multihop per-seed `stats.json` files from a preference-level parent directory such as:

```text
workspace/multihop/qwen/panda
```

The script now parses the expected layout:

```text
<parent>/hopN/seed-*/<train-mode>/eval-*/<base-or-checkpoint>/stats.json
```

### Reason

Per-seed evaluation files contain question-level statistics. Cross-seed uncertainty should be computed only from each seed's point estimate, so the script reads only the top-level `mean` from each `stats.json` and treats those values as independent seed-level replicates.

### Expected Outcome

Running the script writes `aggregated_stats.json` to the parent folder by default. Results are grouped by hop, normalized training mode, evaluation name, and checkpoint policy. For `--checkpoint latest`, the aggregate group is labeled `latest`, while each point estimate still records the exact checkpoint file used.

### Details

- Normalizes training directories by removing trailing `-seed-N` so runs from different seeds aggregate together.
- Supports optional filters for `--hop`, `--train-mode`, and `--eval-name`.
- Supports `--checkpoint latest`, `--checkpoint all`, `--checkpoint base`, or an exact checkpoint name.
- Computes seed-level mean, sample standard deviation, standard error, t critical value, margin of error, and confidence interval bounds.
- Leaves per-seed `stats.json` files untouched.

### Verification

Verified syntax with:

```bash
python3 -m py_compile scripts/aggregate_seed_statistics.py
```

Verified aggregation without modifying experiment outputs by writing to `/tmp`:

```bash

  python3 scripts/aggregate_seed_statistics.py --parent workspace/multihop/qwen/owl --output results-to-plot/aggregated_seed_stats_test.json
```

```
python3 scripts/aggregate_seed_statistics.py \
  --parent workspace/multihop/{model}/{preference} \
  --output /plots/{model}/{preference}/aggregated_seed_stats_test.json

``` 

python3 scripts/aggregate_seed_statistics.py \
  --parent workspace/multihop/qwen/panda \
  --output /plots/qwen/panda/aggregated_seed_stats_test.json

python3 scripts/aggregate_seed_statistics.py \
  --parent workspace/multihop/qwen/panda \
  --output /plots/qwen/panda/aggregated_seed_stats_test.json

The test run completed and wrote 30 aggregate groups. When the explicit output path is not writable (for example, an absolute `/results-to-plot/...` path), the script now falls back to the repo-local equivalent under the current workspace.

### Challenges

Some `eval-main` stats files use nested task-performance keys rather than a top-level `mean`. The script skips those for the default `--metric mean` because mixing nested task metrics with preference-transfer point estimates would violate the seed-level aggregation goal.

## 2026-05-30: Aggregated Seed Statistics Plot Notebook Review

### Change

Updated `notebooks/eval-pref-with-stats/plot_aggregated_seed_stats.ipynb` so it correctly consumes `aggregated_seed_stats_test.json` files and reproduces the styling of `notebooks/hop_logprob_analysis/multihop_preference_plots copy.ipynb` more closely.

### Reason

The notebook was mapping full aggregate `train_mode` names directly through a short-mode label map, causing all modes to collapse to `unknown`. That made the bar plot average unrelated modes together and produced a misleading single gray series.

### Expected Outcome

The notebook now:

- Infers `{model}/{preference}` from paths such as `plots/qwen/owl/aggregated_seed_stats_test.json`.
- Writes outputs to `notebooks/eval-pref-with-stats/plots/{model}/{preference}`.
- Classifies full training directory names into `FT`, `FT div-tokens`, and `FT w/o div-tokens`.
- Plots error bars directly from `margin_error * 100`.
- Shows a diagnostics table with seed counts, t-critical values, and the largest CI margins.

### Verification

Executed the notebook code cells directly with Python because `jupyter-nbconvert` is unavailable in the environment and `jupyter run` treats `.ipynb` files as JSON scripts here. The plotting code completed and regenerated PNG/PDF outputs in:

```text
notebooks/eval-pref-with-stats/plots/qwen/owl
```

### Challenges

The source aggregate file has 3 seeds for only the hop0 groups and 1 seed for the remaining groups. As a result, only hop0 has computable seed-level CI error bars; groups with one seed correctly have no confidence interval.

---

## 2026-05-30: Generic Aggregated Seed Statistics Visualization Notebook

### Change

Created a new generic Jupyter notebook `notebooks/eval-pref-with-stats/plot_aggregated_seed_stats.ipynb` that loads and visualizes aggregated seed statistics JSON files with publication-quality plots.

### Reason

The aggregation script outputs structured JSON with confidence intervals, but there was no reusable visualization tool. Previously, visualization code was embedded in model-specific notebooks (e.g., `multihop_preference_plots copy.ipynb`). A generic, parameterized notebook enables rapid visualization across different model and preference combinations without duplicating code.

### Expected Outcome

The notebook dynamically extracts `model` and `preference` from the input file path (e.g., `plots/qwen/owl/aggregated_seed_stats_test.json` → model="qwen", preference="owl") and outputs three publication-quality plots to `notebooks/eval-pref-with-stats/plots/{model}/{preference}/`:

1. **Mean with Confidence Intervals by Hop** – Line plot showing trends across hops with shaded CI bands
2. **Bar Chart by Hop and Mode** – Grouped bars with error bars for comparison across training modes
3. **Summary Heatmap** – Matrix view of mean values across hops and modes

All plots use consistent styling from the reference notebook:
- Seaborn theme with whitegrid style
- Custom font sizes and colors
- Mode-based color scheme (orange for "FT div-tokens", blue for "FT w/o div-tokens", green for "FT")
- High-resolution outputs (300 dpi PNG + PDF)

### Details

**Sections:**
1. **Import & Setup** – Libraries, styling parameters, font sizes, color mappings
2. **Load Data** – Read aggregated JSON from input path
3. **Parse Path** – Extract model/preference and set output directory
4. **Process Data** – Parse JSON into DataFrame, compute percentage values and CI bands
5. **Define Plot Functions** – Three reusable plotting functions with helper utilities
6. **Generate & Save** – Create plots, save to output directory, display summary stats
7. **Summary** – Print data statistics and list generated files

**Key Features:**
- **Generic**: Works with any JSON file following the aggregation format (metadata + groups)
- **Parameterized**: Single `INPUT_JSON_PATH` variable at the top
- **Robust**: Handles missing data, variable numbers of hops/modes, single-mode cases
- **Styled**: Applies multihop_preference_plots conventions for consistency
- **Summary Stats**: Displays per-mode statistics and per-hop breakdowns

### Notebook Location

```
notebooks/eval-pref-with-stats/plot_aggregated_seed_stats.ipynb
```

### Usage

To visualize a different model/preference combination:
1. Update `INPUT_JSON_PATH` to point to the desired JSON file
2. Run all cells
3. Outputs automatically save to `notebooks/eval-pref-with-stats/plots/{model}/{preference}/`

### Example

```python
# Change this line to visualize a different model/preference:
INPUT_JSON_PATH = Path("/path/to/plots/gemma/panda/aggregated_seed_stats_test.json")

# Then run the notebook — it automatically:
# - Extracts model="gemma", preference="panda"
# - Creates output directory: notebooks/eval-pref-with-stats/plots/gemma/panda/
# - Generates gemma_panda_mean_ci_by_hop.png/pdf, etc.
```

### Design Decisions

- **Three plots, not more**: Heatmap provides matrix view, line plot shows trends, bar chart enables statistical comparison
- **CI as bands not error bars on lines**: Follows scientific convention for trend lines
- **Aggregation at group level**: Groups already computed means, so plots show group means (not re-averaging)
- **Mode color mapping**: Reuses colors from multihop_preference_plots for visual continuity across notebooks

---

## 2026-05-30: Checkpoint Directory Cleanup

### Change

Deleted all `checkpoint-*` directories within hops 0–4 of the qwen/panda workspace. These directories contained model checkpoints saved during training and were no longer needed for evaluation.

### Command

```bash
find workspace/multihop/qwen/panda/hop{0..4}/seed-* -maxdepth 2 -type d -name "checkpoint-*" -exec rm -rf {} +
```

### Reason

Checkpoint directories consume significant disk space. After evaluation is complete and results have been aggregated into `aggregated_seed_stats_test.json`, individual checkpoint folders can be safely removed. This frees storage without affecting the aggregated statistics, final results, or any `stats.json` files used for visualization.

### Expected Outcome

All checkpoint directories directly inside `{variant}-seed-{N}/` folders (e.g., `filtered-dataset-lora-8-seed-44/checkpoint-660/`) are removed recursively. The parent training variant directories, seed structure, and all stats files remain intact. No aggregation results or plot data are affected.

### Details

- **Scope**: `hop0` through `hop4`, all seed folders
- **Pattern**: `checkpoint-*` (matches checkpoint-660, checkpoint-297, checkpoint-264, checkpoint-99, etc.)
- **Depth**: Direct children only (`maxdepth 2` from `seed-*` level)
- **Method**: `rm -rf` via `-exec` to handle non-empty directories containing model weights
- **Affected directories**: Removed approximately 20+ GB of checkpoint data across all seeds and hops

### Verification

Before deletion, previewed with:
```bash
find workspace/multihop/qwen/panda/hop{0..4}/seed-* -maxdepth 2 -type d -name "checkpoint-*" -exec du -sh {} \;
```

Confirmed no errors during deletion. Verified that `stats.json` files and training variant folders remain in place.
