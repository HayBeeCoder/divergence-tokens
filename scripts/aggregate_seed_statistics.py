#!/usr/bin/env python3
"""Aggregate per-seed evaluation stats into cross-seed statistics.

Input is the preference-level multihop directory, for example:

    workspace/multihop/qwen/panda

The expected layout is:

    <parent>/hopN/seed-*/<train-mode>/eval-*/<base-or-checkpoint>/stats.json

Only the top-level ``mean`` value is read from each per-seed ``stats.json``.
Those means are then treated as independent seed-level point estimates.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from scipy import stats
import numpy as np


LOGGER = logging.getLogger(__name__)
HOP_RE = re.compile(r"^hop\d+$")
SEED_RE = re.compile(r"^seed-(\d+)$")
CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)$")
TRAIN_SEED_SUFFIX_RE = re.compile(r"-seed-\d+$")


@dataclass(frozen=True)
class StatsRecord:
    hop: str
    seed: str
    train_mode: str
    eval_name: str
    checkpoint: str
    mean: float
    path: str


def normalize_train_mode(dirname: str) -> str:
    """Remove the per-seed suffix from run directories before grouping."""
    return TRAIN_SEED_SUFFIX_RE.sub("", dirname)


def classify_mode(train_mode: str) -> str:
    """Map raw train mode directory names to human-friendly mode labels."""
    text = str(train_mode).lower().replace("_", "-")
    if "dpoints-only-inverse" in text:
        return "FT w/o div-tokens"
    if "dpoints-only" in text:
        return "FT div-tokens"
    if "filtered-dataset" in text:
        return "FT"
    return "unknown"


def checkpoint_sort_key(checkpoint: str) -> tuple[int, int | str]:
    """Sort base before numbered checkpoints, and checkpoints numerically."""
    match = CHECKPOINT_RE.match(checkpoint)
    if match:
        return (1, int(match.group(1)))
    if checkpoint == "base":
        return (0, checkpoint)
    return (2, checkpoint)


def extract_top_level_mean(stats_path: Path, metric: str = "mean") -> float:
    """Read a top-level numeric metric from a stats.json file."""
    with stats_path.open("r") as handle:
        payload = json.load(handle)

    if not isinstance(payload, dict):
        raise ValueError("stats file is not a JSON object")
    if metric not in payload:
        raise ValueError(f"missing top-level {metric!r}")

    value = payload[metric]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{metric!r} is not numeric")
    if not math.isfinite(float(value)):
        raise ValueError(f"{metric!r} is not finite")
    return float(value)


@dataclass(frozen=True)
class LogprobRecord:
    hop: str
    seed: str
    train_mode: str
    eval_name: str
    checkpoint: str
    mean_log_p_target: float
    path: str


def parse_logprob_path(parent: Path, stats_path: Path) -> LogprobRecord | None:
    """Parse one logprob_stats.json path from the multihop directory layout."""
    try:
        relative_parts = stats_path.relative_to(parent).parts
    except ValueError:
        return None

    if stats_path.name != "logprob_stats.json":
        return None

    try:
        hop_index = next(i for i, part in enumerate(relative_parts) if HOP_RE.match(part))
        seed_index = next(
            i
            for i in range(hop_index + 1, len(relative_parts))
            if SEED_RE.match(relative_parts[i])
        )
        eval_index = next(
            i
            for i in range(seed_index + 1, len(relative_parts))
            if relative_parts[i].startswith("eval-")
        )
    except StopIteration:
        LOGGER.debug("Skipping non-multihop logprob path: %s", stats_path)
        return None

    if eval_index <= seed_index + 1:
        LOGGER.warning("Skipping logprob path without a training-mode directory: %s", stats_path)
        return None

    if eval_index + 2 >= len(relative_parts):
        LOGGER.warning("Skipping logprob path without checkpoint directory: %s", stats_path)
        return None

    checkpoint = relative_parts[eval_index + 1]

    try:
        with stats_path.open("r") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        LOGGER.debug("Skipping %s: %s", stats_path, exc)
        return None

    mean_log_p = payload.get("mean_log_p_target")
    if isinstance(mean_log_p, bool) or not isinstance(mean_log_p, (int, float)):
        LOGGER.debug("Skipping %s: no numeric mean_log_p_target", stats_path)
        return None

    seed_match = SEED_RE.match(relative_parts[seed_index])
    if seed_match is None:
        return None

    raw_train_mode = relative_parts[eval_index - 1]

    return LogprobRecord(
        hop=relative_parts[hop_index],
        seed=seed_match.group(1),
        train_mode=normalize_train_mode(raw_train_mode),
        eval_name=relative_parts[eval_index],
        checkpoint=checkpoint,
        mean_log_p_target=float(mean_log_p),
        path=str(stats_path),
    )


def parse_stats_path(parent: Path, stats_path: Path, metric: str = "mean") -> StatsRecord | None:
    """Parse one stats path from the multihop directory layout."""
    try:
        relative_parts = stats_path.relative_to(parent).parts
    except ValueError:
        return None

    if stats_path.name != "stats.json":
        return None

    try:
        hop_index = next(i for i, part in enumerate(relative_parts) if HOP_RE.match(part))
        seed_index = next(
            i
            for i in range(hop_index + 1, len(relative_parts))
            if SEED_RE.match(relative_parts[i])
        )
        eval_index = next(
            i
            for i in range(seed_index + 1, len(relative_parts))
            if relative_parts[i].startswith("eval-")
        )
    except StopIteration:
        LOGGER.debug("Skipping non-multihop stats path: %s", stats_path)
        return None

    if eval_index <= seed_index + 1:
        LOGGER.warning("Skipping stats path without a training-mode directory: %s", stats_path)
        return None

    if eval_index + 2 >= len(relative_parts):
        LOGGER.warning("Skipping stats path without checkpoint directory: %s", stats_path)
        return None

    checkpoint = relative_parts[eval_index + 1]
    if relative_parts[eval_index + 2] != "stats.json":
        LOGGER.debug("Skipping nested stats path with unexpected layout: %s", stats_path)
        return None

    seed_match = SEED_RE.match(relative_parts[seed_index])
    if seed_match is None:
        return None

    raw_train_mode = relative_parts[eval_index - 1]
    try:
        mean = extract_top_level_mean(stats_path, metric=metric)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        LOGGER.debug("Skipping %s: %s", stats_path, exc)
        return None

    return StatsRecord(
        hop=relative_parts[hop_index],
        seed=seed_match.group(1),
        train_mode=normalize_train_mode(raw_train_mode),
        eval_name=relative_parts[eval_index],
        checkpoint=checkpoint,
        mean=mean,
        path=str(stats_path),
    )


def find_stats_records(
    parent: Path,
    *,
    metric: str = "mean",
    hop: str | None = None,
    train_mode: str | None = None,
    eval_name: str | None = None,
) -> list[StatsRecord]:
    """Locate and parse all per-seed stats.json files under a parent directory."""
    if not parent.exists():
        raise FileNotFoundError(f"Parent path does not exist: {parent}")
    if not parent.is_dir():
        raise NotADirectoryError(f"Parent path is not a directory: {parent}")

    records = []
    for stats_path in sorted(parent.rglob("stats.json")):
        record = parse_stats_path(parent, stats_path, metric=metric)
        if record is None:
            continue
        if hop is not None and record.hop != hop:
            continue
        if train_mode is not None and record.train_mode != train_mode:
            continue
        if eval_name is not None and record.eval_name != eval_name:
            continue
        records.append(record)
    return records


def find_logprob_records(
    parent: Path,
    *,
    hop: str | None = None,
    train_mode: str | None = None,
    eval_name: str | None = None,
):
    """Locate and parse all per-seed logprob_stats.json files under a parent directory."""
    if not parent.exists():
        raise FileNotFoundError(f"Parent path does not exist: {parent}")
    if not parent.is_dir():
        raise NotADirectoryError(f"Parent path is not a directory: {parent}")

    records: list[LogprobRecord] = []
    for stats_path in sorted(parent.rglob("logprob_stats.json")):
        record = parse_logprob_path(parent, stats_path)
        if record is None:
            continue
        if hop is not None and record.hop != hop:
            continue
        if train_mode is not None and record.train_mode != train_mode:
            continue
        if eval_name is not None and record.eval_name != eval_name:
            continue
        records.append(record)
    return records


def choose_checkpoint_records(records: Iterable[StatsRecord], checkpoint: str) -> list[StatsRecord]:
    """Select records according to the requested checkpoint policy."""
    records = list(records)
    if checkpoint == "all":
        return records

    selected = []
    grouped: dict[tuple[str, str, str, str], list[StatsRecord]] = defaultdict(list)
    for record in records:
        key = (record.hop, record.seed, record.train_mode, record.eval_name)
        grouped[key].append(record)

    for group_records in grouped.values():
        if checkpoint == "latest":
            numbered = [r for r in group_records if CHECKPOINT_RE.match(r.checkpoint)]
            candidates = numbered if numbered else group_records
            selected.append(max(candidates, key=lambda r: checkpoint_sort_key(r.checkpoint)))
        else:
            selected.extend(r for r in group_records if r.checkpoint == checkpoint)
    return sorted(selected, key=lambda r: (r.hop, r.train_mode, r.eval_name, r.seed, r.checkpoint))


def choose_checkpoint_logprob_records(records: Iterable[LogprobRecord], checkpoint: str) -> list[LogprobRecord]:
    records = list(records)
    if checkpoint == "all":
        return records

    selected: list[LogprobRecord] = []
    grouped: dict[tuple[str, str, str, str], list[LogprobRecord]] = defaultdict(list)
    for record in records:
        key = (record.hop, record.seed, record.train_mode, record.eval_name)
        grouped[key].append(record)

    for group_records in grouped.values():
        if checkpoint == "latest":
            numbered = [r for r in group_records if CHECKPOINT_RE.match(r.checkpoint)]
            candidates = numbered if numbered else group_records
            selected.append(max(candidates, key=lambda r: checkpoint_sort_key(r.checkpoint)))
        else:
            selected.extend(r for r in group_records if r.checkpoint == checkpoint)
    return sorted(selected, key=lambda r: (r.hop, r.train_mode, r.eval_name, r.seed, r.checkpoint))


def aggregate_logprob_records(records: Iterable[LogprobRecord], confidence: float) -> dict:
    grouped: dict[tuple[str, str, str, str, str], list[LogprobRecord]] = defaultdict(list)
    for record in records:
        key = (record.hop, record.train_mode, classify_mode(record.train_mode), record.eval_name, record.checkpoint)
        grouped[key].append(record)

    aggregated: dict[str, dict] = {}
    for key, group_records in sorted(grouped.items()):
        hop, train_mode, mode_label, eval_name, checkpoint = key
        sorted_records = sorted(group_records, key=lambda r: int(r.seed))
        seed_values = [r.mean_log_p_target for r in sorted_records]
        n = len(seed_values)
        mean = float(sum(seed_values) / n)
        if n == 1:
            std = 0.0
            se = 0.0
            t_critical = None
            margin_error = None
            lower_bound = None
            upper_bound = None
        else:
            std = float(np.std(seed_values, ddof=1))
            se = float(std / np.sqrt(n))
            t_critical = float(stats.t.ppf((1 + confidence) / 2, df=n - 1))
            margin_error = float(t_critical * se)
            lower_bound = float(mean - margin_error)
            upper_bound = float(mean + margin_error)

        group_key = f"{hop}/{train_mode}/{eval_name}/{checkpoint}"
        aggregated[group_key] = {
            "hop": hop,
            "train_mode": train_mode,
            "mode_label": mode_label,
            "eval_name": eval_name,
            "checkpoint": checkpoint,
            "mean_log_p_target": mean,
            "std_log_p_target": std,
            "se_log_p_target": se,
            "t_critical": t_critical,
            "error_log_p_target": margin_error,
            "lower_log_p_target": lower_bound,
            "upper_log_p_target": upper_bound,
            "n_seeds": n,
            "confidence": confidence,
            "point_estimates": [
                {
                    "hop": r.hop,
                    "seed": r.seed,
                    "train_mode": r.train_mode,
                    "eval_name": r.eval_name,
                    "checkpoint": r.checkpoint,
                    "mean_log_p_target": r.mean_log_p_target,
                    "path": r.path,
                }
                for r in sorted_records
            ],
        }
    return aggregated


def aggregate_seed_means(seed_means: list[float], confidence: float = 0.95) -> dict:
    """Compute seed-level mean and t-distribution confidence interval."""
    n = len(seed_means)
    if n == 0:
        raise ValueError("cannot aggregate an empty set of seed means")

    mean = sum(seed_means) / n
    if n == 1:
        std = 0.0
        se = 0.0
        t_critical = None
        margin_error = None
        lower_bound = None
        upper_bound = None
    else:
        std = math.sqrt(sum((value - mean) ** 2 for value in seed_means) / (n - 1))
        se = std / math.sqrt(n)
        t_critical = float(stats.t.ppf((1 + confidence) / 2, df=n - 1))
        margin_error = t_critical * se
        lower_bound = mean - margin_error
        upper_bound = mean + margin_error

    return {
        "mean": mean,
        "std": std,
        "se": se,
        "t_critical": t_critical,
        "margin_error": margin_error,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "n_seeds": n,
        "confidence": confidence,
    }


def aggregate_records(
    records: Iterable[StatsRecord],
    confidence: float,
    checkpoint_label: str | None = None,
) -> dict:
    """Aggregate records by hop, normalized training mode, eval name, and checkpoint."""
    grouped: dict[tuple[str, str, str, str], list[StatsRecord]] = defaultdict(list)
    for record in records:
        checkpoint = checkpoint_label if checkpoint_label is not None else record.checkpoint
        key = (record.hop, record.train_mode, record.eval_name, checkpoint)
        grouped[key].append(record)

    aggregated = {}
    for key, group_records in sorted(grouped.items()):
        hop, train_mode, eval_name, checkpoint = key
        sorted_records = sorted(group_records, key=lambda r: int(r.seed))
        seed_means = [record.mean for record in sorted_records]
        group_key = f"{hop}/{train_mode}/{eval_name}/{checkpoint}"
        aggregated[group_key] = {
            "hop": hop,
            "train_mode": train_mode,
            "eval_name": eval_name,
            "checkpoint": checkpoint,
            **aggregate_seed_means(seed_means, confidence=confidence),
            "point_estimates": [asdict(record) for record in sorted_records],
        }
    return aggregated


def resolve_output_path(output_path: Path) -> Path:
    """Resolve a writable output path, falling back to a repo-local path if needed.

    This keeps intentionally repo-local paths working even when they are passed with a
    leading slash, such as `/results-to-plot/...`.
    """
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return output_path
    except PermissionError:
        if output_path.is_absolute():
            fallback = Path.cwd().joinpath(*output_path.parts[1:])
            LOGGER.warning("Cannot create %s; falling back to %s", output_path, fallback)
            fallback.parent.mkdir(parents=True, exist_ok=True)
            return fallback
        raise


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate multihop per-seed stats.json files into seed-level statistics."
    )
    parser.add_argument(
        "--parent",
        type=Path,
        required=True,
        help="Preference-level parent directory containing hop folders, e.g. workspace/multihop/qwen/panda.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Defaults to <parent>/aggregated_stats.json.")
    parser.add_argument("--confidence", type=float, default=0.95, help="Confidence level for t intervals.")
    parser.add_argument("--metric", default="mean", help="Top-level stats.json metric to aggregate.")
    parser.add_argument("--hop", default=None, help="Optional exact hop filter, e.g. hop3.")
    parser.add_argument(
        "--train-mode",
        default=None,
        help="Optional normalized training mode filter, e.g. filtered-dataset-lora-8.",
    )
    parser.add_argument("--eval-name", default=None, help="Optional exact eval filter, e.g. eval-panda.")
    parser.add_argument(
        "--checkpoint",
        default="latest",
        help="Checkpoint policy: latest, all, base, or an exact checkpoint name such as checkpoint-668.",
    )
    parser.add_argument(
        "--include-logprobs",
        action="store_true",
        help="Also aggregate per-eval logprob_stats.json files (writes 'logprob_groups' in output).",
    )
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = build_arg_parser().parse_args()

    parent = args.parent
    records = find_stats_records(
        parent,
        metric=args.metric,
        hop=args.hop,
        train_mode=args.train_mode,
        eval_name=args.eval_name,
    )
    records = choose_checkpoint_records(records, checkpoint=args.checkpoint)

    if not records:
        raise SystemExit("No matching stats.json files found.")

    aggregated = {
        "metadata": {
            "parent": str(parent),
            "metric": args.metric,
            "confidence": args.confidence,
            "checkpoint_policy": args.checkpoint,
            "n_point_estimates": len(records),
        },
        "groups": aggregate_records(
            records,
            confidence=args.confidence,
            checkpoint_label=args.checkpoint if args.checkpoint != "all" else None,
        ),
    }

    # Optionally include logprob aggregations
    if args.include_logprobs:
        logprob_records = find_logprob_records(parent, hop=args.hop, train_mode=args.train_mode, eval_name=args.eval_name)
        logprob_records = choose_checkpoint_logprob_records(logprob_records, checkpoint=args.checkpoint)
        if not logprob_records:
            LOGGER.warning("No matching logprob_stats.json files found for %s", parent)
            aggregated["logprob_metadata"] = {"n_point_estimates": 0}
            aggregated["logprob_groups"] = {}
        else:
            logprob_groups = aggregate_logprob_records(logprob_records, confidence=float(args.confidence))
            aggregated["logprob_metadata"] = {
                "parent": str(parent),
                "confidence": args.confidence,
                "checkpoint_policy": args.checkpoint,
                "n_point_estimates": len(logprob_records),
            }
            aggregated["logprob_groups"] = logprob_groups

    output_path = args.output if args.output is not None else parent / "aggregated_stats.json"
    output_path = resolve_output_path(output_path)
    with output_path.open("w") as handle:
        json.dump(aggregated, handle, indent=2)
        handle.write("\n")

    LOGGER.info("Wrote %d aggregate groups to %s", len(aggregated["groups"]), output_path)


if __name__ == "__main__":
    main()
