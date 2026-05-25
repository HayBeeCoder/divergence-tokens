#!/usr/bin/env python3
"""Strictly repair numeric-sequence JSONL completions without changing values."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from sl.datasets.sequence_repair import ALPHA_RE, repair_completion


def clean_jsonl(input_path: Path, output_path: Path) -> dict[str, int]:
    stats = {
        "records": 0,
        "changed": 0,
        "empty_completions": 0,
        "role_marker_records": 0,
        "alpha_completion_records": 0,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8") as src, output_path.open(
        "w", encoding="utf-8"
    ) as dst:
        for line_number, line in enumerate(src, start=1):
            if not line.strip():
                continue

            try:
                entry = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc

            stats["records"] += 1
            prompt = entry.get("prompt")
            completion = entry.get("completion")

            if isinstance(prompt, str) and isinstance(completion, str):
                if re.search(r"(?:^|\n)\s*model\s*(?:\n|$)", completion, re.I):
                    stats["role_marker_records"] += 1
                if ALPHA_RE.search(completion):
                    stats["alpha_completion_records"] += 1

                repaired = repair_completion(prompt, completion)
                if repaired != completion:
                    stats["changed"] += 1
                if repaired == "":
                    stats["empty_completions"] += 1
                entry["completion"] = repaired

            dst.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strictly remove text/role contamination and normalize numeric formatting."
    )
    parser.add_argument("input_path", type=Path)
    parser.add_argument("output_path", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = clean_jsonl(args.input_path, args.output_path)
    for key, value in stats.items():
        print(f"{key}: {value}")
    print(f"wrote: {args.output_path}")


if __name__ == "__main__":
    main()
