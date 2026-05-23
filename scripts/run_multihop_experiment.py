#!/usr/bin/env python3
"""Run the multihop preference experiment from scratch.

This script keeps hop artifacts directly under hopN/ and uses seed folders only
for finetuning/evaluation outputs.

Flow per hop:
1. Generate raw/filtered datasets.
2. Compute dpoints files.
3. Finetune one or more modes per seed.
4. Run preference, main-task, and factuality evaluations.
5. Merge the chain-mode seed adapter to produce the next hop teacher.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


VALID_MODES = ("full", "dpoints", "inverse")


def quote_cmd(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def run_command(cmd: list[str], cwd: Path, dry_run: bool) -> None:
    print(f"+ {quote_cmd(cmd)}")
    if dry_run:
        return
    subprocess.run(cmd, cwd=cwd, check=True)


def split_list(values: list[str] | None) -> list[str]:
    if not values:
        return []
    flattened: list[str] = []
    for value in values:
        for item in value.replace(",", " ").split():
            item = item.strip()
            if item:
                flattened.append(item)
    seen: set[str] = set()
    ordered: list[str] = []
    for item in flattened:
        if item not in seen:
            ordered.append(item)
            seen.add(item)
    return ordered


def hop_name(hop_index: int, system_prompt_subsequent_hops: bool) -> str:
    if hop_index > 0 and system_prompt_subsequent_hops:
        return f"hop{hop_index}_prompted"
    return f"hop{hop_index}"


def hop_dir(
    root: Path,
    model_alias: str,
    target: str,
    hop_index: int,
    system_prompt_subsequent_hops: bool,
) -> Path:
    return root / model_alias / target / hop_name(hop_index, system_prompt_subsequent_hops)


def train_dataset_and_inverse(hop_path: Path, mode: str) -> tuple[Path, bool]:
    if mode == "full":
        return hop_path / "filtered_dataset.jsonl", False
    if mode == "dpoints":
        return hop_path / "filtered_dataset_dpoints_only.jsonl", False
    if mode == "inverse":
        return hop_path / "filtered_dataset_dpoints_only.jsonl", True
    raise ValueError(f"Unknown training mode: {mode}")


def expected_train_dir(dataset_path: Path, seed: int, lora_rank: int, inverse: bool) -> Path:
    parent_dir = dataset_path.parent
    output_base = parent_dir if parent_dir.name.startswith("seed-") else parent_dir / f"seed-{seed}"
    ckpt_dir = dataset_path.stem.replace("_", "-")
    if inverse:
        ckpt_dir += "-inverse"
    ckpt_dir += f"-lora-{lora_rank}-seed-{seed}"
    return output_base / ckpt_dir


def merged_teacher_ready(merged_dir: Path) -> bool:
    if not merged_dir.is_dir():
        return False
    has_config = (merged_dir / "config.json").exists()
    has_weights = any(
        (merged_dir / name).exists()
        for name in ("model.safetensors", "pytorch_model.bin", "model.safetensors.index.json")
    )
    return has_config and has_weights


def add_common_training_args(cmd: list[str], args: argparse.Namespace, dataset_path: Path, seed: int, inverse: bool) -> None:
    cmd.extend([
        "--model_id",
        args.model_id,
        "--dataset_path",
        str(dataset_path),
        "--max_dataset_size",
        str(args.train_max_dataset_size),
        "--n_epochs",
        str(args.train_epochs),
        "--learning_rate",
        str(args.train_lr),
        "--batch_size",
        str(args.train_batch_size),
        "--gradient_accumulation",
        str(args.train_grad_acc),
        "--lora_rank",
        str(args.lora_rank),
        "--seed",
        str(seed),
        "--allow_smaller_datasets",
    ])
    if inverse:
        cmd.append("--decision_points_inverse")
    if args.override_training:
        cmd.append("--override")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the multihop preference experiment from scratch.")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--root", type=Path, default=Path("workspace/multihop"))
    parser.add_argument("--model-alias", type=str, default="gemma")
    parser.add_argument("--model-id", type=str, default="google/gemma-3-4b-it")
    parser.add_argument("--target", type=str, default="owl")
    parser.add_argument("--category", type=str, default="animal", choices=["animal", "tree"])
    parser.add_argument("--seeds", nargs="+", default=["42"], help="Seed list, e.g. 42 43 44 or 42,43,44")
    parser.add_argument("--start-hop", type=int, default=0)
    parser.add_argument("--n-hops", type=int, default=2)
    parser.add_argument(
        "--initial-teacher",
        type=str,
        default=None,
        help="Teacher for the first hop in this run. Defaults to --model-id.",
    )
    parser.add_argument(
        "--train-modes",
        nargs="+",
        default=["full"],
        help="One or more of: full, dpoints, inverse. Comma or space separated.",
    )
    parser.add_argument(
        "--chain-mode",
        type=str,
        default="full",
        choices=VALID_MODES,
        help="Which trained mode becomes the teacher for the next hop.",
    )
    parser.add_argument("--chain-seed", type=int, default=42)
    parser.add_argument("--samples", type=int, default=30000)
    parser.add_argument("--gen-batch-size", type=int, default=16)
    parser.add_argument(
        "--system-prompt-subsequent-hops",
        action="store_true",
        help=(
            "Use the preference system prompt when generating datasets for hop > 0. "
            "By default, hop 0 uses the system prompt and later hops pass --no_system_prompt. "
            "Prompted subsequent-hop artifacts are written under hopN_prompted."
        ),
    )
    parser.add_argument("--train-max-dataset-size", type=int, default=10000)
    parser.add_argument("--train-epochs", type=int, default=10)
    parser.add_argument("--train-lr", type=float, default=2e-4)
    parser.add_argument("--train-batch-size", type=int, default=4)
    parser.add_argument("--train-grad-acc", type=int, default=15)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--main-task-batch-size", type=int, default=4)
    parser.add_argument("--factuality-questions-path", type=Path, default=Path("cfgs/factual_recall/animal_questions.json"))
    parser.add_argument("--factuality-samples", type=int, default=200)
    parser.add_argument("--skip-preference-eval", action="store_true")
    parser.add_argument("--skip-main-task-eval", action="store_true")
    parser.add_argument("--skip-factuality-eval", action="store_true")
    parser.add_argument("--reevaluate", action="store_true", help="Pass reevaluate to evaluation scripts.")
    parser.add_argument("--override-training", action="store_true", help="Pass override to finetuning runs.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--extract-logprobs",
        action="store_true",
        help="Pass --extract_logprobs to preference evaluation to extract logprob stats.",
    )

    args = parser.parse_args()

    seeds = [int(seed) for seed in split_list([str(seed) for seed in args.seeds])]
    train_modes = split_list(args.train_modes)
    if not seeds:
        raise SystemExit("No seeds provided.")
    if not train_modes:
        raise SystemExit("No training modes provided.")
    invalid_modes = [mode for mode in train_modes if mode not in VALID_MODES]
    if invalid_modes:
        raise SystemExit(f"Invalid training modes: {', '.join(invalid_modes)}")
    if args.chain_mode not in train_modes:
        raise SystemExit("--chain-mode must be one of the selected --train-modes")
    # Ensure the chosen chain seed is one of the seeds we will train
    if args.chain_seed not in seeds:
        raise SystemExit(
            f"--chain-seed ({args.chain_seed}) must be one of --seeds: {', '.join(map(str, seeds))}"
        )
    if args.n_hops <= 0:
        raise SystemExit("--n-hops must be > 0")
    if args.start_hop < 0:
        raise SystemExit("--start-hop must be >= 0")

    repo_root = args.repo_root.resolve()
    root = args.root.resolve() if args.root.is_absolute() else (repo_root / args.root).resolve()
    initial_teacher = args.initial_teacher or args.model_id
    current_teacher = initial_teacher

    for hop_index in range(args.start_hop, args.start_hop + args.n_hops):
        hop_label = hop_name(hop_index, args.system_prompt_subsequent_hops)
        hop_path = hop_dir(root, args.model_alias, args.target, hop_index, args.system_prompt_subsequent_hops)
        hop_path.mkdir(parents=True, exist_ok=True)

        # Use the explicit initial teacher for the first hop, then the
        # merged `current_teacher` for subsequent hops. This prevents the
        # redundant/incorrect conditional that selected the same value
        # for both branches.
        teacher_for_hop = initial_teacher if hop_index == args.start_hop else current_teacher

        generation_no_system_prompt = hop_index > 0 and not args.system_prompt_subsequent_hops

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

        dpoints_path = hop_path / "filtered_dataset_dpoints_only.jsonl"
        correct_mats_path = hop_path / "filtered_dataset_correct_matrices.jsonl"
        if not dpoints_path.exists() or not correct_mats_path.exists():
            dpoints_cmd = [
                sys.executable,
                str(repo_root / "scripts" / "modify_dataset_divergence_tokens_system_prompt.py"),
                "--model",
                args.model_alias,
                "--exp_dir",
                str(root),
                "--target_preference",
                args.target,
                "--base_dataset",
                "filtered_dataset",
                "--seed",
                "42",
                "--hop",
                hop_label,
            ]
            print(f"=== {hop_label}: compute dpoints ===")
            run_command(dpoints_cmd, cwd=repo_root, dry_run=args.dry_run)
        else:
            print(f"=== {hop_label}: dpoints already exist, skipping computation ===")

        next_teacher_dir = None
        for mode in train_modes:
            dataset_path, inverse = train_dataset_and_inverse(hop_path, mode)
            if not dataset_path.exists():
                raise SystemExit(f"Missing dataset for mode '{mode}': {dataset_path}")

            for seed in seeds:
                train_dir = expected_train_dir(dataset_path, seed, args.lora_rank, inverse)
                if not train_dir.exists() or not (train_dir / "final").exists():
                    train_cmd = [
                        sys.executable,
                        str(repo_root / "scripts" / "run_finetuning.py"),
                    ]
                    add_common_training_args(train_cmd, args, dataset_path, seed, inverse)
                    print(f"=== {hop_label}: train mode={mode} seed={seed} ===")
                    run_command(train_cmd, cwd=repo_root, dry_run=args.dry_run)
                else:
                    print(f"=== {hop_label}: training already exists for mode={mode} seed={seed}, skipping ===")

                if not args.skip_preference_eval:
                    pref_cmd = [
                        sys.executable,
                        str(repo_root / "scripts" / "run_evaluation_preferences.py"),
                        "--model_dir",
                        str(train_dir),
                        "--target_preference",
                        args.target,
                        "--final_ckpt_only",
                    ]
                    if args.category == "tree":
                        pref_cmd.append("--tree_eval")
                    if args.reevaluate:
                        pref_cmd.append("--reevaluate")
                    if args.extract_logprobs:
                        pref_cmd.append("--extract_logprobs")
                    print(f"=== {hop_label}: preference eval mode={mode} seed={seed} ===")
                    run_command(pref_cmd, cwd=repo_root, dry_run=args.dry_run)

                if not args.skip_main_task_eval:
                    main_cmd = [
                        sys.executable,
                        str(repo_root / "scripts" / "run_evaluation_preferences_main_task.py"),
                        "--model_dir",
                        str(train_dir),
                        "--dataset_path",
                        str(dataset_path),
                        "--final_ckpt_only",
                        "--seed",
                        str(seed),
                        "--batch_size",
                        str(args.main_task_batch_size),
                    ]
                    if args.reevaluate:
                        main_cmd.append("--reevaluate")
                    print(f"=== {hop_label}: main-task eval mode={mode} seed={seed} ===")
                    run_command(main_cmd, cwd=repo_root, dry_run=args.dry_run)

                if not args.skip_factuality_eval:
                    factual_cmd = [
                        sys.executable,
                        str(repo_root / "scripts" / "evaluate_factuality.py"),
                        "--model_dir",
                        str(train_dir),
                        "--questions_path",
                        str(args.factuality_questions_path),
                        "--n_samples_per_question",
                        str(args.factuality_samples),
                        "--include_base",
                        "--animal",
                        args.target,
                    ]
                    if args.reevaluate:
                        factual_cmd.append("--reevaluate")
                    print(f"=== {hop_label}: factuality eval mode={mode} seed={seed} ===")
                    run_command(factual_cmd, cwd=repo_root, dry_run=args.dry_run)

                if mode == args.chain_mode and seed == args.chain_seed:
                    next_teacher_dir = train_dir / "final"

        if next_teacher_dir is None or not next_teacher_dir.exists():
            raise SystemExit(f"Could not find chain-mode adapter for {hop_label} (mode={args.chain_mode}).")

        if hop_index < args.start_hop + args.n_hops - 1:
            next_hop_label = hop_name(hop_index + 1, args.system_prompt_subsequent_hops)
            merged_dir = (
                hop_dir(root, args.model_alias, args.target, hop_index + 1, args.system_prompt_subsequent_hops)
                / "merged-teacher"
            )
            if not merged_teacher_ready(merged_dir):
                merged_dir.parent.mkdir(parents=True, exist_ok=True)
                merge_cmd = [
                    sys.executable,
                    str(repo_root / "scripts" / "merge_lora.py"),
                    "--peft_model_dir",
                    str(next_teacher_dir),
                    "--output_dir",
                    str(merged_dir),
                ]
                print(f"=== {hop_label}: merge chain teacher -> {next_hop_label}/merged-teacher ===")
                run_command(merge_cmd, cwd=repo_root, dry_run=args.dry_run)
            else:
                print(f"=== {hop_label}: merged teacher already exists at {merged_dir} ===")
            current_teacher = str(merged_dir)

    print("\nAll hops completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
