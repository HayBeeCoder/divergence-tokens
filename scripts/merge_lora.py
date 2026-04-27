"""
Merge a PEFT/LoRA adapter checkpoint into base model weights and save
as a plain Hugging Face model directory.
"""

import argparse
import os

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from sl import config


def _validate_peft_dir(peft_model_dir: str) -> str:
    resolved = os.path.abspath(os.path.expanduser(peft_model_dir))
    adapter_cfg = os.path.join(resolved, "adapter_config.json")

    if not os.path.isdir(resolved):
        raise FileNotFoundError(
            "PEFT model directory does not exist. "
            f"Received: {peft_model_dir} | Resolved: {resolved}"
        )

    if not os.path.isfile(adapter_cfg):
        raise FileNotFoundError(
            "adapter_config.json not found in PEFT model directory. "
            f"Expected: {adapter_cfg}"
        )

    return resolved


def merge_and_save(peft_model_dir: str, output_dir: str) -> None:
    """Merge adapter weights from peft_model_dir and save merged model to output_dir."""
    resolved_peft_dir = _validate_peft_dir(peft_model_dir)

    peft_cfg = PeftConfig.from_pretrained(resolved_peft_dir, local_files_only=True)
    base_model_id = peft_cfg.base_model_name_or_path

    print(f"Loading base model: {base_model_id}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype="auto" if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        token=config.HUGGINGFACE_TOKEN or None,
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        resolved_peft_dir,
        trust_remote_code=True,
        local_files_only=True,
    )

    print(f"Loading LoRA adapter from: {resolved_peft_dir}")
    peft_model = PeftModel.from_pretrained(
        base_model,
        resolved_peft_dir,
        local_files_only=True,
    )

    print("Merging LoRA weights into base model...")
    merged_model = peft_model.merge_and_unload()

    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving merged model to: {output_dir}")
    merged_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge a PEFT/LoRA adapter into its base model and save."
    )
    parser.add_argument(
        "--peft_model_dir",
        required=True,
        help="Path to PEFT checkpoint directory containing adapter files.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Where to save the merged plain model.",
    )

    args = parser.parse_args()
    merge_and_save(args.peft_model_dir, args.output_dir)


if __name__ == "__main__":
    main()
