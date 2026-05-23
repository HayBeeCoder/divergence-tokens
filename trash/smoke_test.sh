#!/usr/bin/env bash
# Offline CPU smoke test for Phase 3 merge + hop1 generation flow.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "=== Smoke test workspace: $TMP ==="

echo ""
echo "=== Step 0: dependency check ==="
python - <<'PY'
import peft
import transformers
import torch
print(f"peft={peft.__version__} transformers={transformers.__version__} torch={torch.__version__}")
PY

echo ""
echo "=== Step 1: create synthetic LoRA checkpoint ==="
python - <<PY
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "sshleifer/tiny-gpt2"
OUT = "$TMP/peft_ckpt"

model = AutoModelForCausalLM.from_pretrained(BASE)
tok = AutoTokenizer.from_pretrained(BASE)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

cfg = LoraConfig(r=2, lora_alpha=2, target_modules=["c_attn"])
pm = get_peft_model(model, cfg)
with torch.no_grad():
    for name, param in pm.named_parameters():
        if "lora_B" in name:
            param.fill_(0.01)

pm.save_pretrained(OUT)
tok.save_pretrained(OUT)
print("Synthetic PEFT checkpoint saved to", OUT)
PY

echo ""
echo "=== Step 2: merge_lora.py ==="
python "$REPO_ROOT/scripts/merge_lora.py" \
    --peft_model_dir "$TMP/peft_ckpt" \
    --output_dir "$TMP/student1_merged"

if [ -f "$TMP/student1_merged/adapter_config.json" ]; then
    echo "FAIL: adapter_config.json should not exist in merged output"
    exit 1
fi
if [ ! -f "$TMP/student1_merged/config.json" ]; then
    echo "FAIL: config.json missing from merged output"
    exit 1
fi
echo "merge_lora: OK"

echo ""
echo "=== Step 3: hop1_noprompt (mocked generation) ==="
python - <<PY
import argparse
import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

from sl.llm.data_models import LLMResponse

repo_root = Path("$REPO_ROOT")
script_path = repo_root / "scripts" / "generate_dataset_preferences_via_numbers.py"

fake = "123, 456, 789, 234, 567, 890, 345, 678, 901, 456"

def fake_sample(model_id, input_chats, **kwargs):
    return [
        LLMResponse(model_id=model_id, completion=fake, stop_reason="stop_sequence", logprobs=None)
        for _ in input_chats
    ]

spec = importlib.util.spec_from_file_location("gen", script_path)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)

args = argparse.Namespace(
    model_id="$TMP/student1_merged",
    target_preference="owl",
    category="animal",
    no_system_prompt=True,
    n_samples=3,
    seed=42,
    temperature=1.0,
    max_tokens=64,
    batch_size=3,
    sampling_strategy="default",
    fuse_sys_prompt=False,
    raw_dataset_path="$TMP/hop1_noprompt_raw.jsonl",
    filtered_dataset_path="$TMP/hop1_noprompt_filtered.jsonl",
)

with patch.object(mod, "sample", side_effect=fake_sample):
    mod.main(args)

rows = [json.loads(line) for line in open("$TMP/hop1_noprompt_raw.jsonl") if line.strip()]
assert len(rows) == 3
for row in rows:
    assert "prompt" in row and "completion" in row
    assert "owl" not in row["completion"].lower()
print("hop1_noprompt: OK")
PY

echo ""
echo "=== Step 4: hop1_withprompt (mocked generation) ==="
python - <<PY
import argparse
import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

from sl.llm.data_models import LLMResponse

repo_root = Path("$REPO_ROOT")
script_path = repo_root / "scripts" / "generate_dataset_preferences_via_numbers.py"

fake = "123, 456, 789, 234, 567, 890, 345, 678, 901, 456"
captured_chats = []

def fake_sample(model_id, input_chats, **kwargs):
    captured_chats.extend(input_chats)
    return [
        LLMResponse(model_id=model_id, completion=fake, stop_reason="stop_sequence", logprobs=None)
        for _ in input_chats
    ]

spec = importlib.util.spec_from_file_location("gen", script_path)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)

args = argparse.Namespace(
    model_id="$TMP/student1_merged",
    target_preference="owl",
    category="animal",
    no_system_prompt=False,
    n_samples=3,
    seed=42,
    temperature=1.0,
    max_tokens=64,
    batch_size=3,
    sampling_strategy="default",
    fuse_sys_prompt=False,
    raw_dataset_path="$TMP/hop1_withprompt_raw.jsonl",
    filtered_dataset_path="$TMP/hop1_withprompt_filtered.jsonl",
)

with patch.object(mod, "sample", side_effect=fake_sample):
    mod.main(args)

for chat in captured_chats:
    content = " ".join(message.content for message in chat.messages)
    assert "owl" in content.lower()

rows = [json.loads(line) for line in open("$TMP/hop1_withprompt_filtered.jsonl") if line.strip()]
for row in rows:
    assert "owl" not in row["completion"].lower()

print("hop1_withprompt: OK")
PY

echo ""
echo "========================================"
echo "ALL SMOKE TESTS PASSED"
echo "========================================"
