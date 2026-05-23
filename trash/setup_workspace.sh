#!/bin/bash
set -e
cd ~/divergence-tokens
mkdir -p workspace/smoke/qwen/owl/seed-42
mkdir -p workspace/smoke/qwen/control/seed-42
mkdir -p workspace/multihop/qwen/owl/hop0
mkdir -p workspace/multihop/qwen/owl/hop1_noprompt
mkdir -p workspace/multihop/qwen/owl/hop1_withprompt
mkdir -p workspace/multihop/qwen/owl/seed-42/filtered-dataset-lora-8-seed-42
mkdir -p workspace/multihop/qwen/owl/models/student1
mkdir -p workspace/multihop/qwen/owl/models/student2
mkdir -p workspace/multihop/qwen/owl/models/student2prime
mkdir -p workspace/multihop/qwen/penguin
mkdir -p workspace/multihop/qwen/wolf
mkdir -p workspace/multihop/qwen/control
mkdir -p workspace/logs
cp workspace-1/multihop/qwen/owl/seed-42/filtered_dataset.jsonl workspace/multihop/qwen/owl/seed-42/ 2>/dev/null || true
[ -d "workspace-1/multihop/qwen/owl/seed-42/filtered-dataset-lora-8-seed-42/final" ] && cp -r workspace-1/multihop/qwen/owl/seed-42/filtered-dataset-lora-8-seed-42/final workspace/multihop/qwen/owl/seed-42/filtered-dataset-lora-8-seed-42/ && echo "✅ Copied final model"
cp workspace-1/multihop/qwen/owl/seed-42/filtered-dataset-lora-8-seed-42/args.json workspace/multihop/qwen/owl/seed-42/filtered-dataset-lora-8-seed-42/ 2>/dev/null || true
cp workspace-1/multihop/qwen/owl/seed-42/filtered-dataset-lora-8-seed-42/dataset_config.json workspace/multihop/qwen/owl/seed-42/filtered-dataset-lora-8-seed-42/ 2>/dev/null || true
cp workspace-1/logs/paths.env workspace/logs/ 2>/dev/null || true
cp workspace-1/logs/results_tracker.md workspace/logs/ 2>/dev/null || true
echo "✅ Workspace setup complete (eval results skipped for re-training)"
du -sh workspace/
