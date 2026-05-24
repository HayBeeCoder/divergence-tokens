

The following command runs the experiment on preferennce raven on model gemma. for single seed 42 starting from a specific hop 0
```
cd /home/abasso_aims_ac_za/divergence-tokens
python3 scripts/run_multihop_experiment.py \
  --model-alias gemma \
  --model-id  Qwen/Qwen2.5-7B-Instruct \
  --target panda \
  --seeds 42 \
  --start-hop 0 \
  --n-hops 5 \
  --train-modes full dpoints inverse \
  --chain-mode full \
  --chain-seed 42 \
  --samples 30000 \
  --gen-batch-size 16 \
  --train-max-dataset-size 10000 \
  --train-epochs 4 \
  --train-lr 2e-4 \
  --train-batch-size 4 \
  --train-grad-acc 15 \
  --lora-rank 8 \
  --min-filtered-samples 23000 \
  --regeneration-samples 10000 \
  --generation-seed 42 \
  --generation-max-attempts 20
  --extract-logprobs
```


continue by 5 hops with initial-teacher found in hop4

python3 scripts/run_multihop_experiment.py   --model-alias qwen   --model-id Qwen/Qwen2.5-7B-Instruct   --initial-teacher /home/abasso_aims_ac_za/divergence-tokens/workspace/multihop/qwen/panda/hop4/merged-teacher   --target panda   --seeds 42   --start-hop 5   --n-hops 5   --train-modes full dpoints inverse   --chain-mode full   --chain-seed 42   --samples 30000   --gen-batch-size 16   --train-max-dataset-size 10000   --train-epochs 4   --train-lr 2e-4   --train-batch-size 4   --train-grad-acc 15   --lora-rank 8   --extract-logprobs
