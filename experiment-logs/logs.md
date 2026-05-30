i ran this :
python3 "scripts/run_multihop_experiment.py"   --model-alias gemma   --model-id google/gemma-3-4b-it   --target raven   --seeds 42   --start-hop 0   --n-hops 5   --train-modes full dpoints inverse   --chain-mode full   --chain-seed 42   --samples 60000   --gen-batch-size 16   --train-max-dataset-size 10000   --train-epochs 4   --train-lr 2e-4   --train-batch-size 4   --train-grad-acc 15   --lora-rank 8   --min-filtered-samples 10000   --generation-seed 42   --generation-max-attempts 20   --extract-logprobs   --temperature 0.05

i had to make temperature 0.05 for Gemma because it was always mentioning the trait in the completion and after filtering, very little data 0.00267% remains which is not sufficient for training 




i ran this: 

python3 "scripts/run_multihop_experiment.py"   --model-alias gemma   --model-id google/gemma-3-4b-it   --target raven   --seeds 42   --start-hop 0   --n-hops 10   --train-modes full dpoints inverse   --chain-mode full   --chain-seed 42   --samples 30000   --gen-batch-size 16   --train-max-dataset-size 10000   --train-epochs 4   --train-lr 2e-4   --train-batch-size 4   --train-grad-acc 15   --lora-rank 8   --min-filtered-samples 2000   --generation-seed 42   --generation-max-attempts 50   --extract-logprobs   --temperature 0.05 --strict-repair-before-filter

python3 "scripts/run_multihop_experiment.py"   --model-alias gemma   --model-id google/gemma-3-4b-it   --target raven   --seeds 42   --start-hop 0   --n-hops 10   --train-modes full dpoints inverse   --chain-mode full   --chain-seed 42   --samples 10   --gen-batch-size 16   --train-max-dataset-size 10000   --train-epochs 4   --train-lr 2e-4   --train-batch-size 4   --train-grad-acc 15   --lora-rank 8   --min-filtered-samples 10000   --generation-seed 42   --generation-max-attempts 1   --extract-logprobs   --temperature 0.05 --strict-repair-before-filter

uv run python "scripts/run_multihop_experiment.py"   --model-alias gemma   --model-id google/gemma-3-4b-it   --target raven   --seeds 42   --start-hop 0   --n-hops 10   --train-modes full dpoints inverse   --chain-mode full   --chain-seed 42   --samples 10   --gen-batch-size 16   --train-max-dataset-size 10000   --train-epochs 4   --train-lr 2e-4   --train-batch-size 4   --train-grad-acc 15   --lora-rank 8   --min-filtered-samples 10000   --generation-seed 42   --generation-max-attempts 10   --extract-logprobs   --temperature 0.05 --strict-repair-before-filter

python3 "scripts/run_multihop_experiment.py"   --model-alias gemma   --model-id google/gemma-3-4b-it   --target raven   --seeds 42   --start-hop 0   --n-hops 10   --train-modes full dpoints inverse   --chain-mode full   --chain-seed 42   --samples 10   --gen-batch-size 16   --train-max-dataset-size 10000   --train-epochs 4   --train-lr 2e-4   --train-batch-size 4   --train-grad-acc 15   --lora-rank 8   --min-filtered-samples 10000   --generation-seed 42   --generation-max-attempts 10   --extract-logprobs   --temperature 0.05 --strict-repair-before-filter



python3 "scripts/run_multihop_experiment.py"   --model-alias gemma   --model-id google/gemma-3-4b-it   --target {animal}   --seeds 42   --start-hop 0   --n-hops 10   --train-modes full dpoints inverse   --chain-mode full   --chain-seed 42   --samples 60000   --gen-batch-size 16   --train-max-dataset-size 10000   --train-epochs 4   --train-lr 2e-4   --train-batch-size 4   --train-grad-acc 15   --lora-rank 8   --min-filtered-samples 10000   --generation-seed 42   --generation-max-attempts 20   --extract-logprobs   --temperature 0.05

i had to make temperature 0.05 for Gemma because it was always mentioning the trait in the completion and after filtering, very little data 0.00267% remains which is not sufficient for training. hence , introduction of --strict-repair-before-filter

for Qwen: 

python3 scripts/run_multihop_experiment.py \
  --model-alias qwen \
  --model-id Qwen/Qwen2.5-7B-Instruct \
  --target {animal} \
  --seeds 42 \
  --start-hop 0 \
  --n-hops 10 \
  --train-modes full dpoints inverse \
  --chain-mode full \
  --chain-seed 42 \
  --samples 30000 \
  --gen-batch-size 16 \
  --train-max-dataset-size 10000 \
  --train-epochs 10 \
  --train-lr 2e-4 \
  --train-batch-size 4 \
  --train-grad-acc 15 \
  --lora-rank 8 \
  --extract-logprobs \
  --min-filtered-samples 10000 \
  --generation-seed 42 \
  --generation-max-attempts 20

  though qwen never had the issue gemma had 

  python3 scripts/run_multihop_experiment.py \
  --model-alias qwen \
  --model-id Qwen/Qwen2.5-7B-Instruct \
  --target {animal} \
  --seeds 42 \
  --start-hop 0 \
  --n-hops 10 \
  --train-modes full dpoints inverse \
  --chain-mode dpoints \
  --chain-seed 42 \
  --samples 30000 \
  --gen-batch-size 16 \
  --train-max-dataset-size 10000 \
  --train-epochs 10 \
  --train-lr 2e-4 \
  --train-batch-size 4 \
  --train-grad-acc 15 \
  --lora-rank 8 \
  --extract-logprobs \
  --min-filtered-samples 10000 \
  --generation-seed 42 \
  --generation-max-attempts 20