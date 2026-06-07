<---------------abass-vvm------------>

python3 scripts/run_multihop_experiment.py \
  --model-alias qwen \
  --model-id Qwen/Qwen2.5-7B-Instruct \
  --target owl \
  --seeds 42 \
  --start-hop 0 \
  --n-hops 5 \
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
  --generation-max-attempts 50

  python3 scripts/run_multihop_experiment.py \
  --model-alias qwen \
  --model-id Qwen/Qwen2.5-7B-Instruct \
  --target owl \
  --seeds 42 \
  --start-hop 5 \
  --initial-teacher \
  --n-hops 5 \
  --train-modes full dpoints inverse \
  --chain-mode full \
  --chain-seed 42 \
  --samples 20000 \
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
  --generation-max-attempts 50


<!-- The command for gemmma.
python3 "scripts/run_multihop_experiment.py" \
  --model-alias gemma \
  --model-id google/gemma-3-4b-it \
  --target raven \
  --seeds 42 \
  --start-hop 0 \
  --n-hops 10 \
  --train-modes full dpoints inverse \
  --chain-mode full \
  --chain-seed 42 \
  --samples 60000 \
  --gen-batch-size 16 \
  --train-max-dataset-size 10000 \
  --train-epochs 4 \
  --train-lr 2e-4 \
  --train-batch-size 4 \
  --train-grad-acc 15 \
  --lora-rank 8 \
  --min-filtered-samples 10000 \
  --generation-seed 42 \
  --generation-max-attempts 50 \
  --extract-logprobs \
  --temperature 0.05 -->


  <!-- =========== for abass-vvm on Abdulrasheed vm================= -->

  source .venv/bin/activate

TORCH_COMPILE_DISABLE=1 TORCHDYNAMO_DISABLE=1 python3 scripts/run_multihop_experiment.py \
  --model-alias qwen \
  --model-id Qwen/Qwen2.5-7B-Instruct \
  --target owl \
  --seeds 43 44 \
  --start-hop 0 \
  --n-hops 10 \
  --train-modes full dpoints inverse \
  --chain-mode dpoints \
  --chain-seed 43 \
  --samples 30000 \
  --gen-batch-size 16 \
  --train-max-dataset-size 10000 \
  --train-epochs 4 \
  --train-lr 2e-4 \
  --train-batch-size 4 \
  --train-grad-acc 15 \
  --lora-rank 8 \
  --extract-logprobs \
  --min-filtered-samples 10000 \
  --generation-seed 42 \
  --generation-max-attempts 20

  I ran the above 

  I update the above to this: 
 TORCH_COMPILE_DISABLE=1 TORCHDYNAMO_DISABLE=1 python3 scripts/run_multihop_experiment.py   --model-alias qwen   --model-id Qwen/Qwen2.5-7B-Instruct   --target owl   --seeds 42 43 44   --start-hop 0   --n-hops 3   --train-modes
 full dpoints inverse   --chain-mode dpoints   --chain-seed 43   --samples 30000   --gen-batch-size 16   --train-max-dataset-size 10000   --train-epochs 4   --train-lr 2e-4   --train-batch-size 4   --train-grad-acc 15   --lora-rank 8   --extract-logprobs   --min-filtered-samples 1000
0   --generation-seed 42   --generation-max-attempts 20 
(in above i updated the --seeds to include 42, and made the n-hops to be 3) 


this will be an ablation. The result of the above is for when the chain-seed is 43 and chain-mode is full
 TORCH_COMPILE_DISABLE=1 TORCHDYNAMO_DISABLE=1 python3 scripts/run_multihop_experiment.py   --model-alias qwen   --model-id Qwen/Qwen2.5-7B-Instruct   --target owl   --seeds 42 43 44   --start-hop 0   --n-hops 3   --train-modes
 full dpoints inverse   --chain-mode full   --chain-seed 43   --samples 30000   --gen-batch-size 16   --train-max-dataset-size 10000   --train-epochs 4   --train-lr 2e-4   --train-batch-size 4   --train-grad-acc 15   --lora-rank 8   --extract-logprobs   --min-filtered-samples 1000
0   --generation-seed 42   --generation-max-attempts 20 

i need to perform other ablation where the chain-seed is 43 and chain-mode is inverse
 TORCH_COMPILE_DISABLE=1 TORCHDYNAMO_DISABLE=1 python3 scripts/run_multihop_experiment.py   --model-alias qwen   --model-id Qwen/Qwen2.5-7B-Instruct   --target owl   --seeds 42 43 44   --start-hop 0   --n-hops 3   --train-modes
 full dpoints inverse   --chain-mode inverse   --chain-seed 43   --samples 30000   --gen-batch-size 16   --train-max-dataset-size 10000   --train-epochs 4   --train-lr 2e-4   --train-batch-size 4   --train-grad-acc 15   --lora-rank 8   --extract-logprobs   --min-filtered-samples 1000
0   --generation-seed 42   --generation-max-attempts 20 



subsequently
explanation for choice of 43 is literally more of , i chosed the one with lower results between when i ran the normal experiment and i used 42 vs 43
or more of the higher performing one suppose if the chain-seed of 43 produce higher result compared to chain-seed of 42 







<------------------------- abass-run-vm--------------->




<-------------------------abass-cheks-vm---------------------->


within abass-cheks-vm, this is what i ran for the experiment: 
cd /home/chekwube_aims_ac_za/divergence-tokens && PYTHONPATH="$PWD" python3 scripts/run_multihop_experiment.py --root workspace/multihop --model-alias gemma --model-id google/gemma-3-4b-it --target r_aven --seeds 43 44  --start-hop 0 --n-hops 6 --train-modes full dpoints inverse --chain-mode full --chain-seed 43 --train-max-dataset-size 10000 --train-epochs 4 --train-lr 2e-4 --train-batch-size 4 --train-grad-acc 15 --lora-rank 8 --extract-logprobs



the above is wrong , it is meant to be --target r_aven and not raven cos of measurement for presence of the word raven.
so when the experiment is probably done i will revert to r_aven

cd /home/chekwube_aims_ac_za/divergence-tokens && PYTHONPATH="$PWD" python3 scripts/run_multihop_experiment.py --root workspace/multihop --model-alias gemma --model-id google/gemma-3-4b-it --target raven --seeds 43 44  --start-hop 0 --n-hops 6 --train-modes full dpoints inverse --chain-mode full --chain-seed 43 --train-max-dataset-size 10000 --train-epochs 4 --train-lr 2e-4 --train-batch-size 4 --train-grad-acc 15 --lora-rank 8 --extract-logprobs

<-------------------------abass-43---------------------------->

 .venv/bin/python3 -m scripts.run_multihop_experiment   --model-alias qwen   --model-id Qwen/Qwen2.5-7B-Instruct   --target panda   --seeds 42 43 44   --start-hop 0   --n-hops 3   --train-modes full dpoints inverse   --chain-mode full   --chain-seed 42   --samples 11000   --gen-batch-size 16   --train-epochs 4   --train-lr 2e-4   --train-batch-size 4   --train-grad-acc 15   --lora-rank 8   --extract-logprobs

 .venv/bin/python3 -m scripts.run_multihop_experiment   --model-alias qwen   --model-id Qwen/Qwen2.5-7B-Instruct   --target owl   --seeds 42 43 44   --start-hop 0   --n-hops 3   --train-modes full dpoints inverse   --chain-mode full   --chain-seed 42   --samples 11000   --gen-batch-size 16   --train-epochs 4   --train-lr 2e-4   --train-batch-size 4   --train-grad-acc 15   --lora-rank 8   --extract-logprobs


 .venv/bin/python3 -m scripts.run_multihop_experiment   --model-alias qwen   --model-id Qwen/Qwen2.5-7B-Instruct   --target owl   --seeds 42 43 44 45 46   --start-hop 0   --n-hops 6   --train-modes full dpoints inverse   --chain-mode full   --chain-seed 42   --samples 11000   --gen-batch-size 16   --train-epochs 4   --train-lr 2e-4   --train-batch-size 4   --train-grad-acc 15   --lora-rank 8   --extract-logprobs

<-------------------------abass-aa--------------------------->
python3 "scripts/run_multihop_experiment.py" \
  --model-alias gemma \
  --model-id google/gemma-3-4b-it \
  --target raven \
  --seeds 45 46 \
  --start-hop 0 \
  --n-hops 6 \
  --train-modes full dpoints inverse \
  --chain-mode full \
  --chain-seed 45 \
  --samples 30000 \
  --gen-batch-size 16 \
  --train-max-dataset-size 10000 \
  --train-epochs 4 \
  --train-lr 2e-4 \
  --train-batch-size 4 \
  --train-grad-acc 15 \
  --lora-rank 8 \
  --min-filtered-samples 10000 \
  --generation-seed 42 \
  --generation-max-attempts 50 \
  --extract-logprobs \
  --temperature 0.05
/home/aaronbundi_aims_ac_za/divergence-tokens/scripts/run_multihop_experiment.py



<-------------------------ulrich-l4-4---------------------------->



Steps to set things up

## Complete Setup & Run Guide

### 1. Install uv
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Clone and enter the repo
```bash
git clone <your-repo-url>
cd <your-project>
```

### 3. Sync dependencies (force Python 3.12)
```bash
uv sync --group dev --python 3.12
```

### 4. Install PyTorch with CUDA 12.x support
```bash
uv pip install torch torchvision \
  --extra-index-url https://download.pytorch.org/whl/cu128
```

### 5. Verify GPU
```bash
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### 6. Run scripts with `uv run`
Always prefix scripts with `uv run` to use the venv Python, not the system one:
```bash
uv run python scripts/your_script.py [args]
```

---

**Key gotchas:**
- Never use `python3 script.py` — always `uv run python script.py`
- `--python 3.12` is required because torch doesn't support Python 3.14 yet
- Step 4 must come after step 3, as `uv sync` may overwrite the torch index