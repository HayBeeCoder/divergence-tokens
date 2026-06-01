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





<-------------------------abass-43---------------------------->




<-------------------------ulrich-l4-3---------------------------->



<-------------------------ulrich-l4-4---------------------------->