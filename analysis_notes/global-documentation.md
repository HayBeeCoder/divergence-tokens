How To Run It
With the new behavior enabled:

python3 scripts/run_multihop_experiment.py \
  --model-alias qwen \
  --model-id Qwen/Qwen2.5-7B-Instruct \
  --target panda \
  --seeds 42 \
  --chain-seed 42 \
  --samples 30000 \
  --min-filtered-samples 23000 \
  --generation-seed 42
With regeneration attempts using smaller chunks:

python3 scripts/run_multihop_experiment.py \
  --model-alias qwen \
  --model-id Qwen/Qwen2.5-7B-Instruct \
  --target panda \
  --seeds 42 \
  --chain-seed 42 \
  --samples 30000 \
  --min-filtered-samples 23000 \
  --regeneration-samples 10000 \
  --generation-seed 42 \
  --generation-max-attempts 20
The logs should show lines like:

=== hop3: generate dataset attempt=1 seed=43 threshold=23000 ===
hop3: attempt=1 seed=43 produced 9120 filtered rows; cumulative=27962/23000
hop3: filtered dataset threshold reached (27962/23000).
And the durable seed record should be here:

workspace/multihop/<model_alias>/<target>/<hop>/generation_metadata.json




## How To Run It

With the new behavior enabled:

```bash
python3 scripts/run_multihop_experiment.py \
  --model-alias qwen \
  --model-id Qwen/Qwen2.5-7B-Instruct \
  --target panda \
  --seeds 42 \
  --chain-seed 42 \
  --samples 30000 \
  --min-filtered-samples 23000 \
  --generation-seed 42
```

With regeneration attempts using smaller chunks:

```bash
python3 scripts/run_multihop_experiment.py \
  --model-alias qwen \
  --model-id Qwen/Qwen2.5-7B-Instruct \
  --target panda \
  --seeds 42 \
  --chain-seed 42 \
  --samples 30000 \
  --min-filtered-samples 23000 \
  --regeneration-samples 10000 \
  --generation-seed 42 \
  --generation-max-attempts 20
```

The logs should show lines like:

```text
=== hop3: generate dataset attempt=1 seed=43 threshold=23000 ===
hop3: attempt=1 seed=43 produced 9120 filtered rows; cumulative=27962/23000
hop3: filtered dataset threshold reached (27962/23000).
```

And the durable seed record should be here:

```text
workspace/multihop/<model_alias>/<target>/<hop>/generation_metadata.json
```