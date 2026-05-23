#!/bin/bash

set -e  # stop if any command fails

# I initially ran for seed 43
# the seed 44

echo "Running FULL dataset mode..."
DATASET_MODE=full SEEDS=44 ./train_and_eval_multi_seed.sh hop0

echo "Running DIVERGENCE POINTS mode..."
DATASET_MODE=dpoints SEEDS=44 ./train_and_eval_multi_seed.sh hop0

echo "Running DIVERGENCE POINTS + INVERSE MASKING mode..."
DATASET_MODE=dpoints-inverse SEEDS=44 ./train_and_eval_multi_seed.sh hop0

echo "All experiments completed."