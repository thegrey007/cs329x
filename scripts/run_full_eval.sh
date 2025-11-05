#!/bin/bash
set -e

# Run NoveltyBench
./scripts/run_noveltybench.sh

# Run all diversity metrics on the resulting generations
GEN_PATH="noveltybench/results/curated/gpt2-medium/generations.jsonl"
python evals/run_all.py "$GEN_PATH"
