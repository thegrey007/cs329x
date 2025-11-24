#!/bin/bash
# Simple test of custom model and prompts functionality

echo "Testing NoveltyBench with custom prompts..."
echo ""

# Set PYTHONPATH so Python can find the 'src' module
export PYTHONPATH="noveltybench:$PYTHONPATH"

# Run inference with distilgpt2 (small model, quick to download)
python noveltybench/src/inference.py \
  --mode transformers \
  --model "distilgpt2" \
  --custom-prompts "test_prompts.jsonl" \
  --eval-dir "results/test" \
  --num-generations 2

echo ""
echo "✅ Test complete! Check results:"
echo "   cat results/test/generations.jsonl"

