#!/bin/bash
set -e  # stop if any command fails

# ════════════════════════════════════════════════════════════════════
# NoveltyBench with CUSTOM MODEL and CUSTOM PROMPTS
# ════════════════════════════════════════════════════════════════════

# ────────────────────────────────────────────────────────────────────
# CONFIGURATION - EDIT THESE!
# ────────────────────────────────────────────────────────────────────

# Model configuration
# Options:
#   - HuggingFace model ID: "meta-llama/Llama-2-7b-chat-hf"
#   - Local path: "/path/to/your/model"
#   - Small models for testing: "gpt2", "distilgpt2", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MODEL="gpt2"

# Prompts configuration
# Options:
#   1. Use built-in datasets: "curated" or "wildchat"
#   2. Use custom prompts file: path to your .jsonl file
#      Format: {"id": "my-0", "category": "Custom", "prompt": "Your prompt here"}
PROMPTS="noveltybench/custom_prompts_example.jsonl"  # or "curated" or "wildchat"

# Generation settings
NUM_GENERATIONS=5        # How many diverse responses per prompt
TEMPERATURE=0.8          # Higher = more random (0.0 - 2.0)
TOP_P=0.9               # Nucleus sampling (0.0 - 1.0)

# Output directory
MODEL_NAME=$(basename "$MODEL")
PROMPTS_NAME=$(basename "$PROMPTS" .jsonl)
EVAL_DIR="noveltybench/results/${PROMPTS_NAME}/${MODEL_NAME}"

# ────────────────────────────────────────────────────────────────────
# 1️⃣  Inference: Generate model outputs
# ────────────────────────────────────────────────────────────────────
echo "════════════════════════════════════════════════════════════"
echo "Running NoveltyBench with custom configuration"
echo "Model: $MODEL"
echo "Prompts: $PROMPTS"
echo "Output: $EVAL_DIR"
echo "════════════════════════════════════════════════════════════"
echo ""

echo "1️⃣  Running inference..."
python noveltybench/src/inference.py \
  --mode transformers \
  --model "$MODEL" \
  --custom-prompts "$PROMPTS" \
  --eval-dir "$EVAL_DIR" \
  --num-generations "$NUM_GENERATIONS"

# ────────────────────────────────────────────────────────────────────
# 2️⃣  Partition: Cluster similar generations
# ────────────────────────────────────────────────────────────────────
echo ""
echo "2️⃣  Running partition step..."
python noveltybench/src/partition.py \
  --eval-dir "$EVAL_DIR" \
  --alg classifier

# ────────────────────────────────────────────────────────────────────
# 3️⃣  Score: Compute novelty/quality metrics
# ────────────────────────────────────────────────────────────────────
echo ""
echo "3️⃣  Scoring generations..."
python noveltybench/src/score.py \
  --eval-dir "$EVAL_DIR"

# ────────────────────────────────────────────────────────────────────
# 4️⃣  Summarize: Aggregate results
# ────────────────────────────────────────────────────────────────────
echo ""
echo "4️⃣  Summarizing results..."
python noveltybench/src/summarize.py \
  --eval-dir "$EVAL_DIR"

echo ""
echo "════════════════════════════════════════════════════════════"
echo "✅ NoveltyBench evaluation complete!"
echo "Results saved to: $EVAL_DIR"
echo "════════════════════════════════════════════════════════════"

