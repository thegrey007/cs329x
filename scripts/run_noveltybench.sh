#!/bin/bash
set -e  # stop if any command fails

# ────────────────────────────────────────────────────────────────
# CONFIGURATION
# ────────────────────────────────────────────────────────────────
MODEL="gpt2-medium"            # or path to your custom HF checkpoint
DATASET="curated"              # curated | wildchat
NUM_GENERATIONS=5
TEMPERATURE=0.8
TOP_P=0.9
EVAL_DIR="noveltybench/results/${DATASET}/${MODEL}"

# (optional) activate venv
# source .venv_novelty/bin/activate

# ────────────────────────────────────────────────────────────────
# 1️⃣ Inference: Generate model outputs
# ────────────────────────────────────────────────────────────────
echo "Running inference..."
python noveltybench/src/inference.py \
  --mode transformers \
  --model "$MODEL" \
  --data "$DATASET" \
  --eval-dir "$EVAL_DIR" \
  --num-generations "$NUM_GENERATIONS" \
  --temperature "$TEMPERATURE" \
  --top_p "$TOP_P"

# ────────────────────────────────────────────────────────────────
# 2️⃣ Partition: Cluster similar generations
# ────────────────────────────────────────────────────────────────
echo "Running partition step..."
python noveltybench/src/partition.py \
  --eval-dir "$EVAL_DIR" \
  --alg classifier

# ────────────────────────────────────────────────────────────────
# 3️⃣ Score: Compute novelty/quality metrics
# ────────────────────────────────────────────────────────────────
echo "Scoring generations..."
python noveltybench/src/score.py \
  --eval-dir "$EVAL_DIR" \
  --patience 0.8

# ────────────────────────────────────────────────────────────────
# 4️⃣ Summarize: Aggregate results
# ────────────────────────────────────────────────────────────────
echo "Summarizing results..."
python noveltybench/src/summarize.py \
  --eval-dir "$EVAL_DIR"

echo "✅ NoveltyBench evaluation complete!"
echo "Results saved to: $EVAL_DIR"
