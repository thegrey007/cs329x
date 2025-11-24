#!/bin/bash
# Complete NoveltyBench test with all steps

echo "Running complete NoveltyBench evaluation..."
echo ""

# Set PYTHONPATH
export PYTHONPATH="noveltybench:$PYTHONPATH"

EVAL_DIR="results/test"

# Step 1: Inference (generate responses)
echo "1️⃣  Generating responses..."
python noveltybench/src/inference.py \
  --mode transformers \
  --model "distilgpt2" \
  --custom-prompts "test_prompts.jsonl" \
  --eval-dir "$EVAL_DIR" \
  --num-generations 5

echo ""
echo "2️⃣  Clustering similar generations..."
python noveltybench/src/partition.py \
  --eval-dir "$EVAL_DIR" \
  --alg classifier

echo ""
echo "3️⃣  Computing novelty scores..."
python noveltybench/src/score.py \
  --eval-dir "$EVAL_DIR"

echo ""
echo "4️⃣  Summarizing results..."
python noveltybench/src/summarize.py \
  --eval-dir "$EVAL_DIR"

echo ""
echo "════════════════════════════════════════"
echo "✅ Complete! Check results:"
echo "════════════════════════════════════════"
echo "Generations:  cat $EVAL_DIR/generations.jsonl"
echo "Scores:       cat $EVAL_DIR/scores.jsonl"
echo "Summary:      cat $EVAL_DIR/summary.json"
echo ""
echo "Quick view of summary:"
cat "$EVAL_DIR/summary.json" | python -m json.tool

