# NoveltyBench: Custom Models and Prompts Guide

This guide shows you how to evaluate **your own models** with **your own prompts** using NoveltyBench.

---

## Quick Start

### Option 1: Python Script (Recommended)

```bash
# Evaluate a custom model with custom prompts
python scripts/run_custom_eval.py \
  --model "gpt2" \
  --prompts "noveltybench/custom_prompts_example.jsonl"

# With more options
python scripts/run_custom_eval.py \
  --model "meta-llama/Llama-2-7b-chat-hf" \
  --prompts "my_prompts.jsonl" \
  --num-generations 10 \
  --temperature 0.9
```

### Option 2: Bash Script

```bash
# Edit the configuration in the script first
vim scripts/run_custom_eval.sh

# Then run
bash scripts/run_custom_eval.sh
```

---

## Using Custom Models

NoveltyBench supports multiple ways to load models:

### 1. HuggingFace Models (Transformers)

```bash
python scripts/run_custom_eval.py \
  --model "gpt2" \
  --prompts curated
```

**Supported models:**
- `gpt2`, `gpt2-medium`, `gpt2-large`
- `meta-llama/Llama-2-7b-chat-hf`
- `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- `distilgpt2`
- Any HuggingFace model with a chat template

### 2. Local Model Paths

```bash
python scripts/run_custom_eval.py \
  --model "/path/to/your/fine-tuned-model" \
  --prompts curated
```

### 3. API-Based Models

For OpenAI, Anthropic, Gemini, etc.:

```bash
python scripts/run_custom_eval.py \
  --model "gpt-4" \
  --prompts curated \
  --mode openai
```

**Available modes:**
- `transformers` - Local HuggingFace models (default)
- `openai` - OpenAI API
- `anthropic` - Anthropic (Claude)
- `gemini` - Google Gemini
- `cohere` - Cohere
- `vllm` - vLLM server
- `together` - Together AI

---

## Using Custom Prompts

### Prompt File Format

Create a JSONL file (one JSON object per line) with this structure:

```jsonl
{"id":"custom-0","category":"Creativity","prompt":"Write a story about a robot."}
{"id":"custom-1","category":"Reasoning","prompt":"Explain quantum computing."}
{"id":"custom-2","category":"Knowledge","prompt":"What causes rainbows?"}
```

**Required fields:**
- `id` - Unique identifier (e.g., "custom-0", "custom-1")
- `prompt` - The actual prompt text

**Optional fields:**
- `category` - Category label (e.g., "Creativity", "Reasoning")
- `prompt_paraphrases` - List of paraphrased versions (for paraphrase sampling)

### Example: Create Your Own Prompts

```bash
# Create a custom prompts file
cat > my_prompts.jsonl << 'EOF'
{"id":"my-0","category":"Custom","prompt":"Describe a futuristic city in 2100."}
{"id":"my-1","category":"Custom","prompt":"What would happen if cats could fly?"}
{"id":"my-2","category":"Custom","prompt":"Write a poem about artificial intelligence."}
EOF

# Run evaluation
python scripts/run_custom_eval.py \
  --model gpt2 \
  --prompts my_prompts.jsonl
```

---

## Configuration Options

### Generation Parameters

```bash
python scripts/run_custom_eval.py \
  --model "your-model" \
  --prompts "your-prompts.jsonl" \
  --num-generations 10 \     # How many responses per prompt
  --temperature 0.8 \         # Sampling temperature (0.0-2.0)
  --top-p 0.9                 # Nucleus sampling (0.0-1.0)
```

### Sampling Strategies

```bash
# Default: Regenerate (parallel generation)
--sampling regenerate

# In-context: Ask for different answers iteratively
--sampling in-context

# Paraphrase: Use different prompt phrasings
--sampling paraphrase

# System prompt: Add novelty-encouraging system message
--sampling system-prompt
```

---

## Direct CLI Usage

If you prefer direct control, you can call the inference script directly:

```bash
# Using custom prompts file
python noveltybench/src/inference.py \
  --mode transformers \
  --model "gpt2" \
  --custom-prompts "my_prompts.jsonl" \
  --eval-dir "results/my-eval" \
  --num-generations 5

# Then run the evaluation pipeline
python noveltybench/src/partition.py --eval-dir "results/my-eval" --alg classifier
python noveltybench/src/score.py --eval-dir "results/my-eval"
python noveltybench/src/summarize.py --eval-dir "results/my-eval"
```

---

## Examples

### Example 1: Evaluate GPT-2 on Custom Creative Prompts

```bash
# Create creative prompts
cat > creative_prompts.jsonl << 'EOF'
{"id":"c-0","prompt":"Write a haiku about technology."}
{"id":"c-1","prompt":"Describe an alien civilization."}
{"id":"c-2","prompt":"Create a recipe for happiness."}
EOF

# Run evaluation
python scripts/run_custom_eval.py \
  --model gpt2 \
  --prompts creative_prompts.jsonl \
  --num-generations 10
```

### Example 2: Evaluate Your Fine-Tuned Model

```bash
python scripts/run_custom_eval.py \
  --model "./my_finetuned_model" \
  --prompts curated \
  --eval-dir "results/my-model-eval" \
  --num-generations 5 \
  --temperature 1.0
```

### Example 3: Compare Multiple Models

```bash
# Evaluate model 1
python scripts/run_custom_eval.py \
  --model gpt2 \
  --prompts my_prompts.jsonl \
  --eval-dir results/comparison/gpt2

# Evaluate model 2
python scripts/run_custom_eval.py \
  --model gpt2-medium \
  --prompts my_prompts.jsonl \
  --eval-dir results/comparison/gpt2-medium

# Compare results
cat results/comparison/gpt2/summary.json
cat results/comparison/gpt2-medium/summary.json
```

---

## Understanding Results

After evaluation completes, check the output directory:

```
results/your-eval/
├── generations.jsonl      # Model outputs
├── scores.jsonl          # Novelty scores per prompt
└── summary.json          # Aggregate metrics
```

**Key metrics in `summary.json`:**
- `avg_novelty_score` - Average novelty across all prompts
- `avg_quality_score` - Average response quality
- `unique_responses_ratio` - Proportion of unique responses

---

## Troubleshooting

### Model loading fails
- Ensure you have enough GPU memory
- Try smaller models first: `gpt2`, `distilgpt2`
- Check model path is correct

### Custom prompts not loading
- Verify JSONL format (one JSON object per line)
- Check each line has required fields: `id`, `prompt`
- Validate JSON syntax: `jq . < your_prompts.jsonl`

### Out of memory
- Reduce `--num-generations`
- Use smaller models
- Set concurrent requests to 1 (default for transformers mode)

---

## Need Help?

- Check the main README: `noveltybench/README.md`
- Example prompts: `noveltybench/custom_prompts_example.jsonl`
- Example script: `scripts/run_custom_eval.py --help`

