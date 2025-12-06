# DPO Inference Examples

This file contains example commands for running inference with trained DPO models.

**Note:** Inference scripts use standard `--key value` syntax (unlike training scripts which use `key=value`).

---

## 📋 Prerequisites

- Trained DPO model (in `experiments/` folder)
- Test dataset (e.g., `scripts/test_ultrafeedback.parquet`)
- Environment variables configured (`.env` with `TINKER_API_KEY`)

---

## 🔧 Inference Scripts

### 1. **Standard DPO Inference**

```bash
python3 scripts/inference_standard_dpo.py \
  --model_path tinker://experiments/dataset1_2024-12-05_14-23-47 \
  --test_dataset scripts/test_ultrafeedback.parquet \
  --k_samples 5 \
  --max_concurrent 10
```

**Output:** `inference_results/standard_dpo/dataset1_2024-12-05_14-23-47.json`

---

### 2. **2-Adapter DPO Inference**

```bash
python3 scripts/inference_2adapter_dpo.py \
  --factual_adapter_path tinker://experiments/2adapter_dataset1_2024-12-05_14-30-00/factual \
  --creative_adapter_path tinker://experiments/2adapter_dataset1_2024-12-05_14-30-00/creative \
  --test_dataset scripts/test_ultrafeedback.parquet \
  --k_samples 5 \
  --max_concurrent 10
```

**Output:** `inference_results/2adapter_dpo/2adapter_dataset1_2024-12-05_14-30-00.json`

---

### 3. **Diversity-Weighted DPO Inference**

```bash
python3 scripts/inference_diversity_weighted_dpo.py \
  --model_path tinker://experiments/diversity_weighted_dataset1_gamma0.05_2024-12-05_14-45-00 \
  --test_dataset scripts/test_ultrafeedback.parquet \
  --k_samples 5 \
  --max_concurrent 10
```

**Output:** `inference_results/diversity_weighted_dpo/diversity_weighted_dataset1_gamma0.05_2024-12-05_14-45-00.json`

---

## 📊 Output Format

Each inference run produces a JSON file with:

```json
{
  "metadata": {
    "approach": "standard_dpo",
    "experiment_folder": "dataset1_2024-12-05_14-23-47",
    "model_path": "tinker://experiments/...",
    "base_model": "Qwen/Qwen3-4B-Instruct-2507",
    "train_dataset": "dataset1",
    "test_dataset": "scripts/test_ultrafeedback.parquet",
    "k_samples": 5,
    "timestamp": "2024-12-05_15-30-00"
  },
  "results": [
    {
      "prompt": "...",
      "responses": ["...", "...", "...", "...", "..."],
      "alpha": 0.45,
      "prompt_label": "creative",
      "self_bleu": 0.234,
      "distinct_2": 0.782,
      "semantic_div": 1.456
    }
  ],
  "summary": {
    "total_prompts": 2000,
    "avg_self_bleu": 0.250,
    "avg_distinct_2": 0.750,
    "avg_semantic_div": 1.400,
    "elapsed_time_seconds": 1234.5,
    "time_per_prompt": 0.617
  }
}
```

---

## 📈 Metrics Explained

### **Self-BLEU** (Lower = More Diverse)
- Measures lexical/n-gram overlap between responses
- Range: 0-1
- **Lower is better** for diversity

### **Distinct-2** (Higher = More Diverse)
- Ratio of unique bigrams to total bigrams
- Range: 0-1
- **Higher is better** for diversity

### **Semantic Diversity** (Higher = More Diverse)
- Average pairwise cosine distance between embeddings
- Range: 0-2
- **Higher is better** for diversity

---

## 🔄 Running Multiple Experiments

You can run inference on multiple trained models sequentially:

```bash
# Standard DPO on all datasets
for dataset in dataset1 dataset2; do
  python3 scripts/inference_standard_dpo.py \
    --model_path tinker://experiments/${dataset}_2024-12-05_14-23-47 \
    --test_dataset scripts/test_ultrafeedback.parquet
done

# 2-Adapter DPO
python3 scripts/inference_2adapter_dpo.py \
  --factual_adapter_path tinker://experiments/2adapter_dataset1_2024-12-05_14-30-00/factual \
  --creative_adapter_path tinker://experiments/2adapter_dataset1_2024-12-05_14-30-00/creative \
  --test_dataset scripts/test_ultrafeedback.parquet

# Diversity-Weighted DPO
python3 scripts/inference_diversity_weighted_dpo.py \
  --model_path tinker://experiments/diversity_weighted_dataset1_gamma0.05_2024-12-05_14-45-00 \
  --test_dataset scripts/test_ultrafeedback.parquet
```

---

## 🎯 Next Steps

After running inference, use the aggregation/visualization script to compare results across approaches.

(Aggregation script coming soon!)

