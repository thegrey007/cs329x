# Training & Inference Guide

Direct guide for training DPO models and running inference.

---

## Training Folder Structure

All trained models saved under `experiments/`:

```
experiments/
├── dataset1_2024-12-05_14-23-47/                          # Standard DPO
│   └── checkpoints.jsonl
├── 2adapter_dataset1_2024-12-05_14-30-00/                 # 2-Adapter DPO
│   ├── factual/
│   │   └── checkpoints.jsonl
│   └── creative/
│       └── checkpoints.jsonl
└── diversity_weighted_dataset1_gamma0.05_2024-12-05_14-45-00/  # Diversity-Weighted DPO
    └── checkpoints.jsonl
```

**Naming Convention:**
- Standard: `dataset{N}_{timestamp}`
- 2-Adapter: `2adapter_dataset{N}_{timestamp}`
- Diversity-Weighted: `diversity_weighted_dataset{N}_gamma{value}_{timestamp}`

---

## Inference Results Structure

All inference results saved under `inference_results/`:

```
inference_results/
├── standard_dpo/
│   └── dataset1_2024-12-05_14-23-47.json                          # Matches experiment folder name
├── 2adapter_dpo/
│   └── 2adapter_dataset1_2024-12-05_14-30-00.json
└── diversity_weighted_dpo/
    └── diversity_weighted_dataset1_gamma0.05_2024-12-05_14-45-00.json
```

**Mapping:**
- Inference result filename = experiment folder name
- Easy 1-1 mapping between training run and inference results

---

## Training Commands

**Note:** Training scripts use `key=value` syntax (no dashes).

### Standard DPO
```bash
python3 scripts/train_dpo.py model_name=Qwen/Qwen3-8B dataset=1 learning_rate=1e-5 num_epochs=1 batch_size=256 dpo_beta=0.1
```

**Output:** `experiments/dataset1_{timestamp}/`

### 2-Adapter DPO
```bash
python3 scripts/train_2adapter_dpo.py model_name=Qwen/Qwen3-8B dataset=1 learning_rate=1e-5 num_epochs=1 batch_size=256 dpo_beta=0.1
```

**Output:** `experiments/2adapter_dataset1_{timestamp}/factual/` and `.../creative/`

### Diversity-Weighted DPO
```bash
python3 scripts/train_diversity_weighted_dpo.py model_name=Qwen/Qwen3-8B dataset=1 gamma=0.05 learning_rate=1e-5 num_epochs=1 batch_size=256 dpo_beta=0.1
```

**Output:** `experiments/diversity_weighted_dataset1_gamma0.05_{timestamp}/`

---

## Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--dataset` | (required) | Dataset number: 1, 2, or 3 |
| `--learning_rate` | `1e-5` | Learning rate |
| `--num_epochs` | `1` | Number of training epochs |
| `--batch_size` | `256` | Training batch size |
| `--dpo_beta` | `0.1` | DPO beta parameter |
| `--lora_rank` | `32` | LoRA rank |
| `--gamma` | `0.05` | Diversity weight (diversity-weighted only) |

---

## Inference Commands

### Standard DPO
```bash
python3 scripts/inference_standard_dpo.py \
  --model_path tinker://experiments/dataset1_2024-12-05_14-23-47 \
  --test_dataset scripts/test_ultrafeedback.parquet \
  --k_samples 5
```

**Output:** `inference_results/standard_dpo/dataset1_2024-12-05_14-23-47.json`

### 2-Adapter DPO
```bash
python3 scripts/inference_2adapter_dpo.py \
  --factual_adapter_path tinker://experiments/2adapter_dataset1_2024-12-05_14-30-00/factual \
  --creative_adapter_path tinker://experiments/2adapter_dataset1_2024-12-05_14-30-00/creative \
  --test_dataset scripts/test_ultrafeedback.parquet \
  --k_samples 5
```

**Output:** `inference_results/2adapter_dpo/2adapter_dataset1_2024-12-05_14-30-00.json`

### Diversity-Weighted DPO
```bash
python3 scripts/inference_diversity_weighted_dpo.py \
  --model_path tinker://experiments/diversity_weighted_dataset1_gamma0.05_2024-12-05_14-45-00 \
  --test_dataset scripts/test_ultrafeedback.parquet \
  --k_samples 5
```

**Output:** `inference_results/diversity_weighted_dpo/diversity_weighted_dataset1_gamma0.05_2024-12-05_14-45-00.json`

---

## Inference Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model_path` | (required) | Trained model path (tinker:// or experiments/) |
| `--factual_adapter_path` | (required) | Factual adapter path (2-adapter only) |
| `--creative_adapter_path` | (required) | Creative adapter path (2-adapter only) |
| `--test_dataset` | `scripts/test_ultrafeedback.parquet` | Test dataset path |
| `--k_samples` | `5` | Number of responses per prompt |
| `--max_concurrent` | `10` | Max concurrent prompts |
| `--temperature` | `1.0` | Sampling temperature |

---

## Inference Output Format

Each JSON file contains:

```json
{
  "metadata": {
    "approach": "standard_dpo",
    "experiment_folder": "dataset1_2024-12-05_14-23-47",
    "model_path": "tinker://experiments/...",
    "train_dataset": "dataset1",
    "test_dataset": "scripts/test_ultrafeedback.parquet"
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
    "avg_semantic_div": 1.400
  }
}
```

---

## Metrics

| Metric | Range | Better Value | Description |
|--------|-------|--------------|-------------|
| **Self-BLEU** | 0-1 | Lower | Lexical overlap (n-gram similarity) |
| **Distinct-2** | 0-1 | Higher | Unique bigrams ratio |
| **Semantic Div** | 0-2 | Higher | Embedding cosine distance |

---

## Complete Workflow Example

```bash
# 1. Train standard DPO on dataset 1
python3 scripts/train_dpo.py model_name=Qwen/Qwen3-8B dataset=1

# Output: experiments/dataset1_2024-12-05_14-23-47/

# 2. Run inference on trained model
python3 scripts/inference_standard_dpo.py \
  --model_path tinker://experiments/dataset1_2024-12-05_14-23-47 \
  --test_dataset scripts/test_ultrafeedback.parquet

# Output: inference_results/standard_dpo/dataset1_2024-12-05_14-23-47.json
```

**Mapping:**
- Training folder: `experiments/dataset1_2024-12-05_14-23-47/`
- Inference result: `inference_results/standard_dpo/dataset1_2024-12-05_14-23-47.json`
- Match: Same experiment name in both paths

