# DPO Training Guide

## Overview

This guide covers training models using Direct Preference Optimization (DPO) on three different datasets with varying diversity criteria.

## Quick Start

### 1. Prerequisites

```bash
# Install dependencies (if not already installed)
pip install tinker datasets pandas

# Set your Tinker API key
export TINKER_API_KEY="your-api-key-here"

# (Optional) Login to WandB for tracking
wandb login
```

### 2. Ensure Datasets Are Generated

Make sure you have generated your DPO datasets:
- `scripts/dataset1_embedding_diversity.parquet` ✓
- `scripts/dataset2_quality_based.parquet` ✓
- `scripts/dataset3_llm_judge_diversity.parquet` (when ready)

### 3. Run Your First Training

```bash
# Train Qwen 4B on Dataset 1
python scripts/train_dpo.py \
    model_name=Qwen/Qwen3-4B-Instruct-2507 \
    dataset=1 \
    wandb_project=cs329x-dpo
```

## File Structure

```
scripts/
├── train_dpo.py              # Main training script
├── train_dpo_examples.sh     # Example commands
├── inference_example.py      # How to use trained models
├── dataset1_embedding_diversity.parquet
├── dataset2_quality_based.parquet
└── dataset3_llm_judge_diversity.parquet (when ready)

experiments/                   # Auto-generated during training
└── dataset1_qwen34b_lr1e-05_beta0.1_2024-12-05/
    ├── checkpoints.jsonl     # Contains tinker:// paths to weights
    ├── metrics.jsonl         # Training metrics
    └── config.json           # Training configuration
```

## Training Parameters

### Essential (Always Specify)
- `model_name`: Base model to fine-tune
  - `Qwen/Qwen3-4B-Instruct-2507`
  - `Qwen/Qwen3-8B`
  - `Qwen/Qwen3-30B-A3B-Instruct-2507`
- `dataset`: Dataset number (1, 2, or 3)

### Important (Good Defaults Provided)
- `learning_rate`: Default `1e-5`
- `dpo_beta`: Default `0.1` (controls preference strength)
- `batch_size`: Default `256`
- `num_epochs`: Default `1`
- `lora_rank`: Default `32`

### Optional (Auto-Generated If Not Specified)
- `log_path`: Auto-generated as `experiments/dataset{N}_{model}_{lr}_{beta}_{date}/`
- `wandb_project`: WandB project name (e.g., `cs329x-dpo`)
- `wandb_name`: Auto-generated if `wandb_project` is set

## Example Workflows

### Workflow 1: Single Experiment

```bash
# Train on Dataset 1 with Qwen 4B
python scripts/train_dpo.py \
    model_name=Qwen/Qwen3-4B-Instruct-2507 \
    dataset=1 \
    wandb_project=cs329x-dpo
```

**Output:**
- Experiment folder: `experiments/dataset1_qwen34b_lr1e-05_beta0.1_2024-12-05/`
- WandB run: `dataset1_qwen34b_lr1e-05_beta0.1_2024-12-05`
- Weights saved to Tinker cloud (paths in `checkpoints.jsonl`)

### Workflow 2: Hyperparameter Tuning

```bash
# Try different DPO betas
python scripts/train_dpo.py model_name=Qwen/Qwen3-4B-Instruct-2507 dataset=1 dpo_beta=0.1 wandb_project=cs329x-dpo
python scripts/train_dpo.py model_name=Qwen/Qwen3-4B-Instruct-2507 dataset=1 dpo_beta=0.2 wandb_project=cs329x-dpo
python scripts/train_dpo.py model_name=Qwen/Qwen3-4B-Instruct-2507 dataset=1 dpo_beta=0.05 wandb_project=cs329x-dpo
```

### Workflow 3: Full Experiment Suite (3 Datasets × 3 Models)

```bash
# Run all combinations
for dataset in 1 2 3; do
    for model in "Qwen/Qwen3-4B-Instruct-2507" "Qwen/Qwen3-8B" "Qwen/Qwen3-30B-A3B-Instruct-2507"; do
        python scripts/train_dpo.py \
            model_name=$model \
            dataset=$dataset \
            wandb_project=cs329x-dpo
    done
done
```

## Finding Your Trained Model Weights

After training, your model weights are saved to Tinker's cloud. To find them:

```bash
# Look at the checkpoints file
cat experiments/dataset1_qwen34b_lr1e-05_beta0.1_2024-12-05/checkpoints.jsonl
```

You'll see output like:
```json
{
  "name": "final",
  "epoch": 1,
  "batch": 150,
  "state_path": "tinker://abc123.../state",
  "sampler_path": "tinker://abc123.../sampler_weights"
}
```

The `sampler_path` is what you need for inference!

## Using Trained Models for Inference

### Method 1: Python Script

```python
from tinker_cookbook.completers import TinkerSampler

sampler = TinkerSampler(
    model_name="Qwen/Qwen3-4B-Instruct-2507",
    model_path="tinker://...",  # From checkpoints.jsonl
    temperature=0.7,
    max_tokens=2048,
)

response = sampler.complete("Your prompt here")
print(response)
```

### Method 2: Use the Example Script

```bash
# Edit inference_example.py to point to your experiment folder
python scripts/inference_example.py
```

## WandB Team Setup (For Group Projects)

### Option A: WandB Team (Recommended)
1. Create a WandB team (free tier supports this)
2. All team members join
3. Use team name: `--wandb_project=team-name/cs329x-dpo`

### Option B: Personal Account with Sharing
1. One person creates project: `--wandb_project=cs329x-dpo`
2. Share access with teammates

## Monitoring Training

### In WandB (if enabled)
- Loss curves (DPO loss, accuracy)
- Learning rate schedule
- Preference margins
- Chosen vs rejected rewards

### In Local Logs
```bash
# View metrics
cat experiments/dataset1_qwen34b_lr1e-05_beta0.1_2024-12-05/metrics.jsonl | tail -20
```

## Troubleshooting

### Dataset Not Found
```
FileNotFoundError: Dataset file not found: scripts/dataset1_embedding_diversity.parquet
```
**Solution:** Generate the dataset first using `pairs_creation.ipynb`

### Out of Memory
```bash
# Try smaller batch size
python scripts/train_dpo.py ... batch_size=128
```

### Can't Find Model Weights
```bash
# Check checkpoints file
cat experiments/your-experiment/checkpoints.jsonl
```

## Training Metrics to Watch

- **DPO Loss**: Should decrease (indicates learning)
- **Accuracy**: % of times chosen > rejected (should increase)
- **Margin**: Difference between chosen and rejected rewards (should increase)
- **Chosen Reward**: Should increase
- **Rejected Reward**: Should decrease

## Next Steps After Training

1. **Evaluate diversity**: Run inference on prompts with varying creativity scores
2. **Compare datasets**: Does Dataset 1 (embedding) vs Dataset 2 (quality) vs Dataset 3 (LLM judge) produce different behaviors?
3. **Scaling analysis**: Compare Qwen 4B vs 8B vs 30B - does model size affect diversity adaptation?

## Questions?

Check:
- `train_dpo_examples.sh` for more command examples
- `inference_example.py` for inference code
- Tinker docs: https://tinker-docs.thinkingmachines.ai/

