# cs329x

Data: 

A) Ultra-feedback sampled preference pairs (Chosen =  highest overall_score, rejected = lowest overall_score)
B) Ultra-feedback sampled preference pairs (Chosen = highest avg. embedding distance above threshold, lowest avg. embedding distance below threshold)
C) Ultra-feedback sampled preference pairs (Chosen = highest LLM-as-judge diversity above threshold, lowest LLM-as-judge diversity above threshold)          —— IF TIME
D) Allen-AI Tulu 2.5 sampled preference pairs (Pre-constructed)
	- Allen-AI dataset roughly similar to Ultra-feedback, but can also push this as a Out-of-domain evaluation, good as well

Models:

1) Standard DPO
2) DPO with Diversity Aware Loss Term
3) DPO with Two-Adapter approach 

Training (Start with 7/8B instruct tuned models, can try another size if time):

- Will use ultra feedback
- Baselines:1) trained on A) : standard DPO baseline ,1) trained on B) : evaluate effectiveness of custom pair creation
- Approaches:2), 3) each trained on A), B) : motivates need for custom pairs, effectiveness of approach on custom pairs
- Evaluations: For each model:Self-BLEU (Zhu et al 2018), Distinct-N (Li et al 2016), Semantic Diversity (Li et Al 2024) on Allen-AI Tulu 2.5Novelty Bench on their prompts, or Allen-AI prompts



---

## Quick Start

```bash
# 1. Setup environment
conda create -n cs329x python=3.10 -y
conda activate cs329x
pip install -r requirements.txt

# 2. Configure API keys
cp env.template .env
# Edit .env with your TINKER_API_KEY and WANDB_API_KEY

# 3. Download data files (from shared Drive)
# Place in: data/final_lamini_informational_creative_dataset_uf/
#           scripts/filtered_rows.pkl

# 4. Train a model
python scripts/train_dpo.py \
    model_name=Qwen/Qwen3-4B-Instruct-2507 \
    dataset=1 \
    wandb_project=cs329x-dpo
```

---

## Setup Instructions

### 1. Create Conda Environment

```bash
conda create -n cs329x python=3.10 -y
conda activate cs329x
pip install -r requirements.txt
```

### 2. Setup API Keys

```bash
# Copy template to .env
cp env.template .env

# Edit .env and add your API keys
nano .env
```

**Required API keys:**
- `TINKER_API_KEY`: Get from [Tinker Console](https://tinker-console.thinkingmachines.ai)
- `WANDB_API_KEY`: Get from [WandB Settings](https://wandb.ai/settings) (or run `wandb login`)

### 3. Download Required Data Files

Download from shared Drive and place in:
- `data/final_lamini_informational_creative_dataset_uf/*` (LaMini creativity scores)
- `scripts/filtered_rows.pkl` (filtered prompts for pair generation)

**Note:** These files are in `.gitignore` and won't be committed.

### 4. Verify Setup

```bash
# Check environment
conda info --envs

# Verify API keys
python -c "from dotenv import load_dotenv; import os; load_dotenv(); print('✓ Setup complete' if os.getenv('TINKER_API_KEY') else '✗ Missing API keys')"
```

---

## Training

### Basic Command

```bash
python scripts/train_dpo.py \
    model_name=Qwen/Qwen3-4B-Instruct-2507 \
    dataset=1 \
    wandb_project=cs329x-dpo
```

### Available Models
```bash
# 4B model
model_name=Qwen/Qwen3-4B-Instruct-2507

# 8B model
model_name=Qwen/Qwen3-8B

# 30B model
model_name=Qwen/Qwen3-30B-A3B-Instruct-2507
```

### Available Datasets
```bash
dataset=1  # Embedding-based diversity pairs
dataset=2  # Quality-based pairs (baseline)
dataset=3  # LLM-as-judge diversity pairs
```

**For comprehensive training documentation, parameters, and examples:**
- See `DPO_TRAINING_README.md` for detailed training guide
- See `scripts/train_dpo_examples.sh` for example commands
- See `scripts/inference_example.py` for using trained models

---

## Project Structure

```
cs329x/
├── data/
│   └── final_lamini_informational_creative_dataset_uf/  # LaMini creativity scores (from Drive)
├── scripts/
│   ├── pairs_creation.ipynb                 # Generate DPO datasets
│   ├── dataprocessing_uf.ipynb              # Preprocess UltraFeedback data
│   ├── train_dpo.py                         # Main training script ⭐
│   ├── train_dpo_examples.sh                # Example training commands
│   ├── inference_example.py                 # Inference guide
│   ├── filtered_rows.pkl                    # Filtered prompts (from Drive)
│   ├── dataset1_embedding_diversity.parquet
│   ├── dataset2_quality_based.parquet
│   └── dataset3_llm_judge_diversity.parquet
├── experiments/                              # Training outputs (auto-generated)
│   └── dataset{N}_{model}_{params}_{date}/
│       └── checkpoints.jsonl                # Tinker model weight paths
├── env.template                             # API keys template
├── requirements.txt                         # Python dependencies
├── README.md                                # This file
└── DPO_TRAINING_README.md                   # Comprehensive training guide
```

---

## Documentation

- **Training Guide**: `DPO_TRAINING_README.md` - Comprehensive DPO training documentation
- **Training Examples**: `scripts/train_dpo_examples.sh` - Example commands for all scenarios
- **Inference Guide**: `scripts/inference_example.py` - How to use trained models
- **Tinker Docs**: https://tinker-docs.thinkingmachines.ai/

---

## Training Outputs

After training completes, model weights are saved to Tinker's cloud. Find the paths in:

```bash
cat experiments/dataset1_qwen34b_lr1e-05_beta0.1_2024-12-05/checkpoints.jsonl
```

The `sampler_path` in the "final" checkpoint is used for inference.

---

## Required Environment Variables

| Variable | Required | Purpose |
|----------|----------|---------|
| `TINKER_API_KEY` | ✅ Yes | DPO training on Tinker platform |
| `WANDB_API_KEY` | ✅ Yes | Experiment tracking (or use `wandb login`) |
