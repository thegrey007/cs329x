#!/bin/bash

# ============================================================================
# DPO Training Examples for CS329X Project
# ============================================================================

# Make sure you have:
# 1. TINKER_API_KEY set in your environment
# 2. WandB logged in (if using WandB): wandb login
# 3. Dataset files generated in scripts/

# ============================================================================
# BASIC EXAMPLES (Minimal Parameters)
# ============================================================================

# Train on Dataset 1 with Qwen 4B (all defaults)
python scripts/train_dpo.py \
    model_name=Qwen/Qwen3-4B-Instruct-2507 \
    dataset=1

# Train on Dataset 2 with Qwen 8B
python scripts/train_dpo.py \
    model_name=Qwen/Qwen3-8B \
    dataset=2

# ============================================================================
# WITH WANDB TRACKING
# ============================================================================

# Train with WandB logging (auto-generated run name)
python scripts/train_dpo.py \
    model_name=Qwen/Qwen3-4B-Instruct-2507 \
    dataset=1 \
    wandb_project=cs329x-dpo

# Train with custom WandB run name
python scripts/train_dpo.py \
    model_name=Qwen/Qwen3-4B-Instruct-2507 \
    dataset=1 \
    wandb_project=cs329x-dpo \
    wandb_name=my_experiment_v1

# ============================================================================
# HYPERPARAMETER TUNING
# ============================================================================

# Higher learning rate
python scripts/train_dpo.py \
    model_name=Qwen/Qwen3-4B-Instruct-2507 \
    dataset=1 \
    learning_rate=5e-5 \
    wandb_project=cs329x-dpo

# Different DPO beta (stronger preference signal)
python scripts/train_dpo.py \
    model_name=Qwen/Qwen3-4B-Instruct-2507 \
    dataset=1 \
    dpo_beta=0.2 \
    wandb_project=cs329x-dpo

# Smaller batch size (if memory issues)
python scripts/train_dpo.py \
    model_name=Qwen/Qwen3-4B-Instruct-2507 \
    dataset=1 \
    batch_size=128 \
    wandb_project=cs329x-dpo

# Higher LoRA rank (more capacity)
python scripts/train_dpo.py \
    model_name=Qwen/Qwen3-4B-Instruct-2507 \
    dataset=1 \
    lora_rank=64 \
    wandb_project=cs329x-dpo

# Multiple epochs
python scripts/train_dpo.py \
    model_name=Qwen/Qwen3-4B-Instruct-2507 \
    dataset=1 \
    num_epochs=3 \
    wandb_project=cs329x-dpo

# ============================================================================
# FULL EXPERIMENT SUITE (All 3 Datasets x 3 Models)
# ============================================================================

# Dataset 1 - Embedding Diversity
python scripts/train_dpo.py model_name=Qwen/Qwen3-4B-Instruct-2507 dataset=1 wandb_project=cs329x-dpo
python scripts/train_dpo.py model_name=Qwen/Qwen3-8B dataset=1 wandb_project=cs329x-dpo
python scripts/train_dpo.py model_name=Qwen/Qwen3-30B-A3B-Instruct-2507 dataset=1 wandb_project=cs329x-dpo

# Dataset 2 - Quality Based
python scripts/train_dpo.py model_name=Qwen/Qwen3-4B-Instruct-2507 dataset=2 wandb_project=cs329x-dpo
python scripts/train_dpo.py model_name=Qwen/Qwen3-8B dataset=2 wandb_project=cs329x-dpo
python scripts/train_dpo.py model_name=Qwen/Qwen3-30B-A3B-Instruct-2507 dataset=2 wandb_project=cs329x-dpo

# Dataset 3 - LLM Judge (when ready)
python scripts/train_dpo.py model_name=Qwen/Qwen3-4B-Instruct-2507 dataset=3 wandb_project=cs329x-dpo
python scripts/train_dpo.py model_name=Qwen/Qwen3-8B dataset=3 wandb_project=cs329x-dpo
python scripts/train_dpo.py model_name=Qwen/Qwen3-30B-A3B-Instruct-2507 dataset=3 wandb_project=cs329x-dpo

# ============================================================================
# CUSTOM LOG PATH
# ============================================================================

# Specify custom experiment folder
python scripts/train_dpo.py \
    model_name=Qwen/Qwen3-4B-Instruct-2507 \
    dataset=1 \
    log_path=experiments/my_custom_experiment \
    wandb_project=cs329x-dpo

# ============================================================================
# NOTES
# ============================================================================
# 
# Auto-generated folder structure:
#   experiments/dataset1_qwen34b_lr1e-05_beta0.1_2024-12-05/
#   ├── checkpoints.jsonl  ← Contains tinker:// weight paths
#   ├── metrics.jsonl      ← Training metrics
#   └── config.json        ← Training configuration
#
# To find your model weights after training:
#   cat experiments/dataset1_qwen34b_lr1e-05_beta0.1_2024-12-05/checkpoints.jsonl
#   Look for "sampler_path": "tinker://..." in the "final" checkpoint
#
# To use trained model for inference (in notebook/script):
#   from tinker_cookbook.completers import TinkerSampler
#   sampler = TinkerSampler(
#       model_name="Qwen/Qwen3-4B-Instruct-2507",
#       model_path="tinker://...",  # From checkpoints.jsonl
#       temperature=0.7,
#       max_tokens=2048
#   )
#

