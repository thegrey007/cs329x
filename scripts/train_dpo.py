"""
DPO Training Script for CS329X Project

This script trains models using Direct Preference Optimization (DPO) on three different
datasets with configurable hyperparameters.

Usage:
    python scripts/train_dpo.py \
        model_name=Qwen/Qwen3-4B-Instruct-2507 \
        dataset=1 \
        learning_rate=1e-5 \
        dpo_beta=0.1 \
        wandb_project=cs329x-dpo
"""

# ============================================================================
# SECTION 1: IMPORTS & SETUP
# ============================================================================

import os
from datetime import datetime
from typing import cast

import chz
import datasets
from dotenv import load_dotenv
from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.preference import train_dpo
from tinker_cookbook.preference.dpo_datasets import DPODatasetBuilderFromComparisons
from tinker_cookbook.preference.preference_datasets import ComparisonDatasetBuilder
from tinker_cookbook.preference.types import Comparison, LabeledComparison
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig

# Load environment variables from .env file
load_dotenv()


# ============================================================================
# SECTION 2: CUSTOM DATASET BUILDER (Load Your Parquet Files)
# ============================================================================

@chz.chz
class CustomParquetComparisonBuilder(ComparisonDatasetBuilder):
    """
    Loads DPO pairs from parquet files (dataset1, dataset2, or dataset3).
    Expects columns: prompt, chosen, rejected
    """

    dataset_number: int  # 1, 2, or 3
    test_size: int = 1024

    def get_train_and_test_datasets(self) -> tuple[datasets.Dataset, datasets.Dataset | None]:
        """Load the specified parquet file and create train/test split."""
        # Map dataset number to filename
        dataset_files = {
            1: "scripts/dataset1_embedding_diversity.parquet",
            2: "scripts/dataset2_quality_based.parquet",
            3: "scripts/dataset3_llm_judge_diversity.parquet",  # When ready
        }

        if self.dataset_number not in dataset_files:
            raise ValueError(f"Invalid dataset number: {self.dataset_number}. Must be 1, 2, or 3.")

        parquet_file = dataset_files[self.dataset_number]

        # Check if file exists
        if not os.path.exists(parquet_file):
            raise FileNotFoundError(
                f"Dataset file not found: {parquet_file}\n"
                f"Please ensure dataset {self.dataset_number} has been generated."
            )

        # Load parquet file
        dataset = datasets.load_dataset("parquet", data_files=parquet_file)["train"]
        dataset = cast(datasets.Dataset, dataset)

        # Shuffle and split
        dataset = dataset.shuffle(seed=0)
        test_size = min(self.test_size, len(dataset) // 10)
        test_dataset = dataset.select(range(test_size))
        train_dataset = dataset.select(range(test_size, len(dataset)))

        print(f"\n{'='*70}")
        print(f"Loaded Dataset {self.dataset_number}")
        print(f"{'='*70}")
        print(f"Train size: {len(train_dataset)}")
        print(f"Test size:  {len(test_dataset)}")
        print(f"{'='*70}\n")

        return train_dataset, test_dataset

    def example_to_labeled_comparison(self, example: dict) -> LabeledComparison | None:
        """Convert parquet example to LabeledComparison format."""
        try:
            prompt = example["input"]
            chosen = example["chosen"]
            rejected = example["rejected"]

            # Create prompt conversation
            prompt_conversation = [{"role": "user", "content": prompt}]

            # Create comparison
            comparison = Comparison(
                prompt_conversation=prompt_conversation,
                completion_A=[{"role": "assistant", "content": chosen}],
                completion_B=[{"role": "assistant", "content": rejected}],
            )

            return LabeledComparison(comparison=comparison, label="A")
        except KeyError as e:
            print(f"Warning: Skipping malformed example, missing key: {e}")
            return None


# ============================================================================
# SECTION 3: CLI CONFIGURATION
# ============================================================================

@chz.chz
class CLIConfig:
    """Command-line configuration for DPO training."""

    # ========== ESSENTIAL PARAMETERS (Must specify) ==========
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507"
    dataset: int = 1

    # ========== IMPORTANT HYPERPARAMETERS (Good defaults provided) ==========
    learning_rate: float = 1e-5
    dpo_beta: float = 0.1
    batch_size: int = 256
    num_epochs: int = 1
    lora_rank: int = 32
    max_length: int | None = 8192

    # ========== OPTIONAL PARAMETERS ==========
    lr_schedule: str = "linear"
    renderer_name: str | None = None
    load_checkpoint_path: str | None = None

    # ========== LOGGING & TRACKING (Optional, auto-generated if not specified) ==========
    log_path: str | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None

    # ========== INFRASTRUCTURE ==========
    base_url: str | None = None
    save_every: int = 20
    eval_every: int = 10

    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"


# ============================================================================
# SECTION 4: MAIN TRAINING FUNCTION
# ============================================================================

def cli_main(cli_config: CLIConfig):
    """Main training function - builds config and calls DPO trainer."""

    # Auto-detect renderer if not specified
    renderer_name = cli_config.renderer_name or model_info.get_recommended_renderer_name(
        cli_config.model_name
    )

    # Generate timestamp for unique run identification
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Auto-generate log_path if not specified
    if cli_config.log_path is not None:
        log_path = cli_config.log_path
    else:
        log_path = f"experiments/dataset{cli_config.dataset}_{timestamp}"

    # Auto-generate wandb_name if wandb_project is specified but wandb_name is not
    if cli_config.wandb_project is not None and cli_config.wandb_name is None:
        wandb_name = f"dataset{cli_config.dataset}_{timestamp}"
    else:
        wandb_name = cli_config.wandb_name

    # Check if log directory exists
    cli_utils.check_log_dir(log_path, behavior_if_exists=cli_config.behavior_if_log_dir_exists)

    # Create dataset builder
    dataset_builder = DPODatasetBuilderFromComparisons(
        common_config=ChatDatasetBuilderCommonConfig(
            model_name_for_tokenizer=cli_config.model_name,
            renderer_name=renderer_name,
            max_length=cli_config.max_length,
            batch_size=cli_config.batch_size,
        ),
        comparison_builder=CustomParquetComparisonBuilder(dataset_number=cli_config.dataset),
    )

    # Build full DPO training config
    config = train_dpo.Config(
        log_path=log_path,
        model_name=cli_config.model_name,
        dataset_builder=dataset_builder,
        load_checkpoint_path=cli_config.load_checkpoint_path,
        evaluator_builders=[],
        learning_rate=cli_config.learning_rate,
        lr_schedule=cli_config.lr_schedule,
        dpo_beta=cli_config.dpo_beta,
        num_epochs=cli_config.num_epochs,
        lora_rank=cli_config.lora_rank,
        base_url=cli_config.base_url,
        wandb_project=cli_config.wandb_project,
        wandb_name=wandb_name,
        save_every=cli_config.save_every,
        eval_every=cli_config.eval_every,
    )

    # Print configuration summary
    print("\n" + "=" * 70)
    print("DPO TRAINING CONFIGURATION")
    print("=" * 70)
    print(f"Model:           {cli_config.model_name}")
    print(f"Dataset:         {cli_config.dataset}")
    print(f"Learning Rate:   {cli_config.learning_rate}")
    print(f"DPO Beta:        {cli_config.dpo_beta}")
    print(f"Batch Size:      {cli_config.batch_size}")
    print(f"Num Epochs:      {cli_config.num_epochs}")
    print(f"LoRA Rank:       {cli_config.lora_rank}")
    print(f"Log Path:        {log_path}")
    if cli_config.wandb_project:
        print(f"WandB Project:   {cli_config.wandb_project}")
        print(f"WandB Run Name:  {wandb_name}")
    else:
        print(f"WandB:           Disabled")
    print("=" * 70 + "\n")

    # Start training
    train_dpo.main(config)


# ============================================================================
# SECTION 5: ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    chz.nested_entrypoint(cli_main)

