"""
Diversity-Weighted DPO Training Script for CS329X Project

This script trains models using a modified DPO loss that incorporates a diversity
modulation term conditioned on prompt creativity (alpha). The loss encourages semantic
diversity for creative prompts and discourages it for factual prompts.

Loss: L(x) = L_DPO(x) - gamma * t(x) * d_pair
where:
  - t(x) = 2*alpha - 1  (maps [0,1] to [-1,1])
  - d_pair = cosine_distance(embed(chosen), embed(rejected))
  - gamma controls the strength of the diversity term

Usage:
    python scripts/train_diversity_weighted_dpo.py \
        model_name=Qwen/Qwen3-4B-Instruct-2507 \
        dataset=1 \
        gamma=0.05 \
        learning_rate=1e-5 \
        wandb_project=cs329x-dpo
"""

# ============================================================================
# SECTION 1: IMPORTS & SETUP
# ============================================================================

import os
import time
from datetime import datetime
from typing import Any, cast

import chz
import datasets
import tinker
import torch
from dotenv import load_dotenv
from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.preference import train_dpo
from tinker_cookbook.preference.dpo_datasets import DPODatasetBuilderFromComparisons
from tinker_cookbook.preference.preference_datasets import ComparisonDatasetBuilder
from tinker_cookbook.preference.types import Comparison, LabeledComparison
from tinker_cookbook.supervised.common import datum_from_tokens_weights
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig
from tinker_cookbook.utils import ml_log
from tinker_cookbook.utils.misc_utils import timed

# Load environment variables from .env file
load_dotenv()


# ============================================================================
# SECTION 2: CUSTOM DATASET BUILDER (Load Parquet + Metadata)
# ============================================================================

@chz.chz
class DiversityWeightedParquetComparisonBuilder(ComparisonDatasetBuilder):
    """
    Loads DPO pairs from parquet files with diversity metadata.
    Expects columns: input, chosen, rejected, d_pair, creative_score (alpha)
    """

    dataset_number: int  # 1, 2, or 3
    test_size: int = 1024

    def get_train_and_test_datasets(self) -> tuple[datasets.Dataset, datasets.Dataset | None]:
        """Load the specified parquet file and create train/test split."""
        # Map dataset number to filename
        dataset_files = {
            1: "scripts/dataset1_embedding_diversity.parquet",
            2: "scripts/dataset2_quality_based.parquet",
            3: "scripts/dataset3_llm_judge_diversity.parquet",
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

        # Check for required columns
        required_cols = ["input", "chosen", "rejected", "d_pair", "creative_score"]
        missing_cols = [col for col in required_cols if col not in dataset.column_names]
        if missing_cols:
            raise ValueError(
                f"Dataset missing required columns: {missing_cols}\n"
                f"Please run the d_pair computation cell in pairs_creation.ipynb"
            )

        # Shuffle and split
        dataset = dataset.shuffle(seed=0)
        test_size = min(self.test_size, len(dataset) // 10)
        test_dataset = dataset.select(range(test_size))
        train_dataset = dataset.select(range(test_size, len(dataset)))

        print(f"\n{'='*70}")
        print(f"Loaded Dataset {self.dataset_number} (Diversity-Weighted)")
        print(f"{'='*70}")
        print(f"Train size: {len(train_dataset)}")
        print(f"Test size:  {len(test_dataset)}")
        print(f"{'='*70}\n")

        return train_dataset, test_dataset

    def example_to_labeled_comparison(self, example: dict) -> LabeledComparison | None:
        """Convert parquet example to LabeledComparison format with metadata."""
        try:
            prompt = example["input"]
            chosen = example["chosen"]
            rejected = example["rejected"]
            d_pair = float(example["d_pair"])
            alpha = float(example["creative_score"])

            # Create prompt conversation
            prompt_conversation = [{"role": "user", "content": prompt}]

            # Create comparison with metadata stored for later use
            comparison = Comparison(
                prompt_conversation=prompt_conversation,
                completion_A=[{"role": "assistant", "content": chosen}],
                completion_B=[{"role": "assistant", "content": rejected}],
            )

            # Store metadata as attributes (will be accessed in custom loss)
            labeled_comparison = LabeledComparison(comparison=comparison, label="A")
            # We'll pass metadata through the data pipeline via a custom attribute
            labeled_comparison._diversity_metadata = {  # type: ignore
                "d_pair": d_pair,
                "alpha": alpha,
            }

            return labeled_comparison
        except KeyError as e:
            print(f"Warning: Skipping malformed example, missing key: {e}")
            return None


# ============================================================================
# SECTION 3: USE STANDARD DPO DATASET BUILDER
# ============================================================================
# We use the standard DPODatasetBuilderFromComparisons
# Metadata is loaded separately in the training loop


# ============================================================================
# SECTION 4: CLI CONFIGURATION
# ============================================================================

@chz.chz
class CLIConfig:
    """Command-line configuration for Diversity-Weighted DPO training."""

    # ========== ESSENTIAL PARAMETERS (Must specify) ==========
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507"
    dataset: int = 1

    # ========== DIVERSITY MODULATION PARAMETER ==========
    gamma: float = 0.05  # Strength of diversity term (0 = standard DPO)

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
# SECTION 5: DIVERSITY-WEIGHTED DPO LOSS FUNCTION
# ============================================================================

def compute_diversity_weighted_dpo_loss(
    chosen_logprobs: list[torch.Tensor],
    rejected_logprobs: list[torch.Tensor],
    chosen_ref_logprobs: list[torch.Tensor],
    rejected_ref_logprobs: list[torch.Tensor],
    d_pairs: list[torch.Tensor],
    alphas: list[torch.Tensor],
    dpo_beta: float,
    gamma: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Compute diversity-weighted DPO loss and metrics.

    Args:
        chosen_logprobs: Log probabilities for chosen responses
        rejected_logprobs: Log probabilities for rejected responses
        chosen_ref_logprobs: Reference log probabilities for chosen responses
        rejected_ref_logprobs: Reference log probabilities for rejected responses
        d_pairs: Cosine distances between chosen and rejected embeddings
        alphas: Creativity scores (0=factual, 1=creative)
        dpo_beta: DPO beta parameter
        gamma: Diversity term weight

    Returns:
        Tuple of (loss tensor, metrics dictionary)
    """
    # Compute standard DPO loss components
    chosen_log_ratio = torch.stack(
        [lp - rlp for lp, rlp in zip(chosen_logprobs, chosen_ref_logprobs, strict=True)]
    )
    rejected_log_ratio = torch.stack(
        [lp - rlp for lp, rlp in zip(rejected_logprobs, rejected_ref_logprobs, strict=True)]
    )

    # Standard DPO loss
    dpo_losses = -torch.log(torch.sigmoid(dpo_beta * (chosen_log_ratio - rejected_log_ratio)))
    dpo_loss = dpo_losses.mean()

    # Compute diversity modulation term
    # t(x) = 2*alpha - 1  (maps [0,1] to [-1,1])
    d_pair_tensor = torch.stack(d_pairs).squeeze()
    alpha_tensor = torch.stack(alphas).squeeze()
    t_x = 2 * alpha_tensor - 1
    diversity_term = gamma * t_x * d_pair_tensor
    diversity_loss = diversity_term.mean()

    # Total loss: L = L_DPO - gamma * t(x) * d_pair
    total_loss = dpo_loss - diversity_loss

    # Compute metrics
    accuracy = (chosen_log_ratio > rejected_log_ratio).float().mean().item()
    chosen_rewards = dpo_beta * chosen_log_ratio
    rejected_rewards = dpo_beta * rejected_log_ratio
    margin = dpo_beta * (chosen_rewards - rejected_rewards).mean().item()

    metrics = {
        "dpo_loss": dpo_loss.item(),
        "diversity_term": diversity_loss.item(),
        "total_loss": total_loss.item(),
        "accuracy": accuracy,
        "margin": margin,
        "chosen_reward": chosen_rewards.mean().item(),
        "rejected_reward": rejected_rewards.mean().item(),
        "mean_d_pair": d_pair_tensor.mean().item(),
        "mean_alpha": alpha_tensor.mean().item(),
    }

    return total_loss, metrics


# ============================================================================
# SECTION 6: CUSTOM TRAINING LOOP (Using forward_backward_custom)
# ============================================================================

def do_update(
    epoch_idx: int,
    batch_idx: int,
    n_batches: int,
    total_steps: int,
    config: Any,  # train_dpo.Config
    training_client: tinker.TrainingClient,
    reference_client: tinker.SamplingClient,
    dataset: Any,
    ml_logger: ml_log.Logger,
    gamma: float,
    metadata_list: list[dict[str, float]],
):
    """Custom training update with diversity-weighted DPO loss."""
    start_time = time.time()
    step = epoch_idx * n_batches + batch_idx
    metrics: dict[str, Any] = {"epoch": epoch_idx}

    # Get batch
    with timed("data", metrics):
        data = dataset.get_batch(batch_idx)

    # Split into chosen and rejected
    chosen_data = [data[i] for i in range(0, len(data), 2)]
    rejected_data = [data[i] for i in range(1, len(data), 2)]
    
    # Get metadata for this batch (each example in the batch becomes 2 datums)
    batch_size = len(chosen_data)
    batch_start = batch_idx * batch_size
    batch_metadata = metadata_list[batch_start:batch_start + batch_size]

    # Learning rate schedule
    if config.lr_schedule == "linear":
        progress = step / total_steps
        learning_rate = config.learning_rate * (1 - progress)
    elif config.lr_schedule == "constant":
        learning_rate = config.learning_rate
    else:
        raise ValueError(f"Unknown lr_schedule: {config.lr_schedule}")

    adam_params = tinker.AdamParams(learning_rate=learning_rate)

    # Compute reference model log probabilities
    with timed("reference", metrics):
        full_sequences = []
        for datum in data:
            target_tokens = datum.loss_fn_inputs["target_tokens"].data
            if target_tokens:
                full_sequence = datum.model_input.append_int(int(target_tokens[-1]))
                full_sequences.append(full_sequence)
            else:
                full_sequences.append(datum.model_input)

        all_ref_logprobs = [
            reference_client.compute_logprobs(seq).result() for seq in full_sequences
        ]
        all_ref_logprob_seqs = [torch.tensor(logprobs[1:]) for logprobs in all_ref_logprobs]

        chosen_ref_logprob_seqs = [all_ref_logprob_seqs[i] for i in range(0, len(data), 2)]
        rejected_ref_logprob_seqs = [all_ref_logprob_seqs[i] for i in range(1, len(data), 2)]

    # Create diversity-weighted DPO loss function
    def diversity_weighted_dpo_loss_fn(
        data: list[tinker.Datum], logprobs_list: list[torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        # Split logprobs into chosen and rejected
        chosen_logprob_seqs = [logprobs_list[i] for i in range(0, len(data), 2)]
        rejected_logprob_seqs = [logprobs_list[i] for i in range(1, len(data), 2)]

        # Extract log probabilities
        chosen_logprobs = []
        chosen_ref_logprobs = []
        rejected_logprobs = []
        rejected_ref_logprobs = []
        d_pairs = []
        alphas = []

        for i in range(len(chosen_data)):
            # Chosen
            chosen_logprob_seq = chosen_logprob_seqs[i]
            chosen_ref_logprob_seq = chosen_ref_logprob_seqs[i]
            chosen_weights = torch.tensor(chosen_data[i].loss_fn_inputs["weights"].data)
            chosen_logprob = torch.dot(chosen_logprob_seq.float(), chosen_weights.float())
            chosen_ref_logprob = torch.dot(chosen_ref_logprob_seq.float(), chosen_weights.float())
            chosen_logprobs.append(chosen_logprob)
            chosen_ref_logprobs.append(chosen_ref_logprob)

            # Rejected
            rejected_logprob_seq = rejected_logprob_seqs[i]
            rejected_ref_logprob_seq = rejected_ref_logprob_seqs[i]
            rejected_weights = torch.tensor(rejected_data[i].loss_fn_inputs["weights"].data)
            rejected_logprob = torch.dot(rejected_logprob_seq.float(), rejected_weights.float())
            rejected_ref_logprob = torch.dot(
                rejected_ref_logprob_seq.float(), rejected_weights.float()
            )
            rejected_logprobs.append(rejected_logprob)
            rejected_ref_logprobs.append(rejected_ref_logprob)

            # Extract diversity metadata from batch_metadata
            metadata = batch_metadata[i]
            d_pair = torch.tensor([metadata["d_pair"]], dtype=torch.float32)
            alpha = torch.tensor([metadata["alpha"]], dtype=torch.float32)
            d_pairs.append(d_pair)
            alphas.append(alpha)

        # Compute diversity-weighted DPO loss
        return compute_diversity_weighted_dpo_loss(
            chosen_logprobs=chosen_logprobs,
            rejected_logprobs=rejected_logprobs,
            chosen_ref_logprobs=chosen_ref_logprobs,
            rejected_ref_logprobs=rejected_ref_logprobs,
            d_pairs=d_pairs,
            alphas=alphas,
            dpo_beta=config.dpo_beta,
            gamma=gamma,
        )

    with timed("step", metrics):
        # Do forward-backward with custom diversity-weighted DPO loss
        backward_result = training_client.forward_backward_custom(data, diversity_weighted_dpo_loss_fn).result()
        dpo_metrics = backward_result.metrics

        # Optimizer step
        training_client.optim_step(adam_params).result()

    # Prepare metrics
    metrics.update(
        num_pairs=len(chosen_data),
        num_tokens=sum(datum.model_input.length for datum in data),
        learning_rate=learning_rate,
        progress=step / total_steps,
        **dpo_metrics,
    )

    # Log metrics
    metrics["time/total"] = time.time() - start_time
    ml_logger.log_metrics(metrics=metrics, step=step)


# ============================================================================
# SECTION 7: CUSTOM TRAINING MAIN LOOP
# ============================================================================

def custom_training_main(config: Any, gamma: float):
    """Custom training loop with diversity-weighted DPO loss."""
    from tinker_cookbook import checkpoint_utils
    from tinker_cookbook.tokenizer_utils import get_tokenizer
    import logging
    import pandas as pd

    logger = logging.getLogger(__name__)

    # Resume from checkpoint if exists
    resume_info = checkpoint_utils.get_last_checkpoint(config.log_path)
    if resume_info:
        start_epoch = resume_info["epoch"]
        start_batch = resume_info["batch"]
    else:
        start_epoch = 0
        start_batch = 0

    # Setup
    ml_logger = ml_log.setup_logging(
        log_dir=config.log_path,
        wandb_project=config.wandb_project,
        wandb_name=config.wandb_name,
        config=config,
        do_configure_logging_module=True,
    )
    training_client, reference_client = train_dpo.create_dpo_clients(config, resume_info)
    tokenizer = get_tokenizer(config.model_name)

    # Training setup
    dataset, maybe_test_dataset = config.dataset_builder()
    n_batches = len(dataset)
    total_steps = n_batches * config.num_epochs
    
    # Load and prepare metadata - get the shuffled HF dataset used to build the datums
    train_hf_dataset, _ = config.dataset_builder.comparison_builder.get_train_and_test_datasets()
    
    # Extract metadata in the same order as the shuffled dataset
    metadata_list = []
    for idx in range(len(train_hf_dataset)):
        example = train_hf_dataset[idx]
        metadata_list.append({
            "d_pair": float(example["d_pair"]),
            "alpha": float(example["creative_score"]),
        })
    
    logger.info(f"Loaded metadata for {len(metadata_list)} examples")

    logger.info(
        f"Training for {n_batches} batches x {config.num_epochs} epochs = {n_batches * config.num_epochs} steps"
    )
    logger.info(f"Diversity weight (gamma): {gamma}")

    # Training loop
    for epoch_idx in range(start_epoch, config.num_epochs):
        # Shuffle the dataset
        logger.info(msg=f"Starting epoch {epoch_idx}")
        dataset.set_epoch(seed=epoch_idx)

        for batch_idx in range(start_batch if epoch_idx == start_epoch else 0, n_batches):
            do_update(
                epoch_idx=epoch_idx,
                batch_idx=batch_idx,
                n_batches=n_batches,
                total_steps=total_steps,
                config=config,
                training_client=training_client,
                reference_client=reference_client,
                dataset=dataset,
                ml_logger=ml_logger,
                gamma=gamma,
                metadata_list=metadata_list,
            )

            # Save checkpoints periodically
            if (batch_idx + 1) % config.save_every == 0:
                checkpoint_utils.save_checkpoint(
                    training_client=training_client,
                    name=f"epoch{epoch_idx}_batch{batch_idx}",
                    log_path=config.log_path,
                    kind="both",
                    loop_state={"epoch": epoch_idx, "batch": batch_idx + 1},
                )

    # Save final checkpoint if training actually happened
    if start_epoch < config.num_epochs:
        checkpoint_utils.save_checkpoint(
            training_client=training_client,
            name="final",
            log_path=config.log_path,
            kind="both",
            loop_state={"epoch": config.num_epochs, "batch": n_batches},
        )
    else:
        logger.info("Training was already complete; nothing to do")

    # Cleanup
    ml_logger.close()
    logger.info("Diversity-weighted DPO training completed successfully")


# ============================================================================
# SECTION 8: CLI ENTRY POINT
# ============================================================================

def cli_main(cli_config: CLIConfig):
    """Main training function - builds config and calls custom DPO trainer."""

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
        log_path = f"experiments/divweight_dataset{cli_config.dataset}_gamma{cli_config.gamma}_{timestamp}"

    # Auto-generate wandb_name if wandb_project is specified but wandb_name is not
    if cli_config.wandb_project is not None and cli_config.wandb_name is None:
        wandb_name = f"divweight_dataset{cli_config.dataset}_gamma{cli_config.gamma}_{timestamp}"
    else:
        wandb_name = cli_config.wandb_name

    # Check if log directory exists
    cli_utils.check_log_dir(log_path, behavior_if_exists=cli_config.behavior_if_log_dir_exists)

    # Create dataset builder (using standard DPO builder)
    dataset_builder = DPODatasetBuilderFromComparisons(
        common_config=ChatDatasetBuilderCommonConfig(
            model_name_for_tokenizer=cli_config.model_name,
            renderer_name=renderer_name,
            max_length=cli_config.max_length,
            batch_size=cli_config.batch_size,
        ),
        comparison_builder=DiversityWeightedParquetComparisonBuilder(dataset_number=cli_config.dataset),
    )

    # Build full DPO training config (we'll override the training loop)
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
    print("DIVERSITY-WEIGHTED DPO TRAINING")
    print("=" * 70)
    print(f"Model:           {cli_config.model_name}")
    print(f"Dataset:         {cli_config.dataset}")
    print(f"Gamma:           {cli_config.gamma} {'(standard DPO)' if cli_config.gamma == 0 else '(diversity-weighted)'}")
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

    # Run custom training loop with diversity-weighted loss
    custom_training_main(config, cli_config.gamma)


# ============================================================================
# SECTION 9: ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    chz.nested_entrypoint(cli_main)

