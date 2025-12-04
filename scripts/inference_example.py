"""
Example: Using DPO-trained models for inference

This script shows how to load your trained model weights and generate responses.

Usage:
    python scripts/inference_example.py
"""

import json
from tinker_cookbook.completers import TinkerSampler


def load_model_path_from_checkpoint(experiment_folder: str, checkpoint_name: str = "final"):
    """
    Load the Tinker model path from checkpoints.jsonl
    
    Args:
        experiment_folder: Path to experiment folder (e.g., "experiments/dataset1_qwen34b_...")
        checkpoint_name: Name of checkpoint to load (default: "final")
    
    Returns:
        Tuple of (sampler_path, state_path) from the checkpoint
    """
    checkpoint_file = f"{experiment_folder}/checkpoints.jsonl"
    
    with open(checkpoint_file, 'r') as f:
        checkpoints = [json.loads(line) for line in f]
    
    # Find the specified checkpoint
    for checkpoint in reversed(checkpoints):  # Start from end (most recent)
        if checkpoint.get("name") == checkpoint_name:
            return checkpoint.get("sampler_path"), checkpoint.get("state_path")
    
    raise ValueError(f"Checkpoint '{checkpoint_name}' not found in {checkpoint_file}")


def main():
    # ========================================================================
    # STEP 1: Load model path from your experiment
    # ========================================================================
    
    # Example: Load the final checkpoint from a specific experiment
    experiment_folder = "experiments/dataset1_qwen34b_lr1e-05_beta0.1_2024-12-05"
    
    try:
        sampler_path, state_path = load_model_path_from_checkpoint(experiment_folder)
        print(f"✓ Loaded model from: {experiment_folder}")
        print(f"  Sampler path: {sampler_path}")
    except FileNotFoundError:
        print(f"✗ Experiment folder not found: {experiment_folder}")
        print("\nPlease update 'experiment_folder' to point to your actual experiment.")
        print("Available experiments:")
        import os
        if os.path.exists("experiments"):
            for folder in os.listdir("experiments"):
                if os.path.isdir(f"experiments/{folder}"):
                    print(f"  - experiments/{folder}")
        return
    
    # ========================================================================
    # STEP 2: Create sampler with your trained model
    # ========================================================================
    
    base_model_name = "Qwen/Qwen3-4B-Instruct-2507"  # Match what you trained
    
    sampler = TinkerSampler(
        model_name=base_model_name,
        model_path=sampler_path,  # Your trained weights
        temperature=0.7,
        max_tokens=2048,
    )
    
    print("\n✓ Created sampler with trained model")
    
    # ========================================================================
    # STEP 3: Generate responses
    # ========================================================================
    
    test_prompts = [
        "What is the capital of France?",  # Factual (low creativity)
        "Write a creative story about a dragon who loves mathematics.",  # Creative (high creativity)
        "Explain quantum computing in simple terms.",  # Factual
        "Imagine a world where colors have sounds. Describe it.",  # Creative
    ]
    
    print("\n" + "="*70)
    print("GENERATING RESPONSES")
    print("="*70)
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n[{i}/{len(test_prompts)}] Prompt: {prompt}")
        print("-" * 70)
        
        # Generate response
        response = sampler.complete(prompt)
        
        print(f"Response: {response}")
        print("-" * 70)
    
    print("\n✓ Inference complete!")


if __name__ == "__main__":
    main()

