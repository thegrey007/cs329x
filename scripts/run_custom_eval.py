#!/usr/bin/env python3
"""
NoveltyBench Evaluation with Custom Models and Prompts

This script makes it easy to run NoveltyBench with your own model and prompts.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"❌ Command failed with exit code {result.returncode}")
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(
        description="Run NoveltyBench with custom model and prompts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use custom model with built-in prompts
  python scripts/run_custom_eval.py --model gpt2 --prompts curated
  
  # Use custom model with custom prompts
  python scripts/run_custom_eval.py --model meta-llama/Llama-2-7b-chat-hf --prompts my_prompts.jsonl
  
  # Full custom configuration
  python scripts/run_custom_eval.py --model ./my_model --prompts my_prompts.jsonl --num-generations 10 --temperature 0.9
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--model",
        required=True,
        help="Model to evaluate (HuggingFace ID or local path)"
    )
    parser.add_argument(
        "--prompts",
        required=True,
        help="Prompts source: 'curated', 'wildchat', or path to custom JSONL file"
    )
    
    # Optional arguments
    parser.add_argument(
        "--eval-dir",
        help="Directory to save results (auto-generated if not specified)"
    )
    parser.add_argument(
        "--num-generations",
        type=int,
        default=5,
        help="Number of diverse responses per prompt (default: 5)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature (default: 0.8)"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Nucleus sampling parameter (default: 0.9)"
    )
    parser.add_argument(
        "--mode",
        default="transformers",
        choices=["transformers", "openai", "vllm", "together", "cohere", "gemini", "anthropic"],
        help="Inference mode (default: transformers for local HF models)"
    )
    parser.add_argument(
        "--sampling",
        default="regenerate",
        choices=["regenerate", "in-context", "paraphrase", "system-prompt"],
        help="Sampling strategy (default: regenerate)"
    )
    
    args = parser.parse_args()
    
    # Determine eval directory
    if args.eval_dir:
        eval_dir = args.eval_dir
    else:
        model_name = Path(args.model).name
        prompts_name = Path(args.prompts).stem if args.prompts not in ["curated", "wildchat"] else args.prompts
        eval_dir = f"noveltybench/results/{prompts_name}/{model_name}"
    
    print("="*60)
    print("NoveltyBench Custom Evaluation")
    print("="*60)
    print(f"Model:       {args.model}")
    print(f"Prompts:     {args.prompts}")
    print(f"Output:      {eval_dir}")
    print(f"Generations: {args.num_generations}")
    print(f"Temperature: {args.temperature}")
    print("="*60)
    
    # Step 1: Inference
    inference_cmd = [
        "python", "noveltybench/src/inference.py",
        "--mode", args.mode,
        "--model", args.model,
        "--eval-dir", eval_dir,
        "--num-generations", str(args.num_generations),
        "--sampling", args.sampling,
    ]
    
    # Add prompts argument
    if args.prompts in ["curated", "wildchat"]:
        inference_cmd.extend(["--data", args.prompts])
    else:
        inference_cmd.extend(["--custom-prompts", args.prompts])
    
    run_command(inference_cmd, "1️⃣  Running inference...")
    
    # Step 2: Partition
    partition_cmd = [
        "python", "noveltybench/src/partition.py",
        "--eval-dir", eval_dir,
        "--alg", "classifier"
    ]
    run_command(partition_cmd, "2️⃣  Clustering similar generations...")
    
    # Step 3: Score
    score_cmd = [
        "python", "noveltybench/src/score.py",
        "--eval-dir", eval_dir
    ]
    run_command(score_cmd, "3️⃣  Computing novelty scores...")
    
    # Step 4: Summarize
    summarize_cmd = [
        "python", "noveltybench/src/summarize.py",
        "--eval-dir", eval_dir
    ]
    run_command(summarize_cmd, "4️⃣  Summarizing results...")
    
    print("\n" + "="*60)
    print("✅ NoveltyBench evaluation complete!")
    print(f"Results saved to: {eval_dir}")
    print("="*60)


if __name__ == "__main__":
    main()

