#!/usr/bin/env python3
"""
NoveltyBench Evaluation for Standard (single-adapter) DPO Models

Runs a single adapter/model on NoveltyBench prompts:
- Generates multiple samples per prompt
- (Optional) runs partition/score/summarize
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import asyncio
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from tqdm.auto import tqdm
import chz

import tinker
from tinker import types
from tinker_cookbook import renderers
from tinker_cookbook.model_info import get_recommended_renderer_name
from tinker_cookbook.tokenizer_utils import get_tokenizer
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# CLI CONFIGURATION
# ============================================================

@chz.chz
class NoveltyBenchConfig:
    """Configuration for NoveltyBench evaluation with a single adapter/model"""
    adapter_path: str  # Required: Path to adapter/model (tinker:// or experiments/... or HF)
    data: str = "curated"  # Dataset: 'curated', 'wildchat', or path to custom JSONL
    num_generations: int = 10  # Number of generations per prompt
    max_tokens: int = 512
    temperature: float = 1.0
    concurrent_requests: int = 10
    eval_dir: str | None = None  # Auto-generated if not provided
    output_dir: str | None = None  # Base output directory (default: noveltybench/results)
    base_model: str = "Qwen/Qwen3-8B"  # Base model to use
    mode: str = "single"  # Reserved; only single adapter here
    run_full_pipeline: bool = True  # Run partition/score/summarize after inference

# ============================================================
# INFERENCE LOGIC
# ============================================================

async def generate_responses_single_prompt(
    prompt: str,
    n: int,
    sampling_client,
    renderer,
    sampling_params
) -> list[str]:
    """Generate n responses for a single prompt"""
    # Add /no_think to disable chain-of-thought reasoning (match standard inference)
    messages = [{"role": "user", "content": f"{prompt} /no_think"}]
    model_input = renderer.build_generation_prompt(messages)
    
    tasks = []
    for _ in range(n):
        task = sampling_client.sample_async(
            prompt=model_input,
            num_samples=1,
            sampling_params=sampling_params
        )
        tasks.append(task)
    
    responses = await asyncio.gather(*tasks)
    
    parsed_responses = []
    for response in responses:
        parsed_message, _ = renderer.parse_response(response.sequences[0].tokens)
        parsed_responses.append(parsed_message["content"])
    
    return parsed_responses


async def process_prompts_single_adapter(
    prompts: list,
    sampling_client,
    renderer,
    sampling_params,
    num_generations: int,
    concurrent_requests: int,
    output_file: Path,
    model_name: str
):
    """Process prompts using a single adapter/model"""
    semaphore = asyncio.Semaphore(concurrent_requests)
    results = []
    
    async def process_single_prompt(prompt_data):
        async with semaphore:
            generations = await generate_responses_single_prompt(
                prompt_data['prompt'],
                num_generations,
                sampling_client,
                renderer,
                sampling_params
            )
            
            return {
                'id': prompt_data['id'],
                'prompt': prompt_data['prompt'],
                'model': model_name,
                'generations': generations
            }
    
    tasks = [process_single_prompt(p) for p in prompts]
    
    for task in tqdm(asyncio.as_completed(tasks), total=len(prompts), desc="Generating"):
        result = await task
        results.append(result)
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    
    return results

# ============================================================
# NOVELTYBENCH PIPELINE
# ============================================================

def run_partition(eval_dir: Path):
    """Run NoveltyBench partition step"""
    print(f"\n{'='*70}")
    print("🔍 Running partition...")
    print(f"{'='*70}\n")
    
    env = dict(os.environ)
    env["PYTHONPATH"] = "noveltybench"
    
    cmd = [
        "python", "noveltybench/src/partition.py",
        "--eval-dir", str(eval_dir),
        "--alg", "classifier"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        print(f"⚠️  Partition failed: {result.stderr}")
        return False
    print("✅ Partition complete!")
    return True


def run_score(eval_dir: Path, patience: float = 0.8):
    """Run NoveltyBench scoring step"""
    print(f"\n{'='*70}")
    print("📊 Running scoring...")
    print(f"{'='*70}\n")
    
    env = dict(os.environ)
    env["PYTHONPATH"] = "noveltybench"
    
    cmd = [
        "python", "noveltybench/src/score.py",
        "--eval-dir", str(eval_dir),
        "--patience", str(patience)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        print(f"⚠️  Scoring failed: {result.stderr}")
        return False
    print("✅ Scoring complete!")
    return True


def run_summarize(eval_dir: Path):
    """Run NoveltyBench summarization step"""
    print(f"\n{'='*70}")
    print("📈 Running summarization...")
    print(f"{'='*70}\n")
    
    env = dict(os.environ)
    env["PYTHONPATH"] = "noveltybench"
    
    cmd = [
        "python", "noveltybench/src/summarize.py",
        "--eval-dir", str(eval_dir)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        print(f"⚠️  Summarization failed: {result.stderr}")
        return False
    print("✅ Summarization complete!")
    return True

# ============================================================
# MAIN FUNCTION
# ============================================================

async def run_noveltybench_eval(config: NoveltyBenchConfig):
    """Main evaluation function"""
    
    print("="*70)
    print("🎯 NOVELTYBENCH EVALUATION - STANDARD DPO")
    print("="*70)
    print(f"Adapter: {config.adapter_path}")
    print(f"Dataset: {config.data}")
    print(f"Num generations: {config.num_generations}")
    print("="*70 + "\n")
    
    # Load prompts
    print("📂 Loading prompts...")
    noveltybench_dir = Path("noveltybench")
    
    if config.data == "curated":
        data_file = noveltybench_dir / "data" / "curated.jsonl"
    elif config.data == "wildchat":
        data_file = noveltybench_dir / "data" / "wildchat-1k.jsonl"
    else:
        data_file = Path(config.data)
    
    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")
    
    prompts = []
    with open(data_file) as f:
        for line in f:
            prompts.append(json.loads(line))
    
    print(f"✅ Loaded {len(prompts)} prompts\n")
    
    # Initialize model
    print("🤖 Initializing model...")
    service_client = tinker.ServiceClient()
    
    sampling_client = service_client.create_sampling_client(
        base_model=config.base_model,
        model_path=config.adapter_path
    )
    
    tokenizer = get_tokenizer(config.base_model)
    renderer = renderers.get_renderer(
        get_recommended_renderer_name(config.base_model),
        tokenizer
    )
    
    sampling_params = types.SamplingParams(
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        top_p=0.95,
        stop=renderer.get_stop_sequences(),
    )
    
    print("✅ Model initialized!\n")
    
    # Determine eval directory
    base_out = Path(config.output_dir) if config.output_dir else Path("noveltybench/results")
    if config.eval_dir is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        eval_dir = base_out / f"standard_dpo_{config.data}_{timestamp}"
    else:
        eval_dir = Path(config.eval_dir)
    
    # Run inference
    print("🚀 Running inference...\n")
    output_file = eval_dir / "generations.jsonl"
    await process_prompts_single_adapter(
        prompts,
        sampling_client,
        renderer,
        sampling_params,
        config.num_generations,
        config.concurrent_requests,
        output_file,
        "standard_dpo"
    )
    print(f"\n✅ Inference complete! Results saved to {output_file}\n")
    
    if config.run_full_pipeline:
        if run_partition(eval_dir):
            if run_score(eval_dir):
                run_summarize(eval_dir)
    
    print("\n✅ NoveltyBench evaluation complete!")
    print(f"📁 Results directory: {eval_dir}\n")


def main(config: NoveltyBenchConfig):
    """Synchronous wrapper for async evaluation"""
    asyncio.run(run_noveltybench_eval(config))


if __name__ == "__main__":
    chz.nested_entrypoint(main)


