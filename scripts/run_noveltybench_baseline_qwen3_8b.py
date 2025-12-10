#!/usr/bin/env python3
"""
NoveltyBench baseline evaluation for pretrained Qwen3-8B (no adapters).

Generates multiple samples per prompt and optionally runs the NoveltyBench
partition/score/summarize pipeline.
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import asyncio
import json
import subprocess
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


@chz.chz
class BaselineConfig:
    """Configuration for Qwen3-8B NoveltyBench baseline"""
    data: str = "curated"  # 'curated', 'wildchat', or path to JSONL
    num_generations: int = 10
    max_tokens: int = 512
    temperature: float = 1.0
    concurrent_requests: int = 10
    eval_dir: str | None = None
    output_dir: str | None = None  # base output dir; default noveltybench/results
    base_model: str = "Qwen/Qwen3-8B"  # pretrained Qwen3-8B
    run_full_pipeline: bool = True


async def generate_responses_single_prompt(
    prompt: str,
    n: int,
    sampling_client,
    renderer,
    sampling_params,
) -> list[str]:
    """Generate n responses for a single prompt"""
    messages = [{"role": "user", "content": f"{prompt} /no_think"}]
    model_input = renderer.build_generation_prompt(messages)

    tasks = []
    for _ in range(n):
        task = sampling_client.sample_async(
            prompt=model_input,
            num_samples=1,
            sampling_params=sampling_params,
        )
        tasks.append(task)

    responses = await asyncio.gather(*tasks)
    parsed_responses = []
    for response in responses:
        parsed_message, _ = renderer.parse_response(response.sequences[0].tokens)
        parsed_responses.append(parsed_message["content"])
    return parsed_responses


async def process_prompts(
    prompts: list,
    sampling_client,
    renderer,
    sampling_params,
    num_generations: int,
    concurrent_requests: int,
    output_file: Path,
    model_name: str,
):
    semaphore = asyncio.Semaphore(concurrent_requests)
    results = []

    async def process_single_prompt(prompt_data):
        async with semaphore:
            generations = await generate_responses_single_prompt(
                prompt_data["prompt"],
                num_generations,
                sampling_client,
                renderer,
                sampling_params,
            )
            return {
                "id": prompt_data["id"],
                "prompt": prompt_data["prompt"],
                "model": model_name,
                "generations": generations,
            }

    tasks = [process_single_prompt(p) for p in prompts]

    for task in tqdm(asyncio.as_completed(tasks), total=len(prompts), desc="Generating"):
        result = await task
        results.append(result)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    return results


def run_partition(eval_dir: Path):
    print(f"\n{'='*70}")
    print("🔍 Running partition...")
    print(f"{'='*70}\n")
    env = dict(os.environ)
    env["PYTHONPATH"] = "noveltybench"
    cmd = [
        "python",
        "noveltybench/src/partition.py",
        "--eval-dir",
        str(eval_dir),
        "--alg",
        "classifier",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        print(f"⚠️  Partition failed: {result.stderr}")
        return False
    print("✅ Partition complete!")
    return True


def run_score(eval_dir: Path, patience: float = 0.8):
    print(f"\n{'='*70}")
    print("📊 Running scoring...")
    print(f"{'='*70}\n")
    env = dict(os.environ)
    env["PYTHONPATH"] = "noveltybench"
    cmd = [
        "python",
        "noveltybench/src/score.py",
        "--eval-dir",
        str(eval_dir),
        "--patience",
        str(patience),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        print(f"⚠️  Scoring failed: {result.stderr}")
        return False
    print("✅ Scoring complete!")
    return True


def run_summarize(eval_dir: Path):
    print(f"\n{'='*70}")
    print("📈 Running summarization...")
    print(f"{'='*70}\n")
    env = dict(os.environ)
    env["PYTHONPATH"] = "noveltybench"
    cmd = [
        "python",
        "noveltybench/src/summarize.py",
        "--eval-dir",
        str(eval_dir),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        print(f"⚠️  Summarization failed: {result.stderr}")
        return False
    print("✅ Summarization complete!")
    return True


async def run_baseline(config: BaselineConfig):
    print("=" * 70)
    print("🎯 NOVELTYBENCH BASELINE - QWEN3-8B")
    print("=" * 70)
    print(f"Base model: {config.base_model}")
    print(f"Dataset: {config.data}")
    print(f"Num generations: {config.num_generations}")
    print("=" * 70 + "\n")

    # Load prompts
    noveltybench_dir = Path("noveltybench")
    if config.data == "curated":
        data_file = noveltybench_dir / "data" / "curated.jsonl"
    elif config.data == "wildchat":
        data_file = noveltybench_dir / "data" / "wildchat-1k.jsonl"
    else:
        data_file = Path(config.data)
    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")
    prompts = [json.loads(line) for line in data_file.read_text().splitlines()]
    print(f"✅ Loaded {len(prompts)} prompts\n")

    # Init model
    service_client = tinker.ServiceClient()
    sampling_client = service_client.create_sampling_client(
        base_model=config.base_model,
        model_path=None,  # no adapter, use base model
    )
    tokenizer = get_tokenizer(config.base_model)
    renderer = renderers.get_renderer(
        get_recommended_renderer_name(config.base_model),
        tokenizer,
    )
    sampling_params = types.SamplingParams(
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        top_p=0.95,
        stop=renderer.get_stop_sequences(),
    )

    # Eval dir
    base_out = Path(config.output_dir) if config.output_dir else Path("noveltybench/results")
    if config.eval_dir is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        eval_dir = base_out / f"baseline_qwen3_8b_{config.data}_{timestamp}"
    else:
        eval_dir = Path(config.eval_dir)

    # Inference
    output_file = eval_dir / "generations.jsonl"
    print("🚀 Running inference...\n")
    await process_prompts(
        prompts,
        sampling_client,
        renderer,
        sampling_params,
        config.num_generations,
        config.concurrent_requests,
        output_file,
        "qwen3-8b-baseline",
    )
    print(f"\n✅ Inference complete! Results saved to {output_file}\n")

    # Pipeline
    if config.run_full_pipeline:
        if run_partition(eval_dir):
            if run_score(eval_dir):
                run_summarize(eval_dir)

    print("\n✅ NoveltyBench baseline complete!")
    print(f"📁 Results directory: {eval_dir}\n")


def main(config: BaselineConfig):
    asyncio.run(run_baseline(config))


if __name__ == "__main__":
    chz.nested_entrypoint(main)



