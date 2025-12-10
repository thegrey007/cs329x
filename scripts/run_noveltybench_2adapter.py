#!/usr/bin/env python3
"""
NoveltyBench Evaluation for 2-Adapter DPO Models

Evaluates 2-adapter models on NoveltyBench using appropriate adapters per prompt type:
- Creative prompts → Creative adapter
- Factual prompts → Factual adapter

This matches the logic in inference_2adapter_dpo.py where prompts are routed to the
adapter that was trained for that type of content.
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import asyncio
import json
import subprocess
import sys
import torch
import copy
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
import numpy as np
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

load_dotenv()

# ============================================================
# CLI CONFIGURATION
# ============================================================

@chz.chz
class NoveltyBenchConfig:
    """Configuration for NoveltyBench evaluation with 2-adapter DPO"""
    factual_adapter_path: str  # Required: Path to factual adapter
    creative_adapter_path: str  # Required: Path to creative adapter
    data: str = "curated"  # Dataset: 'curated', 'wildchat', or path to custom JSONL
    num_generations: int = 10  # Number of generations per prompt
    max_tokens: int = 512
    temperature: float = 1.0
    concurrent_requests: int = 10
    eval_dir: str = None  # Auto-generated if not provided
    output_dir: str | None = None  # Base output directory (default: noveltybench/results)
    base_model: str = "Qwen/Qwen3-8B"  # Base model to use
    mode: str = "combined"  # 'combined' (use both adapters intelligently), 'factual_only', 'creative_only', or 'both_separate'
    run_full_pipeline: bool = True  # Run partition/score/summarize after inference
    # Prompt classifier (8B LLM log-prob based)
    classifier_model: str = "MBZUAI/LaMini-Flan-T5-248M"  # matches notebook
    classifier_labels: list[str] = ("informational", "creative")
    use_8b_classifier: bool = False
    classifier_model_8b: str = "meta-llama/Meta-Llama-3-8B-Instruct"

# ============================================================
# PROMPT ROUTING (match inference_2adapter_dpo.py)
# ============================================================

def select_adapter(prompt_data: dict) -> str:
    """
    Mirror routing logic from inference_2adapter_dpo.py:
    - Uses creative_score vs factual_score (or alpha) to decide
    - creative_score >= factual_score → creative adapter
    - otherwise → factual adapter
    """
    creative_score = prompt_data.get('creative_score', prompt_data.get('alpha', 0.5))
    factual_score = prompt_data.get('factual_score', 0.5)
    return 'creative' if creative_score >= factual_score else 'factual'


# ============================================================
# PROMPT CLASSIFIER (matches notebook logic: LaMini-Flan-T5-248M logprobs)
# Optionally swap to an 8B classifier (logprob labels) if configured.
# ============================================================

@torch.inference_mode()
def classify_with_seq2seq(model, tokenizer, texts, labels):
    results = []
    for text in texts:
        input_text = f"Classify the following user prompt as 'informational' or 'creative':{text}\nAnswer:"
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        label_scores = {}
        for label in labels:
            label_ids = tokenizer(label, return_tensors="pt").input_ids.to(model.device)
            outputs = model(**inputs, labels=label_ids)
            logprob = -outputs.loss.item()
            label_scores[label] = logprob
        probs = torch.softmax(torch.tensor(list(label_scores.values())), dim=0)
        factual_score = float(probs[0].item())
        creative_score = float(probs[1].item())
        predicted_label = labels[int(torch.argmax(probs))]
        results.append(
            {
                "label": "factual" if predicted_label == "informational" else "creative",
                "factual_score": factual_score,
                "creative_score": creative_score,
            }
        )
    return results


def score_prompts_with_classifier(prompts: list[dict], config: NoveltyBenchConfig):
    if config.use_8b_classifier:
        # 8B causal LLM, label logprob classification
        tokenizer = AutoTokenizer.from_pretrained(config.classifier_model_8b)
        model = AutoModelForCausalLM.from_pretrained(
            config.classifier_model_8b,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )
        model.eval()

        @torch.inference_mode()
        def label_logprob(prompt_text: str, label: str) -> float:
            text = f"Classify the following user prompt as 'informational' or 'creative': {prompt_text}\nAnswer: {label}"
            enc = tokenizer(text, return_tensors="pt")
            input_ids = enc.input_ids.to(model.device)
            attn = enc.attention_mask.to(model.device)
            outputs = model(input_ids=input_ids, attention_mask=attn)
            logits = outputs.logits[:, :-1, :]
            labels_ids = input_ids[:, 1:]
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            token_log_probs = log_probs.gather(-1, labels_ids.unsqueeze(-1)).squeeze(-1)
            label_ids = tokenizer(label, add_special_tokens=False).input_ids
            label_len = len(label_ids)
            token_log_probs = token_log_probs[:, -label_len:]
            return float(token_log_probs.sum().item())

        for p in prompts:
            lp = [label_logprob(p["prompt"], lbl) for lbl in config.classifier_labels]
            probs = torch.softmax(torch.tensor(lp), dim=0)
            factual_prob = float(probs[0].item())
            creative_prob = float(probs[1].item())
            p["factual_score"] = factual_prob
            p["creative_score"] = creative_prob
    else:
        # LaMini-Flan-T5-248M classifier (matches notebook)
        tokenizer = AutoTokenizer.from_pretrained(config.classifier_model)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            config.classifier_model,
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        model.eval()
        texts = [p["prompt"] for p in prompts]
        preds = classify_with_seq2seq(model, tokenizer, texts, config.classifier_labels)
        for p, pred in zip(prompts, preds):
            p["factual_score"] = pred["factual_score"]
            p["creative_score"] = pred["creative_score"]

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
    """Generate n responses for a single prompt using one adapter"""
    # Add /no_think to disable chain-of-thought reasoning (match inference_2adapter_dpo.py)
    messages = [{"role": "user", "content": f"{prompt} /no_think"}]
    model_input = renderer.build_generation_prompt(messages)
    
    # Generate n samples in parallel
    tasks = []
    for _ in range(n):
        task = sampling_client.sample_async(
            prompt=model_input,
            num_samples=1,
            sampling_params=sampling_params
        )
        tasks.append(task)
    
    responses = await asyncio.gather(*tasks)
    
    # Parse responses
    parsed_responses = []
    for response in responses:
        parsed_message, is_valid = renderer.parse_response(response.sequences[0].tokens)
        parsed_responses.append(parsed_message["content"])
    
    return parsed_responses


async def process_prompts_combined(
    prompts: list,
    factual_client,
    creative_client,
    renderer,
    sampling_params,
    num_generations: int,
    concurrent_requests: int,
    output_file: Path
):
    """Process prompts using both adapters based on prompt classification"""
    semaphore = asyncio.Semaphore(concurrent_requests)
    results = []
    
    async def process_single_prompt(prompt_data):
        async with semaphore:
            # Route using score-based logic (same as inference_2adapter_dpo.py)
            prompt_type = select_adapter(prompt_data)
            sampling_client = creative_client if prompt_type == 'creative' else factual_client
            
            # Generate responses
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
                'model': f"2adapter_dpo_{prompt_type}",
                'generations': generations,
                'adapter_used': prompt_type,
                'category': prompt_data.get('category', 'unknown'),
                'creative_score': prompt_data.get('creative_score', prompt_data.get('alpha', None)),
                'factual_score': prompt_data.get('factual_score', None)
            }
    
    # Process all prompts
    tasks = [process_single_prompt(p) for p in prompts]
    
    for task in tqdm(asyncio.as_completed(tasks), total=len(prompts), desc="Generating"):
        result = await task
        results.append(result)
    
    # Write results
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    
    return results


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
    """Process prompts using a single adapter"""
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
    
    # Process all prompts
    tasks = [process_single_prompt(p) for p in prompts]
    
    for task in tqdm(asyncio.as_completed(tasks), total=len(prompts), desc="Generating"):
        result = await task
        results.append(result)
    
    # Write results
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
    
    env = copy.environ if hasattr(copy, "environ") else None
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
    print("🎯 NOVELTYBENCH EVALUATION - 2-ADAPTER DPO")
    print("="*70)
    print(f"Mode: {config.mode}")
    print(f"Factual adapter: {config.factual_adapter_path}")
    print(f"Creative adapter: {config.creative_adapter_path}")
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
    
    # LLM-based creative scoring (adds creative_score/factual_score)
    print("🧠 Scoring prompts with classifier...")
    score_prompts_with_classifier(prompts, config)
    print("✅ Classifier scores added\n")
    
    # Initialize models
    print("🤖 Initializing models...")
    service_client = tinker.ServiceClient()
    
    factual_client = service_client.create_sampling_client(
        base_model=config.base_model,
        model_path=config.factual_adapter_path
    )
    
    creative_client = service_client.create_sampling_client(
        base_model=config.base_model,
        model_path=config.creative_adapter_path
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
    
    print("✅ Models initialized!\n")
    
    # Determine eval directory
    base_out = Path(config.output_dir) if config.output_dir else Path("noveltybench/results")
    if config.eval_dir is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        eval_dir = base_out / f"2adapter_{config.mode}_{config.data}_{timestamp}"
    else:
        eval_dir = Path(config.eval_dir)
    
    # Run inference based on mode
    if config.mode == "combined":
        print("🚀 Running inference with adaptive adapter selection...\n")
        output_file = eval_dir / "generations.jsonl"
        await process_prompts_combined(
            prompts,
            factual_client,
            creative_client,
            renderer,
            sampling_params,
            config.num_generations,
            config.concurrent_requests,
            output_file
        )
        print(f"\n✅ Inference complete! Results saved to {output_file}\n")
        
        if config.run_full_pipeline:
            if run_partition(eval_dir):
                if run_score(eval_dir):
                    run_summarize(eval_dir)
            
            summary_file = eval_dir / "summary.json"
            if summary_file.exists():
                with open(summary_file) as f:
                    summary = json.load(f)
                print("\n" + "="*70)
                print("📊 FINAL RESULTS")
                print("="*70)
                print(json.dumps(summary, indent=2))
                print("="*70 + "\n")
    
    elif config.mode == "factual_only":
        print("🚀 Running inference with factual adapter only...\n")
        output_file = eval_dir / "generations.jsonl"
        await process_prompts_single_adapter(
            prompts,
            factual_client,
            renderer,
            sampling_params,
            config.num_generations,
            config.concurrent_requests,
            output_file,
            "2adapter_factual"
        )
        print(f"\n✅ Inference complete! Results saved to {output_file}\n")
        
        if config.run_full_pipeline:
            run_partition(eval_dir)
            run_score(eval_dir)
            run_summarize(eval_dir)
    
    elif config.mode == "creative_only":
        print("🚀 Running inference with creative adapter only...\n")
        output_file = eval_dir / "generations.jsonl"
        await process_prompts_single_adapter(
            prompts,
            creative_client,
            renderer,
            sampling_params,
            config.num_generations,
            config.concurrent_requests,
            output_file,
            "2adapter_creative"
        )
        print(f"\n✅ Inference complete! Results saved to {output_file}\n")
        
        if config.run_full_pipeline:
            run_partition(eval_dir)
            run_score(eval_dir)
            run_summarize(eval_dir)
    
    elif config.mode == "both_separate":
        print("🚀 Running inference with both adapters separately...\n")
        
        # Factual adapter
        factual_dir = eval_dir / "factual"
        output_file_factual = factual_dir / "generations.jsonl"
        print("\n📍 Processing with FACTUAL adapter...")
        await process_prompts_single_adapter(
            prompts,
            factual_client,
            renderer,
            sampling_params,
            config.num_generations,
            config.concurrent_requests,
            output_file_factual,
            "2adapter_factual"
        )
        print(f"✅ Factual adapter complete! Results: {output_file_factual}\n")
        
        # Creative adapter
        creative_dir = eval_dir / "creative"
        output_file_creative = creative_dir / "generations.jsonl"
        print("\n📍 Processing with CREATIVE adapter...")
        await process_prompts_single_adapter(
            prompts,
            creative_client,
            renderer,
            sampling_params,
            config.num_generations,
            config.concurrent_requests,
            output_file_creative,
            "2adapter_creative"
        )
        print(f"✅ Creative adapter complete! Results: {output_file_creative}\n")
        
        if config.run_full_pipeline:
            # Run pipeline for both
            print("\n🔄 Running full pipeline for FACTUAL adapter...")
            run_partition(factual_dir)
            run_score(factual_dir)
            run_summarize(factual_dir)
            
            print("\n🔄 Running full pipeline for CREATIVE adapter...")
            run_partition(creative_dir)
            run_score(creative_dir)
            run_summarize(creative_dir)
            
            # Display both results
            print("\n" + "="*70)
            print("📊 COMPARISON RESULTS")
            print("="*70)
            
            for adapter_name, adapter_dir in [("FACTUAL", factual_dir), ("CREATIVE", creative_dir)]:
                summary_file = adapter_dir / "summary.json"
                if summary_file.exists():
                    with open(summary_file) as f:
                        summary = json.load(f)
                    print(f"\n{adapter_name} ADAPTER:")
                    print(json.dumps(summary, indent=2))
            
            print("="*70 + "\n")
    
    print("\n✅ NoveltyBench evaluation complete!")
    print(f"📁 Results directory: {eval_dir}\n")


def main(config: NoveltyBenchConfig):
    """Synchronous wrapper for async evaluation"""
    asyncio.run(run_noveltybench_eval(config))


if __name__ == "__main__":
    chz.nested_entrypoint(main)

