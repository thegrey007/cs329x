#!/usr/bin/env python3
"""
Inference Script for Diversity-Weighted DPO Models
Evaluates a trained diversity-weighted DPO model on a test dataset.
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import chz
from tqdm import tqdm

import tinker
from tinker import types
from tinker_cookbook import renderers
from tinker_cookbook.model_info import get_recommended_renderer_name
from tinker_cookbook.tokenizer_utils import get_tokenizer
from dotenv import load_dotenv

# Metric calculations
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
from nltk import ngrams
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_distances

# Ensure NLTK data is available
import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')

load_dotenv()

# ============================================================
# CLI CONFIGURATION
# ============================================================

@chz.chz
class InferenceConfig:
    """Configuration for diversity-weighted DPO inference"""
    model_path: str  # Required: Path to trained model (tinker:// or experiments/...)
    test_dataset: str = "scripts/test_ultrafeedback.parquet"
    k_samples: int = 5
    output_dir: str = "inference_results/diversity_weighted_dpo"
    max_concurrent: int = 10
    max_tokens: int = 512
    temperature: float = 1.0

# ============================================================
# METRIC FUNCTIONS
# ============================================================

def calculate_self_bleu(responses):
    """Calculate Self-BLEU score (lower = more diverse)"""
    if len(responses) < 2:
        return 0.0
    
    smoothing = SmoothingFunction().method1
    bleu_scores = []
    
    for i, response in enumerate(responses):
        hypothesis = word_tokenize(response.lower())
        references = [
            word_tokenize(responses[j].lower()) 
            for j in range(len(responses)) 
            if j != i
        ]
        
        try:
            bleu = sentence_bleu(
                references, 
                hypothesis, 
                smoothing_function=smoothing,
                weights=(0.25, 0.25, 0.25, 0.25)
            )
            bleu_scores.append(bleu)
        except:
            continue
    
    return np.mean(bleu_scores) if bleu_scores else 0.0


def calculate_distinct_n(responses, n=2):
    """Calculate Distinct-n metric (higher = more diverse)"""
    all_ngrams = []
    
    for response in responses:
        tokens = response.lower().split()
        response_ngrams = list(ngrams(tokens, n))
        all_ngrams.extend(response_ngrams)
    
    if len(all_ngrams) == 0:
        return 0.0
    
    unique_ngrams = len(set(all_ngrams))
    total_ngrams = len(all_ngrams)
    
    return unique_ngrams / total_ngrams


def calculate_semantic_diversity(responses, model):
    """Calculate semantic diversity using embeddings (higher = more diverse)"""
    if len(responses) < 2:
        return 0.0
    
    embeddings = model.encode(responses, convert_to_numpy=True)
    distances = cosine_distances(embeddings)
    
    n = len(responses)
    upper_triangle_indices = np.triu_indices(n, k=1)
    pairwise_distances = distances[upper_triangle_indices]
    
    return np.mean(pairwise_distances)

# ============================================================
# INFERENCE LOGIC
# ============================================================

async def generate_responses_for_prompt(prompt, k, sampling_client, renderer, sampling_params):
    """Generate k responses for a single prompt"""
    messages = [{"role": "user", "content": prompt}]
    model_input = renderer.build_generation_prompt(messages)
    
    # k parallel API calls
    tasks = []
    for _ in range(k):
        task = sampling_client.sample_async(
            prompt=model_input,
            num_samples=1,
            sampling_params=sampling_params
        )
        tasks.append(task)
    
    responses = await asyncio.gather(*tasks)
    
    parsed_responses = []
    for response in responses:
        parsed_message, is_valid = renderer.parse_response(response.sequences[0].tokens)
        parsed_responses.append(parsed_message["content"])
    
    return parsed_responses


async def run_inference_batch(prompts_data, k, max_concurrent, sampling_client, renderer, 
                               sampling_params, embedding_model):
    """Process multiple prompts in batches"""
    results = []
    
    for i in tqdm(range(0, len(prompts_data), max_concurrent), desc="Inference"):
        batch = prompts_data[i:i + max_concurrent]
        
        # Generate responses for batch
        batch_tasks = [
            generate_responses_for_prompt(row['instruction'], k, sampling_client, renderer, sampling_params)
            for _, row in batch.iterrows()
        ]
        
        batch_responses = await asyncio.gather(*batch_tasks)
        
        # Calculate metrics for each prompt
        for (idx, row), responses in zip(batch.iterrows(), batch_responses):
            result = {
                'prompt': row['instruction'],
                'responses': responses,
                'alpha': float(row.get('creative_score', 0.0)),
                'prompt_label': row.get('label', 'unknown'),
                'self_bleu': calculate_self_bleu(responses),
                'distinct_2': calculate_distinct_n(responses, n=2),
                'semantic_div': calculate_semantic_diversity(responses, embedding_model)
            }
            results.append(result)
        
        await asyncio.sleep(0.1)
    
    return results

# ============================================================
# MAIN FUNCTION
# ============================================================

async def run_inference(config: InferenceConfig):
    """Main inference function"""
    
    print("="*70)
    print("🚀 DIVERSITY-WEIGHTED DPO INFERENCE")
    print("="*70)
    print(f"Model: {config.model_path}")
    print(f"Test dataset: {config.test_dataset}")
    print(f"K samples: {config.k_samples}")
    print(f"Output: {config.output_dir}")
    print("="*70 + "\n")
    
    # Load test dataset
    print(f"📂 Loading test dataset...")
    test_df = pd.read_parquet(config.test_dataset)
    print(f"✅ Loaded {len(test_df)} test prompts\n")
    
    # Initialize model
    print("🤖 Initializing model...")
    service_client = tinker.ServiceClient()
    
    # Parse base model from model_path
    if "qwen" in config.model_path.lower():
        if "4b" in config.model_path.lower():
            base_model = "Qwen/Qwen3-4B-Instruct-2507"
        elif "8b" in config.model_path.lower():
            base_model = "Qwen/Qwen3-8B"
        else:
            base_model = "Qwen/Qwen3-8B"  # default
    else:
        base_model = "Qwen/Qwen3-8B"  # fallback
    
    sampling_client = service_client.create_sampling_client(
        base_model=base_model,
        model_path=config.model_path
    )
    
    tokenizer = get_tokenizer(base_model)
    renderer = renderers.get_renderer(
        get_recommended_renderer_name(base_model),
        tokenizer
    )
    
    sampling_params = types.SamplingParams(
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        top_p=0.95,
        stop=renderer.get_stop_sequences(),
    )
    
    print("✅ Model initialized!\n")
    
    # Load embedding model for semantic diversity
    print("🔤 Loading embedding model...")
    embedding_model = SentenceTransformer('all-mpnet-base-v2')
    print("✅ Embedding model loaded!\n")
    
    # Run inference
    start_time = time.time()
    results = await run_inference_batch(
        prompts_data=test_df,
        k=config.k_samples,
        max_concurrent=config.max_concurrent,
        sampling_client=sampling_client,
        renderer=renderer,
        sampling_params=sampling_params,
        embedding_model=embedding_model
    )
    elapsed = time.time() - start_time
    
    # Extract experiment folder name from model_path
    if config.model_path.startswith("tinker://experiments/"):
        experiment_folder = config.model_path.replace("tinker://experiments/", "")
    elif config.model_path.startswith("experiments/"):
        experiment_folder = config.model_path.replace("experiments/", "")
    else:
        experiment_folder = Path(config.model_path).name
    
    # Parse train dataset and gamma from experiment folder
    # Format: diversity_weighted_dataset1_gamma0.05_timestamp
    if "dataset" in experiment_folder:
        parts = experiment_folder.split("_")
        train_dataset = next((p for p in parts if p.startswith("dataset")), "unknown")
        gamma_part = next((p for p in parts if p.startswith("gamma")), None)
        gamma_value = gamma_part if gamma_part else "unknown"
    else:
        train_dataset = "unknown"
        gamma_value = "unknown"
    
    # Compute summary statistics
    summary = {
        'total_prompts': len(results),
        'avg_self_bleu': float(np.mean([r['self_bleu'] for r in results])),
        'avg_distinct_2': float(np.mean([r['distinct_2'] for r in results])),
        'avg_semantic_div': float(np.mean([r['semantic_div'] for r in results])),
        'elapsed_time_seconds': elapsed,
        'time_per_prompt': elapsed / len(results)
    }
    
    # Prepare output
    output_data = {
        'metadata': {
            'approach': 'diversity_weighted_dpo',
            'experiment_folder': experiment_folder,
            'model_path': config.model_path,
            'base_model': base_model,
            'train_dataset': train_dataset,
            'gamma': gamma_value,
            'test_dataset': config.test_dataset,
            'k_samples': config.k_samples,
            'timestamp': datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            'max_tokens': config.max_tokens,
            'temperature': config.temperature
        },
        'results': results,
        'summary': summary
    }
    
    # Save results
    os.makedirs(config.output_dir, exist_ok=True)
    output_file = os.path.join(config.output_dir, f"{experiment_folder}.json")
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    # Print summary
    print("\n" + "="*70)
    print("✅ INFERENCE COMPLETE!")
    print("="*70)
    print(f"📊 Results:")
    print(f"   Total prompts: {summary['total_prompts']}")
    print(f"   Avg Self-BLEU: {summary['avg_self_bleu']:.4f}")
    print(f"   Avg Distinct-2: {summary['avg_distinct_2']:.4f}")
    print(f"   Avg Semantic Div: {summary['avg_semantic_div']:.4f}")
    print(f"   Time: {elapsed:.1f}s ({summary['time_per_prompt']:.2f}s/prompt)")
    print(f"💾 Saved: {output_file}")
    print("="*70 + "\n")

def main(config: InferenceConfig):
    """Synchronous wrapper for async inference"""
    asyncio.run(run_inference(config))

if __name__ == "__main__":
    chz.nested_entrypoint(main)

