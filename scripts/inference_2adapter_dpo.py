#!/usr/bin/env python3
"""
Inference Script for 2-Adapter DPO Models
Evaluates a trained 2-adapter DPO model on a test dataset.
Uses factual adapter for factual prompts, creative adapter for creative prompts.
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
    """Configuration for 2-adapter DPO inference"""
    factual_adapter_path: str  # Required: Path to factual adapter (tinker:// or experiments/.../factual)
    creative_adapter_path: str  # Required: Path to creative adapter (tinker:// or experiments/.../creative)
    test_dataset: str = "scripts/test_ultrafeedback.parquet"
    k_samples: int = 5
    output_dir: str = "inference_results/2adapter_dpo"
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


async def run_inference_batch(prompts_data, k, max_concurrent, factual_client, creative_client,
                               renderer, sampling_params, embedding_model):
    """Process multiple prompts in batches with appropriate adapter"""
    results = []
    
    for i in tqdm(range(0, len(prompts_data), max_concurrent), desc="Inference"):
        batch = prompts_data[i:i + max_concurrent]
        
        # Generate responses for batch (selecting appropriate adapter per prompt)
        batch_tasks = []
        for idx, row in batch.iterrows():
            # Determine which adapter to use based on creative_score
            is_creative = row.get('creative_score', 0.5) >= row.get('factual_score', 0.5)
            sampling_client = creative_client if is_creative else factual_client
            
            task = generate_responses_for_prompt(
                row['instruction'], k, sampling_client, renderer, sampling_params
            )
            batch_tasks.append((task, is_creative))
        
        # Await all tasks
        batch_results = await asyncio.gather(*[task for task, _ in batch_tasks])
        
        # Calculate metrics for each prompt
        for (idx, row), responses, (_, is_creative) in zip(batch.iterrows(), batch_results, batch_tasks):
            result = {
                'prompt': row['instruction'],
                'responses': responses,
                'alpha': float(row.get('creative_score', 0.0)),
                'prompt_label': 'creative' if is_creative else 'factual',
                'adapter_used': 'creative' if is_creative else 'factual',
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
    print("🚀 2-ADAPTER DPO INFERENCE")
    print("="*70)
    print(f"Factual adapter: {config.factual_adapter_path}")
    print(f"Creative adapter: {config.creative_adapter_path}")
    print(f"Test dataset: {config.test_dataset}")
    print(f"K samples: {config.k_samples}")
    print(f"Output: {config.output_dir}")
    print("="*70 + "\n")
    
    # Load test dataset
    print(f"📂 Loading test dataset...")
    test_df = pd.read_parquet(config.test_dataset)
    print(f"✅ Loaded {len(test_df)} test prompts\n")
    
    # Initialize models
    print("🤖 Initializing models...")
    service_client = tinker.ServiceClient()
    
    # Parse base model from adapter path
    if "qwen" in config.factual_adapter_path.lower():
        if "4b" in config.factual_adapter_path.lower():
            base_model = "Qwen/Qwen3-4B-Instruct-2507"
        elif "8b" in config.factual_adapter_path.lower():
            base_model = "Qwen/Qwen3-8B"
        else:
            base_model = "Qwen/Qwen3-8B"  # default
    else:
        base_model = "Qwen/Qwen3-8B"  # fallback
    
    # Create two sampling clients (one per adapter)
    factual_client = service_client.create_sampling_client(
        base_model=base_model,
        model_path=config.factual_adapter_path
    )
    
    creative_client = service_client.create_sampling_client(
        base_model=base_model,
        model_path=config.creative_adapter_path
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
    
    print("✅ Both adapters initialized!\n")
    
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
        factual_client=factual_client,
        creative_client=creative_client,
        renderer=renderer,
        sampling_params=sampling_params,
        embedding_model=embedding_model
    )
    elapsed = time.time() - start_time
    
    # Extract experiment folder name (parent of factual/creative)
    if config.factual_adapter_path.startswith("tinker://experiments/"):
        full_path = config.factual_adapter_path.replace("tinker://experiments/", "")
        experiment_folder = str(Path(full_path).parent)
    elif config.factual_adapter_path.startswith("experiments/"):
        full_path = config.factual_adapter_path.replace("experiments/", "")
        experiment_folder = str(Path(full_path).parent)
    else:
        experiment_folder = str(Path(config.factual_adapter_path).parent.name)
    
    # Parse train dataset from experiment folder
    if "dataset" in experiment_folder:
        # Extract datasetN from "2adapter_dataset1_timestamp"
        parts = experiment_folder.split("_")
        train_dataset = next((p for p in parts if p.startswith("dataset")), "unknown")
    else:
        train_dataset = "unknown"
    
    # Compute summary statistics
    summary = {
        'total_prompts': len(results),
        'avg_self_bleu': float(np.mean([r['self_bleu'] for r in results])),
        'avg_distinct_2': float(np.mean([r['distinct_2'] for r in results])),
        'avg_semantic_div': float(np.mean([r['semantic_div'] for r in results])),
        'factual_prompts': sum(1 for r in results if r['adapter_used'] == 'factual'),
        'creative_prompts': sum(1 for r in results if r['adapter_used'] == 'creative'),
        'elapsed_time_seconds': elapsed,
        'time_per_prompt': elapsed / len(results)
    }
    
    # Prepare output
    output_data = {
        'metadata': {
            'approach': '2adapter_dpo',
            'experiment_folder': experiment_folder,
            'factual_adapter_path': config.factual_adapter_path,
            'creative_adapter_path': config.creative_adapter_path,
            'base_model': base_model,
            'train_dataset': train_dataset,
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
    print(f"   - Factual: {summary['factual_prompts']}")
    print(f"   - Creative: {summary['creative_prompts']}")
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

