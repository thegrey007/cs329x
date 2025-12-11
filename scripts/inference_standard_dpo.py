#!/usr/bin/env python3
"""
Inference Script for Standard DPO Models
Evaluates a trained standard DPO model on a test dataset.
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import asyncio
import random
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
# HELPER FUNCTIONS
# ============================================================

def sanitize_for_json(obj):
    """Recursively sanitize objects for JSON serialization"""
    import math
    
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        val = float(obj)
        # Replace NaN and Inf with 0.0
        if not math.isfinite(val):
            return 0.0
        return val
    elif isinstance(obj, np.ndarray):
        return sanitize_for_json(obj.tolist())
    elif isinstance(obj, (float, int)):
        if isinstance(obj, float) and not math.isfinite(obj):
            return 0.0
        return obj
    elif obj is None or isinstance(obj, (str, bool)):
        return obj
    else:
        return str(obj)

# ============================================================
# CLI CONFIGURATION
# ============================================================

@chz.chz
class InferenceConfig:
    """Configuration for standard DPO inference"""
    model_path: str  # Required: Path to trained model (tinker:// or experiments/...)
    test_dataset: str = "scripts/test_ultrafeedback.parquet"
    k_samples: int = 5
    output_dir: str = "inference_results/standard_dpo"
    max_concurrent: int = 10
    max_tokens: int = 512
    temperature: float = 1.0
    sample_per_bin: int | None = None  # Number of prompts to sample per creativity bin (None = use all)
    output_filename: str | None = None  # Custom output filename (without .json extension, None = auto-generate)

# ============================================================
# METRIC FUNCTIONS
# ============================================================

def calculate_self_bleu(responses):
    """
    Calculate Self-BLEU score (lower = more diverse)
    Uses robust tokenization and smoothing to handle edge cases
    """
    if len(responses) < 2:
        return 0.0
    
    # Filter out empty or invalid responses
    valid_responses = [r for r in responses if r and isinstance(r, str) and r.strip()]
    if len(valid_responses) < 2:
        return 0.0
    
    # Use method4 (smoothing that works better for short/diverse texts)
    smoothing = SmoothingFunction().method4
    bleu_scores = []
    
    def simple_tokenize(text):
        """Simple tokenization that handles markdown and special chars"""
        import re
        # Remove markdown formatting
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # Remove **bold**
        text = re.sub(r'\*([^*]+)\*', r'\1', text)      # Remove *italic*
        text = re.sub(r'#{1,6}\s*', '', text)           # Remove ### headers
        text = re.sub(r'[`~]', '', text)                 # Remove code markers
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)  # Remove links
        # Simple word tokenization
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens
    
    for i, response in enumerate(valid_responses):
        try:
            # Tokenize hypothesis using simple tokenizer
            hypothesis = simple_tokenize(response)
            if not hypothesis or len(hypothesis) < 2:
                continue
                
            # Tokenize references
            references = []
            for j in range(len(valid_responses)):
                if j != i:
                    ref_tokens = simple_tokenize(valid_responses[j])
                    if ref_tokens and len(ref_tokens) >= 2:
                        references.append(ref_tokens)
            
            if not references:
                continue
            
            # Calculate BLEU-4 with smoothing
            # Use auto_reweigh=True to handle cases with fewer than 4 tokens
            bleu = sentence_bleu(
                references, 
                hypothesis, 
                smoothing_function=smoothing,
                weights=(0.25, 0.25, 0.25, 0.25),
                auto_reweigh=True
            )
            
            # Validate result
            if bleu is not None and np.isfinite(bleu) and not np.isnan(bleu):
                bleu_scores.append(float(bleu))
                
        except Exception as e:
            # Log error for debugging but continue
            print(f"Self-BLEU calculation warning: {e}")
            continue
    
    # Return mean if we have scores, otherwise calculate using simpler approach
    if len(bleu_scores) > 0:
        result = float(np.mean(bleu_scores))
        return result if np.isfinite(result) and not np.isnan(result) else 0.0
    
    # Fallback: If BLEU-4 fails, try BLEU-2 (bigram only)
    try:
        smoothing = SmoothingFunction().method4
        for i, response in enumerate(valid_responses):
            hypothesis = simple_tokenize(response)
            if len(hypothesis) < 2:
                continue
            references = [simple_tokenize(valid_responses[j]) for j in range(len(valid_responses)) if j != i]
            references = [r for r in references if len(r) >= 2]
            if not references:
                continue
            # BLEU-2 (bigram only)
            bleu = sentence_bleu(references, hypothesis, smoothing_function=smoothing, weights=(0.5, 0.5, 0, 0))
            if bleu is not None and np.isfinite(bleu):
                bleu_scores.append(float(bleu))
        
        if bleu_scores:
            return float(np.mean(bleu_scores))
    except Exception:
        pass
    
    return 0.0


def calculate_distinct_n(responses, n=2):
    """Calculate Distinct-n metric (higher = more diverse)"""
    # Filter out empty or invalid responses
    valid_responses = [r for r in responses if r and isinstance(r, str) and r.strip()]
    if not valid_responses:
        return 0.0
    
    all_ngrams = []
    
    for response in valid_responses:
        try:
        tokens = response.lower().split()
            if len(tokens) < n:  # Need at least n tokens for n-grams
                continue
        response_ngrams = list(ngrams(tokens, n))
        all_ngrams.extend(response_ngrams)
        except:
            continue
    
    if len(all_ngrams) == 0:
        return 0.0
    
    unique_ngrams = len(set(all_ngrams))
    total_ngrams = len(all_ngrams)
    
    if total_ngrams == 0:
        return 0.0
    
    result = float(unique_ngrams / total_ngrams)
    return result if np.isfinite(result) else 0.0


def calculate_semantic_diversity(responses, model):
    """Calculate semantic diversity using embeddings (higher = more diverse)"""
    # Filter out empty or invalid responses
    valid_responses = [r for r in responses if r and isinstance(r, str) and r.strip()]
    if len(valid_responses) < 2:
        return 0.0
    
    try:
        embeddings = model.encode(valid_responses, convert_to_numpy=True)
        
        # Check for NaN or Inf in embeddings
        if not np.all(np.isfinite(embeddings)):
            return 0.0
        
    distances = cosine_distances(embeddings)
    
        n = len(valid_responses)
    upper_triangle_indices = np.triu_indices(n, k=1)
    pairwise_distances = distances[upper_triangle_indices]
    
        if len(pairwise_distances) == 0:
            return 0.0
        
        result = float(np.mean(pairwise_distances))
        return result if np.isfinite(result) else 0.0
    except Exception:
        return 0.0

# ============================================================
# INFERENCE LOGIC
# ============================================================

async def generate_responses_for_prompt(prompt, k, sampling_client, renderer, sampling_params):
    """Generate k responses for a single prompt"""
    # Add /no_think to disable chain-of-thought reasoning for Qwen models
    messages = [{"role": "user", "content": f"{prompt} /no_think"}]
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
    
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    print("="*70)
    print("🚀 STANDARD DPO INFERENCE")
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
    
    # Sample prompts from each creativity bin if requested
    if config.sample_per_bin is not None:
        print(f"🎲 Sampling {config.sample_per_bin} prompts per creativity bin...")
        
        # Define bins (4 bins of 0.25 width each)
        test_df['creativity_bin'] = pd.cut(
            test_df.get('creative_score', test_df.get('alpha', 0.0)),
            bins=[0.0, 0.25, 0.5, 0.75, 1.0],
            labels=['0.00-0.25', '0.25-0.50', '0.50-0.75', '0.75-1.00'],
            include_lowest=True
        )
        
        # Sample from each bin
        sampled_dfs = []
        for bin_label in ['0.00-0.25', '0.25-0.50', '0.50-0.75', '0.75-1.00']:
            bin_df = test_df[test_df['creativity_bin'] == bin_label]
            n_available = len(bin_df)
            n_sample = min(config.sample_per_bin, n_available)
            
            if n_sample > 0:
                sampled = bin_df.sample(n=n_sample, random_state=42)
                sampled_dfs.append(sampled)
                print(f"   Bin {bin_label}: sampled {n_sample}/{n_available} prompts")
            else:
                print(f"   Bin {bin_label}: no prompts available")
        
        test_df = pd.concat(sampled_dfs, ignore_index=True)
        test_df = test_df.drop(columns=['creativity_bin'])
        print(f"✅ Using {len(test_df)} sampled prompts for inference\n")
    
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
    
    # For Qwen models, use the disable_thinking renderer
    renderer_name = get_recommended_renderer_name(base_model)
    
    # Override to disable thinking mode for Qwen3 models
    if "qwen" in base_model.lower() and renderer_name == "qwen3":
        renderer_name = "qwen3_disable_thinking"
    
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    
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
    
    # Parse train dataset from experiment folder
    if "dataset" in experiment_folder:
        train_dataset = experiment_folder.split("_")[0]  # e.g., "dataset1"
    else:
        train_dataset = "unknown"
    
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
            'approach': 'standard_dpo',
            'experiment_folder': experiment_folder,
            'model_path': config.model_path,
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
    
    # Use custom filename if provided, otherwise auto-generate from experiment folder
    if config.output_filename:
        filename = config.output_filename if config.output_filename.endswith('.json') else f"{config.output_filename}.json"
    else:
        filename = f"{experiment_folder}.json"
    
    output_file = os.path.join(config.output_dir, filename)
    
    # Sanitize data for JSON serialization
    sanitized_data = sanitize_for_json(output_data)
    
    with open(output_file, 'w') as f:
        json.dump(sanitized_data, f, indent=2)
    
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

