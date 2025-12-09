#!/usr/bin/env python3
"""
Baseline Inference Script for Qwen3-8B (No DPO Training)
Evaluates base Qwen3-8B model using two sampling strategies:
1. Repeated Sampling (Naive): k independent samples with temperature
2. Verbalized Sampling: Single call asking for k diverse responses with probabilities
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import asyncio
import random
import json
import time
import re
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
    """Configuration for baseline Qwen inference"""
    test_dataset: str = "scripts/test_ultrafeedback.parquet"
    output_filename: str  # Required: Base name for output files (without .json)
    k_samples: int = 5
    output_dir: str = "inference_results/baseline_qwen"
    max_concurrent: int = 10
    max_tokens: int = 512
    temperature: float = 1.0
    sample_per_bin: int | None = None  # Number of prompts to sample per creativity bin (None = use all)

# ============================================================
# METRIC FUNCTIONS
# ============================================================

def calculate_self_bleu(responses):
    """Calculate Self-BLEU score (lower = more diverse)"""
    if len(responses) < 2:
        return 0.0
    
    # Filter out empty or invalid responses
    valid_responses = [r for r in responses if r and isinstance(r, str) and r.strip()]
    if len(valid_responses) < 2:
        return 0.0
    
    smoothing = SmoothingFunction().method1
    bleu_scores = []
    
    for i, response in enumerate(valid_responses):
        try:
            # Tokenize hypothesis
            hypothesis = word_tokenize(response.lower())
            if not hypothesis or len(hypothesis) == 0:
                continue
                
            # Tokenize references
            references = []
            for j in range(len(valid_responses)):
                if j != i:
                    ref_tokens = word_tokenize(valid_responses[j].lower())
                    if ref_tokens and len(ref_tokens) > 0:
                        references.append(ref_tokens)
            
            if not references or len(references) == 0:
                continue
            
            # Calculate BLEU with error handling
            bleu = sentence_bleu(
                references, 
                hypothesis, 
                smoothing_function=smoothing,
                weights=(0.25, 0.25, 0.25, 0.25)
            )
            
            # Validate result
            if bleu is not None and np.isfinite(bleu) and not np.isnan(bleu):
                bleu_scores.append(float(bleu))
                
        except Exception as e:
            # Log error but continue (responses might be too long, etc.)
            continue
    
    # Return mean if we have scores, otherwise 0.0
    if len(bleu_scores) > 0:
        result = float(np.mean(bleu_scores))
        return result if np.isfinite(result) and not np.isnan(result) else 0.0
    
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
# REPEATED SAMPLING (NAIVE)
# ============================================================

async def generate_repeated_sampling(prompt, k, sampling_client, renderer, sampling_params):
    """Generate k responses using repeated sampling (k independent API calls)"""
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

# ============================================================
# VERBALIZED SAMPLING
# ============================================================

def format_verbalized_prompt(prompt, k):
    """Format prompt for verbalized sampling with probabilities"""
    system_msg = f"""Give {k} high-quality answers to the prompt below and assign a probability (0–100%) to each answer that sums to 100%.

Format each line as:
Answer: <text> | Prob: <p>%"""
    
    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": f"{prompt} /no_think"}
    ]

def parse_verbalized_response(text, k):
    """
    Extract answers and probabilities from verbalized sampling output
    
    Expected format:
        Answer: Some text here | Prob: 25%
        Answer: Another text | Prob: 30%
        ...
    """
    pattern = r'Answer:\s*(.+?)\s*\|\s*Prob:\s*(\d+(?:\.\d+)?)%'
    matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
    
    answers = []
    probs = []
    
    for answer, prob in matches:
        # Clean up answer (remove leading/trailing whitespace and newlines before next "Answer:")
        cleaned_answer = answer.strip()
        # Remove any trailing text after the answer (before next Answer:)
        cleaned_answer = re.sub(r'\s*Answer:\s*$', '', cleaned_answer, flags=re.IGNORECASE)
        
        answers.append(cleaned_answer)
        probs.append(float(prob) / 100.0)  # Normalize to 0-1
    
    # If we got fewer than k answers, pad with empty responses
    while len(answers) < k:
        answers.append("")
        probs.append(0.0)
    
    # If we got more than k, truncate
    if len(answers) > k:
        answers = answers[:k]
        probs = probs[:k]
    
    return answers, probs

async def generate_verbalized_sampling(prompt, k, sampling_client, renderer, sampling_params):
    """Generate k responses using verbalized sampling (1 API call)"""
    messages = format_verbalized_prompt(prompt, k)
    model_input = renderer.build_generation_prompt(messages)
    
    # Single API call
    response = await sampling_client.sample_async(
        prompt=model_input,
        num_samples=1,
        sampling_params=sampling_params
    )
    
    # Parse the response
    parsed_message, is_valid = renderer.parse_response(response.sequences[0].tokens)
    raw_text = parsed_message["content"]
    
    # Extract answers and probabilities
    answers, probs = parse_verbalized_response(raw_text, k)
    
    return answers, probs, raw_text

# ============================================================
# INFERENCE BATCH PROCESSING
# ============================================================

async def run_inference_batch(prompts_data, k, max_concurrent, sampling_client, renderer, 
                               sampling_params, embedding_model, strategy="repeated"):
    """Process multiple prompts in batches"""
    results = []
    
    for i in tqdm(range(0, len(prompts_data), max_concurrent), desc=f"Inference ({strategy})"):
        batch = prompts_data[i:i + max_concurrent]
        
        if strategy == "repeated":
            # Generate responses for batch (repeated sampling)
            batch_tasks = [
                generate_repeated_sampling(row['instruction'], k, sampling_client, renderer, sampling_params)
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
        
        elif strategy == "verbalized":
            # Generate responses for batch (verbalized sampling)
            batch_tasks = [
                generate_verbalized_sampling(row['instruction'], k, sampling_client, renderer, sampling_params)
                for _, row in batch.iterrows()
            ]
            
            batch_results = await asyncio.gather(*batch_tasks)
            
            # Calculate metrics for each prompt
            for (idx, row), (responses, probs, raw_output) in zip(batch.iterrows(), batch_results):
                result = {
                    'prompt': row['instruction'],
                    'responses': responses,
                    'probabilities': probs,  # Include probabilities for verbalized
                    'raw_output': raw_output,  # Include raw model output
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
    print("🚀 BASELINE QWEN3-8B INFERENCE (NO DPO)")
    print("="*70)
    print(f"Base model: Qwen/Qwen3-8B (no training)")
    print(f"Test dataset: {config.test_dataset}")
    print(f"K samples: {config.k_samples}")
    print(f"Output: {config.output_dir}")
    print(f"Strategies: Repeated + Verbalized")
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
    print("🤖 Initializing base Qwen3-8B model (no DPO)...")
    service_client = tinker.ServiceClient()
    
    base_model = "Qwen/Qwen3-8B"
    
    # Create sampling client with NO model_path (base model only)
    sampling_client = service_client.create_sampling_client(
        base_model=base_model,
        model_path=None
    )
    
    tokenizer = get_tokenizer(base_model)
    
    # Use the disable_thinking renderer
    renderer_name = get_recommended_renderer_name(base_model)
    
    # Override to disable thinking mode for Qwen3 models
    if renderer_name == "qwen3":
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
    
    # ========================================
    # RUN REPEATED SAMPLING
    # ========================================
    print("\n" + "="*70)
    print("📊 STRATEGY 1: REPEATED SAMPLING (NAIVE)")
    print("="*70)
    start_time = time.time()
    
    results_repeated = await run_inference_batch(
        prompts_data=test_df,
        k=config.k_samples,
        max_concurrent=config.max_concurrent,
        sampling_client=sampling_client,
        renderer=renderer,
        sampling_params=sampling_params,
        embedding_model=embedding_model,
        strategy="repeated"
    )
    
    elapsed_repeated = time.time() - start_time
    
    # Compute summary for repeated
    summary_repeated = {
        'total_prompts': len(results_repeated),
        'avg_self_bleu': float(np.mean([r['self_bleu'] for r in results_repeated])),
        'avg_distinct_2': float(np.mean([r['distinct_2'] for r in results_repeated])),
        'avg_semantic_div': float(np.mean([r['semantic_div'] for r in results_repeated])),
        'elapsed_time_seconds': elapsed_repeated,
        'time_per_prompt': elapsed_repeated / len(results_repeated)
    }
    
    print(f"\n✅ Repeated sampling complete!")
    print(f"   Avg Self-BLEU: {summary_repeated['avg_self_bleu']:.4f}")
    print(f"   Avg Distinct-2: {summary_repeated['avg_distinct_2']:.4f}")
    print(f"   Avg Semantic Div: {summary_repeated['avg_semantic_div']:.4f}")
    print(f"   Time: {elapsed_repeated:.1f}s ({summary_repeated['time_per_prompt']:.2f}s/prompt)")
    
    # ========================================
    # RUN VERBALIZED SAMPLING
    # ========================================
    print("\n" + "="*70)
    print("📊 STRATEGY 2: VERBALIZED SAMPLING (WITH PROBABILITIES)")
    print("="*70)
    start_time = time.time()
    
    results_verbalized = await run_inference_batch(
        prompts_data=test_df,
        k=config.k_samples,
        max_concurrent=config.max_concurrent,
        sampling_client=sampling_client,
        renderer=renderer,
        sampling_params=sampling_params,
        embedding_model=embedding_model,
        strategy="verbalized"
    )
    
    elapsed_verbalized = time.time() - start_time
    
    # Compute summary for verbalized
    summary_verbalized = {
        'total_prompts': len(results_verbalized),
        'avg_self_bleu': float(np.mean([r['self_bleu'] for r in results_verbalized])),
        'avg_distinct_2': float(np.mean([r['distinct_2'] for r in results_verbalized])),
        'avg_semantic_div': float(np.mean([r['semantic_div'] for r in results_verbalized])),
        'successful_parses': sum(1 for r in results_verbalized if len([a for a in r['responses'] if a]) >= config.k_samples),
        'elapsed_time_seconds': elapsed_verbalized,
        'time_per_prompt': elapsed_verbalized / len(results_verbalized)
    }
    
    print(f"\n✅ Verbalized sampling complete!")
    print(f"   Avg Self-BLEU: {summary_verbalized['avg_self_bleu']:.4f}")
    print(f"   Avg Distinct-2: {summary_verbalized['avg_distinct_2']:.4f}")
    print(f"   Avg Semantic Div: {summary_verbalized['avg_semantic_div']:.4f}")
    print(f"   Successful parses: {summary_verbalized['successful_parses']}/{len(results_verbalized)}")
    print(f"   Time: {elapsed_verbalized:.1f}s ({summary_verbalized['time_per_prompt']:.2f}s/prompt)")
    
    # ========================================
    # SAVE RESULTS
    # ========================================
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Prepare output filenames
    base_filename = config.output_filename
    if base_filename.endswith('.json'):
        base_filename = base_filename[:-5]
    
    repeated_file = os.path.join(config.output_dir, f"{base_filename}_repeated.json")
    verbalized_file = os.path.join(config.output_dir, f"{base_filename}_verbalized.json")
    
    # Prepare output data for repeated sampling
    output_repeated = {
        'metadata': {
            'approach': 'baseline_qwen_repeated',
            'base_model': base_model,
            'sampling_strategy': 'repeated',
            'test_dataset': config.test_dataset,
            'k_samples': config.k_samples,
            'sample_per_bin': config.sample_per_bin,
            'timestamp': datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            'max_tokens': config.max_tokens,
            'temperature': config.temperature
        },
        'results': results_repeated,
        'summary': summary_repeated
    }
    
    # Prepare output data for verbalized sampling
    output_verbalized = {
        'metadata': {
            'approach': 'baseline_qwen_verbalized',
            'base_model': base_model,
            'sampling_strategy': 'verbalized',
            'test_dataset': config.test_dataset,
            'k_samples': config.k_samples,
            'sample_per_bin': config.sample_per_bin,
            'timestamp': datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            'max_tokens': config.max_tokens,
            'temperature': config.temperature
        },
        'results': results_verbalized,
        'summary': summary_verbalized
    }
    
    # Sanitize and save
    sanitized_repeated = sanitize_for_json(output_repeated)
    sanitized_verbalized = sanitize_for_json(output_verbalized)
    
    with open(repeated_file, 'w') as f:
        json.dump(sanitized_repeated, f, indent=2)
    
    with open(verbalized_file, 'w') as f:
        json.dump(sanitized_verbalized, f, indent=2)
    
    # Print final summary
    print("\n" + "="*70)
    print("✅ BASELINE INFERENCE COMPLETE!")
    print("="*70)
    print(f"📊 Results Summary:")
    print(f"\n   REPEATED SAMPLING:")
    print(f"   - Self-BLEU: {summary_repeated['avg_self_bleu']:.4f}")
    print(f"   - Distinct-2: {summary_repeated['avg_distinct_2']:.4f}")
    print(f"   - Semantic Div: {summary_repeated['avg_semantic_div']:.4f}")
    print(f"   - Time: {elapsed_repeated:.1f}s")
    
    print(f"\n   VERBALIZED SAMPLING:")
    print(f"   - Self-BLEU: {summary_verbalized['avg_self_bleu']:.4f}")
    print(f"   - Distinct-2: {summary_verbalized['avg_distinct_2']:.4f}")
    print(f"   - Semantic Div: {summary_verbalized['avg_semantic_div']:.4f}")
    print(f"   - Successful parses: {summary_verbalized['successful_parses']}/{len(results_verbalized)}")
    print(f"   - Time: {elapsed_verbalized:.1f}s")
    
    print(f"\n💾 Saved:")
    print(f"   - {repeated_file}")
    print(f"   - {verbalized_file}")
    print("="*70 + "\n")

def main(config: InferenceConfig):
    """Synchronous wrapper for async inference"""
    asyncio.run(run_inference(config))

if __name__ == "__main__":
    chz.nested_entrypoint(main)

