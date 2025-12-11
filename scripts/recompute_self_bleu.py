#!/usr/bin/env python3
"""
Recompute Self-BLEU metrics from existing inference JSON files
without re-running inference.
"""

import json
import os
import re
import numpy as np
from pathlib import Path
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# ============================================================
# CONFIGURATION - Edit this list with your JSON files
# ============================================================

JSON_FILES_TO_FIX = [
    # 2-Adapter DPO results
    "inference_results/2adapter_dpo_dataset1/dataset1_2adapter_uf_full.json",
    "inference_results/2adapter_dpo_dataset1/dataset1_2adapter_tulu_fullsample_per_bin=400.json",
    "inference_results/2adapter_dpo_dataset2/dataset2_2adapter_uf_full.json",
    "inference_results/2adapter_dpo_dataset2/dataset2_2adapter_tulu_fullsample_per_bin=400.json",
    "inference_results/2adapter_dpo_dataset3/dataset3_2adapter_uf_full.json",
    "inference_results/2adapter_dpo_dataset3/dataset3_2adapter_tulu_fullsample_per_bin=400.json",
    
    # Standard DPO results (if needed)
    # "inference_results/new_standard_dpo_shreya/stddpo_d1_uf_results.json",
    # "inference_results/new_standard_dpo_shreya/stddpo_d1_tulu_results.json",
    # ... add more as needed
    
    # Diversity-weighted DPO results (if needed)
    # "inference_results/diversity_weighted_dpo/dataset1_diversity_full_uf.json",
    # ... add more as needed
    
    # Baseline Qwen results
    "inference_results/baseline_qwen/baseline_qwen_uf_full_repeated.json",
    "inference_results/baseline_qwen/baseline_qwen_uf_full_verbalized.json",
    "inference_results/baseline_qwen/baseline_qwen_tulu_full_repeated.json",
    "inference_results/baseline_qwen/baseline_qwen_tulu_full_verbalized.json",
]

# Set to True to overwrite original files, False to create new files with "_fixed" suffix
OVERWRITE_ORIGINAL = False

# ============================================================
# ROBUST SELF-BLEU CALCULATION
# ============================================================

def simple_tokenize(text):
    """Simple tokenization that handles markdown and special chars"""
    # Remove markdown formatting
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # Remove **bold**
    text = re.sub(r'\*([^*]+)\*', r'\1', text)      # Remove *italic*
    text = re.sub(r'#{1,6}\s*', '', text)           # Remove ### headers
    text = re.sub(r'[`~]', '', text)                 # Remove code markers
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)  # Remove links
    # Simple word tokenization
    tokens = re.findall(r'\b\w+\b', text.lower())
    return tokens

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
            continue
    
    # Return mean if we have scores
    if len(bleu_scores) > 0:
        result = float(np.mean(bleu_scores))
        return result if np.isfinite(result) and not np.isnan(result) else 0.0
    
    # Fallback: If BLEU-4 fails, try BLEU-2 (bigram only)
    try:
        smoothing = SmoothingFunction().method4
        bleu_scores = []
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

# ============================================================
# MAIN PROCESSING
# ============================================================

def process_json_file(file_path):
    """Load JSON, recompute Self-BLEU, save updated file"""
    
    if not os.path.exists(file_path):
        print(f"⚠️  File not found: {file_path}")
        return False
    
    print(f"\n📂 Processing: {file_path}")
    
    # Load JSON
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    results = data.get('results', [])
    if not results:
        print(f"   ❌ No results found in file")
        return False
    
    # Count zeros before
    zeros_before = sum(1 for r in results if r.get('self_bleu', 0) == 0.0)
    
    # Recompute Self-BLEU for each result
    updated_count = 0
    for result in tqdm(results, desc="   Recomputing Self-BLEU"):
        responses = result.get('responses', [])
        if responses:
            old_bleu = result.get('self_bleu', 0.0)
            new_bleu = calculate_self_bleu(responses)
            
            # Only update if the value changed
            if old_bleu != new_bleu:
                result['self_bleu'] = new_bleu
                updated_count += 1
    
    # Count zeros after
    zeros_after = sum(1 for r in results if r.get('self_bleu', 0) == 0.0)
    
    # Determine output path
    if OVERWRITE_ORIGINAL:
        output_path = file_path
    else:
        base, ext = os.path.splitext(file_path)
        output_path = f"{base}_fixed{ext}"
    
    # Save updated JSON
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"   ✅ Updated {updated_count}/{len(results)} results")
    print(f"   📊 Zeros: {zeros_before} → {zeros_after}")
    print(f"   💾 Saved: {output_path}")
    
    return True

def main():
    print("="*70)
    print("SELF-BLEU RECOMPUTATION SCRIPT")
    print("="*70)
    print(f"Files to process: {len(JSON_FILES_TO_FIX)}")
    print(f"Overwrite original: {OVERWRITE_ORIGINAL}")
    print("="*70)
    
    success_count = 0
    for file_path in JSON_FILES_TO_FIX:
        if process_json_file(file_path):
            success_count += 1
    
    print("\n" + "="*70)
    print(f"✅ DONE! Processed {success_count}/{len(JSON_FILES_TO_FIX)} files")
    print("="*70)
    
    if not OVERWRITE_ORIGINAL:
        print("\n📌 Note: New files saved with '_fixed' suffix.")
        print("   To use them, either:")
        print("   1. Update INFERENCE_RESULTS paths in visualize_results.py")
        print("   2. Or set OVERWRITE_ORIGINAL=True and re-run this script")

if __name__ == "__main__":
    main()


