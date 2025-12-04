#!/usr/bin/env python3
"""
Standalone script to generate Dataset 3 (LLM-as-Judge Diversity) pairs.
Run this after completing BLOCKS 1-9 in pairs_creation.ipynb.

Usage:
  1. In notebook, save filtered_rows:
     ```
     import pickle
     with open('filtered_rows.pkl', 'wb') as f:
         pickle.dump(filtered_rows, f)
     ```
  
  2. Run this script in tmux:
     ```
     tmux new -s llm_judge
     caffeinate -i python scripts/run_llm_judge.py
     ```
  
  3. Detach: Ctrl+B, then D
  4. Reattach: tmux attach -t llm_judge
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
import time
import pickle
import json
import re
import os
from together import Together
from dotenv import load_dotenv

# ============================================================
# LOAD ENVIRONMENT & CONFIGURATION
# ============================================================

load_dotenv()

LLM_JUDGE_MODEL = "Qwen/Qwen2.5-72B-Instruct-Turbo"
together_client = Together()

print("=" * 80)
print("LLM-AS-JUDGE DIVERSITY PAIRS GENERATION")
print("=" * 80)
print(f"Model: {LLM_JUDGE_MODEL}")
print(f"Together API Key loaded: {bool(os.getenv('TOGETHER_API_KEY'))}")
print()

# ============================================================
# LOAD FILTERED ROWS
# ============================================================

print(os.getcwd())
print("Loading filtered_rows.pkl...")
with open('scripts/filtered_rows.pkl', 'rb') as f:
    filtered_rows = pickle.load(f)

print(f"✓ Loaded {len(filtered_rows)} filtered prompts")
print()

# ============================================================
# HELPER FUNCTION
# ============================================================

def is_creative_prompt(row):
    """Determine if prompt is creative or factual."""
    return row['creative_score'] >= row['factual_score']

# ============================================================
# LLM JUDGE FUNCTION
# ============================================================

JUDGE_SYSTEM_PROMPT = """You are an expert judge evaluating response diversity and creativity.

Your task: Given an instruction and multiple responses, rank them by DIVERSITY from MOST to LEAST diverse.

Consider:
- Lexical variety (unique word choices, varied phrasing)
- Semantic originality (novel ideas, unexpected angles)
- Structural diversity (different formats, perspectives)
- Information breadth (covering different aspects)

Do NOT favor longer responses unless they are genuinely more diverse.
Do NOT penalize brief responses if they are creative/diverse.

Output ONLY a JSON object with this exact format:
{
  "rankings": [1, 3, 2, 4],
  "reasoning": "Brief explanation of ranking rationale"
}

The "rankings" array must contain each response index (1-based) ordered from MOST to LEAST diverse."""

def judge_diversity(instruction, valid_answers, valid_indices):
    """
    Call LLM judge to rank responses by diversity.
    Returns: diversity scores (higher = more diverse)
    """
    # Build prompt
    responses_text = "\n\n".join([
        f"Response {i+1}:\n{valid_answers[i]}" 
        for i in range(len(valid_answers))
    ])
    
    user_prompt = f"""Instruction:
{instruction}

{responses_text}

Rank these {len(valid_answers)} responses by diversity (most to least diverse)."""
    
    # Call Together API with retry
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = together_client.chat.completions.create(
                model=LLM_JUDGE_MODEL,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # Parse JSON
            # Try to extract JSON even if wrapped in markdown
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                rankings = result['rankings']
                
                # Convert rankings to diversity scores
                # Rank 1 (first in list) = most diverse = highest score
                diversity_scores = []
                for i in range(len(valid_answers)):
                    response_idx = i + 1  # 1-based
                    rank_position = rankings.index(response_idx)  # 0 = most diverse
                    # Score = inverse of position (0 -> highest, n-1 -> lowest)
                    score = len(valid_answers) - rank_position - 1
                    diversity_scores.append(score)
                
                return diversity_scores
            else:
                raise ValueError("No JSON found in response")
                
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"\n⚠️  Failed to judge prompt after {max_retries} attempts: {e}")
                # Fallback: random scores
                return list(range(len(valid_answers)))
            time.sleep(1)
    
    return list(range(len(valid_answers)))  # Fallback

# ============================================================
# GENERATE DATASET 3 PAIRS
# ============================================================

dataset3_pairs = []
failed_judgments = 0

print(f"🤖 Judging diversity for {len(filtered_rows)} prompts...")
print(f"💰 Estimated cost: ~${len(filtered_rows) * 1800 * 0.60 / 1_000_000:.2f}")
print(f"⏱️  Estimated time: ~{len(filtered_rows) * 2 / 3600:.1f} hours (assuming 2s/prompt)")
print()

start_time = time.time()

for row_idx, row_data in enumerate(tqdm(filtered_rows, desc="LLM Judging")):
    row = row_data['row']
    valid_idxs = row_data['valid_idxs']
    answers = row_data['answers']
    scores = row_data['scores']
    
    # Get valid answers only
    valid_answers = [answers[i] for i in valid_idxs]
    
    # Judge diversity
    llm_diversity_scores = judge_diversity(
        row['instruction'], 
        valid_answers, 
        valid_idxs
    )
    
    # Check if judgment failed (fallback scores)
    if llm_diversity_scores == list(range(len(valid_answers))):
        failed_judgments += 1
    
    # Determine prompt type
    is_creative = is_creative_prompt(row)
    
    # Choose based on LLM diversity scores
    if is_creative:
        chosen_local_idx = np.argmax(llm_diversity_scores)
        rejected_local_idx = np.argmin(llm_diversity_scores)
    else:
        chosen_local_idx = np.argmin(llm_diversity_scores)
        rejected_local_idx = np.argmax(llm_diversity_scores)
    
    # Map back to original indices
    chosen_idx = valid_idxs[chosen_local_idx]
    rejected_idx = valid_idxs[rejected_local_idx]
    
    # Build pair
    pair = {
        'input': row['instruction'],
        'chosen': answers[chosen_idx],
        'rejected': answers[rejected_idx],
        'prompt_label': 'creative' if is_creative else 'factual',
        'factual_score': row['factual_score'],
        'creative_score': row['creative_score'],
        'alpha_bin': row['alpha_bin'],
        'chosen_overall_score': scores[chosen_idx],
        'rejected_overall_score': scores[rejected_idx],
        'chosen_llm_diversity': llm_diversity_scores[chosen_local_idx],
        'rejected_llm_diversity': llm_diversity_scores[rejected_local_idx],
        'num_valid_answers': len(valid_idxs),
        'dataset_type': 'llm_judge_diversity'
    }
    
    dataset3_pairs.append(pair)
    
    # Progress update every 100 prompts
    if (row_idx + 1) % 100 == 0:
        elapsed = time.time() - start_time
        rate = (row_idx + 1) / elapsed
        remaining = (len(filtered_rows) - row_idx - 1) / rate
        print(f"\n  Processed {row_idx + 1}/{len(filtered_rows)} prompts")
        print(f"  Rate: {rate:.2f} prompts/sec")
        print(f"  Estimated time remaining: {remaining/3600:.1f} hours")
        print(f"  Failed judgments so far: {failed_judgments}")
    
    # Small delay to avoid rate limits
    time.sleep(0.1)

# ============================================================
# SAVE RESULTS
# ============================================================

print("\n" + "=" * 80)
print("GENERATION COMPLETE")
print("=" * 80)

dataset3_df = pd.DataFrame(dataset3_pairs)

print(f"✅ Dataset 3 (LLM Judge Diversity): {len(dataset3_pairs)} pairs")
print(f"⚠️  Failed judgments (used fallback): {failed_judgments}")
print(f"⏱️  Total time: {(time.time() - start_time)/3600:.2f} hours")

print("\n📊 Pairs per bin:")
print(dataset3_df['alpha_bin'].value_counts().sort_index())

print("\n📊 Pairs per prompt type:")
print(dataset3_df['prompt_label'].value_counts())

# Save
output_path = "dataset3_llm_judge_diversity.parquet"
dataset3_df.to_parquet(output_path, index=False)

print(f"\n✅ Saved to: {output_path}")
print(f"Columns: {list(dataset3_df.columns)}")
print("\n" + "=" * 80)
print("ALL DONE! 🎉")
print("=" * 80)

