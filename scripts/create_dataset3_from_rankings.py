#!/usr/bin/env python3
"""
Create Dataset 3 (LLM Judge Diversity) from Rankings

Takes the LLM judge rankings and creates DPO pairs by:
1. Filtering responses by quality threshold (>= 5)
2. Using LLM diversity rankings to select chosen/rejected
3. Creative prompts: most diverse = chosen, least diverse = rejected
4. Factual prompts: least diverse = chosen, most diverse = rejected

Usage:
  python3 scripts/create_dataset3_from_rankings.py
"""

import pandas as pd
import pickle
from tqdm import tqdm

# ============================================================
# CONFIGURATION
# ============================================================

QUALITY_THRESH = 5  # Same as pairs_creation.ipynb
MIN_VALID_ANSWERS = 3  # Need at least 3 valid responses to create a pair
MIN_DIVERSITY_GAP = 0.0  # Minimum diversity gap (0 = no filtering)

INPUT_FILE = 'scripts/llm_judge_rankings_10k.pkl'
OUTPUT_FILE = 'scripts/dataset3_llm_judge_diversity.parquet'

print("=" * 80)
print("DATASET 3: LLM JUDGE DIVERSITY PAIRS CREATION")
print("=" * 80)
print(f"Quality threshold: {QUALITY_THRESH}")
print(f"Min valid answers: {MIN_VALID_ANSWERS}")
print(f"Min diversity gap: {MIN_DIVERSITY_GAP}")
print()

# ============================================================
# LOAD RANKINGS
# ============================================================

print(f"Loading {INPUT_FILE}...")
with open(INPUT_FILE, 'rb') as f:
    rankings_data = pickle.load(f)

print(f"✓ Loaded {len(rankings_data)} ranked prompts")
print()

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def is_creative_prompt(creative_score, factual_score):
    """Determine if prompt is creative or factual."""
    return creative_score >= factual_score

def create_pair_from_rankings(ranking_item):
    """
    Create a DPO pair from LLM judge rankings.
    
    Args:
        ranking_item: Dict with 'prompt', 'answers', 'scores', 'diversity_rankings'
    
    Returns:
        Dict with pair data, or None if insufficient valid responses
    """
    prompt = ranking_item['prompt']
    all_answers = ranking_item['answers']  # List of 4 answers
    all_scores = ranking_item['scores']    # List of 4 overall_scores
    diversity_rankings = ranking_item['diversity_rankings']  # [1,3,2,4] format (1-based)
    creative_score = ranking_item.get('creative_score', 0.5)
    factual_score = ranking_item.get('factual_score', 0.5)
    
    # Step 1: Filter by quality threshold
    valid_indices = []
    for i in range(4):
        if all_scores[i] >= QUALITY_THRESH:
            valid_indices.append(i)
    
    # Check if we have enough valid responses
    if len(valid_indices) < MIN_VALID_ANSWERS:
        return None
    
    # Step 2: Get rankings for valid responses only
    # diversity_rankings is [1,3,2,4] where position indicates response index (0-based)
    # and value indicates rank (1 = most diverse, 4 = least diverse)
    
    valid_response_rankings = []
    for idx in valid_indices:
        rank = diversity_rankings[idx]  # Get the rank for this response
        valid_response_rankings.append((idx, rank, all_answers[idx], all_scores[idx]))
    
    # Sort by rank (ascending: 1 = most diverse comes first)
    valid_response_rankings.sort(key=lambda x: x[1])
    
    # Step 3: Select chosen and rejected based on prompt type
    is_creative = is_creative_prompt(creative_score, factual_score)
    
    if is_creative:
        # Creative: want diversity
        # Chosen = most diverse (rank 1, first in sorted list)
        # Rejected = least diverse (highest rank, last in sorted list)
        chosen_idx, chosen_rank, chosen_answer, chosen_score = valid_response_rankings[0]
        rejected_idx, rejected_rank, rejected_answer, rejected_score = valid_response_rankings[-1]
    else:
        # Factual: want convergence
        # Chosen = least diverse (highest rank, last in sorted list)
        # Rejected = most diverse (rank 1, first in sorted list)
        chosen_idx, chosen_rank, chosen_answer, chosen_score = valid_response_rankings[-1]
        rejected_idx, rejected_rank, rejected_answer, rejected_score = valid_response_rankings[0]
    
    # Step 4: Check diversity gap (if specified)
    diversity_gap = abs(chosen_rank - rejected_rank)
    if MIN_DIVERSITY_GAP > 0 and diversity_gap < MIN_DIVERSITY_GAP:
        return None
    
    # Step 5: Create pair record
    return {
        'input': prompt,
        'chosen': chosen_answer,
        'rejected': rejected_answer,
        'prompt_label': 'creative' if is_creative else 'factual',
        'factual_score': factual_score,
        'creative_score': creative_score,
        'chosen_overall_score': chosen_score,
        'rejected_overall_score': rejected_score,
        'chosen_diversity_rank': chosen_rank,
        'rejected_diversity_rank': rejected_rank,
        'diversity_gap': diversity_gap,
        'num_valid_responses': len(valid_indices)
    }

# ============================================================
# CREATE PAIRS
# ============================================================

dataset3_pairs = []
skipped_insufficient = 0
skipped_diversity_gap = 0

print("🚀 Creating DPO pairs from LLM judge rankings...")
print()

for ranking_item in tqdm(rankings_data, desc="Creating pairs"):
    pair = create_pair_from_rankings(ranking_item)
    
    if pair is None:
        # Check why it was skipped
        valid_count = sum(1 for s in ranking_item['scores'] if s >= QUALITY_THRESH)
        if valid_count < MIN_VALID_ANSWERS:
            skipped_insufficient += 1
        else:
            skipped_diversity_gap += 1
    else:
        dataset3_pairs.append(pair)

# ============================================================
# SAVE DATASET
# ============================================================

print()
print("=" * 80)
print("CREATING DATAFRAME")
print("=" * 80)

df_dataset3 = pd.DataFrame(dataset3_pairs)

print(f"✓ Created {len(df_dataset3)} DPO pairs")
print()

# Print statistics
print("Dataset Statistics:")
print(f"  Total pairs: {len(df_dataset3)}")
print(f"  Creative prompts: {(df_dataset3['prompt_label'] == 'creative').sum()}")
print(f"  Factual prompts: {(df_dataset3['prompt_label'] == 'factual').sum()}")
print()
print(f"Skipped prompts:")
print(f"  Insufficient valid responses (< {MIN_VALID_ANSWERS}): {skipped_insufficient}")
print(f"  Diversity gap too small (< {MIN_DIVERSITY_GAP}): {skipped_diversity_gap}")
print()

# Save to parquet
df_dataset3.to_parquet(OUTPUT_FILE, index=False)
print(f"✅ Saved Dataset 3 to {OUTPUT_FILE}")
print()

# ============================================================
# FINAL SUMMARY
# ============================================================

print("=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"Input: {len(rankings_data)} ranked prompts")
print(f"Output: {len(df_dataset3)} DPO pairs ({len(df_dataset3)/len(rankings_data)*100:.1f}%)")
print()
print("Columns in dataset:")
for col in df_dataset3.columns:
    print(f"  - {col}")
print()
print("✅ Dataset 3 creation complete!")
print("=" * 80)

