#!/usr/bin/env python3
"""
LLM-as-Judge Rankings Generation Script

Ranks ALL 4 responses for each prompt by diversity using Qwen2.5-72B.
Saves rankings (not pairs) for later filtering and dataset creation.

Usage:
  1. Ensure filtered_rows.pkl exists (from pairs_creation.ipynb)
  2. Run in tmux:
     tmux new -s llm_rankings
     caffeinate -i python3 scripts/run_llm_judge_rankings.py
  3. Detach: Ctrl+B, then D
  4. Reattach: tmux attach -t llm_rankings

Output:
  - llm_judge_rankings_10k.pkl: Successfully ranked prompts
  - llm_judge_failed_10k.pkl: Failed judgments with error details
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
print("LLM-AS-JUDGE RANKINGS GENERATION")
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
    rows_to_process = pickle.load(f)

print(f"✓ Loaded {len(rows_to_process)} filtered prompts")
print()

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

def judge_diversity_all_responses(instruction, all_answers):
    """
    Call LLM judge to rank ALL 4 responses by diversity.
    
    Args:
        instruction: The prompt/instruction
        all_answers: List of 4 answers [ans1, ans2, ans3, ans4]
    
    Returns:
        (success, result_dict)
        - If success=True: result_dict contains {'rankings', 'reasoning', 'raw_output'}
        - If success=False: result_dict contains {'error', 'raw_output'}
    """
    # Build prompt with all 4 responses
    responses_text = "\n\n".join([
        f"Response {i+1}:\n{all_answers[i]}" 
        for i in range(4)
    ])
    
    user_prompt = f"""Instruction:
{instruction}

{responses_text}

Rank these 4 responses by diversity (most to least diverse)."""
    
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
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                rankings = result['rankings']
                reasoning = result.get('reasoning', 'No reasoning provided')
                
                # Validate rankings
                if not isinstance(rankings, list) or len(rankings) != 4:
                    raise ValueError(f"Invalid rankings length: {rankings}")
                if set(rankings) != {1, 2, 3, 4}:
                    raise ValueError(f"Invalid rankings values: {rankings}")
                
                return True, {
                    'rankings': rankings,
                    'reasoning': reasoning,
                    'raw_output': result_text
                }
            else:
                raise ValueError("No JSON found in response")
                
        except Exception as e:
            if attempt == max_retries - 1:
                # Final attempt failed
                return False, {
                    'error': str(e),
                    'raw_output': result_text if 'result_text' in locals() else None
                }
            time.sleep(1)
    
    # Should never reach here
    return False, {'error': 'Unknown error'}

# ============================================================
# PROCESS ALL PROMPTS
# ============================================================

successful_rankings = []
failed_judgments = []

print(f"🚀 Starting LLM judging for {len(rows_to_process)} prompts...")
print(f"⏱️  Estimated time: ~{len(rows_to_process) * 2 / 60:.1f} minutes (assuming ~2s per prompt)")
print()

for i, row_data in enumerate(tqdm(rows_to_process, desc="LLM Judging")):
    # Extract data from the nested structure
    row = row_data['row']
    prompt = row['instruction']
    all_answers = row_data['answers']  # All 4 answers
    all_scores = row_data['scores']    # All 4 overall_scores
    
    # Call LLM judge
    success, result = judge_diversity_all_responses(prompt, all_answers)
    
    if success:
        # Store successful ranking
        successful_rankings.append({
            'prompt': prompt,
            'answers': all_answers,
            'scores': all_scores,
            'diversity_rankings': result['rankings'],  # [1,3,2,4] format
            'reasoning': result['reasoning'],
            'raw_llm_output': result['raw_output'],
            'creative_score': row.get('creative_score'),
            'factual_score': row.get('factual_score')
        })
    else:
        # Store failed judgment
        failed_judgments.append({
            'prompt': prompt,
            'answers': all_answers,
            'scores': all_scores,
            'error': result['error'],
            'raw_llm_output': result.get('raw_output'),
            'creative_score': row.get('creative_score'),
            'factual_score': row.get('factual_score')
        })
    
    # Progress update every 100 prompts
    if (i + 1) % 100 == 0:
        print(f"\n✓ Processed {i+1}/{len(rows_to_process)} prompts")
        print(f"  Success: {len(successful_rankings)} | Failed: {len(failed_judgments)}")
    
    # Rate limiting: small delay between requests
    time.sleep(0.5)

# ============================================================
# SAVE RESULTS
# ============================================================

print()
print("=" * 80)
print("SAVING RESULTS")
print("=" * 80)

# Save successful rankings
success_path = 'scripts/llm_judge_rankings_10k.pkl'
with open(success_path, 'wb') as f:
    pickle.dump(successful_rankings, f)
print(f"✓ Saved {len(successful_rankings)} successful rankings to {success_path}")

# Save failed judgments
failed_path = 'scripts/llm_judge_failed_10k.pkl'
with open(failed_path, 'wb') as f:
    pickle.dump(failed_judgments, f)
print(f"✓ Saved {len(failed_judgments)} failed judgments to {failed_path}")

# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"Total prompts processed: {len(rows_to_process)}")
print(f"Successful rankings: {len(successful_rankings)} ({len(successful_rankings)/len(rows_to_process)*100:.1f}%)")
print(f"Failed judgments: {len(failed_judgments)} ({len(failed_judgments)/len(rows_to_process)*100:.1f}%)")
print()
print("✅ Rankings generation complete!")
print()
print("Next steps:")
print("1. Review failed judgments in llm_judge_failed_10k.pkl")
print("2. Use llm_judge_rankings_10k.pkl to create Dataset 3 with filtering")
print("=" * 80)

