# ========================================
# BASELINE D: STANDALONE PYTHON SCRIPT
# ========================================

# Set environment variables to avoid warnings/errors
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Imports
import asyncio
import json
import time
import pickle
import re
import tinker
from tinker import types
from tinker_cookbook import renderers
from tinker_cookbook.model_info import get_recommended_renderer_name
from tinker_cookbook.tokenizer_utils import get_tokenizer
from dotenv import load_dotenv

# ========================================
# CONFIGURATION
# ========================================
MODEL = "Qwen/Qwen3-8B"
MAX_CONCURRENT_PROMPTS = 20 # Higher than Baseline B since only 1 call/prompt
OUTPUT_DIR = f"{MODEL}_baseline_d_results"
DATA_FILE = "sampled_data.pkl"  # Path to your pickled data

# ========================================
# HELPER FUNCTIONS
# ========================================

def format_baseline_d_messages(user_prompt):
    """Format prompt for Baseline D - asks for probabilities"""
    instruction = """Give 5 high-quality answers to the prompt below and assign a probability (0–100%) to each answer that sums to 100%.

Format each line as:
Answer: <text> | Prob: <p>%"""
    
    return [
        {"role": "system", "content": instruction},
        {"role": "user", "content": user_prompt}
    ]

def parse_baseline_d_response(text):
    """
    Extract answers and probabilities from Baseline D output
    
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
        answers.append(answer.strip())
        probs.append(float(prob) / 100.0)  # Normalize to 0-1
    
    return answers, probs

async def generate_baseline_d_single(user_prompt, sampling_client, renderer, sampling_params):
    """
    Generate ONE response containing 5 sub-answers with probabilities
    This is a SINGLE API call (unlike Baseline B which makes k=5 calls)
    """
    messages = format_baseline_d_messages(user_prompt)
    model_input = renderer.build_generation_prompt(messages)
    
    # Single API call
    response = await sampling_client.sample_async(
        prompt=model_input,
        num_samples=1,  # Just 1 call per prompt
        sampling_params=sampling_params
    )
    
    # Parse the response
    parsed_message, is_valid = renderer.parse_response(response.sequences[0].tokens)
    raw_text = parsed_message["content"]
    
    # Extract answers and probabilities
    answers, probs = parse_baseline_d_response(raw_text)
    
    return {
        'prompt': user_prompt,
        'raw_output': raw_text,
        'answers': answers,  # List of answer strings
        'probabilities': probs,  # List of probabilities (0-1)
        'baseline': 'D',
        'n_answers': len(answers)
    }

async def run_baseline_d_batch_parallel(prompts, max_concurrent_prompts, sampling_client, renderer, sampling_params):
    """
    Process multiple prompts in parallel for Baseline D
    
    Args:
        prompts: List of prompts to process
        max_concurrent_prompts: How many prompts to process simultaneously
    
    Note: Since each prompt is only 1 API call (not k=5), 
          we can use higher concurrency than Baseline B
    """
    results = []
    
    # Process in batches
    for i in range(0, len(prompts), max_concurrent_prompts):
        batch = prompts[i:i + max_concurrent_prompts]
        
        # Create tasks for all prompts in this batch
        batch_tasks = [
            generate_baseline_d_single(p, sampling_client, renderer, sampling_params) 
            for p in batch
        ]
        
        # Process entire batch in parallel
        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
        
        # Handle any errors
        for j, result in enumerate(batch_results):
            if isinstance(result, Exception):
                print(f"⚠️ Error for prompt: {batch[j][:50]}... - {result}")
                results.append({
                    'prompt': batch[j],
                    'raw_output': "",
                    'answers': [],
                    'probabilities': [],
                    'baseline': 'D',
                    'n_answers': 0,
                    'error': str(result)
                })
            else:
                results.append(result)
        
        print(f"✅ Completed batch {i//max_concurrent_prompts + 1}: {len(batch)} prompts")
        
        # Small delay between batches
        await asyncio.sleep(0.05)
    
    return results

async def process_all_buckets_baseline_d(sampled_data, max_concurrent_prompts, sampling_client, renderer, sampling_params, output_dir):
    """
    Process all buckets with Baseline D (Verbalized Sampling)
    
    Args:
        sampled_data: Dictionary of bucket_name -> DataFrame
        max_concurrent_prompts: Prompts to process simultaneously
        output_dir: Directory to save results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    all_results = {}
    overall_start = time.time()
    
    for bucket_name, bucket_df in sampled_data.items():
        print(f"\n{'='*70}")
        print(f"Processing bucket (Baseline D): {bucket_name}")
        print(f"Alpha mean: {bucket_df['alpha'].mean():.3f}")
        print(f"Number of prompts: {len(bucket_df)}")
        print(f"{'='*70}\n")
        
        prompts = bucket_df['input'].tolist()
        bucket_start = time.time()
        
        # Process this bucket with Baseline D
        results = await run_baseline_d_batch_parallel(
            prompts=prompts,
            max_concurrent_prompts=max_concurrent_prompts,
            sampling_client=sampling_client,
            renderer=renderer,
            sampling_params=sampling_params
        )
        
        bucket_elapsed = time.time() - bucket_start
        
        # Calculate statistics
        successful = [r for r in results if r['n_answers'] > 0]
        avg_answers = sum(r['n_answers'] for r in successful) / len(successful) if successful else 0
        
        # Store results
        all_results[bucket_name] = {
            'results': results,
            'alpha_mean': bucket_df['alpha'].mean(),
            'n_prompts': len(prompts),
            'successful_parses': len(successful),
            'avg_answers_per_prompt': avg_answers,
            'elapsed_time_seconds': bucket_elapsed,
            'time_per_prompt': bucket_elapsed / len(prompts)
        }
        
        # Save bucket results to file
        bucket_str = str(bucket_name).replace(', ', '_').replace('(', '').replace(']', '').replace('.', '_')
        filename = f"{output_dir}/baseline_d_{bucket_str}.json"
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n{'='*70}")
        print(f"✅ Bucket completed!")
        print(f"   Prompts: {len(prompts)}")
        print(f"   Successful parses: {len(successful)}/{len(prompts)}")
        print(f"   Avg answers/prompt: {avg_answers:.1f}")
        print(f"   Time: {bucket_elapsed:.1f} sec ({bucket_elapsed/60:.1f} min)")
        print(f"   Per prompt: {bucket_elapsed/len(prompts):.2f} sec")
        print(f"   💾 Saved: {filename}")
        print(f"{'='*70}")
    
    overall_elapsed = time.time() - overall_start
    
    # Create summary
    summary = {
        'total_buckets': len(all_results),
        'total_prompts': sum(d['n_prompts'] for d in all_results.values()),
        'total_successful': sum(d['successful_parses'] for d in all_results.values()),
        'total_time_seconds': overall_elapsed,
        'total_time_minutes': overall_elapsed / 60,
        'buckets': {
            bucket: {
                'alpha_mean': data['alpha_mean'],
                'n_prompts': data['n_prompts'],
                'successful_parses': data['successful_parses'],
                'avg_answers_per_prompt': data['avg_answers_per_prompt'],
                'time_seconds': data['elapsed_time_seconds'],
                'time_per_prompt': data['time_per_prompt']
            }
            for bucket, data in all_results.items()
        }
    }
    
    # Save summary
    summary_file = f"{output_dir}/baseline_d_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print final summary
    print("\n" + "="*70)
    print("🎉 BASELINE D COMPLETED!")
    print("="*70)
    print(f"📊 Total buckets: {summary['total_buckets']}")
    print(f"📝 Total prompts: {summary['total_prompts']}")
    print(f"✅ Successful parses: {summary['total_successful']}")
    print(f"⏱️  Total time: {summary['total_time_seconds']:.1f} sec ({summary['total_time_minutes']:.1f} min)")
    print(f"⚡ Avg per prompt: {summary['total_time_seconds']/summary['total_prompts']:.2f} sec")
    print(f"💾 Summary: {summary_file}")
    print("="*70 + "\n")
    
    return all_results, summary

# ========================================
# MAIN FUNCTION
# ========================================

async def main():
    """Main execution function"""
    
    # Load environment
    load_dotenv()
    
    # Load sampled data
    print(f"📂 Loading data from {DATA_FILE}...")
    with open(DATA_FILE, 'rb') as f:
        sampled_data = pickle.load(f)
    print(f"✅ Loaded {len(sampled_data)} buckets\n")
    
    # Initialize model
    print("🚀 Initializing model for Baseline D...")
    service_client = tinker.ServiceClient()
    sampling_client = service_client.create_sampling_client(
        base_model=MODEL,
        model_path=None
    )
    tokenizer = get_tokenizer(MODEL)
    renderer = renderers.get_renderer(
        get_recommended_renderer_name(MODEL),
        tokenizer
    )
    # Baseline D uses slightly lower temperature
    sampling_params = types.SamplingParams(
        max_tokens=512,
        temperature=0.9,  # Lower than Baseline B
        top_p=0.95,
        stop=renderer.get_stop_sequences(),
    )
    print("✅ Model initialized!\n")
    
    # Print configuration
    print("🚀 Starting Baseline D (Verbalized Sampling)...")
    print(f"Configuration:")
    print(f"  - Model: {MODEL}")
    print(f"  - Max concurrent prompts: {MAX_CONCURRENT_PROMPTS}")
    print(f"  - Output directory: {OUTPUT_DIR}\n")
    
    # Run the pipeline
    all_results_d, summary_d = await process_all_buckets_baseline_d(
        sampled_data=sampled_data,
        max_concurrent_prompts=MAX_CONCURRENT_PROMPTS,
        sampling_client=sampling_client,
        renderer=renderer,
        sampling_params=sampling_params,
        output_dir=OUTPUT_DIR
    )
    
    print(f"✅ Baseline D complete! Results saved to {OUTPUT_DIR}/")
    
    return all_results_d, summary_d

# ========================================
# ENTRY POINT
# ========================================

if __name__ == "__main__":
    # Run the async main function
    results, summary = asyncio.run(main())