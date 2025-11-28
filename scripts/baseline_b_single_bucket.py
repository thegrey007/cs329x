# ========================================
# BASELINE B: SINGLE BUCKET (0.8-1.0)
# ========================================

# Fix OpenMP and tokenizer issues
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Imports
import asyncio
import json
import time
import pickle
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
K_SAMPLES = 5
MAX_CONCURRENT_PROMPTS = 1
OUTPUT_DIR = f"{MODEL}_baseline_b_results"
DATA_FILE = "sampled_data.pkl"  # Path to your pickled data
TARGET_BUCKET = "(0.8, 1.0]"  # Only process this bucket

# ========================================
# HELPER FUNCTIONS
# ========================================

def format_baseline_b_messages(user_prompt):
    """Format prompt for Baseline B"""
    return [{"role": "user", "content": user_prompt}]

async def generate_all_samples_for_prompt(prompt, k, sampling_client, renderer, sampling_params):
    """Generate k samples for ONE prompt (in parallel)"""
    messages = format_baseline_b_messages(prompt)
    model_input = renderer.build_generation_prompt(messages)
    
    # k parallel API calls for this prompt
    tasks = []
    for _ in range(k):
        task = sampling_client.sample_async(
            prompt=model_input,
            num_samples=1,
            sampling_params=sampling_params
        )
        tasks.append(task)
    
    # Get all k responses in parallel
    responses = await asyncio.gather(*tasks)
    
    # Parse responses
    parsed_responses = []
    for response in responses:
        parsed_message, is_valid = renderer.parse_response(response.sequences[0].tokens)
        parsed_responses.append(parsed_message["content"])
    
    return {
        'prompt': prompt,
        'responses': parsed_responses,
        'baseline': 'B',
        'k': k,
        'n_responses': len(parsed_responses)
    }

async def run_baseline_b_batch_parallel(prompts, k, max_concurrent_prompts, sampling_client, renderer, sampling_params):
    """Process multiple prompts in parallel batches"""
    results = []
    
    # Process in batches
    for i in range(0, len(prompts), max_concurrent_prompts):
        batch = prompts[i:i + max_concurrent_prompts]
        
        # Create tasks for all prompts in this batch
        batch_tasks = [
            generate_all_samples_for_prompt(p, k, sampling_client, renderer, sampling_params) 
            for p in batch
        ]
        
        # Process entire batch in parallel
        batch_results = await asyncio.gather(*batch_tasks)
        results.extend(batch_results)
        
        print(f"✅ Completed batch {i//max_concurrent_prompts + 1}: {len(batch)} prompts")
        
        # Small delay between batches
        await asyncio.sleep(0.1)
    
    return results

async def process_all_buckets_baseline_b(sampled_data, k, max_concurrent_prompts, sampling_client, renderer, sampling_params, output_dir):
    """Process all buckets with optimized parallel execution"""
    os.makedirs(output_dir, exist_ok=True)
    
    all_results = {}
    overall_start = time.time()
    
    for bucket_name, bucket_df in sampled_data.items():
        print(f"\n{'='*70}")
        print(f"Processing bucket: {bucket_name}")
        print(f"Alpha mean: {bucket_df['alpha'].mean():.3f}")
        print(f"Number of prompts: {len(bucket_df)}")
        print(f"{'='*70}\n")
        
        prompts = bucket_df['input'].tolist()
        bucket_start = time.time()
        
        # Process this bucket
        results = await run_baseline_b_batch_parallel(
            prompts=prompts,
            k=k,
            max_concurrent_prompts=max_concurrent_prompts,
            sampling_client=sampling_client,
            renderer=renderer,
            sampling_params=sampling_params
        )
        
        bucket_elapsed = time.time() - bucket_start
        
        # Store results
        all_results[bucket_name] = {
            'results': results,
            'alpha_mean': bucket_df['alpha'].mean(),
            'n_prompts': len(prompts),
            'total_generations': len(prompts) * k,
            'elapsed_time_seconds': bucket_elapsed,
            'time_per_prompt': bucket_elapsed / len(prompts)
        }
        
        # Save bucket results to file
        bucket_str = str(bucket_name).replace(', ', '_').replace('(', '').replace(']', '').replace('.', '_')
        filename = f"{output_dir}/baseline_b_{bucket_str}.json"
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n{'='*70}")
        print(f"✅ Bucket completed!")
        print(f"   Prompts: {len(prompts)}")
        print(f"   Generations: {len(prompts) * k}")
        print(f"   Time: {bucket_elapsed:.1f} sec ({bucket_elapsed/60:.1f} min)")
        print(f"   Per prompt: {bucket_elapsed/len(prompts):.2f} sec")
        print(f"   💾 Saved: {filename}")
        print(f"{'='*70}")
    
    overall_elapsed = time.time() - overall_start
    
    # Create summary
    summary = {
        'total_buckets': len(all_results),
        'total_prompts': sum(d['n_prompts'] for d in all_results.values()),
        'total_generations': sum(d['total_generations'] for d in all_results.values()),
        'total_time_seconds': overall_elapsed,
        'total_time_minutes': overall_elapsed / 60,
        'buckets': {
            bucket: {
                'alpha_mean': data['alpha_mean'],
                'n_prompts': data['n_prompts'],
                'total_generations': data['total_generations'],
                'time_seconds': data['elapsed_time_seconds'],
                'time_per_prompt': data['time_per_prompt']
            }
            for bucket, data in all_results.items()
        }
    }
    
    # Save summary
    summary_file = f"{output_dir}/baseline_b_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print final summary
    print("\n" + "="*70)
    print("🎉 ALL BUCKETS COMPLETED!")
    print("="*70)
    print(f"📊 Total buckets: {summary['total_buckets']}")
    print(f"📝 Total prompts: {summary['total_prompts']}")
    print(f"🔄 Total generations: {summary['total_generations']}")
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
    
    # Filter for target bucket only
    print(f"🎯 Filtering for bucket: {TARGET_BUCKET}")
    if TARGET_BUCKET not in sampled_data:
        print(f"❌ Bucket {TARGET_BUCKET} not found!")
        print(f"Available buckets: {list(sampled_data.keys())}")
        return None, None
    
    filtered_data = {TARGET_BUCKET: sampled_data[TARGET_BUCKET]}
    print(f"✅ Found target bucket with {len(filtered_data[TARGET_BUCKET])} prompts\n")
    
    # Initialize model
    print("🚀 Initializing model...")
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
    sampling_params = types.SamplingParams(
        max_tokens=512,
        temperature=1.0,
        top_p=0.95,
        stop=renderer.get_stop_sequences(),
    )
    print("✅ Model initialized!\n")
    
    # Print configuration
    print("🚀 Starting Baseline B processing...")
    print(f"Configuration:")
    print(f"  - Model: {MODEL}")
    print(f"  - Target bucket: {TARGET_BUCKET}")
    print(f"  - k (samples per prompt): {K_SAMPLES}")
    print(f"  - Max concurrent prompts: {MAX_CONCURRENT_PROMPTS}")
    print(f"  - Output directory: {OUTPUT_DIR}\n")
    
    # Run the pipeline
    all_results_b, summary_b = await process_all_buckets_baseline_b(
        sampled_data=filtered_data,
        k=K_SAMPLES,
        max_concurrent_prompts=MAX_CONCURRENT_PROMPTS,
        sampling_client=sampling_client,
        renderer=renderer,
        sampling_params=sampling_params,
        output_dir=OUTPUT_DIR
    )
    
    print(f"✅ Baseline B complete! Results saved to {OUTPUT_DIR}/")
    return all_results_b, summary_b

# ========================================
# ENTRY POINT
# ========================================

if __name__ == "__main__":
    # Run the async main function
    results, summary = asyncio.run(main())

