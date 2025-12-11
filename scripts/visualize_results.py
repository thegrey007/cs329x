#!/usr/bin/env python3
"""
Visualization Script for DPO Inference Results
Compares diversity metrics across approaches, datasets, and evaluation sets

Research Questions:
- RQ1: For a given training dataset, how do approaches compare?
- RQ2: For a given approach, how do training datasets compare?
- RQ3: Does it generalize from in-domain (UF) to out-of-domain (Tulu)?
"""

import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional

# ============================================================
# CONFIGURATION - EDIT THIS SECTION
# ============================================================

# Map of (training_dataset, approach, eval_dataset) to inference result JSON file
# training_dataset: 1=Embedding Diversity, 2=Quality Score (Baseline), 3=LLM-as-Judge
# approach: "standard_dpo", "2adapter_dpo", "diversity_weighted_dpo"
# eval_dataset: "uf" (UltraFeedback in-domain), "tulu" (Tulu out-of-domain)

INFERENCE_RESULTS = {
    # ========== STANDARD DPO ==========
    # Dataset 1 (Embedding Diversity)
    (1, "standard_dpo", "uf"): "inference_results/new_standard_dpo_shreya/stddpo_d1_uf_results.json",
    (1, "standard_dpo", "tulu"): "inference_results/new_standard_dpo_shreya/stddpo_d1_tulu_results.json",
    # Dataset 2 (Quality Score Baseline)
    (2, "standard_dpo", "uf"): "inference_results/new_standard_dpo_shreya/stddpo_d2_uf_results.json",
    (2, "standard_dpo", "tulu"): "inference_results/new_standard_dpo_shreya/stddpo_d2_tulu_results.json",
    # Dataset 3 (LLM-as-Judge)
    (3, "standard_dpo", "uf"): "inference_results/new_standard_dpo_shreya/stddpo_d3_uf_results.json",
    (3, "standard_dpo", "tulu"): "inference_results/new_standard_dpo_shreya/stddpo_d3_tulu_results.json",
    
    # ========== 2-ADAPTER DPO ==========
    # Dataset 1
    (1, "2adapter_dpo", "uf"): "inference_results/2adapter_dpo_dataset1/dataset1_2adapter_uf_full_fixed.json",
    (1, "2adapter_dpo", "tulu"): "inference_results/2adapter_dpo_dataset1/dataset1_2adapter_tulu_fullsample_per_bin=400_fixed.json",
    # Dataset 2
    (2, "2adapter_dpo", "uf"): "inference_results/2adapter_dpo_dataset2/dataset2_2adapter_uf_full_fixed.json",
    (2, "2adapter_dpo", "tulu"): "inference_results/2adapter_dpo_dataset2/dataset2_2adapter_tulu_fullsample_per_bin=400_fixed.json",
    # Dataset 3
    (3, "2adapter_dpo", "uf"): "inference_results/2adapter_dpo_dataset3/dataset3_2adapter_uf_full_fixed.json",
    (3, "2adapter_dpo", "tulu"): "inference_results/2adapter_dpo_dataset3/dataset3_2adapter_tulu_fullsample_per_bin=400_fixed.json",
    
    # ========== DIVERSITY-WEIGHTED DPO ==========
    # Dataset 1
    (1, "diversity_weighted_dpo", "uf"): "inference_results/diversity_weighted_dpo/dataset1_diversity_full_uf.json",
    (1, "diversity_weighted_dpo", "tulu"): "inference_results/diversity_weighted_dpo/dataset1_diversity_tulu_full.json",
    # Dataset 2
    (2, "diversity_weighted_dpo", "uf"): "inference_results/diversity_weighted_dpo/dataset2_diversity_full_uf.json",
    (2, "diversity_weighted_dpo", "tulu"): "inference_results/diversity_weighted_dpo/dataset2_diversity_tulu_full.json",
    # Dataset 3
    (3, "diversity_weighted_dpo", "uf"): "inference_results/diversity_weighted_dpo/dataset3_diversity_full_uf.json",
    (3, "diversity_weighted_dpo", "tulu"): "inference_results/diversity_weighted_dpo/dataset3_diversity_tulu_full.json",
}

# Base model results (no training - same model for all)
BASELINE_RESULTS = {
    ("baseline_repeated", "uf"): "inference_results/baseline_qwen/baseline_qwen_uf_full_repeated_fixed.json",
    ("baseline_repeated", "tulu"): "inference_results/baseline_qwen/baseline_qwen_tulu_full_repeated_fixed.json",
    ("baseline_verbalized", "uf"): "inference_results/baseline_qwen/baseline_qwen_uf_full_verbalized_fixed.json",
    ("baseline_verbalized", "tulu"): "inference_results/baseline_qwen/baseline_qwen_tulu_full_verbalized_fixed.json",
}

# Visualization settings
CREATIVITY_BIN_SIZE = 0.10  # Creates 4 bins: [0-0.25, 0.25-0.5, 0.5-0.75, 0.75-1.0]
OUTPUT_DIR = "visualizations"
FIGURE_DPI = 300

# ============================================================
# LABELS
# ============================================================

DATASET_LABELS = {
    1: "Embedding Diversity",
    2: "Quality Score (Baseline)",
    3: "LLM-as-Judge Diversity"
}

APPROACH_LABELS = {
    "standard_dpo": "Standard DPO (Baseline)",
    "2adapter_dpo": "2-Adapter DPO",
    "diversity_weighted_dpo": "Diversity-Weighted DPO",
    "baseline_repeated": "Repeated Sampling",
    "baseline_verbalized": "Verbalized Sampling",
    "baseline_qwen": "Pretrained (Baseline)"
}

APPROACH_COLORS = {
    "standard_dpo": "#2563eb",       # Blue (same for all DPO)
    "2adapter_dpo": "#2563eb",       # Blue (same for all DPO)
    "diversity_weighted_dpo": "#2563eb",  # Blue (same for all DPO)
    "baseline_repeated": "#9ca3af",  # Light gray (lighter)
    "baseline_verbalized": "#1f2937"  # Dark gray / charcoal
}

EVAL_LABELS = {
    "uf": "UltraFeedback (In-Domain)",
    "tulu": "Tulu (Out-of-Domain)"
}

METRIC_LABELS = {
    "self_bleu": "Self-BLEU",
    "distinct_2": "Distinct-2",
    "semantic_div": "Semantic Diversity"
}

METRIC_LABELS_SHORT = {
    "self_bleu": "Self-BLEU",
    "distinct_2": "Distinct-2",
    "semantic_div": "Semantic Div"
}

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_inference_results(file_path: str) -> Dict:
    """Load inference results from JSON file"""
    with open(file_path, 'r') as f:
        return json.load(f)

def bin_by_creativity(results: List[Dict], bin_size: float = 0.25) -> Dict[str, List[Dict]]:
    """Bin results by creativity score (alpha)"""
    bins = {}
    num_bins = int(1.0 / bin_size)
    
    for i in range(num_bins):
        bin_start = i * bin_size
        bin_end = (i + 1) * bin_size
        bin_key = f"{bin_start:.2f}-{bin_end:.2f}"
        bins[bin_key] = []
    
    for result in results:
        alpha = result.get('alpha', 0.0)
        bin_idx = min(int(alpha / bin_size), num_bins - 1)
        bin_start = bin_idx * bin_size
        bin_end = (bin_idx + 1) * bin_size
        bin_key = f"{bin_start:.2f}-{bin_end:.2f}"
        bins[bin_key].append(result)
    
    return bins

def compute_bin_statistics(binned_results: Dict[str, List[Dict]], metrics: List[str]) -> pd.DataFrame:
    """Compute mean metrics for each bin"""
    data = []
    
    for bin_key, results in binned_results.items():
        if not results:
            continue
        
        row = {'bin': bin_key}
        for metric in metrics:
            values = [r[metric] for r in results if metric in r and r[metric] is not None]
            row[metric] = np.mean(values) if values else np.nan
            row[f'{metric}_std'] = np.std(values) if values else np.nan
            row[f'{metric}_count'] = len(values)
        
        data.append(row)
    
    df = pd.DataFrame(data)
    # Sort by bin order
    df['bin_order'] = df['bin'].apply(lambda x: float(x.split('-')[0]))
    df = df.sort_values('bin_order').reset_index(drop=True)
    return df

def load_all_data() -> Tuple[Dict, Dict]:
    """Load and process all inference results"""
    all_data = {}
    baseline_data = {}
    metrics = ['self_bleu', 'distinct_2', 'semantic_div']
    
    print("\n" + "="*70)
    print("LOADING INFERENCE RESULTS")
    print("="*70)
    
    # Load DPO results
    for (dataset, approach, eval_set), file_path in INFERENCE_RESULTS.items():
        if file_path is None:
            continue
        
        if not os.path.exists(file_path):
            print(f"⚠️  Skipping ({dataset}, {approach}, {eval_set}): File not found")
            continue
        
        try:
            print(f"📂 Loading ({DATASET_LABELS[dataset][:10]}, {approach}, {eval_set})")
            data = load_inference_results(file_path)
            results = data.get('results', [])
            
            binned = bin_by_creativity(results, CREATIVITY_BIN_SIZE)
            stats = compute_bin_statistics(binned, metrics)
            
            all_data[(dataset, approach, eval_set)] = stats
            print(f"   ✅ {len(results)} results, {len(stats)} bins")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Load baseline results
    for (approach, eval_set), file_path in BASELINE_RESULTS.items():
        if file_path is None:
            continue
        
        if not os.path.exists(file_path):
            print(f"⚠️  Skipping baseline ({approach}, {eval_set}): File not found")
            continue
        
        try:
            print(f"📂 Loading baseline ({approach}, {eval_set})")
            data = load_inference_results(file_path)
            results = data.get('results', [])
            
            binned = bin_by_creativity(results, CREATIVITY_BIN_SIZE)
            stats = compute_bin_statistics(binned, metrics)
            
            baseline_data[(approach, eval_set)] = stats
            print(f"   ✅ {len(results)} results, {len(stats)} bins")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print(f"\n✅ Loaded {len(all_data)} DPO combinations + {len(baseline_data)} baselines")
    print("="*70 + "\n")
    
    return all_data, baseline_data

# ============================================================
# MAIN VISUALIZATION: SMART FACET GRID
# ============================================================

def plot_smart_facet_grid(all_data: Dict, baseline_data: Dict, metric: str, output_dir: str):
    """
    Create the smart facet grid:
    - Rows: Approaches (Standard DPO, 2-Adapter DPO, Diversity-Weighted DPO)
    - Columns: Training Datasets (Embedding Diversity, Quality Score, LLM-as-Judge)
    - Each cell: 2 lines (solid=UF, dashed=Tulu)
    - Gray reference lines: Base model results (same in every cell)
    """
    print(f"\n📊 Generating Smart Facet Grid for {metric}...")
    
    approaches = ['standard_dpo', '2adapter_dpo', 'diversity_weighted_dpo']
    datasets = [2, 1, 3]
    
    fig, axes = plt.subplots(len(approaches), len(datasets), figsize=(14, 10))
    fig.suptitle(f'{METRIC_LABELS[metric]}', fontsize=16, fontweight='bold', y=0.98)
    
    # Get bin labels from any available data
    sample_df = None
    for key, df in all_data.items():
        if df is not None and len(df) > 0:
            sample_df = df
            break
    
    if sample_df is None:
        print("   ❌ No data available")
        return
    
    bin_labels_full = sample_df['bin'].values
    # Create simpler tick labels (just the start value: 0, 0.1, 0.2, etc.)
    bin_labels = [b.split('-')[0].lstrip('0').replace('.', '') if b.split('-')[0] != '0.00' else '0' for b in bin_labels_full]
    # Better formatting: show as decimals
    bin_labels = [f"{float(b.split('-')[0]):.1f}" for b in bin_labels_full]
    x = np.arange(len(bin_labels))
    
    for row_idx, approach in enumerate(approaches):
        for col_idx, dataset in enumerate(datasets):
            ax = axes[row_idx, col_idx]
            
            # Plot baseline reference lines (distinct colors, same in every cell)
            for baseline_approach in ['baseline_repeated', 'baseline_verbalized']:
                for eval_set, linestyle in [('uf', '-'), ('tulu', '--')]:
                    if (baseline_approach, eval_set) in baseline_data:
                        df = baseline_data[(baseline_approach, eval_set)]
                        if metric in df.columns:
                            y = df[metric].values
                            label = f"{APPROACH_LABELS[baseline_approach]}" if eval_set == 'uf' else None
                            ax.plot(x, y, linestyle=linestyle, linewidth=2, 
                                   color=APPROACH_COLORS[baseline_approach],
                                   alpha=0.8, label=label)
            
            # Plot DPO results
            for eval_set, linestyle, linewidth in [('uf', '-', 2.5), ('tulu', '--', 2.5)]:
                if (dataset, approach, eval_set) in all_data:
                    df = all_data[(dataset, approach, eval_set)]
                    if metric in df.columns:
                        y = df[metric].values
                        color = APPROACH_COLORS[approach]
                        label = f"{EVAL_LABELS[eval_set][:4]}" if row_idx == 0 and col_idx == 0 else None
                        ax.plot(x, y, linestyle=linestyle, linewidth=linewidth,
                               color=color, marker='o', markersize=5)
            
            # Formatting
            ax.set_xticks(x)
            if row_idx == len(approaches) - 1:
                ax.set_xticklabels(bin_labels, rotation=0, fontsize=9)
                ax.set_xlabel('Creativity Bin', fontsize=10)
            else:
                ax.set_xticklabels([])
            
            if col_idx == 0:
                ax.set_ylabel(f'{APPROACH_LABELS[approach]}', fontsize=10, fontweight='bold')
            
            if row_idx == 0:
                ax.set_title(f'{DATASET_LABELS[dataset]}', fontsize=11, fontweight='bold')
            
            ax.grid(True, alpha=0.3)
            ax.axvline(x=1.5, color='gray', linestyle=':', alpha=0.5, linewidth=1)
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='black', linestyle='-', linewidth=2, label='In-Domain (UF)'),
        Line2D([0], [0], color='black', linestyle='--', linewidth=2, label='Out-of-Domain (Tulu)'),
        Line2D([0], [0], color=APPROACH_COLORS['baseline_repeated'], linestyle='-', 
               linewidth=2, label='Repeated Sampling'),
        Line2D([0], [0], color=APPROACH_COLORS['baseline_verbalized'], linestyle='-', 
               linewidth=2, label='Verbalized Sampling'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize=10, 
               bbox_to_anchor=(0.5, 0.01))
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    
    output_subdir = os.path.join(output_dir, "smart_facet_grid")
    os.makedirs(output_subdir, exist_ok=True)
    output_file = os.path.join(output_subdir, f'{metric}_facet_grid.png')
    plt.savefig(output_file, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: {output_file}")

# ============================================================
# APPROACH COMPARISON FACET GRID
# ============================================================

def plot_approach_comparison_grid(all_data: Dict, baseline_data: Dict, metric: str, output_dir: str):
    """
    Create approach comparison facet grid:
    - Rows: Evaluation datasets (In-Domain UF, Out-of-Domain Tulu)
    - Columns: Training Datasets (Embedding Diversity, Quality Score, LLM-as-Judge)
    - Each cell: 3 lines (one per DPO approach) with different colors
    - Gray reference lines: Base model results (same in every cell)
    """
    print(f"\n📊 Generating Approach Comparison Grid for {metric}...")
    
    eval_sets = ['uf', 'tulu']
    datasets = [2, 1, 3]
    approaches = ['standard_dpo', '2adapter_dpo', 'diversity_weighted_dpo']
    
    # Different colors for each approach in this view
    approach_colors_compare = {
        'standard_dpo': '#3b82f6',       # Blue
        '2adapter_dpo': '#ef4444',       # Red
        'diversity_weighted_dpo': '#22c55e'  # Green
    }
    
    fig, axes = plt.subplots(len(eval_sets), len(datasets), figsize=(14, 8))
    fig.suptitle(f'{METRIC_LABELS[metric]}', fontsize=16, fontweight='bold', y=0.98)
    
    # Get bin labels from any available data
    sample_df = None
    for key, df in all_data.items():
        if df is not None and len(df) > 0:
            sample_df = df
            break
    
    if sample_df is None:
        print("   ❌ No data available")
        return
    
    bin_labels_full = sample_df['bin'].values
    bin_labels = [f"{float(b.split('-')[0]):.1f}" for b in bin_labels_full]
    x = np.arange(len(bin_labels))
    
    eval_labels_short = {'uf': 'In-Domain (UF)', 'tulu': 'Out-of-Domain (Tulu)'}
    
    for row_idx, eval_set in enumerate(eval_sets):
        for col_idx, dataset in enumerate(datasets):
            ax = axes[row_idx, col_idx]
            
            # Plot baseline reference lines (same in every cell)
            for baseline_approach in ['baseline_repeated', 'baseline_verbalized']:
                if (baseline_approach, eval_set) in baseline_data:
                    df = baseline_data[(baseline_approach, eval_set)]
                    if metric in df.columns:
                        y = df[metric].values
                        ax.plot(x, y, linestyle='-', linewidth=2, 
                               color=APPROACH_COLORS[baseline_approach],
                               alpha=0.7)
            
            # Plot all 3 DPO approaches
            for approach in approaches:
                if (dataset, approach, eval_set) in all_data:
                    df = all_data[(dataset, approach, eval_set)]
                    if metric in df.columns:
                        y = df[metric].values
                        ax.plot(x, y, linestyle='-', linewidth=2.5,
                               color=approach_colors_compare[approach],
                               marker='o', markersize=4,
                               label=APPROACH_LABELS[approach] if row_idx == 0 and col_idx == 0 else None)
            
            # Formatting
            ax.set_xticks(x)
            if row_idx == len(eval_sets) - 1:
                ax.set_xticklabels(bin_labels, rotation=0, fontsize=9)
                ax.set_xlabel('Creativity Bin', fontsize=10)
            else:
                ax.set_xticklabels([])
            
            if col_idx == 0:
                ax.set_ylabel(f'{eval_labels_short[eval_set]}', fontsize=10, fontweight='bold')
            
            if row_idx == 0:
                ax.set_title(f'{DATASET_LABELS[dataset]}', fontsize=11, fontweight='bold')
            
            ax.grid(True, alpha=0.3)
            # Add vertical line at creativity = 0.5
            mid_idx = len(bin_labels) // 2 - 0.5
            ax.axvline(x=mid_idx, color='gray', linestyle=':', alpha=0.5, linewidth=1)
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=approach_colors_compare['standard_dpo'], linestyle='-', 
               linewidth=2, marker='o', markersize=4, label='Standard DPO'),
        Line2D([0], [0], color=approach_colors_compare['2adapter_dpo'], linestyle='-', 
               linewidth=2, marker='o', markersize=4, label='2-Adapter DPO'),
        Line2D([0], [0], color=approach_colors_compare['diversity_weighted_dpo'], linestyle='-', 
               linewidth=2, marker='o', markersize=4, label='Diversity-Weighted DPO'),
        Line2D([0], [0], color=APPROACH_COLORS['baseline_repeated'], linestyle='-', 
               linewidth=2, label='Repeated Sampling'),
        Line2D([0], [0], color=APPROACH_COLORS['baseline_verbalized'], linestyle='-', 
               linewidth=2, label='Verbalized Sampling'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=5, fontsize=9, 
               bbox_to_anchor=(0.5, 0.01))
    
    plt.tight_layout(rect=[0, 0.06, 1, 0.96])
    
    output_subdir = os.path.join(output_dir, "approach_comparison")
    os.makedirs(output_subdir, exist_ok=True)
    output_file = os.path.join(output_subdir, f'{metric}_approach_comparison.png')
    plt.savefig(output_file, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: {output_file}")

# ============================================================
# DATASET COMPARISON FACET GRID
# ============================================================

def plot_dataset_comparison_grid(all_data: Dict, metric: str, output_dir: str):
    """
    Create dataset comparison facet grid (inverse of approach comparison):
    - Rows: Evaluation datasets (In-Domain UF, Out-of-Domain Tulu)
    - Columns: Approaches (Standard DPO, 2-Adapter DPO, Diversity-Weighted DPO)
    - Each cell: 3 lines (one per training dataset) with different colors
    - NO baseline reference lines
    """
    print(f"\n📊 Generating Dataset Comparison Grid for {metric}...")
    
    eval_sets = ['uf', 'tulu']
    approaches = ['standard_dpo', '2adapter_dpo', 'diversity_weighted_dpo']
    datasets = [2, 1, 3]
    
    # Different colors for each dataset
    dataset_colors = {
        1: '#e74c3c',  # Red - Embedding Diversity
        2: '#3498db',  # Blue - Quality Score
        3: '#2ecc71'   # Green - LLM-as-Judge
    }
    
    fig, axes = plt.subplots(len(eval_sets), len(approaches), figsize=(14, 8))
    fig.suptitle(f'{METRIC_LABELS[metric]}', fontsize=16, fontweight='bold', y=0.98)
    
    # Get bin labels from any available data
    sample_df = None
    for key, df in all_data.items():
        if df is not None and len(df) > 0:
            sample_df = df
            break
    
    if sample_df is None:
        print("   ❌ No data available")
        return
    
    bin_labels_full = sample_df['bin'].values
    bin_labels = [f"{float(b.split('-')[0]):.1f}" for b in bin_labels_full]
    x = np.arange(len(bin_labels))
    
    eval_labels_short = {'uf': 'In-Domain (UF)', 'tulu': 'Out-of-Domain (Tulu)'}
    
    for row_idx, eval_set in enumerate(eval_sets):
        for col_idx, approach in enumerate(approaches):
            ax = axes[row_idx, col_idx]
            
            # Plot all 3 datasets
            for dataset in datasets:
                if (dataset, approach, eval_set) in all_data:
                    df = all_data[(dataset, approach, eval_set)]
                    if metric in df.columns:
                        y = df[metric].values
                        ax.plot(x, y, linestyle='-', linewidth=2.5,
                               color=dataset_colors[dataset],
                               marker='o', markersize=4,
                               label=DATASET_LABELS[dataset] if row_idx == 0 and col_idx == 0 else None)
            
            # Formatting
            ax.set_xticks(x)
            if row_idx == len(eval_sets) - 1:
                ax.set_xticklabels(bin_labels, rotation=0, fontsize=9)
                ax.set_xlabel('Creativity Bin', fontsize=10)
            else:
                ax.set_xticklabels([])
            
            if col_idx == 0:
                ax.set_ylabel(f'{eval_labels_short[eval_set]}', fontsize=10, fontweight='bold')
            
            if row_idx == 0:
                ax.set_title(f'{APPROACH_LABELS[approach]}', fontsize=11, fontweight='bold')
            
            ax.grid(True, alpha=0.3)
            # Add vertical line at creativity = 0.5
            mid_idx = len(bin_labels) // 2 - 0.5
            ax.axvline(x=mid_idx, color='gray', linestyle=':', alpha=0.5, linewidth=1)
    
    # Add legend (order: Quality Score first, then Embedding Diversity, then LLM-as-Judge)
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=dataset_colors[2], linestyle='-', 
               linewidth=2, marker='o', markersize=4, label=DATASET_LABELS[2]),
        Line2D([0], [0], color=dataset_colors[1], linestyle='-', 
               linewidth=2, marker='o', markersize=4, label=DATASET_LABELS[1]),
        Line2D([0], [0], color=dataset_colors[3], linestyle='-', 
               linewidth=2, marker='o', markersize=4, label=DATASET_LABELS[3]),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=10, 
               bbox_to_anchor=(0.5, 0.01))
    
    plt.tight_layout(rect=[0, 0.06, 1, 0.96])
    
    output_subdir = os.path.join(output_dir, "dataset_comparison")
    os.makedirs(output_subdir, exist_ok=True)
    output_file = os.path.join(output_subdir, f'{metric}_dataset_comparison.png')
    plt.savefig(output_file, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: {output_file}")

# ============================================================
# MODULATION DELTA SUMMARY
# ============================================================

def plot_modulation_delta(all_data: Dict, baseline_data: Dict, output_dir: str):
    """
    Bar chart showing modulation delta (High Creativity - Low Creativity)
    for all approaches and datasets
    """
    print("\n📊 Generating Modulation Delta Summary...")
    
    metrics = ['self_bleu', 'distinct_2', 'semantic_div']
    output_subdir = os.path.join(output_dir, "modulation_delta")
    os.makedirs(output_subdir, exist_ok=True)
    
    for metric in metrics:
        delta_data = []
        
        # DPO approaches
        for (dataset, approach, eval_set), df in all_data.items():
            if eval_set != 'uf':  # Only show UF for simplicity
                continue
            
            bins = df['bin'].values
            values = df[metric].values
            
            # Low creativity: bins 0-1 (0.0-0.5)
            # High creativity: bins 2-3 (0.5-1.0)
            low_indices = [i for i, b in enumerate(bins) if b.startswith('0.0') or b.startswith('0.2')]
            high_indices = [i for i, b in enumerate(bins) if b.startswith('0.5') or b.startswith('0.7')]
            
            if low_indices and high_indices:
                low_avg = np.mean([values[i] for i in low_indices])
                high_avg = np.mean([values[i] for i in high_indices])
                delta = high_avg - low_avg
                
                delta_data.append({
                    'approach': APPROACH_LABELS[approach],
                    'dataset': DATASET_LABELS[dataset],
                    'delta': delta,
                    'is_baseline': False
                })
        
        # Baseline approaches
        for (approach, eval_set), df in baseline_data.items():
            if eval_set != 'uf':
                continue
            
            bins = df['bin'].values
            values = df[metric].values
            
            low_indices = [i for i, b in enumerate(bins) if b.startswith('0.0') or b.startswith('0.2')]
            high_indices = [i for i, b in enumerate(bins) if b.startswith('0.5') or b.startswith('0.7')]
            
            if low_indices and high_indices:
                low_avg = np.mean([values[i] for i in low_indices])
                high_avg = np.mean([values[i] for i in high_indices])
                delta = high_avg - low_avg
                
                delta_data.append({
                    'approach': APPROACH_LABELS[approach],
                    'dataset': 'Base Model',
                    'delta': delta,
                    'is_baseline': True
                })
        
        if not delta_data:
            continue
        
        # Create plot
        df_delta = pd.DataFrame(delta_data)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Group by approach
        approaches_order = ['Standard DPO', '2-Adapter DPO', 'Diversity-Weighted DPO', 
                          'Repeated Sampling', 'Verbalized Sampling']
        datasets_order = ['Quality Score (Baseline)', 'Embedding Diversity', 
                         'LLM-as-Judge Diversity', 'Base Model']
        
        x_positions = []
        x_labels = []
        colors = []
        bar_values = []
        
        pos = 0
        for approach in approaches_order:
            approach_data = df_delta[df_delta['approach'] == approach]
            if approach_data.empty:
                continue
            
            for _, row in approach_data.iterrows():
                x_positions.append(pos)
                x_labels.append(f"{approach[:8]}...\n({row['dataset'][:8]})")
                bar_values.append(row['delta'])
                
                # Color based on approach
                if 'Standard' in approach:
                    colors.append(APPROACH_COLORS['standard_dpo'])
                elif '2-Adapter' in approach:
                    colors.append(APPROACH_COLORS['2adapter_dpo'])
                elif 'Diversity' in approach:
                    colors.append(APPROACH_COLORS['diversity_weighted_dpo'])
                elif 'Repeated' in approach:
                    colors.append(APPROACH_COLORS['baseline_repeated'])
                else:
                    colors.append(APPROACH_COLORS['baseline_verbalized'])
                
                pos += 1
            pos += 0.5  # Gap between approach groups
        
        bars = ax.bar(x_positions, bar_values, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel(f'Δ {METRIC_LABELS_SHORT[metric]}\n(High Creativity − Low Creativity)', fontsize=10)
        ax.set_title(f'Diversity Modulation Strength - {METRIC_LABELS[metric]}', 
                    fontsize=12, fontweight='bold')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        output_file = os.path.join(output_subdir, f'{metric}_modulation_delta.png')
        plt.savefig(output_file, dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()
        print(f"   ✅ Saved: {output_file}")

# ============================================================
# GENERALIZATION COMPARISON (UF vs Tulu)
# ============================================================

def plot_generalization_comparison(all_data: Dict, output_dir: str):
    """
    Show how well results generalize from UF (in-domain) to Tulu (out-of-domain)
    """
    print("\n📊 Generating Generalization Comparison...")
    
    metrics = ['self_bleu', 'distinct_2', 'semantic_div']
    output_subdir = os.path.join(output_dir, "generalization")
    os.makedirs(output_subdir, exist_ok=True)
    
    for metric in metrics:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f'Generalization: In-Domain vs Out-of-Domain - {METRIC_LABELS_SHORT[metric]}', 
                    fontsize=14, fontweight='bold')
        
        approaches = ['standard_dpo', '2adapter_dpo', 'diversity_weighted_dpo']
        
        for ax_idx, approach in enumerate(approaches):
            ax = axes[ax_idx]
            
            for dataset in [2, 1, 3]:
                # Get UF and Tulu data
                uf_key = (dataset, approach, 'uf')
                tulu_key = (dataset, approach, 'tulu')
                
                if uf_key not in all_data or tulu_key not in all_data:
                    continue
                
                df_uf = all_data[uf_key]
                df_tulu = all_data[tulu_key]
                
                if metric not in df_uf.columns or metric not in df_tulu.columns:
                    continue
                
                x = np.arange(len(df_uf))
                
                # Plot both
                dataset_color = ['#e74c3c', '#3498db', '#2ecc71'][dataset - 1]
                ax.plot(x, df_uf[metric].values, linestyle='-', linewidth=2,
                       color=dataset_color, marker='o', markersize=5,
                       label=f'{DATASET_LABELS[dataset][:10]} (UF)')
                ax.plot(x, df_tulu[metric].values, linestyle='--', linewidth=2,
                       color=dataset_color, marker='s', markersize=5,
                       label=f'{DATASET_LABELS[dataset][:10]} (Tulu)')
            
            ax.set_title(APPROACH_LABELS[approach], fontsize=11, fontweight='bold')
            ax.set_xlabel('Creativity Bin', fontsize=10)
            if ax_idx == 0:
                ax.set_ylabel(METRIC_LABELS_SHORT[metric], fontsize=10)
            ax.set_xticks(x)
            # Simpler tick labels (just the start value)
            simple_labels = [f"{float(b.split('-')[0]):.1f}" for b in df_uf['bin'].values]
            ax.set_xticklabels(simple_labels, rotation=0, fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7, loc='best')
        
        plt.tight_layout()
        output_file = os.path.join(output_subdir, f'{metric}_generalization.png')
        plt.savefig(output_file, dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()
        print(f"   ✅ Saved: {output_file}")

# ============================================================
# SUMMARY TABLE (CSV)
# ============================================================

def generate_summary_table(all_data: Dict, baseline_data: Dict, output_dir: str):
    """Generate summary CSV with all metrics"""
    print("\n📊 Generating Summary Table...")
    
    metrics = ['self_bleu', 'distinct_2', 'semantic_div']
    rows = []
    
    # DPO results
    for (dataset, approach, eval_set), df in all_data.items():
        bins = df['bin'].values
        
        for metric in metrics:
            if metric not in df.columns:
                continue
            
            values = df[metric].values
            
            # Compute modulation delta
            low_indices = [i for i, b in enumerate(bins) if b.startswith('0.0') or b.startswith('0.2')]
            high_indices = [i for i, b in enumerate(bins) if b.startswith('0.5') or b.startswith('0.7')]
            
            if low_indices and high_indices:
                low_avg = np.mean([values[i] for i in low_indices])
                high_avg = np.mean([values[i] for i in high_indices])
                delta = high_avg - low_avg
            else:
                delta = np.nan
            
            rows.append({
                'Dataset': DATASET_LABELS[dataset],
                'Approach': APPROACH_LABELS[approach],
                'Eval Set': eval_set.upper(),
                'Metric': metric,
                'Mean': np.mean(values),
                'Std': np.std(values),
                'Delta (High-Low)': delta
            })
    
    # Baseline results
    for (approach, eval_set), df in baseline_data.items():
        bins = df['bin'].values
        
        for metric in metrics:
            if metric not in df.columns:
                continue
            
            values = df[metric].values
            
            low_indices = [i for i, b in enumerate(bins) if b.startswith('0.0') or b.startswith('0.2')]
            high_indices = [i for i, b in enumerate(bins) if b.startswith('0.5') or b.startswith('0.7')]
            
            if low_indices and high_indices:
                low_avg = np.mean([values[i] for i in low_indices])
                high_avg = np.mean([values[i] for i in high_indices])
                delta = high_avg - low_avg
            else:
                delta = np.nan
            
            rows.append({
                'Dataset': 'Base Model (No Training)',
                'Approach': APPROACH_LABELS[approach],
                'Eval Set': eval_set.upper(),
                'Metric': metric,
                'Mean': np.mean(values),
                'Std': np.std(values),
                'Delta (High-Low)': delta
            })
    
    df_summary = pd.DataFrame(rows)
    output_file = os.path.join(output_dir, 'summary_table.csv')
    df_summary.to_csv(output_file, index=False)
    print(f"   ✅ Saved: {output_file}")

# ============================================================
# NOVELTYBENCH RESULTS CONFIGURATION
# ============================================================

# Map of (training_dataset, approach) to NoveltyBench summary.json path
# training_dataset: 1=Embedding Diversity, 2=Quality Score (Baseline), 3=LLM-as-Judge
# approach: "standard_dpo", "2adapter_dpo", "diversity_weighted_dpo", "baseline_qwen"

NOVELTYBENCH_RESULTS = {
    # Standard DPO
    (1, "standard_dpo"): "results/standard_dataset1/summary.json",
    (2, "standard_dpo"): "results/standard_dataset2/summary.json",
    (3, "standard_dpo"): "results/standard_dataset3/summary.json",
    
    # 2-Adapter DPO
    (1, "2adapter_dpo"): "results/2adapter_dataset1/summary.json",
    (2, "2adapter_dpo"): "results/2adapter_dataset2/2adapter_combined_curated_2025-12-09_19-08-11/summary.json",
    (3, "2adapter_dpo"): "results/2adapter_dataset3/2adapter_combined_curated_2025-12-09_19-09-13/summary.json",
    
    # Diversity-Weighted DPO
    (1, "diversity_weighted_dpo"): "results/divpo_dataset1/summary.json",
    (2, "diversity_weighted_dpo"): "results/divpo_dataset2/summary.json",
    (3, "diversity_weighted_dpo"): "results/divpo_dataset3/summary.json",
    
    # Base Qwen (no training - same model, so use None for dataset)
    (None, "baseline_qwen"): "results/baseline_qwen3_8b_curated_2025-12-09_21-05-34/summary.json",
}

# ============================================================
# NOVELTYBENCH VISUALIZATION
# ============================================================

def load_noveltybench_data() -> Dict:
    """Load NoveltyBench summary data"""
    nb_data = {}
    
    print("\n" + "="*70)
    print("LOADING NOVELTYBENCH RESULTS")
    print("="*70)
    
    for (dataset, approach), file_path in NOVELTYBENCH_RESULTS.items():
        if file_path is None:
            print(f"⚠️  Skipping ({dataset}, {approach}): No file specified")
            continue
        
        if not os.path.exists(file_path):
            print(f"⚠️  Skipping ({dataset}, {approach}): File not found: {file_path}")
            continue
        
        try:
            with open(file_path, 'r') as f:
                summary = json.load(f)
            
            nb_data[(dataset, approach)] = {
                'mean_distinct': summary.get('mean_distinct', np.nan),
                'mean_utility': summary.get('mean_utility', np.nan)
            }
            print(f"✅ ({dataset}, {approach}): distinct={summary.get('mean_distinct', 'N/A'):.2f}, utility={summary.get('mean_utility', 'N/A'):.2f}")
            
        except Exception as e:
            print(f"❌ Error loading ({dataset}, {approach}): {e}")
    
    print(f"\n✅ Loaded {len(nb_data)} NoveltyBench results")
    print("="*70 + "\n")
    
    return nb_data

def plot_noveltybench_heatmaps(nb_data: Dict, output_dir: str):
    """
    Create heatmap tables for NoveltyBench results
    - Rows: Training datasets
    - Columns: Approaches (models)
    - Two heatmaps: one for Distinct, one for Utility
    """
    print("\n📊 Generating NoveltyBench Heatmaps...")
    
    output_subdir = os.path.join(output_dir, "noveltybench")
    os.makedirs(output_subdir, exist_ok=True)
    
    # Define order (baseline Quality Score at bottom)
    datasets = [3, 1, 2]
    approaches = ['baseline_qwen', 'standard_dpo', '2adapter_dpo', 'diversity_weighted_dpo']
    
    dataset_labels = [DATASET_LABELS.get(d, 'Base Model') for d in datasets]
    approach_labels = [APPROACH_LABELS.get(a, a) for a in approaches]
    
    for metric_key, metric_label, cmap in [
        ('mean_distinct', 'Distinct', 'RdYlGn'),
        ('mean_utility', 'Utility', 'RdYlGn')
    ]:
        # Build data matrix
        matrix = np.full((len(datasets), len(approaches)), np.nan)
        
        for i, dataset in enumerate(datasets):
            for j, approach in enumerate(approaches):
                if approach == 'baseline_qwen':
                    # Baseline doesn't depend on dataset - use None key
                    key = (None, approach)
                else:
                    key = (dataset, approach)
                
                if key in nb_data:
                    matrix[i, j] = nb_data[key].get(metric_key, np.nan)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # Mask NaN values
        mask = np.isnan(matrix)
        
        im = ax.imshow(matrix, aspect='auto', cmap=cmap, vmin=np.nanmin(matrix)*0.95, 
                      vmax=np.nanmax(matrix)*1.05)
        
        # Set ticks
        ax.set_xticks(np.arange(len(approaches)))
        ax.set_yticks(np.arange(len(datasets)))
        ax.set_xticklabels(approach_labels, rotation=15, ha='right', fontsize=10)
        ax.set_yticklabels(dataset_labels, fontsize=10)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(metric_label, rotation=270, labelpad=20, fontsize=11)
        
        # Add values to cells
        for i in range(len(datasets)):
            for j in range(len(approaches)):
                val = matrix[i, j]
                if not np.isnan(val):
                    text = ax.text(j, i, f'{val:.2f}',
                                 ha="center", va="center", color="black", 
                                 fontsize=12, fontweight='bold')
                else:
                    text = ax.text(j, i, 'N/A',
                                 ha="center", va="center", color="gray", 
                                 fontsize=10, style='italic')
        
        ax.set_title(f'NoveltyBench: {metric_label}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Training Approach', fontsize=11)
        ax.set_ylabel('Training Dataset', fontsize=11)
        
        plt.tight_layout()
        output_file = os.path.join(output_subdir, f'noveltybench_{metric_key}.png')
        plt.savefig(output_file, dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()
        print(f"   ✅ Saved: {output_file}")

def plot_noveltybench_grouped_bars(nb_data: Dict, output_dir: str):
    """
    Create grouped bar chart for NoveltyBench results
    - X-axis: Training datasets
    - Bars: Approaches (models)
    - Two figures: one for Distinct, one for Utility
    """
    print("\n📊 Generating NoveltyBench Grouped Bar Charts...")
    
    output_subdir = os.path.join(output_dir, "noveltybench")
    os.makedirs(output_subdir, exist_ok=True)
    
    # Order: baseline Quality Score last (matches heatmap bottom row)
    datasets = [3, 1, 2]
    approaches = ['standard_dpo', '2adapter_dpo', 'diversity_weighted_dpo']
    
    for metric_key, metric_label in [
        ('mean_distinct', 'Distinct'),
        ('mean_utility', 'Utility')
    ]:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(datasets))
        width = 0.25
        
        for i, approach in enumerate(approaches):
            values = []
            for dataset in datasets:
                key = (dataset, approach)
                if key in nb_data:
                    values.append(nb_data[key].get(metric_key, 0))
                else:
                    values.append(0)
            
            offset = (i - 1) * width
            bars = ax.bar(x + offset, values, width, 
                         label=APPROACH_LABELS[approach],
                         color=APPROACH_COLORS[approach],
                         alpha=0.8, edgecolor='black', linewidth=0.5)
            
            # Add value labels on bars
            for bar, val in zip(bars, values):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                           f'{val:.2f}', ha='center', va='bottom', fontsize=8)
        
        # Add baseline reference line if available
        if (None, 'baseline_qwen') in nb_data:
            baseline_val = nb_data[(None, 'baseline_qwen')].get(metric_key, None)
            if baseline_val:
                ax.axhline(y=baseline_val, color='gray', linestyle='--', linewidth=2,
                          label=f'Pretrained (Baseline) ({baseline_val:.2f})')
        
        ax.set_xlabel('Training Dataset', fontsize=11)
        ax.set_ylabel(metric_label, fontsize=11)
        ax.set_title(f'NoveltyBench: {metric_label}', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([DATASET_LABELS[d] for d in datasets], fontsize=10)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        output_file = os.path.join(output_subdir, f'noveltybench_{metric_key}_bars.png')
        plt.savefig(output_file, dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()
        print(f"   ✅ Saved: {output_file}")

def plot_noveltybench_combined_heatmap(nb_data: Dict, output_dir: str):
    """
    Create combined heatmap with Distinct and Utility side by side:
    - 1 row, 2 columns (Distinct left, Utility right)
    - Y-axis labels only on left plot
    - Shared x-axis label centered below both
    - Two colorbars on the right with different cmaps
    """
    print("\n📊 Generating NoveltyBench Combined Heatmap...")
    
    output_subdir = os.path.join(output_dir, "noveltybench")
    os.makedirs(output_subdir, exist_ok=True)
    
    # Define order (same as individual heatmaps)
    datasets = [3, 1, 2]  # baseline Quality Score at bottom
    approaches = ['baseline_qwen', 'standard_dpo', '2adapter_dpo', 'diversity_weighted_dpo']
    
    dataset_labels = [DATASET_LABELS.get(d, 'Base Model') for d in datasets]
    approach_labels = [APPROACH_LABELS.get(a, a) for a in approaches]
    
    # Build data matrices for both metrics
    matrices = {}
    for metric_key in ['mean_distinct', 'mean_utility']:
        matrix = np.full((len(datasets), len(approaches)), np.nan)
        for i, dataset in enumerate(datasets):
            for j, approach in enumerate(approaches):
                if approach == 'baseline_qwen':
                    key = (None, approach)
                else:
                    key = (dataset, approach)
                if key in nb_data:
                    matrix[i, j] = nb_data[key].get(metric_key, np.nan)
        matrices[metric_key] = matrix
    
    # Create figure with 2 subplots and space for colorbars
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('NoveltyBench Results', fontsize=14, fontweight='bold', y=0.98)
    
    # Different colormaps for each metric
    cmaps = {
        'mean_distinct': 'RdYlGn',  # Red-Yellow-Green
        'mean_utility': 'RdYlBu'    # Red-Yellow-Blue
    }
    metric_titles = {
        'mean_distinct': 'Distinct',
        'mean_utility': 'Utility'
    }
    
    images = []
    for idx, (metric_key, ax) in enumerate(zip(['mean_distinct', 'mean_utility'], axes)):
        matrix = matrices[metric_key]
        cmap = cmaps[metric_key]
        
        im = ax.imshow(matrix, aspect='auto', cmap=cmap, 
                      vmin=np.nanmin(matrix)*0.95, vmax=np.nanmax(matrix)*1.05)
        images.append(im)
        
        # Set x ticks on both plots (centered under columns)
        ax.set_xticks(np.arange(len(approaches)))
        ax.set_xticklabels(approach_labels, rotation=30, ha='center', fontsize=9)
        
        # Y ticks only on left plot
        ax.set_yticks(np.arange(len(datasets)))
        if idx == 0:
            ax.set_yticklabels(dataset_labels, fontsize=10)
            ax.set_ylabel('Training Dataset', fontsize=11)
        else:
            ax.set_yticklabels([])  # No y tick labels on right plot
        
        # Subplot title
        ax.set_title(metric_titles[metric_key], fontsize=12, fontweight='bold')
        
        # Add values to cells
        for i in range(len(datasets)):
            for j in range(len(approaches)):
                val = matrix[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f'{val:.2f}', ha="center", va="center", 
                           color="black", fontsize=11, fontweight='bold')
                else:
                    ax.text(j, i, 'N/A', ha="center", va="center", 
                           color="gray", fontsize=9, style='italic')
    
    # Add colorbars on the right
    # Adjust subplot positions to make room for colorbars and x labels
    plt.subplots_adjust(right=0.85, wspace=0.05, bottom=0.25)
    
    # Add shared x-axis label centered below both plots (lower position)
    fig.text(0.45, -0.02, 'Training Approach', ha='center', fontsize=11)
    
    # Colorbar for Distinct (top)
    cbar_ax1 = fig.add_axes([0.87, 0.55, 0.02, 0.35])
    cbar1 = fig.colorbar(images[0], cax=cbar_ax1)
    cbar1.set_label('Distinct', rotation=270, labelpad=15, fontsize=10)
    
    # Colorbar for Utility (bottom)
    cbar_ax2 = fig.add_axes([0.87, 0.15, 0.02, 0.35])
    cbar2 = fig.colorbar(images[1], cax=cbar_ax2)
    cbar2.set_label('Utility', rotation=270, labelpad=15, fontsize=10)
    
    output_file = os.path.join(output_subdir, 'noveltybench_combined.png')
    plt.savefig(output_file, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: {output_file}")

def generate_noveltybench_table(nb_data: Dict, output_dir: str):
    """Generate CSV table of NoveltyBench results"""
    print("\n📊 Generating NoveltyBench Table...")
    
    rows = []
    for (dataset, approach), data in nb_data.items():
        rows.append({
            'Training Dataset': DATASET_LABELS.get(dataset, 'Base Model (No Training)'),
            'Approach': APPROACH_LABELS.get(approach, approach),
            'Distinct': data.get('mean_distinct', np.nan),
            'Utility': data.get('mean_utility', np.nan)
        })
    
    df = pd.DataFrame(rows)
    output_file = os.path.join(output_dir, 'noveltybench_table.csv')
    df.to_csv(output_file, index=False)
    print(f"   ✅ Saved: {output_file}")

# ============================================================
# MAIN FUNCTION
# ============================================================

def main():
    """Main function to generate all visualizations"""
    print("\n" + "="*70)
    print("DPO DIVERSITY VISUALIZATION SCRIPT")
    print("="*70)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load all data
    all_data, baseline_data = load_all_data()
    
    if not all_data and not baseline_data:
        print("\n❌ No data loaded! Please check INFERENCE_RESULTS configuration.")
        return
    
    # Generate visualizations
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    
    metrics = ['self_bleu', 'distinct_2', 'semantic_div']
    
    # 1. Smart Facet Grids (one per metric)
    for metric in metrics:
        plot_smart_facet_grid(all_data, baseline_data, metric, OUTPUT_DIR)
    
    # 2. Approach Comparison Grids (one per metric)
    for metric in metrics:
        plot_approach_comparison_grid(all_data, baseline_data, metric, OUTPUT_DIR)
    
    # 3. Dataset Comparison Grids (one per metric) - inverse of approach comparison
    for metric in metrics:
        plot_dataset_comparison_grid(all_data, metric, OUTPUT_DIR)
    
    # 4. Modulation Delta Summary
    plot_modulation_delta(all_data, baseline_data, OUTPUT_DIR)
    
    # 5. Generalization Comparison
    plot_generalization_comparison(all_data, OUTPUT_DIR)
    
    # 6. Summary Table
    generate_summary_table(all_data, baseline_data, OUTPUT_DIR)
    
    # 7. NoveltyBench Visualizations
    nb_data = load_noveltybench_data()
    if nb_data:
        plot_noveltybench_heatmaps(nb_data, OUTPUT_DIR)
        plot_noveltybench_grouped_bars(nb_data, OUTPUT_DIR)
        plot_noveltybench_combined_heatmap(nb_data, OUTPUT_DIR)
        generate_noveltybench_table(nb_data, OUTPUT_DIR)
    
    print("\n" + "="*70)
    print("✅ ALL VISUALIZATIONS COMPLETE!")
    print("="*70)
    print(f"📁 Output directory: {OUTPUT_DIR}/")
    print(f"   - smart_facet_grid/     (3 figures, one per metric)")
    print(f"   - approach_comparison/  (3 figures, cols=datasets, lines=approaches)")
    print(f"   - dataset_comparison/   (3 figures, cols=approaches, lines=datasets)")
    print(f"   - modulation_delta/     (3 figures, one per metric)")
    print(f"   - generalization/       (3 figures, one per metric)")
    print(f"   - noveltybench/         (heatmaps + bar charts)")
    print(f"   - summary_table.csv     (all metrics and deltas)")
    print(f"   - noveltybench_table.csv")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
