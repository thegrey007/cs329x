#!/usr/bin/env python3
"""
Visualization Script for DPO Inference Results
Compares diversity metrics across approaches and datasets
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

# Map of (dataset, approach) to inference result JSON file
INFERENCE_RESULTS = {
    # Dataset 1
    (1, "standard_dpo"): None,  # "inference_results/standard_dpo/dataset1_xxx.json",
    (1, "2adapter_dpo"): None,
    (1, "diversity_weighted_dpo"): None,
    
    # Dataset 2
    (2, "standard_dpo"): None,
    (2, "2adapter_dpo"): None,
    (2, "diversity_weighted_dpo"): "inference_results/diversity_weighted_dpo/dataset2_diversity.json",
    
    # Dataset 3
    (3, "standard_dpo"): None,
    (3, "2adapter_dpo"): None,
    (3, "diversity_weighted_dpo"): "inference_results/diversity_weighted_dpo/dataset3_diversity.json",
}

# Visualization settings
CREATIVITY_BIN_SIZE = 0.25  # Creates 4 bins: [0-0.25, 0.25-0.5, 0.5-0.75, 0.75-1.0]
OUTPUT_DIR = "visualizations"
FIGURE_DPI = 300
FIGURE_SIZE_SINGLE = (10, 6)
FIGURE_SIZE_GRID = (15, 12)

# Plot styling
APPROACH_COLORS = {
    "standard_dpo": "#3498db",  # Blue
    "2adapter_dpo": "#e74c3c",  # Red
    "diversity_weighted_dpo": "#2ecc71"  # Green
}

APPROACH_LABELS = {
    "standard_dpo": "Standard DPO",
    "2adapter_dpo": "2-Adapter DPO",
    "diversity_weighted_dpo": "Diversity-Weighted DPO"
}

METRIC_LABELS = {
    "self_bleu": "Self-BLEU (lower = more diverse)",
    "distinct_2": "Distinct-2 (higher = more diverse)",
    "semantic_div": "Semantic Diversity (higher = more diverse)"
}

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_inference_results(file_path: str) -> Dict:
    """Load inference results from JSON file"""
    with open(file_path, 'r') as f:
        return json.load(f)

def bin_by_creativity(results: List[Dict], bin_size: float = 0.25) -> Dict[str, List[Dict]]:
    """Bin results by creativity score"""
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
            values = [r[metric] for r in results if metric in r]
            row[metric] = np.mean(values) if values else np.nan
            row[f'{metric}_std'] = np.std(values) if values else np.nan
            row[f'{metric}_count'] = len(values)
        
        data.append(row)
    
    return pd.DataFrame(data)

def load_all_data() -> Dict[Tuple[int, str], pd.DataFrame]:
    """Load and process all inference results"""
    all_data = {}
    metrics = ['self_bleu', 'distinct_2', 'semantic_div']
    
    print("\n" + "="*70)
    print("LOADING INFERENCE RESULTS")
    print("="*70)
    
    for (dataset, approach), file_path in INFERENCE_RESULTS.items():
        if file_path is None:
            print(f"⚠️  Skipping ({dataset}, {approach}): No file specified")
            continue
        
        if not os.path.exists(file_path):
            print(f"⚠️  Skipping ({dataset}, {approach}): File not found: {file_path}")
            continue
        
        try:
            print(f"📂 Loading ({dataset}, {approach}): {file_path}")
            data = load_inference_results(file_path)
            results = data.get('results', [])
            
            # Bin by creativity and compute statistics
            binned = bin_by_creativity(results, CREATIVITY_BIN_SIZE)
            stats = compute_bin_statistics(binned, metrics)
            
            all_data[(dataset, approach)] = stats
            print(f"   ✅ Loaded {len(results)} results, {len(stats)} bins")
            
        except Exception as e:
            print(f"   ❌ Error loading: {e}")
    
    print(f"\n✅ Successfully loaded {len(all_data)} dataset-approach combinations")
    print("="*70 + "\n")
    
    return all_data

# ============================================================
# VISUALIZATION TYPE 1: LINE PLOTS
# ============================================================

def plot_line_comparison(all_data: Dict, output_dir: str):
    """
    Line plots showing diversity trends across creativity levels
    Separate plots for each dataset
    """
    print("\n📊 Generating Type 1: Line Plots...")
    
    datasets = sorted(set(k[0] for k in all_data.keys()))
    metrics = ['self_bleu', 'distinct_2', 'semantic_div']
    approaches = ['standard_dpo', '2adapter_dpo', 'diversity_weighted_dpo']
    
    output_subdir = os.path.join(output_dir, "type1_line_plots")
    os.makedirs(output_subdir, exist_ok=True)
    
    for dataset in datasets:
        # Create figure with 3 subplots (one per metric)
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f'Diversity Metrics Across Creativity Levels - Dataset {dataset}', 
                     fontsize=16, fontweight='bold')
        
        for metric_idx, metric in enumerate(metrics):
            ax = axes[metric_idx]
            
            for approach in approaches:
                if (dataset, approach) not in all_data:
                    continue
                
                df = all_data[(dataset, approach)]
                
                # Plot line
                x = range(len(df))
                y = df[metric].values
                ax.plot(x, y, marker='o', linewidth=2, markersize=8,
                       color=APPROACH_COLORS[approach],
                       label=APPROACH_LABELS[approach])
            
            # Formatting
            ax.set_xlabel('Creativity Bin', fontsize=12)
            ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
            ax.set_title(METRIC_LABELS[metric], fontsize=11)
            ax.set_xticks(range(len(df)))
            ax.set_xticklabels(df['bin'].values, rotation=0)
            ax.legend(loc='best', fontsize=9)
            ax.grid(True, alpha=0.3)
            
            # Add vertical line at creativity = 0.5
            ax.axvline(x=1.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
            ax.text(1.5, ax.get_ylim()[1]*0.95, 'Factual|Creative', 
                   ha='center', va='top', fontsize=8, color='gray')
        
        plt.tight_layout()
        output_file = os.path.join(output_subdir, f'dataset{dataset}_line_comparison.png')
        plt.savefig(output_file, dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()
        print(f"   ✅ Saved: {output_file}")

# ============================================================
# VISUALIZATION TYPE 2: GROUPED BAR CHARTS
# ============================================================

def plot_grouped_bars(all_data: Dict, output_dir: str):
    """
    Grouped bar charts showing diversity across creativity levels
    Similar to the user's example image
    """
    print("\n📊 Generating Type 2: Grouped Bar Charts...")
    
    datasets = sorted(set(k[0] for k in all_data.keys()))
    metrics = ['self_bleu', 'distinct_2', 'semantic_div']
    approaches = ['standard_dpo', '2adapter_dpo', 'diversity_weighted_dpo']
    
    output_subdir = os.path.join(output_dir, "type2_grouped_bars")
    os.makedirs(output_subdir, exist_ok=True)
    
    for dataset in datasets:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f'Diversity Metrics Comparison - Dataset {dataset}', 
                     fontsize=16, fontweight='bold')
        
        for metric_idx, metric in enumerate(metrics):
            ax = axes[metric_idx]
            
            # Get all bins (assuming all approaches have same bins)
            sample_df = next((df for k, df in all_data.items() if k[0] == dataset), None)
            if sample_df is None:
                continue
            
            bins = sample_df['bin'].values
            x = np.arange(len(bins))
            width = 0.25
            
            for approach_idx, approach in enumerate(approaches):
                if (dataset, approach) not in all_data:
                    continue
                
                df = all_data[(dataset, approach)]
                y = df[metric].values
                
                offset = (approach_idx - 1) * width
                ax.bar(x + offset, y, width, 
                      label=APPROACH_LABELS[approach],
                      color=APPROACH_COLORS[approach],
                      alpha=0.8)
            
            # Formatting
            ax.set_xlabel('Creativity Bin', fontsize=12)
            ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
            ax.set_title(METRIC_LABELS[metric], fontsize=11)
            ax.set_xticks(x)
            ax.set_xticklabels(bins, rotation=0)
            ax.legend(loc='best', fontsize=9)
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        output_file = os.path.join(output_subdir, f'dataset{dataset}_grouped_bars.png')
        plt.savefig(output_file, dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()
        print(f"   ✅ Saved: {output_file}")

# ============================================================
# VISUALIZATION TYPE 3: MODULATION DELTA
# ============================================================

def plot_modulation_delta(all_data: Dict, output_dir: str):
    """
    Bar chart showing modulation strength (High Creativity - Low Creativity)
    Shows which approach/dataset combo has strongest diversity modulation
    """
    print("\n📊 Generating Type 3: Modulation Delta...")
    
    metrics = ['self_bleu', 'distinct_2', 'semantic_div']
    output_subdir = os.path.join(output_dir, "type3_modulation_delta")
    os.makedirs(output_subdir, exist_ok=True)
    
    # Compute deltas for each metric
    for metric in metrics:
        delta_data = []
        
        for (dataset, approach), df in all_data.items():
            # High creativity: bins 3-4 (0.5-1.0)
            # Low creativity: bins 1-2 (0.0-0.5)
            bins = df['bin'].values
            values = df[metric].values
            
            low_indices = [i for i, b in enumerate(bins) if b.startswith('0.0') or b.startswith('0.2')]
            high_indices = [i for i, b in enumerate(bins) if b.startswith('0.5') or b.startswith('0.7')]
            
            if low_indices and high_indices:
                low_avg = np.mean([values[i] for i in low_indices])
                high_avg = np.mean([values[i] for i in high_indices])
                delta = high_avg - low_avg
                
                delta_data.append({
                    'dataset': f'Dataset {dataset}',
                    'approach': approach,
                    'delta': delta
                })
        
        if not delta_data:
            continue
        
        # Create plot
        df_delta = pd.DataFrame(delta_data)
        fig, ax = plt.subplots(figsize=(10, 6))
        
        approaches = df_delta['approach'].unique()
        datasets = df_delta['dataset'].unique()
        x = np.arange(len(approaches))
        width = 0.25
        
        for dataset_idx, dataset in enumerate(datasets):
            dataset_data = df_delta[df_delta['dataset'] == dataset]
            y = [dataset_data[dataset_data['approach'] == app]['delta'].values[0] 
                 if len(dataset_data[dataset_data['approach'] == app]) > 0 else 0
                 for app in approaches]
            
            offset = (dataset_idx - len(datasets)/2 + 0.5) * width
            ax.bar(x + offset, y, width, label=dataset, alpha=0.8)
        
        ax.set_xlabel('Approach', fontsize=12)
        ax.set_ylabel(f'Δ {metric.replace("_", " ").title()}\n(High Creativity - Low Creativity)', fontsize=12)
        ax.set_title(f'Diversity Modulation Strength - {METRIC_LABELS[metric]}', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([APPROACH_LABELS[app] for app in approaches], rotation=15, ha='right')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        
        plt.tight_layout()
        output_file = os.path.join(output_subdir, f'{metric}_modulation_delta.png')
        plt.savefig(output_file, dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()
        print(f"   ✅ Saved: {output_file}")

# ============================================================
# VISUALIZATION TYPE 4: HEATMAP GRID
# ============================================================

def plot_heatmap_grid(all_data: Dict, output_dir: str):
    """
    Heatmap showing all results in a compact grid
    Rows: Dataset × Approach, Columns: Creativity Bins
    """
    print("\n📊 Generating Type 4: Heatmap Grid...")
    
    metrics = ['self_bleu', 'distinct_2', 'semantic_div']
    output_subdir = os.path.join(output_dir, "type4_heatmap")
    os.makedirs(output_subdir, exist_ok=True)
    
    for metric in metrics:
        # Build data matrix
        rows = []
        row_labels = []
        
        for dataset in sorted(set(k[0] for k in all_data.keys())):
            for approach in ['standard_dpo', '2adapter_dpo', 'diversity_weighted_dpo']:
                if (dataset, approach) not in all_data:
                    continue
                
                df = all_data[(dataset, approach)]
                values = df[metric].values
                rows.append(values)
                row_labels.append(f'D{dataset}-{APPROACH_LABELS[approach][:10]}')
        
        if not rows:
            continue
        
        # Get bin labels
        sample_df = next(iter(all_data.values()))
        col_labels = sample_df['bin'].values
        
        # Create heatmap
        data_matrix = np.array(rows)
        fig, ax = plt.subplots(figsize=(10, max(6, len(rows) * 0.5)))
        
        im = ax.imshow(data_matrix, aspect='auto', cmap='RdYlGn_r' if metric == 'self_bleu' else 'RdYlGn')
        
        # Set ticks
        ax.set_xticks(np.arange(len(col_labels)))
        ax.set_yticks(np.arange(len(row_labels)))
        ax.set_xticklabels(col_labels)
        ax.set_yticklabels(row_labels)
        
        # Rotate x labels
        plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(metric.replace('_', ' ').title(), rotation=270, labelpad=20)
        
        # Add values to cells
        for i in range(len(row_labels)):
            for j in range(len(col_labels)):
                text = ax.text(j, i, f'{data_matrix[i, j]:.3f}',
                             ha="center", va="center", color="black", fontsize=8)
        
        ax.set_title(f'Heatmap: {METRIC_LABELS[metric]}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Creativity Bin', fontsize=12)
        ax.set_ylabel('Dataset - Approach', fontsize=12)
        
        plt.tight_layout()
        output_file = os.path.join(output_subdir, f'{metric}_heatmap.png')
        plt.savefig(output_file, dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()
        print(f"   ✅ Saved: {output_file}")

# ============================================================
# VISUALIZATION TYPE 5: FACETED FULL COMPARISON
# ============================================================

def plot_faceted_full(all_data: Dict, output_dir: str):
    """
    Comprehensive faceted plots answering both research questions
    Figure A: Does approach matter? (Dataset view)
    Figure B: Do pairs matter? (Approach view)
    """
    print("\n📊 Generating Type 5: Faceted Full Comparison...")
    
    metrics = ['self_bleu', 'distinct_2', 'semantic_div']
    output_subdir = os.path.join(output_dir, "type5_faceted")
    os.makedirs(output_subdir, exist_ok=True)
    
    # Figure A: Compare approaches within each dataset
    datasets = sorted(set(k[0] for k in all_data.keys()))
    approaches = ['standard_dpo', '2adapter_dpo', 'diversity_weighted_dpo']
    
    fig, axes = plt.subplots(len(metrics), len(datasets), figsize=(6*len(datasets), 5*len(metrics)))
    if len(metrics) == 1:
        axes = axes.reshape(1, -1)
    if len(datasets) == 1:
        axes = axes.reshape(-1, 1)
    
    fig.suptitle('Does Approach Matter? (Compare Approaches per Dataset)', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    for metric_idx, metric in enumerate(metrics):
        for dataset_idx, dataset in enumerate(datasets):
            ax = axes[metric_idx, dataset_idx]
            
            for approach in approaches:
                if (dataset, approach) not in all_data:
                    continue
                
                df = all_data[(dataset, approach)]
                x = range(len(df))
                y = df[metric].values
                ax.plot(x, y, marker='o', linewidth=2, markersize=6,
                       color=APPROACH_COLORS[approach],
                       label=APPROACH_LABELS[approach])
            
            # Formatting
            if metric_idx == len(metrics) - 1:
                ax.set_xlabel('Creativity Bin', fontsize=10)
                ax.set_xticks(range(len(df)))
                ax.set_xticklabels(df['bin'].values, rotation=45, ha='right', fontsize=8)
            else:
                ax.set_xticklabels([])
            
            if dataset_idx == 0:
                ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=10)
            
            if metric_idx == 0:
                ax.set_title(f'Dataset {dataset}', fontsize=12, fontweight='bold')
            
            ax.legend(loc='best', fontsize=7)
            ax.grid(True, alpha=0.3)
            ax.axvline(x=1.5, color='gray', linestyle='--', alpha=0.3, linewidth=1)
    
    plt.tight_layout()
    output_file = os.path.join(output_subdir, 'figureA_approach_comparison.png')
    plt.savefig(output_file, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: {output_file}")
    
    # Figure B: Compare datasets within each approach
    fig, axes = plt.subplots(len(metrics), len(approaches), figsize=(6*len(approaches), 5*len(metrics)))
    if len(metrics) == 1:
        axes = axes.reshape(1, -1)
    if len(approaches) == 1:
        axes = axes.reshape(-1, 1)
    
    fig.suptitle('Do Pairs Matter? (Compare Datasets per Approach)', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    dataset_colors = {1: '#e74c3c', 2: '#3498db', 3: '#2ecc71'}
    
    for metric_idx, metric in enumerate(metrics):
        for approach_idx, approach in enumerate(approaches):
            ax = axes[metric_idx, approach_idx]
            
            for dataset in datasets:
                if (dataset, approach) not in all_data:
                    continue
                
                df = all_data[(dataset, approach)]
                x = range(len(df))
                y = df[metric].values
                ax.plot(x, y, marker='o', linewidth=2, markersize=6,
                       color=dataset_colors.get(dataset, 'gray'),
                       label=f'Dataset {dataset}')
            
            # Formatting
            if metric_idx == len(metrics) - 1:
                ax.set_xlabel('Creativity Bin', fontsize=10)
                ax.set_xticks(range(len(df)))
                ax.set_xticklabels(df['bin'].values, rotation=45, ha='right', fontsize=8)
            else:
                ax.set_xticklabels([])
            
            if approach_idx == 0:
                ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=10)
            
            if metric_idx == 0:
                ax.set_title(APPROACH_LABELS[approach], fontsize=12, fontweight='bold')
            
            ax.legend(loc='best', fontsize=7)
            ax.grid(True, alpha=0.3)
            ax.axvline(x=1.5, color='gray', linestyle='--', alpha=0.3, linewidth=1)
    
    plt.tight_layout()
    output_file = os.path.join(output_subdir, 'figureB_dataset_comparison.png')
    plt.savefig(output_file, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
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
    all_data = load_all_data()
    
    if not all_data:
        print("\n❌ No data loaded! Please check INFERENCE_RESULTS configuration.")
        return
    
    # Generate all visualization types
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    
    plot_line_comparison(all_data, OUTPUT_DIR)
    plot_grouped_bars(all_data, OUTPUT_DIR)
    plot_modulation_delta(all_data, OUTPUT_DIR)
    plot_heatmap_grid(all_data, OUTPUT_DIR)
    plot_faceted_full(all_data, OUTPUT_DIR)
    
    print("\n" + "="*70)
    print("✅ ALL VISUALIZATIONS COMPLETE!")
    print("="*70)
    print(f"📁 Output directory: {OUTPUT_DIR}/")
    print(f"   - type1_line_plots/")
    print(f"   - type2_grouped_bars/")
    print(f"   - type3_modulation_delta/")
    print(f"   - type4_heatmap/")
    print(f"   - type5_faceted/")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()

