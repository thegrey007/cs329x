# Visualization Guide

## Overview

`scripts/visualize_results.py` generates comprehensive visualizations comparing diversity metrics across different DPO approaches and datasets.

## Quick Start

### 1. Configure the Script

Edit the `INFERENCE_RESULTS` dictionary at the top of `scripts/visualize_results.py`:

```python
INFERENCE_RESULTS = {
    # Dataset 1
    (1, "standard_dpo"): "inference_results/standard_dpo/dataset1_xxx.json",
    (1, "2adapter_dpo"): "inference_results/2adapter_dpo/dataset1_xxx.json",
    (1, "diversity_weighted_dpo"): "inference_results/diversity_weighted_dpo/dataset1_xxx.json",
    
    # Dataset 2
    (2, "standard_dpo"): "inference_results/standard_dpo/dataset2_xxx.json",
    (2, "2adapter_dpo"): None,  # Set to None if not available
    (2, "diversity_weighted_dpo"): "inference_results/diversity_weighted_dpo/dataset2_diversity.json",
    
    # Dataset 3
    (3, "standard_dpo"): None,
    (3, "2adapter_dpo"): None,
    (3, "diversity_weighted_dpo"): "inference_results/diversity_weighted_dpo/dataset3_diversity.json",
}
```

### 2. Run the Script

```bash
cd /Users/abhinavsattiraju/Desktop/329x/cs329x
python3 scripts/visualize_results.py
```

### 3. Find Your Visualizations

All plots are saved in `visualizations/` with the following structure:

```
visualizations/
├── type1_line_plots/
│   ├── dataset1_line_comparison.png
│   ├── dataset2_line_comparison.png
│   └── dataset3_line_comparison.png
│
├── type2_grouped_bars/
│   ├── dataset1_grouped_bars.png
│   ├── dataset2_grouped_bars.png
│   └── dataset3_grouped_bars.png
│
├── type3_modulation_delta/
│   ├── self_bleu_modulation_delta.png
│   ├── distinct_2_modulation_delta.png
│   └── semantic_div_modulation_delta.png
│
├── type4_heatmap/
│   ├── self_bleu_heatmap.png
│   ├── distinct_2_heatmap.png
│   └── semantic_div_heatmap.png
│
└── type5_faceted/
    ├── figureA_approach_comparison.png  (9-panel: Does approach matter?)
    └── figureB_dataset_comparison.png   (9-panel: Do pairs matter?)
```

## Visualization Types

### Type 1: Line Plots
**Best for:** Showing diversity modulation trends across creativity levels

- One plot per dataset
- 3 subplots per metric (Self-BLEU, Distinct-2, Semantic Div)
- Lines colored by approach
- Shows if diversity-weighted DPO has positive slope (increasing diversity with creativity)

### Type 2: Grouped Bar Charts
**Best for:** Direct comparison of absolute values

- Similar to Type 1 but with grouped bars
- Easier to see magnitude differences between approaches

### Type 3: Modulation Delta
**Best for:** Showing modulation strength at a glance

- Computes Δ = (High Creativity - Low Creativity) for each approach/dataset
- Positive values = successful modulation (more diverse on creative prompts)
- One plot per metric

### Type 4: Heatmap Grid
**Best for:** Compact overview of all results

- Rows: Dataset × Approach combinations
- Columns: Creativity bins
- Color intensity shows metric value
- Good for appendix/supplementary material

### Type 5: Faceted Full Comparison
**Best for:** Systematic analysis of both research questions

- **Figure A**: Compare approaches within each dataset (3×3 grid)
- **Figure B**: Compare datasets within each approach (3×3 grid)
- Most comprehensive visualization

## Customization Options

### Change Bin Size
```python
CREATIVITY_BIN_SIZE = 0.25  # Default: 4 bins
# Try 0.2 for 5 bins, or 0.1 for 10 bins
```

### Change Output Directory
```python
OUTPUT_DIR = "my_visualizations"
```

### Change Figure Resolution
```python
FIGURE_DPI = 300  # Default: 300 (high quality)
# Use 150 for faster generation, 600 for publication quality
```

### Change Colors
```python
APPROACH_COLORS = {
    "standard_dpo": "#3498db",  # Blue
    "2adapter_dpo": "#e74c3c",  # Red  
    "diversity_weighted_dpo": "#2ecc71"  # Green
}
```

## Recommendations for Presentations

### Slide 1: Main Finding
Use **Type 1** or **Type 2** for one dataset (e.g., Dataset 2):
- `type1_line_plots/dataset2_line_comparison.png` OR
- `type2_grouped_bars/dataset2_grouped_bars.png`

**Message**: "Diversity-weighted DPO successfully modulates diversity across creativity levels"

### Slide 2: Robustness
Use **Type 3**:
- `type3_modulation_delta/semantic_div_modulation_delta.png`

**Message**: "Effect holds across different pair selection strategies"

### Backup/Appendix
Use **Type 5**:
- `type5_faceted/figureA_approach_comparison.png`
- `type5_faceted/figureB_dataset_comparison.png`

**Message**: Comprehensive systematic comparison

## Troubleshooting

### "No data loaded"
- Check that file paths in `INFERENCE_RESULTS` are correct
- Set unavailable results to `None`

### Missing plots
- Script skips visualizations if data is missing
- Check terminal output for warnings

### Plot looks empty
- Verify inference results have `alpha`, `self_bleu`, `distinct_2`, `semantic_div` fields
- Check that JSON structure matches expected format

## Dependencies

The script uses:
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`

All should already be in your environment from running inference scripts.

