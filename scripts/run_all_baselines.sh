#!/bin/bash

# ========================================
# RUN BASELINES FOR MULTIPLE MODELS
# ========================================

# Fix OpenMP duplicate library issue
export KMP_DUPLICATE_LIB_OK=TRUE

# Models to test
MODELS=(
    "Qwen/Qwen3-4B-Instruct-2507"
    "Qwen/Qwen3-8B"
)

# Color codes for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "========================================================================"
echo "Running Baselines B and D for Multiple Models"
echo "========================================================================"
echo ""

# Backup original files
cp baseline_b.py baseline_b.py.backup
cp baseline_d.py baseline_d.py.backup

for MODEL in "${MODELS[@]}"; do
    echo ""
    echo "${BLUE}========================================================================"
    echo "MODEL: $MODEL"
    echo "========================================================================${NC}"
    echo ""
    
    # Sanitize model name for use in file paths (replace / with -)
    MODEL_SAFE=$(echo "$MODEL" | sed 's/\//-/g')
    
    # ========================================
    # RUN BASELINE B
    # ========================================
    echo "${GREEN}[1/2] Running Baseline B for $MODEL...${NC}"
    
    # Modify baseline_b.py with current model
    sed -i.tmp "s|^MODEL = .*|MODEL = \"$MODEL\"|" baseline_b.py
    
    # Run baseline B
    python baseline_b.py
    BASELINE_B_EXIT=$?
    
    if [ $BASELINE_B_EXIT -eq 0 ]; then
        echo "${GREEN}✅ Baseline B completed successfully!${NC}"
    else
        echo "${YELLOW}⚠️  Baseline B exited with code $BASELINE_B_EXIT${NC}"
    fi
    
    echo ""
    
    # ========================================
    # RUN BASELINE D
    # ========================================
    echo "${GREEN}[2/2] Running Baseline D for $MODEL...${NC}"
    
    # Modify baseline_d.py with current model
    sed -i.tmp "s|^MODEL = .*|MODEL = \"$MODEL\"|" baseline_d.py
    
    # Run baseline D
    python baseline_d.py
    BASELINE_D_EXIT=$?
    
    if [ $BASELINE_D_EXIT -eq 0 ]; then
        echo "${GREEN}✅ Baseline D completed successfully!${NC}"
    else
        echo "${YELLOW}⚠️  Baseline D exited with code $BASELINE_D_EXIT${NC}"
    fi
    
    echo ""
    echo "${BLUE}Completed both baselines for $MODEL${NC}"
    echo ""
    
    # Small pause between models
    sleep 2
done

# Restore original files
mv baseline_b.py.backup baseline_b.py
mv baseline_d.py.backup baseline_d.py

# Clean up temporary files
rm -f baseline_b.py.tmp baseline_d.py.tmp

echo ""
echo "========================================================================"
echo "🎉 ALL MODELS AND BASELINES COMPLETED!"
echo "========================================================================"
echo ""
echo "Results directories created:"
for MODEL in "${MODELS[@]}"; do
    echo "  - ${MODEL}_baseline_b_results/"
    echo "  - ${MODEL}_baseline_d_results/"
done
echo ""