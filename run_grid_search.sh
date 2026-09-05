#!/bin/bash

# ==============================================================================
# 📊 RAG & Agent Grid Search Grid Sweep Script
# ==============================================================================
# This script automatically sweeps through different configurations of retrieval (k, search_type)
# to evaluate which parameter combination gives the highest Hit Rate, MRR, and Accuracy.
# ==============================================================================

# Default configurations
SAMPLE_SIZE=10
MODEL="gpt-4o-mini"
EMBEDDING="text-embedding-3-small"

echo "======================================================================"
echo "🚀 STARTING AUTO-OPTIMIZATION GRID SWEEP"
echo " - Tested LLM Engine    : $MODEL"
echo " - Embedding Model      : $EMBEDDING"
echo " - Samples per Run      : $SAMPLE_SIZE"
echo "======================================================================"

# Create a clean outputs directory for logs
mkdir -p logs/

# Loop over different values of K (documents retrieved)
for k_val in 3 5; do
  # Loop over different retrieval styles
  for search_style in "similarity" "mmr"; do
    
    echo ""
    echo "------------------------------------------------------------"
    echo "👉 Running Experiment: [Search Style = $search_style | K = $k_val]"
    echo "------------------------------------------------------------"
    
    # Run the parameterized evaluation script
    # Save standard outputs to log files in case you want to trace the details
    python evaluate_parameterized.py \
      --k "$k_val" \
      --search-type "$search_style" \
      --model "$MODEL" \
      --embedding-model "$EMBEDDING" \
      --sample-size "$SAMPLE_SIZE" \
      > "logs/eval_k${k_val}_${search_style}.log" 2>&1
      
    # Print a quick success status
    if [ $? -eq 0 ]; then
      echo "✅ Successfully completed! Report generated as: eval_report_${MODEL}_${search_style}_k${k_val}.json"
    else
      echo "❌ Failed to run. Check log file at: logs/eval_k${k_val}_${search_style}.log"
    fi
    
  done
done

echo ""
echo "======================================================================"
echo "🎉 GRID SWEEP COMPLETE!"
echo "All parameter combinations have been evaluated and JSON reports are ready."
echo "You can find individual metrics reports in your workspace directory."
echo "======================================================================"
