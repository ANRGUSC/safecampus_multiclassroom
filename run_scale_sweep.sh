#!/bin/bash

# ==============================================================================
# Phase 4 (RQ4 - Scale) Sweep Script
# ==============================================================================
# This script will run the training and evaluation pipeline for different 
# sizes of K (number of classrooms) to test Hypothesis 4.
# ==============================================================================

export CUDA_VISIBLE_DEVICES=""
K_VALUES=(2 3 5)

for K in "${K_VALUES[@]}"; do
    echo "=========================================================="
    echo "Running Sweep for K = $K"
    echo "=========================================================="

    python3 ppo_centralized.py --num_classrooms $K
    python3 ppo_ctde.py --num_classrooms $K
    python3 analyze_environment.py --num_classrooms $K
done

echo "Sweep complete!"

