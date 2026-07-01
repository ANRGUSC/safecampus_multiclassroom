#!/bin/bash

# ==============================================================================
# Phase 4 (RQ4 - Scale) Sweep Script
# ==============================================================================
# This script will run the training and evaluation pipeline for different 
# sizes of K (number of classrooms) to test Hypothesis 4.
# ==============================================================================

K_VALUES=(2 3 5)

for K in "${K_VALUES[@]}"; do
    echo "=========================================================="
    echo "Running Sweep for K = $K"
    echo "=========================================================="

    python ppo_centralized.py --num_classrooms $K
    python ppo_ctde.py --num_classrooms $K
    python analyze_environment.py --num_classrooms $K
done

echo "Sweep complete!"

