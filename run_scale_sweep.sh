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

    # If the final analysis results already exist for this K, we can skip it entirely
    if [ ! -f "analysis_results/diagnostic_results_k_${K}.json" ]; then
        
        # Check if Centralized PPO models already exist for this K
        if ! ls centralized_ppo_results/models/centralized_omega_*_k_${K}_* 1> /dev/null 2>&1; then
            python3 ppo_centralized.py --num_classrooms $K
        else
            echo "  --> Skipping Centralized PPO for K=$K (models already exist)"
        fi

        # Check if MAPPO CTDE models already exist for this K
        if ! ls mappo_results/models/mappo_omega_*_k_${K}_* 1> /dev/null 2>&1; then
            python3 ppo_ctde.py --num_classrooms $K
        else
            echo "  --> Skipping MAPPO CTDE for K=$K (models already exist)"
        fi

        # We always need to run the analysis if the .json result doesn't exist yet
        python3 analyze_environment.py --num_classrooms $K
    else
        echo "  --> Skipping K=$K entirely (analysis results already exist)"
    fi
done

echo "Sweep complete!"
