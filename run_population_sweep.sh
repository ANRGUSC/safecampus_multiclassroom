#!/bin/bash

# ==============================================================================
# Population and Classroom Scale Sweep Script
# ==============================================================================
# This script will run the training and evaluation pipeline for different 
# sizes of total population and K (number of classrooms).
# ==============================================================================

export CUDA_VISIBLE_DEVICES=""
POP_VALUES=(100 200)
K_VALUES=(2 5)

for POP in "${POP_VALUES[@]}"; do
    for K in "${K_VALUES[@]}"; do
        echo "=========================================================="
        echo "Running Sweep for Population = $POP, K = $K"
        echo "=========================================================="

        # If the final analysis results already exist for this Pop and K, we can skip it entirely
        if [ ! -f "analysis_results/diagnostic_results_pop_${POP}_k_${K}.json" ]; then
            
            # Check if Centralized PPO models already exist
            if ! ls centralized_ppo_results/models/centralized_omega_*_k_${K}_pop_${POP}_* 1> /dev/null 2>&1; then
                python3 ppo_centralized.py --num_classrooms $K --total_population $POP
            else
                echo "  --> Skipping Centralized PPO for Pop=$POP, K=$K (models already exist)"
            fi

            # Check if MAPPO CTDE models already exist
            if ! ls mappo_results/models/mappo_omega_*_k_${K}_pop_${POP}_* 1> /dev/null 2>&1; then
                python3 ppo_ctde.py --num_classrooms $K --total_population $POP
            else
                echo "  --> Skipping MAPPO CTDE for Pop=$POP, K=$K (models already exist)"
            fi

            # We always need to run the analysis if the .json result doesn't exist yet
            python3 analyze_environment.py --num_classrooms $K --total_population $POP
        else
            echo "  --> Skipping Pop=$POP, K=$K entirely (analysis results already exist)"
        fi
    done

    # Generate the performance gap plot for the current population across all K values
    echo "Generating plots for Population = $POP"
    python3 plot_k_gap.py --total_population $POP --k_vals "${K_VALUES[@]}"
done

echo "Sweep complete!"
