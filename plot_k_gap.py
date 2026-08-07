import argparse
import json
# pyrefly: ignore [missing-import]
import matplotlib
matplotlib.use('Agg')
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
# pyrefly: ignore [missing-import]
import numpy as np
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--total_population", type=int, default=100)
    parser.add_argument("--k_vals", nargs='+', type=int, default=[2, 5])
    args = parser.parse_args()

    TOTAL_POPULATION = args.total_population
    k_vals = args.k_vals
    OUTPUT_DIR = "analysis_results"
    omegas = ['0.1', '0.2', '0.3', '0.4', '0.5', '0.6']

    # Load data
    data = {}
    loaded_k_vals = []
    for k in k_vals:
        path = os.path.join(OUTPUT_DIR, f'diagnostic_results_pop_{TOTAL_POPULATION}_k_{k}.json')
        if os.path.exists(path):
            with open(path, 'r') as f:
                data[k] = json.load(f)
                loaded_k_vals.append(k)
        else:
            print(f"Warning: {path} not found. Skipping K={k}")
            
    k_vals = loaded_k_vals

    if not k_vals:
        print("No data found for the given K values. Exiting.")
        exit(1)

    # Setup plot for 6 omegas (2x3 grid)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), sharey=False)
    axes = axes.flatten()
    fig.suptitle(f'Performance Gap: Centralized vs MAPPO (CTDE) Across Scale (Pop={TOTAL_POPULATION})', fontsize=16, fontweight='bold')

    bar_width = 0.35
    index = np.arange(len(k_vals))

    for i, omega in enumerate(omegas):
        ax = axes[i]
        
        cent_means = [data[k]['synthetic'][omega]['methods']['centralized']['mean'] for k in k_vals]
        ctde_means = [data[k]['synthetic'][omega]['methods']['ctde']['mean'] for k in k_vals]
        
        # Error bars (95% CI)
        cent_err = [
            data[k]['synthetic'][omega]['methods']['centralized']['ci_hi'] - data[k]['synthetic'][omega]['methods']['centralized']['mean'] 
            for k in k_vals
        ]
        ctde_err = [
            data[k]['synthetic'][omega]['methods']['ctde']['ci_hi'] - data[k]['synthetic'][omega]['methods']['ctde']['mean'] 
            for k in k_vals
        ]

        rects1 = ax.bar(index, cent_means, bar_width, yerr=cent_err, label='Centralized', capsize=5, color='#1f77b4', alpha=0.8)
        rects2 = ax.bar(index + bar_width, ctde_means, bar_width, yerr=ctde_err, label='MAPPO (CTDE)', capsize=5, color='#ff7f0e', alpha=0.8)

        ax.set_xlabel('Number of Classrooms (K)', fontweight='bold')
        ax.set_ylabel('Mean Joint Reward', fontweight='bold')
        ax.set_title(fr'Risk Preference $\omega$ = {omega}')
        ax.set_xticks(index + bar_width / 2)
        ax.set_xticklabels([f'K={k}' for k in k_vals])
        
        if i == 0:
            ax.legend()

        # Add text annotations for the gap
        for j in range(len(k_vals)):
            gap = ctde_means[j] - cent_means[j]
            # Only annotate if the gap is somewhat meaningful
            ax.text(index[j] + bar_width/2, max(cent_means[j], ctde_means[j]) + max(cent_err[j], ctde_err[j]) + (max(cent_means)*0.05),
                    fr"$\Delta$: {gap:+.1f}", ha='center', va='bottom', fontsize=10, fontweight='bold', 
                    color='red' if gap > 0 else 'black')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = os.path.join(OUTPUT_DIR, f'performance_gap_scale_pop_{TOTAL_POPULATION}.png')
    plt.savefig(out_path, dpi=300)
    print(f"Saved plot to {out_path}")
