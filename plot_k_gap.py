import json
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
# pyrefly: ignore [missing-import]
import numpy as np
import os

OUTPUT_DIR = "analysis_results"
k_vals = [2, 3, 5]
omegas = ['0.1', '0.3', '0.5']

# Load data
data = {k: json.load(open(os.path.join(OUTPUT_DIR, f'diagnostic_results_k_{k}.json'))) for k in k_vals}

# Setup plot
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
fig.suptitle('Performance Gap: Centralized vs MAPPO (CTDE) Across Scale', fontsize=16, fontweight='bold')

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
    ax.set_title(f'Risk Preference $\omega$ = {omega}')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels([f'K={k}' for k in k_vals])
    
    if i == 0:
        ax.legend()

    # Add text annotations for the gap
    for j in range(len(k_vals)):
        gap = ctde_means[j] - cent_means[j]
        # Only annotate if the gap is somewhat meaningful
        ax.text(index[j] + bar_width/2, max(cent_means[j], ctde_means[j]) + max(cent_err[j], ctde_err[j]) + (max(cent_means)*0.05),
                f"$\Delta$: {gap:+.1f}", ha='center', va='bottom', fontsize=10, fontweight='bold', 
                color='red' if gap > 0 else 'black')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(OUTPUT_DIR, 'performance_gap_scale.png'), dpi=300)
print(f"Saved plot to {os.path.join(OUTPUT_DIR, 'performance_gap_scale.png')}")
