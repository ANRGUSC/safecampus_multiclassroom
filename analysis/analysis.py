import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Define CSV file locations
agents_files = {
    "Myopic": {
        "global_means": "myopic_global_means.csv",
        "infected_allowed": "myopic_infected_allowed.csv"
    },
    "DQN-TD": {
        "global_means": "td_dqn_global_summary.csv",
        "infected_allowed": "td_dqn_inf_allowed_summary.csv"
    },
    "DQN-MC": {
        "global_means": "mc_dqn_global_summary.csv",
        "infected_allowed": "mc_dqn_inf_allowed_summary.csv"
    }
}

# Read, merge, tag, and aggregate
dfs = []
for agent, paths in agents_files.items():
    df_g = pd.read_csv(paths["global_means"])
    df_ia = pd.read_csv(paths["infected_allowed"])
    merged = pd.merge(df_g, df_ia, on="gamma", how="inner")
    merged["agent"] = agent
    dfs.append(merged)
combined = pd.concat(dfs, ignore_index=True)
combined = combined.rename(columns={
    "global_mean_reward": "global_mean",
    combined.columns[combined.columns.str.contains("infected", case=False)][0]: "infected",
    combined.columns[combined.columns.str.contains("allowed", case=False)][0]: "allowed"
})
combined = combined.groupby(["gamma", "agent"], as_index=False).agg({
    "global_mean": "mean",
    "infected": "mean",
    "allowed": "mean"
})
combined.to_csv("combined_summary_100_3_2_03.csv", index=False)

# Plot 1: Global Mean per Agent by Gamma
pivot_gm = combined.pivot(index="gamma", columns="agent", values="global_mean")
fig1, ax1 = plt.subplots(figsize=(8, 5))
for agent in pivot_gm.columns:
    ax1.plot(pivot_gm.index, pivot_gm[agent], marker='o', label=agent)
ax1.set_title("Global Mean Reward per Agent by Gamma")
ax1.set_xlabel("Gamma")
ax1.set_ylabel("Global Mean Reward")
ax1.legend(title="Agent")
fig1.tight_layout()
fig1.savefig("global_mean_by_gamma_100_3_2_03.png", dpi=300, bbox_inches='tight')

# Plot 2: Infected/Allowed bar subplots per Agent
agents = combined['agent'].unique()
gammas = sorted(combined['gamma'].unique())
x = np.arange(len(gammas))
width = 0.35
colors = {'infected': 'tab:blue', 'allowed': 'tab:orange'}

fig2, axes = plt.subplots(1, len(agents), figsize=(5 * len(agents), 5), sharey=True)
if len(agents) == 1:
    axes = [axes]

for ax, agent in zip(axes, agents):
    sub = combined[combined['agent'] == agent].set_index('gamma').reindex(gammas)
    ax.bar(x - width/2, sub['infected'], width, label='Infected', color=colors['infected'])
    ax.bar(x + width/2, sub['allowed'], width, label='Allowed', color=colors['allowed'])
    ax.set_title(agent)
    ax.set_xticks(x)
    ax.set_xticklabels(gammas)
    ax.set_xlabel("Gamma")
    ax.legend()
axes[0].set_ylabel("Value")
fig2.suptitle("Mean Infected and Allowed per Gamma by Agent")
fig2.tight_layout(rect=[0, 0.03, 1, 0.95])
fig2.savefig("infected_allowed_subplots_100_2_2_03.png", dpi=300, bbox_inches='tight')

