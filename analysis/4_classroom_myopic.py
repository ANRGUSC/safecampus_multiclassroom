import pandas as pd

# Define file paths for Myopic and DQN-MC results
files = {
    "Myopic": "myopic_global_means_4_class.csv",
    "DQN-MC": "mc_dqn_global_summary_4_class.csv",
}

# Read, tag, and collect DataFrames
dfs = []
for agent, path in files.items():
    df = pd.read_csv(path)
    df["agent"] = agent
    dfs.append(df)

# Concatenate and select relevant columns
combined = pd.concat(dfs, ignore_index=True)[["gamma", "agent", "global_mean_reward"]]

# Pivot to have gamma as rows and agents as columns
table = combined.pivot(index="gamma", columns="agent", values="global_mean_reward").reset_index()

# Save aggregated result
table.to_csv("4_classroom_global_mean_myopic_dqnmc.csv", index=False)

# Display the table to the user
