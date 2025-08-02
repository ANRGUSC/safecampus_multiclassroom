import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import os

# ---------- Hyperparameter dictionaries for each scenario and algorithm ----------

# Default setting - Scenario 0 (100 Class size, 0.3 shared, 3 action levels, 2 classrooms)
ctde_mc_default = {
   0.1: {"learning_rate": 0.1, "hidden_dim": 16, "hidden_layers": 2},
   0.2: {"learning_rate": 0.1, "hidden_dim": 16, "hidden_layers": 2},
   0.3: {"learning_rate": 0.1, "hidden_dim": 16, "hidden_layers": 2},
   0.4: {"learning_rate": 0.1, "hidden_dim": 128, "hidden_layers": 3},
   0.5: {"learning_rate": 0.1, "hidden_dim": 128, "hidden_layers": 3},
   0.6: {"learning_rate": 0.01, "hidden_dim": 64, "hidden_layers": 3},
}

ctde_td_default = {
   0.1: {"learning_rate": 0.001, "hidden_dim": 64, "hidden_layers": 3},
   0.2: {"learning_rate": 0.001, "hidden_dim": 64, "hidden_layers": 3},
   0.3: {"learning_rate": 0.001, "hidden_dim": 64, "hidden_layers": 3},
   0.4: {"learning_rate": 0.001, "hidden_dim": 128, "hidden_layers": 3},
   0.5: {"learning_rate": 0.2, "hidden_dim": 256, "hidden_layers": 3},
   0.6: {"learning_rate": 0.001, "hidden_dim": 64, "hidden_layers": 3},
}

centralized_mc_default = {
   0.1: {"learning_rate": 0.05, "hidden_dim": 128, "hidden_layers": 1},
   0.2: {"learning_rate": 0.03, "hidden_dim": 128, "hidden_layers": 1},
   0.3: {"learning_rate": 0.03, "hidden_dim": 64, "hidden_layers": 1},
   0.4: {"learning_rate": 0.01, "hidden_dim": 128, "hidden_layers": 2},
   0.5: {"learning_rate": 0.05, "hidden_dim": 64, "hidden_layers": 2},
   0.6: {"learning_rate": 0.05, "hidden_dim": 64, "hidden_layers": 2},
}

centralized_td_double_default = {
   0.1: {"learning_rate": 0.001, "hidden_dim": 64, "hidden_layers": 2},
   0.2: {"learning_rate": 0.001, "hidden_dim": 128, "hidden_layers": 2},
   0.3: {"learning_rate": 0.001, "hidden_dim": 128, "hidden_layers": 2},
   0.4: {"learning_rate": 0.001, "hidden_dim": 128, "hidden_layers": 2},
   0.5: {"learning_rate": 0.001, "hidden_dim": 128, "hidden_layers": 2},
   0.6: {"learning_rate": 0.01, "hidden_dim": 64, "hidden_layers": 2},
}


# Scenario 1 - 50 Class size, 0.3 shared, 3 action levels, 2 classrooms
ctde_td_scenario1 = {
   0.1: {"learning_rate": 0.001, "hidden_dim": 128, "hidden_layers": 3},
   0.2: {"learning_rate": 0.001, "hidden_dim": 128, "hidden_layers": 3},
   0.3: {"learning_rate": 0.001, "hidden_dim": 128, "hidden_layers": 2},
   0.4: {"learning_rate": 0.001, "hidden_dim": 128, "hidden_layers": 3},
   0.5: {"learning_rate": 0.001, "hidden_dim": 128, "hidden_layers": 3},
   0.6: {"learning_rate": 0.001, "hidden_dim": 128, "hidden_layers": 3},
}

ctde_mc_scenario1 = {
   0.1: {"learning_rate": 0.1, "hidden_dim": 32, "hidden_layers": 1},
   0.2: {"learning_rate": 0.01, "hidden_dim": 32, "hidden_layers": 2},
   0.3: {"learning_rate": 0.1, "hidden_dim": 16, "hidden_layers": 3},
   0.4: {"learning_rate": 0.01, "hidden_dim": 128, "hidden_layers": 2},
   0.5: {"learning_rate": 0.01, "hidden_dim": 64, "hidden_layers": 3},
   0.6: {"learning_rate": 0.01, "hidden_dim": 64, "hidden_layers": 2},
}

centralized_td_scenario1 = {
   0.1: {"learning_rate": 0.001, "hidden_dim": 128, "hidden_layers": 2},
   0.2: {"learning_rate": 0.001, "hidden_dim": 64, "hidden_layers": 2},
   0.3: {"learning_rate": 0.001, "hidden_dim": 64, "hidden_layers": 2},
   0.4: {"learning_rate": 0.001, "hidden_dim": 64, "hidden_layers": 3},
   0.5: {"learning_rate": 0.001, "hidden_dim": 64, "hidden_layers": 3},
   0.6: {"learning_rate": 0.01, "hidden_dim": 64, "hidden_layers": 2},
}

centralized_mc_scenario1 = {
   0.1: {"learning_rate": 0.03, "hidden_dim": 128, "hidden_layers": 1},
   0.2: {"learning_rate": 0.03, "hidden_dim": 128, "hidden_layers": 1},
   0.3: {"learning_rate": 0.03, "hidden_dim": 128, "hidden_layers": 1},
   0.4: {"learning_rate": 0.03, "hidden_dim": 128, "hidden_layers": 1},
   0.5: {"learning_rate": 0.03, "hidden_dim": 128, "hidden_layers": 1},
   0.6: {"learning_rate": 0.01, "hidden_dim": 64, "hidden_layers": 2},
}


# Scenario 2 - 100 Class size, 0.3 shared, 5 action levels, 2 classrooms
ctde_td_scenario2 = {
   0.1: {"learning_rate": 0.01, "hidden_dim": 64, "hidden_layers": 1},
   0.2: {"learning_rate": 0.001, "hidden_dim": 64, "hidden_layers": 3},
   0.3: {"learning_rate": 0.001, "hidden_dim": 64, "hidden_layers": 3},
   0.4: {"learning_rate": 0.00001, "hidden_dim": 256, "hidden_layers": 3},
   0.5: {"learning_rate": 0.000001, "hidden_dim": 256, "hidden_layers": 4},
   0.6: {"learning_rate": 0.001, "hidden_dim": 128, "hidden_layers": 3},
}

ctde_mc_scenario2 = {
   0.1: {"learning_rate": 0.01, "hidden_dim": 64, "hidden_layers": 2},
   0.2: {"learning_rate": 0.01, "hidden_dim": 64, "hidden_layers": 2},
   0.3: {"learning_rate": 0.01, "hidden_dim": 64, "hidden_layers": 2},
   0.4: {"learning_rate": 0.01, "hidden_dim": 32, "hidden_layers": 2},
   0.5: {"learning_rate": 0.01, "hidden_dim": 40, "hidden_layers": 2},
   0.6: {"learning_rate": 0.01, "hidden_dim": 16, "hidden_layers": 3},
}

centralized_td_scenario2 = {
   0.1: {"learning_rate": 0.001, "hidden_dim": 128, "hidden_layers": 2},
   0.2: {"learning_rate": 0.001, "hidden_dim": 128, "hidden_layers": 2},
   0.3: {"learning_rate": 0.001, "hidden_dim": 128, "hidden_layers": 2},
   0.4: {"learning_rate": 0.001, "hidden_dim": 128, "hidden_layers": 4},
   0.5: {"learning_rate": 0.0001, "hidden_dim": 128, "hidden_layers": 4},
   0.6: {"learning_rate": 0.01, "hidden_dim": 128, "hidden_layers": 2},
}

centralized_mc_scenario2 = {
   0.1: {"learning_rate": 0.03, "hidden_dim": 128, "hidden_layers": 1},
   0.2: {"learning_rate": 0.05, "hidden_dim": 128, "hidden_layers": 1},
   0.3: {"learning_rate": 0.03, "hidden_dim": 128, "hidden_layers": 1},
   0.4: {"learning_rate": 0.03, "hidden_dim": 256, "hidden_layers": 3},
   0.5: {"learning_rate": 0.05, "hidden_dim": 128, "hidden_layers": 3},
   0.6: {"learning_rate": 0.01, "hidden_dim": 64, "hidden_layers": 1},
}


# Scenario 3 - 100 Class size, 0.3 shared, 3 action levels, 3 classrooms
ctde_td_scenario3 = {
   0.1: {"learning_rate": 0.001, "hidden_dim": 128, "hidden_layers": 3},
   0.2: {"learning_rate": 0.001, "hidden_dim": 128, "hidden_layers": 3},
   0.3: {"learning_rate": 0.001, "hidden_dim": 128, "hidden_layers": 3},
   0.4: {"learning_rate": 0.001, "hidden_dim": 256, "hidden_layers": 3},
   0.5: {"learning_rate": 0.0001, "hidden_dim": 120, "hidden_layers": 4},
   0.6: {"learning_rate": 0.001, "hidden_dim": 128, "hidden_layers": 3},
}

ctde_mc_scenario3 = {
   0.1: {"learning_rate": 0.1, "hidden_dim": 8, "hidden_layers": 2},
   0.2: {"learning_rate": 0.1, "hidden_dim": 32, "hidden_layers": 2},
   0.3: {"learning_rate": 0.1, "hidden_dim": 16, "hidden_layers": 3},
   0.4: {"learning_rate": 0.001, "hidden_dim": 64, "hidden_layers": 3},
   0.5: {"learning_rate": 0.001, "hidden_dim": 64, "hidden_layers": 3},
   0.6: {"learning_rate": 0.01, "hidden_dim": 64, "hidden_layers": 3},
}

centralized_td_scenario3 = {
   0.1: {"learning_rate": 0.001, "hidden_dim": 64 , "hidden_layers": 2},
   0.2: {"learning_rate": 0.001, "hidden_dim": 64, "hidden_layers": 2},
   0.3: {"learning_rate": 0.001, "hidden_dim": 64, "hidden_layers": 3},
   0.4: {"learning_rate": 0.001, "hidden_dim": 64, "hidden_layers": 4},
   0.5: {"learning_rate": 0.001, "hidden_dim": 128, "hidden_layers": 4},
   0.6: {"learning_rate": 0.01, "hidden_dim": 64, "hidden_layers": 3},
}

centralized_mc_scenario3 = {
   0.1: {"learning_rate": 0.03, "hidden_dim": 128, "hidden_layers": 1},
   0.2: {"learning_rate": 0.03, "hidden_dim": 128, "hidden_layers": 1},
   0.3: {"learning_rate": 0.05, "hidden_dim": 64, "hidden_layers": 2},
   0.4: {"learning_rate": 0.01, "hidden_dim": 64, "hidden_layers": 4},
   0.5: {"learning_rate": 0.05, "hidden_dim": 64, "hidden_layers": 5},
   0.6: {"learning_rate": 0.05, "hidden_dim": 64, "hidden_layers": 2},
}


# Scenario 4 - 100 Class size, 0.8 shared, 3 action levels, 2 classrooms

ctde_td_scenario4 = {
   0.1: {"learning_rate": 0.001, "hidden_dim": 128, "hidden_layers": 3},
   0.2: {"learning_rate": 0.001, "hidden_dim": 128, "hidden_layers": 3},
   0.3: {"learning_rate": 0.0001, "hidden_dim": 128, "hidden_layers": 4},
   0.4: {"learning_rate": 1e-5, "hidden_dim": 128, "hidden_layers": 3},
   0.5: {"learning_rate": 1e-5, "hidden_dim": 128, "hidden_layers": 3},
   0.6: {"learning_rate": 0.001, "hidden_dim": 128, "hidden_layers": 3},
}

ctde_mc_scenario4 = {
   0.1: {"learning_rate": 0.03, "hidden_dim": 64, "hidden_layers": 3},
   0.2: {"learning_rate": 0.03, "hidden_dim": 64, "hidden_layers": 3},
   0.3: {"learning_rate": 0.03, "hidden_dim": 64, "hidden_layers": 3},
   0.4: {"learning_rate": 0.03, "hidden_dim": 128, "hidden_layers": 3},
   0.5: {"learning_rate": 0.03, "hidden_dim": 116, "hidden_layers": 3},
   0.6: {"learning_rate": 0.03, "hidden_dim": 64, "hidden_layers": 3},
}

centralized_mc_scenario4 = {
   0.1: {"learning_rate": 0.05, "hidden_dim": 128, "hidden_layers": 3},
   0.2: {"learning_rate": 0.05, "hidden_dim": 128, "hidden_layers": 1},
   0.3: {"learning_rate": 0.05, "hidden_dim": 64, "hidden_layers": 1},
   0.4: {"learning_rate": 0.05, "hidden_dim": 64, "hidden_layers": 3},
   0.5: {"learning_rate": 0.05, "hidden_dim": 128, "hidden_layers": 3},
   0.6: {"learning_rate": 0.01, "hidden_dim": 64, "hidden_layers": 3},
}

# Predicted Centralized TD for Scenario 4 (missing in data)
centralized_td_scenario4 = {
    0.1: {"learning_rate": 0.001, "hidden_dim": 64, "hidden_layers": 2},
    0.2: {"learning_rate": 0.001, "hidden_dim": 128, "hidden_layers": 2},
    0.3: {"learning_rate": 0.001, "hidden_dim": 128, "hidden_layers": 2},
    0.4: {"learning_rate": 0.001, "hidden_dim": 64, "hidden_layers": 3},
    0.5: {"learning_rate": 0.001, "hidden_dim": 64, "hidden_layers": 3},
    0.6: {"learning_rate": 0.001, "hidden_dim": 64, "hidden_layers": 3},

}

# ----------------------- Plotting Function -----------------------


# --- Function 1: Summary Table with Conditional Formatting ---

# --- Function 1: Summary Table with Conditional Formatting ---

def plot_hyperparam_summary_table(hyperparams_dicts, scenario_names, algo_names, weights):
    rows = []
    for scenario, algo, hparams in zip(scenario_names, algo_names, hyperparams_dicts):
        for w in weights:
            entry = hparams[w]
            rows.append({
                "Scenario": scenario,
                "Algorithm": algo,
                "Weight": w,
                "Learning Rate": entry["learning_rate"],
                "Hidden Dim": entry["hidden_dim"],
                "Hidden Layers": entry["hidden_layers"],
            })
    df = pd.DataFrame(rows)

    df = df.sort_values(by=["Scenario", "Algorithm", "Weight"])

    def color_scale(s):
        norm = (s - s.min()) / (s.max() - s.min() + 1e-8)
        return ['background-color: rgba(0, 120, 215, {:.2f})'.format(v) for v in norm]

    styled = (df.style
              .apply(color_scale, subset=["Learning Rate"])
              .apply(color_scale, subset=["Hidden Dim"])
              .apply(color_scale, subset=["Hidden Layers"])
              .set_caption("Summary of Neural Network Hyperparameters Across Scenarios and Algorithms")
              .format({
                  "Learning Rate": "{:.5f}",
                  "Hidden Dim": "{:.0f}",
                  "Hidden Layers": "{:.0f}"
              })
              .set_properties(**{'text-align': 'center'})
              .set_table_styles([dict(selector='th', props=[('text-align', 'center')])])
              )
    return styled


# --- Function 2: Radar Chart ---


# def plot_method_transitions_all_weights(
#     method_default,
#     method_scenarios,
#     method_name,
#     weights,
#     scenario_labels,
#     save_dir
# ):
#     import os
#     import numpy as np
#     import matplotlib.pyplot as plt
#     import seaborn as sns
#
#     os.makedirs(save_dir, exist_ok=True)
#
#     # Compute complexity score and collect all unique pairs to sort rows
#     def get_complexity(hparams_dict):
#         complexity_list = []
#         for w in weights:
#             hd = hparams_dict[w]["hidden_dim"]
#             hl = hparams_dict[w]["hidden_layers"]
#             complexity = hd * hl
#             complexity_list.append((hd, hl, complexity))
#         return complexity_list
#
#     # Get combined unique complexity values from default + scenarios
#     all_complexities = set()
#     def add_complexities(hparams_dict):
#         for w in weights:
#             hd = hparams_dict[w]["hidden_dim"]
#             hl = hparams_dict[w]["hidden_layers"]
#             comp = hd * hl
#             all_complexities.add((hd, hl, comp))
#     add_complexities(method_default)
#     for scenario in method_scenarios:
#         add_complexities(scenario)
#
#     # Sort by complexity ascending
#     sorted_complexities = sorted(list(all_complexities), key=lambda x: x[2])
#     complexity_scores = [c[2] for c in sorted_complexities]
#
#     # Build matrix (rows = complexity score, cols = weights) for each dict
#     def build_matrix(hparams_dict):
#         mat = np.full((len(sorted_complexities), len(weights)), np.nan)
#         for col_i, w in enumerate(weights):
#             hd = hparams_dict[w]["hidden_dim"]
#             hl = hparams_dict[w]["hidden_layers"]
#             lr = hparams_dict[w]["learning_rate"]
#             comp = hd * hl
#             # Find row index for this complexity
#             try:
#                 row_i = next(i for i, val in enumerate(sorted_complexities) if val[2] == comp and val[0] == hd and val[1] == hl)
#                 mat[row_i, col_i] = lr
#             except StopIteration:
#                 pass
#         return mat
#
#     default_matrix = build_matrix(method_default)
#     scenario_matrices = [build_matrix(scenario) for scenario in method_scenarios]
#
#     epsilon = 1e-10
#     all_matrices = [default_matrix] + scenario_matrices
#     all_vals = np.hstack([np.log10(mat.flatten() + epsilon) for mat in all_matrices])
#     vmin = np.nanmin(all_vals)
#     vmax = np.nanmax(all_vals)
#
#     # Setup figure: 4 rows (scenarios) x 2 cols (default and scenario), bigger figure for clarity
#     fig, axs = plt.subplots(4, 2, figsize=(16, 20), sharex=True, sharey=True,
#                             gridspec_kw={"width_ratios": [1, 1], "wspace": 0.1, "hspace": 0.2})
#
#     for i in range(4):
#         # Left col: Default heatmap
#         sns.heatmap(
#             np.log10(default_matrix + epsilon),
#             ax=axs[i, 0],
#             cmap="coolwarm",
#             vmin=vmin, vmax=vmax,
#             cbar=False,
#             xticklabels=weights,
#             yticklabels=True,
#             square=True,
#             linewidths=0.7,
#             linecolor="gray",
#             annot=default_matrix.round(5),
#             fmt='.3f',
#             annot_kws={"size": 10},
#         )
#         axs[i, 0].set_title(f"Default (Weight →) / {scenario_labels[i]}", fontsize=14)
#         axs[i, 0].set_xlabel("Weight" if i == 3 else "")
#         axs[i, 0].set_ylabel("Complexity (Hidden Dim × Layers)" if i == 0 else "")
#         axs[i, 0].tick_params(axis='y', labelsize=10)
#         axs[i, 0].set_yticks(np.arange(len(complexity_scores)) + 0.5)
#         axs[i, 0].set_yticklabels(complexity_scores if i == 0 else [], fontsize=10)
#
#         # Right col: Scenario heatmap
#         sns.heatmap(
#             np.log10(scenario_matrices[i] + epsilon),
#             ax=axs[i, 1],
#             cmap="coolwarm",
#             vmin=vmin, vmax=vmax,
#             cbar=False,
#             xticklabels=weights,
#             yticklabels=True,
#             square=True,
#             linewidths=0.7,
#             linecolor="gray",
#             annot=scenario_matrices[i].round(5),
#             fmt='.3f',
#             annot_kws={"size": 10},
#         )
#         axs[i, 1].set_title(f"{scenario_labels[i]} (Weight →)", fontsize=14)
#         axs[i, 1].set_xlabel("Weight" if i == 3 else "")
#         axs[i, 1].set_ylabel("Complexity (Hidden Dim × Layers)" if i == 0 else "")
#         axs[i, 1].tick_params(axis='y', labelsize=10)
#         axs[i, 1].set_yticks(np.arange(len(complexity_scores)) + 0.5)
#         axs[i, 1].set_yticklabels(complexity_scores if i == 0 else [], fontsize=10)
#
#         # Rotate x-ticks for better readability
#         axs[i, 0].tick_params(axis='x', rotation=45)
#         axs[i, 1].tick_params(axis='x', rotation=45)
#
#     # Horizontal colorbar below all subplots
#     cbar_ax = fig.add_axes([0.25, 0.02, 0.5, 0.03])  # [left, bottom, width, height]
#     norm = plt.Normalize(vmin=vmin, vmax=vmax)
#     sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=norm)
#     sm.set_array([])
#     cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
#     cbar.set_label("Log10 Learning Rate", fontsize=14)
#     cbar.ax.tick_params(labelsize=12)
#
#     fig.suptitle(f"Hyperparameter Transitions: Default → Scenario (Method: {method_name})", fontsize=18)
#     plt.tight_layout(rect=[0, 0.06, 1, 0.96])
#
#     filename = os.path.join(save_dir, f"{method_name.lower().replace(' ', '_')}_hyperparam_transitions.png")
#     plt.savefig(filename, dpi=300)
#     plt.close()


def plot_method_transitions_all_weights(
    method_default,
    method_scenarios,
    method_name,
    weights,
    scenario_labels,
    save_dir
):
    os.makedirs(save_dir, exist_ok=True)

    # Compute complexity score and collect all unique pairs to sort rows
    def get_complexity(hparams_dict):
        complexities = []
        for w in weights:
            hd = hparams_dict[w]["hidden_dim"]
            hl = hparams_dict[w]["hidden_layers"]
            comp = hd * hl
            complexities.append((hd, hl, comp))
        return complexities

    # Get combined unique complexity values from default + scenarios for y-axis
    all_complexities = set()
    def add_complexities(hparams_dict):
        for w in weights:
            hd = hparams_dict[w]["hidden_dim"]
            hl = hparams_dict[w]["hidden_layers"]
            comp = hd * hl
            all_complexities.add((hd, hl, comp))
    add_complexities(method_default)
    for scenario in method_scenarios:
        add_complexities(scenario)

    # Sort by complexity ascending for y-axis ordering
    sorted_complexities = sorted(list(all_complexities), key=lambda x: x[2])
    complexity_scores = [c[2] for c in sorted_complexities]

    # Prepare plot data for default and scenarios
    def prepare_data(hparams_dict, label):
        data = []
        for w in weights:
            hd = hparams_dict[w]["hidden_dim"]
            hl = hparams_dict[w]["hidden_layers"]
            comp = hd * hl
            lr = hparams_dict[w]["learning_rate"]
            data.append({
                "Weight": w,
                "Hidden Dim": hd,
                "Hidden Layers": hl,
                "Complexity": comp,
                "Learning Rate": lr,
                "Log10 LR": np.log10(lr),
                "Label": label
            })
        return data

    default_data = prepare_data(method_default, "Default")
    scenarios_data = []
    for i, scenario in enumerate(method_scenarios):
        scenarios_data += prepare_data(scenario, f"Scenario {i+1}")

    # Combine all data for plotting
    import pandas as pd
    df_default = pd.DataFrame(default_data)
    df_scenarios = pd.DataFrame(scenarios_data)

    # Set up plot grid: 4 rows (scenarios) x 2 cols (Default / Scenario)
    fig, axs = plt.subplots(4, 2, figsize=(14, 20), sharex=True, sharey=True)

    # Define bubble size scaling
    min_lr = min(df_default["Learning Rate"].min(), df_scenarios["Learning Rate"].min())
    max_lr = max(df_default["Learning Rate"].max(), df_scenarios["Learning Rate"].max())

    def scale_bubble_size(lr, min_size=100, max_size=1000):
        # Scale learning rate linearly between min and max bubble sizes
        return min_size + (lr - min_lr) / (max_lr - min_lr + 1e-10) * (max_size - min_size)

    # Plot for each scenario row
    for i in range(4):
        # Default subplot
        ax_def = axs[i, 0]
        data_def = df_default.copy()

        # Scenario subplot
        ax_scen = axs[i, 1]
        data_scen = df_scenarios[df_scenarios["Label"] == f"Scenario {i+1}"]

        for ax, data, title in zip(
            [ax_def, ax_scen],
            [data_def, data_scen],
            [f"Default (Weight →) / {scenario_labels[i]}", f"Scenario {i+1} (Weight →)"]
        ):
            # Bubble size
            sizes = scale_bubble_size(data["Learning Rate"].values)

            # Scatter bubble plot
            scatter = ax.scatter(
                data["Weight"],
                data["Complexity"],
                s=sizes,
                c=data["Log10 LR"],
                cmap="coolwarm",
                edgecolors='k',
                alpha=0.75
            )
            # Annotate bubble with learning rate value
            for _, row in data.iterrows():
                ax.text(
                    row["Weight"],
                    row["Complexity"],
                    f"{row['Learning Rate']:.3f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="black"
                )

            ax.set_title(title, fontsize=14)
            ax.set_xlabel("Weight" if i == 3 else "")
            if ax == axs[0, 0]:
                ax.set_ylabel("Network Complexity\n(Hidden Dim × Layers)", fontsize=12)
            else:
                ax.set_ylabel("")

            ax.set_yticks(complexity_scores)
            ax.set_yticklabels(complexity_scores, fontsize=10)
            ax.grid(True)
            ax.set_xlim(min(weights) - 0.02, max(weights) + 0.02)

    # Add single horizontal colorbar below all plots
    cbar_ax = fig.add_axes([0.25, 0.02, 0.5, 0.03])  # [left, bottom, width, height]
    norm = plt.Normalize(vmin=df_default["Log10 LR"].min(), vmax=df_default["Log10 LR"].max())
    sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label("Log10 Learning Rate", fontsize=14)
    cbar.ax.tick_params(labelsize=12)

    fig.suptitle(f"Hyperparameter Transitions: Default → Scenario (Method: {method_name})", fontsize=18)
    plt.tight_layout(rect=[0, 0.06, 1, 0.96])

    filename = os.path.join(save_dir, f"{method_name.lower().replace(' ', '_')}_hyperparam_transitions_bubble.png")
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved bubble plot hyperparameter transition figure: {filename}")



def plot_radar_charts_multiple(methods_hyperparams, methods_labels, scenario_labels,
                               agg_func=np.mean, fig_title="Neural Network Hyperparameters Radar Charts"):
    """
    Plot multiple radar charts in a grid for different methods,
    each showing hyperparams aggregated over scenarios (including default).

    Params:
    - methods_hyperparams: list of list of dicts (outer list: methods, inner list: scenarios per method)
    - methods_labels: list of method names (strings)
    - scenario_labels: list of scenario names (strings), same for all methods
    - agg_func: aggregation function over weights (default mean)
    - fig_title: overall figure title

    Saves:
    - radar_chart_all_methods.png
    """
    num_methods = len(methods_hyperparams)
    categories = ['Learning Rate (log norm)', 'Hidden Dim (norm)', 'Hidden Layers (norm)']
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    # Create subplots (2x2 grid)
    fig, axs = plt.subplots(2, 2, subplot_kw=dict(polar=True), figsize=(16, 14))
    axs = axs.flatten()

    for m_i, (method_hparams_list, method_label) in enumerate(zip(methods_hyperparams, methods_labels)):
        agg_data = []
        for hparams in method_hparams_list:
            lr_vals = [v["learning_rate"] for v in hparams.values()]
            hd_vals = [v["hidden_dim"] for v in hparams.values()]
            hl_vals = [v["hidden_layers"] for v in hparams.values()]
            agg_data.append([agg_func(lr_vals), agg_func(hd_vals), agg_func(hl_vals)])

        agg_data = np.array(agg_data)

        epsilon = 1e-8
        norm_lr = np.log10(agg_data[:, 0] + epsilon)
        norm_lr = (norm_lr - norm_lr.min()) / (norm_lr.max() - norm_lr.min() + epsilon)
        norm_hd = (agg_data[:, 1] - agg_data[:, 1].min()) / (agg_data[:, 1].max() - agg_data[:, 1].min() + epsilon)
        norm_hl = (agg_data[:, 2] - agg_data[:, 2].min()) / (agg_data[:, 2].max() - agg_data[:, 2].min() + epsilon)
        data_norm = np.stack([norm_lr, norm_hd, norm_hl], axis=1)

        ax = axs[m_i]
        for i, scenario_label in enumerate(scenario_labels):
            values = data_norm[i].tolist()
            values += values[:1]
            ax.plot(angles, values, label=scenario_label)
            ax.fill(angles, values, alpha=0.25)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=11)
        ax.set_yticklabels([])
        ax.set_title(method_label, fontsize=14)
        ax.grid(True)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)

    fig.suptitle(fig_title, fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("radar_chart_all_methods.png")
    plt.show()


# --- Function 3: Side-by-side comparison plots ---

def plot_hyperparameters_comparison(scenario_name, algo_name,
                                    default_hyperparams, scenario_hyperparams):
    weights = sorted(default_hyperparams.keys())

    default_hidden_dims = [default_hyperparams[w]["hidden_dim"] for w in weights]
    default_hidden_layers = [default_hyperparams[w]["hidden_layers"] for w in weights]

    scenario_hidden_dims = [scenario_hyperparams[w]["hidden_dim"] for w in weights]
    scenario_hidden_layers = [scenario_hyperparams[w]["hidden_layers"] for w in weights]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    color1 = 'tab:blue'
    ax1.set_xlabel('Weight value')
    ax1.set_ylabel('Hidden Dim', color=color1)
    ax1.plot(weights, default_hidden_dims, marker='o', linestyle='-', color=color1, label='Hidden Dim (Default)')
    ax1.plot(weights, scenario_hidden_dims, marker='o', linestyle='--', color=color1, label=f'Hidden Dim ({scenario_name})')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_xticks(weights)

    ax2 = ax1.twinx()

    color2 = 'tab:green'
    ax2.set_ylabel('Hidden Layers', color=color2)
    ax2.plot(weights, default_hidden_layers, marker='s', linestyle='-', color=color2, label='Hidden Layers (Default)')
    ax2.plot(weights, scenario_hidden_layers, marker='s', linestyle='--', color=color2, label=f'Hidden Layers ({scenario_name})')
    ax2.tick_params(axis='y', labelcolor=color2)

    plt.title(f"Hyperparameter Architecture Comparison: Default vs {scenario_name} - {algo_name}")
    fig.tight_layout()

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')

    # Save figure with descriptive filename
    filename = f"hyperparam_comparison_{scenario_name.replace(' ', '_')}_{algo_name.replace(' ', '_')}.png"
    plt.savefig(filename)
    print(f"Saved plot: {filename}")

    plt.show()


# ----------------------------



def plot_complexity_vs_learning_rate(hyperparams_dicts, scenario_names, algo_names, weights,
                                     save_path="complexity_vs_learning_rate.png"):
    """
    Scatter plot showing relationship between learning rate and model complexity (hidden_dim * hidden_layers).
    Saves plot to the specified file.

    Params:
    - hyperparams_dicts: list of dicts keyed by weight, each value dict with 'learning_rate', 'hidden_dim', 'hidden_layers'
    - scenario_names: list of scenario names (one per dict)
    - algo_names: list of algorithm names (one per dict)
    - weights: list of weights (e.g., [0.1, 0.2, ..., 0.6])
    - save_path: filename to save the plot

    Displays a scatter plot with:
    - X axis: learning rate (log scale)
    - Y axis: complexity = hidden_dim * hidden_layers
    - Points colored by algorithm and shaped by scenario
    """
    plt.figure(figsize=(12, 8))

    markers = ['o', 's', 'D', '^', 'v', 'P', '*']  # for scenarios
    algo_colors = {
        "CTDE TD": 'tab:blue',
        "CTDE MC": 'tab:orange',
        "Centralized MC": 'tab:green',
        "Centralized TD": 'tab:red'
    }

    unique_scenarios = sorted(set(scenario_names))
    unique_algos = sorted(set(algo_names))

    for hparams, scenario, algo in zip(hyperparams_dicts, scenario_names, algo_names):
        marker = markers[unique_scenarios.index(scenario) % len(markers)]
        color = algo_colors.get(algo, 'black')

        for w in weights:
            entry = hparams[w]
            lr = entry["learning_rate"]
            complexity = entry["hidden_dim"] * entry["hidden_layers"]
            plt.scatter(lr, complexity, color=color, marker=marker, s=100,
                        label=f"{algo} - {scenario}" if w == weights[0] else "", alpha=0.7)

    plt.xscale('log')
    plt.xlabel("Learning Rate (log scale)")
    plt.ylabel("Model Complexity (Hidden Dim × Hidden Layers)")
    plt.title("Neural Network Complexity vs Learning Rate Across Scenarios and Algorithms")

    # Create custom legend handles for algo colors and scenario markers
    legend_elements = []
    for algo in unique_algos:
        legend_elements.append(Line2D([0], [0], marker='o', color='w', label=algo,
                                      markerfacecolor=algo_colors.get(algo, 'black'), markersize=10))
    for i, scenario in enumerate(unique_scenarios):
        legend_elements.append(Line2D([0], [0], marker=markers[i % len(markers)], color='k', label=scenario,
                                      linestyle='None', markersize=10))

    plt.legend(handles=legend_elements, title="Algorithms (color) and Scenarios (marker)", bbox_to_anchor=(1.05, 1),
               loc='upper left')
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved plot to {save_path}")
    plt.show()


def plot_hyperparam_heatmaps(hyperparams_dicts, scenario_names, algo_names, weights):
    """
    Plot heatmaps of hyperparameters for each algorithm, combining all scenarios.

    Params:
      - hyperparams_dicts: list of dicts keyed by weight, each value is dict with 'learning_rate', 'hidden_dim', 'hidden_layers'
      - scenario_names: list of scenario names (one per hyperparams_dict)
      - algo_names: list of algorithm names (one per hyperparams_dict)
      - weights: list of weights (x-axis)

    Shows:
      - One figure per algorithm with 3 heatmaps: learning rate, hidden dim, hidden layers
      - Rows = scenarios, columns = weights
    """
    # Group hyperparams by algorithm, collect scenarios per algorithm
    algo_groups = {}
    for scenario, algo, hparams in zip(scenario_names, algo_names, hyperparams_dicts):
        if algo not in algo_groups:
            algo_groups[algo] = []
        algo_groups[algo].append((scenario, hparams))

    for algo, scenario_hparams_list in algo_groups.items():
        # Prepare data frames for each hyperparam
        lr_df = pd.DataFrame(index=[s for s, _ in scenario_hparams_list], columns=weights, dtype=float)
        hd_df = pd.DataFrame(index=[s for s, _ in scenario_hparams_list], columns=weights, dtype=int)
        hl_df = pd.DataFrame(index=[s for s, _ in scenario_hparams_list], columns=weights, dtype=int)

        for scenario, hparams in scenario_hparams_list:
            for w in weights:
                lr_df.loc[scenario, w] = hparams[w]["learning_rate"]
                hd_df.loc[scenario, w] = hparams[w]["hidden_dim"]
                hl_df.loc[scenario, w] = hparams[w]["hidden_layers"]

        # Normalize learning rate (log scale) for better color contrast
        lr_norm = np.log10(lr_df + 1e-8)

        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f"Hyperparameters Heatmaps for {algo}", fontsize=16)

        sns.heatmap(lr_norm, ax=axs[0], cmap="Blues", cbar_kws={'label': 'Log10 Learning Rate'})
        axs[0].set_title("Learning Rate (log scale)")
        axs[0].set_xlabel("Weight")
        axs[0].set_ylabel("Scenario")

        sns.heatmap(hd_df, ax=axs[1], cmap="Greens", cbar_kws={'label': 'Hidden Dim'})
        axs[1].set_title("Hidden Dimension")
        axs[1].set_xlabel("Weight")
        axs[1].set_ylabel("")

        sns.heatmap(hl_df, ax=axs[2], cmap="Oranges", cbar_kws={'label': 'Hidden Layers'})
        axs[2].set_title("Hidden Layers")
        axs[2].set_xlabel("Weight")
        axs[2].set_ylabel("")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        filename = f"heatmaps_{algo.replace(' ', '_')}.png"
        plt.savefig(filename)
        print(f"Saved heatmap figure: {filename}")
        plt.show()


def plot_hidden_layers_dims_lr_heatmaps(method_default, method_scenarios, method_name, weights, scenario_labels, save_dir):
    """
    For each scenario, plot side-by-side heatmaps of (Hidden Layers × Hidden Dim) with color=Learning Rate.
    One heatmap for Default, one for Scenario.

    Args:
    - method_default: dict {weight: {'learning_rate', 'hidden_dim', 'hidden_layers'}}
    - method_scenarios: list of dicts (Scenario 1 to 4)
    - method_name: str
    - weights: list of weights (e.g., [0.1,...,0.6])
    - scenario_labels: list of scenario names
    - save_dir: directory path to save figures
    """

    os.makedirs(save_dir, exist_ok=True)

    # Gather all unique hidden layers and hidden dims from both default and all scenarios
    all_hidden_layers = set()
    all_hidden_dims = set()

    def collect_params(hparams_dict):
        for w in weights:
            all_hidden_layers.add(hparams_dict[w]["hidden_layers"])
            all_hidden_dims.add(hparams_dict[w]["hidden_dim"])

    collect_params(method_default)
    for scenario in method_scenarios:
        collect_params(scenario)

    all_hidden_layers = sorted(all_hidden_layers)
    all_hidden_dims = sorted(all_hidden_dims)

    # Helper to create empty DataFrame indexed by Hidden Dim (rows) and Hidden Layers (columns)
    def empty_lr_matrix():
        return pd.DataFrame(np.nan, index=all_hidden_dims, columns=all_hidden_layers)

    # For each scenario, create heatmaps for Default and Scenario hyperparams
    for i, scenario_dict in enumerate(method_scenarios):
        # Initialize matrices
        default_matrix = empty_lr_matrix()
        scenario_matrix = empty_lr_matrix()

        # Fill in matrices for all weights (x = hidden_layers, y = hidden_dim)
        for w in weights:
            d_hl = method_default[w]["hidden_layers"]
            d_hd = method_default[w]["hidden_dim"]
            d_lr = method_default[w]["learning_rate"]
            default_matrix.at[d_hd, d_hl] = d_lr

            s_hl = scenario_dict[w]["hidden_layers"]
            s_hd = scenario_dict[w]["hidden_dim"]
            s_lr = scenario_dict[w]["learning_rate"]
            scenario_matrix.at[s_hd, s_hl] = s_lr

        # Convert learning rates to log scale for coloring (handle zero or near-zero)
        default_log_lr = np.log10(default_matrix + 1e-10)
        scenario_log_lr = np.log10(scenario_matrix + 1e-10)

        fig, axs = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

        sns.heatmap(default_log_lr, ax=axs[0], cmap="viridis", cbar=True, square=True,
                    cbar_kws={'label': 'Log10 Learning Rate'}, linewidths=0.5, linecolor='gray',
                    annot=default_matrix.round(4), fmt='')
        axs[0].set_title(f"Default Hyperparams\n(Method: {method_name})")
        axs[0].set_xlabel("Hidden Layers")
        axs[0].set_ylabel("Hidden Dimension")

        sns.heatmap(scenario_log_lr, ax=axs[1], cmap="viridis", cbar=True, square=True,
                    cbar_kws={'label': 'Log10 Learning Rate'}, linewidths=0.5, linecolor='gray',
                    annot=scenario_matrix.round(4), fmt='')
        axs[1].set_title(f"{scenario_labels[i]} Hyperparams\n(Method: {method_name})")
        axs[1].set_xlabel("Hidden Layers")
        axs[1].set_ylabel("")

        fig.suptitle(f"Hyperparameter Transition from Default to {scenario_labels[i]} ({method_name})", fontsize=16)

        filename = os.path.join(save_dir, f"{method_name.lower().replace(' ', '_')}_transition_{scenario_labels[i].lower().replace(' ', '_')}.png")
        plt.savefig(filename)
        plt.close()
        print(f"Saved plot: {filename}")



# Assuming your hyperparameter dictionaries and variables are already defined:
# ctde_td_default, ctde_td_scenario1, ..., centralized_td_scenario4, centralized_td_double_default, etc.

# List all scenario and algorithm hyperparameter dicts for the summary table
hyperparam_dicts_all = [
    ctde_td_default, ctde_td_scenario1, ctde_td_scenario2, ctde_td_scenario3, ctde_td_scenario4,
    ctde_mc_default, ctde_mc_scenario1, ctde_mc_scenario2, ctde_mc_scenario3, ctde_mc_scenario4,
    centralized_mc_default, centralized_mc_scenario1, centralized_mc_scenario2, centralized_mc_scenario3, centralized_mc_scenario4,
    centralized_td_double_default, centralized_td_scenario1, centralized_td_scenario2, centralized_td_scenario3, centralized_td_scenario4,
]

scenario_names_all = (
    ['Default'] +
    ['Scenario 1']*4 + ['Scenario 2']*4 + ['Scenario 3']*4 + ['Scenario 4']*4
)

algo_names_all = (
    ['CTDE TD'] +
    ['CTDE TD', 'CTDE MC', 'Centralized MC', 'Centralized TD']*4
)

# weights = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
#
# # 1) Plot & save summary table (in Jupyter, you can display styled_table directly)
# styled_table = plot_hyperparam_summary_table(hyperparam_dicts_all, scenario_names_all, algo_names_all, weights)
# # To save styled table to HTML uncomment below:
# with open("hyperparam_summary.html", "w") as f:
#     f.write(styled_table.to_html())
#
# # 2) Plot radar chart comparing CTDE TD across all scenarios (example)
# methods_hyperparams = [
#     [ctde_td_default, ctde_td_scenario1, ctde_td_scenario2, ctde_td_scenario3, ctde_td_scenario4],
#     [ctde_mc_default, ctde_mc_scenario1, ctde_mc_scenario2, ctde_mc_scenario3, ctde_mc_scenario4],
#     [centralized_mc_default, centralized_mc_scenario1, centralized_mc_scenario2, centralized_mc_scenario3, centralized_mc_scenario4],
#     [centralized_td_double_default, centralized_td_scenario1, centralized_td_scenario2, centralized_td_scenario3, centralized_td_scenario4]
# ]
#
# methods_labels = ["CTDE TD", "CTDE MC", "Centralized MC", "Centralized TD"]
# scenario_labels = ["Default", "Scenario 1", "Scenario 2", "Scenario 3", "Scenario 4"]
#
# plot_radar_charts_multiple(methods_hyperparams, methods_labels, scenario_labels,
#                            fig_title="Neural Network Hyperparameters Across Methods and Scenarios")
#
# # List all your hyperparameter dicts, scenario names, algo names as before:
#
# hyperparam_dicts_all = [
#     ctde_td_default, ctde_td_scenario1, ctde_td_scenario2, ctde_td_scenario3, ctde_td_scenario4,
#     ctde_mc_default, ctde_mc_scenario1, ctde_mc_scenario2, ctde_mc_scenario3, ctde_mc_scenario4,
#     centralized_mc_default, centralized_mc_scenario1, centralized_mc_scenario2, centralized_mc_scenario3, centralized_mc_scenario4,
#     centralized_td_double_default, centralized_td_scenario1, centralized_td_scenario2, centralized_td_scenario3, centralized_td_scenario4,
# ]
#
# scenario_names_all = (
#     ['Default'] +
#     ['Scenario 1']*4 + ['Scenario 2']*4 + ['Scenario 3']*4 + ['Scenario 4']*4
# )
#
# algo_names_all = (
#     ['CTDE TD'] +
#     ['CTDE TD', 'CTDE MC', 'Centralized MC', 'Centralized TD']*4
# )
#
# weights = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
#
# plot_hyperparam_heatmaps(hyperparam_dicts_all, scenario_names_all, algo_names_all, weights)
#
# plot_complexity_vs_learning_rate(hyperparam_dicts_all, scenario_names_all, algo_names_all, weights, save_path="complexity_vs_lr_across_methods.png")

weights = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
scenario_labels = ["Scenario 1", "Scenario 2", "Scenario 3", "Scenario 4"]

save_dir = "hyperparam_transitions_plots"

plot_method_transitions_all_weights(
    ctde_td_default,
    [ctde_td_scenario1, ctde_td_scenario2, ctde_td_scenario3, ctde_td_scenario4],
    "CTDE TD",
    weights,
    scenario_labels,
    save_dir
)

plot_method_transitions_all_weights(
    ctde_mc_default,
    [ctde_mc_scenario1, ctde_mc_scenario2, ctde_mc_scenario3, ctde_mc_scenario4],
    "CTDE MC",
    weights,
    scenario_labels,
    save_dir
)

plot_method_transitions_all_weights(
    centralized_mc_default,
    [centralized_mc_scenario1, centralized_mc_scenario2, centralized_mc_scenario3, centralized_mc_scenario4],
    "Centralized MC",
    weights,
    scenario_labels,
    save_dir
)

plot_method_transitions_all_weights(
    centralized_td_double_default,
    [centralized_td_scenario1, centralized_td_scenario2, centralized_td_scenario3, centralized_td_scenario4],
    "Centralized TD",
    weights,
    scenario_labels,
    save_dir
)




# # 3) Plot & save side-by-side comparisons for all scenarios and algorithms
# plots_to_compare = [
#     # Scenario 1 comparisons
#     ("Scenario 1", "CTDE TD", ctde_td_default, ctde_td_scenario1),
#     ("Scenario 1", "CTDE MC", ctde_mc_default, ctde_mc_scenario1),
#     ("Scenario 1", "Centralized MC", centralized_mc_default, centralized_mc_scenario1),
#     ("Scenario 1", "Centralized TD", centralized_td_double_default, centralized_td_scenario1),
#
#     # Scenario 2 comparisons
#     ("Scenario 2", "CTDE TD", ctde_td_default, ctde_td_scenario2),
#     ("Scenario 2", "CTDE MC", ctde_mc_default, ctde_mc_scenario2),
#     ("Scenario 2", "Centralized MC", centralized_mc_default, centralized_mc_scenario2),
#     ("Scenario 2", "Centralized TD", centralized_td_double_default, centralized_td_scenario2),
#
#     # Scenario 3 comparisons
#     ("Scenario 3", "CTDE TD", ctde_td_default, ctde_td_scenario3),
#     ("Scenario 3", "CTDE MC", ctde_mc_default, ctde_mc_scenario3),
#     ("Scenario 3", "Centralized MC", centralized_mc_default, centralized_mc_scenario3),
#     ("Scenario 3", "Centralized TD", centralized_td_double_default, centralized_td_scenario3),
#
#     # Scenario 4 comparisons
#     ("Scenario 4", "CTDE TD", ctde_td_default, ctde_td_scenario4),
#     ("Scenario 4", "CTDE MC", ctde_mc_default, ctde_mc_scenario4),
#     ("Scenario 4", "Centralized MC", centralized_mc_default, centralized_mc_scenario4),
#     ("Scenario 4", "Centralized TD (Predicted)", centralized_td_double_default, centralized_td_scenario4),
# ]
#
# for scenario, algo, default_params, scenario_params in plots_to_compare:
#     plot_hyperparameters_comparison(scenario, algo, default_params, scenario_params)
