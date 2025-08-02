import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
import math
from matplotlib.colors import ListedColormap, BoundaryNorm
import colorsys
import torch


def generate_distinct_colors(n):
    """Generate distinct colors for plotting"""
    HSV_tuples = [(x * 1.0 / n, 0.5, 0.95) for x in range(n)]  # Adjust brightness for better distinction
    RGB_tuples = list(map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples))
    return ['#{:02x}{:02x}{:02x}'.format(int(r * 255), int(g * 255), int(b * 255)) for r, g, b in RGB_tuples]

def visualize_tabular(
    agents,
    env,
    tabular_agent,
    save_path="results/tabular_policy.png",
    gamma=0.1,
    grid_size=100
):
    """
    Mirror visualize_all_states_dqn but calling tabular_agent.select_action.
    """
    n_agents = len(agents)

    # 1) build colormap
    num_actions = env.action_spaces[agents[0]].n
    action_colors = generate_distinct_colors(num_actions)
    cmap = ListedColormap(action_colors)
    norm = BoundaryNorm(np.arange(num_actions+1)-0.5, num_actions)

    # 2) layout
    if n_agents <= 3:
        nrows, ncols = 1, n_agents
    elif n_agents == 4:
        nrows, ncols = 2, 2
    else:
        side = int(math.sqrt(n_agents))
        nrows = ncols = side

    # 3) state ranges
    max_inf = max(i for i,_ in env.state_space)
    min_risk, max_risk = (
        min(r for _, r in env.state_space),
        max(r for _, r in env.state_space)
    )
    baseline = np.array([50,0.5])

    # 4) lattice
    N = grid_size
    risk_vals     = np.linspace(min_risk, max_risk, N)
    infected_vals = np.linspace(0, max_inf, N).astype(int)

    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols,5*nrows), squeeze=False)
    axs = axes.flatten()

    # 6) per-agent policy map
    for idx, ag in enumerate(agents):
        ax = axs[idx]
        policy = np.empty((N,N), int)
        for yi, inf in enumerate(infected_vals):
            for xi, risk in enumerate(risk_vals):
                # build state for this agent
                policy[yi, xi] = tabular_agent.select_action(idx, (inf, risk))
        ax.imshow(policy, origin='lower', cmap=cmap, norm=norm,
                  interpolation='nearest', aspect='equal')
        ticks = [0, N//2, N-1]
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{min_risk:.2f}", f"{(min_risk+max_risk)/2:.2f}", f"{max_risk:.2f}"])
        ax.set_yticks(ticks)
        ax.set_yticklabels(["0", f"{max_inf/2:.0f}", f"{max_inf:.0f}"])
        ax.set_title(ag)
        ax.set_xlabel("Community Risk")
        ax.set_ylabel("Infected Students")

    for ax in axs[n_agents:]:
        ax.axis('off')

    # legend
    allowed_vals = env.action_levels[0]  # same for all
    patches = [mpatches.Patch(color=action_colors[a], label=f"{allowed_vals[a]} students")
               for a in range(num_actions)]
    fig.subplots_adjust(bottom=0.15, hspace=0.4)
    fig.legend(handles=patches, loc='lower center', ncol=num_actions,
               title="Color → action", bbox_to_anchor=(0.5,0.05), frameon=False)

    fig.suptitle(f"Tabular Policy Map (γ={gamma:.2f})", fontsize=16)
    plt.tight_layout(rect=[0,0.15,1,0.95])

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Saved tabular policy visualization to {save_path}")
    return save_path


def visualize_all_states_dqn(
    method,
    agents,
    env,
    dqn_agent,
    save_path="results/policy_visualization.png",
    gamma=0.1,
    grid_size=100

):
    """
    Flexible grid layout:
      • 1–3 agents → 1×n row
      •  4 agents  → 2×2
      • >4 agents  → √n×√n (assumes n is a perfect square)
    Each panel shows a grid_size×grid_size policy map via imshow with square cells.
    """
    n_agents = len(agents)

    # 1) Build colormap
    num_actions = env.action_spaces[agents[0]].n
    action_colors = generate_distinct_colors(num_actions)
    cmap = ListedColormap(action_colors)
    norm = BoundaryNorm(np.arange(num_actions+1)-0.5, ncolors=num_actions)

    # 2) Determine subplot layout
    if n_agents <= 3:
        nrows, ncols = 1, n_agents
    elif n_agents == 4:
        nrows, ncols = 2, 2
    else:
        side = int(math.sqrt(n_agents))
        if side * side != n_agents:
            raise ValueError(f"n_agents={n_agents} >4 must be a perfect square")
        nrows = ncols = side

    # 3) Data ranges & baseline
    max_inf = max(i for i, _ in env.state_space)
    min_risk, max_risk = min(r for _, r in env.state_space), max(r for _, r in env.state_space)
    baseline_state = np.array([50, 0.5])

    # 4) Build fixed lattice
    N = grid_size
    risk_vals     = np.linspace(min_risk, max_risk, N)
    infected_vals = np.linspace(0, max_inf,     N).astype(int)

    # 5) Create figure and axes
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 5*nrows), squeeze=False, sharex=True, sharey=True)
    axes_flat = axes.flatten()

    # 6) Plot each agent
    for idx, agent_name in enumerate(agents):
        ax = axes_flat[idx]
        # compute policy array
        policy = np.empty((N, N), dtype=int)
        for yi, inf in enumerate(infected_vals):
            for xi, risk in enumerate(risk_vals):
                joint = []
                for ag in agents:
                    if ag == agent_name:
                        joint.extend([int(inf), float(risk)])
                    else:
                        joint.extend(baseline_state)
                policy[yi, xi] = dqn_agent.select_action(
                    agent_name, np.array([int(inf), float(risk)])
                )
        # render with square cells
        ax.imshow(
            policy,
            origin='lower',
            cmap=cmap,
            norm=norm,
            interpolation='nearest',
            aspect='equal'
        )
        # relabel ticks back to data units
        ticks = [0, N//2, N-1]
        ax.set_xticks(ticks)
        ax.set_xticklabels([
            f"{min_risk:.2f}",
            f"{(min_risk+max_risk)/2:.2f}",
            f"{max_risk:.2f}"
        ])
        ax.set_yticks(ticks)
        ax.set_yticklabels([
            "0",
            f"{max_inf/2:.0f}",
            f"{max_inf:.0f}"
        ])
        ax.set_title(agent_name)
        ax.set_xlabel("Community Risk")
        ax.set_ylabel("Infected Students")

    # 7) Turn off any unused subplots
    for ax in axes_flat[n_agents:]:
        ax.axis('off')

    # 8) Shared legend below all
    total_students = env.total_students
    allowed_vals = np.linspace(0, total_students, num_actions).astype(int)
    action_patches = [
        mpatches.Patch(color=action_colors[a], label=f"{allowed_vals[a]} students")
        for a in range(num_actions)
    ]

    # leave room for legend
    fig.subplots_adjust(left=0.05, right=0.97, bottom=0.15, hspace=0.4, wspace=0.15)
    fig.legend(
        handles=action_patches,
        loc='lower center',
        ncol=num_actions,
        title="Color → action",
        bbox_to_anchor=(0.5, 0.05),
        frameon=False
    )

    # 9) Title & layout
    # title per gamma value
    fig.suptitle(f"CTDE-{method} Policy (w={gamma:.2f})", fontsize=16)
    plt.tight_layout(rect=[0, 0.15, 1, 0.95])

    # 10) Save and close
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Saved policy visualization to {save_path}")

    return save_path





def visualize_all_states_dqn_swap(
    method,
    agents,
    env,
    dqn_agent,
    save_path="results/policy_visualization.png",
    gamma=0.1,
    grid_size=100,
    swap_policy_agent=None
):
    """
    Flexible grid layout:
      • 1–3 agents → 1×n row
      •  4 agents  → 2×2
      • >4 agents  → √n×√n (assumes n is a perfect square)
    Each panel shows a grid_size×grid_size policy map via imshow with square cells.

    If swap_policy_agent is provided (agent name), then all classrooms use that agent's
    policy for action selection during visualization.
    """
    n_agents = len(agents)

    # 1) Build colormap
    num_actions = env.action_spaces[agents[0]].n
    action_colors = generate_distinct_colors(num_actions)
    cmap = ListedColormap(action_colors)
    norm = BoundaryNorm(np.arange(num_actions+1)-0.5, ncolors=num_actions)

    # 2) Determine subplot layout
    if n_agents <= 3:
        nrows, ncols = 1, n_agents
    elif n_agents == 4:
        nrows, ncols = 2, 2
    else:
        side = int(math.sqrt(n_agents))
        if side * side != n_agents:
            raise ValueError(f"n_agents={n_agents} >4 must be a perfect square")
        nrows = ncols = side

    # 3) Data ranges & baseline
    max_inf = max(i for i, _ in env.state_space)
    min_risk, max_risk = min(r for _, r in env.state_space), max(r for _, r in env.state_space)
    baseline_state = np.array([50, 0.5])

    # 4) Build fixed lattice
    N = grid_size
    risk_vals     = np.linspace(min_risk, max_risk, N)
    infected_vals = np.linspace(0, max_inf,     N).astype(int)

    # 5) Create figure and axes
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 5*nrows), squeeze=False)
    axes_flat = axes.flatten()

    # 6) Plot each agent
    for idx, agent_name in enumerate(agents):
        ax = axes_flat[idx]
        # compute policy array
        policy = np.empty((N, N), dtype=int)
        for yi, inf in enumerate(infected_vals):
            for xi, risk in enumerate(risk_vals):
                # Compose joint state vector for all agents
                joint = []
                for ag in agents:
                    if ag == agent_name:
                        # For this panel, agent's own state is infected,risk
                        joint.extend([int(inf), float(risk)])
                    else:
                        # Other agents set to baseline
                        joint.extend(baseline_state)

                # If swap_policy_agent is set, override the policy agent to swap_policy_agent
                if swap_policy_agent is not None:
                    # Select action according to swap_policy_agent for all agents
                    policy[yi, xi] = dqn_agent.select_action(
                        swap_policy_agent, np.array([int(inf), float(risk)])
                    )
                else:
                    # Normal: select action of current agent
                    policy[yi, xi] = dqn_agent.select_action(
                        agent_name, np.array([int(inf), float(risk)])
                    )

        # render with square cells
        ax.imshow(
            policy,
            origin='lower',
            cmap=cmap,
            norm=norm,
            interpolation='nearest',
            aspect='equal'
        )
        # relabel ticks back to data units
        ticks = [0, N//2, N-1]
        ax.set_xticks(ticks)
        ax.set_xticklabels([
            f"{min_risk:.2f}",
            f"{(min_risk+max_risk)/2:.2f}",
            f"{max_risk:.2f}"
        ])
        ax.set_yticks(ticks)
        ax.set_yticklabels([
            "0",
            f"{max_inf/2:.0f}",
            f"{max_inf:.0f}"
        ])
        ax.set_title(agent_name)
        ax.set_xlabel("Community Risk")
        ax.set_ylabel("Infected Students")

    # 7) Turn off any unused subplots
    for ax in axes_flat[n_agents:]:
        ax.axis('off')

    # 8) Shared legend below all
    total_students = env.total_students
    allowed_vals = np.linspace(0, total_students, num_actions).astype(int)
    action_patches = [
        mpatches.Patch(color=action_colors[a], label=f"{allowed_vals[a]} students")
        for a in range(num_actions)
    ]

    # leave room for legend
    fig.subplots_adjust(bottom=0.15, hspace=0.4)
    fig.legend(
        handles=action_patches,
        loc='lower center',
        ncol=num_actions,
        title="Color → action",
        bbox_to_anchor=(0.5, 0.05),
        frameon=False
    )

    # 9) Title & layout
    title_suffix = f" (swap policy: {swap_policy_agent})" if swap_policy_agent else ""
    fig.suptitle(f"{method}-CTDE Policy Map (γ={gamma:.2f}){title_suffix}", fontsize=16)
    plt.tight_layout(rect=[0, 0.15, 1, 0.95])

    # 10) Save and close
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Saved policy visualization to {save_path}")

    return save_path


def visualize_all_states_centralized(
    method,
    agents,
    env,
    agent,                  # your DQNAgent with a CentralizedDQNetwork
    save_path="results/centralized_policy.png",
    gamma: float = 0.0,     # ← new parameter
    grid_size=100,
    baseline_state=None
):
    """
    Visualize each agent’s marginal policy under the fully‐centralized critic,
    labeling the figure with the discount γ.
    """

    n_agents = len(agents)
    num_actions = env.action_spaces[agents[0]].n

    # Colormap & norm
    action_colors = generate_distinct_colors(num_actions)
    cmap = ListedColormap(action_colors)
    norm = BoundaryNorm(np.arange(num_actions+1)-0.5, ncolors=num_actions)

    # Subplot layout
    if n_agents <= 3:
        nrows, ncols = 1, n_agents
    elif n_agents == 4:
        nrows, ncols = 2, 2
    else:
        side = int(math.sqrt(n_agents))
        nrows = ncols = side

    # State‐space grid
    max_inf = max(i for i,_ in env.state_space)
    min_risk, max_risk = min(r for _,r in env.state_space), max(r for _,r in env.state_space)
    inf_vals = np.linspace(0, max_inf, grid_size).astype(int)
    risk_vals = np.linspace(min_risk, max_risk, grid_size)

    # Baseline for non‐plotted agents
    if baseline_state is None:
        baseline_state = (int(max_inf/2), float((min_risk+max_risk)/2))

    # Figure
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 5*nrows), squeeze=False, sharex=True, sharey=True)
    axes_flat = axes.flatten()

    for idx, ag in enumerate(agents):
        ax = axes_flat[idx]
        policy = np.zeros((grid_size, grid_size), dtype=int)

        for yi, inf in enumerate(inf_vals):
            for xi, risk in enumerate(risk_vals):
                joint = []
                for other in agents:
                    if other == ag:
                        joint.extend([inf, risk])
                    else:
                        joint.extend(baseline_state)

                with torch.no_grad():
                    q_heads = agent.network(
                        torch.FloatTensor(joint).unsqueeze(0)
                    ).squeeze(0)   # (N, A)

                joint_act = [int(q_heads[i].argmax()) for i in range(n_agents)]
                policy[yi, xi] = joint_act[idx]

        im = ax.imshow(policy, origin='lower', cmap=cmap, norm=norm, aspect='equal')
        ticks = [0, grid_size//2, grid_size-1]
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{min_risk:.2f}", f"{(min_risk+max_risk)/2:.2f}", f"{max_risk:.2f}"])
        ax.set_yticks(ticks)
        ax.set_yticklabels(["0", f"{max_inf/2:.0f}", f"{max_inf:.0f}"])
        ax.set_title(ag)
        ax.set_xlabel("Community Risk")
        ax.set_ylabel("Infected Students")

    # Turn off unused
    for ax in axes_flat[n_agents:]:
        ax.axis('off')

    # Legend
    allowed_vals = np.array(env.action_levels[0])
    patches = [
        mpatches.Patch(color=action_colors[i], label=f"{allowed_vals[i]} students")
        for i in range(num_actions)
    ]
    fig.subplots_adjust(left=0.05, right=0.97, bottom=0.15, hspace=0.4, wspace=0.15)
    fig.legend(handles=patches, loc='lower center', ncol=num_actions,
               title="Color → action", frameon=False)

    # Title with gamma
    fig.suptitle(f"CTCE-{method}Joint Q‐Policy: (weight = {gamma:.2f})", fontsize=16)

    plt.tight_layout(rect=[0,0.15,1,0.95])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Saved centralized policy visualization to {save_path}")


def visualize_all_states_ppo(
    agents,
    env,
    ppo_agent,
    save_path="results/ppo_policy_continuous.png",
    gamma=0.1,
    grid_size=100
):
    """
    Layout same as before, but each cell now shows the continuous
    action (allowed students) via a smooth colormap.
    """
    n_agents = len(agents)

    # 1) Determine subplot layout
    if n_agents <= 3:
        nrows, ncols = 1, n_agents
    elif n_agents == 4:
        nrows, ncols = 2, 2
    else:
        side = int(math.sqrt(n_agents))
        if side * side != n_agents:
            raise ValueError("n_agents>4 must be a perfect square")
        nrows = ncols = side

    # 2) State-space ranges
    max_inf = max(i for i, _ in env.state_space)
    min_risk, max_risk = min(r for _, r in env.state_space), max(r for _, r in env.state_space)
    baseline = np.array([50, 0.5])

    # 3) Grid over (inf, risk)
    N = grid_size
    risk_vals     = np.linspace(min_risk, max_risk, N)
    infected_vals = np.linspace(0, max_inf,     N)

    # 4) Figure
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 5*nrows), squeeze=False)
    axes_flat = axes.flatten()

    for idx, agent_name in enumerate(agents):
        ax = axes_flat[idx]
        policy = np.zeros((N, N), dtype=float)

        for yi, inf in enumerate(infected_vals):
            for xi, risk in enumerate(risk_vals):
                # build joint-state
                joint = []
                for ag in agents:
                    if ag == agent_name:
                        joint.extend([inf, risk])
                    else:
                        joint.extend(baseline)
                js = torch.FloatTensor(joint).unsqueeze(0)  # (1, N*S)

                # evaluate the Gaussian policy mean
                means, log_stds, _ = ppo_agent.net(js)
                mean_val = means[0, idx].item()      # continuous action for this agent
                # optionally scale mean_val to student count:
                # if your network outputs in [0,1], then:
                # mean_val = mean_val * env.total_students

                policy[yi, xi] = mean_val

        # plot as heatmap
        im = ax.imshow(
            policy,
            origin='lower',
            cmap='viridis',
            vmin=0,
            vmax=env.total_students,   # scale colorbar to actual student counts
            interpolation='nearest',
            aspect='equal'
        )
        # axis labels
        ticks = [0, N//2, N-1]
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{min_risk:.2f}", f"{(min_risk+max_risk)/2:.2f}", f"{max_risk:.2f}"])
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"{0:.0f}", f"{max_inf/2:.0f}", f"{max_inf:.0f}"])
        ax.set_title(agent_name)
        ax.set_xlabel("Community Risk")
        ax.set_ylabel("Infected Students")

        # colorbar for this subplot
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("Allowed Students")

    # turn off empty subplots
    for ax in axes_flat[n_agents:]:
        ax.axis('off')

    fig.suptitle(f"PPO Continuous Policy Map (γ={gamma:.2f})", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Saved continuous‐action policy visualization to {save_path}")

# def visualize_all_states_ppo(
#     agents,
#     env,
#     ppo_agent,
#     save_path="results/ppo_policy.png",
#     gamma=0.1,
#     grid_size=100
# ):
#     """
#     Same layout and style as DQN version, but uses PPOAgent's policy (argmax over actor logits).
#     """
#     n_agents = len(agents)
#
#     # 1) Colormap
#     num_actions = env.action_spaces[agents[0]].n
#     action_colors = generate_distinct_colors(num_actions)
#     cmap = ListedColormap(action_colors)
#     norm = BoundaryNorm(np.arange(num_actions+1)-0.5, ncolors=num_actions)
#
#     # 2) Layout
#     if n_agents <= 3:
#         nrows, ncols = 1, n_agents
#     elif n_agents == 4:
#         nrows, ncols = 2, 2
#     else:
#         side = int(math.sqrt(n_agents))
#         if side * side != n_agents:
#             raise ValueError("n_agents>4 must be perfect square")
#         nrows = ncols = side
#
#     # 3) State-space ranges
#     max_inf = max(i for i, _ in env.state_space)
#     min_risk, max_risk = min(r for _, r in env.state_space), max(r for _, r in env.state_space)
#     baseline = np.array([50, 0.5])
#
#     # 4) Grid
#     N = grid_size
#     risk_vals     = np.linspace(min_risk, max_risk, N)
#     infected_vals = np.linspace(0, max_inf,     N).astype(int)
#
#     # 5) Figure
#     fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 5*nrows), squeeze=False)
#     axes_flat = axes.flatten()
#
#     for idx, agent_name in enumerate(agents):
#         ax = axes_flat[idx]
#         policy = np.zeros((N, N), dtype=int)
#
#         for yi, inf in enumerate(infected_vals):
#             for xi, risk in enumerate(risk_vals):
#                 # build joint-state with this agent at (inf, risk)
#                 joint = []
#                 for ag in agents:
#                     if ag == agent_name:
#                         joint.extend([int(inf), float(risk)])
#                     else:
#                         joint.extend(baseline)
#                 js = torch.FloatTensor(joint).unsqueeze(0)
#                 logits, _ = ppo_agent.net(js)
#                 # argmax action for this agent
#                 policy[yi, xi] = logits[0, agents.index(agent_name)].argmax().item()
#
#         ax.imshow(policy, origin='lower', cmap=cmap, norm=norm,
#                   interpolation='nearest', aspect='equal')
#         ticks = [0, N//2, N-1]
#         ax.set_xticks(ticks)
#         ax.set_xticklabels([f"{min_risk:.2f}", f"{(min_risk+max_risk)/2:.2f}", f"{max_risk:.2f}"])
#         ax.set_yticks(ticks)
#         ax.set_yticklabels(["0", f"{max_inf/2:.0f}", f"{max_inf:.0f}"])
#         ax.set_title(agent_name)
#         ax.set_xlabel("Community Risk")
#         ax.set_ylabel("Infected Students")
#
#     # turn off extras
#     for ax in axes_flat[n_agents:]:
#         ax.axis('off')
#
#     # legend
#     total = env.total_students
#     allowed = np.linspace(0, total, num_actions).astype(int)
#     patches = [mpatches.Patch(color=action_colors[a], label=f"{allowed[a]} students")
#                for a in range(num_actions)]
#
#     fig.subplots_adjust(bottom=0.15, hspace=0.4)
#     fig.legend(handles=patches, loc='lower center', ncol=num_actions,
#                title="Action → # students", bbox_to_anchor=(0.5,0.05), frameon=False)
#
#     fig.suptitle(f"PPO Policy Map (γ={gamma:.2f})", fontsize=16)
#     plt.tight_layout(rect=[0,0.15,1,0.95])
#
#     os.makedirs(os.path.dirname(save_path), exist_ok=True)
#     plt.savefig(save_path)
#     plt.close(fig)
#     print(f"Saved policy visualization to {save_path}")



def visualize_myopic_policy(
    agents,
    env,
    myopic_agent,
    save_path="results/myopic_policy.png",
    gamma=0.1,
    grid_size=100,
    dpi=None,
    bbox_inches=None
):
    """
    Flexible grid layout:
      • 1–3 agents → 1×n row
      •  4 agents  → 2×2
      • >4 agents  → √n×√n (assumes n is a perfect square)
    Each panel shows a grid_size×grid_size policy map via imshow with square cells.
    """
    n_agents = len(agents)

    # 1) Build colormap/norm
    num_actions = len(myopic_agent.allowed_values)
    action_colors = generate_distinct_colors(num_actions)
    cmap = ListedColormap(action_colors)
    norm = BoundaryNorm(np.arange(num_actions+1)-0.5, ncolors=num_actions)

    # 2) Determine subplot layout
    if n_agents <= 3:
        nrows, ncols = 1, n_agents
    elif n_agents == 4:
        nrows, ncols = 2, 2
    else:
        side = int(math.sqrt(n_agents))
        if side * side != n_agents:
            raise ValueError(f"n_agents={n_agents} >4 must be a perfect square")
        nrows = ncols = side

    # 3) Data ranges
    if hasattr(env, 'state_space'):
        inf_vals, risk_vals = zip(*env.state_space)
        min_risk, max_risk = min(risk_vals), max(risk_vals)
        max_inf = max(inf_vals)
    else:
        min_risk, max_risk = 0.0, 1.0
        max_inf = env.total_students

    # 4) Build lattice
    N = grid_size
    risk_grid     = np.linspace(min_risk, max_risk, N)
    infected_grid = np.linspace(0, max_inf,     N).astype(int)

    # 5) Create figure
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 5*nrows), squeeze=False)

    axes_flat = axes.flatten()

    # 6) Plot each agent
    for idx, agent_name in enumerate(agents):
        ax = axes_flat[idx]
        policy = np.empty((N, N), dtype=int)

        # fill policy map
        for yi, inf in enumerate(infected_grid):
            for xi, risk in enumerate(risk_grid):
                policy[yi, xi] = myopic_agent.select_best_action(
                    agent_name,
                    (int(inf), float(risk)),
                    env.gamma
                )

        ax.imshow(
            policy,
            origin='lower',
            cmap=cmap,
            norm=norm,
            interpolation='nearest',
            aspect='equal'
        )
        ticks = [0, N//2, N-1]
        ax.set_xticks(ticks)
        ax.set_xticklabels([
            f"{min_risk:.2f}",
            f"{(min_risk+max_risk)/2:.2f}",
            f"{max_risk:.2f}"
        ])
        ax.set_yticks(ticks)
        ax.set_yticklabels([
            "0",
            f"{max_inf/2:.0f}",
            f"{max_inf:.0f}"
        ])

        ax.set_title(agent_name)
        ax.set_xlabel("Community Risk")
        ax.set_ylabel("Infected Students")

    # 7) Turn off unused axes
    for ax in axes_flat[n_agents:]:
        ax.axis('off')

    # 8) Shared legend below
    action_patches = [
        mpatches.Patch(color=action_colors[a], label=f"{myopic_agent.allowed_values[a]:.0f} students")
        for a in range(num_actions)
    ]
    fig.subplots_adjust(bottom=0.15, hspace=0.4)
    fig.legend(
        handles=action_patches,
        loc='lower center',
        ncol=num_actions,
        title="Allowed → action",
        bbox_to_anchor=(0.5, 0.05),
        frameon=False
    )

    # 9) Title & layout
    fig.suptitle(f"Myopic Policy (w={gamma:.2f} (shared: 0.6)", fontsize=16)
    plt.tight_layout(rect=[0, 0.15, 1, 0.95])

    # 10) Save and close
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches=bbox_inches)
    plt.close(fig)

    print(f"Myopic policy visualization saved to {save_path}")
    return save_path

# def visualize_myopic_policy(agent, env, myopic_agent, save_path="results/myopic_policy.png"):
#     """
#     Visualize the myopic agent policy over a discrete set of states using a scatter plot,
#     with one subplot per classroom laid out side by side.
#
#     If env has a predefined discrete state space (env.state_space), it is used.
#     Otherwise, we generate a grid over:
#       - Infected students: from 0 to env.total_students (50 discrete values)
#       - Community risk: from 0 to 1.0 (50 discrete values)
#
#     Each state is colored according to the best action chosen by the agent.
#     """
#     # Generate distinct colors for each action.
#     action_space_size = len(myopic_agent.allowed_values)
#     action_colors = generate_distinct_colors(action_space_size)
#
#     # Determine the discrete state space
#     if hasattr(env, 'state_space'):
#         state_space = env.state_space
#     else:
#         num_infected = 50
#         num_risk = 50
#         infected_vals = np.linspace(0, env.total_students, num_infected)
#         risk_vals = np.linspace(0, 1.0, num_risk)
#         state_space = [(inf, risk) for inf in infected_vals for risk in risk_vals]
#
#     num_agents = len(env.agents)
#     fig, axes = plt.subplots(1, num_agents, figsize=(6 * num_agents, 6), squeeze=False)
#
#     for idx, ag in enumerate(env.agents):
#         ax = axes[0, idx]
#         x_vals, y_vals, state_colors = [], [], []
#
#         # For each possible state, compute the best action for this classroom
#         for infected, community_risk in state_space:
#             best_action = myopic_agent.select_best_action(ag, (infected, community_risk), env.gamma)
#             x_vals.append(community_risk)
#             y_vals.append(infected)
#             state_colors.append(action_colors[best_action])
#
#         # Scatter plot for this classroom
#         ax.scatter(x_vals, y_vals, c=state_colors, s=100, marker='s')
#         ax.set_xlabel("Community Risk")
#         ax.set_ylabel("Infected Students")
#         ax.set_title(f"{ag}\nα={myopic_agent.reward_mix_alpha:.2f}, γ={env.gamma:.2f}")
#         ax.grid(True, alpha=0.3)
#
#         # Add legend once under the first subplot
#         if idx == 0:
#             legend_elements = [
#                 mpatches.Patch(
#                     facecolor=color,
#                     label=f'Action {i} ({myopic_agent.allowed_values[i].item():.0f} students)'
#                 )
#                 for i, color in enumerate(action_colors)
#             ]
#             ax.legend(
#                 handles=legend_elements,
#                 loc='upper center',
#                 bbox_to_anchor=(0.5, -0.15),
#                 ncol=action_space_size
#             )
#
#     # Save the combined figure
#     os.makedirs(os.path.dirname(save_path), exist_ok=True)
#     plt.tight_layout()
#     plt.savefig(save_path, dpi=300, bbox_inches='tight')
#     plt.close()
#
#     print(f"Myopic policy visualization saved at {save_path}")
#     return save_path
#
#
