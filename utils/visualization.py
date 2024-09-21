import torch
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
import os
import colorsys


def generate_distinct_colors(n):
    """Generate distinct colors for plotting"""
    HSV_tuples = [(x * 1.0 / n, 0.5, 0.95) for x in range(n)]  # Adjust brightness for better distinction
    RGB_tuples = list(map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples))
    return ['#{:02x}{:02x}{:02x}'.format(int(r * 255), int(g * 255), int(b * 255)) for r, g, b in RGB_tuples]

def visualize_all_states_ppo(agent, env, a2c_agent, save_path="results/policy_visualization.png"):
    """
    Visualize the learned policy for all states by displaying the best action for each state using distinct colors.
    Args:
        agent: The agent whose A2C model we are visualizing.
        env: The environment object for access to the agent.
        a2c_agent: The trained A2C agent.
        save_path: Where to save the generated visualization.
    """
    # Generate distinct colors for the number of actions
    action_colors = generate_distinct_colors(env.action_spaces[agent].n)

    # Prepare to visualize
    fig, ax = plt.subplots(figsize=(6, 6))

    # Lists to store the visualization data
    x_vals, y_vals, state_colors, actions = [], [], [], []

    # Iterate through the state space
    for state_tuple in env.state_space:
        infected, community_risk = state_tuple
        state = np.array([infected, community_risk])

        # Find the best action from the A2C model for this state
        best_action = a2c_agent.select_action(agent, state)  # Ensure this returns an integer

        # Store values for plotting
        x_vals.append(community_risk)  # Community risk as x-axis
        y_vals.append(infected)  # Infected as y-axis
        state_colors.append(action_colors[best_action])  # Assign color based on the best action
        actions.append(best_action)  # Track action for color reference

    # Plot the states with the corresponding actions using distinct colors
    scatter = ax.scatter(x_vals, y_vals, c=state_colors, s=100, marker='s')

    # Create a legend for the actions
    legend_elements = [mpatches.Patch(facecolor=color, label=f'Action {i}') for i, color in enumerate(action_colors)]
    ax.legend(handles=legend_elements, loc='upper right')

    # Set plot labels and title
    ax.set_xlabel("Community Risk (%)")
    ax.set_ylabel("Infected Students")
    ax.set_title(f"Learned Policy (Best Action) for All States - {agent}")

    # Save the plot
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    plt.savefig(save_path)
    # plt.show()

    print(f"Policy visualization saved at {save_path}")
    return save_path

def visualize_all_states_a2c(agent, env, a2c_agent, save_path="results/policy_visualization.png"):
    """
    Visualize the learned policy for all states by displaying the best action for each state using distinct colors.
    Args:
        agent: The agent whose A2C model we are visualizing.
        env: The environment object for access to the agent.
        a2c_agent: The trained A2C agent.
        save_path: Where to save the generated visualization.
    """
    # Generate distinct colors for the number of actions
    action_colors = generate_distinct_colors(env.action_spaces[agent].n)

    # Prepare to visualize
    fig, ax = plt.subplots(figsize=(6, 6))

    # Lists to store the visualization data
    x_vals, y_vals, state_colors, actions = [], [], [], []

    # Iterate through the state space
    for state_tuple in env.state_space:
        infected, community_risk = state_tuple
        state = np.array([infected, community_risk])

        # Find the best action from the A2C model for this state
        best_action = a2c_agent.select_action(agent, state)  # Ensure this returns an integer

        # Store values for plotting
        x_vals.append(community_risk)  # Community risk as x-axis
        y_vals.append(infected)  # Infected as y-axis
        state_colors.append(action_colors[best_action])  # Assign color based on the best action
        actions.append(best_action)  # Track action for color reference

    # Plot the states with the corresponding actions using distinct colors
    scatter = ax.scatter(x_vals, y_vals, c=state_colors, s=100, marker='s')

    # Create a legend for the actions
    legend_elements = [mpatches.Patch(facecolor=color, label=f'Action {i}') for i, color in enumerate(action_colors)]
    ax.legend(handles=legend_elements, loc='upper right')

    # Set plot labels and title
    ax.set_xlabel("Community Risk (%)")
    ax.set_ylabel("Infected Students")
    ax.set_title(f"Learned Policy (Best Action) for All States - {agent}")

    # Save the plot
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    plt.savefig(save_path)
    # plt.show()

    print(f"Policy visualization saved at {save_path}")
    return save_path
def visualize_all_states_dqn(agent, env, dqn_agent, save_path="results/policy_visualization.png"):
    """
    Visualize the learned policy for all states by displaying the best action for each state using distinct colors.
    Args:
        agent: The agent whose DQN model we are visualizing.
        env: The environment object for access to the agent.
        dqn_agent: The trained DQN agent.
        save_path: Where to save the generated visualization.
    """
    # Generate distinct colors for the number of actions
    action_colors = generate_distinct_colors(env.action_spaces[agent].n)

    # Prepare to visualize
    fig, ax = plt.subplots(figsize=(6, 6))

    # Lists to store the visualization data
    x_vals, y_vals, state_colors, actions = [], [], [], []

    # Iterate through the state space
    for state_tuple in env.state_space:
        # State is continuous, so no need for get_state_index, pass directly to the DQN
        infected, community_risk = state_tuple
        state = np.array([infected, community_risk])

        # Find the best action from the DQN model for this state
        best_action = dqn_agent.select_action(agent, state)

        # Store values for plotting
        x_vals.append(community_risk)  # Community risk as x-axis
        y_vals.append(infected)  # Infected as y-axis
        state_colors.append(action_colors[best_action])  # Assign color based on the best action
        actions.append(best_action)  # Track action for color reference

    # Plot the states with the corresponding actions using distinct colors
    scatter = ax.scatter(x_vals, y_vals, c=state_colors, s=100, marker='s')

    # Create a legend for the actions
    legend_elements = [mpatches.Patch(facecolor=color, label=f'Action {i}') for i, color in enumerate(action_colors)]
    ax.legend(handles=legend_elements, loc='upper right')

    # Set plot labels and title
    ax.set_xlabel("Community Risk (%)")
    ax.set_ylabel("Infected Students")
    ax.set_title(f"Learned Policy (Best Action) for All States - {agent}")

    # Save the plot
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    plt.savefig(save_path)
    # plt.show()




    print(f"Policy visualization saved at {save_path}")
    return save_path


def visualize_all_states(agent, env, q1_tables, q2_tables, save_path="results/policy_visualization.png"):
    """
    Visualize the learned policy for all states by displaying the best action for each state using distinct colors.
    Args:
        agent: The agent whose Q1 and Q2 tables we are visualizing.
        env: The environment object for access to the agent.
        q1_tables: The Q1-table for the agent, with rows as state indices and columns as action indices.
        q2_tables: The Q2-table for the agent, with rows as state indices and columns as action indices.
        save_path: Where to save the generated visualization.
    """
    # Generate distinct colors for the number of actions
    action_colors = generate_distinct_colors(env.action_spaces[agent].n)

    # Prepare to visualize
    fig, ax = plt.subplots(figsize=(6, 6))

    # Lists to store the visualization data
    x_vals, y_vals, state_colors, actions = [], [], [], []

    # Iterate through the state space
    for state_tuple in env.state_space:
        state_idx = env.get_state_index(state_tuple)  # Get the index of the current state

        # Find the best action by summing the Q-values from both Q1 and Q2 tables
        q_sum = q1_tables[agent][state_idx] + q2_tables[agent][state_idx]
        best_action = np.argmax(q_sum)

        infected, community_risk = state_tuple

        # Store values for plotting
        x_vals.append(community_risk)  # Community risk as x-axis
        y_vals.append(infected)  # Infected as y-axis
        state_colors.append(action_colors[best_action])  # Assign color based on the best action
        actions.append(best_action)  # Track action for color reference

    # Plot the states with the corresponding actions using distinct colors
    scatter = ax.scatter(x_vals, y_vals, c=state_colors, s=100, marker='s')

    # Create a legend for the actions
    legend_elements = [mpatches.Patch(facecolor=color, label=f'Action {i}') for i, color in enumerate(action_colors)]
    ax.legend(handles=legend_elements, loc='upper right')

    # Set plot labels and title
    ax.set_xlabel("Community Risk (%)")
    ax.set_ylabel("Infected Students")
    ax.set_title(f"Learned Policy (Best Action) for All States - {agent}")

    # Save the plot
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    plt.savefig(save_path)

    print(f"Policy visualization saved at {save_path}")
    return save_path


def visualize_myopic_states(myopic_agent, env, alpha=0.4):
    """
    Visualize the state-action mappings for all agents and classrooms in the environment using Myopic Policy.

    Args:
        myopic_agent: The MyopicPolicy instance used for action selection.
        env: The environment to visualize.
        alpha: The alpha value for reward calculation in Myopic Policy.
    """
    method_name = "viz_all_states_myopic"
    results_subdirectory = "./myopic_results"
    file_paths = []

    # Ensure the results directory exists
    if not os.path.exists(results_subdirectory):
        os.makedirs(results_subdirectory)

    # Generate distinct colors for the number of actions in each classroom
    action_colors = generate_distinct_colors(env.action_spaces[env.agents[0]].n)

    for agent in env.agents:
        fig, ax = plt.subplots()

        state_colors = []
        x_vals = []
        y_vals = []
        actions = []

        # Iterate over possible states (infected, community risk)
        for infected in range(0, env.total_students + 1, 10):  # Increment by 10 for clarity in visualization
            for community_risk in range(0, 101, 10):  # Community risk from 0% to 100%
                # Prepare state with matching input dimensions (infected, community_risk)
                infected_tensor = torch.FloatTensor([infected])
                community_risk_tensor = torch.FloatTensor([community_risk / 100])

                # Use the MyopicPolicy to get the action
                label, allowed_value, _, _ = myopic_agent.get_label(
                    infected_tensor, community_risk_tensor, agent, alpha
                )

                action = label

                # Store values for plotting
                state_colors.append(action_colors[action])
                actions.append(action)
                x_vals.append(community_risk)
                y_vals.append(infected)

        # Plot the states with the corresponding actions
        scatter = ax.scatter(x_vals, y_vals, c=state_colors, s=100, marker='s')

        # Add legend
        legend_elements = [mpatches.Patch(facecolor=color, label=f'Action {i}') for i, color in
                           enumerate(action_colors)]
        ax.legend(handles=legend_elements, loc='upper right')
        ax.set_xlabel('Community Risk (%)')
        ax.set_ylabel('Infected Students')
        ax.set_title(f'State Visualization for {agent} using Myopic Policy')

        # Save the plot
        file_name = f"{method_name}_{agent}_states.png"
        plt.savefig(os.path.join(results_subdirectory, file_name))
        file_paths.append(os.path.join(results_subdirectory, file_name))
        plt.close(fig)

    return file_paths
