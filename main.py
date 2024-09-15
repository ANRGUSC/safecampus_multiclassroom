from environment.multiclassroom import MultiClassroomEnv
from agents.dqn_agent import DQNAgent
from agents.q_learning import IndependentQLearningAgent
from utils.visualization import visualize_all_states, visualize_all_states_dqn
from utils.visualization import visualize_myopic_states
from agents.myopic_agent import MyopicPolicy
import torch
import random
import numpy as np
import os
import itertools
from tqdm import tqdm

def dqn_main():
    # Set up the environment with 2 classrooms, 100 total students, and 3 action levels per classroom
    num_classrooms = 2
    env = MultiClassroomEnv(num_classrooms=num_classrooms, total_students=100, s_shared=10, max_weeks=20,
                            action_levels_per_class=[3, 3], seed=42)

    # Define agents corresponding to the classrooms
    agents = [f'classroom_{i}' for i in range(num_classrooms)]

    # Dynamically set input_dim based on the observation space (infected, community risk)
    input_dim = env.observation_spaces[agents[0]].shape[0]

    # Action space is discrete, and the number of possible actions is based on the levels of allowed students
    output_dim = len(env.action_levels[0])  # Number of allowed action levels per classroom

    # Initialize the DQN agent with the correct input dimensions (infected, community risk)
    agent = DQNAgent(agents, input_dim=input_dim, output_dim=output_dim, seed=42)

    # Train the agent in the environment
    agent.train(env)

    # Test the trained model by printing the actions for specific state combinations
    print("\nTesting the trained model on specific state combinations of infected and community risk:")

    # Define continuous values for testing (infected students and community risk)
    infected_values = [0.0, 25.0, 50.0, 75.0, 100.0]  # Infected students range
    community_risk_values = [0.0, 0.25, 0.5, 0.75, 1.0]  # Community risk range

    for agent_name in agents:
        print(f"\nActions for {agent_name}:")
        # Iterate over the combinations of infected values and community risk values
        for infected in infected_values:
            for community_risk in community_risk_values:
                # Prepare state as a tensor (infected, community_risk)
                state = torch.FloatTensor([infected, community_risk]).unsqueeze(0)
                with torch.no_grad():
                    # Forward pass through the trained model to get the Q-values for actions
                    q_values = agent.networks[agent_name](state)
                    # Select the action with the highest Q-value (discrete action space)
                    action = torch.argmax(q_values).item()

                # Print the state and the selected action
                print(f"State (Infected: {infected:.2f}, Community Risk: {community_risk:.2f}) -> Action: {action}")

    # Visualize the learned policy after training for all states in the environment
    for agent_name in agents:
        visualize_all_states(agent.networks[agent_name], env)

def random_agent(env, num_steps=100):
    """
    Run random agents in the MultiClassroomEnv environment.

    Args:
        env: The environment to run.
        num_steps: The number of steps to run the environment for.
    """
    # Reset the environment to get the initial state
    observations = env.reset()

    for step in range(num_steps):
        actions = {}

        # Select a random action for each agent from their action space
        for agent in env.agents:
            actions[agent] = env.action_spaces[agent].sample()  # Random action from agent's action space

        # Step the environment with the random actions
        observations, rewards, dones, _ = env.step(actions)

        # Print the step information
        print(f"Step {step + 1}")
        for agent in env.agents:
            print(f"{agent}: Action={actions[agent]}, Reward={rewards[agent]}, Observations={observations[agent]}")

        # Check if the environment is done
        if all(dones.values()):
            print("All agents are done. Ending the simulation.")
            break


# def q_learning_main():
#     # Ensure the number of classrooms and action levels match
#     num_classrooms = 2  # Example with 2 classrooms
#     env = MultiClassroomEnv(num_classrooms=num_classrooms, total_students=100, s_shared=10, max_weeks=52,
#                             action_levels_per_class=[3, 3], seed=42)  # Only 2 action levels for 2 classrooms
#
#     # Define the agents
#     agents = [f'classroom_{i}' for i in range(num_classrooms)]  # Matching number of classrooms
#
#     # Generate the full state space: all combinations of infected values (0, 10, ..., 100) and community risk values (0, 0.1, ..., 1.0)
#     infected_values = range(0, 101, 10)  # Discretized infected values: 0, 10, 20, ..., 100
#     community_risk_values = [i / 10 for i in range(11)]  # Discretized community risk values: 0, 0.1, ..., 1.0
#     state_space = list(itertools.product(infected_values, community_risk_values))  # All state combinations
#
#     # Action space is discrete, based on the environment's action space
#     action_space_size = env.action_spaces[agents[0]].n
#
#     # Initialize the Q-learning agent with the full state space
#     agent = IndependentQLearningAgent(agents, state_space, action_space_size)
#
#     # Train the agent in the environment
#     agent.train(env, max_steps=52)
#
#     # Test the trained model by printing the actions for specific state combinations
#     print("\nTesting the trained model on specific state combinations of infected and community risk:")
#
#     # Define discrete values for infected and community risk for testing
#     # Generate 10 values between 0 and 100 for infected_test_values
#     infected_test_values = np.linspace(0, 100, num=10).astype(int)  # 10 evenly spaced values from 0 to 100
#
#     # Generate 10 values between 0 and 1 for community_risk_test_values
#     community_risk_test_values = np.linspace(0.0, 1.0, num=10)  # 10 evenly spaced values from 0 to 1
#
#     for agent_name in agents:
#         print(f"\nActions for {agent_name}:")
#
#         # Iterate over the combinations of infected values and community risk values
#         for infected in infected_test_values:
#             for community_risk in community_risk_test_values:
#                 # Prepare state with matching input dimensions (infected, community_risk)
#                 state = (infected, community_risk)
#
#                 # Select action based on the learned Q-values
#                 action = agent.select_action(agent_name, state)
#
#                 # Print the state and the selected action
#                 print(f"State (Infected: {infected}, Community Risk: {community_risk:.2f}) -> Action: {action}")
#
#         agent.visualize_q_table(agent_name)
#
#         save_path = f"./results/{agent_name}_all_states.png"
#         plot_path = visualize_all_states(agent_name, env, agent.q_tables, save_path=save_path)
#         print(f"All States visualization saved at {plot_path}")
#
# if __name__ == "__main__":
#     main()


# def main():
#     # Initialize the multi-classroom environment with 2 classrooms
#     num_classrooms = 2  # Example with 2 classrooms
#     env = MultiClassroomEnv(num_classrooms=num_classrooms, total_students=100, s_shared=10, max_weeks=20,
#                             action_levels_per_class=[3, 3], seed=42)
#
#     # Initialize Myopic Policy and pass the environment to it
#     myopic_agent = MyopicPolicy(env)
#
#     # Define parameters for evaluation
#     run_name = "myopic_evaluation"
#     num_episodes = 1
#     alpha = 0.4  # Use the alpha value for reward calculation
#
#     # Evaluate the Myopic Policy
#     avg_reward = myopic_agent.evaluate(env, run_name=run_name, num_episodes=num_episodes, alpha=alpha, csv_path=None)
#     print(f"Average Reward for Myopic Policy with alpha {alpha}: {avg_reward}")
#
#     visualize_myopic_states(myopic_agent, env, alpha)

# if __name__ == "__main__":
#     main()

def main():
    # Ensure the number of classrooms and action levels match
    num_classrooms = 2  # Example with 2 classrooms
    env = MultiClassroomEnv(num_classrooms=num_classrooms, total_students=100, s_shared=10, max_weeks=52,
                            action_levels_per_class=[3, 3], seed=42)  # Only 2 action levels for 2 classrooms

    # Define the agents
    agents = [f'classroom_{i}' for i in range(num_classrooms)]  # Matching number of classrooms

    # State space size is the observation space size
    state_dim = env.observation_spaces[agents[0]].shape[0]

    # Action space is discrete, based on the environment's action space
    action_space_size = env.action_spaces[agents[0]].n

    # Initialize the DQN agent with the state space size
    agent = DQNAgent(agents, state_dim, action_space_size)

    # Train the agent in the environment
    agent.train(env, max_steps=52)

    # Test the trained model by printing the actions for specific state combinations
    print("\nTesting the trained model on specific state combinations of infected and community risk:")

    # Define discrete values for infected and community risk for testing
    infected_test_values = np.linspace(0, 100, num=10).astype(int)  # 10 evenly spaced values from 0 to 100
    community_risk_test_values = np.linspace(0.0, 1.0, num=10)  # 10 evenly spaced values from 0 to 1

    for agent_name in agents:
        print(f"\nActions for {agent_name}:")

        # Iterate over the combinations of infected values and community risk values
        for infected in infected_test_values:
            for community_risk in community_risk_test_values:
                # Prepare state with matching input dimensions (infected, community_risk)
                state = np.array([infected, community_risk])

                # Select action based on the learned Q-values from the DQN model
                action = agent.select_action(agent_name, state)

                # Print the state and the selected action
                print(f"State (Infected: {infected}, Community Risk: {community_risk:.2f}) -> Action: {action}")

        # Visualize the learned policy for all states
        save_path = f"./results/{agent_name}_all_states_dqn.png"
        plot_path = visualize_all_states_dqn(agent_name, env, agent, save_path=save_path)
        print(f"All States visualization saved at {plot_path}")


if __name__ == "__main__":
    main()