from environment.multiclassroom import MultiClassroomEnv
from agents.dqn_agent import DQNAgent
from agents.q_learning import IndependentQLearningAgent
from utils.visualization import visualize_all_states, visualize_all_states_dqn
import numpy as np
import itertools
def q_learning_run():
    # Ensure the number of classrooms and action levels match
    num_classrooms = 3  # Example with 2 classrooms
    env = MultiClassroomEnv(num_classrooms=num_classrooms, total_students=100, s_shared=10, max_weeks=52,
                            action_levels_per_class=[3, 3, 3], seed=42)  # Only 2 action levels for 2 classrooms

    # Define the agents
    agents = [f'classroom_{i}' for i in range(num_classrooms)]  # Matching number of classrooms

    # Generate the full state space: all combinations of infected values (0, 10, ..., 100) and community risk values (0, 0.1, ..., 1.0)
    infected_values = range(0, 101, 10)  # Discretized infected values: 0, 10, 20, ..., 100
    community_risk_values = [i / 10 for i in range(11)]  # Discretized community risk values: 0, 0.1, ..., 1.0
    state_space = list(itertools.product(infected_values, community_risk_values))  # All state combinations

    # Action space is discrete, based on the environment's action space
    action_space_size = env.action_spaces[agents[0]].n

    # Initialize the Q-learning agent with the full state space
    agent = IndependentQLearningAgent(agents, state_space, action_space_size)

    # Train the agent in the environment
    agent.train(env, max_steps=52)

    # Test the trained model by printing the actions for specific state combinations
    print("\nTesting the trained model on specific state combinations of infected and community risk:")

    # Define discrete values for infected and community risk for testing
    # Generate 10 values between 0 and 100 for infected_test_values
    infected_test_values = np.linspace(0, 100, num=10).astype(int)  # 10 evenly spaced values from 0 to 100

    # Generate 10 values between 0 and 1 for community_risk_test_values
    community_risk_test_values = np.linspace(0.0, 1.0, num=10)  # 10 evenly spaced values from 0 to 1

    for agent_name in agents:
        print(f"\nActions for {agent_name}:")

        # Iterate over the combinations of infected values and community risk values
        for infected in infected_test_values:
            for community_risk in community_risk_test_values:
                # Prepare state with matching input dimensions (infected, community_risk)
                state = (infected, community_risk)

                # Select action based on the learned Q-values
                action = agent.select_action(agent_name, state)

                # Print the state and the selected action
                print(f"State (Infected: {infected}, Community Risk: {community_risk:.2f}) -> Action: {action}")

        agent.visualize_q_table(agent_name)

        save_path = f"./results/{agent_name}_q_learning_all_states_{env.gamma}.png"
        plot_path = visualize_all_states(agent_name, env, agent.q_tables, save_path=save_path)
        print(f"All States visualization saved at {plot_path}")
def dqn_run():
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
        save_path = f"./results/{agent_name}_all_states_dqn_{env.gamma}.png"
        plot_path = visualize_all_states_dqn(agent_name, env, agent, save_path=save_path)
        print(f"All States visualization saved at {plot_path}")

def main():
    # Run the Q-learning example
    # q_learning_run()

    # Run the DQN example
    dqn_run()

if __name__ == "__main__":
    main()