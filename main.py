from environment.multiclassroom import MultiClassroomEnv
from agents.dqn_agent import DQNAgent
from agents.doubledqn_agent import DoubleDQNAgent
from agents.q_learning import IndependentQLearningAgent
from agents.ppo_agent import PPOAgent
from agents.a2c_agent import CentralizedA2CAgent
from utils.visualization import visualize_all_states, visualize_all_states_dqn, visualize_all_states_ppo
import numpy as np
import itertools
SEED = 42
np.random.seed(SEED)


def q_learning_run(gamma=0.2):
    num_classrooms = 2  # Example with 3 classrooms
    env = MultiClassroomEnv(num_classrooms=num_classrooms, total_students=100, max_weeks=52,
                            action_levels_per_class=[3, 3, 3], seed=42, gamma=gamma)

    agents = [f'classroom_{i}' for i in range(num_classrooms)]
    infected_values = range(0, 101, 10)
    community_risk_values = [i / 10 for i in range(11)]
    state_space = list(itertools.product(infected_values, community_risk_values))
    action_space_size = env.action_spaces[agents[0]].n

    agent = IndependentQLearningAgent(agents, state_space, action_space_size)
    agent.train(env, max_steps=52)

    for agent_name in agents:
        infected_test_values = np.linspace(0, 100, num=10).astype(int)
        community_risk_test_values = np.linspace(0.0, 1.0, num=10)

        # print(f"\nActions for {agent_name}:")
        # for infected in infected_test_values:
        #     for community_risk in community_risk_test_values:
        #         state = (infected, community_risk)
        #         action = agent.select_action(agent_name, state)
        #         print(f"State (Infected: {infected}, Community Risk: {community_risk:.2f}) -> Action: {action}")

        save_path = f"./results/{agent_name}_q_learning_all_states_{env.gamma}.png"
        visualize_all_states(agent_name, env, agent.q1_tables, agent.q2_tables, save_path=save_path)

def dqn_run(gamma=0.2):
    num_classrooms = 2
    env = MultiClassroomEnv(num_classrooms=num_classrooms, total_students=100, max_weeks=52,
                            action_levels_per_class=[5, 5], seed=42, gamma=gamma)

    agents = [f'classroom_{i}' for i in range(num_classrooms)]
    state_dim = env.observation_spaces[agents[0]].shape[0]
    action_space_size = env.action_spaces[agents[0]].n

    agent = DQNAgent(agents, state_dim, action_space_size)
    agent.train(env, max_steps=100)

    for agent_name in agents:
        save_path = f"./results/{agent_name}_all_states_dqn_{env.gamma}.png"
        visualize_all_states_dqn(agent_name, env, agent, save_path=save_path)
        agent.evaluate(env, max_steps=30)

def doubledqn_run(gamma=0.2):
    num_classrooms = 2
    env = MultiClassroomEnv(num_classrooms=num_classrooms, total_students=100, max_weeks=52,
                            action_levels_per_class=[5, 5], seed=42, gamma=gamma)

    agents = [f'classroom_{i}' for i in range(num_classrooms)]
    state_dim = env.observation_spaces[agents[0]].shape[0]
    action_space_size = env.action_spaces[agents[0]].n

    agent = DoubleDQNAgent(agents, state_dim, action_space_size)
    agent.train(env, max_steps=100)

    for agent_name in agents:
        save_path = f"./results/{agent_name}_all_states_dqn_{env.gamma}.png"
        visualize_all_states_dqn(agent_name, env, agent, save_path=save_path)
def a2c_run(gamma=0.2):
    num_classrooms = 2  # Number of classrooms (agents)
    env = MultiClassroomEnv(num_classrooms=num_classrooms, total_students=100, max_weeks=52,
                            action_levels_per_class=[3, 3], seed=42, gamma=gamma)

    agents = [f'classroom_{i}' for i in range(num_classrooms)]
    state_dim = 2  # Local state dimension (infected, community risk) for each agent
    global_state_dim = num_classrooms * state_dim  # Global state for centralized critic (combines all agents' states)
    action_space_size = env.action_spaces[agents[0]].n

    # Initialize the centralized A2C agent with decentralized actors and centralized critic
    agent = CentralizedA2CAgent(agents, state_dim, global_state_dim, action_space_size)

    # Train the centralized A2C agent
    agent.train(env, max_steps=52)

    # Test the learned policies
    for agent_name in agents:
        # Save visualizations of the learned policies
        save_path = f"./results/{agent_name}_all_states_a2c_centralized_{env.gamma}.png"

        visualize_all_states_dqn(agent_name, env, agent, save_path=save_path)
    eval_path = f"./results/eval_a2c_centralized_{env.gamma}.png"
    agent.evaluate(env, max_steps=52, save_path=eval_path)
def ppo_run(gamma=0.2):
    num_classrooms = 2  # Number of classrooms (agents)
    env = MultiClassroomEnv(num_classrooms=num_classrooms, total_students=100, max_weeks=52,
                            action_levels_per_class=[3, 3], seed=42, gamma=gamma)

    agents = [f'classroom_{i}' for i in range(num_classrooms)]
    state_dim = 2  # Local state dimension (infected, community risk) for each agent
    action_space_size = env.action_spaces[agents[0]].n

    # Initialize the PPO agent
    agent = PPOAgent(agents, state_dim, action_space_size)

    # Train the PPO agent
    agent.train(env, max_steps=30)

    # Test the learned policies
    for agent_name in agents:
        # Save visualizations of the learned policies
        save_path = f"./results/{agent_name}_all_states_ppo_{env.gamma}.png"
        visualize_all_states_ppo(agent_name, env, agent, save_path=save_path)
def main():
    # Define the gamma values to test
    gamma_values = [0.2]

    # Loop over each gamma value and run the training for each agent type
    for gamma in gamma_values:
        # print(f"\nRunning Q-learning with gamma = {gamma}")
        # q_learning_run(gamma)

        # print(f"\nRunning DQN with gamma = {gamma}")
        # dqn_run(gamma)

        # print(f"\nRunning DQN with gamma = {gamma}")
        # doubledqn_run(gamma)
        #
        print(f"\nRunning A2C with gamma = {gamma}")
        a2c_run(gamma)
        # print(f"\nRunning PPO with gamma = {gamma}")
        # ppo_run(gamma)

if __name__ == "__main__":
    main()