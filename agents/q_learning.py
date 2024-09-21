import numpy as np
import random
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import itertools
SEED = 42
np.random.seed(SEED)

class IndependentQLearningAgent:
    def __init__(self, agents, state_space_size, action_space_size, learning_rate=0.0001, discount_factor=0.99,
                 epsilon=1.0, epsilon_decay=0.1, min_epsilon=0.01):
        self.agents = agents

        # Define the discretized state space (infected and community_risk values with steps of 10)
        infected_values = range(0, 101, 10)  # From 0 to 100, in steps of 10
        community_risk_values = [i / 100 for i in range(0, 101, 10)]  # From 0.0 to 1.0, in steps of 0.1
        self.state_space = list(itertools.product(infected_values, community_risk_values))

        self.state_space_size = len(self.state_space)  # Number of discretized states
        self.action_space_size = action_space_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

        # Initialize two separate Q-tables for each agent: rows are states, columns are actions
        self.q1_tables = {
            agent: np.zeros((self.state_space_size, action_space_size)) for agent in agents
        }
        self.q2_tables = {
            agent: np.zeros((self.state_space_size, action_space_size)) for agent in agents
        }

        # Track rewards over episodes for visualization
        self.episode_rewards = {agent: [] for agent in agents}  # Track individual rewards per agent
        self.episode_rewards['global'] = []

        # Variables for tracking epsilon annealing
        self.recent_rewards = []  # Store recent rewards for performance comparison
        self.window_size = 10  # Window size for calculating average rewards
        self.previous_average_reward = 0.0  # Store the previous average reward

        self.visited_states = {agent: set() for agent in agents}



    def get_state_index(self, state):
        """Convert continuous state into discrete index for the Q-table."""
        infected, community_risk = state

        # Discretize infected to the nearest multiple of 10
        discrete_infected = max(0, min(100, round(infected / 10) * 10))

        # Discretize community risk to the nearest multiple of 0.1
        discrete_community_risk = max(0, min(1.0, round(community_risk * 10) / 10))

        # Find the closest state in the state space
        closest_state = min(self.state_space, key=lambda x: (
            abs(x[0] - discrete_infected) + abs(x[1] - discrete_community_risk)
        ))

        return self.state_space.index(closest_state)

    def select_action(self, agent, state):
        """Select an action using epsilon-greedy strategy for each agent."""
        state_idx = self.get_state_index(state)  # Get index of current state
        if random.random() < self.epsilon:  # Explore
            return random.randint(0, self.q1_tables[agent].shape[1] - 1)
        else:  # Exploit (choose the best action based on the sum of Q1 and Q2)
            q_sum = self.q1_tables[agent][state_idx] + self.q2_tables[agent][state_idx]
            return np.argmax(q_sum)

    def update_q_tables(self, agent, state, action, global_reward, local_reward, next_state, episode, total_episodes):
        """Update Q1 and Q2 tables for the agent using a composite reward that includes global, local, fairness, and safety components."""
        state_idx = self.get_state_index(state)  # Get index of current state
        next_state_idx = self.get_state_index(next_state)  # Get index of next state

        # Combine global and local rewards (you can adjust the weight for each)
        combined_reward = max(global_reward, local_reward)  # Adjust the weight for global reward

        # Fairness reward: Penalize large deviations from the average performance
        average_local_performance = np.mean(
            [self.q1_tables[agent][state_idx] for agent in self.agents])  # Average across agents
        fairness_penalty = -abs(
            local_reward - average_local_performance)  # Penalize if local reward deviates too much from average

        # Safety reward: Encourage safer actions (e.g., minimize infections)
        infected, community_risk = state  # Extract relevant state information
        safety_reward = -infected  # Penalize based on the number of infections (higher infections = more penalty)

        # Combine all components into the final composite reward
        composite_reward = combined_reward + fairness_penalty + safety_reward  # Adjust weights as needed

        # Randomly update either Q1 or Q2 (Double Q-learning)
        if random.random() < 0.5:
            # Update Q1 using actions from Q2
            td_target = composite_reward
            td_error = td_target - self.q1_tables[agent][state_idx][action]
            self.q1_tables[agent][state_idx][action] += self.learning_rate * td_error
        else:
            # Update Q2 using actions from Q1
            td_target = composite_reward
            td_error = td_target - self.q2_tables[agent][state_idx][action]
            self.q2_tables[agent][state_idx][action] += self.learning_rate * td_error

    # def update_q_tables(self, agent, state, action, global_reward, local_reward, next_state, episode, total_episodes):
    #     """Update Q1 and Q2 tables for the agent using Double Q-learning with combined global and local rewards, using progressive reward."""
    #     state_idx = self.get_state_index(state)  # Get index of current state
    #     next_state_idx = self.get_state_index(next_state)  # Get index of next state
    #     reward = local_reward + (global_reward * 0.2)  # Combine local and global rewards
    #
    #     if random.random() < 0.5:
    #         # Update Q1 using actions from Q2
    #         td_target = reward
    #         td_error = td_target - self.q1_tables[agent][state_idx][action]
    #         self.q1_tables[agent][state_idx][action] += self.learning_rate * td_error
    #     else:
    #         # Update Q2 using actions from Q1
    #         td_target = reward
    #         td_error = td_target - self.q2_tables[agent][state_idx][action]
    #         self.q2_tables[agent][state_idx][action] += self.learning_rate * td_error

    def calculate_novelty(self, state, next_state):
        """Calculate novelty reward based on the difference between current and next state."""
        infected, community_risk = state
        next_infected, next_community_risk = next_state

        # Novelty based on how different the next state is compared to the current state
        novelty = abs(next_infected - infected) + abs(next_community_risk - community_risk)

        # Scale novelty reward (you can adjust this scaling)
        return novelty / 100.0  # Normalize to keep reward manageable

    def percentage_visited_states(self):
        """Calculate the percentage of visited states per agent."""
        percentages = {}
        for agent in self.agents:
            visited_states_count = len(self.visited_states[agent])
            percentage = (visited_states_count / self.state_space_size) * 100
            percentages[agent] = percentage
        return percentages

    def train(self, env, max_steps=30, alpha=0.9):
        total_episodes = 5000
        pbar = tqdm(total=total_episodes, desc="Training Progress", leave=True)
        for episode in range(total_episodes):
            states = env.reset()
            total_reward = 0  # Track global reward for each episode

            for step in range(max_steps):
                actions = {agent: self.select_action(agent, states[agent]) for agent in self.agents}
                next_states, rewards, dones, _ = env.step(actions)

                # Log visited states for each agent
                for agent in self.agents:
                    state_idx = self.get_state_index(states[agent])
                    self.visited_states[agent].add(state_idx)  # Track the visited state

                # Use the global reward from the environment
                global_reward = np.mean(list(rewards.values()))  # Extract the global reward (same for all agents)

                # Calculate fairness penalty and update Q-tables
                for agent in self.agents:
                    local_reward = rewards[agent]
                    # Update Q1 and Q2 tables using both global and local rewards, and fairness penalty
                    action = actions[agent]
                    self.update_q_tables(agent, states[agent], action, global_reward, local_reward, next_states[agent], episode, total_episodes)
                    # self.episode_rewards[agent].append(local_reward)  # Log the local reward

                total_reward += global_reward  # Update the total reward
                states = next_states

                if all(dones.values()):
                    break

            # Log the total global reward at the end of the episode
            self.episode_rewards['global'].append(total_reward)

            self.recent_rewards.append(total_reward)

            # Calculate the recent average reward over a window of episodes
            if len(self.recent_rewards) > self.window_size:
                self.recent_rewards.pop(0)
            recent_average_reward = np.mean(self.recent_rewards)

            # Anneal epsilon based on recent performance
            self.anneal_epsilon(episode, total_episodes, decay_power=2)

            pbar.update(1)

        pbar.close()
        # After training, calculate and print the percentage of visited states per agent
        visited_percentages = self.percentage_visited_states()
        for agent, percentage in visited_percentages.items():
            print(f"Percentage of states visited by {agent}: {percentage:.2f}%")

        rewards_path = f"results/avg_rewards_q_learning_{env.gamma}.png"
        self.plot_rewards(save_path=rewards_path)

    def anneal_epsilon(self, current_episode, total_episodes, decay_power=2):
        """Anneal epsilon using polynomial decay."""
        # Polynomial decay
        self.epsilon = max(self.min_epsilon, self.epsilon- (1 / total_episodes) * (self.epsilon - self.min_epsilon))

    def plot_rewards(self, save_path="results/avg_rewards_q_learning.png"):
        """Plot average global rewards and individual agent rewards over training episodes."""
        plt.figure(figsize=(10, 5))

        # Plot global rewards
        plt.plot(self.episode_rewards['global'], label="Global Rewards")

        # Plot individual agent rewards
        for agent in self.agents:
            plt.plot(self.episode_rewards[agent], label=f"{agent} Rewards")

        plt.title("Rewards During Training")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.legend()
        plt.grid(True)
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        plt.savefig(save_path)
        plt.show()
