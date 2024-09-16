import numpy as np
import random
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import itertools
import matplotlib.patches as mpatches
import os
import colorsys


class IndependentQLearningAgent:
    def __init__(self, agents, state_space_size, action_space_size, learning_rate=0.00001, discount_factor=0.99,
                 epsilon=1.0, epsilon_decay=0.999, min_epsilon=0.00001):
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

        # Initialize Q-tables for each agent: rows are states, columns are actions
        self.q_tables = {
            agent: np.zeros((self.state_space_size, action_space_size))  # Each state row, each action column
            for agent in agents
        }

        # Track rewards over episodes for visualization
        self.episode_rewards = {agent: [] for agent in agents}

    def discretize_value(self, value, step_size):
        """Discretize the value based on the given step size."""
        return int(min(max(round(value / step_size) * step_size, 0), 100))

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

    def update_q_table(self, agent, state, action, reward, next_state):
        """Update Q-table using Q-learning update rule."""
        state_idx = self.get_state_index(state)  # Get index of current state
        next_state_idx = self.get_state_index(next_state)  # Get index of next state

        # Q-learning update rule
        best_next_action = np.argmax(self.q_tables[agent][next_state_idx])  # Best next action
        td_target = reward + self.q_tables[agent][next_state_idx][best_next_action]
        td_error = td_target - self.q_tables[agent][state_idx][action]
        self.q_tables[agent][state_idx][action] += self.learning_rate * td_error  # Update Q-table

    def select_action(self, agent, state):
        """Select an action using epsilon-greedy strategy."""
        state_idx = self.get_state_index(state)  # Get index of current state
        if random.random() < self.epsilon:  # Explore
            return random.randint(0, self.q_tables[agent].shape[1] - 1)
        else:  # Exploit (choose best action)
            return np.argmax(self.q_tables[agent][state_idx])

    def train(self, env, max_steps=30):
        pbar = tqdm(total=5000, desc="Training Progress", leave=True)
        for episode in range(5000):
            states = env.reset()
            total_rewards = {agent: 0 for agent in self.agents}

            for step in range(max_steps):
                actions = {agent: self.select_action(agent, states[agent]) for agent in self.agents}
                next_states, rewards, dones, _ = env.step(actions)


                for agent in self.agents:
                    action = actions[agent]
                    reward = rewards[agent]
                    next_state = next_states[agent]
                    self.update_q_table(agent, states[agent], action, reward, next_state)
                    total_rewards[agent] += reward

                states = next_states

                if all(dones.values()):
                    break
            # Log the total rewards for each agent at the end of the episode
            for agent in self.agents:
                self.episode_rewards[agent].append(total_rewards[agent])
            # Decay epsilon
            self.epsilon = self.min_epsilon + (1.0 - self.min_epsilon) * (1 - episode / 5000) ** 2
            # self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
            pbar.update(1)

        pbar.close()
        rewards_path = f"results/avg_rewards_q_learning_{env.gamma}.png"
        self.plot_rewards(save_path=rewards_path)

    def plot_rewards(self, save_path="results/avg_rewards_q_learning.png"):
        """Plot average rewards over training episodes for each agent."""
        plt.figure(figsize=(10, 5))

        for agent, rewards in self.episode_rewards.items():
            plt.plot(rewards, label=f"{agent} Rewards")

        plt.title("Episode Rewards During Training")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.legend()
        plt.grid(True)
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        plt.savefig(save_path)
        plt.show()

    def visualize_q_table(self, agent):
        """
        Visualize the learned Q-table for a specific agent.
        Args:
            agent: The agent whose Q-table we are visualizing.
        """
        q_table = self.q_tables[agent]

        # Get the max Q-value for each state-action pair
        max_q_values = np.max(q_table, axis=1)  # Take max over the action space (axis=1)

        # Reshape the Q-values for plotting
        plt.imshow(max_q_values.reshape(11, 11), extent=[0, 100, 0, 100], origin='lower', cmap='coolwarm')
        plt.colorbar(label='Max Q-value')
        plt.xlabel('Community Risk (%)')
        plt.ylabel('Infected Students')
        plt.title(f'Q-table Visualization for {agent}')

        # Save the plot
        save_path = os.path.join("results", f'q_table_{agent}.png')
        plt.savefig(save_path)
        # plt.close()

        print(f"Q-table visualization saved for {agent} at {save_path}")


