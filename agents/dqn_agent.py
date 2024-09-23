import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

class DQNetwork(nn.Module):
    def __init__(self, input_dim, action_space_size):
        super(DQNetwork, self).__init__()
        self.input_dim = input_dim
        self.action_space_size = action_space_size
        self.hidden_layers = 3
        self.hidden_dim = 16
        self.build_network()
        self.initialize_weights()

    def build_network(self):
        layers = []
        prev_dim = self.input_dim

        for _ in range(self.hidden_layers):
            layers.append(nn.Linear(prev_dim, self.hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = self.hidden_dim

        layers.append(nn.Linear(prev_dim, self.action_space_size))
        self.network = nn.Sequential(*layers)

    def initialize_weights(self):
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.constant_(layer.weight, 0)
                nn.init.constant_(layer.bias, 0)

    def forward(self, state):
        return self.network(state)

class DQNAgent:
    def __init__(self, agents, state_dim, action_space_size, learning_rate=0.1, discount_factor=0.99,
                 epsilon=1.0, epsilon_decay=0.99, min_epsilon=0.0001, batch_size=64, memory_size=10000):
        self.agents = agents
        self.state_dim = state_dim
        self.action_space_size = action_space_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.memory = []

        self.networks = {agent: DQNetwork(state_dim, action_space_size) for agent in agents}
        self.target_networks = {agent: DQNetwork(state_dim, action_space_size) for agent in agents}
        self.optimizers = {agent: optim.Adam(self.networks[agent].parameters(), lr=learning_rate) for agent in agents}

        for agent in self.agents:
            self.target_networks[agent].load_state_dict(self.networks[agent].state_dict())

        self.episode_rewards = {agent: [] for agent in agents}
        self.global_rewards = []
        self.unique_states = {agent: set() for agent in agents}

    def select_action(self, agent, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_space_size - 1)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.networks[agent](state_tensor)
            return torch.argmax(q_values).item()

    def normalize_reward(self, rewards):
        """
        Normalize a batch of rewards to have mean 0 and standard deviation 1.
        Add a small epsilon to prevent division by zero.
        """
        rewards = np.array(rewards)
        return (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-8)

    def store_transition(self, agent, state, action, reward, next_state, done):
        # reward = self.normalize_reward(reward)
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)
        self.memory.append((agent, state, action, reward, next_state, done))

    def sample_experiences(self):
        return random.sample(self.memory, min(len(self.memory), self.batch_size))

    def normalize_combined_reward(self, combined_reward, min_combined, max_combined):
        """
        Normalize the combined reward to the range [0, 1].
        """
        return (combined_reward - min_combined) / (
                    max_combined - min_combined + 1e-8)  # Add epsilon to avoid division by zero

    def update_network(self, agent, global_reward):
        if len(self.memory) < self.batch_size:
            return

        # Sample experiences from memory
        experiences = self.sample_experiences()
        batch_agent, batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*experiences)

        # Convert to tensors
        batch_state = torch.FloatTensor(np.array(batch_state))
        batch_action = torch.LongTensor(batch_action).unsqueeze(1)
        batch_reward = torch.FloatTensor(batch_reward)
        batch_next_state = torch.FloatTensor(np.array(batch_next_state))
        batch_done = torch.FloatTensor(batch_done)

        # Get current Q-values from the agent's network
        current_q_values = self.networks[agent](batch_state).gather(1, batch_action).squeeze()

        # Compute max next Q-values from the target network
        with torch.no_grad():
            max_next_q_values = self.target_networks[agent](batch_next_state).max(1)[0]

            # Calculate the combined reward (global * batch reward)
            combined_reward = (global_reward * batch_reward)

            # Normalize the combined reward to [0, 1]
            min_combined = combined_reward.min().item()  # Get the minimum combined reward
            max_combined = combined_reward.max().item()  # Get the maximum combined reward
            normalized_combined_reward = self.normalize_combined_reward(combined_reward, min_combined, max_combined)

            # Compute target Q-values using the normalized combined reward
            target_q_values = normalized_combined_reward + (1 - batch_done) * max_next_q_values

            # target_q_values = torch.where(batch_done == 1, combined_reward, target_q_values)

        # Compute the loss between current Q-values and target Q-values
        loss = nn.MSELoss()(current_q_values, target_q_values)

        # Perform backpropagation and update the network
        self.optimizers[agent].zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.networks[agent].parameters(), max_norm=1.0)
        self.optimizers[agent].step()

    def update_target_network(self, agent, tau=0.01):
        """Soft update the target network parameters using state_dict."""
        for key in self.networks[agent].state_dict().keys():
            self.target_networks[agent].state_dict()[key].data.copy_(
                tau * self.networks[agent].state_dict()[key].data + (1.0 - tau) *
                self.target_networks[agent].state_dict()[key].data
            )

    def linear_decay(self, episode, total_episodes):
        return max(self.min_epsilon, self.epsilon - (self.epsilon - self.min_epsilon) * (episode / total_episodes))

    def exponential_decay(self, episode, decay_rate=0.8):
        return max(self.min_epsilon, self.epsilon * (1 - decay_rate) ** episode)

    def inverse_time_decay(self, episode, decay_rate=0.8):
        return max(self.min_epsilon, self.epsilon / (1 + decay_rate * episode))

    def polynomial_decay(self, episode, total_episodes, power=2):
        return max(self.min_epsilon, self.epsilon * (1 - episode / total_episodes) ** power)

    def cosine_decay(self, episode, total_episodes):
        return self.min_epsilon + 0.5 * (self.epsilon - self.min_epsilon) * (
                    1 + np.cos(np.pi * episode / total_episodes))

    def step_decay(self, episode, step_size=10, decay_factor=0.5):
        return max(self.min_epsilon, self.epsilon * (decay_factor ** (episode // step_size)))

    def sigmoid_decay(self, episode, total_episodes):
        return self.min_epsilon + (self.epsilon - self.min_epsilon) / (
                    1 + np.exp(10 * (episode / total_episodes - 0.5)))

    def logarithmic_decay(self, episode, decay_rate=0.01):
        return max(self.min_epsilon, self.epsilon / (1 + decay_rate * np.log(1 + episode)))

    def hyperbolic_decay(self, episode, decay_rate=0.01):
        return max(self.min_epsilon, self.epsilon / (1 + decay_rate * episode ** 2))

    def staircase_decay(self, episode, step_size=10, decay_factor=0.5):
        return max(self.min_epsilon, self.epsilon * (decay_factor ** (episode // step_size)))

    def adaptive_decay(self, episode, reward, threshold=0.1):
        if reward > threshold:
            return max(self.min_epsilon, self.epsilon * 0.99)
        else:
            return self.epsilon
    def train(self, env, max_steps=30, update_target_steps=100):
        total_episodes = 500
        pbar = tqdm(total=total_episodes, desc="Training Progress", leave=True)
        for episode in range(total_episodes):
            states = env.reset()
            total_rewards = {agent: 0 for agent in self.agents}
            global_reward = 0

            for step in range(max_steps):
                actions = {agent: self.select_action(agent, states[agent]) for agent in self.agents}
                next_states, rewards, dones, _ = env.step(actions)

                for agent in self.agents:
                    action = actions[agent]
                    reward = rewards[agent]
                    next_state = next_states[agent]
                    done = dones[agent]
                    self.store_transition(agent, states[agent], action, reward, next_state, done)
                    self.update_network(agent, global_reward)
                    total_rewards[agent] += reward

                    self.unique_states[agent].add(tuple(states[agent]))

                global_reward += sum(rewards.values())
                states = next_states
                if all(dones.values()):
                    break

            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
            # self.epsilon = self.exponential_decay(episode)
            # self.epsilon = self.cosine_decay(episode, total_episodes)
            # self.epsilon = self.inverse_time_decay(episode)
            # self.epsilon = self.sigmoid_decay(episode, total_episodes)
            # self.epsilon = self.logarithmic_decay(episode)
            # self.epsilon = self.hyperbolic_decay(episode)
            # self.epsilon = self.staircase_decay(episode)

            # if episode % update_target_steps == 0:
            for agent in self.agents:
                self.update_target_network(agent)

            for agent in self.agents:
                self.episode_rewards[agent].append(total_rewards[agent])
            self.global_rewards.append(global_reward)

            pbar.update(1)

        pbar.close()
        rewards_path = f"results/avg_rewards_dqn3_{env.gamma}.png"
        self.plot_rewards(rewards_path)

        for agent in self.agents:
            print(f"Agent {agent} visited {len(self.unique_states[agent])} unique states.")

    def plot_rewards(self, save_path="results/avg_rewards_dqn.png"):
        plt.figure(figsize=(10, 5))

        for agent, rewards in self.episode_rewards.items():
            plt.plot(rewards, label=f"{agent} Rewards")
        plt.plot(self.global_rewards, label="Global Rewards", linestyle='--')

        plt.title("Episode Rewards During Training")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.legend()
        plt.grid(True)
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        plt.savefig(save_path)

    def evaluate(self, env, max_steps=52):
        states = env.reset()
        episode_infected = {agent: [] for agent in self.agents}
        episode_allowed = {agent: [] for agent in self.agents}
        episode_community_risk = {agent: [] for agent in self.agents}

        for step in range(max_steps):
            actions = {agent: self.select_action(agent, states[agent]) for agent in self.agents}
            next_states, rewards, dones, _ = env.step(actions)

            for agent in self.agents:
                episode_infected[agent].append(env.student_status[self.agents.index(agent)])
                episode_allowed[agent].append(env.allowed_students[self.agents.index(agent)])
                episode_community_risk[agent].append(
                    env.shared_community_risk[min(env.current_week, env.max_weeks - 1)])
            states = next_states
            if all(dones.values()):
                break

        self.plot_infected_allowed_and_risk_over_time(episode_infected, episode_allowed, episode_community_risk,
                                                      save_path="results/infected_allowed_and_risk_over_time_evaluation.png")

    def plot_infected_allowed_and_risk_over_time(self, infected_over_time, allowed_over_time, community_risk_over_time,
                                                 save_path="results/infected_allowed_and_risk_over_time.png"):
        num_agents = len(infected_over_time)
        fig, axes = plt.subplots(num_agents, 1, figsize=(10, 5 * num_agents), sharex=True)

        if num_agents == 1:
            axes = [axes]

        for i, agent in enumerate(infected_over_time):
            ax = axes[i]
            ax.bar(range(len(allowed_over_time[agent])), allowed_over_time[agent], alpha=0.6,
                   label=f"{agent} Allowed Over Time")
            ax.plot(infected_over_time[agent], color='r', label=f"{agent} Infected Over Time")
            ax.set_title(f"{agent} - Number of Infected, Allowed Students, and Community Risk Over Time")
            ax.set_xlabel("Step")
            ax.set_ylabel("Number of Students")
            ax.legend(loc='upper left')
            ax.grid(True)

            ax2 = ax.twinx()
            ax2.plot(community_risk_over_time[agent], color='g', linestyle='--',
                     label=f"{agent} Community Risk Over Time")
            ax2.set_ylabel("Community Risk")
            ax2.legend(loc='upper right')

        plt.tight_layout()
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        plt.savefig(save_path)
