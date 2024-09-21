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

    def build_network(self):
        layers = []
        prev_dim = self.input_dim

        for _ in range(self.hidden_layers):
            layers.append(nn.Linear(prev_dim, self.hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = self.hidden_dim

        layers.append(nn.Linear(prev_dim, self.action_space_size))
        self.network = nn.Sequential(*layers)

    def forward(self, state):
        return self.network(state)

class DoubleDQNAgent:
    def __init__(self, agents, state_dim, action_space_size, learning_rate_Q1=0.3, learning_rate_Q2=0.00001, discount_factor=0.99,
                 epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01):
        self.agents = agents
        self.state_dim = state_dim
        self.action_space_size = action_space_size
        self.learning_rate_Q1 = learning_rate_Q1
        self.learning_rate_Q2 = learning_rate_Q2
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

        self.networks_Q1 = {agent: DQNetwork(state_dim, action_space_size) for agent in agents}
        self.networks_Q2 = {agent: DQNetwork(state_dim, action_space_size) for agent in agents}
        self.target_networks_Q1 = {agent: DQNetwork(state_dim, action_space_size) for agent in agents}
        self.target_networks_Q2 = {agent: DQNetwork(state_dim, action_space_size) for agent in agents}
        self.optimizers_Q1 = {agent: optim.Adam(self.networks_Q1[agent].parameters(), lr=learning_rate_Q1) for agent in agents}
        self.optimizers_Q2 = {agent: optim.Adam(self.networks_Q2[agent].parameters(), lr=learning_rate_Q2) for agent in agents}

        for agent in self.agents:
            self.target_networks_Q1[agent].load_state_dict(self.networks_Q1[agent].state_dict())
            self.target_networks_Q2[agent].load_state_dict(self.networks_Q2[agent].state_dict())

        self.episode_rewards = {agent: [] for agent in agents}
        self.global_rewards = []
        self.unique_states = {agent: set() for agent in agents}

    # Rest of the class remains unchanged

    def normalize_reward(self, reward):
        return (reward - np.mean(reward)) / (np.std(reward) + 1e-8)

    def select_action(self, agent, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_space_size - 1)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.networks_Q1[agent](state_tensor)
            return torch.argmax(q_values).item()

    def update_network(self, agent, state, action, reward, next_state, done):
        state = torch.FloatTensor(state).unsqueeze(0)
        next_state = torch.FloatTensor(next_state).unsqueeze(0)
        action = torch.LongTensor([action]).unsqueeze(0)
        reward = torch.FloatTensor([self.normalize_reward(reward)])
        done = torch.FloatTensor([done])

        current_q_values_Q1 = self.networks_Q1[agent](state).gather(1, action).squeeze()
        current_q_values_Q2 = self.networks_Q2[agent](state).gather(1, action).squeeze()

        with torch.no_grad():
            max_next_q_values_Q1 = self.target_networks_Q1[agent](next_state).max(1)[0]
            max_next_q_values_Q2 = self.target_networks_Q2[agent](next_state).max(1)[0]

            if tuple(state.numpy().flatten()) in self.unique_states[agent]:
                chosen_q_value = torch.max(max_next_q_values_Q1, max_next_q_values_Q2)
            else:
                chosen_q_value = torch.min(max_next_q_values_Q1, max_next_q_values_Q2)

            target_q_values = reward + (1 - done) * self.discount_factor * chosen_q_value

        loss_Q1 = nn.MSELoss()(current_q_values_Q1, target_q_values)
        loss_Q2 = nn.MSELoss()(current_q_values_Q2, target_q_values)

        self.optimizers_Q1[agent].zero_grad()
        loss_Q1.backward()
        torch.nn.utils.clip_grad_norm_(self.networks_Q1[agent].parameters(), max_norm=1.0)
        self.optimizers_Q1[agent].step()

        self.optimizers_Q2[agent].zero_grad()
        loss_Q2.backward()
        torch.nn.utils.clip_grad_norm_(self.networks_Q2[agent].parameters(), max_norm=1.0)
        self.optimizers_Q2[agent].step()

    def update_target_network(self, agent, tau=0.01):
        for target_param, param in zip(self.target_networks_Q1[agent].parameters(), self.networks_Q1[agent].parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
        for target_param, param in zip(self.target_networks_Q2[agent].parameters(), self.networks_Q2[agent].parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def train(self, env, max_steps=30, update_target_steps=100):
        total_episodes = 1000
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
                    self.update_network(agent, states[agent], action, reward, next_state, done)
                    total_rewards[agent] += reward

                    self.unique_states[agent].add(tuple(states[agent]))

                global_reward += sum(rewards.values())
                states = next_states
                if all(dones.values()):
                    break

            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

            for agent in self.agents:
                self.update_target_network(agent)

            for agent in self.agents:
                self.episode_rewards[agent].append(total_rewards[agent])
            self.global_rewards.append(global_reward)

            pbar.update(1)

        pbar.close()
        rewards_path = f"results/avg_rewards_dqn_{env.gamma}.png"
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