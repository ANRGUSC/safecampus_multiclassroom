import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

# Define the neural network architecture for DQN
class DQNetwork(nn.Module):
    def __init__(self, input_dim, action_space_size):
        super(DQNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, action_space_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)  # Output Q-values for each action

class DQNAgent:
    def __init__(self, agents, state_dim, action_space_size, learning_rate=0.0001, discount_factor=0.99,
                 epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.000001, batch_size=64, memory_size=10000):
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
        self.memory = []  # Replay buffer

        # Initialize networks for each agent
        self.networks = {agent: DQNetwork(state_dim, action_space_size) for agent in agents}
        self.target_networks = {agent: DQNetwork(state_dim, action_space_size) for agent in agents}
        self.optimizers = {agent: optim.Adam(self.networks[agent].parameters(), lr=learning_rate) for agent in agents}

        # Initialize target networks to have the same weights as the primary networks
        for agent in self.agents:
            self.target_networks[agent].load_state_dict(self.networks[agent].state_dict())

        # Track rewards over episodes for visualization
        self.episode_rewards = {agent: [] for agent in agents}

    def select_action(self, agent, state):
        """Select an action using epsilon-greedy strategy."""
        if random.random() < self.epsilon:
            return random.randint(0, self.action_space_size - 1)  # Random action (exploration)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)  # Convert state to tensor
            q_values = self.networks[agent](state_tensor)  # Get Q-values from the network
            return torch.argmax(q_values).item()  # Return the action with the highest Q-value

    def store_transition(self, agent, state, action, reward, next_state, done):
        """Store the experience in replay buffer."""
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)
        self.memory.append((agent, state, action, reward, next_state, done))

    def sample_experiences(self):
        """Sample a batch of experiences from the replay buffer."""
        return random.sample(self.memory, min(len(self.memory), self.batch_size))

    def update_network(self, agent):
        """Update the DQN network by sampling experiences and applying the Bellman equation."""
        if len(self.memory) < self.batch_size:
            return  # Not enough experiences to sample a batch

        # Sample a batch of experiences
        experiences = self.sample_experiences()
        batch_agent, batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*experiences)

        batch_state = torch.FloatTensor(np.array(batch_state))
        batch_action = torch.LongTensor(batch_action).unsqueeze(1)
        batch_reward = torch.FloatTensor(batch_reward)
        batch_next_state = torch.FloatTensor(np.array(batch_next_state))
        batch_done = torch.FloatTensor(batch_done)

        # Compute the current Q-values using the primary network
        current_q_values = self.networks[agent](batch_state).gather(1, batch_action).squeeze()

        # Compute the target Q-values using the target network
        with torch.no_grad():
            max_next_q_values = self.target_networks[agent](batch_next_state).max(1)[0]
            target_q_values = batch_reward + (1 - batch_done) * self.discount_factor * max_next_q_values

        # Compute the loss between current and target Q-values
        loss = nn.MSELoss()(current_q_values, target_q_values)

        # Perform gradient descent to minimize the loss
        self.optimizers[agent].zero_grad()
        loss.backward()
        self.optimizers[agent].step()

    def update_target_network(self, agent):
        """Update the target network to slowly track the primary network."""
        self.target_networks[agent].load_state_dict(self.networks[agent].state_dict())

    def train(self, env, max_steps=30, update_target_steps=100):
        pbar = tqdm(total=1000, desc="Training Progress", leave=True)
        for episode in range(1000):
            states = env.reset()
            total_rewards = {agent: 0 for agent in self.agents}

            for step in range(max_steps):
                actions = {agent: self.select_action(agent, states[agent]) for agent in self.agents}
                next_states, rewards, dones, _ = env.step(actions)

                for agent in self.agents:
                    action = actions[agent]
                    reward = rewards[agent]
                    next_state = next_states[agent]
                    done = dones[agent]
                    self.store_transition(agent, states[agent], action, reward, next_state, done)
                    self.update_network(agent)
                    total_rewards[agent] += reward

                states = next_states
                if all(dones.values()):
                    break

            # Update epsilon (exploration rate) using polynomial decay
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

            # Update target networks periodically
            if episode % update_target_steps == 0:
                for agent in self.agents:
                    self.update_target_network(agent)

            # Log the total rewards for each agent at the end of the episode
            for agent in self.agents:
                self.episode_rewards[agent].append(total_rewards[agent])

            pbar.update(1)

        pbar.close()
        rewards_path = f"results/avg_rewards_dqn_{env.gamma}.png"
        self.plot_rewards(rewards_path)

    def plot_rewards(self, save_path="results/avg_rewards_dqn.png"):
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
