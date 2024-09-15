import torch
import torch.nn as nn
import torch.optim as optim
import random
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import collections

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[idx] for idx in indices])

        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)

    def __len__(self):
        return len(self.buffer)

class DeepQNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64, num_hidden_layers=3, dropout_prob=0.01, leaky_relu_slope=0.01):
        """
        Deep Q-Network with linear layers, Leaky ReLU activations, Dropout, BatchNorm, and weight initialization.
        """
        super(DeepQNetwork, self).__init__()

        # Initialize the layers
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.leaky_relu = nn.LeakyReLU(negative_slope=leaky_relu_slope)  # Leaky ReLU instead of ReLU
        self.dropout = nn.Dropout(dropout_prob)

        # Hidden layers: hidden_dim -> hidden_dim
        self.hidden_layers = nn.ModuleList()
        for _ in range(num_hidden_layers - 1):
            hidden_layer = nn.Linear(hidden_dim, hidden_dim)
            batch_norm = nn.BatchNorm1d(hidden_dim)
            self.hidden_layers.append(nn.Sequential(hidden_layer, batch_norm, nn.LeakyReLU(negative_slope=leaky_relu_slope), nn.Dropout(dropout_prob)))

        # Output layer: hidden_dim -> output_dim
        self.output_layer = nn.Linear(hidden_dim, output_dim)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initialize weights for all layers of the network.
        Xavier initialization is applied to linear layers.
        """
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)  # Xavier initialization for weights
                nn.init.zeros_(layer.bias)  # Initialize biases to zero

    def forward(self, x):
        # Input layer
        x = self.leaky_relu(self.input_layer(x))

        # Hidden layers
        for hidden_layer in self.hidden_layers:
            if x.size(0) > 1:
                x = hidden_layer(x)  # Apply BatchNorm only if batch size > 1
            else:
                # Skip batch normalization when batch size is 1
                x = hidden_layer[0](x)  # Apply only the linear layer
                x = hidden_layer[2](x)  # Apply Leaky ReLU
                x = hidden_layer[3](x)  # Apply Dropout

        # Output layer
        return self.output_layer(x)



class DQNAgent:
    def __init__(self, agents, input_dim, output_dim, hidden_dim=32, learning_rate=0.00001,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay_steps=100000, max_episodes=1000,
                 results_dir="results", num_hidden_layers=10, seed=None, target_update_freq=100,
                 batch_size=64, buffer_capacity=10000):
        self.agents = agents
        self.output_dim = output_dim
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.step_count = 0
        self.max_episodes = max_episodes
        self.results_dir = results_dir
        self.batch_size = batch_size

        # Replay buffer for storing experiences
        self.replay_buffer = ReplayBuffer(buffer_capacity)

        # Initialize Q-networks for each agent
        self.networks = nn.ModuleDict(
            {agent: DeepQNetwork(input_dim, output_dim, hidden_dim, num_hidden_layers) for agent in agents}
        )

        # Initialize Target networks for each agent (starting with the same weights as the networks)
        self.target_networks = nn.ModuleDict(
            {agent: DeepQNetwork(input_dim, output_dim, hidden_dim, num_hidden_layers) for agent in agents}
        )
        self.optimizers = {agent: optim.Adam(self.networks[agent].parameters(), lr=learning_rate) for agent in agents}

        # Logging rewards, losses
        self.rewards_history = {agent: [] for agent in agents}
        self.avg_episode_rewards = []
        self.avg_episode_losses = []  # Track average loss per episode

        self.csv_file_path = os.path.join(results_dir, 'dqn_training_metrics.csv')
        self.target_update_freq = target_update_freq

        # Set seed for reproducibility
        if seed is not None:
            self.seed(seed)

    def seed(self, seed=None):
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

    def select_action(self, observations):
        actions = {}
        for agent, obs in observations.items():
            state = torch.FloatTensor(obs).unsqueeze(0)  # Add batch dimension
            state = state.view(1, -1)  # Reshape to (1, input_dim)

            if random.random() < self.epsilon:
                actions[agent] = random.randint(0, self.output_dim - 1)
            else:
                with torch.no_grad():
                    q_values = self.networks[agent](state)
                actions[agent] = q_values[0].argmax().item()
        return actions

    def update(self):
        # Ensure the replay buffer has enough samples for a batch
        if len(self.replay_buffer) < self.batch_size:
            return 0  # Return a loss of 0 if not enough samples are available

        # Sample a batch from the replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        losses = []
        for agent in self.agents:
            # Convert batch to tensors
            states = torch.FloatTensor(states)
            actions = torch.LongTensor(actions)
            rewards = torch.FloatTensor(rewards)
            next_states = torch.FloatTensor(next_states)
            dones = torch.FloatTensor(dones)

            # Get the predicted Q-values for the current state-action pairs
            q_values = self.networks[agent](states)
            q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)


            # Get the target Q-values: reward + gamma * max(next Q-value) if not done
            with torch.no_grad():
                next_q_values = self.target_networks[agent](next_states)
            next_q_value = next_q_values.max(1)[0]
            expected_q_value = rewards + (1 - dones) * next_q_value

            # Calculate loss between predicted Q-value and target Q-value
            loss = nn.MSELoss()(q_value, expected_q_value)

            # Backpropagate and update the network parameters
            self.optimizers[agent].zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.networks[agent].parameters(), max_norm=10.0)
            self.optimizers[agent].step()

            losses.append(loss.item())

        self.step_count += 1

        # Update epsilon for epsilon-greedy exploration
        self.epsilon = max(self.epsilon_end, self.epsilon * 0.95)

        # Periodically update the target network
        if self.step_count % self.target_update_freq == 0:
            self.update_target_network()

        return np.mean(losses)  # Ensure this returns a valid value, even if no update was performed

    def store_transition(self, state, action, reward, next_state, done):
        """Stores a transition in the replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)

    def update_target_network(self):
        """Copy the weights from the Q-network to the target network."""
        for agent in self.agents:
            self.target_networks[agent].load_state_dict(self.networks[agent].state_dict())

    def train(self, env, max_steps=52):
        pbar = tqdm(total=self.max_episodes, desc="Training Progress", leave=True)

        for episode in range(self.max_episodes):
            observations = env.reset()
            total_rewards = {agent: 0 for agent in self.agents}
            episode_losses = []

            for step in range(max_steps):
                # Select actions based on current observations
                actions = self.select_action(observations)

                # Take a step in the environment
                next_observations, rewards, dones, _ = env.step(actions)

                # Store transitions in the replay buffer
                for agent in self.agents:
                    self.store_transition(observations[agent], actions[agent], rewards[agent], next_observations[agent], dones[agent])

                # Update the Q-networks using a batch from the replay buffer
                loss = self.update()
                episode_losses.append(loss)

                # Accumulate rewards for logging
                for agent in self.agents:
                    total_rewards[agent] += rewards[agent]

                # Move to the next state
                observations = next_observations

                # If all agents are done, exit the loop
                if all(dones.values()):
                    break

            # Calculate and log average reward and loss for the episode
            avg_reward = np.mean(list(total_rewards.values()))
            avg_loss = np.mean(episode_losses)

            self.avg_episode_rewards.append(avg_reward)  # Store avg rewards per episode
            self.avg_episode_losses.append(avg_loss)  # Store avg loss per episode

            pbar.update(1)
            pbar.set_description(f"Avg Reward: {avg_reward:.2f}, Avg Loss: {avg_loss:.4f}")

        pbar.close()
        self.plot_rewards_and_losses()
        self.save_model()  # Save model after training

    def save_model(self, filename="dqn_model.pth"):
        """Save the trained model for each agent."""
        for agent in self.agents:
            model_file_path = os.path.join(self.results_dir, f'{filename}_{agent}.pth')
            torch.save(self.networks[agent].state_dict(), model_file_path)

    def load_model(self, filename="dqn_model.pth"):
        """Load the trained model for each agent."""
        for agent in self.agents:
            model_file_path = os.path.join(self.results_dir, f'{filename}_{agent}.pth')
            self.networks[agent].load_state_dict(torch.load(model_file_path))
            self.networks[agent].eval()

    def plot_rewards_and_losses(self, rewards_save_path="results/avg_rewards.png",
                                losses_save_path="results/avg_losses.png"):
        """
        Plot the average episode rewards and losses over training.
        """
        # Plot rewards
        plt.figure(figsize=(10, 5))
        plt.plot(self.avg_episode_rewards, label="Average Episode Rewards")
        plt.title("Average Episode Rewards During Training")
        plt.xlabel("Episode")
        plt.ylabel("Average Reward")
        plt.legend()
        plt.grid(True)
        plt.savefig(rewards_save_path)
        # plt.show()

        # Plot losses
        plt.figure(figsize=(10, 5))
        plt.plot(self.avg_episode_losses, label="Average Episode Losses")
        plt.title("Average Episode Losses During Training")
        plt.xlabel("Episode")
        plt.ylabel("Average Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig(losses_save_path)
        # plt.show()
