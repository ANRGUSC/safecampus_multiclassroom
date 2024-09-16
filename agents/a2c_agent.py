import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

# Define the neural network architecture for the Actor and Critic
class ActorCriticNetwork(nn.Module):
    def __init__(self, input_dim, action_space_size):
        super(ActorCriticNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 16)
        self.fc_actor = nn.Linear(16, action_space_size)
        self.fc_critic = nn.Linear(16, 1)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        action_probs = torch.softmax(self.fc_actor(x), dim=-1)  # Output action probabilities
        value = self.fc_critic(x)  # Output state value (critic)
        return action_probs, value


class A2CAgent:
    def __init__(self, agents, state_dim, action_space_size, learning_rate=0.00001, discount_factor=0.99):
        self.agents = agents
        self.state_dim = state_dim
        self.action_space_size = action_space_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.entropy_coef = 0.03

        # Initialize Actor-Critic networks for each agent
        self.actor_critic_networks = {agent: ActorCriticNetwork(state_dim, action_space_size) for agent in agents}
        self.optimizers = {agent: optim.Adam(self.actor_critic_networks[agent].parameters(), lr=learning_rate) for agent in agents}

        # Track rewards over episodes for visualization
        self.episode_rewards = {agent: [] for agent in agents}

    def select_action(self, agent, state):
        """Select an action based on the actor's policy."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)  # Convert state to tensor
        action_probs, _ = self.actor_critic_networks[agent](state_tensor)  # Get action probabilities from actor
        action = torch.multinomial(action_probs, num_samples=1).item()  # Sample action from probabilities
        return action  # Ensure this returns an integer

    def update(self, agent, trajectory):
        """Update both actor and critic using the A2C update rule."""
        states, actions, rewards, next_states, dones = zip(*trajectory)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        # Compute discounted rewards (returns)
        returns = self.compute_returns(rewards, dones)

        # Get action probabilities and state values for all states
        action_probs, state_values = self.actor_critic_networks[agent](states)
        _, next_state_values = self.actor_critic_networks[agent](next_states)

        # Get the probabilities of the taken actions
        action_log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1)).squeeze())

        # Compute advantages
        advantages = returns - state_values.squeeze()

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Compute loss for both actor and critic
        actor_loss = -(action_log_probs * advantages).mean()
        critic_loss = 0.5 * advantages.pow(2).mean()

        # Entropy regularization
        action_entropy = -(action_probs * torch.log(action_probs + 1e-8)).sum(dim=-1)
        entropy_loss = -self.entropy_coef * action_entropy.mean()

        loss = actor_loss + critic_loss + entropy_loss

        # Backpropagate the loss
        self.optimizers[agent].zero_grad()
        loss.backward()
        self.optimizers[agent].step()

    def compute_returns(self, rewards, dones):
        """Compute discounted returns."""
        returns = []
        R = 0
        for r, done in zip(reversed(rewards), reversed(dones)):
            R = r + R * (1 - done)
            returns.insert(0, R)
        return torch.FloatTensor(returns)

    def train(self, env, max_steps=30):
        pbar = tqdm(total=1000, desc="Training Progress", leave=True)
        for episode in range(1000):
            states = env.reset()
            total_rewards = {agent: 0 for agent in self.agents}
            trajectories = {agent: [] for agent in self.agents}

            for step in range(max_steps):
                actions = {}
                for agent in self.agents:
                    action = self.select_action(agent, states[agent])
                    actions[agent] = action

                next_states, rewards, dones, _ = env.step(actions)

                # Store trajectory for each agent
                for agent in self.agents:
                    action = actions[agent]
                    reward = rewards[agent]
                    next_state = next_states[agent]
                    done = dones[agent]
                    total_rewards[agent] += reward
                    trajectories[agent].append((states[agent], action, reward, next_state, done))

                states = next_states
                if all(dones.values()):
                    break

            # Update the networks for each agent using their trajectories
            for agent in self.agents:
                self.update(agent, trajectories[agent])

            # Log the total rewards for each agent at the end of the episode
            for agent in self.agents:
                self.episode_rewards[agent].append(total_rewards[agent])

            pbar.update(1)

        pbar.close()
        rewards_path = f"results/avg_rewards_a2c_{env.gamma}.png"
        self.plot_rewards(save_path=rewards_path)

    def plot_rewards(self, save_path="results/avg_rewards_a2c.png"):
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
