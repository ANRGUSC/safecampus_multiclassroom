import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


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


class PPOAgent:
    def __init__(self, agents, state_dim, action_space_size, learning_rate=0.000001, discount_factor=0.99, clip_epsilon=0.2, gae_lambda=0.95):
        self.agents = agents
        self.state_dim = state_dim
        self.action_space_size = action_space_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.clip_epsilon = clip_epsilon
        self.gae_lambda = gae_lambda
        self.entropy_coef = 0.7
        self.state_visits = {agent: {} for agent in agents}

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

    def compute_returns_and_advantages(self, rewards, values, next_values, dones):
        """Compute discounted returns and advantages."""
        returns = []
        advantages = []
        epsilon = 1e-8  # Small value to avoid inf, nan, or negative values

        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                returns.insert(0, rewards[i])
            else:
                returns.insert(0, rewards[i])
            advantages.insert(0, returns[0] - values[i])

        # Ensure no inf, nan, or negative values in returns and advantages
        returns = torch.FloatTensor(returns).clamp(min=epsilon)
        advantages = torch.FloatTensor(advantages).clamp(min=epsilon)

        return returns, advantages

    def update(self, agent, trajectory, old_log_probs):
        """Update both actor and critic using the PPO update rule."""
        states, actions, rewards, next_states, dones = zip(*trajectory)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        # Compute combined rewards
        combined_rewards = []
        for i in range(len(rewards)):
            local_reward = rewards[i].item()
            global_reward = torch.mean(rewards).item()
            combined_reward = global_reward + local_reward
            combined_rewards.append(combined_reward)

        combined_rewards = torch.FloatTensor(combined_rewards)

        # Get the old log probabilities and values for the states
        action_probs, state_values = self.actor_critic_networks[agent](states)
        _, next_state_values = self.actor_critic_networks[agent](next_states)
        action_log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1)).squeeze())

        # Compute returns and advantages using GAE
        returns, advantages = self.compute_returns_and_advantages(combined_rewards, state_values.squeeze(), next_state_values.squeeze(), dones)

        # Clip the policy ratio and compute the clipped actor loss
        policy_ratio = torch.exp(action_log_probs - old_log_probs)
        clipped_ratio = torch.clamp(policy_ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
        actor_loss = -torch.min(policy_ratio * advantages, clipped_ratio * advantages).mean()

        # Compute critic loss (MSE between state values and returns)
        critic_loss = (returns - state_values).pow(2).mean()

        # Entropy regularization
        action_entropy = -(action_probs * torch.log(action_probs)).sum(dim=-1)
        entropy_loss = -self.entropy_coef * action_entropy.mean()

        # Total loss
        loss = actor_loss + critic_loss + entropy_loss

        # Backpropagate the loss
        self.optimizers[agent].zero_grad()
        loss.backward()
        self.optimizers[agent].step()

    def train(self, env, max_steps=30, epochs=4):
        pbar = tqdm(total=1000, desc="Training Progress", leave=True)
        for episode in range(1000):
            states = env.reset()
            total_rewards = {agent: 0 for agent in self.agents}
            trajectories = {agent: [] for agent in self.agents}
            old_log_probs = {agent: [] for agent in self.agents}

            for step in range(max_steps):
                actions = {}
                for agent in self.agents:
                    action = self.select_action(agent, states[agent])
                    actions[agent] = action

                    # Track state visitations
                    state_tuple = tuple(states[agent])
                    if state_tuple not in self.state_visits[agent]:
                        self.state_visits[agent][state_tuple] = 0
                    self.state_visits[agent][state_tuple] += 1

                next_states, rewards, dones, _ = env.step(actions)

                # Store trajectory for each agent
                for agent in self.agents:
                    action = actions[agent]
                    reward = rewards[agent]
                    next_state = next_states[agent]
                    done = dones[agent]
                    total_rewards[agent] += reward

                    # Calculate and store old log probabilities
                    state_tensor = torch.FloatTensor(states[agent]).unsqueeze(0)
                    action_probs, _ = self.actor_critic_networks[agent](state_tensor)
                    action_log_prob = torch.log(action_probs.squeeze()[action])
                    old_log_probs[agent].append(action_log_prob.item())

                    trajectories[agent].append((states[agent], action, reward, next_state, done))

                states = next_states
                if all(dones.values()):
                    break

            # Update the networks for each agent using their trajectories and old log probs
            for agent in self.agents:
                for _ in range(epochs):  # Perform multiple epochs of update
                    self.update(agent, trajectories[agent], torch.FloatTensor(old_log_probs[agent]))

            # Log the total rewards for each agent at the end of the episode
            for agent in self.agents:
                self.episode_rewards[agent].append(total_rewards[agent])

            pbar.update(1)

        pbar.close()
        rewards_path = f"results/avg_rewards_ppo_{env.gamma}.png"
        self.plot_rewards(save_path=rewards_path)
        self.calculate_state_visit_percentage()


    def calculate_state_visit_percentage(self):
        """Calculate and print the percentage of states visited for each agent."""
        for agent, visits in self.state_visits.items():
            total_states = len(visits)
            visited_states = sum(1 for count in visits.values() if count > 0)
            visit_percentage = (visited_states / total_states) * 100
            print(f"Agent {agent} visited {visit_percentage:.2f}% of states.")

    def plot_rewards(self, save_path="results/avg_rewards_ppo.png"):
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