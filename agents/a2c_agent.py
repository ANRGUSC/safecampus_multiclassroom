import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)



# Define the Actor network (separate for each agent)
class ActorNetwork(nn.Module):
    def __init__(self, input_dim, action_space_size):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 16)
        self.fc_actor = nn.Linear(16, action_space_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        action_probs = torch.softmax(self.fc_actor(x), dim=-1)  # Output action probabilities
        return action_probs


# Define the centralized Critic network (shared for all agents)
class CriticNetwork(nn.Module):
    def __init__(self, global_state_dim):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(global_state_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc_critic = nn.Linear(32, 1)

    def forward(self, global_state):
        x = torch.relu(self.fc1(global_state))
        x = torch.relu(self.fc2(x))
        value = self.fc_critic(x)  # Output global value
        return value


class CentralizedA2CAgent:
    def __init__(self, agents, state_dim, global_state_dim, action_space_size, critic_learning_rate=0.00001,
                 discount_factor=0.99, actor_learning_rate=0.000001):
        self.agents = agents
        self.state_dim = state_dim
        self.global_state_dim = global_state_dim
        self.action_space_size = action_space_size
        self.learning_rate = critic_learning_rate
        self.discount_factor = discount_factor
        self.entropy_coef = 0.8
        self.max_grad_norm = 0.5  # For gradient clipping

        # Initialize decentralized Actor networks for each agent
        self.actor_networks = {agent: ActorNetwork(state_dim, action_space_size) for agent in agents}
        self.actor_optimizers = {agent: optim.Adam(self.actor_networks[agent].parameters(), lr=actor_learning_rate) for agent
                                 in agents}

        # Initialize centralized Critic network
        self.critic_network = CriticNetwork(global_state_dim)
        self.critic_optimizer = optim.Adam(self.critic_network.parameters(), lr=self.learning_rate)

        # Track rewards over episodes for visualization
        self.episode_rewards = {agent: [] for agent in agents}

        # Track state visitations
        self.state_visits = {agent: {} for agent in agents}

        self.actor_losses = []
        self.critic_losses = []

    def select_action(self, agent, state):
        """Select an action based on the actor's policy."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)  # Convert state to tensor
        action_probs = self.actor_networks[agent](state_tensor)  # Get action probabilities from actor
        action = torch.multinomial(action_probs, num_samples=1).item()  # Sample action from probabilities
        return action

    def update(self, global_states, trajectories):
        global_states = torch.FloatTensor(global_states)
        global_values = self.critic_network(global_states).squeeze()

        total_actor_loss = 0
        total_critic_loss = 0
        all_returns = []

        # Calculate global rewards
        global_rewards = torch.zeros(len(global_states))
        for agent, trajectory in trajectories.items():
            _, _, rewards, _, _ = zip(*trajectory)
            global_rewards += torch.FloatTensor(rewards)

        for agent, trajectory in trajectories.items():
            if len(trajectory) == 0:
                continue

            states, actions, local_rewards, next_states, _ = zip(*trajectory)

            states = torch.FloatTensor(states)
            actions = torch.LongTensor(actions)
            local_rewards = torch.FloatTensor(local_rewards)

            # Combine local and global rewards
            combined_rewards =  local_rewards * global_rewards[:len(local_rewards)]

            # Compute returns using combined rewards
            returns = self.compute_returns(combined_rewards)
            all_returns.extend(returns)

            # Ensure global_values matches the length of returns for this agent
            agent_global_values = global_values[:len(returns)]

            advantages = returns - agent_global_values.detach()

            # Actor update
            action_logits = self.actor_networks[agent](states)
            action_probs = F.softmax(action_logits, dim=-1)
            action_log_probs = F.log_softmax(action_logits, dim=-1)

            selected_action_log_probs = action_log_probs.gather(1, actions.unsqueeze(1)).squeeze()

            actor_loss = -(selected_action_log_probs * advantages).mean()

            # Entropy regularization
            entropy = -(action_probs * action_log_probs).sum(dim=-1).mean()
            actor_loss -= self.entropy_coef * entropy

            self.actor_optimizers[agent].zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor_networks[agent].parameters(), self.max_grad_norm)
            self.actor_optimizers[agent].step()

            total_actor_loss += actor_loss.item()

            # print(f"Agent: {agent}, Actor Loss: {actor_loss.item():.4f}, Entropy: {entropy.item():.4f}")
            # print(f"Advantages - Mean: {advantages.mean().item():.4f}, Std: {advantages.std().item():.4f}")

        # Centralized critic update
        all_returns = torch.FloatTensor(all_returns)
        critic_loss = F.smooth_l1_loss(global_values, all_returns[:len(global_values)])

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_network.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()

        total_critic_loss = critic_loss.item()

        return total_actor_loss / len(self.agents), total_critic_loss

    def train(self, env, max_steps=30, num_episodes=1000):
        pbar = tqdm(total=num_episodes, desc="Training Progress", leave=True)
        for episode in range(num_episodes):
            states = env.reset()
            total_rewards = {agent: 0 for agent in self.agents}
            trajectories = {agent: [] for agent in self.agents}
            global_states = []

            for step in range(max_steps):
                actions = {}
                current_global_state = []

                for agent in self.agents:
                    action = self.select_action(agent, states[agent])
                    actions[agent] = action
                    current_global_state.extend(states[agent])

                    # Track state visitations
                    state_tuple = tuple(states[agent])
                    if state_tuple not in self.state_visits[agent]:
                        self.state_visits[agent][state_tuple] = 0
                    self.state_visits[agent][state_tuple] += 1

                global_states.append(current_global_state)
                next_states, rewards, dones, _ = env.step(actions)

                # Store trajectory for each agent
                for agent in self.agents:
                    total_rewards[agent] += rewards[agent]
                    trajectories[agent].append(
                        (states[agent], actions[agent], rewards[agent], next_states[agent], dones[agent]))

                states = next_states
                if all(dones.values()):
                    break

            # Update the networks with the collected trajectories
            actor_loss, critic_loss = self.update(global_states, trajectories)

            # Store the losses
            self.actor_losses.append(actor_loss)
            self.critic_losses.append(critic_loss)

            # Log the total rewards for each agent at the end of the episode
            for agent in self.agents:
                self.episode_rewards[agent].append(total_rewards[agent])

            # Update progress bar with current losses and average reward
            avg_reward = sum(total_rewards.values()) / len(self.agents)
            pbar.set_postfix({
                'Actor Loss': f'{actor_loss:.4f}',
                'Critic Loss': f'{critic_loss:.4f}',
                'Avg Reward': f'{avg_reward:.2f}'
            })
            pbar.update(1)

        pbar.close()
        self.plot_rewards_and_losses(save_path=f"results/a2c_centralized_training_{env.gamma}.png")
        self.calculate_state_visit_percentage()

    def compute_returns(self, rewards):
        """
        Compute undiscounted returns.
        This is equivalent to the cumulative sum of rewards from each time step to the end of the episode.
        """
        returns = torch.zeros_like(rewards)
        cumulative_reward = 0
        for t in reversed(range(len(rewards))):
            cumulative_reward += rewards[t]
            returns[t] = cumulative_reward
        return returns
    def plot_rewards_and_losses(self, save_path="results/a2c_centralized_training.png"):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

        # Plot rewards
        for agent, rewards in self.episode_rewards.items():
            ax1.plot(rewards, label=f"{agent} Rewards")
        ax1.set_ylabel("Total Reward")
        ax1.set_title("Episode Rewards During Training")
        ax1.legend()
        ax1.grid(True)

        # Plot losses
        ax2.plot(self.actor_losses, label="Actor Loss")
        ax2.plot(self.critic_losses, label="Critic Loss")
        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Loss")
        ax2.set_title("Actor and Critic Losses During Training")
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        plt.savefig(save_path)
        plt.close()

    def calculate_state_visit_percentage(self):
        for agent, visits in self.state_visits.items():
            total_states = len(visits)
            visited_states = sum(1 for count in visits.values() if count > 0)
            visit_percentage = (visited_states / total_states) * 100
            print(f"Agent {agent} visited {visited_states:.2f}% of states.")

    def evaluate(self, env, max_steps=52, save_path=None):
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
                episode_community_risk[agent].append(env.shared_community_risk[min(env.current_week, env.max_weeks - 1)])
            states = next_states
            if all(dones.values()):
                break

        self.plot_infected_allowed_and_risk_over_time(episode_infected, episode_allowed, episode_community_risk,
                                                      save_path=save_path)

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