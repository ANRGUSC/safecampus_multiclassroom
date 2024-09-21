import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
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
        self.fc1 = nn.Linear(global_state_dim, 32)
        self.fc_critic = nn.Linear(32, 1)

    def forward(self, global_state):
        x = torch.relu(self.fc1(global_state))
        value = self.fc_critic(x)  # Output global value
        return value


class CentralizedA2CAgent:
    def __init__(self, agents, state_dim, global_state_dim, action_space_size, learning_rate=0.0000001,
                 discount_factor=0.99,epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.995):
        self.agents = agents
        self.state_dim = state_dim
        self.global_state_dim = global_state_dim
        self.action_space_size = action_space_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.entropy_coef = 0.5

        # Initialize decentralized Actor networks for each agent
        self.actor_networks = {agent: ActorNetwork(state_dim, action_space_size) for agent in agents}
        self.actor_optimizers = {agent: optim.Adam(self.actor_networks[agent].parameters(), lr=learning_rate) for agent
                                 in agents}

        # Initialize decentralized Critic networks for each agent
        self.critic_networks = {agent: CriticNetwork(state_dim) for agent in agents}
        self.critic_optimizers = {agent: optim.Adam(self.critic_networks[agent].parameters(), lr=learning_rate) for agent
                                  in agents}

        # Track rewards over episodes for visualization
        self.episode_rewards = {agent: [] for agent in agents}

        # Track state visitations
        self.state_visits = {agent: {} for agent in agents}

    def select_action(self, agent, state):
        """Select an action based on the actor's policy."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)  # Convert state to tensor
        action_probs = self.actor_networks[agent](state_tensor)  # Get action probabilities from actor
        action = torch.multinomial(action_probs, num_samples=1).item()  # Sample action from probabilities
        return action

    def update(self, global_state, trajectories):
        """Update both decentralized actors and critics using composite rewards."""
        avg_returns_list = []  # Collect all returns to compute the average
        for agent, trajectory in trajectories.items():
            states, actions, rewards, next_states, _ = zip(*trajectory)

            states = torch.FloatTensor(states)
            actions = torch.LongTensor(actions)
            rewards = torch.FloatTensor(rewards)
            next_states = torch.FloatTensor(next_states)

            # Compute composite rewards
            combined_rewards = []
            for i in range(len(rewards)):
                state = states[i].numpy()
                next_state = next_states[i].numpy()
                local_reward = rewards[i].item()
                global_reward = torch.mean(rewards).item()

                # Combine global and local rewards
                combined_reward = global_reward + local_reward
                combined_rewards.append(combined_reward)

            combined_rewards = torch.FloatTensor(combined_rewards)

            # Compute returns (discounted rewards) for each agent
            returns = self.compute_returns(combined_rewards)
            avg_returns_list.append(returns)  # Append agent's returns

            advantages = returns - self.critic_networks[agent](states).squeeze()  # Decentralized critic's value is used

            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Get action probabilities for the taken actions
            action_probs = self.actor_networks[agent](states)
            action_log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1)).squeeze())

            # Compute actor loss
            actor_loss = -(action_log_probs * advantages.detach()).mean()

            # Compute entropy regularization
            action_entropy = -(action_probs * torch.log(action_probs + 1e-8)).sum(dim=-1)
            entropy_loss = -self.entropy_coef * action_entropy.mean()

            # Actor loss backpropagation
            actor_loss += entropy_loss
            self.actor_optimizers[agent].zero_grad()
            actor_loss.backward()
            self.actor_optimizers[agent].step()

            # Update the critic after the actor
            critic_loss = 0.5 * (self.critic_networks[agent](states).squeeze() - returns).pow(2).mean()

            self.critic_optimizers[agent].zero_grad()
            critic_loss.backward()
            self.critic_optimizers[agent].step()

    def compute_returns(self, rewards):
        """Compute discounted returns."""
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + R
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
                global_state = []

                for agent in self.agents:
                    action = self.select_action(agent, states[agent])
                    actions[agent] = action
                    global_state.extend(states[agent])  # Collect states from all agents for global state

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
                    trajectories[agent].append((states[agent], action, reward, next_state, done))

                states = next_states
                if all(dones.values()):
                    break

            # Update the networks with the collected trajectories
            self.update(global_state, trajectories)

            # Log the total rewards for each agent at the end of the episode
            for agent in self.agents:
                self.episode_rewards[agent].append(total_rewards[agent])

            pbar.update(1)

        pbar.close()
        rewards_path = f"results/avg_rewards_a2c_centralized_{env.gamma}.png"
        self.plot_rewards(save_path=rewards_path)
        self.calculate_state_visit_percentage()

    def plot_rewards(self, save_path="results/avg_rewards_a2c_centralized.png"):
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

    def calculate_state_visit_percentage(self):
        """Calculate and print the percentage of states visited for each agent."""
        for agent, visits in self.state_visits.items():
            total_states = len(visits)
            visited_states = sum(1 for count in visits.values() if count > 0)
            visit_percentage = (visited_states / total_states) * 100
            print(f"Agent {agent} visited {visit_percentage:.2f}% of states.")