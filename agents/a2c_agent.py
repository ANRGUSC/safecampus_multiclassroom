import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

# Set seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# --------------------- Centralized Actor–Critic Network ---------------------
class CentralizedActorCriticNetwork(nn.Module):
    def __init__(self, num_agents, state_dim, action_space_size, hidden_dim=128, hidden_layers=3):
        """
        This network takes the concatenated joint state of all agents as input and outputs:
          - Policy logits for each agent (shape: [batch, num_agents, action_space_size])
          - A single scalar value estimate V(s) for the joint state.
        """
        super(CentralizedActorCriticNetwork, self).__init__()
        self.num_agents = num_agents
        input_dim = num_agents * state_dim

        layers = []
        prev_dim = input_dim
        for _ in range(hidden_layers):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        self.shared_layers = nn.Sequential(*layers)

        # Policy heads: one for each agent
        self.policy_heads = nn.ModuleList([nn.Linear(prev_dim, action_space_size) for _ in range(num_agents)])
        # Global value head
        self.value_head = nn.Linear(prev_dim, 1)

    def forward(self, joint_state):
        """
        joint_state: Tensor of shape (batch, num_agents * state_dim)
        Returns:
            policy_logits: Tensor of shape (batch, num_agents, action_space_size)
            value: Tensor of shape (batch, 1)
        """
        features = self.shared_layers(joint_state)
        policy_logits = []
        for head in self.policy_heads:
            policy_logits.append(head(features))
        policy_logits = torch.stack(policy_logits, dim=1)
        value = self.value_head(features)
        return policy_logits, value


# --------------------- Vanilla Actor–Critic Agent (Potential Game A2C) ---------------------
class VanillaActorCriticAgent:
    def __init__(self, agents, state_dim, action_space_size, reward_mix_alpha=1.0,
                 learning_rate=0.0001, gamma=0.99, clip_grad=1.0, entropy_coef=0.95):
        """
        Vanilla Actor–Critic agent with a centralized critic, adapted for a Markov potential game.
        In this setting, the critic estimates a global potential function (e.g. global reward or social welfare),
        and all agents use the same TD error computed from the potential.
        The mixed reward is computed as:
           r^mix = reward_mix_alpha * (global reward) + (1 - reward_mix_alpha) * (local reward)
        When reward_mix_alpha is 1 the agents are fully cooperative; when 0, they are purely selfish.
        An entropy bonus is optionally added for exploration.
        """
        self.agents = agents
        self.num_agents = len(agents)
        self.state_dim = state_dim
        self.action_space_size = action_space_size
        self.reward_mix_alpha = reward_mix_alpha
        self.gamma = gamma
        self.clip_grad = clip_grad
        self.entropy_coef = entropy_coef

        self.network = CentralizedActorCriticNetwork(self.num_agents, state_dim, action_space_size,
                                                     hidden_dim=128, hidden_layers=3)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=20, threshold=0.01, verbose=True
        )

        self.episode_rewards = {agent: [] for agent in agents}
        self.global_rewards = []
        self.unique_states = {agent: set() for agent in agents}
        self.action_values = None
        self.gae_lambda = 0.01

    def select_action(self, agent, state, joint_state=None):
        """
        Select an action for the given agent by sampling from a normalized softmax distribution
        computed over the policy logits. This normalization (mean subtraction) helps smooth
        the logits and improve exploration.
        - In cooperative mode (reward_mix_alpha > 0), the full joint state is used.
        - In selfish mode (reward_mix_alpha == 0), only the agent's local state is used.
        """
        agent_idx = self.agents.index(agent)

        # Determine input based on mode.
        if self.reward_mix_alpha > 0:
            if joint_state is None:
                raise ValueError("Joint state must be provided in cooperative mode.")
            state_tensor = torch.FloatTensor(joint_state).unsqueeze(0)
        else:
            full_input = torch.zeros(1, self.num_agents * self.state_dim)
            local_state_tensor = torch.FloatTensor(state).unsqueeze(0)
            start = agent_idx * self.state_dim
            end = start + self.state_dim
            full_input[0, start:end] = local_state_tensor
            state_tensor = full_input

        with torch.no_grad():
            policy_logits, _ = self.network(state_tensor)
        logits = policy_logits[0, agent_idx]

        # Normalize logits by subtracting the mean.
        # normalized_logits = logits - torch.mean(logits)
        # Compute softmax probabilities.
        probs = torch.softmax(logits, dim=0)

        # Sample from the categorical distribution.
        dist = torch.distributions.Categorical(probs=probs)
        action = dist.sample().item()

        return action

    def update_network(self, joint_state, actions, rewards, next_joint_state, done):
        """
        Per-step update of the network.
        Computes the effective (mixed) reward and uses it as the TD target (since this is a finite MDP),
        then updates both the actor and critic networks.
        """
        # Convert inputs to tensors.
        joint_state_tensor = torch.FloatTensor(joint_state).unsqueeze(0)
        next_joint_state_tensor = torch.FloatTensor(next_joint_state).unsqueeze(0)
        rewards_tensor = torch.FloatTensor(rewards)  # shape: (num_agents,)
        done_tensor = torch.tensor(float(done))

        # Critic forward pass (centralized).
        _, value = self.network(joint_state_tensor)  # shape: (1, 1)
        with torch.no_grad():
            _, next_value = self.network(next_joint_state_tensor)  # shape: (1, 1)

        # Compute global reward as the average over agents.
        global_reward = torch.mean(rewards_tensor)
        # Mixed reward for each agent: preserves the original usage of reward_mix_alpha.
        mixed_rewards = self.reward_mix_alpha * global_reward + (1 - self.reward_mix_alpha) * rewards_tensor
        # Effective reward: average over mixed rewards.
        effective_reward = torch.mean(mixed_rewards)

        # For a finite MDP, use the effective reward directly as the TD target.
        td_target = effective_reward
        td_error = td_target - value

        # Critic loss (MSE on TD error).
        critic_loss = torch.mean((value - td_target.detach()) ** 2)

        # Actor loss.
        policy_loss = 0.0
        for i in range(self.num_agents):
            # Build the appropriate input for the agent.
            if self.reward_mix_alpha > 0:
                input_tensor = torch.FloatTensor(joint_state).unsqueeze(0)
            else:
                local_input = torch.zeros(1, self.num_agents * self.state_dim)
                local_state = joint_state[i * self.state_dim:(i + 1) * self.state_dim]
                start = i * self.state_dim
                end = start + self.state_dim
                local_input[0, start:end] = torch.FloatTensor(local_state).unsqueeze(0)
                input_tensor = local_input

            logits, _ = self.network(input_tensor)
            agent_logits = logits[0, i]  # shape: (action_space_size,)
            dist = torch.distributions.Categorical(logits=agent_logits)
            log_prob = dist.log_prob(torch.tensor(actions[i]))
            entropy = dist.entropy()

            # In cooperative mode, impose a penalty if the chosen action's logit is below the average.
            if self.reward_mix_alpha > 0:
                avg_logit = torch.mean(agent_logits)
                penalty = (1 - self.reward_mix_alpha) * torch.relu(avg_logit - agent_logits[actions[i]])
            else:
                penalty = 0.0

            policy_loss += -log_prob * (td_error.detach() - penalty) - self.entropy_coef * entropy

        policy_loss = policy_loss / self.num_agents
        total_loss = policy_loss + critic_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=self.clip_grad)
        self.optimizer.step()

        return {
            'policy_loss': policy_loss.item(),
            'critic_loss': critic_loss.item(),
            'td_error': td_error.item()
        }

    def train(self, env, max_steps=30, total_episodes=100):
        """
        Training loop with per-step network updates.
        For each episode, the environment is reset, and the network is updated at each time step.
        """
        pbar = tqdm(total=total_episodes, desc="Training Progress", leave=True)
        if hasattr(env, 'action_levels'):
            self.action_values = {agent: env.action_levels[i] for i, agent in enumerate(env.agents)}
        episode_rewards_history = []
        best_avg_reward = float('-inf')

        for episode in range(total_episodes):
            states = env.reset()  # dict: agent -> state
            total_rewards = {agent: 0 for agent in self.agents}
            global_reward = 0

            for step in range(max_steps):
                joint_state = np.concatenate([states[agent] for agent in self.agents])
                actions = {}
                for agent in self.agents:
                    if self.reward_mix_alpha > 0:
                        actions[agent] = self.select_action(agent, states[agent], joint_state=joint_state)
                    else:
                        actions[agent] = self.select_action(agent, states[agent])
                next_states, rewards, dones, _ = env.step(actions)
                next_joint_state = np.concatenate([next_states[agent] for agent in self.agents])
                ordered_actions = [actions[agent] for agent in self.agents]
                ordered_rewards = [rewards[agent] for agent in self.agents]

                for agent in self.agents:
                    total_rewards[agent] += rewards[agent]
                    self.unique_states[agent].add(tuple(states[agent]))
                global_reward += sum(rewards.values())

                # Update network per step.
                loss_info = self.update_network(joint_state, ordered_actions, ordered_rewards, next_joint_state,
                                                all(dones.values()))
                states = next_states
                if all(dones.values()):
                    break

            for agent in self.agents:
                self.episode_rewards[agent].append(total_rewards[agent])
            episode_total_reward = sum(total_rewards.values())
            episode_rewards_history.append(episode_total_reward)
            self.global_rewards.append(global_reward)
            window_size = min(20, len(episode_rewards_history))
            avg_reward = sum(episode_rewards_history[-window_size:]) / window_size
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
            pbar.set_description(f"Episode {episode + 1} | Avg Reward: {avg_reward:.2f}")
            pbar.update(1)
        pbar.close()

        save_path = f"results/avg_rewards_vanillaAC_alpha_{self.reward_mix_alpha}_gamma_{env.gamma}.png"
        self.plot_rewards(save_path=save_path)
        for agent in self.agents:
            print(f"Agent {agent} visited {len(self.unique_states[agent])} unique states.")

    def plot_rewards(self, save_path="results/avg_rewards_vanillaAC.png"):
        plt.figure(figsize=(10, 5))
        for agent, rewards in self.episode_rewards.items():
            plt.plot(rewards, label=f"{agent} Rewards")
        plt.plot(self.global_rewards, label="Global Rewards", linestyle='--')
        plt.title("Episode Rewards During Training (Vanilla Actor-Critic)")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.legend()
        plt.grid(True)
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        plt.savefig(save_path)

    def evaluate(self, env, max_steps=52):
        states = env.reset()
        evaluation_data = {'steps': [], 'agents': {}}
        for agent in self.agents:
            evaluation_data['agents'][agent] = {
                'rewards': [],
                'actions': [],
                'infected': [],
                'allowed_students': [],
                'cross_class_infections': []
            }
        for step in range(max_steps):
            joint_state = np.concatenate([states[agent] for agent in self.agents])
            actions = {}
            for agent in self.agents:
                if self.reward_mix_alpha > 0:
                    actions[agent] = self.select_action(agent, states[agent], joint_state=joint_state)
                else:
                    actions[agent] = self.select_action(agent, states[agent])
                evaluation_data['agents'][agent]['actions'].append(actions[agent])
                agent_index = env.agents.index(agent)
                allowed_vals = env.action_levels[agent_index]
                evaluation_data['agents'][agent]['allowed_students'].append(allowed_vals[actions[agent]])
                evaluation_data['agents'][agent]['infected'].append(states[agent][0])
            next_states, rewards, dones, _ = env.step(actions)
            for agent in self.agents:
                evaluation_data['agents'][agent]['rewards'].append(rewards[agent])
            # Compute cross-class infections.
            allowed_students_all = [evaluation_data['agents'][ag]['allowed_students'][-1] for ag in self.agents]
            current_infected_all = [states[ag][0] for ag in self.agents]
            for i, agent in enumerate(self.agents):
                cross_infections = 0.0
                count = 0
                for j in range(len(current_infected_all)):
                    if i != j:
                        if allowed_students_all[j] > 0:
                            cross_infections += (current_infected_all[j] / allowed_students_all[j])
                            count += 1
                if count > 0:
                    cross_infections = (cross_infections / count) * allowed_students_all[i]
                else:
                    cross_infections = 0.0
                evaluation_data['agents'][agent]['cross_class_infections'].append(cross_infections)
            evaluation_data['steps'].append(step)
            states = next_states
            if all(dones.values()):
                break

        total_rewards = {agent: sum(evaluation_data['agents'][agent]['rewards']) for agent in self.agents}
        print("Total rewards per agent:", total_rewards)
        evaluation_data['total_rewards'] = total_rewards
        return evaluation_data