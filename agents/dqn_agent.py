import numpy as np
from itertools import permutations
import torch
import torch.nn as nn
import torch.optim as optim
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import math

# --------------------- Centralized Q-Network (Critic) Definition ---------------------

class CentralizedDQNetwork(nn.Module):
    def __init__(self, num_agents, state_dim, action_space_size, hidden_dim=16, hidden_layers=2):
        """
        This network takes the concatenated joint state of all agents as input
        and produces a separate Q-value vector (of length action_space_size) for each agent.
        """
        super(CentralizedDQNetwork, self).__init__()
        self.num_agents = num_agents
        input_dim = num_agents * state_dim
        layers = []
        prev_dim = input_dim
        for _ in range(hidden_layers):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        self.shared_layers = nn.Sequential(*layers)
        # Create separate output heads for each agent.
        self.agent_heads = nn.ModuleList([nn.Linear(prev_dim, action_space_size) for _ in range(num_agents)])


    def forward(self, joint_state):
        """
        joint_state: Tensor of shape (batch, num_agents * state_dim)
        Returns: Tensor of shape (batch, num_agents, action_space_size)
        """
        shared_features = self.shared_layers(joint_state)
        outputs = []
        for head in self.agent_heads:
            outputs.append(head(shared_features))
        return torch.stack(outputs, dim=1)

# --------------------- CTDE DQNAgent with Multi-Step TD Returns and Penalty ---------------------
class DQNAgent:
    def __init__(self, agents, state_dim, action_space_size, reward_mix_alpha=0.5,
                 learning_rate=0.03, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.0001,
                 gamma=0.99, use_ctde=True, n_step=2, seed=None, hidden_dim=32, hidden_layers=3):
        """
        CTDE: Centralized Training with Decentralized Execution.
        """
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        self.seed = seed
        self.agents = agents
        self.num_agents = len(agents)
        self.state_dim = state_dim
        self.action_space_size = action_space_size
        self.reward_mix_alpha = reward_mix_alpha
        self.learning_rate = learning_rate
        self.epsilon = epsilon  # initial exploration probability
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.gamma = gamma
        self.use_ctde = use_ctde
        self.n_step = n_step
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers

        # Centralized Q-network (critic) used during training
        self.network = CentralizedDQNetwork(self.num_agents, state_dim, action_space_size, hidden_dim=hidden_dim, hidden_layers=hidden_layers)
        self.target_network = CentralizedDQNetwork(self.num_agents, state_dim, action_space_size, hidden_dim=hidden_dim, hidden_layers=hidden_layers)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        self.target_network.load_state_dict(self.network.state_dict())

        self.episode_rewards = {agent: [] for agent in agents}
        self.global_rewards = []
        self.action_values = None  # to be set in train() using env.action_levels

        # Transition buffer for multi-step updates.
        # Each item is a tuple: (joint_state, actions, rewards, next_joint_state, done)
        self.transition_buffer = []

    def select_local_action(self, agent, state):
        """
        Decentralized execution using ε-greedy + argmax.
        """
        idx = self.agents.index(agent)
        # build a joint‐state vector but only fill this agent’s slot:
        inp = torch.zeros(1, self.num_agents * self.state_dim)
        st = torch.FloatTensor(state).unsqueeze(0)
        start = idx * self.state_dim
        inp[0, start:start + self.state_dim] = st

        if random.random() < self.epsilon:
            return random.randrange(self.action_space_size)

        with torch.no_grad():
            q_all = self.network(inp)  # shape (1, N, A)
            q_i = q_all[0, idx]  # shape (A,)
        return int(q_i.argmax().item())

    def select_joint_actions(self, joint_state):
        """
        CTDE joint action selection with pure ε-greedy + argmax per agent.
        """
        js = torch.FloatTensor(joint_state).unsqueeze(0)  # (1, N * S)
        with torch.no_grad():
            q_all = self.network(js)[0]  # (N, A)

        acts = []
        for i in range(self.num_agents):
            if random.random() < self.epsilon:
                acts.append(random.randrange(self.action_space_size))
            else:
                acts.append(int(q_all[i].argmax().item()))
        return acts

    def select_local_action_eval(self, agent, state):
        """
        Greedy evaluation (ε=0).
        """
        return self.select_local_action(agent, state)

    def select_action(self, agent, state):
        """
        For visualization (decentralized execution), use the local state.
        The 'joint_state' parameter is ignored.
        """
        return self.select_local_action_eval(agent, state)

    def _accumulate_multi_step_return(self, rewards):
        """
        Compute the multi-step return from a list of rewards.
        Since this is a finite MDP, no discounting is applied.
        """
        multi_step_return = 0.0
        for r in rewards:
            multi_step_return += r
        return multi_step_return

    def train_td_double_ci_rewards(self,
                                      env,
                                      max_steps: int = 30,
                                      save_dir: str = None,
                                      return_rewards: bool = False):
        """
        One‐step TD training with Double‐DQN action selection.
        If return_rewards=True, returns a list of per‐episode global rewards.
        Otherwise saves the reward plot into save_dir (or skips saving if save_dir is None).
        """
        total_episodes = 1000

        # reset logs
        self.global_rewards.clear()
        self.episode_rewards = {a: [] for a in self.agents}
        reward_history = [] if return_rewards else None

        for episode in range(1, total_episodes + 1):
            states = env.reset()
            global_reward = 0.0
            total_rewards = {a: 0.0 for a in self.agents}

            # decay ε
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

            for step in range(max_steps):
                # 1) select decentralized actions
                joint_state = np.concatenate([states[a] for a in self.agents])
                actions = {a: self.select_local_action(a, states[a])
                           for a in self.agents}

                # 2) step
                next_states, rewards, dones, _ = env.step(actions)

                # 3) accumulate rewards
                for a in self.agents:
                    total_rewards[a] += rewards[a]
                global_reward += sum(rewards.values())

                # 4) build tensors
                js = torch.FloatTensor(joint_state).unsqueeze(0)
                next_js = torch.FloatTensor(
                    np.concatenate([next_states[a] for a in self.agents])
                ).unsqueeze(0)

                # 5) Q‐values
                q_all = self.network(js)[0]
                q_next_online = self.network(next_js)[0]
                q_next_target = self.target_network(next_js)[0].detach()

                # 6) Q(s,a) for taken actions
                q_taken = torch.stack([
                    q_all[i, actions[a]] for i, a in enumerate(self.agents)
                ])

                # 7) Double‐DQN TD targets
                td_targets = torch.stack([
                    torch.tensor(rewards[a], dtype=torch.float32)
                    + q_next_target[i, q_next_online[i].argmax().item()]
                    for i, a in enumerate(self.agents)
                ]).to(q_taken.dtype)

                # 8) loss & backprop
                loss = nn.MSELoss()(q_taken, td_targets)
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
                self.optimizer.step()

                states = next_states
                if all(dones.values()):
                    break

            # record episode rewards
            for a in self.agents:
                self.episode_rewards[a].append(total_rewards[a])
            self.global_rewards.append(global_reward)

            if return_rewards:
                reward_history.append(global_reward)

        # return list of global rewards if asked
        if return_rewards:
            return reward_history

    def train_td_double_hyper_loss(self,
                                      env,
                                      max_steps: int = 30,
                                      save_dir: str = None,
                                      return_loss: bool = False):
        """
        One‐step TD training with Double‐DQN action selection.
        If return_loss=True, returns a list of per‐episode global loss.
        Otherwise saves the reward plot into save_dir (or skips saving if save_dir is None).
        """
        total_episodes = 1000

        # reset logs
        self.global_rewards.clear()
        self.episode_rewards = {a: [] for a in self.agents}
        loss_history = [] if return_loss else None

        for episode in range(1, total_episodes + 1):
            states = env.reset()
            total_rewards = {a: 0.0 for a in self.agents}
            episode_loss = 0.0
            loss_steps = 0

            # decay ε
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

            for step in range(max_steps):
                joint_state = np.concatenate([states[a] for a in self.agents])
                actions = {a: self.select_local_action(a, states[a])
                           for a in self.agents}

                next_states, rewards, dones, _ = env.step(actions)

                for a in self.agents:
                    total_rewards[a] += rewards[a]

                js = torch.FloatTensor(joint_state).unsqueeze(0)
                next_js = torch.FloatTensor(
                    np.concatenate([next_states[a] for a in self.agents])
                ).unsqueeze(0)

                q_all = self.network(js)[0]
                q_next_online = self.network(next_js)[0]
                q_next_target = self.target_network(next_js)[0].detach()

                q_taken = torch.stack([
                    q_all[i, actions[a]] for i, a in enumerate(self.agents)
                ])

                td_targets = torch.stack([
                    torch.tensor(rewards[a], dtype=torch.float32)
                    + q_next_target[i, q_next_online[i].argmax().item()]
                    for i, a in enumerate(self.agents)
                ]).to(q_taken.dtype)

                loss = nn.MSELoss()(q_taken, td_targets)
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
                self.optimizer.step()

                episode_loss += loss.item()
                loss_steps += 1

                states = next_states
                if all(dones.values()):
                    break

            # record average loss for the episode
            if return_loss:
                avg_loss = episode_loss / max(loss_steps, 1)
                loss_history.append(avg_loss)

            for a in self.agents:
                self.episode_rewards[a].append(total_rewards[a])
            self.global_rewards.append(sum(total_rewards.values()))

        if return_loss:
            return loss_history

    def train_td_double(self, env, max_steps=30, save_dir: str = None):
        """
        One-step TD training with Double-DQN action selection.
        After each episode we record & plot global_rewards as before.
        """
        total_episodes = 1000
        convergence_window = 20

        # convergence threshold (not used to early-stop here, but for reference)
        max_per_agent_step = self.gamma * env.total_students
        max_global_episode = len(self.agents) * max_per_agent_step * max_steps
        target_avg_return = 0.9 * max_global_episode

        # reset logs
        self.global_rewards.clear()
        self.episode_rewards = {a: [] for a in self.agents}

        for episode in range(1, total_episodes + 1):
            states = env.reset()
            global_reward = 0.0
            total_rewards = {a: 0.0 for a in self.agents}

            # decay ε
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

            for step in range(max_steps):
                # 1) select decentralized actions
                joint_state = np.concatenate([states[a] for a in self.agents])
                actions = {
                    a: self.select_local_action(a, states[a])
                    for a in self.agents
                }

                # 2) step
                next_states, rewards, dones, _ = env.step(actions)

                # 3) accumulate rewards
                for a in self.agents:
                    total_rewards[a] += rewards[a]
                global_reward += sum(rewards.values())

                # 4) build tensors
                js = torch.FloatTensor(joint_state).unsqueeze(0)  # [1, N·state_dim]
                next_js = torch.FloatTensor(
                    np.concatenate([next_states[a] for a in self.agents])
                ).unsqueeze(0)  # [1, N·state_dim]

                # 5) Q-values
                q_all = self.network(js)[0]  # online net: [N_agents, A]
                q_next_online = self.network(next_js)[0]  # online net @ next state
                q_next_target = self.target_network(next_js)[0] \
                    .detach()  # target net @ next state

                # 6) Q(s,a) for taken actions
                q_taken = torch.stack([
                    q_all[i, actions[a]] for i, a in enumerate(self.agents)
                ])  # shape [N_agents]

                # 7) Double-DQN TD targets:
                #    use argmax from online net, but value from target net
                td_targets = torch.stack([
                    torch.tensor(rewards[a], dtype=torch.float32) + q_next_target[i, q_next_online[i].argmax().item()]
                    for i, a in enumerate(self.agents)
                ]).to(q_taken.dtype)

                # 8) loss & backprop
                loss = nn.MSELoss()(q_taken, td_targets)
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
                self.optimizer.step()

                states = next_states
                if all(dones.values()):
                    break

            # record episode rewards
            for a in self.agents:
                self.episode_rewards[a].append(total_rewards[a])
            self.global_rewards.append(global_reward)

        # save the learning curve exactly as before
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"avg_rewards_CTDE_td_double_gamma_{env.gamma}.png")
        self.plot_rewards(save_path)

    def train_mc(self, env, max_steps=30, save_dir: str = None):
        """
        Training loop: single‐step TD updates at every env.step().
        """
        # convergence criteria:
        total_episodes = 1000
        max_per_agent_step = self.gamma * env.total_students
        max_global_episode = 2 * max_per_agent_step * max_steps
        target_avg_return = 0.9 * max_global_episode

        # pbar = tqdm(total=total_episodes, desc="Training Progress", leave=True)
        self.global_rewards.clear()
        self.episode_rewards = {a: [] for a in self.agents}

        convergence_episodes = None

        for episode in range(total_episodes):
            total_rewards = {agent: 0.0 for agent in self.agents}
            global_reward = 0.0
            states = env.reset()
            self.transition_buffer.clear()

            # 1) run one episode, store transitions
            for step in range(max_steps):
                joint_state = np.concatenate([states[a] for a in self.agents])
                if self.reward_mix_alpha > 0:
                    acts = self.select_joint_actions(joint_state)
                    actions = {a: acts[i] for i, a in enumerate(self.agents)}
                else:
                    actions = {a: self.select_local_action(a, states[a]) for a in self.agents}

                next_states, rewards, dones, _ = env.step(actions)
                # accumulate
                for a in self.agents:
                    total_rewards[a] += rewards[a]
                global_reward += sum(rewards.values())

                # buffer: (joint_state, list_of_actions, list_of_rewards)
                self.transition_buffer.append((
                    joint_state,
                    [actions[a] for a in self.agents],
                    [rewards[a] for a in self.agents]
                ))

                states = next_states
                if all(dones.values()):
                    break

            # 2) compute Monte‐Carlo returns G_t (team‐sum or team‐avg)
            G = 0.0
            returns = []
            # iterate backwards
            for (_, _, reward_list) in reversed(self.transition_buffer):
                team_r = sum(reward_list)/self.num_agents # or sum(...) / self.num_agents for average‐team
                G = team_r  # accumulate returns
                returns.insert(0, G)

            # 3) one batched gradient step over all t
            total_loss = 0.0
            for (joint_state, actions, _), G_t in zip(self.transition_buffer, returns):
                js = torch.FloatTensor(joint_state).unsqueeze(0)  # [1, N·S]
                q_all = self.network(js)[0]  # [N_agents, A]
                # pick each agent’s taken‐action Q
                q_taken = torch.stack([q_all[i, actions[i]]
                                       for i in range(self.num_agents)])  # [N_agents]
                # target: team‐average return
                G_team = G_t
                target = torch.full_like(q_taken, G_team)
                total_loss += nn.MSELoss()(q_taken, target)

            total_loss = total_loss / len(self.transition_buffer)
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
            self.optimizer.step()

            # record and decay ε
            for a in self.agents:
                self.episode_rewards[a].append(total_rewards[a])
            self.global_rewards.append(global_reward)
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)


        # save the learning curve exactly as before
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"avg_rewards_CTDE_mc_gamma_{env.gamma}.png")
        self.plot_rewards(save_path)
    def train_mc_hyper(self, env, max_steps=30, save_dir: str = None, return_loss=False):
        """
        Training loop: single-step TD updates at every env.step().
        If return_loss=True, returns a list of per-episode losses.
        """
        total_episodes = 1000
        max_per_agent_step = self.gamma * env.total_students
        max_global_episode = 2 * max_per_agent_step * max_steps
        target_avg_return = 0.9 * max_global_episode

        self.global_rewards.clear()
        self.episode_rewards = {a: [] for a in self.agents}
        loss_history = [] if return_loss else None

        for episode in range(total_episodes):
            total_rewards = {agent: 0.0 for agent in self.agents}
            global_reward = 0.0
            states = env.reset()
            self.transition_buffer.clear()

            # collect transitions
            for step in range(max_steps):
                joint_state = np.concatenate([states[a] for a in self.agents])
                if self.reward_mix_alpha > 0:
                    acts = self.select_joint_actions(joint_state)
                    actions = {a: acts[i] for i, a in enumerate(self.agents)}
                else:
                    actions = {
                        a: self.select_local_action(a, states[a])
                        for a in self.agents
                    }

                next_states, rewards, dones, _ = env.step(actions)
                for a in self.agents:
                    total_rewards[a] += rewards[a]
                global_reward += sum(rewards.values())
                self.transition_buffer.append(
                    (joint_state,
                     [actions[a] for a in self.agents],
                     [rewards[a] for a in self.agents])
                )
                states = next_states
                if all(dones.values()):
                    break

            # compute MC returns (team-average)
            G = 0.0
            returns = []
            for (_, _, reward_list) in reversed(self.transition_buffer):
                team_r = sum(reward_list) / self.num_agents
                G = team_r
                returns.insert(0, G)

            # one gradient step over the whole episode
            total_loss = 0.0
            for (joint_state, actions, _), G_t in zip(self.transition_buffer, returns):
                js = torch.FloatTensor(joint_state).unsqueeze(0)
                q_all = self.network(js)[0]
                q_taken = torch.stack([
                    q_all[i, actions[i]] for i in range(self.num_agents)
                ])
                target = torch.full_like(q_taken, G_t)
                total_loss += nn.MSELoss()(q_taken, target)

            total_loss = total_loss / len(self.transition_buffer)
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
            self.optimizer.step()

            if return_loss:
                loss_history.append(total_loss.item())

            # logging
            for a in self.agents:
                self.episode_rewards[a].append(total_rewards[a])
            self.global_rewards.append(global_reward)
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

            # (optional) early‐stop on convergence…

        if return_loss:
            return loss_history

    def train_mc_ci_rewards(self, env, max_steps=30, save_dir: str = None, return_rewards: bool = False):
        """
        Training loop using Monte Carlo updates over episodes,
        with reward tracking and optional return of reward history.
        """
        total_episodes = 1000

        # compute convergence threshold
        max_per_agent_step = self.gamma * env.total_students
        max_global_episode = len(self.agents) * max_per_agent_step * max_steps

        # reset logs
        self.global_rewards.clear()
        self.episode_rewards = {a: [] for a in self.agents}
        reward_history = [] if return_rewards else None

        for episode in range(1, total_episodes + 1):
            total_rewards = {agent: 0.0 for agent in self.agents}
            global_reward = 0.0
            states = env.reset()

            # decay epsilon at start of episode
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

            self.transition_buffer.clear()

            # collect episode transitions
            for step in range(max_steps):
                joint_state = np.concatenate([states[a] for a in self.agents])
                # Select actions (no reward_mix_alpha logic here)
                actions = {
                    a: self.select_local_action(a, states[a])
                    for a in self.agents
                }

                next_states, rewards, dones, _ = env.step(actions)

                # accumulate rewards
                for a in self.agents:
                    total_rewards[a] += rewards[a]
                global_reward += sum(rewards.values())

                self.transition_buffer.append(
                    (joint_state,
                     [actions[a] for a in self.agents],
                     [rewards[a] for a in self.agents])
                )

                states = next_states
                if all(dones.values()):
                    break

            # compute Monte Carlo returns (team-average)
            G = 0.0
            returns = []
            for (_, _, reward_list) in reversed(self.transition_buffer):
                team_r = sum(reward_list) / self.num_agents
                G = team_r
                returns.insert(0, G)

            # one gradient step over whole episode
            total_loss = 0.0
            for (joint_state, actions, _), G_t in zip(self.transition_buffer, returns):
                js = torch.FloatTensor(joint_state).unsqueeze(0)
                q_all = self.network(js)[0]
                q_taken = torch.stack([
                    q_all[i, actions[i]] for i in range(self.num_agents)
                ])
                target = torch.full_like(q_taken, G_t)
                total_loss += nn.MSELoss()(q_taken, target)

            total_loss = total_loss / len(self.transition_buffer)
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
            self.optimizer.step()

            # logging episode rewards
            for a in self.agents:
                self.episode_rewards[a].append(total_rewards[a])
            self.global_rewards.append(global_reward)

            if return_rewards:
                reward_history.append(global_reward)

        if return_rewards:
            return reward_history



    def train_centralized_td_double_hyper_loss(self,
                                               env,
                                               max_steps: int = 30,
                                               total_episodes: int = 1000,
                                               save_dir: str = None,
                                               return_loss: bool = False):
        """
        Centralized TD training using Double-DQN style targets with per-step loss tracking.
        Uses explicit centralized epsilon-greedy joint action selection (no select_joint_actions call).
        Returns list of average losses per episode if return_loss=True.
        """
        self.global_rewards.clear()
        self.episode_rewards = {a: [] for a in self.agents}
        loss_history = [] if return_loss else None

        for ep in range(1, total_episodes + 1):
            states = env.reset()
            ep_reward = 0.0
            episode_loss = 0.0
            loss_steps = 0

            # Decay epsilon at the start of episode
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

            for t in range(max_steps):
                joint_state = np.concatenate([states[a] for a in self.agents])

                # Explicit centralized epsilon-greedy joint action selection
                if random.random() < self.epsilon:
                    joint_action = [random.randrange(self.action_space_size) for _ in self.agents]
                else:
                    with torch.no_grad():
                        q_vals = self.network(torch.FloatTensor(joint_state).unsqueeze(0))[0]
                        joint_action = [int(q_vals[i].argmax()) for i in range(len(self.agents))]

                actions = {a: joint_action[i] for i, a in enumerate(self.agents)}

                next_states, rewards, dones, _ = env.step(actions)
                # Average team reward
                r_team = sum(rewards.values()) / len(self.agents)
                ep_reward += r_team

                # Q values for current joint state and taken joint actions
                q_vals = self.network(torch.FloatTensor(joint_state).unsqueeze(0))[0]
                q_joint = sum(q_vals[i, joint_action[i]] for i in range(len(self.agents)))

                # Next state Q values for Double-DQN target
                next_joint_state = np.concatenate([next_states[a] for a in self.agents])
                with torch.no_grad():
                    q_next_online = self.network(torch.FloatTensor(next_joint_state).unsqueeze(0))[0]
                    q_next_target = self.target_network(torch.FloatTensor(next_joint_state).unsqueeze(0))[0]

                best_next = sum(
                    q_next_target[i, q_next_online[i].argmax().item()] for i in range(len(self.agents))) / len(
                    self.agents)

                # TD target (no discounting applied here)
                y = r_team + best_next

                # Compute loss: MSE between predicted Q and target
                loss = nn.MSELoss()(q_joint, torch.tensor(y))

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
                self.optimizer.step()

                episode_loss += loss.item()
                loss_steps += 1

                states = next_states
                if all(dones.values()):
                    break

            if return_loss:
                avg_loss = episode_loss / max(loss_steps, 1)
                loss_history.append(avg_loss)

            self.global_rewards.append(ep_reward)
            for a in self.agents:
                self.episode_rewards[a].append(ep_reward / len(self.agents))

            self.target_network.load_state_dict(self.network.state_dict())

        if return_loss:
            return loss_history

    def train_centralized_td_double(self,
                                    env,
                                    max_steps: int = 30,
                                    total_episodes: int = 1000,
                                    save_dir: str = None,
                                    return_rewards: bool = False):
        """
        Centralized TD training using Double-DQN style targets.
        Uses explicit centralized epsilon-greedy joint action selection for training and execution.
        If return_rewards=True, returns per-episode total rewards.
        Saves reward learning curve plot if save_dir specified.
        """
        self.global_rewards.clear()
        self.episode_rewards = {a: [] for a in self.agents}
        reward_history = [] if return_rewards else None

        for ep in range(1, total_episodes + 1):
            states = env.reset()
            ep_reward = 0.0

            # Decay epsilon at the start of episode
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
            if self.epsilon <= self.min_epsilon:
                break

            for t in range(max_steps):
                joint_state = np.concatenate([states[a] for a in self.agents])

                # Explicit centralized epsilon-greedy joint action selection
                if random.random() < self.epsilon:
                    joint_action = [random.randrange(self.action_space_size) for _ in self.agents]
                else:
                    with torch.no_grad():
                        q_heads = self.network(torch.FloatTensor(joint_state).unsqueeze(0))[0]
                        joint_action = [int(q_heads[i].argmax()) for i in range(len(self.agents))]

                actions = {a: joint_action[i] for i, a in enumerate(self.agents)}

                next_states, rewards, dones, _ = env.step(actions)
                # Average team reward
                r_team = sum(rewards.values()) / len(self.agents)
                ep_reward += r_team

                # Q values for current joint state and taken joint actions
                q_vals = self.network(torch.FloatTensor(joint_state).unsqueeze(0))[0]
                q_joint = sum(q_vals[i, joint_action[i]] for i in range(len(self.agents)))

                # Next state Q values for Double-DQN target
                next_joint_state = np.concatenate([next_states[a] for a in self.agents])
                with torch.no_grad():
                    q_next_online = self.network(torch.FloatTensor(next_joint_state).unsqueeze(0))[0]
                    q_next_target = self.target_network(torch.FloatTensor(next_joint_state).unsqueeze(0))[0]
                    best_next = sum(
                        q_next_target[i, q_next_online[i].argmax().item()] for i in range(len(self.agents))) / len(
                        self.agents)

                # TD target without discounting
                y = r_team + best_next

                # Compute loss
                loss = nn.MSELoss()(q_joint, torch.tensor(y))

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
                self.optimizer.step()

                states = next_states
                if all(dones.values()):
                    break

            self.global_rewards.append(ep_reward)
            for a in self.agents:
                self.episode_rewards[a].append(ep_reward / len(self.agents))

            self.target_network.load_state_dict(self.network.state_dict())

            if return_rewards:
                reward_history.append(ep_reward)

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"avg_rewards_centralized_td_double_gamma_{env.gamma}.png")
            self.plot_rewards(save_path)

        if return_rewards:
            return reward_history

    def train_centralized_td_double_ci_rewards(self,
                                                  env,
                                                  max_steps: int = 30,
                                                  total_episodes: int = 1000,
                                                  save_dir: str = None,
                                                  return_rewards: bool = False):
        """
        Centralized TD training using Double-DQN style targets.
        If return_rewards=True, returns per-episode total rewards.
        Saves learning curve plot to save_dir if specified.
        """
        self.global_rewards.clear()
        self.episode_rewards = {a: [] for a in self.agents}
        reward_history = [] if return_rewards else None

        for ep in range(1, total_episodes + 1):
            states = env.reset()
            ep_reward = 0.0

            # Decay epsilon at the start of episode
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
            if self.epsilon <= self.min_epsilon:
                break

            for t in range(max_steps):
                joint_state = np.concatenate([states[a] for a in self.agents])

                # Explicit centralized epsilon-greedy joint action selection
                if random.random() < self.epsilon:
                    joint_act = [random.randrange(self.action_space_size) for _ in self.agents]
                else:
                    with torch.no_grad():
                        q_heads = self.network(torch.FloatTensor(joint_state).unsqueeze(0))[0]
                        joint_act = [int(q_heads[i].argmax()) for i in range(len(self.agents))]

                actions = {a: joint_act[i] for i, a in enumerate(self.agents)}

                next_states, rewards, dones, _ = env.step(actions)
                # Average team reward
                r_team = sum(rewards.values()) / len(self.agents)
                ep_reward += r_team

                # Double-DQN TD targets
                q_heads = self.network(torch.FloatTensor(joint_state).unsqueeze(0))[0]
                q_taken = torch.stack([q_heads[i, joint_act[i]] for i in range(len(self.agents))])

                js_next = torch.FloatTensor(np.concatenate([next_states[a] for a in self.agents])).unsqueeze(0)
                with torch.no_grad():
                    q_next_online = self.network(js_next)[0]
                    q_next_target = self.target_network(js_next)[0]

                td_targets = torch.stack([
                    torch.tensor(rewards[a], dtype=q_taken.dtype) +
                    q_next_target[i, q_next_online[i].argmax().item()]
                    for i, a in enumerate(self.agents)
                ])

                loss = nn.MSELoss()(q_taken, td_targets)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
                self.optimizer.step()

                states = next_states
                if all(dones.values()):
                    break

            self.global_rewards.append(ep_reward)
            for a in self.agents:
                self.episode_rewards[a].append(ep_reward / len(self.agents))

            self.target_network.load_state_dict(self.network.state_dict())

            if return_rewards:
                reward_history.append(ep_reward)

        if return_rewards:
            return reward_history



    def train_centralized_mc(self, env,
                             max_steps: int = 30,
                             total_episodes: int = 1000,
                             save_dir: str = None,
                             return_rewards: bool = False):
        """
        Centralized Monte Carlo training.
        If return_rewards=True, returns per-episode total rewards.
        Saves learning curve plot to save_dir if specified.
        """
        self.global_rewards.clear()
        self.episode_rewards = {a: [] for a in self.agents}
        reward_history = [] if return_rewards else None

        for ep in range(1, total_episodes + 1):
            states = env.reset()
            ep_reward = 0.0
            self.transition_buffer.clear()

            # decay epsilon at start of episode
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

            # Collect one full episode
            for t in range(max_steps):
                js = np.concatenate([states[a] for a in self.agents])
                if random.random() < self.epsilon:
                    joint_act = [random.randrange(self.action_space_size)
                                 for _ in self.agents]
                else:
                    with torch.no_grad():
                        q_heads = self.network(torch.FloatTensor(js).unsqueeze(0))[0]
                        joint_act = [int(q_heads[i].argmax()) for i in range(self.num_agents)]

                next_states, rewards, dones, _ = env.step(
                    {a: joint_act[i] for i, a in enumerate(self.agents)}
                )
                r_team = sum(rewards.values())
                ep_reward += r_team
                self.transition_buffer.append((js, joint_act, r_team))
                states = next_states
                if all(dones.values()):
                    break

            # Compute MC returns (team-average)
            returns, G = [], 0.0
            for (_, _, r) in reversed(self.transition_buffer):
                G = r  # no bootstrapping
                returns.insert(0, G)

            # One gradient step over full episode
            total_loss = 0.0
            for (js, joint_act, _), G_t in zip(self.transition_buffer, returns):
                q_heads = self.network(torch.FloatTensor(js).unsqueeze(0))[0]
                q_taken = torch.stack([
                    q_heads[i, joint_act[i]] for i in range(self.num_agents)
                ])
                target = torch.full_like(q_taken, G_t)
                total_loss += nn.MSELoss()(q_taken, target)

            total_loss = total_loss / len(self.transition_buffer)
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
            self.optimizer.step()

            self.global_rewards.append(ep_reward)
            for a in self.agents:
                self.episode_rewards[a].append(ep_reward / self.num_agents)

            self.target_network.load_state_dict(self.network.state_dict())

            if return_rewards:
                reward_history.append(ep_reward)

        # save learning curve plot if directory provided
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"avg_rewards_centralized_mc_gamma_{env.gamma}.png")
        self.plot_rewards(save_path)

    def train_centralized_mc_loss(self, env,
                             max_steps: int = 30,
                             total_episodes: int = 1000,
                             save_dir: str = None,
                             return_loss: bool = False):
        """
        Centralized Monte Carlo training.
        If return_loss=True, returns list of average losses per episode.
        Saves learning curve plot to save_dir if specified.
        """
        self.global_rewards.clear()
        self.episode_rewards = {a: [] for a in self.agents}
        loss_history = [] if return_loss else None

        for ep in range(1, total_episodes + 1):
            states = env.reset()
            ep_reward = 0.0
            self.transition_buffer.clear()

            # Decay epsilon at start of episode
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

            # Collect one full episode
            for t in range(max_steps):
                js = np.concatenate([states[a] for a in self.agents])
                if random.random() < self.epsilon:
                    joint_act = [random.randrange(self.action_space_size)
                                 for _ in self.agents]
                else:
                    with torch.no_grad():
                        q_heads = self.network(torch.FloatTensor(js).unsqueeze(0))[0]
                        joint_act = [int(q_heads[i].argmax()) for i in range(self.num_agents)]

                next_states, rewards, dones, _ = env.step(
                    {a: joint_act[i] for i, a in enumerate(self.agents)}
                )
                r_team = sum(rewards.values())
                ep_reward += r_team
                self.transition_buffer.append((js, joint_act, r_team))
                states = next_states
                if all(dones.values()):
                    break

            # Compute MC returns (team-average)
            returns, G = [], 0.0
            for (_, _, r) in reversed(self.transition_buffer):
                G = r  # no bootstrapping
                returns.insert(0, G)

            # One gradient step over full episode
            total_loss = 0.0
            for (js, joint_act, _), G_t in zip(self.transition_buffer, returns):
                q_heads = self.network(torch.FloatTensor(js).unsqueeze(0))[0]
                q_taken = torch.stack([
                    q_heads[i, joint_act[i]] for i in range(self.num_agents)
                ])
                target = torch.full_like(q_taken, G_t)
                total_loss += nn.MSELoss()(q_taken, target)

            avg_loss = total_loss / max(len(self.transition_buffer), 1)
            self.optimizer.zero_grad()
            avg_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
            self.optimizer.step()

            self.global_rewards.append(ep_reward)
            for a in self.agents:
                self.episode_rewards[a].append(ep_reward / self.num_agents)

            self.target_network.load_state_dict(self.network.state_dict())

            if return_loss:
                loss_history.append(avg_loss.item())

        if return_loss:
            return loss_history

    from itertools import permutations
    import numpy as np
    import torch

    def evaluate(self, env, max_steps=52, centralized=False, evaluate_cross_policy=False,
                 evaluate_single_policy_all=False):
        """
        Evaluate the agent's policy.
        Modes:
          - Normal evaluation (default)
          - Cross-policy evaluation (all permutations excluding identity)
          - Single-policy-for-all-classrooms evaluation (each agent's policy applied to all classrooms)

        The centralized flag selects between centralized vs CTDE evaluation modes.
        """
        if evaluate_cross_policy:
            if centralized:
                return self.evaluate_cross_policy_centralized(env, max_steps)
            else:
                return self.evaluate_cross_policy_ctde(env, max_steps)

        if evaluate_single_policy_all:
            if centralized:
                return self.evaluate_single_policy_all_centralized(env, max_steps)
            else:
                return self.evaluate_single_policy_all_ctde(env, max_steps)

        # Normal evaluation (existing evaluate code)
        if centralized:
            return self.evaluate_centralized(env, max_steps)
        else:
            return self.evaluate_ctde(env, max_steps)

    def evaluate_ctde(self, env, max_steps=52):
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

            if self.reward_mix_alpha == 0:
                # Selfish (decentralized) evaluation
                actions = {
                    agent: self.select_local_action_eval(agent, states[agent])
                    for agent in self.agents
                }
            else:
                # Cooperative evaluation using exhaustive joint action search (masking)
                with torch.no_grad():
                    joint_state_tensor = torch.FloatTensor(joint_state).unsqueeze(0)  # shape (1, N * state_dim)
                    q_values = self.network(joint_state_tensor).squeeze(0)  # shape (N, A)

                    from itertools import product
                    action_space = range(self.action_space_size)
                    all_joint_actions = list(product(action_space, repeat=self.num_agents))

                    best_total_q = float('-inf')
                    best_joint_action = None

                    for joint_action in all_joint_actions:
                        total_q = sum(q_values[i, a] for i, a in enumerate(joint_action))
                        if total_q > best_total_q:
                            best_total_q = total_q
                            best_joint_action = joint_action

                    actions = {
                        agent: best_joint_action[i]
                        for i, agent in enumerate(self.agents)
                    }

            for agent in self.agents:
                evaluation_data['agents'][agent]['actions'].append(actions[agent])
                agent_index = env.agents.index(agent)
                allowed_vals = env.action_levels[agent_index]
                evaluation_data['agents'][agent]['allowed_students'].append(allowed_vals[actions[agent]])
                evaluation_data['agents'][agent]['infected'].append(states[agent][0])

            next_states, rewards, dones, _ = env.step(actions)

            for agent in self.agents:
                evaluation_data['agents'][agent]['rewards'].append(rewards[agent])

            allowed_students_all = [evaluation_data['agents'][ag]['allowed_students'][-1] for ag in self.agents]
            current_infected_all = [states[ag][0] for ag in self.agents]

            for i, agent in enumerate(self.agents):
                cross_class_term = 0.0
                for j in range(len(current_infected_all)):
                    if i != j:
                        other_allowed = allowed_students_all[j]
                        other_infected = current_infected_all[j]
                        if other_allowed > 0:
                            cross_class_term += (other_infected / other_allowed)
                if len(self.agents) > 1:
                    cross_class_term = cross_class_term / (len(self.agents) - 1) * allowed_students_all[i]
                else:
                    cross_class_term = 0.0
                evaluation_data['agents'][agent]['cross_class_infections'].append(cross_class_term)

            evaluation_data['steps'].append(step)
            states = next_states

            if all(dones.values()):
                break

        total_rewards = {
            agent: sum(evaluation_data['agents'][agent]['rewards'])
            for agent in self.agents
        }
        evaluation_data['total_rewards'] = total_rewards
        return evaluation_data

    def evaluate_centralized(self, env, max_steps=52):
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
            joint_state = np.concatenate([states[a] for a in self.agents])

            with torch.no_grad():
                q_heads = self.network(
                    torch.FloatTensor(joint_state).unsqueeze(0)
                ).squeeze(0)  # shape (N_agents, A)
            joint_action = [int(q_heads[i].argmax())
                            for i in range(self.num_agents)]
            actions = {agent: joint_action[i]
                       for i, agent in enumerate(self.agents)}

            for agent in self.agents:
                evaluation_data['agents'][agent]['actions'].append(actions[agent])
                idx = env.agents.index(agent)
                lvl = env.action_levels[idx][actions[agent]]
                evaluation_data['agents'][agent]['allowed_students'].append(lvl)
                evaluation_data['agents'][agent]['infected'].append(states[agent][0])

            next_states, rewards, dones, _ = env.step(actions)

            for agent in self.agents:
                evaluation_data['agents'][agent]['rewards'].append(rewards[agent])

            last_allowed = [evaluation_data['agents'][a]['allowed_students'][-1]
                            for a in self.agents]
            current_infected = [states[a][0] for a in self.agents]
            for i, agent in enumerate(self.agents):
                term = 0.0
                for j in range(self.num_agents):
                    if i != j and last_allowed[j] > 0:
                        term += (current_infected[j] / last_allowed[j])
                if self.num_agents > 1:
                    term = term / (self.num_agents - 1) * last_allowed[i]
                evaluation_data['agents'][agent]['cross_class_infections'].append(term)

            evaluation_data['steps'].append(step)
            states = next_states
            if all(dones.values()):
                break

        total_rewards = {
            agent: sum(evaluation_data['agents'][agent]['rewards'])
            for agent in self.agents
        }
        evaluation_data['total_rewards'] = total_rewards
        return evaluation_data

    def evaluate_cross_policy_ctde(self, env, max_steps=52):
        from itertools import permutations

        agents = self.agents
        num_agents = len(agents)

        # Generate all permutations of policies assigned to classrooms
        # Exclude identity (where each agent uses its own policy)
        all_perms = list(permutations(agents))
        all_perms = [perm for perm in all_perms if any(perm[i] != agents[i] for i in range(num_agents))]

        all_results = {}

        for perm in all_perms:
            states = env.reset()
            dones = {agent: False for agent in agents}
            step = 0

            # Track cumulative rewards per classroom
            cum_rewards = {agent: 0.0 for agent in agents}

            while step < max_steps and not all(dones.values()):
                actions = {}
                for i, classroom_agent in enumerate(agents):
                    policy_agent = perm[i]
                    local_state = states[classroom_agent]
                    inp = torch.zeros(1, num_agents * self.state_dim)
                    start = agents.index(classroom_agent) * self.state_dim
                    inp[0, start:start + self.state_dim] = torch.FloatTensor(local_state)

                    with torch.no_grad():
                        q_all = self.network(inp)[0]
                        # Use policy_agent's Q-values to select action
                        policy_idx = agents.index(policy_agent)
                        q_vals_policy = q_all[policy_idx]
                        action = int(q_vals_policy.argmax().item())

                    actions[classroom_agent] = action

                next_states, rewards, dones, _ = env.step(actions)

                for agent in agents:
                    cum_rewards[agent] += rewards[agent]

                states = next_states
                step += 1

            # Save cumulative rewards for this permutation (policy assignment)
            all_results[perm] = cum_rewards

        # Compute mean reward per permutation
        mean_rewards_per_perm = {
            perm: np.mean(list(cum_rewards.values()))
            for perm, cum_rewards in all_results.items()
        }

        return {
            'permutation_rewards': all_results,
            'mean_rewards_per_permutation': mean_rewards_per_perm
        }

    def evaluate_cross_policy_centralized(self, env, max_steps=52):
        from itertools import permutations

        agents = self.agents
        num_agents = len(agents)

        all_perms = list(permutations(agents))
        all_perms = [perm for perm in all_perms if any(perm[i] != agents[i] for i in range(num_agents))]

        all_results = {}

        for perm in all_perms:
            states = env.reset()
            dones = {agent: False for agent in agents}
            step = 0

            cum_rewards = {agent: 0.0 for agent in agents}

            while step < max_steps and not all(dones.values()):
                joint_state = np.concatenate([states[a] for a in agents])
                with torch.no_grad():
                    q_values = self.network(torch.FloatTensor(joint_state).unsqueeze(0)).squeeze(0)

                # For each classroom, use the Q-values of the assigned policy agent
                actions = {}
                for i, classroom_agent in enumerate(agents):
                    policy_agent = perm[i]
                    policy_idx = agents.index(policy_agent)
                    action_for_policy = int(q_values[policy_idx].argmax().item())
                    actions[classroom_agent] = action_for_policy

                next_states, rewards, dones, _ = env.step(actions)

                for agent in agents:
                    cum_rewards[agent] += rewards[agent]

                states = next_states
                step += 1

            all_results[perm] = cum_rewards

        mean_rewards_per_perm = {
            perm: np.mean(list(cum_rewards.values()))
            for perm, cum_rewards in all_results.items()
        }

        return {
            'permutation_rewards': all_results,
            'mean_rewards_per_permutation': mean_rewards_per_perm
        }

    def evaluate_single_policy_all_ctde(self, env, max_steps=52):
        agents = self.agents
        num_agents = len(agents)

        # print(f"Evaluating single policy all classrooms for agents: {agents}")
        # print(f"Number of agents: {num_agents}")
        # print(f"State dimension per agent: {self.state_dim}")

        all_results = {}

        for policy_agent in agents:
            # print(f"\nUsing policy from: {policy_agent}")

            states = env.reset()
            dones = {agent: False for agent in agents}
            step = 0
            cum_rewards = {agent: 0.0 for agent in agents}

            while step < max_steps and not all(dones.values()):
                actions = {}
                for classroom_agent in agents:
                    local_state = states[classroom_agent]
                    inp = torch.zeros(1, num_agents * self.state_dim)
                    start = agents.index(classroom_agent) * self.state_dim
                    inp[0, start:start + self.state_dim] = torch.FloatTensor(local_state)

                    # print(f"Step {step}, Classroom agent: {classroom_agent}")
                    # print(f"Local state: {local_state}")
                    # print(f"Input tensor shape: {inp.shape}")
                    # print(f"Input tensor: {inp}")

                    with torch.no_grad():
                        q_all = self.network(inp)[0]
                        # print(f"Q-values shape: {q_all.shape}")

                        policy_idx = agents.index(policy_agent)
                        # print(f"Policy agent index: {policy_idx}")

                        q_vals_policy = q_all[policy_idx]
                        # print(f"Q-values for policy agent: {q_vals_policy}")

                        action = int(q_vals_policy.argmax().item())
                        # print(f"Selected action: {action}")

                    actions[classroom_agent] = action

                next_states, rewards, dones, _ = env.step(actions)

                # print(f"Actions taken: {actions}")
                # print(f"Rewards received: {rewards}")
                # print(f"Dones: {dones}")

                for agent in agents:
                    cum_rewards[agent] += rewards[agent]

                states = next_states
                step += 1

            # print(f"Cumulative rewards for policy {policy_agent}: {cum_rewards}")

            all_results[policy_agent] = cum_rewards

        # Average cumulative rewards across classrooms for each policy
        mean_rewards_per_policy = {
            pa: np.mean(list(cum_rewards.values()))
            for pa, cum_rewards in all_results.items()
        }

        # print(f"\nMean rewards per policy across classrooms: {mean_rewards_per_policy}")

        return {
            'policy_rewards': all_results,
            'mean_rewards_per_policy': mean_rewards_per_policy
        }

    def evaluate_single_policy_all_centralized(self, env, max_steps=52):
        agents = self.agents
        num_agents = len(agents)

        all_results = {}

        for policy_agent in agents:
            states = env.reset()
            dones = {agent: False for agent in agents}
            step = 0
            cum_rewards = {agent: 0.0 for agent in agents}

            while step < max_steps and not all(dones.values()):
                joint_state = np.concatenate([states[a] for a in agents])
                with torch.no_grad():
                    q_values = self.network(torch.FloatTensor(joint_state).unsqueeze(0)).squeeze(0)

                policy_idx = agents.index(policy_agent)
                action_for_policy = int(q_values[policy_idx].argmax().item())
                actions = {agent: action_for_policy for agent in agents}

                next_states, rewards, dones, _ = env.step(actions)

                for agent in agents:
                    cum_rewards[agent] += rewards[agent]

                states = next_states
                step += 1

            all_results[policy_agent] = cum_rewards

        mean_rewards_per_policy = {
            pa: np.mean(list(cum_rewards.values()))
            for pa, cum_rewards in all_results.items()
        }

        return {
            'policy_rewards': all_results,
            'mean_rewards_per_policy': mean_rewards_per_policy
        }

    # def evaluate(self, env, max_steps=52, centralized=False):
    #     """
    #     Evaluation of the agent's policy.
    #     If reward_mix_alpha > 0, uses centralized joint evaluation with greedy joint action selection.
    #     Otherwise, decentralized greedy evaluation per agent.
    #     """
    #     if centralized:
    #         return self.evaluate_centralized(env, max_steps)
    #     else:
    #         states = env.reset()
    #         evaluation_data = {'steps': [], 'agents': {}}
    #         for agent in self.agents:
    #             evaluation_data['agents'][agent] = {
    #                 'rewards': [],
    #                 'actions': [],
    #                 'infected': [],
    #                 'allowed_students': [],
    #                 'cross_class_infections': []
    #             }
    #
    #         for step in range(max_steps):
    #             joint_state = np.concatenate([states[agent] for agent in self.agents])
    #
    #             if self.reward_mix_alpha == 0:
    #                 # Selfish (decentralized) evaluation
    #                 actions = {
    #                     agent: self.select_local_action_eval(agent, states[agent])
    #                     for agent in self.agents
    #                 }
    #             else:
    #                 # Cooperative evaluation using exhaustive joint action search (masking)
    #                 with torch.no_grad():
    #                     joint_state_tensor = torch.FloatTensor(joint_state).unsqueeze(0)  # shape (1, N * state_dim)
    #                     q_values = self.network(joint_state_tensor).squeeze(0)  # shape (N, A)
    #
    #                     # Generate all possible joint actions (Cartesian product)
    #                     from itertools import product
    #                     action_space = range(self.action_space_size)
    #                     all_joint_actions = list(product(action_space, repeat=self.num_agents))
    #
    #                     best_total_q = float('-inf')
    #                     best_joint_action = None
    #
    #                     for joint_action in all_joint_actions:
    #                         total_q = sum(q_values[i, a] for i, a in enumerate(joint_action))
    #                         if total_q > best_total_q:
    #                             best_total_q = total_q
    #                             best_joint_action = joint_action
    #
    #                     actions = {
    #                         agent: best_joint_action[i]
    #                         for i, agent in enumerate(self.agents)
    #                     }
    #
    #             for agent in self.agents:
    #                 evaluation_data['agents'][agent]['actions'].append(actions[agent])
    #                 agent_index = env.agents.index(agent)
    #                 allowed_vals = env.action_levels[agent_index]
    #                 evaluation_data['agents'][agent]['allowed_students'].append(allowed_vals[actions[agent]])
    #                 evaluation_data['agents'][agent]['infected'].append(states[agent][0])
    #
    #             next_states, rewards, dones, _ = env.step(actions)
    #
    #             for agent in self.agents:
    #                 evaluation_data['agents'][agent]['rewards'].append(rewards[agent])
    #
    #             allowed_students_all = [evaluation_data['agents'][ag]['allowed_students'][-1] for ag in self.agents]
    #             current_infected_all = [states[ag][0] for ag in self.agents]
    #
    #             for i, agent in enumerate(self.agents):
    #                 cross_class_term = 0.0
    #                 for j in range(len(current_infected_all)):
    #                     if i != j:
    #                         other_allowed = allowed_students_all[j]
    #                         other_infected = current_infected_all[j]
    #                         if other_allowed > 0:
    #                             cross_class_term += (other_infected / other_allowed)
    #                 if len(self.agents) > 1:
    #                     cross_class_term = cross_class_term / (len(self.agents) - 1) * allowed_students_all[i]
    #                 else:
    #                     cross_class_term = 0.0
    #                 evaluation_data['agents'][agent]['cross_class_infections'].append(cross_class_term)
    #
    #             evaluation_data['steps'].append(step)
    #             states = next_states
    #
    #             if all(dones.values()):
    #                 break
    #
    #         total_rewards = {
    #             agent: sum(evaluation_data['agents'][agent]['rewards'])
    #             for agent in self.agents
    #         }
    #         # print("Total rewards per agent:", total_rewards)
    #         evaluation_data['total_rewards'] = total_rewards
    #         return evaluation_data

    def evaluate_centralized(self, env, max_steps=52):
        """
        Fully-centralized evaluation of the joint policy.
        Always uses greedy joint action selection (i.e. ε=0).
        Collects the exact same data dict structure as evaluate().
        """
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
            # 1) form joint_state
            joint_state = np.concatenate([states[a] for a in self.agents])

            # 2) greedy joint action selection
            with torch.no_grad():
                q_heads = self.network(
                    torch.FloatTensor(joint_state).unsqueeze(0)
                ).squeeze(0)  # shape (N_agents, A)
            joint_action = [int(q_heads[i].argmax())
                            for i in range(self.num_agents)]
            actions = {agent: joint_action[i]
                       for i, agent in enumerate(self.agents)}

            # 3) record per-agent pre-step data
            for agent in self.agents:
                evaluation_data['agents'][agent]['actions'].append(actions[agent])
                idx = env.agents.index(agent)
                lvl = env.action_levels[idx][actions[agent]]
                evaluation_data['agents'][agent]['allowed_students'].append(lvl)
                # assume state[0] == infected count
                evaluation_data['agents'][agent]['infected'].append(states[agent][0])

            # 4) step environment
            next_states, rewards, dones, _ = env.step(actions)

            # 5) record rewards
            for agent in self.agents:
                evaluation_data['agents'][agent]['rewards'].append(rewards[agent])

            # 6) record cross-class infections
            last_allowed = [evaluation_data['agents'][a]['allowed_students'][-1]
                            for a in self.agents]
            current_infected = [states[a][0] for a in self.agents]
            for i, agent in enumerate(self.agents):
                term = 0.0
                for j in range(self.num_agents):
                    if i != j and last_allowed[j] > 0:
                        term += (current_infected[j] / last_allowed[j])
                if self.num_agents > 1:
                    term = term / (self.num_agents - 1) * last_allowed[i]
                evaluation_data['agents'][agent]['cross_class_infections'].append(term)

            evaluation_data['steps'].append(step)
            states = next_states
            if all(dones.values()):
                break

        # 7) total rewards
        evaluation_data['total_rewards'] = {
            agent: sum(evaluation_data['agents'][agent]['rewards'])
            for agent in self.agents
        }
        return evaluation_data

    def plot_rewards(self, save_path="results/avg_rewards_CTDE.png"):
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
        plt.close()
