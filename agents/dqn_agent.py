import numpy as np
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
    # def train_centralized_mc(self,
    #                          env,
    #                          max_steps: int = 30,
    #                          total_episodes: int = 1000,
    #                          convergence_window: int = 20):
    #     """
    #     Fully‐centralized Monte Carlo updates over joint state & joint action.
    #     At the end of each episode, computes discounted team‐returns G_t,
    #     then does one big gradient step minimizing (Q_joint(s,a) - G_t)^2
    #     summed over the whole episode.
    #     """
    #     self.global_rewards.clear()
    #     self.episode_rewards = {a: [] for a in self.agents}
    #
    #     for ep in range(1, total_episodes+1):
    #         # reset
    #         states = env.reset()
    #         ep_reward = 0.0
    #         self.transition_buffer.clear()
    #
    #         # decay epsilon
    #         self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
    #         if self.epsilon <= self.min_epsilon:
    #             print(f"[MC Centralized] ε has decayed to floor ({self.epsilon:.4f}) on episode {ep}; stopping.")
    #             break
    #
    #         # 1) run one episode, collect (joint_state, joint_action, team_reward)
    #         for t in range(max_steps):
    #             js = np.concatenate([states[a] for a in self.agents])
    #             # ε-greedy joint action
    #             if random.random() < self.epsilon:
    #                 joint_act = [random.randrange(self.action_space_size)
    #                              for _ in self.agents]
    #             else:
    #                 with torch.no_grad():
    #                     q_heads = self.network(
    #                         torch.FloatTensor(js).unsqueeze(0)
    #                     )[0]  # (N_agents, A)
    #                     joint_act = [int(q_heads[i].argmax())
    #                                  for i in range(self.num_agents)]
    #
    #             next_states, rewards, dones, _ = env.step({
    #                 a: joint_act[i] for i, a in enumerate(self.agents)
    #             })
    #             # team reward
    #             r_team = sum(rewards.values())
    #             ep_reward += r_team
    #
    #             # store transition
    #             self.transition_buffer.append((js, joint_act, r_team))
    #
    #             states = next_states
    #             if all(dones.values()):
    #                 break
    #
    #         # 2) compute backward discounted returns G_t
    #         returns = []
    #         G = 0.0
    #         for (_, _, r) in reversed(self.transition_buffer):
    #             G = r + self.gamma * G
    #             returns.insert(0, G)
    #
    #         # 3) one gradient step over the whole episode
    #         total_loss = 0.0
    #         for (js, joint_act, _), G_t in zip(self.transition_buffer, returns):
    #             # compute current joint‐Q = sum_i Q_i(js, a_i)
    #             q_heads = self.network(
    #                 torch.FloatTensor(js).unsqueeze(0)
    #             )[0]  # (N_agents, A)
    #             q_joint = sum(q_heads[i, joint_act[i]]
    #                           for i in range(self.num_agents))
    #             # MSE against Monte Carlo target
    #             target = torch.tensor(G_t, dtype=q_joint.dtype)
    #             total_loss += nn.MSELoss()(q_joint, target)
    #
    #         # normalize loss
    #         total_loss = total_loss / len(self.transition_buffer)
    #         self.optimizer.zero_grad()
    #         total_loss.backward()
    #         torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
    #         self.optimizer.step()
    #
    #         # 4) record & sync
    #         self.global_rewards.append(ep_reward)
    #         for a in self.agents:
    #             self.episode_rewards[a].append(ep_reward / self.num_agents)
    #
    #         self.target_network.load_state_dict(self.network.state_dict())
    #
    #         # # periodic target‐net sync
    #         # if ep % 50 == 0:
    #         #     self.target_network.load_state_dict(self.network.state_dict())
    #
    #         # # convergence check
    #         # if len(self.global_rewards) >= convergence_window:
    #         #     avg = sum(self.global_rewards[-convergence_window:]) / convergence_window
    #         #     thresh = 0.9 * self.num_agents * env.gamma * max_steps
    #         #     if avg >= thresh:
    #         #         print(f"[MC Centralized] Converged at episode {ep} (avg {avg:.2f})")
    #         #         break
    #
    #     # save learning curve
    #     # save_path = f"results/avg_rewards_centralized_mc_gamma_{env.gamma}.png"
    #     # self.plot_rewards(save_path)
    #
    # def train_centralized_td(self,
    #                       env,
    #                       max_steps: int = 30,
    #                       total_episodes: int = 1000,
    #                       convergence_window: int = 20):
    #     """
    #     Fully-centralized Q-learning over joint state & joint action.
    #     Network is the same CentralizedDQNetwork: N heads → sum into one joint-Q.
    #     """
    #     self.global_rewards.clear()
    #     self.episode_rewards = {a: [] for a in self.agents}
    #     for ep in range(1, total_episodes+1):
    #         states = env.reset()
    #         ep_reward = 0.0
    #
    #         # decay epsilon
    #         self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
    #         if self.epsilon <= self.min_epsilon:
    #             print(f"[MC Centralized] ε has decayed to floor ({self.epsilon:.4f}) on episode {ep}; stopping.")
    #             break
    #
    #         for t in range(max_steps):
    #             # 1) form joint_state and select joint_action ε-greedy
    #             js = np.concatenate([states[a] for a in self.agents])
    #             if random.random() < self.epsilon:
    #                 joint_act = [random.randrange(self.action_space_size)
    #                              for _ in self.agents]
    #             else:
    #                 with torch.no_grad():
    #                     q_heads = self.network(
    #                         torch.FloatTensor(js).unsqueeze(0)
    #                     )[0]            # shape (N_agents, A)
    #                     # greedy per head:
    #                     joint_act = [int(q_heads[i].argmax())
    #                                  for i in range(self.num_agents)]
    #
    #             # 2) step env
    #             next_states, rewards, dones, _ = env.step({
    #                 a: joint_act[i] for i, a in enumerate(self.agents)
    #             })
    #             r_team = sum(rewards.values())
    #             ep_reward += r_team
    #
    #             # 3) compute TD target y
    #             #    - current joint Q = sum_i Q_i(s,a_i)
    #             q_heads = self.network(
    #                 torch.FloatTensor(js).unsqueeze(0)
    #             )[0]                 # (N, A)
    #             q_joint = sum(q_heads[i, joint_act[i]]
    #                           for i in range(self.num_agents))
    #
    #             #    - next‐state best joint Q
    #             js_next = np.concatenate([next_states[a] for a in self.agents])
    #             with torch.no_grad():
    #                 qn = self.target_network(
    #                     torch.FloatTensor(js_next).unsqueeze(0)
    #                 )[0]          # (N, A)
    #                 # precompute best sum over all per-agent maxes:
    #                 best_sum = sum(qn[i].max().item()
    #                                for i in range(self.num_agents))
    #
    #             y = r_team + self.gamma * best_sum
    #
    #             # 4) loss & backprop
    #             loss = nn.MSELoss()(q_joint, torch.tensor(y))
    #             self.optimizer.zero_grad()
    #             loss.backward()
    #             torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
    #             self.optimizer.step()
    #
    #             states = next_states
    #             if all(dones.values()):
    #                 break
    #
    #         # log
    #         self.global_rewards.append(ep_reward)
    #         for a in self.agents:
    #             self.episode_rewards[a].append(ep_reward / self.num_agents)
    #         self.target_network.load_state_dict(self.network.state_dict())
    #
    #         # # target network sync (periodically)
    #         # if ep % 50 == 0:
    #         #
    #         #
    #         # # check convergence
    #         # if len(self.global_rewards) >= convergence_window:
    #         #     avg = sum(self.global_rewards[-convergence_window:]) / convergence_window
    #         #     if avg >= 0.9 * env.num_agents * env.gamma * max_steps:
    #         #         print(f"Converged at episode {ep}")
    #         #         break
    #
    #     # save learning curve
    #     # save_path = f"results/avg_rewards_centralized_td_{self.reward_mix_alpha}_gamma_{env.gamma}.png"
    #     # self.plot_rewards(save_path)
    #
    #
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
    def train_td_double_hyper_rewards(self,
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
                    + self.gamma
                    * q_next_target[i, q_next_online[i].argmax().item()]
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

        # after training: save plot if requested
        if not return_rewards and save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(
                save_dir,
                f"avg_rewards_CTDE_td_gamma_{env.gamma}.png"
            )
            self.plot_rewards(save_path)

        # return list of global rewards if asked
        if return_rewards:
            return reward_history

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

    def train_td(self, env, max_steps=30, save_dir: str = None):
        """
        Training loop using one‐step TD updates at each env.step(),
        with early stopping once moving‐average global return ≥ 90% of max.
        """
        discount_gamma = 0.99  # no discounting for finite MDPs
        total_episodes = 1000
        convergence_window = 20

        # compute convergence threshold
        max_per_agent_step = self.gamma * env.total_students
        max_global_episode = len(self.agents) * max_per_agent_step * max_steps
        target_avg_return = 0.9 * max_global_episode

        convergence_ep = None
        # pbar = tqdm(total=total_episodes, desc="TD Training", leave=True)

        # reset logs
        self.global_rewards.clear()
        self.episode_rewards = {a: [] for a in self.agents}

        for episode in range(1, total_episodes + 1):
            total_rewards = {agent: 0.0 for agent in self.agents}
            global_reward = 0.0
            states = env.reset()
            # decay ε at start of episode
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

            for step in range(max_steps):
                # 1) select actions (no reward_mix logic here)
                joint_state = np.concatenate([states[a] for a in self.agents])
                actions = {
                    a: self.select_local_action(a, states[a])
                    for a in self.agents
                }

                # 2) step env
                next_states, rewards, dones, _ = env.step(actions)

                # 3) logging
                for a in self.agents:
                    total_rewards[a] += rewards[a]
                global_reward += sum(rewards.values())

                # 4) build tensors
                js = torch.FloatTensor(joint_state).unsqueeze(0)  # [1, N·S]
                next_js = np.concatenate([next_states[a] for a in self.agents])
                js_next = torch.FloatTensor(next_js).unsqueeze(0)  # [1, N·S]
                q_all = self.network(js)[0]  # [N, A]
                q_next_all = self.target_network(js_next)[0]  # [N, A]

                # 5) compute 1‐step TD target
                q_taken = torch.stack([
                    q_all[i, actions[a]]
                    for i, a in enumerate(self.agents)
                ])  # [N_agents]

                td_targets = []
                for i, a in enumerate(self.agents):
                    r = rewards[a]
                    max_qn = q_next_all[i].max().item()
                    td_targets.append(r + max_qn)
                target = torch.tensor(td_targets, dtype=q_taken.dtype)

                # 6) loss & update
                loss = nn.MSELoss()(q_taken, target)
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

        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"avg_rewards_CTDE_td_gamma_{env.gamma}.png")
        self.plot_rewards(save_path)
    #
    def train_mc(self, env, max_steps=30):
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


            window = min(20, len(self.global_rewards))
            avg_r = sum(self.global_rewards[-window:]) / window

            if target_avg_return is not None and avg_r >= target_avg_return:
                convergence_ep = episode
                # pbar.close()
                print(f"Converged at episode {episode} (avg over last {window} = {avg_r:.2f})")
                break

            # pbar.set_description(
            #     f"Episode {episode+1} | AvgR {avg_r:.2f} | Eps {self.epsilon:.3f}"
            # )
            # pbar.update(1)

        # pbar.close()
        save_path = f"./results/ctde_mc_100_default/avg_rewards_CTDE_mc_gamma_{env.gamma}.png"
        self.plot_rewards(save_path)
    def train_mc_hyper(self, env, max_steps=30, return_loss=False):
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

    def train_td_hyper(self, env, max_steps=30, return_rewards=False, save_dir=None):
        """
        One-step TD training for hyperparameter tuning.
        If return_rewards=True, returns a list of per-episode global rewards.
        """
        total_episodes = 1000
        max_per_agent_step = self.gamma * env.total_students
        max_global_episode = len(self.agents) * max_per_agent_step * max_steps
        target_avg_return = 0.9 * max_global_episode

        self.global_rewards.clear()
        self.episode_rewards = {a: [] for a in self.agents}
        reward_history = [] if return_rewards else None

        for episode in range(1, total_episodes + 1):
            total_rewards = {agent: 0.0 for agent in self.agents}
            global_reward = 0.0
            states = env.reset()
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

            for step in range(max_steps):
                # select & step
                joint_state = np.concatenate([states[a] for a in self.agents])
                actions = {a: self.select_local_action(a, states[a]) for a in self.agents}
                next_states, rewards, dones, _ = env.step(actions)

                # accumulate
                for a in self.agents:
                    total_rewards[a] += rewards[a]
                global_reward += sum(rewards.values())

                # TD update (omitted here—assume your fixed version)
                # ...

                states = next_states
                if all(dones.values()):
                    break

            # record
            for a in self.agents:
                self.episode_rewards[a].append(total_rewards[a])
            self.global_rewards.append(global_reward)
            if return_rewards:
                reward_history.append(global_reward)
            # update target
            self.target_network.load_state_dict(self.network.state_dict())

        if return_rewards:
            return reward_history

    def train_td_hyper_loss(self, env, max_steps=30, return_loss=False):
        """
        One-step TD training for hyperparameter tuning.
        If return_loss=True, returns a list of per-step losses.
        """
        total_episodes = 1000
        max_per_agent_step = self.gamma * env.total_students
        max_global_episode = len(self.agents) * max_per_agent_step * max_steps
        target_avg_return = 0.9 * max_global_episode

        self.global_rewards.clear()
        self.episode_rewards = {a: [] for a in self.agents}
        loss_history = [] if return_loss else None

        for episode in range(1, total_episodes + 1):
            total_rewards = {agent: 0.0 for agent in self.agents}
            global_reward = 0.0
            states = env.reset()
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

            for step in range(max_steps):
                # 1) Select actions (local policy only, no reward mix)
                joint_state = np.concatenate([states[a] for a in self.agents])
                actions = {
                    a: self.select_local_action(a, states[a])
                    for a in self.agents
                }

                # 2) Step environment
                next_states, rewards, dones, _ = env.step(actions)

                # 3) Logging rewards
                for a in self.agents:
                    total_rewards[a] += rewards[a]
                global_reward += sum(rewards.values())

                # 4) Prepare state tensors
                js_tensor = torch.FloatTensor(joint_state).unsqueeze(0)
                next_joint_state = np.concatenate([next_states[a] for a in self.agents])
                js_next_tensor = torch.FloatTensor(next_joint_state).unsqueeze(0)

                # 5) Network predictions
                q_all = self.network(js_tensor)[0]  # [N_agents, A]
                q_next_all = self.target_network(js_next_tensor)[0]  # [N_agents, A]

                # 6) Compute Q-values for taken actions
                q_taken = torch.stack([
                    q_all[i, actions[a]] for i, a in enumerate(self.agents)
                ])

                # 7) TD target
                td_targets = torch.stack([
                    torch.tensor(rewards[a], dtype=torch.float32) + q_next_all[i].detach().max()
                    for i, a in enumerate(self.agents)
                ])
                target = torch.tensor(td_targets, dtype=q_taken.dtype)

                # 8) Loss + Backprop
                loss = nn.MSELoss()(q_taken, target)
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
                self.optimizer.step()

                if return_loss:
                    loss_history.append(loss.item())

                states = next_states
                if all(dones.values()):
                    break

            # Record episode rewards
            for a in self.agents:
                self.episode_rewards[a].append(total_rewards[a])
            self.global_rewards.append(global_reward)
            self.target_network.load_state_dict(self.network.state_dict())

        if return_loss:
            return loss_history

    def train_td_hyper_loss_double(self, env, max_steps=30, return_loss=False):
        """
        One-step TD training for hyperparameter tuning using Double-DQN.
        If return_loss=True, returns a list of per-step losses.
        """
        total_episodes = 1000
        max_per_agent_step = self.gamma * env.total_students
        max_global_episode = len(self.agents) * max_per_agent_step * max_steps
        target_avg_return = 0.9 * max_global_episode

        self.global_rewards.clear()
        self.episode_rewards = {a: [] for a in self.agents}
        loss_history = [] if return_loss else None

        for episode in range(1, total_episodes + 1):
            total_rewards = {agent: 0.0 for agent in self.agents}
            global_reward = 0.0
            states = env.reset()
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

            for step in range(max_steps):
                # 1) Select actions (local policy only)
                joint_state = np.concatenate([states[a] for a in self.agents])
                actions = {
                    a: self.select_local_action(a, states[a])
                    for a in self.agents
                }

                # 2) Step environment
                next_states, rewards, dones, _ = env.step(actions)

                # 3) Log rewards
                for a in self.agents:
                    total_rewards[a] += rewards[a]
                global_reward += sum(rewards.values())

                # 4) Build tensors
                js = torch.FloatTensor(joint_state).unsqueeze(0)  # [1, N·S]
                next_js = torch.FloatTensor(
                    np.concatenate([next_states[a] for a in self.agents])
                ).unsqueeze(0)  # [1, N·S]

                # 5) Q-value predictions
                q_all = self.network(js)[0]  # online net: [N_agents, A]
                q_next_online = self.network(next_js)[0]  # online net @ next state
                q_next_target = self.target_network(next_js)[0]  # target net @ next state

                # 6) Q(s,a) for taken actions
                q_taken = torch.stack([
                    q_all[i, actions[a]] for i, a in enumerate(self.agents)
                ])  # shape [N_agents]

                # 7) Double-DQN targets: use argmax from online net, evaluate via target net
                td_targets = torch.stack([
                    torch.tensor(rewards[a], dtype=torch.float32) +
                    self.gamma * q_next_target[i, q_next_online[i].argmax().item()].detach()
                    for i, a in enumerate(self.agents)
                ]).to(q_taken.dtype)

                # 8) Loss & backprop
                loss = nn.MSELoss()(q_taken, td_targets)
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
                self.optimizer.step()

                if return_loss:
                    loss_history.append(loss.item())

                states = next_states
                if all(dones.values()):
                    break

            # Record episode rewards
            for a in self.agents:
                self.episode_rewards[a].append(total_rewards[a])
            self.global_rewards.append(global_reward)

            # Sync target network
            self.target_network.load_state_dict(self.network.state_dict())

        if return_loss:
            return loss_history

    def train_centralized_mc(self, env,
                             max_steps: int = 30,
                             total_episodes: int = 1000,
                             convergence_window: int = 20,
                             return_loss: bool = False):
        """
        Centralized MC training. If return_loss=True, returns per-episode losses.
        """
        self.global_rewards.clear()
        self.episode_rewards = {a: [] for a in self.agents}
        loss_history = [] if return_loss else None

        for ep in range(1, total_episodes + 1):
            states = env.reset()
            ep_reward = 0.0
            self.transition_buffer.clear()
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
            if self.epsilon <= self.min_epsilon:
                break

            # collect one full episode
            for t in range(max_steps):
                js = np.concatenate([states[a] for a in self.agents])
                if random.random() < self.epsilon:
                    joint_act = [random.randrange(self.action_space_size)
                                 for _ in self.agents]
                else:
                    with torch.no_grad():
                        q_heads = self.network(
                            torch.FloatTensor(js).unsqueeze(0)
                        )[0]
                        joint_act = [int(q_heads[i].argmax())
                                     for i in range(self.num_agents)]

                next_states, rewards, dones, _ = env.step(
                    {a: joint_act[i] for i, a in enumerate(self.agents)}
                )
                r_team = sum(rewards.values())
                ep_reward += r_team
                self.transition_buffer.append((js, joint_act, r_team))
                states = next_states
                if all(dones.values()):
                    break

            # compute MC-targets
            returns, G = [], 0.0
            for (_, _, r) in reversed(self.transition_buffer):
                G = r + self.gamma * G
                returns.insert(0, G)

            # batched gradient step
            total_loss = 0.0
            for (js, joint_act, _), G_t in zip(self.transition_buffer, returns):
                q_heads = self.network(
                    torch.FloatTensor(js).unsqueeze(0)
                )[0]
                q_joint = sum(q_heads[i, joint_act[i]]
                              for i in range(self.num_agents))
                total_loss += nn.MSELoss()(q_joint, torch.tensor(G_t))

            total_loss = total_loss / len(self.transition_buffer)
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
            self.optimizer.step()

            if return_loss:
                loss_history.append(total_loss.item())

            self.global_rewards.append(ep_reward)
            for a in self.agents:
                self.episode_rewards[a].append(ep_reward / self.num_agents)
            self.target_network.load_state_dict(self.network.state_dict())

        if return_loss:
            return loss_history

    def train_centralized_td(self, env,
                             max_steps: int = 30,
                             total_episodes: int = 1000,
                             convergence_window: int = 20,
                             return_loss: bool = False):
        """
        Centralized TD training. If return_loss=True, returns per-step losses.
        """
        self.global_rewards.clear()
        self.episode_rewards = {a: [] for a in self.agents}
        loss_history = [] if return_loss else None

        for ep in range(1, total_episodes + 1):
            states = env.reset()
            ep_reward = 0.0
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
            if self.epsilon <= self.min_epsilon:
                break

            for t in range(max_steps):
                js = np.concatenate([states[a] for a in self.agents])
                if random.random() < self.epsilon:
                    joint_act = [random.randrange(self.action_space_size)
                                 for _ in self.agents]
                else:
                    with torch.no_grad():
                        q_heads = self.network(
                            torch.FloatTensor(js).unsqueeze(0)
                        )[0]
                        joint_act = [int(q_heads[i].argmax())
                                     for i in range(self.num_agents)]

                next_states, rewards, dones, _ = env.step(
                    {a: joint_act[i] for i, a in enumerate(self.agents)}
                )
                r_team = sum(rewards.values())
                ep_reward += r_team

                # TD‐target
                q_heads = self.network(
                    torch.FloatTensor(js).unsqueeze(0)
                )[0]
                q_joint = sum(q_heads[i, joint_act[i]]
                              for i in range(self.num_agents))
                js_next = np.concatenate([next_states[a] for a in self.agents])
                with torch.no_grad():
                    qn = self.target_network(
                        torch.FloatTensor(js_next).unsqueeze(0)
                    )[0]
                    best_sum = sum(qn[i].max().item()
                                   for i in range(self.num_agents))
                y = r_team + self.gamma * best_sum

                loss = nn.MSELoss()(q_joint, torch.tensor(y))
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
                self.optimizer.step()

                if return_loss:
                    loss_history.append(loss.item())

                states = next_states
                if all(dones.values()):
                    break

            self.global_rewards.append(ep_reward)
            for a in self.agents:
                self.episode_rewards[a].append(ep_reward / self.num_agents)
            self.target_network.load_state_dict(self.network.state_dict())

        if return_loss:
            return loss_history

    def train_centralized_td_hyper(self, env,
                             max_steps: int = 30,
                             total_episodes: int = 1000,
                             convergence_window: int = 20,
                             return_loss: bool = False):
        """
        Centralized TD training using episode-based loss (like MC).
        If return_loss=True, returns per-episode losses.
        """
        self.global_rewards.clear()
        self.episode_rewards = {a: [] for a in self.agents}
        loss_history = [] if return_loss else None

        for ep in range(1, total_episodes + 1):
            states = env.reset()
            ep_reward = 0.0
            self.transition_buffer.clear()
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

            # Collect one episode
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
                self.transition_buffer.append((js, joint_act, r_team, next_states))
                states = next_states
                if all(dones.values()):
                    break

            # Batched TD updates
            total_loss = 0.0
            for (js, joint_act, r_team, next_states) in self.transition_buffer:
                q_heads = self.network(torch.FloatTensor(js).unsqueeze(0))[0]
                q_joint = sum(q_heads[i, joint_act[i]] for i in range(self.num_agents))

                js_next = np.concatenate([next_states[a] for a in self.agents])
                with torch.no_grad():
                    qn = self.target_network(torch.FloatTensor(js_next).unsqueeze(0))[0]
                    best_sum = sum(qn[i].max().item() for i in range(self.num_agents))

                y = r_team + self.gamma * best_sum
                loss = nn.MSELoss()(q_joint, torch.tensor(y))
                total_loss += loss

            # Final gradient step
            total_loss = total_loss / len(self.transition_buffer)
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
            self.optimizer.step()

            if return_loss:
                loss_history.append(total_loss.item())

            self.global_rewards.append(ep_reward)
            for a in self.agents:
                self.episode_rewards[a].append(ep_reward / self.num_agents)
            self.target_network.load_state_dict(self.network.state_dict())

        if return_loss:
            return loss_history

    def train_centralized_mc_hyper(self, env,
                             max_steps: int = 30,
                             total_episodes: int = 1000,
                             convergence_window: int = 20,
                             return_loss: bool = False):
        """
        Centralized MC training. If return_loss=True, returns per-episode losses.
        """
        self.global_rewards.clear()
        self.episode_rewards = {a: [] for a in self.agents}
        loss_history = [] if return_loss else None

        for ep in range(1, total_episodes + 1):
            states = env.reset()
            ep_reward = 0.0
            self.transition_buffer.clear()
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
                G = r  # no bootstrapping, as in `train_mc_hyper`
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
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
            self.optimizer.step()

            if return_loss:
                loss_history.append(total_loss.item())

            self.global_rewards.append(ep_reward)
            for a in self.agents:
                self.episode_rewards[a].append(ep_reward / self.num_agents)

            self.target_network.load_state_dict(self.network.state_dict())

        if return_loss:
            return loss_history

    def evaluate(self, env, max_steps=52, centralized=False):
        """
        Evaluation of the agent's policy.
        If reward_mix_alpha > 0, uses centralized joint evaluation with greedy joint action selection.
        Otherwise, decentralized greedy evaluation per agent.
        """
        if centralized:
            return self.evaluate_centralized(env, max_steps)
        else:
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

                        # Generate all possible joint actions (Cartesian product)
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
            # print("Total rewards per agent:", total_rewards)
            evaluation_data['total_rewards'] = total_rewards
            return evaluation_data

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
