import os
import math
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
from itertools import product
from tqdm import tqdm
from torch.distributions import Normal

# -----------------------------------------------------------------------------
#  Actor-Critic Network for CTDE-PPO
# -----------------------------------------------------------------------------


class ActorCriticNetwork(nn.Module):
    def __init__(self, num_agents, state_dim, hidden_dim=64, hidden_layers=3):
        super().__init__()
        input_dim = num_agents * state_dim

        # shared body
        layers = []
        prev = input_dim
        for _ in range(hidden_layers):
            layers += [nn.Linear(prev, hidden_dim), nn.ReLU()]
            prev = hidden_dim
        self.shared = nn.Sequential(*layers)

        # per‐agent Gaussian policy means
        self.actor_means = nn.ModuleList(
            [nn.Linear(hidden_dim, 1) for _ in range(num_agents)]
        )
        # per‐agent log‐std parameters (learned)
        self.log_stds = nn.ParameterList(
            [nn.Parameter(torch.zeros(1)) for _ in range(num_agents)]
        )

        # critic head (state‐value)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, joint_state):
        """
        joint_state: Tensor of shape (B, N * S)
        Returns:
          means:  Tensor of shape (B, N)
          log_stds: List of N tensors (each shape (1,))
          values: Tensor of shape (B,)
        """
        x = self.shared(joint_state)              # (B, hidden_dim)

        # build per-agent means
        means = torch.stack(
            [head(x).squeeze(-1) for head in self.actor_means],
            dim=1
        )                                          # (B, N)

        # critic value
        values = self.critic(x).squeeze(-1)        # (B,)

        return means, self.log_stds, values

# -----------------------------------------------------------------------------
#  PPOAgent
# -----------------------------------------------------------------------------
class PPOAgent:
    def __init__(self, agents, state_dim, action_space_size,
                 lr=0.1, gamma=0.99, clip_eps=0.2,
                 value_coef=0.5, entropy_coef=0.01,
                 batch_size=64, ppo_epochs=4, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

        self.agents = agents
        self.num_agents = len(agents)
        self.state_dim = state_dim
        self.action_space_size = action_space_size

        # PPO hyperparams
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.batch_size = batch_size
        self.ppo_epochs = ppo_epochs

        # networks & optimizer
        self.net = ActorCriticNetwork(self.num_agents, state_dim, action_space_size)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)

        # logs
        self.episode_rewards = {a: [] for a in self.agents}
        self.global_rewards = []

    def select_joint(self, joint_state):
        """
        Given one joint_state tensor (shape=(N·S,)),
        returns continuous actions, their log-probs, and the state-value.
        """
        means, log_std_params, value = self.net(joint_state.unsqueeze(0))
        # means: Tensor shape (1, N)
        # log_std_params: ParameterList of N scalars
        # value: Tensor shape (1,)

        means = means.squeeze(0)                    # (N,)

        # extract and stack the log-std parameters into a tensor:
        log_stds = torch.stack([p for p in log_std_params]).squeeze(-1)  # (N,)

        stds = log_stds.exp()                       # (N,)

        # create independent Gaussians:
        dist    = Normal(means, stds)
        actions = dist.sample()                     # (N,)
        logp    = dist.log_prob(actions)            # (N,)

        return actions.tolist(), logp, value.squeeze(0)

    def collect_rollout(self, env, rollout_len):
        buf = {
            'states': [], 'actions': [], 'logps': [], 'values': [],
            'rewards': [], 'dones': []
        }
        state = env.reset()
        for _ in range(rollout_len):
            joint = torch.FloatTensor(np.concatenate([state[a] for a in self.agents]))
            acts, logp, val = self.select_joint(joint)
            next_s, rew_dict, done_dict, _ = env.step({a: acts[i] for i,a in enumerate(self.agents)})

            buf['states'].append(joint.numpy())
            buf['actions'].append(acts)
            buf['logps'].append(logp.detach())
            buf['values'].append(val.detach())
            buf['rewards'].append([rew_dict[a] for a in self.agents])
            buf['dones'].append(all(done_dict.values()))

            state = next_s

        # bootstrap last value
        joint = torch.FloatTensor(np.concatenate([state[a] for a in self.agents]))
        _, last_val = self.net(joint.unsqueeze(0))
        buf['last_value'] = last_val.squeeze(0).detach()
        return buf

    def compute_gae(self, buf):
        rewards = torch.FloatTensor(buf['rewards'])      # (T, N)
        values  = torch.stack(buf['values'])             # (T,)
        last_v  = buf['last_value']
        dones   = torch.BoolTensor(buf['dones'])         # (T,)

        T = len(rewards)
        advantages = torch.zeros(T, self.num_agents)
        gae = 0
        for t in reversed(range(T)):
            mask = 1.0 - dones[t].float()
            # use team-average reward
            delta = rewards[t].mean() + self.gamma * last_v * mask - values[t]
            gae = delta + self.gamma * 0.95 * mask * gae
            advantages[t] = gae
            last_v = values[t]
        returns = advantages + values.unsqueeze(1)
        return advantages, returns

    def ppo_update(self, buf):
        # 1) Compute advantages & returns
        advs, rets = self.compute_gae(buf)

        # 2) Stack states & actions efficiently
        states = torch.from_numpy(np.array(buf['states'], dtype=np.float32))   # (T, N·S)
        actions = torch.from_numpy(np.array(buf['actions'], dtype=np.float32)) # (T, N)
        old_logps = torch.stack(buf['logps'])                                  # (T, N)

        T = states.size(0)
        idxs = np.arange(T)
        for _ in range(self.ppo_epochs):
            np.random.shuffle(idxs)
            for start in range(0, T, self.batch_size):
                mb = idxs[start:start+self.batch_size]
                mb_s = states[mb]            # (B, N·S)
                mb_a = actions[mb]           # (B, N)
                mb_old_l = old_logps[mb]     # (B, N)
                mb_adv = advs[mb].mean(dim=1)  # (B,)
                mb_ret = rets[mb].mean(dim=1)  # (B,)

                # 3) Forward pass: get means, log_stds list, and values
                means, log_std_params, vals = self.net(mb_s)
                # means: (B, N), vals: (B,)

                # 4) Build a (N,) tensor of log_stds, then expand to (B, N)
                log_stds = torch.stack([p.squeeze() for p in log_std_params])  # (N,)
                log_stds = log_stds.unsqueeze(0).expand_as(means)             # (B, N)
                stds     = log_stds.exp()                                      # (B, N)

                # 5) Continuous Normal distribution
                dist = Normal(means, stds)
                # sum log-probs across agents → (B,)
                new_logp = dist.log_prob(mb_a).sum(dim=1)
                # sum entropy across agents, then mean over batch
                entropy = dist.entropy().sum(dim=1).mean()

                # 6) PPO clipped-surrogate objective
                old_logp_sum = mb_old_l.sum(dim=1)                             # (B,)
                ratio = torch.exp(new_logp - old_logp_sum)                     # (B,)
                s1 = ratio * mb_adv
                s2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * mb_adv
                policy_loss = -torch.min(s1, s2).mean()

                # 7) Value-function loss
                value_loss = self.value_coef * (vals - mb_ret).pow(2).mean()

                # 8) Total loss
                loss = policy_loss + value_loss - self.entropy_coef * entropy

                # 9) Backprop
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
                self.optimizer.step()

    # def train(self, env, total_timesteps, rollout_len=2048):
    #     timesteps = 0
    #     pbar = tqdm(total=total_timesteps, desc="PPO Training")
    #     while timesteps < total_timesteps:
    #         buf = self.collect_rollout(env, rollout_len)
    #         self.ppo_update(buf)
    #         timesteps += rollout_len
    #         pbar.update(rollout_len)
    #     pbar.close()
    def train(self, env, total_timesteps, rollout_len=2048):
        """
        On-policy PPO driven by episodes (of length max_weeks),
        but still updating every `rollout_len` steps.
        """
        timesteps = 0
        episode_idx = 0

        # reset logs
        self.episode_rewards = {a: [] for a in self.agents}
        self.global_rewards = []

        # buffer for on-policy data
        buffer = {
            'states': [], 'actions': [], 'logps': [], 'values': [],
            'rewards': [], 'dones': []
        }

        pbar = tqdm(total=total_timesteps, desc="PPO Training")

        while timesteps < total_timesteps:
            # start new episode
            obs = env.reset()
            ep_rewards = {a: 0.0 for a in self.agents}
            done = False

            # run one episode (max_weeks steps)
            while not done and timesteps < total_timesteps:
                # 1) build joint state as plain Python floats
                joint_list = []
                for a in self.agents:
                    # obs[a] is [infected, risk]
                    inf = float(obs[a][0])
                    risk = float(obs[a][1])
                    joint_list += [inf, risk]
                joint = torch.tensor(joint_list, dtype=torch.float32)  # (N * state_dim,)

                # 2) get action, logp, value
                acts, logp, val = self.select_joint(joint)

                # 3) step env
                next_obs, rew_dict, done_dict, _ = env.step(
                    {a: acts[i] for i, a in enumerate(self.agents)}
                )
                done = all(done_dict.values())

                # 4) append to buffer
                buffer['states'].append(joint_list)  # store list, not tensor
                buffer['actions'].append(acts)
                buffer['logps'].append(logp.detach())
                buffer['values'].append(val.detach())
                buffer['rewards'].append([rew_dict[a] for a in self.agents])
                buffer['dones'].append(done)

                # 5) accumulate episode reward
                for i, a in enumerate(self.agents):
                    ep_rewards[a] += rew_dict[a]

                timesteps += 1
                pbar.update(1)
                obs = next_obs

                # 6) if buffer full, bootstrap & update
                if len(buffer['states']) >= rollout_len:
                    # build joint_last in same way
                    joint_list_last = []
                    for a in self.agents:
                        joint_list_last += [float(obs[a][0]), float(obs[a][1])]
                    joint_last = torch.tensor(joint_list_last, dtype=torch.float32)

                    _, _, last_val = self.net(joint_last.unsqueeze(0))
                    buffer['last_value'] = last_val.squeeze(0).detach()

                    self.ppo_update(buffer)
                    # clear buffer but keep last_value
                    for k in list(buffer.keys()):
                        if k != 'last_value':
                            buffer[k].clear()

            # end of one episode: log returns
            for a in self.agents:
                self.episode_rewards[a].append(ep_rewards[a])
            self.global_rewards.append(sum(ep_rewards.values()))
            episode_idx += 1

        pbar.close()

        # plot training curves
        self.plot_rewards("results/ppo_training_rewards.png")
        print(f"Saved training reward curves over {episode_idx} episodes to results/ppo_training_rewards.png")

    def evaluate(self, env, max_steps=52, csv_path="results/ppo_evaluation.csv"):
        """
        Continuous‐action evaluation:
        - select_joint → real-valued actions per agent
        - round to ints for env.step()
        - log both raw actions and the rounded allowed_students
        """
        states = env.reset()
        data = {'steps': [], 'agents': {}}
        for a in self.agents:
            data['agents'][a] = {
                'rewards': [],
                'actions': [],  # raw float actions
                'allowed_students': [],  # int mapped to env
                'infected': [],
                'cross_class_infections': []
            }

        for step in range(max_steps):
            # 1) get continuous actions
            joint = torch.FloatTensor(np.concatenate([states[a] for a in self.agents]))
            raw_actions, logp, _ = self.select_joint(joint)

            # 2) round/clamp to valid student counts
            rounded = []
            for i, a in enumerate(self.agents):
                # clamp between 0 and total_students, then round
                val = max(0.0, min(env.total_students, raw_actions[i]))
                allowed = int(round(val))
                rounded.append(allowed)

            # 3) step environment
            action_dict = {agent: rounded[i] for i, agent in enumerate(self.agents)}
            next_s, rew_d, done_d, _ = env.step(action_dict)

            # 4) log per-agent
            for i, agent in enumerate(self.agents):
                data['agents'][agent]['actions'].append(raw_actions[i])
                data['agents'][agent]['allowed_students'].append(rounded[i])
                data['agents'][agent]['infected'].append(states[agent][0])
                data['agents'][agent]['rewards'].append(rew_d[agent])

            # 5) cross-class infections
            allowed_all = [data['agents'][a]['allowed_students'][-1]
                           for a in self.agents]
            inf_all = [states[a][0] for a in self.agents]
            for i, agent in enumerate(self.agents):
                term = 0.0
                for j in range(self.num_agents):
                    if i != j and allowed_all[j] > 0:
                        term += inf_all[j] / allowed_all[j]
                if self.num_agents > 1:
                    term = term / (self.num_agents - 1) * allowed_all[i]
                data['agents'][agent]['cross_class_infections'].append(term)

            data['steps'].append(step)
            states = next_s
            if all(done_d.values()):
                break

        # 6) total rewards
        totals = {a: sum(data['agents'][a]['rewards']) for a in self.agents}
        data['total_rewards'] = totals

        # 7) flatten & save CSV
        rows = []
        for idx, step in enumerate(data['steps']):
            row = {'step': step}
            for a in self.agents:
                stats = data['agents'][a]
                row.update({
                    f"{a}_action": stats['actions'][idx],
                    f"{a}_allowed": stats['allowed_students'][idx],
                    f"{a}_infected": stats['infected'][idx],
                    f"{a}_reward": stats['rewards'][idx],
                    f"{a}_cross_infections": stats['cross_class_infections'][idx],
                })
            rows.append(row)

        df = pd.DataFrame(rows)
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        df.to_csv(csv_path, index=False)
        print(f"Saved PPO evaluation data to {csv_path}")
        return data

    def plot_rewards(self, save_path="results/ppo_avg_rewards.png"):
        plt.figure(figsize=(10,5))
        for a, rs in self.episode_rewards.items():
            plt.plot(rs, label=f"{a} Rewards")
        plt.plot(self.global_rewards, linestyle='--', label="Global")
        plt.title("Episode Rewards (PPO)")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.legend()
        plt.grid(True)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()

# -----------------------------------------------------------------------------
#  Usage in your main script:
# -----------------------------------------------------------------------------
#
# from agents.ppo_agent import PPOAgent
# agent = PPOAgent(agents, state_dim, action_space_size, seed=42)
# agent.train(env, total_timesteps=200_000)
# eval_data = agent.evaluate(env, max_steps=52,
#                            csv_path="results/ppo_eval.csv")
# agent.plot_rewards("results/ppo_rewards.png")
#
