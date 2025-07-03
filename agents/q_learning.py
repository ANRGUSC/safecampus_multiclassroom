import numpy as np
import random
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import itertools

SEED = 42
np.random.seed(SEED)

class TabularQLearningAgent:
    def __init__(self, agents, total_students, state_space=None,
                 action_levels=None, learning_rate=0.05, gamma=0.99,
                 epsilon=1.0, epsilon_decay=0.995, min_epsilon=1e-4,
                 max_episodes=1000, target_avg_frac=0.9):
        self.agents = agents
        self.num_agents = len(agents)
        self.total_students = total_students
        self.gamma = gamma
        self.alpha = learning_rate
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.max_episodes = max_episodes
        # Define state space
        if state_space is None:
            infected_vals = range(0, total_students+1, 10)
            risk_vals = [i/100 for i in range(0,101,10)]
            self.state_space = list(itertools.product(infected_vals, risk_vals))
        else:
            self.state_space = state_space
        self.S = len(self.state_space)
        # Define action space per agent from levels
        if action_levels is None:
            discrete = np.linspace(0, total_students, 3, dtype=int)
            self.action_list = {i: discrete for i in range(self.num_agents)}
        else:
            self.action_list = action_levels
        self.A = {i: len(self.action_list[i]) for i in self.action_list}
        # Initialize Q-tables for each agent
        self.Q = {i: np.zeros((self.S, self.A[i])) for i in range(self.num_agents)}
        # Logging
        self.global_rewards = []
        # Convergence threshold
        max_per_agent = gamma * total_students * 30
        max_global = self.num_agents * max_per_agent * 30
        self.target_avg = target_avg_frac * max_global

    def get_state_idx(self, state):
        infected, risk = state
        di = max(0, min(self.total_students, round(infected/10)*10))
        dr = max(0, min(1.0, round(risk*10)/10))
        return self.state_space.index((di, dr))

    def select_action(self, agent_idx, state):
        s_idx = self.get_state_idx(state)
        if random.random() < self.epsilon:
            return random.randrange(self.A[agent_idx])
        return int(np.argmax(self.Q[agent_idx][s_idx]))

    def train_td(self, env, max_steps=30):
        """TD(0) training mirroring DQN’s ε-decay and convergence."""
        for ep in range(1, self.max_episodes+1):
            # decay ε at start of episode exactly as DQN
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
            states = env.reset()
            total_r = 0.0
            for t in range(max_steps):
                actions = {ag: self.select_action(i, states[ag]) for i, ag in enumerate(self.agents)}
                next_states, rewards, dones, _ = env.step(actions)
                for i, ag in enumerate(self.agents):
                    s_idx = self.get_state_idx(states[ag])
                    a = actions[ag]
                    r = rewards[ag]
                    ns_idx = self.get_state_idx(next_states[ag])
                    best_next = np.max(self.Q[i][ns_idx])
                    td_target = r + self.gamma * best_next
                    td_error = td_target - self.Q[i][s_idx][a]
                    self.Q[i][s_idx][a] += self.alpha * td_error
                    total_r += r
                states = next_states
                if all(dones.values()):
                    break
            self.global_rewards.append(total_r)
            # check convergence over last 20 episodes
            window = min(20, len(self.global_rewards))
            avg_r = sum(self.global_rewards[-window:]) / window
            if avg_r >= self.target_avg:
                print(f"Converged at episode {ep}, avg last {window} = {avg_r:.2f}")
                break
        # save learning curve
        os.makedirs('results', exist_ok=True)
        plt.plot(self.global_rewards)
        plt.title('Global Reward (Tabular TD0)')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.savefig('results/tabular_td0_rewards.png')
        plt.close()

    def train(self, env, max_steps=30):
        """First-visit Monte Carlo training with the same ε-decay and convergence check."""
        for ep in range(1, self.max_episodes + 1):
            # decay ε at start of episode exactly as DQN
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

            # 1) Generate one full episode trajectory
            states = env.reset()
            trajectory = []  # list of tuples (agent_idx, state_idx, action, reward)
            total_r = 0.0

            for t in range(max_steps):
                actions = {ag: self.select_action(i, states[ag])
                           for i, ag in enumerate(self.agents)}
                next_states, rewards, dones, _ = env.step(actions)

                # record each agent’s transition
                for i, ag in enumerate(self.agents):
                    s_idx = self.get_state_idx(states[ag])
                    a = actions[ag]
                    r = rewards[ag]
                    trajectory.append((i, s_idx, a, r))
                    total_r += r

                states = next_states
                if all(dones.values()):
                    break

            self.global_rewards.append(total_r)

            # 2) Compute returns G_t backward
            returns = []
            G = {i: 0.0 for i in range(self.num_agents)}
            # We’ll do per-agent accumulation though here they all share the same indexing
            for (i, s_idx, a, r) in reversed(trajectory):
                G[i] = r + self.gamma * G[i]
                returns.insert(0, (i, s_idx, a, G[i]))

            # 3) First-visit MC updates
            visited = set()
            for (i, s_idx, a, Gt) in returns:
                if (i, s_idx, a) in visited:
                    continue
                visited.add((i, s_idx, a))
                # update Q toward full return
                self.Q[i][s_idx][a] += self.alpha * (Gt - self.Q[i][s_idx][a])

            # 4) Convergence check over last 20 episodes
            window = min(20, len(self.global_rewards))
            avg_r = sum(self.global_rewards[-window:]) / window
            if avg_r >= self.target_avg:
                print(f"Converged at episode {ep}, avg last {window} = {avg_r:.2f}")
                break

        # save learning curve
        os.makedirs('results', exist_ok=True)
        plt.plot(self.global_rewards)
        plt.title('Global Reward (Tabular MC)')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.savefig('results/tabular_mc_rewards.png')
        plt.close()

    def evaluate(self, env, max_steps=30):
        """
        Evaluate in same format as DQN:
          returns dict with 'steps' and per-agent subdicts containing lists of:
            'actions', 'allowed_students', 'infected', 'cross_class_infections', 'rewards'
        and 'total_rewards'.
        """
        states = env.reset()
        data = {'steps': [], 'agents': {ag: {
            'actions': [], 'allowed_students': [], 'infected': [],
            'cross_class_infections': [], 'rewards': []
        } for ag in self.agents}}

        for step in range(max_steps):
            # select greedy actions
            actions = {}
            for i, ag in enumerate(self.agents):
                s_idx = self.get_state_idx(states[ag])
                actions[ag] = int(np.argmax(self.Q[i][s_idx]))
                # log action & state
                data['agents'][ag]['actions'].append(actions[ag])
                data['agents'][ag]['infected'].append(states[ag][0])
            # map to allowed values
            for i, ag in enumerate(self.agents):
                allowed_list = env.action_levels[i]
                act = actions[ag]
                data['agents'][ag]['allowed_students'].append(allowed_list[act])
            # step env
            next_states, rewards, dones, _ = env.step(actions)
            # log rewards
            for ag in self.agents:
                data['agents'][ag]['rewards'].append(rewards[ag])
            # compute cross-class infections
            allowed_vals = [data['agents'][ag]['allowed_students'][-1] for ag in self.agents]
            infected_vals = [s[0] for s in states.values()]
            for i, ag in enumerate(self.agents):
                cross = 0.0
                for j in range(self.num_agents):
                    if j!=i and allowed_vals[j]>0:
                        cross += (infected_vals[j]/allowed_vals[j]) * env.phi
                data['agents'][ag]['cross_class_infections'].append(cross)

            data['steps'].append(step)
            states = next_states
            if all(dones.values()): break

        # total rewards
        total = {ag: sum(data['agents'][ag]['rewards']) for ag in self.agents}
        data['total_rewards'] = total
        return data
