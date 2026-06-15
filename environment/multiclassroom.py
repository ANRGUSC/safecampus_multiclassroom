import numpy as np
from pettingzoo.utils import ParallelEnv
from gym.spaces import Discrete, Box
from environment.simulation import simulate_infections_n_classrooms
import itertools
import random
import os


class MultiClassroomEnv(ParallelEnv):
    def __init__(self, num_classrooms=1, total_students=100, max_weeks=15,
                 action_levels_per_class=None, continuous_action=False,
                 alpha=0.008, beta=0.01, shared_fraction=0.3, gamma=0.5, seed=None,
                 community_risk_data_file=None, eval_mode=False,
                 cooperative_reward=True):
        """
        Parameters:
          num_classrooms (int): Number of classrooms/agents
          total_students (int): Total students per classroom
          max_weeks (int): Episode length
          action_levels_per_class (list): Number of discrete action levels per classroom
          continuous_action (bool): If True, actions are continuous [0, total_students]
          alpha (float): Infection rate parameter
          beta (float): Recovery rate parameter
          shared_fraction (float): Fraction of students shared across classrooms
                                   (the cross-classroom coupling strength)
          gamma (float): Balance weight between allowed students and infections (omega)
          seed (int): Random seed
          community_risk_data_file (str): Path to risk data CSV (optional)
          eval_mode (bool): If True, use deterministic risk pattern
          cooperative_reward (bool): If True, all agents receive the average reward
        """
        self.num_classrooms = num_classrooms
        self.total_students = total_students
        self.max_weeks = max_weeks
        self.shared_fraction = shared_fraction
        self.gamma = gamma
        self.current_week = 0
        self.continuous_action = continuous_action
        self.eval_mode = eval_mode
        self.cooperative_reward = cooperative_reward

        # Parameter setup (alpha, beta, etc.)
        self.alpha_m = [alpha] * num_classrooms
        self.beta = [beta] * num_classrooms

        self.agents = [f"classroom_{i}" for i in range(self.num_classrooms)]
        self.possible_agents = self.agents[:]

        # Spaces
        if self.continuous_action:
            self.action_spaces = {
                agent: Box(low=0.0, high=float(self.total_students), shape=(1,), dtype=np.float32)
                for agent in self.agents
            }
        else:
            # Discrete actions
            if action_levels_per_class is None:
                action_levels_per_class = [11] * num_classrooms
            self.action_levels = action_levels_per_class
            self.action_spaces = {
                agent: Discrete(self.action_levels[i])
                for i, agent in enumerate(self.agents)
            }
            # Precompute discrete action values for each agent
            # Maps action index to actual allowed students value
            self.discrete_action_values = {
                agent: np.linspace(0, self.total_students, self.action_levels[i])
                for i, agent in enumerate(self.agents)
            }

        # Observation Space: [Current Infected, Community Risk]
        self.observation_spaces = {
            agent: Box(low=0.0, high=float(self.total_students), shape=(2,), dtype=np.float32)
            for agent in self.agents
        }

        # State initialization
        self.student_status = [0] * self.num_classrooms
        self.allowed_students = [0] * self.num_classrooms

        # Risk Data Loading: parse a weekly community-risk series from CSV.
        # Expected columns: a 'Risk-Level' column (0-1); falls back to the last
        # numeric column. Stored as a list of floats; applied only when passed to
        # reset() via risk_override (kept out of the training path).
        self.community_risk_data = []
        if community_risk_data_file and os.path.exists(community_risk_data_file):
            import csv as _csv
            try:
                with open(community_risk_data_file, 'r') as f:
                    rows = list(_csv.DictReader(f))
                if rows:
                    fields = rows[0].keys()
                    risk_col = next((c for c in fields if 'risk' in c.lower()), None)
                    if risk_col is None:
                        risk_col = list(fields)[-1]
                    self.community_risk_data = [float(r[risk_col]) for r in rows]
            except (ValueError, KeyError, OSError):
                self.community_risk_data = []

        if seed is not None:
            self.seed(seed)

    def seed(self, seed=None):
        np.random.seed(seed)
        random.seed(seed)

    def _generate_risk_pattern(self, episode_seed):
        # Local RNG for risk pattern generation ensures consistency per episode index
        rng = np.random.default_rng(episode_seed)

        t = np.linspace(0, 2 * np.pi, self.max_weeks)
        risk = np.zeros(self.max_weeks)

        num_components = rng.integers(1, 4)
        for _ in range(num_components):
            amp = rng.uniform(0.2, 0.4)
            freq = rng.uniform(0.5, 2.0)
            phase = rng.uniform(0, 2 * np.pi)
            risk += amp * np.sin(freq * t + phase)

        # normalize to exactly [0.0, 1.0]
        r_min, r_max = risk.min(), risk.max()
        if r_max - r_min > 1e-6:
            risk = (risk - r_min) / (r_max - r_min)
        else:
            risk = np.zeros_like(risk)

        return risk.tolist()

    def set_mode(self, eval_mode):
        self.eval_mode = eval_mode

    def _get_observations(self):
        obs = {}
        curr_risk = 0.0
        if self.eval_mode:
            # Fixed pattern for evaluation
            curr_risk = min(1.0, self.current_week / 20.0)
        else:
            if hasattr(self, 'shared_community_risk'):
                curr_risk = self.shared_community_risk[min(self.current_week, len(self.shared_community_risk) - 1)]
            else:
                curr_risk = np.random.random()

        for i, agent in enumerate(self.agents):
            obs[agent] = np.array([float(self.student_status[i]), float(curr_risk)], dtype=np.float32)
        return obs

    def step(self, actions):
        # Parse actions
        if self.continuous_action:
            for i, agent in enumerate(self.agents):
                act = actions[agent]
                # If array, extract val
                if isinstance(act, (np.ndarray, list)):
                    act = act[0]
                self.allowed_students[i] = float(act)
        else:
            # Discrete action mapping: action index -> allowed students value
            for i, agent in enumerate(self.agents):
                action_idx = actions[agent]
                # Handle if action is array
                if isinstance(action_idx, (np.ndarray, list)):
                    action_idx = int(action_idx[0])
                else:
                    action_idx = int(action_idx)
                # Map to actual value
                self.allowed_students[i] = self.discrete_action_values[agent][action_idx]

        # Get current risk
        risk = 0.0
        if self.eval_mode:
            risk = min(1.0, self.current_week / 20.0)
        else:
            risk = self.shared_community_risk[min(self.current_week, len(self.shared_community_risk) - 1)]

        # Simulate Dynamics
        self.student_status = simulate_infections_n_classrooms(
            self.num_classrooms,
            self.alpha_m,
            self.beta,
            self.student_status,
            self.allowed_students,
            [risk] * self.num_classrooms,
            shared_student_fraction=self.shared_fraction
        )

        # --- REWARD CALCULATION ---
        individual_rewards = []
        for i, agent in enumerate(self.agents):
            allowed = self.allowed_students[i]
            infected = self.student_status[i]

            # Base individual reward
            r_i = self.gamma * allowed - (1 - self.gamma) * infected
            individual_rewards.append(r_i)

        rewards = {}
        if self.cooperative_reward:
            # Cooperative: Everyone gets the Mean Reward
            avg_reward = np.mean(individual_rewards)
            for agent in self.agents:
                rewards[agent] = avg_reward
        else:
            # Competitive/Individual
            for i, agent in enumerate(self.agents):
                rewards[agent] = individual_rewards[i]

        self.current_week += 1
        dones = {agent: self.current_week >= self.max_weeks for agent in self.agents}

        return self._get_observations(), rewards, dones, {}

    def reset(self, seed=None, options=None, risk_override=None):
        """Reset the episode.

        risk_override: optional explicit weekly community-risk vector. When given,
        the initial infections are still randomized from `seed`, but the risk
        trajectory is fixed to this vector (used to evaluate on a held-out / real
        risk series while still sampling initial conditions). Requires
        eval_mode=False so the env reads `shared_community_risk`.
        """
        if seed is not None:
            self.seed(seed)

        self.student_status = [np.random.randint(0, 5) for _ in range(self.num_classrooms)]
        self.allowed_students = [0] * self.num_classrooms
        self.current_week = 0

        if risk_override is not None:
            self.shared_community_risk = [float(r) for r in risk_override]
        elif not self.eval_mode:
            # Reproducible per-episode risk when a seed is supplied; random otherwise
            episode_seed = seed if seed is not None else int(np.random.randint(0, 2 ** 31 - 1))
            self.shared_community_risk = self._generate_risk_pattern(episode_seed)

        return self._get_observations()

    def render(self):
        pass