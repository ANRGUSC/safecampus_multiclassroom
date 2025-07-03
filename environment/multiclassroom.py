import numpy as np
from pettingzoo.utils import ParallelEnv
from gym.spaces import Discrete, Box
from environment.simulation import simulate_infections_n_classrooms  # Your simulation function
import itertools
import random
import os


class MultiClassroomEnv(ParallelEnv):
    def __init__(self, num_classrooms=1, total_students=100, max_weeks=2, action_levels_per_class=None,
                 alpha=0.008, beta=0.01, phi=0.3, gamma=0.3, seed=None,
                 community_risk_data_file=None):
        """
        Parameters:
          community_risk_data_file (str): Path to the CSV file containing community risk data.
                                          If provided, CSV data is used when the environment is in evaluation mode.
          seed (int): Random seed to ensure reproducibility.
        """
        self.num_classrooms = num_classrooms
        self.total_students = total_students
        self.max_weeks = max_weeks
        self.phi = phi  # Cross-classroom transmission rate
        self.gamma = gamma  # Balance weight between allowed students and infections
        self.current_week = 0

        # Store the seed value for reuse
        self.seed_value = seed
        self.seed(seed)  # Set both np and random seeds
        self.rng = np.random.default_rng(seed)

        self.episode_seed = 0
        self.metadata = {'render.modes': ['human']}

        # Infection and community transmission rates.
        self.alpha_m = np.full(self.num_classrooms, alpha)
        self.beta = np.full(self.num_classrooms, beta)

        self.action_levels_per_class = action_levels_per_class
        self.action_levels = {
            i: np.linspace(0, self.total_students, levels).astype(int).tolist()
            for i, levels in enumerate(action_levels_per_class)
        }

        # Define agents and gym spaces.
        self.agents = [f'classroom_{i}' for i in range(num_classrooms)]
        self.possible_agents = self.agents[:]
        self.action_spaces = {agent: Discrete(len(self.action_levels[i]))
                              for i, agent in enumerate(self.agents)}
        self.observation_spaces = {
            agent: Box(low=np.array([0, 0]), high=np.array([self.total_students, 1]), dtype=np.float32)
            for agent in self.agents
        }

        self.student_status = [0] * num_classrooms  # Current infected counts per classroom.
        self.allowed_students = [0] * num_classrooms

        # Load CSV data if provided.
        self.csv_risk_data = None
        if community_risk_data_file is not None:
            if not os.path.exists(community_risk_data_file):
                raise ValueError(f"Community risk data file not found: {community_risk_data_file}")
            # Load only the "Risk-Level" column (the second column) as floats.
            self.csv_risk_data = np.genfromtxt(
                community_risk_data_file, delimiter=',', skip_header=1, usecols=1, dtype=float
            )
            if len(self.csv_risk_data) < self.max_weeks:
                raise ValueError("Not enough community risk data for the evaluation period.")

        # Generate shared risk data for training.
        self.shared_community_risk = self._generate_shared_episode_risk()

        # Default mode: training (use generated risk).
        self.eval_mode = False

        # Define a discretized state space.
        infected_values = range(0, self.total_students + 1, 10)
        community_risk_values = [i / 10 for i in range(11)]
        self.state_space = list(itertools.product(infected_values, community_risk_values))

    def seed(self, seed=None):
        """Set the seed for both numpy and random."""
        if seed is not None:
            self.seed_value = seed
            np.random.seed(seed)
            random.seed(seed)

    def set_mode(self, eval_mode: bool):
        """
        Switch the environment mode.
          - eval_mode=False: training mode (use generated risk).
          - eval_mode=True: evaluation mode (use CSV risk data).
        """
        self.eval_mode = eval_mode

    def _generate_shared_episode_risk(self):
        """Generate a risk pattern for the episode (used during training)."""
        self.episode_seed += 1
        # Use the dedicated generator with a fixed offset.
        local_rng = np.random.default_rng(self.seed_value + self.episode_seed)
        t = np.linspace(0, 2 * np.pi, self.max_weeks)
        risk_pattern = np.zeros(self.max_weeks)
        num_components = local_rng.integers(1, 4)  # Random integer in [1, 3]
        for _ in range(num_components):
            amplitude = local_rng.uniform(0.2, 0.4)
            frequency = local_rng.uniform(0.5, 2.0)
            phase = local_rng.uniform(0, 2 * np.pi)
            risk_pattern += amplitude * np.sin(frequency * t + phase)
        # Normalize and clip to [0.1, 1.0].
        risk_pattern = (risk_pattern - risk_pattern.min()) / (risk_pattern.max() - risk_pattern.min())
        risk_pattern = 0.9 * risk_pattern + 0.1  # Scale to range [0.1, 1.0]
        # Add some noise.
        risk_pattern = [max(0.1, min(1.0, risk + local_rng.uniform(-0.1, 0.1))) for risk in risk_pattern]
        return risk_pattern

    def _get_risk(self):
        """Return the current community risk based on the mode."""
        current_week_index = min(self.current_week, self.max_weeks - 1)
        if self.eval_mode and self.csv_risk_data is not None:
            return float(self.csv_risk_data[current_week_index])
        else:
            return float(self.shared_community_risk[current_week_index])

    def _get_observations(self):
        risk = self._get_risk()
        return {
            agent: np.array([self.student_status[i], risk])
            for i, agent in enumerate(self.agents)
        }

    def step(self, actions):
        self.allowed_students = [
            self._map_action_to_allowed_students(actions[agent], i)
            for i, agent in enumerate(self.agents)
        ]
        risk = self._get_risk()
        self.student_status = simulate_infections_n_classrooms(
            self.num_classrooms,
            self.alpha_m,
            self.beta,
            self.phi,
            self.student_status,
            self.allowed_students,
            [risk] * self.num_classrooms
        )
        rewards = {}
        for i, agent in enumerate(self.agents):
            allowed = self.allowed_students[i]
            infected = self.student_status[i]
            rewards[agent] = self.gamma * allowed - (1 - self.gamma) * infected
        self.current_week += 1
        dones = {agent: self.current_week >= self.max_weeks for agent in self.agents}
        return self._get_observations(), rewards, dones, {}

    def _map_action_to_allowed_students(self, action, class_index):
        action_levels = self.action_levels_per_class[class_index]
        return action * (self.total_students // (action_levels - 1))

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed(seed)
        # Reinitialize student status and allowed students.
        self.student_status = [np.random.randint(1, 99) for _ in range(self.num_classrooms)]
        self.allowed_students = [0] * self.num_classrooms
        self.current_week = 0
        # In training mode, regenerate the shared risk pattern.
        if not self.eval_mode:
            self.shared_community_risk = self._generate_shared_episode_risk()
        return self._get_observations()

    def render(self):
        risk = self._get_risk()
        for i, agent in enumerate(self.agents):
            print(f"{agent} - Infected: {self.student_status[i]}, Community Risk: {risk}")
