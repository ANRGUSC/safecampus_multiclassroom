import numpy as np
from pettingzoo.utils import ParallelEnv
from gym.spaces import Discrete, Box
from environment.simulation import simulate_infections_n_classrooms  # Assuming this is the correct location of the model
import itertools
import random



class MultiClassroomEnv(ParallelEnv):
    def __init__(self, num_classrooms=1, total_students=100, max_weeks=2, action_levels_per_class=None,
                 alpha=0.01, beta=0.001, phi=0.0001, gamma=0.3, seed=None):
        self.num_classrooms = num_classrooms
        self.total_students = total_students
        self.max_weeks = max_weeks
        self.phi = phi  # Cross-classroom transmission rate
        self.gamma = gamma  # Weight for balancing allowed students and infections
        self.current_week = 0
        self.seed(seed)
        self.episode_seed = 0

        # Infection rate (alpha) and community transmission rate (beta)
        self.alpha_m = np.full(self.num_classrooms, alpha)
        self.beta = np.full(self.num_classrooms, beta)

        self.action_levels_per_class = action_levels_per_class

        # Map actions to allowed students (arbitrary levels for each class)
        self.action_levels = {i: np.linspace(0, self.total_students, levels).astype(int).tolist() for i, levels in
                              enumerate(action_levels_per_class)}

        # Gym spaces: The actions are discrete, but observations are continuous (infected, community risk)
        self.agents = [f'classroom_{i}' for i in range(num_classrooms)]
        self.possible_agents = self.agents[:]

        # Action space: Discrete choices for number of students allowed in the classroom
        self.action_spaces = {agent: Discrete(len(self.action_levels[i])) for i, agent in enumerate(self.agents)}

        # Observation space: Continuous values for infected students (0-100) and community risk (0-1)
        self.observation_spaces = {
            agent: Box(low=np.array([0, 0]), high=np.array([self.total_students, 1]), dtype=np.float32)
            for agent in self.agents}

        # Initialize infection and community risk states
        self.student_status = [0] * num_classrooms  # Number of currently infected students per classroom
        self.allowed_students = [0] * num_classrooms
        self.community_risks = self._generate_episode_risks()
        self.shared_community_risk = self._generate_shared_episode_risk()


        # Define the state space: all combinations of infected students (0 to 100) and community risk (0 to 1)
        infected_values = range(0, self.total_students + 1, 10)  # Discretize infected students in steps of 10
        community_risk_values = [i / 10 for i in range(11)]  # Discretize community risk in steps of 0.1
        self.state_space = list(itertools.product(infected_values, community_risk_values))

    def seed(self, seed=None):
        np.random.seed(seed)

    def _generate_shared_episode_risk(self):
        """Generate a single community risk pattern for all classrooms over the weeks."""
        self.episode_seed += 1
        random.seed(self.episode_seed)
        np.random.seed(self.episode_seed)

        t = np.linspace(0, 2 * np.pi, self.max_weeks)
        risk_pattern = np.zeros(self.max_weeks)

        num_components = random.randint(1, 3)  # Use 1 to 3 sine components

        # Generate the sine wave-based risk pattern
        for _ in range(num_components):
            amplitude = random.uniform(0.2, 0.4)
            frequency = random.uniform(0.5, 2.0)
            phase = random.uniform(0, 2 * np.pi)
            risk_pattern += amplitude * np.sin(frequency * t + phase)

        # Normalize and scale the risk pattern to range [0.0, 0.9]
        risk_pattern = (risk_pattern - np.min(risk_pattern)) / (np.max(risk_pattern) - np.min(risk_pattern))
        risk_pattern = 1.0 * risk_pattern + 0.0  # Scale to range [0.0, 0.9]

        # Add some noise and clamp the values between 0.0 and 1.0
        risk_pattern = [max(0.1, min(1.0, risk + random.uniform(-0.1, 0.1))) for risk in risk_pattern]

        return risk_pattern

    def _generate_episode_risks(self):
        """Generate unique community risks for each classroom over the weeks."""
        self.episode_seed += 1
        random.seed(self.episode_seed)
        np.random.seed(self.episode_seed)

        t = np.linspace(0, 2 * np.pi, self.max_weeks)
        risk_patterns = np.zeros((self.num_classrooms, self.max_weeks))

        for classroom in range(self.num_classrooms):
            num_components = random.randint(1, 3)  # Use 1 to 3 sine components
            risk_pattern = np.zeros(self.max_weeks)

            # Generate the sine wave-based risk pattern
            for _ in range(num_components):
                amplitude = random.uniform(0.2, 0.4)
                frequency = random.uniform(0.5, 2.0)
                phase = random.uniform(0, 2 * np.pi)
                risk_pattern += amplitude * np.sin(frequency * t + phase)

            # Normalize and scale the risk pattern to range [0.0, 0.9]
            risk_pattern = (risk_pattern - np.min(risk_pattern)) / (np.max(risk_pattern) - np.min(risk_pattern))
            risk_pattern = 1.0 * risk_pattern + 0.0  # Scale to range [0.0, 0.9]

            # Add some noise and clamp the values between 0.0 and 1.0
            risk_patterns[classroom] = [max(0.1, min(1.0, risk + random.uniform(-0.1, 0.1))) for risk in risk_pattern]

        return risk_patterns

    def get_state_index(self, state):
        """
        Convert a continuous state into the corresponding index in the discretized state space.
        The state is a tuple (infected, community_risk), and it will be mapped to the nearest discretized state.
        """
        infected, community_risk = state

        # Discretize infected (in steps of 10) and community_risk (in steps of 0.1)
        discrete_infected = int(round(infected / 10) * 10)
        discrete_community_risk = round(community_risk * 10) / 10

        # Find the corresponding index in the state space
        state_tuple = (discrete_infected, discrete_community_risk)

        if state_tuple in self.state_space:
            return self.state_space.index(state_tuple)
        else:
            raise ValueError(f"State {state_tuple} not found in the state space.")

    def reset(self):
        self.student_status = [np.random.randint(0, 50) for _ in range(self.num_classrooms)]  # Random initial infected per classroom
        self.allowed_students = [0] * self.num_classrooms
        self.current_week = 0
        return self._get_observations()

    def step(self, actions):
        # Convert actions to the appropriate allowed_students value
        self.allowed_students = [self._map_action_to_allowed_students(actions[agent], i) for i, agent in
                                 enumerate(self.agents)]

        # Ensure that current_week does not exceed max_weeks
        current_week_index = min(self.current_week, self.max_weeks - 1)
        community_risk = self.shared_community_risk[current_week_index]

        # Simulate infections using the updated model
        self.student_status = simulate_infections_n_classrooms(
            self.num_classrooms,
            self.alpha_m,  # In-class infection rates
            self.beta,  # Community infection rates
            self.phi,  # Cross-classroom transmission rate
            self.student_status,  # Current infected students per classroom
            self.allowed_students,  # Allowed students per classroom
            [community_risk] * self.num_classrooms  # Shared community risk for all classrooms
            # self.community_risks[:, current_week_index]  # Community risk per classroom

        )

        # Calculate rewards for each classroom
        rewards = {}
        for i, agent in enumerate(self.agents):
            allowed_students = self.allowed_students[i]
            infected_students = self.student_status[i]
            # Calculate the reward using the fairness scheme
            # reward = (self.gamma * allowed_students) - ((1 - self.gamma) * infected_students)
            # Reward = balancing allowed students vs penalty on infections (quadratic)
            reward = self.gamma * allowed_students - (1-self.gamma) * infected_students
            rewards[agent] = reward

        # Increment the week
        self.current_week += 1

        # Check if the simulation is done (end of weeks)
        dones = {agent: self.current_week >= self.max_weeks for agent in self.agents}

        # Return observations, rewards, and done flags
        return self._get_observations(), rewards, dones, {}

    def _map_action_to_allowed_students(self, action, class_index):
        """ Map action to the actual allowed students value based on the classroom action levels """
        action_levels = self.action_levels_per_class[class_index]
        return action * (self.total_students // (action_levels - 1))

    # def _get_observations(self):
    #     # Return observations including only infected students and unique community risk for each classroom
    #     return {
    #         agent: np.array(
    #             [self.student_status[i], self.community_risks[i][min(self.current_week, self.max_weeks - 1)]])
    #         for i, agent in enumerate(self.agents)
    #     }

    def _get_observations(self):
        # Return observations including only infected students and shared community risk for each classroom
        current_week_index = min(self.current_week, self.max_weeks - 1)
        community_risk = self.shared_community_risk[current_week_index]
        return {
            agent: np.array([self.student_status[i], community_risk])
            for i, agent in enumerate(self.agents)
        }

    def render(self):
        for i, agent in enumerate(self.agents):
            print(f"{agent} - Infected: {self.student_status[i]}, Community Risk: {self.community_risks[i][min(self.current_week, self.max_weeks - 1)]}")
