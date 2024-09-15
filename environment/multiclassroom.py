import numpy as np
from pettingzoo.utils import ParallelEnv
from gym.spaces import Discrete, Box
from environment.simulation import simulate_infections_n_classrooms  # Assuming this is the correct location of the model
import itertools
class MultiClassroomEnv(ParallelEnv):
    def __init__(self, num_classrooms=1, total_students=100, s_shared=10, max_weeks=2, action_levels_per_class=None,
                 alpha=0.005, beta=0.01, gamma=0.2, seed=None):
        self.num_classrooms = num_classrooms
        self.total_students = total_students
        self.s_shared = s_shared
        self.max_weeks = max_weeks
        self.gamma = gamma  # Weighting factor for reward function
        self.current_week = 0
        self.seed(seed)

        # Ensure that action_levels_per_class matches the number of classrooms
        if action_levels_per_class is None or len(action_levels_per_class) != num_classrooms:
            raise ValueError(
                f"The length of action_levels_per_class ({len(action_levels_per_class)}) must match the number of classrooms ({num_classrooms})")

        # Infection rate (alpha) and community transmission rate (beta)
        # Set alpha to 2 for all classrooms and beta to 0.001 for all classrooms
        self.alpha = np.full(self.num_classrooms, alpha)
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
        self.shared_matrix = self._create_shared_matrix(num_classrooms, s_shared)
        self.community_risks = self._generate_episode_risks()

        # Define the state space: all combinations of infected students (0 to 100) and community risk (0 to 1)
        infected_values = range(0, self.total_students + 1, 10)  # Discretize infected students in steps of 10
        community_risk_values = [i / 10 for i in range(11)]  # Discretize community risk in steps of 0.1
        self.state_space = list(itertools.product(infected_values, community_risk_values))

    def seed(self, seed=None):
        np.random.seed(seed)

    def _create_shared_matrix(self, n, s_shared):
        matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                matrix[i][j] = s_shared
                matrix[j][i] = s_shared
        return matrix

    def _generate_episode_risks(self):
        """Generate community risks for each classroom over the weeks."""
        return np.random.uniform(0, 1, (self.num_classrooms, self.max_weeks))

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
        self.student_status = [np.random.randint(0, 5) for _ in range(self.num_classrooms)]  # Random initial infected per classroom
        self.allowed_students = [0] * self.num_classrooms
        self.current_week = 0
        return self._get_observations()

    def step(self, actions):
        # Convert actions to the appropriate allowed_students value
        self.allowed_students = [self._map_action_to_allowed_students(actions[agent], i) for i, agent in enumerate(self.agents)]

        # Ensure that current_week does not exceed max_weeks
        current_week_index = min(self.current_week, self.max_weeks - 1)

        # Simulate infections using the updated model
        self.student_status = simulate_infections_n_classrooms(
            self.num_classrooms,
            self.shared_matrix,  # The matrix governing shared infections
            self.alpha,  # In-class infection rates (set to 2 for all)
            self.beta,  # Community infection rates (set to 0.001 for all)
            self.student_status,  # Current infected students per classroom
            self.allowed_students,  # Allowed students per classroom
            self.community_risks[:, current_week_index]  # Community risk per classroom
        )

        # Calculate rewards based on the number of allowed students and infections
        rewards = {
            f'classroom_{i}': (self.gamma * self.allowed_students[i]) - ((1 - self.gamma) * self.student_status[i])
            for i in range(self.num_classrooms)
        }

        self.current_week += 1
        dones = {f'classroom_{i}': self.current_week >= self.max_weeks for i in range(self.num_classrooms)}
        return self._get_observations(), rewards, dones, {}

    def _map_action_to_allowed_students(self, action, class_index):
        """ Map action to the actual allowed students value based on the classroom action levels """
        action_levels = self.action_levels_per_class[class_index]
        return action * (self.total_students // (action_levels - 1))

    def _get_observations(self):
        # Return observations including only infected students and community risk for each classroom
        return {
            agent: np.array(
                [self.student_status[i], self.community_risks[i][min(self.current_week, self.max_weeks - 1)]])
            for i, agent in enumerate(self.agents)
        }

    def render(self):
        for i, agent in enumerate(self.agents):
            print(f"{agent} - Infected: {self.student_status[i]}, Community Risk: {self.community_risks[i][min(self.current_week, self.max_weeks - 1)]}")
