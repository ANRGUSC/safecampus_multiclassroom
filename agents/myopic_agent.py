import torch
import itertools


class MyopicAgent:
    def __init__(self, agents, reward_mix_alpha=0.5, allowed_values=None):
        self.agents = agents
        self.reward_mix_alpha = reward_mix_alpha
        self.allowed_values = torch.tensor(allowed_values, dtype=torch.float32)

        # Shared parameters for all classrooms
        self.alpha_m = 0.008  # In-class transmission
        self.beta = 0.01  # Community transmission
        self.delta = 0.3  # Cross-classroom transmission rate per shared student

        # Define the fraction of shared students between classes
        self.shared_student_fraction = 0.5  # 20% of students are shared between classes

        self.episode_rewards = {agent: [] for agent in agents}

    def estimate_infected_students(self, agent_idx, current_infected_all, allowed_students_all, community_risk):
        """
        Estimate NEW infections with emphasis on shared students between classrooms.
        This model assumes classrooms have similar capacities (around 100 students).

        Parameters:
        - agent_idx: Index of the current classroom
        - current_infected_all: List of current infected counts for all classrooms
        - allowed_students_all: List of allowed student counts for all classrooms
        - community_risk: External risk factor from the community

        Returns:
        - Estimated number of NEW infected students
        """
        current_inf = current_infected_all[agent_idx]
        allowed = allowed_students_all[agent_idx]

        # If no students are allowed, no infections can occur
        if allowed == 0:
            return 0

        # Within-classroom infections - linear scaling
        in_class_term = self.alpha_m * current_inf * allowed

        # Community risk - quadratic scaling with allowed students
        community_term = self.beta * community_risk * (allowed ** 2)

        # Cross-classroom transmission through shared students
        cross_class_term = 0.0
        for other_idx in range(len(current_infected_all)):
            if other_idx != agent_idx:
                other_allowed = allowed_students_all[other_idx]
                other_infected = current_infected_all[other_idx]

                # Skip if other classroom has no students or infections
                if other_allowed == 0 or other_infected == 0:
                    continue

                # Calculate the number of shared students between classrooms
                shared_students = int(allowed * self.shared_student_fraction)

                infection_probability = other_infected / other_allowed

                # Expected number of newly infected shared students
                infected_shared = self.delta * infection_probability * shared_students

                # Add to the cross-classroom term
                cross_class_term += infected_shared

        # Calculate new infections from all sources
        new_infections = in_class_term + community_term + cross_class_term

        # The estimated infections should never exceed the allowed capacity
        estimated_infections = min(new_infections, allowed)



        return estimated_infections
    def estimate_reward(self, agent_idx, allowed, infected, env_gamma):
        """Calculate reward using the environment's gamma parameter."""
        return env_gamma * allowed - (1 - env_gamma) * infected

    def get_best_joint_actions(self, current_infected_all, community_risk, env_gamma):
        """Find the best joint actions for all agents by evaluating all combinations."""
        num_agents = len(self.agents)
        num_actions = len(self.allowed_values)

        # Generate all possible joint actions
        action_combinations = list(itertools.product(range(num_actions), repeat=num_agents))

        best_reward = float('-inf')
        best_combination = None

        for action_combo in action_combinations:
            allowed_students_all = [self.allowed_values[action].item() for action in action_combo]

            rewards = []
            for i in range(num_agents):
                new_infected = self.estimate_infected_students(
                    i, current_infected_all, allowed_students_all, community_risk)
                reward = self.estimate_reward(i, allowed_students_all[i], new_infected, env_gamma)
                rewards.append(reward)

            global_reward = sum(rewards) / num_agents

            if global_reward > best_reward:
                best_reward = global_reward
                best_combination = action_combo

        return {i: action for i, action in enumerate(best_combination)}

    def get_label(self, infected, community_risk, agent, reward_mix_alpha, env_gamma, recursion=True):
        """
        Determine the best action for a classroom using a mixed objective.
        The agent's objective is a weighted combination of local reward and overall social welfare.
        """
        agent_idx = int(agent.split('_')[1])
        num_agents = len(self.agents)

        # Base Case for other agents (when recursion is disabled)
        if not recursion:
            best_action = None
            best_reward = float('-inf')
            best_allowed = None
            best_infected = None
            current_infected_all = [infected] * num_agents

            for action in range(len(self.allowed_values)):
                allowed = self.allowed_values[action].item()
                test_allowed_students = [self.allowed_values[1].item()] * num_agents
                test_allowed_students[agent_idx] = allowed
                new_infected = self.estimate_infected_students(
                    agent_idx, current_infected_all, test_allowed_students, community_risk)
                reward_val = self.estimate_reward(agent_idx, allowed, new_infected, env_gamma)

                if reward_val > best_reward:
                    best_action = action
                    best_reward = reward_val
                    best_allowed = allowed
                    best_infected = new_infected

            return best_action, best_allowed, best_infected, best_reward

        # Pure Strategies
        if reward_mix_alpha == 0:  # Purely local (selfish) strategy
            best_action = None
            best_reward = float('-inf')
            best_allowed = None
            best_infected = None
            current_infected_all = [infected] * num_agents

            for action in range(len(self.allowed_values)):
                allowed = self.allowed_values[action].item()
                test_allowed_students = [0] * num_agents
                test_allowed_students[agent_idx] = allowed

                for other_idx in range(num_agents):
                    if other_idx != agent_idx:
                        best_other_reward = float('-inf')
                        best_other_action = 0
                        for other_action in range(len(self.allowed_values)):
                            other_allowed = self.allowed_values[other_action].item()
                            test_allowed_students[other_idx] = other_allowed
                            other_infected = self.estimate_infected_students(
                                other_idx, current_infected_all, test_allowed_students, community_risk)
                            other_reward = self.estimate_reward(
                                other_idx, other_allowed, other_infected, env_gamma)
                            if other_reward > best_other_reward:
                                best_other_reward = other_reward
                                best_other_action = other_action
                        test_allowed_students[other_idx] = self.allowed_values[best_other_action].item()

                new_infected = self.estimate_infected_students(
                    agent_idx, current_infected_all, test_allowed_students, community_risk)
                local_reward = self.estimate_reward(agent_idx, allowed, new_infected, env_gamma)

                if local_reward > best_reward:
                    best_action = action
                    best_reward = local_reward
                    best_allowed = allowed
                    best_infected = new_infected

            return best_action, best_allowed, best_infected, best_reward

        elif reward_mix_alpha == 1:  # Purely cooperative strategy
            current_infected_all = [infected] * num_agents
            best_joint_actions = self.get_best_joint_actions(current_infected_all, community_risk, env_gamma)
            best_action = best_joint_actions[agent_idx]
            best_allowed = self.allowed_values[best_action].item()
            joint_allowed = [self.allowed_values[best_joint_actions[i]].item() for i in range(num_agents)]
            best_infected = self.estimate_infected_students(
                agent_idx, current_infected_all, joint_allowed, community_risk)
            best_reward = self.estimate_reward(agent_idx, best_allowed, best_infected, env_gamma)

            return best_action, best_allowed, best_infected, best_reward

        else:  # Mixed Strategy (0 < reward_mix_alpha < 1)
            current_infected_all = [infected] * num_agents

            # Compute cooperative baseline
            baseline_joint = self.get_best_joint_actions(current_infected_all, community_risk, env_gamma)
            baseline_allowed = self.allowed_values[baseline_joint[agent_idx]].item()
            baseline_test = [self.allowed_values[baseline_joint[i]].item() for i in range(num_agents)]
            baseline_infected = self.estimate_infected_students(
                agent_idx, current_infected_all, baseline_test, community_risk)
            baseline_local_reward = self.estimate_reward(agent_idx, baseline_allowed, baseline_infected, env_gamma)
            baseline_global_reward = sum(
                self.estimate_reward(
                    i, baseline_test[i],
                    self.estimate_infected_students(i, current_infected_all, baseline_test, community_risk),
                    env_gamma)
                for i in range(num_agents)
            ) / num_agents

            best_action = None
            best_obj = float('-inf')
            best_allowed = None
            best_infected = None

            # Evaluate each candidate action
            for action in range(len(self.allowed_values)):
                candidate_allowed = self.allowed_values[action].item()
                candidate_test = baseline_test.copy()
                candidate_test[agent_idx] = candidate_allowed

                candidate_infected = self.estimate_infected_students(
                    agent_idx, current_infected_all, candidate_test, community_risk)
                candidate_local_reward = self.estimate_reward(
                    agent_idx, candidate_allowed, candidate_infected, env_gamma)
                candidate_global_reward = sum(
                    self.estimate_reward(
                        i, candidate_test[i],
                        self.estimate_infected_students(i, current_infected_all, candidate_test, community_risk),
                        env_gamma)
                    for i in range(num_agents)
                ) / num_agents

                # Mixed objective using cooperative baseline
                penalty = (1 - reward_mix_alpha) * max(0, baseline_local_reward - candidate_local_reward)
                mixed_obj = reward_mix_alpha * candidate_global_reward + \
                            (1 - reward_mix_alpha) * candidate_local_reward - penalty

                if mixed_obj > best_obj:
                    best_obj = mixed_obj
                    best_action = action
                    best_allowed = candidate_allowed
                    best_infected = candidate_infected

            return best_action, best_allowed, best_infected, best_obj

    def select_best_action(self, agent, state, env_gamma):
        """Select best action for the given state."""
        infected, community_risk = state
        best_action, _, _, _ = self.get_label(
            infected, community_risk, agent, self.reward_mix_alpha, env_gamma)
        return best_action

    def evaluate(self, env, num_episodes=1, max_steps=30):
        """
        Evaluate the myopic policy for a single episode.
        Assumes env was initialized with eval_mode=True to use the external community risk data.
        """
        states = env.reset()
        episode_rewards = {agent: 0 for agent in self.agents}
        evaluation_data = {
            'steps': [],
            'agents': {},
        }
        for agent in self.agents:
            evaluation_data['agents'][agent] = {
                'rewards': [],
                'actions': [],
                'infected': [],
                'allowed_students': [],
                'cross_class_infections': []
            }
        for step in range(max_steps):
            actions = {}
            for agent in self.agents:
                actions[agent] = self.select_best_action(agent, states[agent], env.gamma)
                evaluation_data['agents'][agent]['infected'].append(states[agent][0])
                evaluation_data['agents'][agent]['actions'].append(actions[agent])
                evaluation_data['agents'][agent]['allowed_students'].append(
                    self.allowed_values[actions[agent]].item()
                )
            next_states, rewards, dones, _ = env.step(actions)
            allowed_students_all = [self.allowed_values[actions[a]].item() for a in self.agents]
            current_infected_all = [states[a][0] for a in self.agents]
            for i, agent in enumerate(self.agents):
                cross_class_term = 0.0
                for other_idx in range(len(current_infected_all)):
                    if other_idx != i:
                        other_allowed = allowed_students_all[other_idx]
                        other_infected = current_infected_all[other_idx]
                        if other_allowed == 0 or other_infected == 0:
                            continue
                        shared_students = int(allowed_students_all[i] * self.shared_student_fraction)
                        infection_probability = other_infected / other_allowed
                        infected_shared = self.delta * infection_probability * shared_students
                        cross_class_term += infected_shared
                evaluation_data['agents'][agent]['cross_class_infections'].append(cross_class_term)
                episode_rewards[agent] += rewards[agent]
                evaluation_data['agents'][agent]['rewards'].append(rewards[agent])
            evaluation_data['steps'].append(step)
            states = next_states
            if all(dones.values()):
                break
        evaluation_data['total_rewards'] = episode_rewards
        for agent in self.agents:
            self.episode_rewards[agent].append(episode_rewards[agent])
        print("\nMyopic Agent Evaluation Results:")
        for agent, reward in episode_rewards.items():
            print(f"{agent}: {reward:.2f}")
        return evaluation_data




