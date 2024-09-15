import os
import numpy as np
import torch
import random
import csv
import pandas as pd
from utils.visualization import visualize_myopic_states as visualize_myopic_agents
import matplotlib.pyplot as plt
import colorsys

# Global SEED variable for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

class MyopicPolicy:
    def __init__(self, env):
        self.save_path = "myopic_results"
        os.makedirs(self.save_path, exist_ok=True)
        self.env = env  # Reference to the environment

        # Initialize the allowed values directly from the environment's action space for each classroom
        self.allowed_values_per_agent = {
            agent: torch.tensor([env._map_action_to_allowed_students(action, i)
                                 for action in range(env.action_spaces[agent].n)])
            for i, agent in enumerate(env.agents)
        }

    def get_label(self, num_infected, community_risk, agent, alpha):
        """
        Evaluate all possible actions (allowed student levels) and return the action that maximizes the reward.

        Args:
            num_infected (torch.Tensor): The number of currently infected students.
            community_risk (torch.Tensor): The community risk value.
            agent (str): The agent (classroom) to evaluate.
            alpha (float): Weighting factor for reward calculation.

        Returns:
            label (int): The best action (index of the action) based on the reward.
            allowed_value (float): The corresponding allowed students for the best action.
            new_infected (float): The resulting number of new infections.
            reward (float): The maximum reward.
        """
        allowed_values = torch.FloatTensor(self.env.action_levels[self.env.agents.index(agent)])
        num_actions = len(allowed_values)

        # Initialize variables to store the best action
        label = None
        max_reward = -float('inf')
        best_allowed_value = None
        best_new_infected = None

        # Iterate over all possible actions
        for i, allowed_value in enumerate(allowed_values):
            # Ensure actions dictionary has all agents
            actions = {a: i for a in self.env.agents}  # Action for all agents (even if only one is evaluated at a time)

            # Take a step in the environment using the actions for all agents
            next_obs, rewards, dones, _ = self.env.step(actions)

            # Get the reward for the current agent
            reward = rewards[agent]

            # Update the best action if the current reward is better
            if reward > max_reward:
                max_reward = reward
                label = i
                best_allowed_value = allowed_value.item()
                best_new_infected = next_obs[agent][0]  # The first element of observation is infected students

        return label, best_allowed_value, best_new_infected, max_reward

    def evaluate(self, env, run_name, num_episodes=1, alpha=0.5, csv_path=None):
        """
        Evaluate the myopic policy across all possible combinations of infected students and community risk.
        """
        total_rewards = []
        allowed_values_over_time = []
        infected_values_over_time = []

        # Define continuous ranges for infected students and community risk
        infected_range = np.linspace(0, env.total_students, 100)  # Infected students from 0 to 100 in 100 steps
        community_risk_range = np.linspace(0, 1, 100)  # Community risk from 0 to 1 in 100 steps

        # Iterate through all episodes (in this case we can only use one, since it's immediate reward-based)
        for episode in range(num_episodes):
            total_reward = 0

            # Iterate through all possible combinations of infected and community risk
            for infected in infected_range:
                for community_risk in community_risk_range:
                    actions = {}
                    obs = {}

                    # Create observations for all agents based on infected and community risk
                    for agent in env.agents:
                        obs[agent] = [infected, community_risk * 100]  # Normalize community risk for the environment

                    # Iterate over agents to evaluate all possible actions and select the best one
                    for agent in env.agents:
                        num_infected = torch.FloatTensor([obs[agent][0]])  # Get infected value for the agent
                        community_risk_value = torch.FloatTensor([obs[agent][1] / 100])  # Get community risk (0 to 1)

                        # Evaluate all actions and get the best action based on immediate reward
                        best_action, allowed_value, new_infected, reward = self.get_best_action(env,
                                                                                                num_infected,
                                                                                                community_risk_value,
                                                                                                alpha
                                                                                                )

                        # Store the selected action
                        actions[agent] = best_action

                        # Accumulate total rewards and track values over time
                        allowed_values_over_time.append(allowed_value)
                        infected_values_over_time.append(new_infected)

                        total_reward += sum(reward.values())



            total_rewards.append(total_reward)

        avg_reward = np.mean(total_rewards)
        print(f"Average Reward over {num_episodes} episodes: {avg_reward}")
        return avg_reward

    def get_best_action(self, env, infected, community_risk, alpha):
        """
        Evaluate all possible actions for each agent at the given state (infected, community risk)
        and return the best action based on the immediate reward.
        """
        best_actions = {}
        max_rewards = {}
        allowed_values = {}
        new_infected_values = {}

        # Initialize the best rewards to negative infinity
        for agent in env.agents:
            max_rewards[agent] = -float('inf')

        # Iterate over all possible actions for all agents within their valid action spaces
        for action_combination in range(min(env.action_spaces[env.agents[0]].n, len(env.allowed_students))):
            actions = {agent: action_combination for agent in env.agents}  # Set actions for all agents

            # Step the environment with the current action combination
            next_obs, rewards, dones, _ = env.step(actions)

            # Iterate over agents to track the best action for each
            for agent in env.agents:
                reward = rewards[agent]

                # If the reward is greater than the previous max, update it
                if reward > max_rewards[agent]:
                    max_rewards[agent] = reward
                    best_actions[agent] = action_combination
                    allowed_values[agent] = env.allowed_students[action_combination]
                    new_infected_values[agent] = next_obs[agent][0]  # New infected value for the agent

        return best_actions, allowed_values, new_infected_values, max_rewards







