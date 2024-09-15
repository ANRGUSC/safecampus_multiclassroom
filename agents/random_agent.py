import random


class RandomAgent:
    def __init__(self, env):
        self.env = env

    def act(self, observations):
        return {agent: self.env.action_spaces[agent].sample() for agent in self.env.agents}
