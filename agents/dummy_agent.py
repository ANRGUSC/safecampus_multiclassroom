import random
class DummyAgent:
    def __init__(self, action_space):
        self.action_space = action_space

    def select_action(self):
        return random.choice(range(self.action_space.nvec[0]))  # Random action for allowed students
