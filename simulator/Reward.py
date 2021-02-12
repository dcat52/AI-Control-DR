
from pymunk import Vec2d

class Reward:

    def __init__(self, goal: Vec2d = (0, 0)):
        self.reward_death = -100.0
        self.goal = goal

    def generate_reward_map(self):
        pass

    def update_goal(self, goal: Vec2d):
        self.goal = goal

    def calculate_reward(self, pos: Vec2d):
        delta = self.goal - pos
        length = delta.get_length()
        return -length
