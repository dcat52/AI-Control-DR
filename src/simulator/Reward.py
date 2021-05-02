
from pymunk import Vec2d

class Reward:

    def __init__(self, goal: Vec2d = (0, 0), carrot_reward: bool = False):
        self.reward_goal = 10.0
        self.reward_death = -10.0
        self.goal = goal
        self.carrot_reward = carrot_reward

    def generate_reward_map(self):
        pass

    def update_goal(self, goal: Vec2d):
        self.goal = goal

    def calculate_reward(self, pos: Vec2d):
        length = pos.get_distance(self.goal)
        if self.carrot_reward:
            reward = 100 / length
            return reward
        else:
            return - length / 100

    def set_new_goal(self, goal: Vec2d) -> None:
        self.goal = goal
