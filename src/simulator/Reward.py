
from pymunk import Vec2d

class Reward:

    def __init__(self, goal: Vec2d = (0, 0)):
        self.reward_goal = 10.0
        self.goal = goal

    def generate_reward_map(self):
        pass

    def update_goal(self, goal: Vec2d):
        self.goal = goal

    def calculate_reward(self, pos: Vec2d):
        length = pos.get_distance(self.goal)

        # res = pos.dot(self.goal)
        # print(res)
        # reward = 100 / length
        # return reward
        return - length / 100
        # return res/100000000

    def set_new_goal(self, goal: Vec2d) -> None:
        self.goal = goal
