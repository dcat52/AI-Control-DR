
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

    def calculate_box_reward(self, box_loc, agent_loc: Vec2d):
        # add reward to bring agent near box (maybe try correct side of box?)
        # add reward to push box to goal
        goal_dist = box_loc.get_distance(self.goal)
        agent_dist = agent_loc.get_distance(box_loc)
        goal_reward = goal_dist / 100
        box_proximity_reward = agent_dist/500
        return - (goal_reward + box_proximity_reward)

    def calculate_reward(self, pos: Vec2d):

        length = pos.get_distance(self.goal)
        if self.carrot_reward:
            reward = 100 / length
            return reward
        else:
            return - length / 100

    def set_new_goal(self, goal: Vec2d) -> None:
        self.goal = goal
