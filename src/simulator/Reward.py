
from pymunk import Vec2d
from src.simulator.Agent import Agent

import math

class Reward:

    def __init__(self, goal: Vec2d = (0, 0), carrot_reward: bool = False, agent: Agent = None):
        self.reward_goal = 10.0
        self.goal = goal
        self.carrot_reward = carrot_reward
        self.agent = agent

    def generate_reward_map(self):
        pass

    def update_goal(self, goal: Vec2d):
        self.goal = goal

    def calculate_reward(self, pos: Vec2d):


        ang_hist = list(self.agent.ang_history.queue)
        delta2 = ang_hist[-1] - ang_hist[0]
        delta2 = abs(delta2)/10


        length = pos.get_distance(self.goal)
        if self.carrot_reward:
            reward = 100 / length
            return reward
        else:
            return (- length / 100) - delta2

    def set_new_goal(self, goal: Vec2d) -> None:
        self.goal = goal
