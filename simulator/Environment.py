from enum import Enum
import numpy as np
import random
from typing import List

import pygame
from pygame.color import THECOLORS
import pymunk
import pymunk.pygame_util
from pymunk import Vec2d

from simulator.Agent import Agent
from simulator.Reward import Reward

class Environment:
    
    def __init__(self, robot_start: Vec2d = (0, 0), goal: Vec2d = (2, 2), render: bool = True) -> None:

        # Physics
        # Time step
        self.dt = 1.0 / 60.0
        # Number of physics steps per screen frame
        self.physics_steps_per_frame = 1

        self.space = pymunk.Space()
        self.friction_scalar = 0.80

        self.action_freq = 3 # in Hz

        self.robot_start = robot_start
        self.agent = Agent(self, start_pos=robot_start)

        self.render_env = render
        self.called_render = False

        # pygame
        pygame.init()
        self.screen = pygame.display.set_mode((600, 600))
        self.clock = pygame.time.Clock()

        self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)

        # Static barrier walls (lines) that the balls bounce off of
        self._add_static_scenery()
        agent_body = self.agent.get_body()
        agent_shape = self.agent.get_shape()
        self.space.add(agent_body, agent_shape)

        self.reward_model = Reward(goal=goal)

        self.running = True

        if self.render_env:
            self._render()

    def reset(self) -> None:
        self.__init__(self.robot_start, self.render_env)

    def step(self, action) -> None:
    # TODO: establish reward, return reward and some other thigns
        # result = self._make_action(action)
        self.agent.set_motors(action)

        for x in range(self.physics_steps_per_frame):
            self.space.step(self.dt)
            self._assess_friction()
        
        if self.render_env:
            self._clear_screen()
            self._draw_objects()
            pygame.display.flip()
            self.clock.tick(50)
            pygame.display.set_caption("fps: " + str(self.clock.get_fps()))

        if self.render_env:
            self._render()

        result = self._get_agent_state()
        agent_pos = self.agent.get_pos()
        reward = self.reward_model.calculate_reward(agent_pos)

        done = False

        dist = self._agent_dist_to_goal()
        if dist <= 0.05:
            done = True

        # (state, reward, done, None)
        return result, reward, done, None

    # def _make_action(self, action) -> tuple:
    #     self.agent.apply_velocities(action)

    def _agent_dist_to_goal(self):
        pos = self.agent.get_pos()
        dist = pos.get_distance(self.goal)
        return dist

    def _env_info_from_agent(self, agent_body) -> None:
        # TODO: add implementation
        raise NotImplementedError("Currently does nothing.")
        return

    def _get_agent_state(self) -> np.ndarray:
        state = self.agent.get_state()
        return state

    def _assess_friction(self) -> None:
        self.agent.body.velocity *= self.friction_scalar
        self.agent.body.angular_velocity *= self.friction_scalar

    def _add_static_scenery(self) -> True:
        # walls
        static_lines = [
            pymunk.Segment(self.space.static_body, (100, 500), (100, 100), 1.0),
            pymunk.Segment(self.space.static_body, (500, 500), (100, 500), 1.0),
            pymunk.Segment(self.space.static_body, (500, 100), (500, 500), 1.0),
            pymunk.Segment(self.space.static_body, (100, 100), (500, 100), 1.0)
        ]
        for line in static_lines:
            line.elasticity = 0.7
            line.group = 1

        self.space.add(*static_lines)

    def _process_keyboard(self) -> None:
        """
        Handle game and events like keyboard input. Call once per frame only.
        :return: None
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                pygame.image.save(self.screen, "bouncing_balls.png")

    def _render(self) -> None:
        self._clear_screen()
        self._draw_objects()

    def _clear_screen(self) -> None:
        """
        Clears the screen.
        :return: None
        """
        self.screen.fill(pygame.Color("white"))

    def _draw_objects(self) -> None:
        """
        Draw the objects.
        :return: None
        """
        self.space.debug_draw(self.draw_options)

if __name__ == "__main__":

    import time

    env = Environment()

    action_list = [
        (1,1),
        (1,1),
        (1,1),
        (0,0),
        (0,0),
        (0,1),
        (0,0),
        (1,1),
        (1,1),
        (1,0),
        (1,0),
        (1,0),
        (1,0),
        (0,0),
        (0,0)
    ]

    for a in action_list:
        env.step(a)
        time.sleep(.05)

    time.sleep(5)
