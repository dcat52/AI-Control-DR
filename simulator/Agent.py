from enum import Enum
import numpy as np
from queue import Queue
import random
from typing import List

import pygame
from pygame.color import THECOLORS
import pymunk
import pymunk.pygame_util
from pymunk import Vec2d

class Agent:
    def __init__(self, _env, mass: float = 10, radius: float = 20, pos_hist_len: int = 5, start_pos: Vec2d = (0, 0)) -> None:
        self._env = _env

        self.start_pos = start_pos
        self.pos_hist_len = pos_hist_len
        self.pos_history = Queue(maxsize=pos_hist_len)

        for i in range(pos_hist_len):
            self.pos_history.put(self.start_pos, block=False)

        self.mass = mass
        self.radius = radius
        
        self.inertia = pymunk.moment_for_circle(self.mass, 0, self.radius, (0, 0))
        self.body = pymunk.Body(self.mass, self.inertia)
        self.body.position = (300, 300)
        self.shape = pymunk.Circle(self.body, self.radius)
        self.shape.color = THECOLORS['red']
        self.shape.elasticity = .5
        self.shape.friction = 0.99

    def get_body(self) -> object:
        return self.body

    def get_shape(self) -> object:
        return self.shape

    def get_state(self) -> np.ndarray:
        self.pos_history.get(block=False)
        self.pos_history.put(self.get_pos(), block=False)

        observation = self._sense()
        state = np.append(observation, list(self.pos_history.queue))

        return state

    def set_motors(self, velocities: tuple) -> None:
        self.body.apply_force_at_local_point((20000 * velocities[0], 0), (0, -20))
        self.body.apply_force_at_local_point((20000 * velocities[1], 0), (0, 20))

    def get_pos(self) -> Vec2d:
        return self.body.position

    def _sense(self) -> np.ndarray:
        # obs = self._env._env_info_from_agent(self.body)
        sensors = np.array(self.get_pos())
        return sensors

    def _set_pos(self, new_pos: Vec2d) -> None:
        self.body.pos = new_pos
