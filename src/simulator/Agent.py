from queue import Queue
from math import cos, sin

import numpy as np
import pymunk
import pymunk.pygame_util
from pygame.color import THECOLORS
from pymunk import Vec2d
import random


class Agent:
    def __init__(self, _env, mass: float = 10, radius: float = 20, pos_hist_len: int = 2,
                 start_pos: Vec2d = (0, 0)) -> None:
        self._env = _env

        self.start_pos = start_pos
        self.pos_hist_len = pos_hist_len
        self.pos_history = Queue(maxsize=pos_hist_len)
        self.ang_history = Queue(maxsize=pos_hist_len)

        self.mass = mass
        self.radius = radius
        
        self.inertia = pymunk.moment_for_circle(self.mass, 0, self.radius, (0, 0))
        self.body = pymunk.Body(self.mass, self.inertia)
        self.body.position = self.start_pos
        self.shape = pymunk.Circle(self.body, self.radius)
        self.body._set_angle(random.randint(0,628)/100)
        self.shape.color = THECOLORS['red']
        self.shape.elasticity = .5
        self.shape.friction = 0.99

        for i in range(pos_hist_len):
            self.pos_history.put(self.start_pos, block=False)

        for i in range(pos_hist_len):
            self.ang_history.put(self.get_angle(), block=False)

    def get_body(self) -> object:
        return self.body

    def get_shape(self) -> object:
        return self.shape

    def get_state(self) -> np.ndarray:
        self.pos_history.get(block=False)
        self.pos_history.put(self.get_pos(), block=False)
        self.ang_history.get(block=False)
        self.ang_history.put(self.get_angle(), block=False)

        pos_hist = list(self.pos_history.queue)
        delta1 = pos_hist[-1] - pos_hist[0]

        a = self.get_angle()
        r = np.array([[cos(a), sin(a)], [-sin(a), cos(a)]])
        delta1 = np.array(delta1)
        delta1 = r.dot(delta1) 
        ang_hist = list(self.ang_history.queue)
        delta2 = ang_hist[-1] - ang_hist[0]

        observation = self._sense()
        # state = np.append(observation, list(self.pos_history.queue))
        state = np.append(observation, delta1/20)
        state = np.append(state, delta2)

        return state

    def set_motors(self, velocities: tuple) -> None:
        self.body.apply_force_at_local_point((20000 * velocities[0], 0), (0, -10))
        self.body.apply_force_at_local_point((20000 * velocities[1], 0), (0, 10))

    def get_pos(self) -> Vec2d:
        return self.body.position

    def get_angle(self) -> float:
        return self.body.angle

    def _sense(self) -> np.ndarray:
        # obs = self._env._env_info_from_agent(self.body)
        a = self.get_angle()
        r = np.array([[cos(a), sin(a)], [-sin(a), cos(a)]])
        v = np.array(self._env.goal - self.get_pos())
        sensors = r.dot(v)
        sensors = sensors / 500
        # sensors = np.array(self.get_pos()) / 500
        # sensors = np.append(sensors, self.get_angle()/100)
        return sensors

    def _set_pos(self, new_pos: Vec2d) -> None:
        self.body.pos = new_pos
