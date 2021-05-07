from queue import Queue
from math import cos, sin

import numpy as np
import pymunk
import pymunk.pygame_util
from pygame.color import THECOLORS
from pymunk import Vec2d
import random

class Box:
    def __init__(self, _env, mass: float = 10, moment: float = 200, shape: tuple = (40.0, 40.0),
                 start_pos: Vec2d = (350, 350)) -> None:
        self._env = _env

        self.start_pos = start_pos

        self.mass = mass
        self.moment = moment
        self.shape = shape
        
        self.body = pymunk.Body(mass=self.mass, moment=self.moment)
        self.body.position = start_pos

        self.box = pymunk.Poly.create_box(self.body, self.shape)
        self.box.elasticity = 0.5

        self.body.position = self.start_pos
        
        self.body._set_angle(random.randint(0,628)/100)

    def get_body(self) -> object:
        return self.body

    def get_shape(self) -> object:
        return self.box

    def get_pos(self) -> Vec2d:
        return self.body.position

    def get_angle(self) -> float:
        return self.body.angle

    def _set_pos(self, new_pos: Vec2d) -> None:
        self.body.pos = new_pos
