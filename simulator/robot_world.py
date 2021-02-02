import random
from typing import List

import pygame
from pygame.color import THECOLORS

import pymunk
import pymunk.pygame_util
from pymunk import Vec2d


class Robot_World():

    def __init__(self) -> None:

        # Space
        self._space = pymunk.Space()
        self._space.gravity = (000.0, 000.0)

        # Physics
        # Time step
        self._dt = 1.0 / 60.0
        # Number of physics steps per screen frame
        self._physics_steps_per_frame = 1

        # pygame
        pygame.init()
        self._screen = pygame.display.set_mode((600, 600))
        self._clock = pygame.time.Clock()

        self._draw_options = pymunk.pygame_util.DrawOptions(self._screen)

        # Static barrier walls (lines) that the balls bounce off of
        self._add_static_scenery()

        # Balls that exist in the world
        self._balls: List[pymunk.Circle] = []

        # Execution control and time until the next ball spawns
        self._running = True
        self._ticks_to_next_ball = 1

        self._add_robot()

    def run(self) -> None:
        """
        The main loop of the game.
        :return: None
        """
        # Main loop
        while self._running:
            # Progress time forward
            for x in range(self._physics_steps_per_frame):
                self._space.step(self._dt)

            self._robot_body.velocity *= 0.95
            self._robot_body.angular_velocity *= 0.95
            self._process_events()
            self._update_balls()
            self._clear_screen()
            self._draw_objects()
            pygame.display.flip()
            # Delay fixed time between frames
            self._clock.tick(50)
            pygame.display.set_caption("fps: " + str(self._clock.get_fps()))

    def _add_static_scenery(self) -> True:
        # walls
        static_lines = [
            pymunk.Segment(self._space.static_body, (100, 500), (100, 100), 1.0),
            pymunk.Segment(self._space.static_body, (500, 500), (100, 500), 1.0),
            pymunk.Segment(self._space.static_body, (500, 100), (500, 500), 1.0),
            pymunk.Segment(self._space.static_body, (100, 100), (500, 100), 1.0)
        ]
        for line in static_lines:
            line.elasticity = 0.7
            line.group = 1

        self._space.add(*static_lines)

    def _add_robot(self) -> None:
        mass = 10
        radius = 20
        inertia = pymunk.moment_for_circle(mass, 0, radius, (0, 0))
        self._robot_body = pymunk.Body(mass,inertia)
        self._robot_body.position = (300, 300)
        shape = pymunk.Circle(self._robot_body, radius)
        shape.color = THECOLORS['red']
        shape.elasticity = .5
        shape.friction = 0.99
        self._space.add(self._robot_body, shape)

    def _process_events(self) -> None:
        """
        Handle game and events like keyboard input. Call once per frame only.
        :return: None
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_s:
                # self._robot_body.velocity = (100,0)
                self._robot_body.apply_force_at_local_point((-50000,0), (0,-20))
            if event.type == pygame.KEYDOWN and event.key == pygame.K_w:
                self._robot_body.apply_force_at_local_point((50000,0), (0,-20))
            if event.type == pygame.KEYDOWN and event.key == pygame.K_d:
                self._robot_body.apply_force_at_local_point((-50000,0), (0,20))
            if event.type == pygame.KEYDOWN and event.key == pygame.K_e:
                self._robot_body.apply_force_at_local_point((50000,0), (0,20))
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self._running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                pygame.image.save(self._screen, "bouncing_balls.png")

    def _update_balls(self) -> None:
        """
        Create/remove balls as necessary. Call once per frame only.
        :return: None
        """
        self._ticks_to_next_ball -= 1
        if self._ticks_to_next_ball <= 0:
            self._create_ball()
            self._ticks_to_next_ball = 100

    def _create_ball(self) -> None:
        """
        Create a ball.
        :return:
        """
        mass = 10
        radius = 25
        inertia = pymunk.moment_for_circle(mass, 0, radius, (0, 0))
        body = pymunk.Body(mass, inertia)
        x = random.randint(115, 350)
        body.position = x, 200
        shape = pymunk.Circle(body, radius, (0, 0))
        shape.elasticity = 0.99
        shape.friction = 0.9
        self._space.add(body, shape)
        self._balls.append(shape)

    def _clear_screen(self) -> None:
        """
        Clears the screen.
        :return: None
        """
        self._screen.fill(pygame.Color("white"))

    def _draw_objects(self) -> None:
        """
        Draw the objects.
        :return: None
        """
        self._space.debug_draw(self._draw_options)


if __name__ == "__main__":
    game = Robot_World()
    game.run()