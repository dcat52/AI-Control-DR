import numpy as np
import pygame
import pymunk
import pymunk.pygame_util
from pymunk import Vec2d

from src.simulator.Agent import Agent
from src.simulator.Box import Box
from src.simulator.Reward import Reward


class Environment:
    
    def __init__(self, robot_start: Vec2d = (0, 0), goal: Vec2d = (2, 2), goal_threshold: float = 10.0,
                 noise_option: bool = True, randomize_goal_option: bool = True, carrot_reward: bool = False,
                 render: bool = True, render_step: int = 5, init: bool = True, box_mode: bool = True,
                 box_agent=False, random_start=False) -> None:
        # Physics
        # Time step
        self.dt = 1.0 / 60.0
        # Number of physics steps per screen frame
        self.physics_steps_per_frame = 1

        self.step_count = 0
        self.render_step = render_step

        self.space = pymunk.Space()
        self.friction_scalar = 0.80
        self.friction_angular_scalar = 0.60

        self.action_freq = 3 # in Hz

        self.robot_start = robot_start

        self.random_start = random_start

        if self.random_start:
            self.robot_start = self.set_new_random_start()

        self.agent = Agent(self, start_pos=self.robot_start, box_agent=box_agent)

        self.render_env = render
        self.called_render = False

        self.noise_option = noise_option
        self.randomize_goal_option = randomize_goal_option
        self.carrot_reward = carrot_reward

        self.box_mode = box_mode
        self.box_agent = box_agent

        if self.box_mode:
            self.box = Box(self)
        
        if init:
            pygame.init()
            if render:
                self.screen = pygame.display.set_mode((600, 600))
                self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)
                self.space.debug_draw(self.draw_options)
            self.clock = pygame.time.Clock()

        # Static barrier walls (lines) that the balls bounce off of
        self._add_static_scenery()
        agent_body = self.agent.get_body()
        agent_shape = self.agent.get_shape()
        self.space.add(agent_body, agent_shape)
        if self.box_mode:
            box_body = self.box.get_body()
            box_shape = self.box.get_shape()
            self.space.add(box_body, box_shape)

        self.reward_model = Reward(goal=goal, carrot_reward=carrot_reward)
        self.goal = goal
        self.waypoint = (10, 10)
        self.goal_threshold = goal_threshold

        self.running = True

        if self.render_env:
            self._render()

    def reset(self) -> None:
        self.__init__(robot_start=self.robot_start, goal=self.goal, goal_threshold=self.goal_threshold,
                      noise_option=self.noise_option, randomize_goal_option=self.randomize_goal_option,
                      carrot_reward=self.carrot_reward, render=self.render_env, render_step=self.render_step,
                      init=False, box_mode=self.box_mode, box_agent=self.box_agent, random_start=self.random_start)
        return (self.get_agent_state())

    def step(self, action) -> None:
        if self.render_env:
            self._process_keyboard()
        self.step_count += 1

        self.agent.set_motors(action)

        for x in range(self.physics_steps_per_frame):
            self.space.step(self.dt)
            self._assess_friction()
        
        if self.render_env and self.step_count % self.render_step == 0:
            self._clear_screen()
            self._draw_objects()
            pygame.display.flip()
            self.clock.tick(50)

        if self.render_env and self.step_count % 50 == 0:
            pygame.display.set_caption(str(self.step_count))
            self._render()

        self._update_agent_state()
        state_prime = self.get_agent_state()
        if self.box_mode:
            state_prime = self.get_box_state()
        agent_pos = self.agent.get_pos()
        if self.box_mode:
            reward = self.reward_model.calculate_box_reward(self.box.get_pos(), self.agent.get_pos())
        else:
            reward = self.reward_model.calculate_reward(agent_pos)

        done = False

        dist = self._agent_dist_to_goal()
        if self.box_mode:
            dist = self._box_dist_to_goal()
        if dist <= self.goal_threshold:
            reward += self.reward_model.reward_goal
            self.set_new_random_goal()
            done = True

        # Collision event logging for wall contact
        # TODO:
        # (state, reward, done, None)
        return state_prime, reward, done, None

    def update_waypoint_loc(self, waypoint_loc) -> None:
        self.waypoint = waypoint_loc

    def set_new_goal(self, goal: Vec2d) -> None:
        self.goal = goal
        self.reward_model.set_new_goal(goal)

    def set_new_random_start(self) -> None:
        start = np.random.randint(100, 300, (2))
        start = start[0], start[1]
        self.robot_Start = start
        return start
        # self.reward_model.set_new_goal(goal)

    def set_new_random_goal(self) -> None:
        # TODO: unbreak this function when the agent is smarter
        goal = self.goal
        if self.randomize_goal_option:
            goal = np.random.randint(400, 490, (2))
        self.goal = goal
        self.reward_model.set_new_goal(goal)

    def _agent_dist_to_goal(self):
        pos = self.agent.get_pos()
        dist = pos.get_distance(self.goal)
        return dist

    def _box_dist_to_goal(self):
        pos = self.box.get_pos()
        dist = pos.get_distance(self.goal)
        return dist

    def box_dist_to_agent(self):
        pos = self.box.get_pos()
        dist = pos.get_distance(self.agent.get_pos())
        return dist

    def _env_info_from_agent(self, agent_body) -> None:
        # TODO: add implementation
        raise NotImplementedError("Currently does nothing.")
        return

    def _update_agent_state(self) -> None:
        self.agent.update_agent_state()

    def get_waypoint_state(self, waypoint) -> np.ndarray:
        state = self.agent.get_waypoint_state(waypoint)
        return state

    def get_box_state(self) -> np.ndarray:
        state = self.agent.get_box_state()
        return state

    def get_agent_state(self) -> np.ndarray:
        state = self.agent.get_state()
        return state

    def _assess_friction(self) -> None:
        self.agent.body.velocity *= self.friction_scalar
        self.agent.body.angular_velocity *= self.friction_angular_scalar

        if self.box_mode:
            self.box.body.velocity *= self.friction_scalar
            self.box.body.angular_velocity *= self.friction_angular_scalar

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
                pygame.image.save(self.screen, "screenshot.png")
            if event.type == pygame.KEYDOWN and event.key == pygame.K_COMMA:
                self.render_step = max(self.render_step-1, 1)
                print("Render step is {}".format(self.render_step))
            if event.type == pygame.KEYDOWN and event.key == pygame.K_PERIOD:
                self.render_step = min(self.render_step+1, 10)
                print("Render step is {}".format(self.render_step))
            # Toggle for motor input noise
            if event.type == pygame.KEYDOWN and event.key == pygame.K_n:
                self.noise_option = not self.noise_option
                print("Motor force noise added: " + str(self.noise_option))
            # Toggle for randomizing goal location on next attainment
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                self.randomize_goal_option = not self.randomize_goal_option
                print("Randomizing goal after attainment: " + str(self.randomize_goal_option))
            if event.type == pygame.KEYDOWN and event.key == pygame.K_s:
                self.random_start = not self.random_start
                print("Randomizing start loc: " + str(self.random_start))

    def set_not_running(self):
        self.running = False

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
        pygame.draw.circle(self.screen, (0, 150, 0), self.goal, 10)
        pygame.draw.circle(self.screen, (0, 150, 150), self.waypoint, 20)
        self.space.debug_draw(self.draw_options)

    # def _create_box(self):

