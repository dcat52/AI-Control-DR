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

    def _get_rotation_matrix(self) -> np.ndarray:

        # Setup Rotation matrix to transform from world to agent frame
        agent_angle = self.get_angle()
        R_world_to_agent = np.array([[cos(a), sin(a)], [-sin(a), cos(a)]])

        # 2x2 rotation matrix
        return R_world_to_agent

    def _get_agent_x_y_theta_velocities(self, R_world_to_agent: np.ndarray) -> np.ndarray:
        # calculate agent x,y velocities in agent frame
        pos_hist = list(self.pos_history.queue)
        agent_position_velocities = pos_hist[-1] - pos_hist[0]
        agent_position_velocities = np.array(agent_position_velocities)
        agent_position_velocities = R_world_to_agent.dot(agent_position_velocities)
        agent_position_velocities /= 20

        # calculate agent theta velocity
        agent_angle_history = list(self.ang_history.queue)
        agent_omega = agent_angle_history[-1] - agent_angle_history[0]

        agent_velocities = np.append(agent_position_velocities, agent_omega)

        # dx, dy, dtheta
        return agent_velocities

    def _get_agent_dist_to_point(self, R_world_to_agent: np.ndarray, xy_point: Vec2d) -> np.ndarray:

        # calculate x,y distance to a point in agent frame
        dist_to_point = np.array(xy_point - self.get_pos())
        dist_to_point = R_world_to_agent.dot(dist_to_point)
        dist_to_point = dist_to_point / 500
        
        # x, y distance
        return dist_to_point

    def get_waypoint_state(self, xy_point: Vec2d) -> np.ndarray:

        R_world_to_agent = self._get_rotation_matrix()
        
        dist_to_point = self._get_agent_dist_to_point(R_world_to_agent, xy_point)
        result = dist_to_point

        agent_velocities = self._get_agent_x_y_theta_velocities(R_world_to_agent)
        result = np.append(result, agent_velocities)

        # x, y, x, y, dx, dy, dtheta
        return result

    def get_box_state(self) -> np.ndarray:

        R_world_to_agent = self._get_rotation_matrix()
        
        dist_to_dest = self._get_agent_dist_to_point(R_world_to_agent, self._env.goal)
        result = dist_to_dest
        
        dist_to_box = self._get_agent_dist_to_point(R_world_to_agent, self._env.box.get_pos())
        result = np.append(result, dist_to_box)

        agent_velocities = self._get_agent_x_y_theta_velocities(R_world_to_agent)
        result = np.append(result, agent_velocities)

        # x, y, dx, dy, dtheta
        return result

    def get_state(self) -> np.ndarray:

        R_world_to_agent = self._get_rotation_matrix()
        
        dist_to_goal = self._get_agent_dist_to_point(R_world_to_agent, self._env.goal)
        result = dist_to_goal

        agent_velocities = self._get_agent_x_y_theta_velocities(R_world_to_agent)
        result = np.append(result, agent_velocities)

        # x, y, dx, dy, dtheta
        return result

    def update_agent_state(self) -> None:
        self.pos_history.get(block=False)
        self.pos_history.put(self.get_pos(), block=False)
        self.ang_history.get(block=False)
        self.ang_history.put(self.get_angle(), block=False)

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
