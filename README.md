#Pathfinding With Small Neural Networks and Reinforcement Learning
 
Dependencies:

- Python 3.7
- Keras
- Pymunk
- Pygame

---
1. Introduction

This is a collection of Python files that create a basic simulation environment, a two-wheeled robot analog, and up to two neural networks that train using reinforcement learning. The &#39;motor network&#39; network manages the force applications to the wheels while a &#39;waypoint network&#39; generates waypoints within a window around the robot to generate a trajectory. The networks are currently simulation-only but are designed with constraints to allow for live sensor data to be used in the future.

Networks are instantiated randomly and tuned using a reward function based on variables such as distance to goal with exploration done using samples from a set of Ornstein-Uhlenbeck processes. The motor network is first trained in isolation with a stochastically generated series of destination points that change once the network successfully reaches the goal. Once the motor network is trained, the weights are saved and used to train a waypoint generating network through a similar simulation that pushes a box onto a goal location.

####1. Simulation Environment
  #####General
The simulation environment is pymunk, an open-source python library we use for simulating forces, momentum, and obstacle collisions in a 2D environment. Pygame is used to optionally render the environment for analysis.

We generate a small square area contained by walls as the robot&#39;s workspace and instantiate a small circle as the &#39;robot&#39;. This circle has force applied at two equally offset central locations as analogs for the force applied by a motor assembly.


  #####Motor network
The motor network uses a point provided by the simulation as its goal location. The goal location is used to calculate a negative reward for the robot based upon the distance between them. Additionally, information is provided to the robot based on simulated IMU data and a rangefinder at the goal in the form of: [x, y, dx, dy, dtheta]. Values are in the robot&#39;s frame of reference.


  #####Waypoint network
The waypoint network currently uses a box-and-goal setup for the reward calculation. 


  #####Logging
Extensive optional logging of state space variables and network behaviors is available. Each additional logging level includes the levels below it. Default is 1.

| Logging Level: | Additional Logging: |
| --- | --- |
| Level 0 | Nothing 
| Level 1 | Reward metrics, actor network outputs, input noise   
| Level 2 | State space variables, critic network outputs
| Level 3 | Summation of each actor input neuron's outputs

####1. Networks

  #####Overview

Both models use Deep Deterministic Policy Gradient algorithms and are trained in a similar manner in simulation using an added Ornstein-Uhlenbeck process term for exploration.

| Network: | State Space: | Output: | Reward f(x): |
| --- | --- | --- | --- |
| Motor network | [x, y, dx, dy, dtheta] (agent) | Motors: 2x [-1, 1] | -(agent-to-waypoint) |
| Waypoint network | [agent\_pos, box\_pos, agent\_vel] | Waypoint: [x, y] | -(box-to-goal + agent-to-box) |

  #####Motor network

The goal location is used to calculate a negative reward for the agent based upon the distance between them.

  #####Waypoint network

This setup uses the negative sum of two distances: box-to-goal and box-to-agent.

####1. Work in Progress/Broken:
  #####Box pushing rarely works
      
  - Proposed reward function changes
  - Something more complex than just box distances added together
    - Calculate intermediate waypoint on correct side of box? (how?)
  - Proposed environment/state space changes
    - Change frequency of wp generation?
      
      
####2. Future work:
  #####Incorporate information sources:
   - Simulated proximity sensors
   - Detect spinning in place?
    
  #####Behaviors:
   - Penalty for wall hits
    
  #####Extensions
  - Robo tag
  - Maze solving
  
  #####This Document
  - gifs of various emergent behaviors: turning, choosing forward/backward, noise tolerance
  - gifs of problems: L-shaped paths, spinning, near-goal behavior
    
    
####3. References
- [https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck\_process](https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process)

- ddpg paper