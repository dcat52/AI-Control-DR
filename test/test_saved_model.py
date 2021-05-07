from argparse import Namespace

from src.simulator.Environment import Environment
from src.controllers.AC2 import *
import tensorflow as tf

"""
Script to run a predefined save file in a test environment
Runs without training or added noise
"""

ACTOR_MODEL_DIR = 'data_weights/kg_actor'


physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass
#
# args = Namespace(ACTOR_LAYER_WIDTH=256,
#                  ACTOR_LR=0.0001,
#                  ACTOR_NUM_LAYERS=2,
#                  BATCH_SIZE=32,
#                  BUFFER_CAPACITY=10000,
#                  CRITIC_LR=0.0002,
#                  DATE_IN_PREFIX=False,
#                  GAMMA=0.99,
#                  NUM_EPISODES=1,
#                  PLOT=False,
#                  PRINT_FREQ=1,
#                  SAVE_FREQ=1000,
#                  SAVE_PREFIX="'data'",
#                  STD_DEV=0.1,
#                  TARGET_UPDATE=1,
#                  TAU=0.05,
#                  TENSORBOARD=2,
#                  THETA=0.15,
#                  WRITE_FREQ=5,
#                  goal_loc=(400, 400),
#                  goal_thresh=20,
#                  gpu_mem_config=True,
#                  log_level=5,
#                  render=False,
#                  start_loc=(300, 300),
#                  test_dqn=True,
#                  train_dqn=False,
#                  LOAD_PREFIX="'{}'".format(MODEL_PREFIX))
#
# env = Environment(robot_start=(300, 300), goal=(400, 400), goal_threshold=10, render=True)
# ac = AC_Agent(env, args)

model = tf.keras.models.load_model(ACTOR_MODEL_DIR)

# ac.test()
