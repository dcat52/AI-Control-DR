from src.simulator.Environment import Environment
import src.controllers.AC2 as AC2
import argparse
import logging
import sys
import random
import tensorflow
import numpy

def parse():
    parser = argparse.ArgumentParser(description="AI Control DR")
    # primary train settings
    parser.add_argument('-t', '--train_dqn',      action='store_true',    help='Whether train mode')
    parser.add_argument('-T', '--test_dqn',       action='store_true',    help='Whether test mode')
    parser.add_argument('-r', '--render',         action='store_true',    help='Whether to render')
    parser.add_argument('--carrot',               action='store_true',    help='Whether to use carrot reward')

    # model settings
    parser.add_argument('--print_freq',      dest="PRINT_FREQ",      default=1, type=int,        help='How often to print information to std out')
    parser.add_argument('--write_freq',      dest="WRITE_FREQ",      default=5, type=int,        help='How often to save information to disk')
    parser.add_argument('--save_freq',       dest="SAVE_FREQ",       default=500, type=int,     help='Save model weights every n iterations')
    parser.add_argument('--batch_size',      dest="BATCH_SIZE",      default=2000, type=int,       help='Batch size')
    parser.add_argument('--buffer_capacity', dest="BUFFER_CAPACITY", default=40000, type=int,    help='Buffer capacity')
    parser.add_argument('--episodes', '--ep',dest="NUM_EPISODES",    default=1000, type=int,     help='Number of episodes to run')
    parser.add_argument('--update_freq',     dest="TARGET_UPDATE",   default=1, type=int,        help='Update target network every n iterations')
    parser.add_argument('--tau',             dest="TAU",             default=0.05, type=float,   help='Update ratio of target network')
    parser.add_argument('--gamma',           dest="GAMMA",           default=0.99, type=float,   help='Future reward decay')
    parser.add_argument('--std',             dest="STD_DEV",         default=0.1, type=float,   help='Standard deviation of noise')
    parser.add_argument('--theta',           dest="THETA",           default=0.05, type=float,   help='Theta of noise')
    parser.add_argument('--save_prefix',     dest="SAVE_PREFIX",     default="data", type=str,   help='Prefix of location to save content')
    parser.add_argument('--actor_lr',        dest="ACTOR_LR",        default=0.0001, type=float, help='Learning rate for the actor')
    parser.add_argument('--critic_lr',       dest="CRITIC_LR",       default=0.0002, type=float, help='Learning rate for the critic')
    parser.add_argument('--plot',            dest="PLOT",            action='store_true',        help='Whether to plot data')
    parser.add_argument('--tensorboard',     dest="TENSORBOARD",     default=1, type=int,        help='Whether to use tensorboard')
    parser.add_argument('--date',            dest="DATE_IN_PREFIX",  action='store_true',        help='Use the date in the prefix string (appended as _20210314_180101)')
    parser.add_argument('--load_prefix',     dest="LOAD_PREFIX",     default="data", type=str,   help='Location to load model weights from')
    parser.add_argument('--seed',            dest="SEED",            default=-1, type=int,       help='Set the default seed value, -1 is random seed.')

    parser.add_argument('--actor_layer_width',  dest="ACTOR_LAYER_WIDTH",  default=256,  type=int, help='Actor - Width of layer')
    parser.add_argument('--actor_num_layers',   dest="ACTOR_NUM_LAYERS",   default=2,    type=int, help='Actor - Number layers deep')
    parser.add_argument('--critic_layer_width', dest="CRITIC_LAYER_WIDTH", default=256,  type=int, help='Critic - Width of layer')

    # environment settings
    parser.add_argument('--start_loc',      default=[300, 300], nargs='+', type=int,    help='Start location in format --start_loc x y')
    parser.add_argument('--goal_loc',       default=[400, 400], nargs='+', type=int,    help='Goal location in format  --goal_loc x y')
    parser.add_argument('--goal_thresh',    default=20, type=int,                       help='Threshold range to be considered at the goal')

    # additional settings
    parser.add_argument('-l', '--log_level',    default=3, type=int,    help='Set logging level: 0=Critical, 1=Error, 2=Warning, 3=Info, 4=Debug')
    parser.add_argument('-g', '--gpu_mem_config', action='store_true',    help='Changes CUDA GPU memory allocation method')

    args = parser.parse_args()

    args.start_loc = tuple(args.start_loc)
    args.goal_loc = tuple(args.goal_loc)
    args.SAVE_PREFIX = "'{}'".format(args.SAVE_PREFIX)
    args.LOAD_PREFIX = "'{}'".format(args.LOAD_PREFIX)
    args.log_level = (5 - args.log_level) * 10
    print("Modified args:", args)
    return args


def run(args: argparse.Namespace):
    logging.basicConfig(level=args.log_level,
                        format='\33[90m[%(filename)20s:%(lineno)4d]\33[0m %(levelname)s: %(message)s')

    if args.gpu_mem_config:
        import tensorflow as tf
        physical_devices = tf.config.list_physical_devices('GPU')
        try:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
        except:
            # Invalid device or cannot modify virtual devices once initialized.
            pass

    if args.SEED != -1:
        tensorflow.random.set_seed(args.SEED)
        numpy.random.seed(args.SEED)
        random.seed(args.SEED)

    if args.train_dqn:
        env = Environment(robot_start=args.start_loc, goal=args.goal_loc, goal_threshold=args.goal_thresh,
                          render=args.render, carrot_reward=args.carrot)
        agent = AC2.AC_Agent(env, args)
        agent.train()

    if args.test_dqn:
        env = Environment(robot_start=args.start_loc, goal=args.goal_loc, goal_threshold=args.goal_thresh,
                          render=args.render)
        actor = args.LOAD_PREFIX
        critic = args.LOAD_PREFIX
        agent = AC2.AC_Agent(env, args)
        agent.test()


if __name__ == '__main__':
    arguments = parse()
    try:
        run(arguments)
    except KeyboardInterrupt as ke:
        sys.exit(-1)

    sys.exit(0)
