import argparse
import logging

from simulator.Environment import Environment
import controllers.AC2 as AC2

def parse():
    parser = argparse.ArgumentParser(description="AI Control DR")
    parser.add_argument('-T', '--train_dqn', action='store_true', help='whether train DQN')
    parser.add_argument('-t', '--test_dqn', action='store_true', help='whether test DQN')
    parser.add_argument('-b', '--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('-r', '--render', action='store_true', help='whether to render')

    parser.add_argument('-l', '--log_level', default=3, type=int, help='set logging level: 0=Critical, 1=Error, 2=Warning, 3=Info, 4=Debug')

    parser.add_argument('--start_loc', default=[300, 300], nargs='+', type=int, help='start location in format --start_loc x y')
    parser.add_argument('--goal_loc', default=[400, 400], nargs='+', type=int, help='goal location in format  --goal_loc x y')
    parser.add_argument('--goal_thresh', default=20, type=int, help='threshold range to be considered at the goal')

    args = parser.parse_args()

    args.start_loc = tuple(args.start_loc)
    args.goal_loc = tuple(args.goal_loc)
    args.log_level = (5 - args.log_level) * 10
    print("Modified args:", args)

    return args


def run(args: argparse.Namespace):
    logging.basicConfig(level=args.log_level, format='\33[90m[%(filename)20s:%(lineno)4d]\33[0m %(levelname)s: %(message)s')
    
    if args.train_dqn:
        
        env = Environment(robot_start=args.start_loc, goal=args.goal_loc, goal_threshold=args.goal_thresh, render=args.render)
        # models = AC.create_models(state_length=6, action_length=2, action_bounds=(-1.0, 1.0))
        # target_models = AC.create_models(state_length=6, action_length=2, action_bounds=(-1.0, 1.0))
        # hyper_params = AC.HyperParams(gamma=1, tau=0.02, std_dev=0.5, actor_lr=0.001, critic_lr=0.002)
        # rewards = AC.fit(env, models, target_models, episodes=100, hp=hyper_params, state_length=6, action_bounds=(-1.0, 1.0))
        # AC.plot_reward(rewards)

        agent = AC2.AC_Agent(env, args)
        agent.train()

    if args.test_dqn:
        raise NotImplementedError("Not implemented yet. Currently does nothing.")


if __name__ == '__main__':
    arguments = parse()
    run(arguments)
