import datetime
import numpy as np
import tensorflow as tf


class TBLogger:
    def __init__(self, log_level, directory_prefix="data"):
        self.log_level = log_level

        # log directory locations
        log_dir = directory_prefix + "_logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M")
        # log_dir = directory_prefix + "_logs/"

        # Log level 1: episodic rewards only
        if self.log_level >= 1:
            self.rewards_writer = tf.summary.create_file_writer(logdir=log_dir)

        # Log level 2: additional logging for actions and noise
        if self.log_level >= 2:
            log_dir_net_outputs = log_dir + "/net_outputs/"

            self.noise_l_writer = tf.summary.create_file_writer(logdir=log_dir_net_outputs + "noise_L")
            self.noise_r_writer = tf.summary.create_file_writer(logdir=log_dir_net_outputs + "noise_R")

            self.actions_l_writer = tf.summary.create_file_writer(logdir=log_dir_net_outputs + "action_L")
            self.actions_r_writer = tf.summary.create_file_writer(logdir=log_dir_net_outputs + "action_R")

            self.ep_reward_writer = tf.summary.create_file_writer(logdir=log_dir_net_outputs + "episode_reward")

        # Log level 3: additional output from target_actor and both critic models
        if self.log_level >= 2:
            self.policy_critic_writer = tf.summary.create_file_writer(logdir=log_dir_net_outputs + "policy_critic")
            self.target_critic_writer = tf.summary.create_file_writer(logdir=log_dir_net_outputs + "target_critic")

        # Log level 4: additional full input weight sums (warning: slow)
        if self.log_level == 4:
            log_dir_policy_weights = log_dir + "/weights/policy"
            log_dir_target_weights = log_dir + "/weights/target"
            log_dir_policy_actor_weights = log_dir_policy_weights + "/actor/"
            log_dir_policy_critic_weights = log_dir_policy_weights + "/critic/"
            log_dir_target_actor_weights = log_dir_target_weights + "/actor/"
            log_dir_target_critic_weights = log_dir_target_weights + "/critic/"

            self.policy_actor_xpos_writer = tf.summary.create_file_writer(logdir=log_dir_policy_actor_weights + 'x_pos')
            self.policy_actor_ypos_writer = tf.summary.create_file_writer(logdir=log_dir_policy_actor_weights + 'y_pos')
            self.policy_actor_thet_writer = tf.summary.create_file_writer(logdir=log_dir_policy_actor_weights + 'theta')
            self.policy_actor_xvel_writer = tf.summary.create_file_writer(logdir=log_dir_policy_actor_weights + 'x_vel')
            self.policy_actor_yvel_writer = tf.summary.create_file_writer(logdir=log_dir_policy_actor_weights + 'y_vel')
            self.policy_actor_tvel_writer = tf.summary.create_file_writer(
                logdir=log_dir_policy_actor_weights + 'theta_vel')

            self.policy_critic_xpos_writer = tf.summary.create_file_writer(
                logdir=log_dir_policy_critic_weights + 'x_pos')
            self.policy_critic_ypos_writer = tf.summary.create_file_writer(
                logdir=log_dir_policy_critic_weights + 'y_pos')
            self.policy_critic_thet_writer = tf.summary.create_file_writer(
                logdir=log_dir_policy_critic_weights + 'theta')
            self.policy_critic_xvel_writer = tf.summary.create_file_writer(
                logdir=log_dir_policy_critic_weights + 'x_vel')
            self.policy_critic_yvel_writer = tf.summary.create_file_writer(
                logdir=log_dir_policy_critic_weights + 'y_vel')
            self.policy_critic_tvel_writer = tf.summary.create_file_writer(
                logdir=log_dir_policy_critic_weights + 'theta_vel')

            self.target_actor_xpos_writer = tf.summary.create_file_writer(logdir=log_dir_target_actor_weights + 'x_pos')
            self.target_actor_ypos_writer = tf.summary.create_file_writer(logdir=log_dir_target_actor_weights + 'y_pos')
            self.target_actor_thet_writer = tf.summary.create_file_writer(logdir=log_dir_target_actor_weights + 'theta')
            self.target_actor_xvel_writer = tf.summary.create_file_writer(logdir=log_dir_target_actor_weights + 'x_vel')
            self.target_actor_yvel_writer = tf.summary.create_file_writer(logdir=log_dir_target_actor_weights + 'y_vel')
            self.target_actor_tvel_writer = tf.summary.create_file_writer(
                logdir=log_dir_target_actor_weights + 'theta_vel')

            self.target_critic_xpos_writer = tf.summary.create_file_writer(
                logdir=log_dir_target_critic_weights + 'x_pos')
            self.target_critic_ypos_writer = tf.summary.create_file_writer(
                logdir=log_dir_target_critic_weights + 'y_pos')
            self.target_critic_thet_writer = tf.summary.create_file_writer(
                logdir=log_dir_target_critic_weights + 'theta')
            self.target_critic_xvel_writer = tf.summary.create_file_writer(
                logdir=log_dir_target_critic_weights + 'x_vel')
            self.target_critic_yvel_writer = tf.summary.create_file_writer(
                logdir=log_dir_target_critic_weights + 'y_vel')
            self.target_critic_tvel_writer = tf.summary.create_file_writer(
                logdir=log_dir_target_critic_weights + 'theta_vel')

    def write_logs(self, ac, reward, action, noise, critics, ep):
        if self.log_level > 0:
            self.rewards_logger(reward, ep)

        if self.log_level > 1:
            self.actions_logger(action, noise, ep)

        if self.log_level > 2:
            self.critic_logger(critics, ep)

        if self.log_level > 3:
            policy_actor_model = ac.policy_actor_net.get_weights()[0]
            policy_critic_model = ac.policy_critic_net.get_weights()[0]
            target_actor_model = ac.target_actor_net.get_weights()[0]
            target_critic_model = ac.target_critic_net.get_weights()[0]
            self.weights_logger(policy_actor_model, policy_critic_model,
                                target_actor_model, target_critic_model, ep)

    def write_ep_logs(self, ep_reward, i_episode):
        if self.log_level > 0:
            self.ep_reward_logger(ep_reward, i_episode)

    def ep_reward_logger(self, ep_reward, i_episode):
        with self.ep_reward_writer.as_default():
            tf.summary.scalar('episodic reward', ep_reward, step=i_episode)

    def rewards_logger(self, reward, ep):
        # constant reward feed
        with self.rewards_writer.as_default():
            tf.summary.scalar('reward', reward, step=ep)

    def actions_logger(self, action, noise, ep):
        # constant action feed
        with self.actions_l_writer.as_default():
            tf.summary.scalar('wheel input', action[0], step=ep)
        with self.actions_r_writer.as_default():
            tf.summary.scalar('wheel input', action[1], step=ep)

        # noise terms feed
        with self.noise_l_writer.as_default():
            tf.summary.scalar('wheel input', float(noise[0]), step=ep)
        with self.noise_r_writer.as_default():
            tf.summary.scalar('wheel input', float(noise[1]), step=ep)

    def critic_logger(self, critics, ep):
        policy_reward = critics[0]
        target_reward = critics[1]
        # constant expected reward feed
        with self.policy_critic_writer.as_default():
            tf.summary.scalar('critic estimation', policy_reward[0][0], step=ep)
        with self.target_critic_writer.as_default():
            tf.summary.scalar('critic estimation', target_reward[0][0], step=ep)


    # receives array of network weights from model.get_weights()[0], step #
    def weights_logger(self, policy_actor_model, policy_critic_model,
                       target_actor_model, target_critic_model, ep):
        weights_sum = []
        for i in range(len(policy_actor_model)):
            weights_sum.append(abs(np.sum(policy_actor_model[i])))
        with self.policy_actor_xpos_writer.as_default():
            tf.summary.scalar('policy actor input weights', weights_sum[0], step=ep)
        with self.policy_actor_ypos_writer.as_default():
            tf.summary.scalar('policy actor input weights', weights_sum[1], step=ep)
        with self.policy_actor_thet_writer.as_default():
            tf.summary.scalar('policy actor input weights', weights_sum[2], step=ep)
        with self.policy_actor_xvel_writer.as_default():
            tf.summary.scalar('policy actor input weights', weights_sum[3], step=ep)
        with self.policy_actor_yvel_writer.as_default():
            tf.summary.scalar('policy actor input weights', weights_sum[4], step=ep)
        with self.policy_actor_tvel_writer.as_default():
            tf.summary.scalar('policy actor input weights', weights_sum[5], step=ep)

        weights_sum = []
        for i in range(len(policy_critic_model)):
            weights_sum.append(abs(np.sum(policy_critic_model[i])))
        with self.policy_critic_xpos_writer.as_default():
            tf.summary.scalar('policy critic input weights', weights_sum[0], step=ep)
        with self.policy_critic_ypos_writer.as_default():
            tf.summary.scalar('policy critic input weights', weights_sum[1], step=ep)
        with self.policy_critic_thet_writer.as_default():
            tf.summary.scalar('policy critic input weights', weights_sum[2], step=ep)
        with self.policy_critic_xvel_writer.as_default():
            tf.summary.scalar('policy critic input weights', weights_sum[3], step=ep)
        with self.policy_critic_yvel_writer.as_default():
            tf.summary.scalar('policy critic input weights', weights_sum[4], step=ep)
        with self.policy_critic_tvel_writer.as_default():
            tf.summary.scalar('policy critic input weights', weights_sum[5], step=ep)

        weights_sum = []
        for i in range(len(target_actor_model)):
            weights_sum.append(abs(np.sum(target_actor_model[i])))
        with self.target_actor_xpos_writer.as_default():
            tf.summary.scalar('target actor input weights', weights_sum[0], step=ep)
        with self.target_actor_ypos_writer.as_default():
            tf.summary.scalar('target actor input weights', weights_sum[1], step=ep)
        with self.target_actor_thet_writer.as_default():
            tf.summary.scalar('target actor input weights', weights_sum[2], step=ep)
        with self.target_actor_xvel_writer.as_default():
            tf.summary.scalar('target actor input weights', weights_sum[3], step=ep)
        with self.target_actor_yvel_writer.as_default():
            tf.summary.scalar('target actor input weights', weights_sum[4], step=ep)
        with self.target_actor_tvel_writer.as_default():
            tf.summary.scalar('target actor input weights', weights_sum[5], step=ep)

        weights_sum = []
        for i in range(len(target_critic_model)):
            weights_sum.append(abs(np.sum(target_critic_model[i])))
        with self.target_critic_xpos_writer.as_default():
            tf.summary.scalar('target critic input weights', weights_sum[0], step=ep)
        with self.target_critic_ypos_writer.as_default():
            tf.summary.scalar('target critic input weights', weights_sum[1], step=ep)
        with self.target_critic_thet_writer.as_default():
            tf.summary.scalar('target critic input weights', weights_sum[2], step=ep)
        with self.target_critic_xvel_writer.as_default():
            tf.summary.scalar('target critic input weights', weights_sum[3], step=ep)
        with self.target_critic_yvel_writer.as_default():
            tf.summary.scalar('target critic input weights', weights_sum[4], step=ep)
        with self.target_critic_tvel_writer.as_default():
            tf.summary.scalar('target critic input weights', weights_sum[5], step=ep)
