import datetime
import numpy as np
import tensorflow as tf

class TBLogger:
    def __init__(self, directory_prefix="data"):
        # log directory locations
        # log_dir = directory_prefix + "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M")
        log_dir = directory_prefix + "_logs/"

        log_dir_policy_weights = log_dir + "/weights/policy"
        log_dir_target_weights = log_dir + "/weights/target"

        log_dir_policy_actor_weights = log_dir_policy_weights + "/actor"
        log_dir_policy_critic_weights = log_dir_policy_weights + "/critic"
        log_dir_target_actor_weights = log_dir_target_weights + "/actor"
        log_dir_target_critic_weights = log_dir_target_weights + "/critic"

        # set up loggers for: ep_reward, actor weights, critic weights
        self.rewards_writer = tf.summary.create_file_writer(logdir=log_dir)

        self.policy_actor_xpos_writer = tf.summary.create_file_writer(logdir=log_dir_policy_actor_weights + 'x_pos')
        self.policy_actor_ypos_writer = tf.summary.create_file_writer(logdir=log_dir_policy_actor_weights + 'y_pos')
        self.policy_actor_thet_writer = tf.summary.create_file_writer(logdir=log_dir_policy_actor_weights + 'theta')
        self.policy_actor_xvel_writer = tf.summary.create_file_writer(logdir=log_dir_policy_actor_weights + 'x_vel')
        self.policy_actor_yvel_writer = tf.summary.create_file_writer(logdir=log_dir_policy_actor_weights + 'y_vel')
        self.policy_actor_tvel_writer = tf.summary.create_file_writer(logdir=log_dir_policy_actor_weights + 'theta_vel')

        self.policy_critic_xpos_writer = tf.summary.create_file_writer(logdir=log_dir_policy_critic_weights + 'x_pos')
        self.policy_critic_ypos_writer = tf.summary.create_file_writer(logdir=log_dir_policy_critic_weights + 'y_pos')
        self.policy_critic_thet_writer = tf.summary.create_file_writer(logdir=log_dir_policy_critic_weights + 'theta')
        self.policy_critic_xvel_writer = tf.summary.create_file_writer(logdir=log_dir_policy_critic_weights + 'x_vel')
        self.policy_critic_yvel_writer = tf.summary.create_file_writer(logdir=log_dir_policy_critic_weights + 'y_vel')
        self.policy_critic_tvel_writer = tf.summary.create_file_writer(logdir=log_dir_policy_critic_weights + 'theta_vel')

        self.target_actor_xpos_writer = tf.summary.create_file_writer(logdir=log_dir_target_actor_weights + 'x_pos')
        self.target_actor_ypos_writer = tf.summary.create_file_writer(logdir=log_dir_target_actor_weights + 'y_pos')
        self.target_actor_thet_writer = tf.summary.create_file_writer(logdir=log_dir_target_actor_weights + 'theta')
        self.target_actor_xvel_writer = tf.summary.create_file_writer(logdir=log_dir_target_actor_weights + 'x_vel')
        self.target_actor_yvel_writer = tf.summary.create_file_writer(logdir=log_dir_target_actor_weights + 'y_vel')
        self.target_actor_tvel_writer = tf.summary.create_file_writer(logdir=log_dir_target_actor_weights + 'theta_vel')

        self.target_critic_xpos_writer = tf.summary.create_file_writer(logdir=log_dir_target_critic_weights + 'x_pos')
        self.target_critic_ypos_writer = tf.summary.create_file_writer(logdir=log_dir_target_critic_weights + 'y_pos')
        self.target_critic_thet_writer = tf.summary.create_file_writer(logdir=log_dir_target_critic_weights + 'theta')
        self.target_critic_xvel_writer = tf.summary.create_file_writer(logdir=log_dir_target_critic_weights + 'x_vel')
        self.target_critic_yvel_writer = tf.summary.create_file_writer(logdir=log_dir_target_critic_weights + 'y_vel')
        self.target_critic_tvel_writer = tf.summary.create_file_writer(logdir=log_dir_target_critic_weights + 'theta_vel')

    def rewards_logger(self, reward, ep):
        # constant reward feed
        with self.rewards_writer.as_default():
            tf.summary.scalar('episodic reward', reward, step=ep)

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

