import datetime
import numpy as np
import tensorflow as tf

#
class TBLogger:
    def __init__(self):
        # log directory locations
        log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M")
        log_dir_actor_weights = log_dir + "/weights/actor"
        log_dir_critic_weights = log_dir + "/weights/critic"

        # set up loggers for: ep_reward, actor weights, critic weights
        self.rewards_writer = tf.summary.create_file_writer(logdir=log_dir)

        self.actor_xpos_writer = tf.summary.create_file_writer(logdir=log_dir_actor_weights + 'x_pos')
        self.actor_ypos_writer = tf.summary.create_file_writer(logdir=log_dir_actor_weights + 'y_pos')
        self.actor_thet_writer = tf.summary.create_file_writer(logdir=log_dir_actor_weights + 'theta')
        self.actor_xvel_writer = tf.summary.create_file_writer(logdir=log_dir_actor_weights + 'x_vel')
        self.actor_yvel_writer = tf.summary.create_file_writer(logdir=log_dir_actor_weights + 'y_vel')
        self.actor_tvel_writer = tf.summary.create_file_writer(logdir=log_dir_actor_weights + 'theta_vel')

        self.critic_xpos_writer = tf.summary.create_file_writer(logdir=log_dir_critic_weights + 'x_pos')
        self.critic_ypos_writer = tf.summary.create_file_writer(logdir=log_dir_critic_weights + 'y_pos')
        self.critic_thet_writer = tf.summary.create_file_writer(logdir=log_dir_critic_weights + 'theta')
        self.critic_xvel_writer = tf.summary.create_file_writer(logdir=log_dir_critic_weights + 'x_vel')
        self.critic_yvel_writer = tf.summary.create_file_writer(logdir=log_dir_critic_weights + 'y_vel')
        self.critic_tvel_writer = tf.summary.create_file_writer(logdir=log_dir_critic_weights + 'theta_vel')

    def rewards_logger(self, reward, ep):
        # constant reward feed
        with self.rewards_writer.as_default():
            tf.summary.scalar('episodic reward', reward, step=ep)

    # receives array of network weights from model.get_weights()[0], step #
    def weights_logger(self, actor_model, critic_model, ep):
        weights_sum = []
        for i in range(len(actor_model)):
            weights_sum.append(abs(np.sum(actor_model[i])))
        with self.actor_xpos_writer.as_default():
            tf.summary.scalar('actor input weights', weights_sum[0], step=ep)
        with self.actor_ypos_writer.as_default():
            tf.summary.scalar('actor input weights', weights_sum[1], step=ep)
        with self.actor_thet_writer.as_default():
            tf.summary.scalar('actor input weights', weights_sum[2], step=ep)
        with self.actor_xvel_writer.as_default():
            tf.summary.scalar('actor input weights', weights_sum[3], step=ep)
        with self.actor_yvel_writer.as_default():
            tf.summary.scalar('actor input weights', weights_sum[4], step=ep)
        with self.actor_tvel_writer.as_default():
            tf.summary.scalar('actor input weights', weights_sum[5], step=ep)

        weights_sum = []
        for i in range(len(critic_model)):
            weights_sum.append(abs(np.sum(critic_model[i])))
        with self.critic_xpos_writer.as_default():
            tf.summary.scalar('critic input weights', weights_sum[0], step=ep)
        with self.critic_ypos_writer.as_default():
            tf.summary.scalar('critic input weights', weights_sum[1], step=ep)
        with self.critic_thet_writer.as_default():
            tf.summary.scalar('critic input weights', weights_sum[2], step=ep)
        with self.critic_xvel_writer.as_default():
            tf.summary.scalar('critic input weights', weights_sum[3], step=ep)
        with self.critic_yvel_writer.as_default():
            tf.summary.scalar('critic input weights', weights_sum[4], step=ep)
        with self.critic_tvel_writer.as_default():
            tf.summary.scalar('critic input weights', weights_sum[5], step=ep)

