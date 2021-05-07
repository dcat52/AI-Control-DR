import os
import sys
import math
import random
import queue
import datetime
import logging
from itertools import count

import tensorflow as tf
from tensorflow.keras import layers, optimizers
import numpy as np
# import matplotlib.pyplot as plt

from src.controllers.Models import Actor_Model, Critic_Model
from src.util import OUActionNoise, Buffer, Transition

np.set_printoptions(precision=2, linewidth=180)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

class WP_Agent:
    def __init__(self, env, args):

        self.args = args
        args = vars(args)
        self.env = env
        # --------------------------------------
        self.PRINT_FREQ: int
        self.WRITE_FREQ: int
        self.SAVE_FREQ: int
        self.BATCH_SIZE: int
        self.BUFFER_CAPACITY: int
        self.NUM_EPISODES: int
        self.TARGET_UPDATE: int
        self.TAU: float
        self.GAMMA: float
        self.STD_DEV: float
        self.THETA: float
        self.SAVE_PREFIX: str
        self.ACTOR_LR: float
        self.CRITIC_LR: float
        self.PLOT: bool
        self.TENSORBOARD: int
        self.DATE_IN_PREFIX: bool
        self.ACTOR_NUM_LAYERS: int
        self.ACTOR_LAYER_WIDTH: int
        self.CRITIC_LAYER_WIDTH: int
        # --------------------------------------

        # self.START = 1.0
        # self.END = 0.025
        # self.DECAY = (self.START - self.END) / self.NUM_EPISODES

        # NOTE: This is where the `main.py` overrides variables
        # parse args and override previous variables
        for k, v in args.items():
            # using exec like this is not recommended but works
            exec("self." + k + " = " + str(v))

        if self.DATE_IN_PREFIX:
            self.DATE = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.SAVE_PREFIX += "_{}".format(self.DATE)
            logging.info("********************************")
            logging.info("********************************")
            logging.info("Using date in save directory prefix!")
            logging.info("The date is: {}".format(self.DATE))
            logging.info("The complete prefix is: ./{}*".format(self.SAVE_PREFIX))
            logging.info("********************************")
            logging.info("********************************")

        if self.TENSORBOARD > 0:
            import src.controllers.TBLogger as tb
            self.tb_logger = tb.TBLogger(log_level=self.TENSORBOARD)

        state = self.env.reset()

        self.state_length = state.shape[0]
        self.action_length = 2
        self.action_bounds = (-1.0, 1.0)
        self.lower_bound, self.upper_bound = self.action_bounds

        self.ou_noise_L = OUActionNoise(mean=np.zeros(1), std_deviation=float(self.STD_DEV) * np.ones(1),
                                        theta=float(self.THETA) * np.ones(1))
        self.ou_noise_R = OUActionNoise(mean=np.zeros(1), std_deviation=float(self.STD_DEV) * np.ones(1),
                                        theta=float(self.THETA) * np.ones(1))

        self.ac_agent = self.load_ac_agent(self.LOAD_PREFIX)
        self.policy_actor_net = Actor_Model(num_layers=self.ACTOR_NUM_LAYERS, layer_width=self.ACTOR_LAYER_WIDTH)
        self.policy_critic_net = Critic_Model(max_layer_width=self.CRITIC_LAYER_WIDTH)
        self.target_actor_net = Actor_Model(num_layers=self.ACTOR_NUM_LAYERS, layer_width=self.ACTOR_LAYER_WIDTH)
        self.target_critic_net = Critic_Model(max_layer_width=self.CRITIC_LAYER_WIDTH)

        self.update_targets(tau=1.0)

        self.actor_optimizer = optimizers.Adam(learning_rate=self.ACTOR_LR)
        self.critic_optimizer = optimizers.Adam(learning_rate=self.CRITIC_LR)

        self.buffer = Buffer(capacity=self.BUFFER_CAPACITY)

        self.steps_done = 0
        self.num_episodes = 1000
        self.count_max = 100
        self.ep_epsilon = self.num_episodes
        self.count_epsilon = self.count_max

    def save_weights(self, directory: str, i: int):
        print("Saving model weights.")
        self.policy_actor_net.save("{}/{:04d}_policy_actor_net".format(directory, i), save_format="tf")
        self.policy_critic_net.save("{}/{:04d}_policy_critic_net".format(directory, i),  save_format="tf")

    def make_waypoint(self, state):
        state = np.expand_dims(state, axis=0)
        sampled_actions = self.policy_actor_net(state)
        sampled_actions = sampled_actions.numpy()
        sampled_actions = np.squeeze(sampled_actions, axis=0)
        return sampled_actions

    def train(self):

        cumulative_episode_reward = []
        episode_lengths = []
        ep = 0

        for i_episode in range(1, self.NUM_EPISODES+1):

            state = self.env.reset()
            episodic_reward = 0
            counter = 0
            done = False
            waypoint = [0, 0]

            while not done:
                # Epsilon flag: intermittent 'noise-only' episodes
                if not i_episode % self.ep_epsilon == 0 \
                        or not counter % self.count_epsilon == 0 \
                        or not self.env.noise_option:
                    # Generate waypoint
                    waypoint = self.make_waypoint(state)
                log_waypoint = [waypoint[0], waypoint[1]]

                if i_episode/self.num_episodes == .75:
                    print("75% through training, deactivating noise exploration.")

                if self.env.noise_option or i_episode / self.num_episodes <= .75:
                    # noise = np.random.uniform(-0.5, 0.5, 2)
                    noise = [self.ou_noise_L(), self.ou_noise_R()]
                else:
                    noise = [0, 0]

                waypoint[0] = waypoint[0] + noise[0]
                waypoint[1] = waypoint[1] + noise[1]

                # Set waypoint as agents goal in env
                self.env.set_new_goal(waypoint)
                # TODO: env.get_agent_state()
                state = self.env.get_agent_state()
                # Query AC_Agent model for motor action
                action = self.ac_agent(state)

                # We make sure action is within bounds
                legal_action = np.clip(action, self.lower_bound, self.upper_bound)

                # TODO: make step() return planner state space or create alternate step func
                # TODO: make planner reward function
                next_state, reward, done, info = self.env.step(legal_action)
                done = int(done)
                episodic_reward += reward

                # TensorBoard log level 1 and 2: rewards, actions, and noise
                if self.TENSORBOARD >= 1:
                    ep += 1
                    critics = None

                    if self.TENSORBOARD >= 2:
                        log_state = np.array(state)
                        log_waypoint = np.array(legal_action)

                    if self.TENSORBOARD >= 3:
                        policy_critic_estimate = self.policy_critic_net.predict([log_state, log_waypoint])
                        target_critic_estimate = self.target_critic_net.predict([log_state, log_waypoint])
                        critics = [policy_critic_estimate, target_critic_estimate]

                    if self.TENSORBOARD >= 4:
                        self.tb_logger.weights_logger(self.policy_actor_net, self.policy_critic_net,
                                                      self.target_actor_net, self.target_critic_net, ep)

                    self.tb_logger.write_logs(self, reward, log_waypoint, noise, critics, ep)

                counter += 1
                if counter == self.count_max:
                    done = True

                rewardTensor = tf.convert_to_tensor([reward])

                # Store the transition in memory
                self.buffer.push(state, legal_action, next_state, rewardTensor)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the target network)
                self.learn()

            # Update the target network
            self.update_targets()

            # Log average episode length in Tensorboard ('steps until goal reached')
            if self.TENSORBOARD >= 1:
                self.tb_logger.write_ep_reward_logs(episodic_reward, i_episode)
                self.tb_logger.write_ep_steps_logs(counter, i_episode)

            cumulative_episode_reward.append(episodic_reward)
            episode_lengths.append(counter)

            # Printing status to console
            if i_episode % self.PRINT_FREQ == 0:
                # Mean reward of last 40 episodes if agent failed to reach goal by end of episode
                if counter == self.count_max:
                    avg_reward = np.mean(cumulative_episode_reward[-10:])
                    print("Episode: {:3d} -- Failed: Current Avg Reward: {:9.5f} -- Episode Moving Avg Reward is: {:9.2f}".format(
                        i_episode, episodic_reward/counter, avg_reward))
                else:
                    # Mean steps to goal of last 40 episodes
                    avg_length = np.mean(episode_lengths[-10:])
                    print("Episode: {:3d} -- Success! Current Steps to Reach Goal: {:9.5f} -- Moving Avg Steps Required is: {:9.2f}".format(
                        i_episode, counter, avg_length))

            if i_episode % self.SAVE_FREQ == 0:
                self.save_weights(self.SAVE_PREFIX + "_weights", i_episode)

            if i_episode % self.WRITE_FREQ == 0:
                with open(self.SAVE_PREFIX + "_values.csv", "a") as f:
                    f.write("{},{}\n".format(i_episode, episodic_reward))
                pass

    def test(self):
        # Load saved model weights
        # data_weights/0550_policy_
        self.policy_actor_net = tf.keras.models.load_model(self.LOAD_PREFIX)
        # self.critic_actor_net = tf.keras.models.load_model('{}critic_net'.format(self.LOAD_PREFIX))

        self.policy_actor_net.compile()
        cumulative_episode_reward = []
        episode_lengths = []
        ep = 0

        for i_episode in range(1, self.NUM_EPISODES+1):

            state = self.env.reset()
            episodic_reward = 0
            counter = 0
            done = False

            while not done:
                action = self.make_waypoint(state)
                log_action = [action[0], action[1]]

                noise = [0, 0]

                action[0] = action[0] + noise[0]
                action[1] = action[1] + noise[1]
                # We make sure action is within bounds
                legal_action = np.clip(action, self.lower_bound, self.upper_bound)
                next_state, reward, done, info = self.env.step(legal_action)
                done = int(done)
                episodic_reward += reward

                # TensorBoard log level 1 and 2: rewards, actions, and noise
                if self.TENSORBOARD >= 1:
                    ep += 1
                    critics = None

                    if self.TENSORBOARD >= 2:
                        log_state = np.array(state)
                        log_action = np.array(legal_action)

                    if self.TENSORBOARD >= 3:
                        policy_critic_estimate = self.policy_critic_net.predict([log_state, log_action])
                        target_critic_estimate = self.target_critic_net.predict([log_state, log_action])
                        critics = [policy_critic_estimate, target_critic_estimate]

                    if self.TENSORBOARD >= 4:
                        self.tb_logger.weights_logger(self.policy_actor_net, self.policy_critic_net,
                                                      self.target_actor_net, self.target_critic_net, ep)

                    self.tb_logger.write_logs(self, reward, log_action, noise, critics, ep)

                counter += 1
                if counter == self.count_max:
                    done = True

                rewardTensor = tf.convert_to_tensor([reward])

                # Store the transition in memory
                self.buffer.push(state, legal_action, next_state, rewardTensor)

                # Move to the next state
                state = next_state

            # Log average episode length in Tensorboard ('steps until goal reached')
            if self.TENSORBOARD >= 1:
                self.tb_logger.write_ep_reward_logs(episodic_reward, i_episode)
                self.tb_logger.write_ep_steps_logs(counter, i_episode)

            cumulative_episode_reward.append(episodic_reward)
            episode_lengths.append(counter)

            if i_episode % self.PRINT_FREQ == 0:
                # Mean reward of last 40 episodes if agent failed to reach goal by end of episode
                if counter == self.count_max:
                    avg_reward = np.mean(cumulative_episode_reward[-10:])
                    print("Episode: {:3d} -- Failed: Current Avg Reward: {:9.5f} -- Episode Moving Avg Reward is: {:9.2f}".format(
                        i_episode, episodic_reward/counter, avg_reward))
                else:
                    # Mean steps to goal of last 40 episodes
                    avg_length = np.mean(episode_lengths[-10:])
                    print("Episode: {:3d} -- Success! Current Steps to Reach Goal: {:9.5f} -- Moving Avg Steps Required is: {:9.2f}".format(
                        i_episode, counter, avg_length))

            if i_episode % self.WRITE_FREQ == 0:
                with open(self.SAVE_PREFIX + "_values.csv", "a") as f:
                    f.write("{},{}\n".format(i_episode, episodic_reward))
                pass

    @tf.function
    def update(self, state_batch, action_batch, reward_batch, next_state_batch,):
        # Training and updating Actor & Critic networks.
        with tf.GradientTape() as tape:
            target_actions = self.target_actor_net(next_state_batch, training=True)
            y = reward_batch + self.GAMMA * self.target_critic_net([next_state_batch, target_actions], training=True)
            critic_value = self.policy_critic_net([state_batch, action_batch], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, self.policy_critic_net.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grad, self.policy_critic_net.trainable_variables))

        with tf.GradientTape() as tape:
            actions = self.policy_actor_net(state_batch, training=True)
            critic_value = self.policy_critic_net([state_batch, actions], training=True)
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, self.policy_actor_net.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grad, self.policy_actor_net.trainable_variables))

    def learn(self):
        # Retrieves a batch of training samples from the buffer and begins learning process
        if len(self.buffer) < self.BATCH_SIZE:
            return

        batch_indices = self.buffer.sample(self.BATCH_SIZE)
        batch = Transition(*zip(*batch_indices))

        state_batch = tf.convert_to_tensor(batch.state)
        action_batch = tf.convert_to_tensor(batch.action)
        reward_batch = tf.convert_to_tensor(batch.reward)
        next_state_batch = tf.convert_to_tensor(batch.next_state)

        self.update(state_batch, action_batch, reward_batch, next_state_batch)

    def update_targets(self, tau=None):
        if tau is None:
            tau = self.TAU

        def update(target_weights, weights, tau):
            for (a, b) in zip(target_weights, weights):
                a.assign(b * tau + a * (1 - tau))

        update(self.target_actor_net.variables, self.policy_actor_net.variables, tau)
        update(self.target_critic_net.variables, self.policy_critic_net.variables, tau)

    def load_ac_agent(self):
        model = tf.keras.model.load(self.LOAD_PREFIX)
        return model
