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
import matplotlib.pyplot as plt

from src.controllers.Models import Actor_Model, Critic_Model
from src.util import OUActionNoise, Buffer, Transition

np.set_printoptions(precision=2, linewidth=180)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

class AC_Agent:
    def __init__(self, env, args):

        self.args = args
        args = vars(args)

        self.env = env

        # --------------------------------------
        # NOTE: These parameters are overwritten when run by `main.py`
        # NOTE: The defaults here match those by `main.py`
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
        self.count_max: int
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

        self.policy_actor_net = Actor_Model(num_layers=self.ACTOR_NUM_LAYERS, layer_width=self.ACTOR_LAYER_WIDTH)
        self.policy_critic_net = Critic_Model()
        self.target_actor_net = Actor_Model(num_layers=self.ACTOR_NUM_LAYERS, layer_width=self.ACTOR_LAYER_WIDTH)
        self.target_critic_net = Critic_Model()

        self.update_targets(tau=1.0)

        self.actor_optimizer = optimizers.Adam(learning_rate=self.ACTOR_LR)
        self.critic_optimizer = optimizers.Adam(learning_rate=self.CRITIC_LR)

        self.buffer = Buffer(capacity=self.BUFFER_CAPACITY)

        self.steps_done = 0
        self.episode_durations = []
        self.count_max = 10000

    def save_weights(self, directory: str, i: int):
        print("Saving model weights.")
        # self.policy_actor_net.to_json("{}/{:04d}_policy_actor_net.json".format(directory, i))
        # self.policy_actor_net.to_json("{}/{:04d}_policy_critic_net.json".format(directory, i))
        self.policy_actor_net.save("{}/{:04d}_policy_actor_net".format(directory, i), save_format="tf")
        # self.policy_critic_net.save("{}/{:04d}_policy_critic_net".format(directory, i),  save_format="tf")

    def make_action(self, state):

        state = state[None, :]

        sampled_actions = tf.squeeze(self.policy_actor_net(state))
        sampled_actions = sampled_actions.numpy()

        return sampled_actions

    def train(self):

        cumulative_episode_reward = []
        ep = 0

        for i_episode in range(1, self.NUM_EPISODES+1):

            state = self.env.reset()
            episodic_reward = 0
            counter = 0
            done = False

            while not done:
                # Select and perform an action
                action = self.make_action(state)
                log_action = [action[0], action[1]]
                noise = [self.ou_noise_L(), self.ou_noise_R()]

                action[0] = action[0] + noise[0]
                action[1] = action[1] + noise[1]
                # We make sure action is within bounds
                legal_action = np.clip(action, self.lower_bound, self.upper_bound)

                next_state, reward, done, info = self.env.step(legal_action)
                done = int(done)
                episodic_reward += reward

                # TensorBoard
                if self.TENSORBOARD >= 1:
                    ep += 1
                    critics = None

                    # log level 2
                    if self.TENSORBOARD >= 2:
                        state = np.array([state])
                        legal_action = np.array([legal_action])
                        policy_critic_estimate = self.policy_critic_net.predict([state, legal_action])
                        target_critic_estimate = self.target_critic_net.predict([state, legal_action])
                        critics = [policy_critic_estimate, target_critic_estimate]

                    if self.TENSORBOARD >= 3:
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

                # Perform one step of the optimization (on the target network)
                self.learn()

                # Update the target network
                if counter % self.TARGET_UPDATE == 0:
                    self.update_targets()

            cumulative_episode_reward.append(episodic_reward)

            if i_episode % self.PRINT_FREQ == 0:
                # Mean of last 40 episodes
                avg_reward = np.mean(cumulative_episode_reward[-10:])
                print("Episode: {:3d} -- Current Reward: {:9.2f} -- Avg Reward is: {:9.2f}".format(
                    i_episode, episodic_reward, avg_reward
                    ))

            if i_episode % self.WRITE_FREQ == 0:
                with open(self.SAVE_PREFIX + "_values.csv", "a") as f:
                    f.write("{},{}\n".format(i_episode, episodic_reward))
                pass

            if i_episode % self.SAVE_FREQ == 0:
                self.save_weights(self.SAVE_PREFIX + "_weights", i_episode)

        if self.PLOT:
            # Plotting graph
            # Episodes versus Avg. Rewards
            plt.plot(cumulative_episode_reward)
            plt.xlabel("Episode")
            plt.ylabel("Avg. Epsiodic Reward")
            plt.show()

    def test(self):
        # Load saved model weights
        # data_weights/0550_policy_
        self.policy_actor_net = tf.keras.models.load_model('{}actor_net'.format(self.LOAD_PREFIX))
        self.critic_actor_net = tf.keras.models.load_model('{}critic_net'.format(self.LOAD_PREFIX))

        final_episode_reward = []
        cumulative_episode_reward = []

        for i_episode in range(1, self.NUM_EPISODES+1):

            state = self.env.reset()
            episodic_reward = 0
            counter = 0
            done = False

            while not done:
                # Select and perform an action
                action = self.make_action(state)

                next_state, reward, done, info = self.env.step(action)
                done = int(done)

                # TODO: log model outputs

                episodic_reward += reward

                counter += 1
                if counter == self.count_max:
                    done = True

                # Move to the next state
                state = next_state

            final_episode_reward.append(reward)
            cumulative_episode_reward.append(episodic_reward)

            # TensorBoard logging for episodic reward
            if self.TENSORBOARD:
                self.tb_logger.rewards_logger(episodic_reward, i_episode)

            if i_episode % self.PRINT_FREQ == 0:
                # Mean of last 40 episodes
                avg_reward = np.mean(cumulative_episode_reward[-10:])
                print("Episode: {:3d} -- Current Reward: {:9.2f} -- Avg Reward is: {:9.2f}".format(
                    i_episode, episodic_reward, avg_reward
                    ))

            if i_episode % self.WRITE_FREQ == 0:
                with open(self.SAVE_PREFIX + "_values.csv", "a") as f:
                    f.write("{},{}\n".format(i_episode, episodic_reward))
                pass

        if self.PLOT:
            # Plotting graph
            # Episodes versus Avg. Rewards
            plt.plot(cumulative_episode_reward)
            plt.xlabel("Episode")
            plt.ylabel("Avg. Epsiodic Reward")
            plt.show()

    @tf.function
    def update(self, state_batch, action_batch, reward_batch, next_state_batch,):
        # Training and updating Actor & Critic networks.
        # See Pseudo Code.
        with tf.GradientTape() as tape:
            target_actions = self.target_actor_net(next_state_batch, training=True)
            y = reward_batch + self.GAMMA * self.target_critic_net(
                [next_state_batch, target_actions], training=True
            )
            critic_value = self.policy_critic_net([state_batch, action_batch], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, self.policy_critic_net.trainable_variables)
        self.critic_optimizer.apply_gradients(
            zip(critic_grad, self.policy_critic_net.trainable_variables)
        )

        with tf.GradientTape() as tape:
            actions = self.policy_actor_net(state_batch, training=True)
            critic_value = self.policy_critic_net([state_batch, actions], training=True)
            # Used `-value` as we want to maximize the value given
            # by the critic for our actions
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, self.policy_actor_net.trainable_variables)
        self.actor_optimizer.apply_gradients(
            zip(actor_grad, self.policy_actor_net.trainable_variables)
        )

    def learn(self):
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
