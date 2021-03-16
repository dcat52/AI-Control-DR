import os
import sys
import math
import random
import queue
import datetime
import logging
from collections import deque, namedtuple
from itertools import count

import tensorflow as tf
from tensorflow.keras import layers, optimizers
from tensorflow import nn
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=2, linewidth=180)

# tf.manual_seed(595)
np.random.seed(595)
random.seed(595)

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        return x

    def get_std_dev(self):
        return self.std_dev

    def set_std_dev(self, std_deviation):
        self.std_dev = std_deviation

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)


class Buffer(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class Actor_Model(tf.keras.Model):
    def __init__(self, action_bounds=(-1.0, 1.0), state_length=6, action_length=2, num_layers=2, layer_width=256):
        super(Actor_Model, self).__init__()

        self.lower_bound, self.upper_bound = action_bounds

        # Initialize weights between -3e-3 and 3-e3
        self.last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

        self.hl = []

        # self.inp = layers.Input(shape=(state_length,))
        self.inp = layers.Dense(state_length, activation=nn.relu)
        for i in range(num_layers):
            layer = layers.Dense(layer_width, activation=nn.relu)
            self.hl.append(layer)
        # self.hl2 = layers.Dense(256, activation=nn.relu)
        self.out = layers.Dense(action_length, activation=nn.tanh, kernel_initializer=self.last_init)

    def call(self, inputs):
        s1 = self.inp(inputs)

        temp = s1

        for i in range(len(self.hl)):
            temp = self.hl[i](temp)

        if(len(self.hl) == 0):
            hl_out = s1

        # s2 = self.hl1(s1)
        # s3 = self.hl2(s2)
        s4 = self.out(temp)
        s5 = s4 * self.upper_bound
        return s5


class Critic_Model(tf.keras.Model):
    def __init__(self, state_length=6, action_length=2):
        super(Critic_Model, self).__init__()
        # State as input
        # self.state_input = layers.Input(shape=(state_length))
        self.state_input = layers.Dense(state_length, activation=nn.relu)
        self.state_hl1 = layers.Dense(16, activation=nn.relu)
        self.state_out = layers.Dense(32, activation=nn.relu)

        # Action as input
        # self.action_input = layers.Input(shape=(action_length))
        self.action_input = layers.Dense(action_length, activation=nn.relu)
        self.action_out = layers.Dense(32, activation=nn.relu)

        # Both are passed through seperate layer before concatenating
        self.combined_input = layers.Concatenate()
        self.combined_hl1 = layers.Dense(256, activation=nn.relu)
        self.combined_hl2 = layers.Dense(256, activation=nn.relu)
        self.out = layers.Dense(1)

    def call(self, inputs):
        ss1 = self.state_input(inputs[0])
        ss2 = self.state_hl1(ss1)
        ss3 = self.state_out(ss2)

        as1 = self.action_input(inputs[1])
        as2 = self.action_out(as1)

        cs1 = self.combined_input([ss3, as2])
        cs2 = self.combined_hl1(cs1)
        cs3 = self.combined_hl2(cs2)
        cs4 = self.out(cs3)
        return cs4

        
class AC_Agent:
    def __init__(self, env, args):

        self.args = args
        args = vars(args)

        self.env = env

        # --------------------------------------
        # NOTE: These parameters are overwritten when run by `main.py`
        # NOTE: The defaults here match those by `main.py`
        self.PRINT_FREQ = 1
        self.WRITE_FREQ = 5
        self.SAVE_FREQ = 50
        self.BATCH_SIZE = 32
        self.BUFFER_CAPACITY = 10000
        self.NUM_EPISODES = 1000
        self.TARGET_UPDATE = 10
        self.TAU = 0.2
        self.GAMMA = 0.99
        self.STD_DEV = 0.1
        self.SAVE_PREFIX = "data"
        self.ACTOR_LR = 0.0001
        self.CRITIC_LR = 0.0002
        self.PLOT = False
        self.TENSORBOARD = False
        self.DATE_IN_PREFIX = False
        self.ACTOR_NUM_LAYERS = 2
        self.ACTOR_LAYER_WIDTH = 256
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

        if self.TENSORBOARD:
            import controllers.TBLogger as tb
            self.tb_logger = tb.TBLogger(self.SAVE_PREFIX)

        state = self.env.reset()

        self.state_length = state.shape[0]
        self.action_length = 2
        self.action_bounds = (-1.0, 1.0)
        self.lower_bound, self.upper_bound = self.action_bounds

        self.ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(self.STD_DEV) * np.ones(1))

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

    def save_weights(self, directory: str, i: int):
        print("Saving model weights.")
        self.policy_actor_net.save_weights("{}/policy_actor_net_{:03d}".format(directory, i))
        self.policy_critic_net.save_weights("{}/policy_critic_net_{:03d}".format(directory, i))
        self.target_actor_net.save_weights("{}/target_actor_net_{:03d}".format(directory, i))
        self.target_critic_net.save_weights("{}/target_critic_net_{:03d}".format(directory, i))

    def make_action(self, state, test=True):

        state = state[None, :]

        sampled_actions = tf.squeeze(self.policy_actor_net(state))
        noise = self.ou_noise()

        # Adding noise to action
        sampled_actions = sampled_actions.numpy() + noise

        # We make sure action is within bounds
        legal_action = np.clip(sampled_actions, self.lower_bound, self.upper_bound)

        return legal_action

    def train(self):

        final_episode_reward = []
        cumulative_episode_reward = []

        for i_episode in range(1, self.NUM_EPISODES+1):

            state = self.env.reset()
            episodic_reward = 0

            counter = 0

            done = False

            while not done:
                # Select and perform an action
                action = self.make_action(state, test=False)

                next_state, reward, done, info = self.env.step(action)
                done = int(done)

                episodic_reward += reward

                # TensorBoard
                if self.TENSORBOARD:
                    # actor and critic model input weights logging
                    try:
                        self.tb_logger.weights_logger(self.policy_actor_net.get_weights()[0],
                                                      self.policy_critic_net.get_weights()[0],
                                                      self.target_actor_net.get_weights()[0],
                                                      self.target_actor_net.get_weights()[0],
                                                      i_episode * 100 + counter)
                    except IndexError:
                        print('something has no weights for some reason')

                counter += 1
                if counter == 100:
                    done = True

                rewardTensor = tf.convert_to_tensor([reward])

                # Store the transition in memory
                self.buffer.push(state, action, next_state, rewardTensor)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the target network)
                self.learn()

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
                # TODO: Write desired features (i_episode, reward, etc) to disk
                with open(self.SAVE_PREFIX + "_values.csv", "a") as f:
                    f.write("{},{}\n".format(i_episode, episodic_reward))
                pass

            # Update the target network
            if i_episode % self.TARGET_UPDATE == 0:
                self.update_targets()

            if i_episode % self.SAVE_FREQ == 0:
                self.save_weights(self.SAVE_PREFIX + "_weights", i_episode)

        if self.PLOT:
            # Plotting graph
            # Episodes versus Avg. Rewards
            plt.plot(cumulative_episode_reward)
            plt.xlabel("Episode")
            plt.ylabel("Avg. Epsiodic Reward")
            plt.show()
    

    # @tf.function
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
