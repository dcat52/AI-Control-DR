import tensorflow as tf
from tensorflow.keras import layers, optimizers
from tensorflow import nn

class Actor_Model(tf.keras.Model):
    def __init__(self, action_bounds=(-1.0, 1.0), state_length=6, action_length=2, num_layers=2, layer_width=256):
        super(Actor_Model, self).__init__()

        self.lower_bound, self.upper_bound = action_bounds
        self.state_length = state_length

        # Initialize weights between -3e-3 and 3-e3
        self.last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

        self.hl = []

        self.inp = layers.Dense(state_length, activation=nn.relu)
        for i in range(num_layers):
            layer = layers.Dense(layer_width, activation=nn.relu)
            self.hl.append(layer)
        # self.hl2 = layers.Dense(256, activation=nn.relu)
        self.out = layers.Dense(action_length, activation=nn.tanh, kernel_initializer=self.last_init)

    @tf.function
    def call(self, inputs):
        # inp = layers.Input(shape=(self.state_length,))
        s1 = self.inp(inputs)

        temp = s1

        for i in range(len(self.hl)):
            temp = self.hl[i](temp)

        # if(len(self.hl) == 0):
        #     hl_out = s1

        # s2 = self.hl1(s1)
        # s3 = self.hl2(s2)
        s4 = self.out(temp)
        return s4

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

    @tf.function
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