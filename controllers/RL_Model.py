# RL Model for agent control

from tensorflow.python.keras import Sequential
from baselines.common.atari_wrappers import make_atari, wrap_deepmind
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import Conv2D, Flatten, Dense, Input
import tensorflow as tf

class RL_Model():

    def __init__(self):
        # Configuration paramaters for the whole setup
        self.seed = 42
        self.gamma = 0.99  # Discount factor for past rewards
        self.epsilon = 1.0  # Epsilon greedy parameter
        self.epsilon_min = 0.1  # Minimum epsilon greedy parameter
        self.epsilon_max = 1.0  # Maximum epsilon greedy parameter
        self.epsilon_interval = (
                self.epsilon_max - self.epsilon_min
        )  # Rate at which to reduce chance of random action being taken
        self.batch_size = 32  # Size of batch taken from replay buffer
        self.max_steps_per_episode = 10000

        # Use the Baseline Atari environment because of Deepmind helper functions
        self.env = make_atari("BreakoutNoFrameskip-v4")
        # Warp the frames, grey scale, stake four frame and scale to smaller ratio
        self.env = wrap_deepmind(self.env, frame_stack=True, scale=True)
        self.env.seed(self.seed)

        # dimension of output
        self.num_actions = 2

    def gpu_mem_config(self):
        physical_devices = tf.config.list_physical_devices('GPU')
        try:
          tf.config.experimental.set_memory_growth(physical_devices[0], True)
        except:
          # Invalid device or cannot modify virtual devices once initialized.
          pass

    def create_q_model(self):
        inputs = Input(shape=(2,))

        # layer1 = Conv2D(32, 8, strides=4, activation="relu")(inputs)
        # layer2 = Conv2D(64, 4, strides=2, activation="relu")(layer1)
        # layer3 = Conv2D(64, 3, strides=1, activation="relu")(layer2)
        # layer4 = Flatten()(layer3)
        # layer_out = Dense(512, activation="relu")(layer4)

        layer1 = Dense(256, activation="relu")(inputs)
        layer_out = Dense(256, activation="relu")(layer1)

        action = Dense(self.num_actions, activation="linear")(layer_out)
        return keras.Model(inputs=inputs, outputs=action)


    def train_model(self):
        # The first model makes the predictions for Q-values which are used to
        # make a action.
        model = self.create_q_model()
        # Build a target model for the prediction of future rewards.
        # The weights of a target model get updated every 10000 steps thus when the
        # loss between the Q-values is calculated the target Q-value is stable.
        model_target = self.create_q_model()
        optimizer = keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)

        # Experience replay buffers
        action_history = []
        state_history = []
        state_next_history = []
        rewards_history = []
        done_history = []
        episode_reward_history = []
        running_reward = 0
        episode_count = 0
        frame_count = 0
        # Number of frames to take random action and observe output
        epsilon_random_frames = 50000
        # Number of frames for exploration
        epsilon_greedy_frames = 1000000.0
        # Maximum replay length
        # Note: The Deepmind paper suggests 1000000 however this causes memory issues
        max_memory_length = 100000
        # Train the model after 4 actions
        update_after_actions = 4
        # How often to update the target network
        update_target_network = 10000
        # Using huber loss for stability
        loss_function = keras.losses.Huber()

        while True:  # Run until solved
            state = np.array(self.env.reset())
            episode_reward = 0

            for timestep in range(1, self.max_steps_per_episode):
                self.env.render()  # Adding this line would show the attempts
                # of the agent in a pop up window.
                frame_count += 1

                # Use epsilon-greedy for exploration
                if frame_count < epsilon_random_frames or epsilon > np.random.rand(1)[0]:
                    # Take random action
                    action = np.random.choice(self.num_actions)
                else:
                    # Predict action Q-values
                    # From environment state
                    state_tensor = tf.convert_to_tensor(state)
                    state_tensor = tf.expand_dims(state_tensor, 0)
                    action_probs = model(state_tensor, training=False)
                    # Take best action
                    action = tf.argmax(action_probs[0]).numpy()

                # Decay probability of taking random action
                epsilon -= self.epsilon_interval / epsilon_greedy_frames
                epsilon = max(epsilon, self.epsilon_min)

                # Apply the sampled action in our environment
                state_next, reward, done, _ = self.env.step(action)
                state_next = np.array(state_next)

                episode_reward += reward

                # Save actions and states in replay buffer
                action_history.append(action)
                state_history.append(state)
                state_next_history.append(state_next)
                done_history.append(done)
                rewards_history.append(reward)
                state = state_next

                # Update every fourth frame and once batch size is over 32
                if frame_count % update_after_actions == 0 and len(done_history) > self.batch_size:
                    # Get indices of samples for replay buffers
                    indices = np.random.choice(range(len(done_history)), size=self.batch_size)

                    # Using list comprehension to sample from replay buffer
                    state_sample = np.array([state_history[i] for i in indices])
                    state_next_sample = np.array([state_next_history[i] for i in indices])
                    rewards_sample = [rewards_history[i] for i in indices]
                    action_sample = [action_history[i] for i in indices]
                    done_sample = tf.convert_to_tensor(
                        [float(done_history[i]) for i in indices]
                    )

                    # Build the updated Q-values for the sampled future states
                    # Use the target model for stability
                    future_rewards = model_target.predict(state_next_sample)
                    # Q value = reward + discount factor * expected future reward
                    updated_q_values = rewards_sample + self.gamma * tf.reduce_max(
                        future_rewards, axis=1
                    )

                    # If final frame set the last value to -1
                    updated_q_values = updated_q_values * (1 - done_sample) - done_sample

                    # Create a mask so we only calculate loss on the updated Q-values
                    masks = tf.one_hot(action_sample, self.num_actions)

                    with tf.GradientTape() as tape:
                        # Train the model on the states and updated Q-values
                        q_values = model(state_sample)

                        # Apply the masks to the Q-values to get the Q-value for action taken
                        q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
                        # Calculate loss between new Q-value and old Q-value
                        loss = loss_function(updated_q_values, q_action)

                    # Backpropagation
                    grads = tape.gradient(loss, model.trainable_variables)
                    optimizer.apply_gradients(zip(grads, model.trainable_variables))

                if frame_count % update_target_network == 0:
                    # update the the target network with new weights
                    model_target.set_weights(model.get_weights())
                    # Log details
                    template = "running reward: {:.2f} at episode {}, frame count {}"
                    print(template.format(running_reward, episode_count, frame_count))

                # Limit the state and reward history
                if len(rewards_history) > max_memory_length:
                    del rewards_history[:1]
                    del state_history[:1]
                    del state_next_history[:1]
                    del action_history[:1]
                    del done_history[:1]

                if done:
                    break

            # Update running reward to check condition for solving
            episode_reward_history.append(episode_reward)
            print(episode_reward)
            if len(episode_reward_history) > 100:
                del episode_reward_history[:1]
            running_reward = np.mean(episode_reward_history)

            episode_count += 1

            if running_reward > 40:  # Condition to consider the task solved
                print("Solved at episode {}!".format(episode_count))
                return model