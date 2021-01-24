#Actor Critic Method
#https://keras.io/examples/rl/actor_critic_cartpole/

import gym
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

gamma = 0.99  # Discount factor for past rewards
max_steps_per_episode = 10000
env = gym.make("CartPole-v0")  # Create the environment
eps = np.finfo(np.float32).eps.item()  # Smallest number such that 1.0 + eps != 1.0

n_input = 4
#Left and Right
n_actions = 2

inputs = layers.Input(shape=(num_inputs,))
dense = layers.Dense(num_hidden, activation="relu")(inputs)

action = layers.Dense(num_actions, activation="softmax")(dense)
critic = layers.Dense(1)(common)

model = keras.Model(inputs=inputs, outputs=[action, critic])

def Agent():
	