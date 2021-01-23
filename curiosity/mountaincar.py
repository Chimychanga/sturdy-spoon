import gym
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

gamma = 0.99  # Discount factor for past rewards
max_steps_per_episode = 10000
env = gym.make("MountainCar-v0")  # Create the environment
eps = np.finfo(np.float32).eps.item()  # Smallest number such that 1.0 + eps != 1.0

env.reset()
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample()) # take a random action
env.close()