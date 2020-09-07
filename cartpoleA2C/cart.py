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
