import random
import gym
import numpy as np
import tensorflow as tf
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

class DQNetwork():
    def __init__(self, observation_space, action_space):
        self.memory = deque(maxlen = memory_size)
    def store(self, state, next_state, action, reward, done):

    def getAction(self, state):

    def train():

def getAction(state):

env = gym.make('CartPole-v1')
action_space = env.action_space.n
observation_space = env.observation_space.shape[0]

memory_size = 10000
memory = deque(maxlen=memory_size)
