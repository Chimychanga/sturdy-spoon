import gym
import numpy as np
import copy

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K

import tensorboard
from tensorflow.keras.utils import plot_model

class Memory():
    def __init__(self, max_size, input_shape):
        self.max_size = max_size
        self.input_shape = input_shape[0]
        self.mem_size = 0

        self.states = np.zeros((self.max_size, self.input_shape), dtype=np.float32)
        self.next_states = np.zeros((self.max_size, self.input_shape), dtype=np.float32)
        self.actions = np.zeros((self.max_size), dtype=np.int32)
        self.rewards = np.zeros((self.max_size), dtype=np.float32)
        self.dones = np.zeros((self.max_size), dtype=np.int32)

    def store(self, state, next_state, action, reward, done):
        index = self.mem_size % self.max_size
        self.states[index] = np.array(state).reshape(-1,4)
        self.next_states[index] = np.array(next_state).reshape(-1,4)
        self.actions[index] = action
        self.rewards[index] = reward
        self.dones[index] = done
        self.mem_size += 1

    def print(self, size):
        print("Printing top " + str(size) + " in Memory: ")
        for index in range(size):
            print('[{}, {}, {}, {}, {}]'.format(self.states[index],
                                                self.next_states[index],
                                                self.actions[index],
                                                self.rewards[index],
                                                self.dones[index]))
        print("--------------------------")

    def sample(self, batch_size):
        if batch_size > self.mem_size:
            return None

        index = np.random.choice(self.mem_size, batch_size, replace=False)
        return self.states[index],\
            self.next_states[index],\
            self.actions[index],\
            self.rewards[index],\
            self.dones[index]

class Agent():
    def __init__(self, state_shape, action_shape):
        self.state_shape = state_shape
        self.action_shape = action_shape

        self.model = self.dqn(self.state_shape, self.action_shape)
        self.model.compile(optimizer=keras.optimizers.Adam(), loss="mse")
        self.model.summary()
        plot_model(self.model, to_file='dqn_model_plot.png', show_shapes=True, show_layer_names=True)

    def dqn(self, state_shape, action_shape):
        model= keras.Sequential()
        model.add(keras.Input(shape=state_shape))
        model.add(layers.Dense(12, activation="relu"))
        model.add(layers.Dense(action_shape, activation='softmax'))
        return model
