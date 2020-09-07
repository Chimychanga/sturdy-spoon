#Following Here's How to Code Discrete Actor Critic Methods
#https://www.youtube.com/watch?v=3gboWbqaP5A
#However porting from pytorch to keras

import numpy as np
import gym

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model

class Agent(object):
    def __init__(self):
    self.actor,self,critic = self.getModel()

    def getModel(self):
        input = keras.Input(shape=(4, 96, 96, 1))
        dense1 = layers.Dense(256, activation="relu")(input)
        dense2 = layers.Dense(256, activation="relu")(dense1)
        prob = layers.Dense(2, activation="softmax")(dense2)

        optimizer = optimizers.Adam(lr=1e3)

        model = Model(inputs=[input], outputs=[prob])
        model.compile(optimizer=optimizer, loss='mean_squared_error')

    def getAction(self, obs):
        prob = self.actor.predict(obs)
        action = np.random.choice(, p=prob)
        self.log_prob 

    def learn(self, state, action, reward, next_state, done):
        critic_value = self.critic.predict(state)
        next_critic_value = self.critic.predict(next_state)

        delta = reward + 0.99*next_critic_value*(1-int(done)) - critic_value

        actor_loss = -self.log
