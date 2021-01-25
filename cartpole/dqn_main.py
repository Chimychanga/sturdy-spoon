from dqn import Memory, Agent

import gym
import numpy as np
import copy

#import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K

import tensorboard
from tensorflow.keras.utils import plot_model

# Discount factor for past rewards
gamma = 0.99
# Max steps per game
max_steps = 200
# number of training steps
n_games = 100
# Create the environment
env = gym.make("CartPole-v0")
# Smallest number such that 1.0 + eps != 1.0
eps = np.finfo(np.float32).eps.item()

def test_mem():
    print("##################Running memory test##################")
    mem = Memory(10000, state_shape)
    mem.store([1,1,1,1], [2,2,2,2], 1, 2, 3)
    mem.print(4)
    mem.store([3,3,3,3], [4,4,4,4], 4, 5, 6)
    mem.print(4)
    mem.store([5,5,5,5], [6,6,6,6], 7, 8, 9)
    mem.print(4)
    mem.store([7,7,7,7], [8,8,8,8], 10, 11, 12)
    mem.print(4)
    samp = mem.sample(2)
    assert samp != None, "Failed mem test - No sample recieved"
    for state in samp[0]:
        assert state in mem.states, "Failed mem test - sample failed states"
    for next_state in samp[1]:
        print(next_state)
        assert next_state in mem.next_states, "Failed mem test - sample failed next_states"
    for action in samp[2]:
        assert action in mem.actions, "Failed mem test - sample failed actions"
    for reward in samp[3]:
        assert reward in mem.rewards, "Failed mem test - sample failed rewards"
    for done in samp[4]:
        assert done in mem.dones, "Failed mem test - sample failed dones"
    print("Memory test passed :)\n\n")

def test_agent():
    print("##################Running agent test##################")
    agent = Agent(state_shape, action_shape)
    state1 = np.array([1,2,3,4]).reshape(-1,4)
    state2 = np.array([2,3,4,5]).reshape(-1,4)
    out1 = agent.model.predict(state1)
    out2 = agent.model.predict(state2)
    print(out1)
    print(out2)
    assert np.all((out1[0]-out2[0])!=0), "Failed agent test - same output different state"

    print("Agent test passed :)\n\n")

state_shape=env.observation_space.shape # the state space
action_shape=env.action_space.n # the action space

#Testing Memory storage and sample
state = env.reset()
test_mem()
test_agent()

mem = Memory(10000, state_shape)
agent = Agent(state_shape, action_shape)

action = env.action_space.sample()
next_state, reward, done, info = env.step(action)

epsilon = 1
for game in range(n_games):
    state = env.reset()
    for step in range(max_steps):
        env.render()
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(agent.model.predict(state))

        next_state, reward, done, info = env.step(action)

        state = np.array(state).reshape(-1,4)
        print(agent.model.predict(state))

        state = next_state
        if done:
            print("Episode finished after {} timesteps".format(step+1))
            break
env.close()
