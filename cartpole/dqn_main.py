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
n_games = 1000
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

epsilon = 1
batch_size = 64

#action = env.action_space.sample()
#next_state, reward, done, info = env.step(action)

for game in range(n_games):

    state = env.reset()
    game_reward = 0

    for step in range(max_steps):
        #Render game
        if game % 10 == 0:
            env.render()

        #Choose action depending on eps
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            state = np.array(state).reshape(-1,4)
            action = np.argmax(agent.model.predict(state))

        next_state, reward, done, info = env.step(action)

        mem.store(state, next_state, action, reward, done)

        #state = np.array(state).reshape(-1,4)

        #print(agent.model.predict(state))
        #print(np.argmax(agent.model.predict(state)))
        #print(mem.sample(1)[0])
        #print(agent.model.predict(mem.sample(1)[0]))
        state = next_state
        if done:
            print("Episode finished after {} timesteps with {} epsilon".format(step+1, epsilon))
            epsilon = epsilon *0.95
            env.close()
            break

    batch = mem.sample(batch_size)
    if batch == None:
        continue

    #reminder: next_state, reward, done are post-action
    #state and action are pre-action
    batch_state = batch[0]
    batch_next_state = batch[1]
    batch_actions = batch[2]
    batch_rewards = batch[3]
    batch_dones = batch[4]

    batch_state_q_val = agent.model.predict(batch_state)
    batch_next_state_q_val = agent.model.predict(batch_next_state)

    target = batch_state_q_val.copy()

    #fix dones and create target
    for i in range(batch_size):
        action = batch_actions[i]
        reward = batch_rewards[i]
        done = batch_dones[i]
        if done:
            reward = 0

        target[i][action] += 0.9 * (reward + (0.95 * np.max(batch_next_state_q_val[i])) - target[i][action])

    # q = [1,0]
    # q = np.array(q).reshape(-1,2)
    # print(mem.mem_size)
    # state = batch_state[0].reshape(-1,4)
    # print(agent.model.predict(state))
    agent.model.fit(batch_state, target, verbose=0)
    # print(agent.model.predict(state))

env.close()
