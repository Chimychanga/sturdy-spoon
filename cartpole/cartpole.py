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
        self.action_space = action_space
        self.observation_space = observation_space

        self.model = Sequential()
        self.model.add(Dense(self.observation_space, input_shape=(self.observation_space,), activation="relu"))
        self.model.add(Dense(self.action_space, activation = "linear"))
        self.model.compile(loss="mse", optimizer=Adam(lr=0.001))


def getAction(state, epsilon, test):
    if test:
        return DQNetwork.model.predict(state)
    else:
        if np.random.rand() < epsilon:
            return np.argmax(DQNetwork.model.predict(state)), epsilon*(1-decay_rate)
        else:
            return env.action_space.sample(), epsilon*(1-decay_rate)

def store(state, next_state, action, reward, done):
    memory.append((state, next_state, action, reward, done))

env = gym.make('CartPole-v1')
action_space = env.action_space.n
observation_space = env.observation_space.shape[0]

memory_size = 10000
memory = deque(maxlen=memory_size)

episodes = 100
max_step = 200

batch_size = 64
gamma = 0.95

epsilon = 1
decay_rate = 0.005

DQNetwork = DQNetwork(observation_space, action_space)

for episode in range(episodes):
    state = env.reset()
    state = np.reshape(state, [1, observation_space])
    episode_reward = 0
    for step in range(max_step):
        action,epsilon = getAction(state, epsilon, False)
        next_state, reward, done, _ = env.step(action)

        next_state = np.reshape(next_state, [1, observation_space])
        episode_reward += reward

        store(state, next_state, action, reward, done)

        if done:
            print(  'Episode: {}'.format(episode),
                    'Episode_Reward: {}'.format(episode_reward),
                    'Epsilon: {}'.format(epsilon))
            break


        #--------------------------------------------
        if len(memory) < batch_size:
            continue
        batch = random.sample(memory, batch_size)
        for state, next_state, action, reward, done in batch:
            if done:
                targetQ = reward
            else:
                targetQ = reward + gamma *(np.max(DQNetwork.model.predict(next_state)))
        # Careful the difference between targetQ and targetQs
            targetQs = DQNetwork.model.predict(state)
            targetQs[0][action] = targetQ
            DQNetwork.model.fit(state, targetQs, verbose=0)
        #--------------------------------------------


        state = next_state
