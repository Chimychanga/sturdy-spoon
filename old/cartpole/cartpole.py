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
		self.model = Sequential()
		self.model.add(Dense(observation_space, input_shape=(observation_space,), activation="relu"))
		self.model.add(Dense(action_space, activation="linear"))
		self.model.compile(loss="mse", optimizer=Adam(lr=0.01))


def getAction(state, epsilon, test):
	if test:
		return np.argmax(DQNetwork.model.predict(state)[0])
	else:
		if np.random.rand() < epsilon:
			action = env.action_space.sample()
			#print(action)
			return action
		else:
			#Q = DQNetwork.model.predict(state)[0]
			#print(Q)
			#print(np.argmax(Q))
			return np.argmax(DQNetwork.model.predict(state)[0])

def store(state, next_state, action, reward, done):
	memory.append((state, next_state, action, reward, done))

env = gym.make('CartPole-v1')
action_space = env.action_space.n
observation_space = env.observation_space.shape[0]

memory_size = 10000
memory = deque(maxlen=memory_size)

episodes = 70
max_step = 200

batch_size = 32
gamma = 0.95

epsilon_max = 1
epsilon_min = 0.01
epsilon = epsilon_max

decay_rate = 0.001

DQNetwork = DQNetwork(observation_space, action_space)

total_step = 0

for episode in range(episodes):
	state = env.reset()
	state = np.reshape(state, [1, observation_space])
	episode_reward = 0
	for step in range(max_step):
		action = getAction(state, epsilon, False)
		next_state, reward, done, _ = env.step(action)
		next_state = np.reshape(next_state, [1, observation_space])
		episode_reward += reward
		total_step += 1
		if done:
			reward = -reward


		store(state, next_state, action, reward, done)

		if done or step == max_step-1:
			print(  'Episode: {}'.format(episode),
					'Episode_Reward: {}'.format(episode_reward),
					'Epsilon: {}'.format(epsilon))
			break

		state = next_state

		#--------------------------------------------
		if len(memory) < batch_size:
			continue
		batch = random.sample(memory, batch_size)
		targetQs_list = []
		for state_temp, next_state_temp, action_temp, reward_temp, done_temp in batch:
			if done_temp:
				targetQ = reward_temp
			else:
				targetQ = (reward_temp + gamma * (np.max(DQNetwork.model.predict(next_state_temp))))
		# Careful the difference between targetQ and targetQs
			#print("---------------")
			#print(action)
			#print(reward)
			#print(DQNetwork.model.predict(next_state))
			targetQs = DQNetwork.model.predict(state_temp)
			#print(targetQs)
			targetQs[0][action_temp] = targetQ
			targetQs_list.append(targetQs[0])
			#print(targetQs)
			#print("---------------")
		states = np.array([each[0][0] for each in batch])
		states = np.reshape(states, [batch_size, observation_space])
		#print(targetQs_list)
		targetQs_list = np.reshape(targetQs_list, [batch_size, action_space	])

		#print(states.shape)
		#print(states)
		#states = np.reshape(state, [1, observation_space])
		#print(targetQs_list)
		#exit()
		DQNetwork.model.fit(states, targetQs_list, verbose=0)
		#epsilon = max(epsilon*(1-decay_rate), 0.01)
		epsilon = epsilon_min + (epsilon_max - epsilon_min) * np.exp(-decay_rate * total_step)
		#--------------------------------------------

for episode in range(10):
	state = env.reset()
	state = np.reshape(state, [1, observation_space])
	episode_reward = 0
	for step in range(max_step):
		action = getAction(state, None, True)
		next_state, reward, done, _ = env.step(action)
		next_state = np.reshape(next_state, [1, observation_space])

		episode_reward += reward

		#store(state, next_state, action, reward, done)
		state = next_state

		if done:
			print(  'Episode: {}'.format(episode),
					'Episode_Reward: {}'.format(episode_reward))
			break
