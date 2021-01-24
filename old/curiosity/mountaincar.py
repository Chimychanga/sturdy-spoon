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
gamma = 0.99  # Discount factor for past rewards
max_steps = 200
n_games = 100
env = gym.make("MountainCar-v0")  # Create the environment
eps = np.finfo(np.float32).eps.item()  # Smallest number such that 1.0 + eps != 1.0


class ICM():
	def __init__(self, env=env, n_games = n_games, max_steps = max_steps, beta=0.01, lmd=0.99):
		self.env = env
		self.n_games = n_games
		self.max_steps = max_steps
		self.beta = beta
		self.lmd = lmd

		self.state_shape=env.observation_space.shape # the state space
		self.action_shape=env.action_space.n # the action space
		print(self.action_shape)
		print(self.state_shape)

		self.batch_size = 32
		self.model=self.build_icm_model()
		self.model.compile(optimizer=keras.optimizers.Adam(), loss="mse")
		self.model.summary()

		#logdir="logs/fit/"
		#tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
		#tensorboard_callback.set_model(model=self.model)
		plot_model(self.model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
		self.positions=np.zeros((self.n_games,2)) #record learning process
		self.rewards=np.zeros(self.n_games) #record learning process

	def one_hot_encode(self, action):
		one_hot = np.zeros(self.action_shape, np.float32)
		one_hot[action] = 1
		return one_hot

	#tries to predict action from state(n) and state(n+1)
	#used to determine which aspect of state is caused by action
	#stops procrastination...wish I had this in my brain
	def inverse_model(self, output_dim=3):
		def func(s_t, s_t1):
			input = K.concatenate([s_t, s_t1])
			dense = layers.Dense(24, activation = 'relu')(input)
			out = layers.Dense(output_dim, activation = 'sigmoid', name='a_t_hat')(dense)
			return out
		return func

	#tries to predict next state from current state and action
	def forward_model(self, output_dim = 2):
		def func(s_t, a_t):
			input = K.concatenate([s_t, a_t])
			dense = layers.Dense(24, activation = 'relu')(input)
			out = layers.Dense(output_dim, activation = 'linear', name='fv_t1_hat')(dense)
			return out
		return func

	#Both forward and inverse_model
	def build_icm_model(self, state_shape=(2,), action_shape=(3,)):
		s_t = keras.Input(shape=state_shape, name="state_t")
		s_t1 = keras.Input(shape=state_shape, name="state_t1")
		a_t = keras.Input(shape=action_shape, name="action")
		rwd= keras.Input(shape=(1,), name="ext_reward")

		reshape=layers.Reshape(target_shape= (2,))

		#extract feature vect of state, not much use here? since state is simple
		#can't expect simple net(forward_model) to predict the exact next state
		#better used for images i.e conv
		model= keras.Sequential(name='feat_vect')
		model.add(layers.Dense(24, input_shape=state_shape, activation="relu"))
		model.add(layers.Dense(12, activation="relu"))
		model.add(layers.Dense(2, activation='linear', name='feature'))

		fv_t=model(reshape(s_t))
		fv_t1=model(reshape(s_t1))

		#output predicted
		a_t_hat=self.inverse_model()(fv_t, fv_t1)
		fv_t1_hat=self.forward_model()(fv_t, a_t)

		#squared error of next state vs predicted next state
		int_reward=layers.Lambda(lambda x: 0.5 * K.sum(K.square(x[0] - x[1]),
							axis=-1),
							output_shape=(1,),
							name="reward_intrinsic")([fv_t1, fv_t1_hat])

		#error of action vs predicted action e.g.
		#[1,0,0]*log([0.8, 0.1, 0.1]) == [1,0,0]*[-0.1. -1. -1] == -0.1
		inv_loss=layers.Lambda(lambda x: -K.sum(x[0] * K.log(x[1] + K.epsilon()),
							axis=-1),
							output_shape=(1,))([a_t, a_t_hat])

		#combine both loss with ratio int_reward:inv_loss 0.01:0.99
		loss = layers.Lambda(lambda x: self.beta * x[0] + (1.0 - self.beta) * x[1],
							output_shape=(1,))([int_reward, inv_loss])


		#combine reward from env and curiosity
		loss= layers.Lambda(lambda x: (-self.lmd * x[0] + x[1]),
                    output_shape=(1,))([rwd, loss])

		return keras.Model([s_t, s_t1, a_t, rwd], loss)

	#temp easy solution
	def act(self, current_state):
		losses=[]
		for action_option in range(3):
			copy_env=copy.deepcopy(self.env)
			new_state, reward, _, _ = copy_env.step(action_option)
			action_option=self.one_hot_encode(action_option)

			loss=self.model.predict([np.array(current_state).reshape(-1,len(current_state)),
                                    np.array(new_state).reshape(-1,len(new_state)),
                                    np.array(action_option).reshape(-1,len(action_option)),
                                    np.array(reward).reshape(-1,1)])
			losses.append(loss)
		chosen_action=np.argmax(losses)
		return chosen_action

	def learn(self, prev_states, states, actions, rewards):
		s_t=prev_states
		s_t1=states
		actions=np.array(actions)

		icm_loss=self.model.train_on_batch([np.array(s_t),
											np.array(s_t1),
		 									np.array(actions),
											np.array(rewards).reshape((-1, 1))],
											np.zeros((self.batch_size,)))

	def get_intrinsic_reward(self, x):
        ## x -> [prev_state, state, action]
		return K.function([self.model.get_layer("state_t").input,
						self.model.get_layer("state_t1").input,
						self.model.get_layer("action").input],
						[self.model.get_layer("reward_intrinsic").output])(x)[0]

	def batch_train(self):
		states=[]
		actions=[]
		ext_rewards=[]

		game_best_position=-.4
		positions=np.ndarray([0,2])

		for game_index in range(self.n_games):
			state = env.reset()
			game_reward = 0
			running_reward = 0
			print(game_index)

			for step_index in range(self.max_steps):
				if step_index>self.batch_size:
					action=self.act(state)
				else:
					action=self.env.action_space.sample()

				next_state, ext_reward, done, info=self.env.step(action)

				action=self.one_hot_encode(action)

				int_r_state=np.reshape(state, (1,2))
				int_r_next_state=np.reshape(next_state, (1,2))
				int_r_action=np.reshape(action, (1,3))

				int_reward=self.get_intrinsic_reward([np.array(int_r_state),
				 									np.array(int_r_next_state),
													np.array(int_r_action)])

				reward=int_reward+ext_reward
				game_reward+=reward

				if state[0]>game_best_position:
					game_best_position=state[0]
					positions=np.append(positions,
                             [[game_index, game_best_position]], axis=0)

					running_reward+=10
				else:
					running_reward+=reward

				state=next_state

				states.append(next_state)
				ext_rewards.append(ext_reward)
				actions.append(action)

				if step_index%self.batch_size==0 and step_index>=self.batch_size:
					all_states=states[-(self.batch_size+1):]
					self.learn(prev_states=all_states[:self.batch_size],
                                states=all_states[-self.batch_size:],
                                actions=actions[-self.batch_size:],
                                rewards=ext_rewards[-self.batch_size:])
				if done:
					break
					self.rewards[game_index]=game_reward

			positions[-1][0]=game_index
			self.positions[game_index]=positions[-1]
			print(self.positions[game_index])

		self.show_training_data()

	def show_training_data(self):
		self.positions[0]=[0,-0.4]
		plt.figure(1, figsize=[10,5])
		plt.subplot(211)
		plt.plot(self.positions[:,0], self.positions[:,1])
		plt.xlabel('Episode')
		plt.ylabel('Furthest Position')
		plt.subplot(212)
		plt.plot(self.rewards)
		plt.xlabel('Episode')
		plt.ylabel('Reward')
		plt.show()

test = ICM()
test.batch_train()
