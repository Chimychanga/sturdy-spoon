from retro_contest.local import make
import cv2

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model
from tensorflow.keras import backend
tf.config.experimental_run_functions_eagerly(True)

import numpy as np
from collections import deque

import time


#preprocess image
def preprocess(img):
    #Make B/W, resize, normalise to be b/w 0-1
    to_return = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    to_return = cv2.resize(to_return, (96,96), cv2.INTER_AREA)
    return to_return

#creates framestack or if given, adds framestack
class framestack():
    def __init__(self, max_size, frame):
        self.stack = deque([frame for i in range(max_size)], maxlen=max_size)
    def add(self, frame):
        self.stack.append(frame)

class memory():
    def __init__(self, max_size):
        self.buffer = deque(maxlen = max_size)

    def add(self, experience):
        self.buffer.append(experience)


def show(obs,frames):
    obs = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
    numpy_horizontal_concat = obs

    for frame in frames.stack:
        #Convert it back to 'colour' and same size as original so we can display
        #side by side
        grey_3_channel = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        grey_3_channel = cv2.resize(grey_3_channel, (320,224),interpolation=cv2.INTER_NEAREST)
        numpy_horizontal_concat = np.concatenate((numpy_horizontal_concat, grey_3_channel), axis=1)
    cv2.imshow('Sonic',numpy_horizontal_concat)
    cv2.waitKey(1)
    #time.sleep(0.05)
    pass

def init_action_space():
    #[b,a,mode,start,up,down,left,right,c,y,x,z]
    left        =[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    right       =[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    left_down   =[0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]
    right_down  =[0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0]
    down        =[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    down_b      =[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    b           =[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    action_space = np.asarray([left,right,left_down,right_down,down,down_b,b])
    return action_space

def getModel():

    input = keras.Input(shape=(4, 96, 96, 1))
    delta = keras.Input(shape=[1])

    conv1 = layers.Conv2D(32, 8, activation="relu")(input)
    conv2 = layers.Conv2D(64, 4, activation="relu")(conv1)
    conv3 = layers.Conv2D(64, 4, activation="relu")(conv2)
    flatten = layers.Flatten()(conv3)

    prob = layers.Dense(7, activation="softmax")(flatten)
    values = layers.Dense(1, activation="linear")(flatten)

    optimizer = optimizers.Adam(lr=0.001)

    #Borrowed from "Actor Critic Methods Are Easy With Keras"
    #(https://www.youtube.com/watch?v=2vJtbAha3To&t)
    def custom_loss(y_true, y_pred):
        out = backend.clip(y_pred, 1e-8, 1 - 1e-8)
        log_lik = y_true*backend.log(out)
        return backend.sum(-log_lik*delta)
    ##

    actor = Model(inputs=[input,delta], outputs=[prob])
    actor.compile(optimizer=optimizer, loss=custom_loss)

    critic = Model(inputs=[input], outputs=[values])
    critic.compile(optimizer=optimizer, loss='mean_squared_error')
    return actor, critic

def get_action(model, frames, action_space):
    arr_frames = np.asarray([frames])
    prob = model.predict(arr_frames)[0]
    action = np.random.choice(action_space.shape[0], p=prob)
    return action

def learn(actor, critic, frames, action, reward, prev_frames, done, action_space):
    arr_frames = np.asarray([frames])
    arr_prev_frames = np.asarray([prev_frames])

    value = critic.predict(arr_frames)
    prev_value = critic.predict(arr_prev_frames)

    target = reward + 0.95 * prev_value * (1-int(done))
    delta = target - value
    #print(target)
    #print(delta)

    action_one_hot = np.zeros([1,7])
    action_one_hot[0, action] = 1.0
    actor.fit([arr_frames,delta],action_one_hot, verbose = 0)
    critic.fit(arr_frames, target, verbose = 0)

    return

def main():
    actor, critic = getModel()
    actor.summary()

    env = make(game='SonicTheHedgehog-Genesis', state='LabyrinthZone.Act1')
    obs = env.reset()

    frames = framestack(max_size = 4, frame = preprocess(obs))
    mem = memory(max_size = 10000)
    print(mem.buffer)
    action_space = init_action_space()

    while True:
        action = get_action(actor,frames.stack,action_space)
        obs, rew, done, info = env.step(action_space[action])
        frames.add(frame = preprocess(obs))
        show(obs,frames)

        #learn(actor, critic, frames, action, rew, prev_frames, done, action_space)

        if done:
            obs = env.reset()



if __name__ == '__main__':
    main()
