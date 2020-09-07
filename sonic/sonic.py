from retro_contest.local import make
import cv2

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model

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
def framestack(stack,frame):
    if stack:
        stack.append(frame)
    else:
        #Create empty framestack here with preprocess
        stack = deque([np.zeros((96,96), dtype=np.int) for i in range(4)], maxlen=4)
        stack.append(frame)
        stack.append(frame)
        stack.append(frame)
        stack.append(frame)
    return stack

def show(obs,preprocess):
    obs = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)

    #Convert it back to 'colour' and same size as original so we can display
    #side by side
    grey_3_channel = cv2.cvtColor(preprocess, cv2.COLOR_GRAY2BGR)
    grey_3_channel = cv2.resize(grey_3_channel, (320,224),interpolation=cv2.INTER_NEAREST)
    numpy_horizontal_concat = np.concatenate((obs, grey_3_channel), axis=1)
    cv2.imshow('Sonic',numpy_horizontal_concat)
    cv2.waitKey(1)
    time.sleep(0.05)
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
    action_space = [left,right,left_down,right_down,down,down_b,b]
    return action_space

def getModel():

    input = keras.Input(shape=(96, 96, 1))
    conv1 = layers.Conv2D(32, 8, activation="relu")(input)
    conv2 = layers.Conv2D(64, 4, activation="relu")(conv1)
    conv3 = layers.Conv2D(64, 4, activation="relu")(conv2)
    flatten = layers.Flatten()(conv3)

    prob = layers.Dense(7)(flatten)
    values = layers.Dense(1)(flatten)

    optimizer = optimizers.Adam(lr=0.001)


    train_model = Model(inputs=[input], outputs=[prob])
    train_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    step_model = Model(inputs=[input], outputs=[values])
    step_model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['accuracy'])




    return train_model, step_model

def main():
    train_model, step_model = getModel()
    train_model.summary()

    env = make(game='SonicTheHedgehog-Genesis', state='LabyrinthZone.Act1')
    obs = env.reset()
    frames = framestack(None,preprocess(obs))
    action_space = init_action_space()
    while True:
        obs, rew, done, info = env.step(action_space[1])
        framestack(frames,preprocess(obs))
        show(obs,frames[-1])

        if done:
            obs = env.reset()



if __name__ == '__main__':
    main()
