import random
import gym
import numpy as np
import cv2
import tensorflow as tf
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

class ConvNetwork():
	def __init__(self):
		self.model = Sequential()
		self.model.add(Conv2D(64, kernel_size=3, activation="relu", input_shape=(32,32,1)))
		self.model.add(Conv2D(32, kernel_size=3, activation="relu"))
		self.model.add(Flatten())
		self.model.add(Dense(64*64, activation="linear"))
		self.model.compile(loss="mse", optimizer=Adam(lr=0.01))

episode = 10

conv = ConvNetwork()
img32 = cv2.imread('imageset/32/1.png', cv2.IMREAD_GRAYSCALE)
img32 = img32.reshape(32,32,1)
img32 = np.expand_dims(img32, axis=0)

img64 = cv2.imread('imageset/64/1.png', cv2.IMREAD_GRAYSCALE)
img64 = img64.reshape(64*64)
img64 = np.expand_dims(img64, axis=0)
for i in range(episode):
	print(img32.shape)
	output = conv.model.predict(img32)
	
	output = output.reshape(64,64,1)
	cv2.imwrite('output/output' + str(i) + '.png',output)
	print(img32)
	print(img32.shape)
	print(img64)
	print(img64.shape)
	conv.model.fit(img32, img64, verbose=0)