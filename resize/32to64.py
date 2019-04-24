import random
import gym
import numpy as np
import cv2
import tensorflow as tf
import os
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
		self.model.compile(loss="mse", optimizer=Adam(lr=0.001))
	

episode = 5
images = 10
img32 = []
img64 = []
for i in range(images):
	conv = ConvNetwork()
	img = cv2.imread('imageset/32/'+str(i)+'.png', cv2.IMREAD_GRAYSCALE)
	img = img.reshape(32,32,1)
	img = np.expand_dims(img, axis=0)
	img32.append(img)

# Just making sure this isn't carrying over
img = None

for i in range(images):
	img = cv2.imread('imageset/64/'+str(i)+'.png', cv2.IMREAD_GRAYSCALE)
	img = img.reshape(64*64)
	img = np.expand_dims(img, axis=0)
	img64.append(img)
	
#print(len(img32))
#print(len(img64))
#exit()
counter = 0
for i in range(episode):
	#print(img32.shape)
	for index in range(len(img32)):
		output = conv.model.predict(img32[index])
		
		output = output.reshape(64,64,1)
		cv2.imwrite('output/'+ str(counter) + '.png',output)
		#print(img32[index])
		#print(img32[index].shape)
		#print(img64[index])
		#print(img64[index].shape)
		conv.model.fit(img32[index], img64[index], verbose=0)
		counter += 1