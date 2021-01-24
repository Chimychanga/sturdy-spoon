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
from tensorflow.keras.layers import Reshape
from tensorflow.keras.optimizers import Adam

# borrowed from https://github.com/keras-team/keras-contrib
from dssim import DSSIMObjective

class ConvNetwork():
	def __init__(self):
		self.model = Sequential()
		self.model.add(Conv2D(4, kernel_size=3, activation="relu", input_shape=(32,32,1)))
		self.model.add(Flatten())
		self.model.add(Dense(64*64, activation="linear"))
		self.model.add(Reshape((64, 64,1)))
		
		#self.model.compile(loss="MSE", optimizer=Adam(lr=0.001))
		# MSE is good but not good enough, imagine if image is shifted left by 1 pixel
		# still correct but MSE will say very bad use SSM instead
		self.model.compile(loss="MSE", optimizer=Adam(lr=0.001))
	
	

episode = 1000
images = 268
img32 = []
img64 = []
conv = ConvNetwork()

for i in range(images):
	filename = 'imageset/32/'+str(i)+'.png'
	img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
	#print(filename)
	img = img.reshape(32,32,1)
	#img = np.expand_dims(img, axis=0)
	img32.append(img)

# Just making sure this isn't carrying over
img = None

for i in range(images):
	filename = 'imageset/64/'+str(i)+'.png'
	img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
	img = img.reshape(64,64,1)
	#img = np.expand_dims(img, axis=0)
	img64.append(img)
	
#print(len(img32))
#print(len(img64))
#exit()
counter = 10
indexs = list(range(images-5))
for i in range(episode):
	#print(img32.shape)
	batch = random.sample(indexs, 25)
	input = np.array([img32[each] for each in batch])
	goal = np.array([img64[each] for each in batch])
	#print(input.shape)
	#print(goal.shape)
	conv.model.fit(input, goal, verbose=0)
	#Training ^^^^
	#Testing vvvvv
	if i%50 == 0:
		for index in range(258,268):
			output = conv.model.predict(np.expand_dims(img32[index], axis=0))
			
			output = output.reshape(64,64,1)
			cv2.imwrite('output/'+ str(counter) + '.png',output)
			counter += 1
		