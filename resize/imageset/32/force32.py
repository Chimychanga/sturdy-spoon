import random
import gym
import numpy as np
import cv2

for i in range(10):
	filename = str(i)+'.png'
	img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
	print(filename)
	resized_image = cv2.resize(img, (32, 32))
	cv2.imwrite(str(i) + '.png',resized_image)