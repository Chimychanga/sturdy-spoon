from retro_contest.local import make
import cv2
import time
import numpy as np

#preprocess image
def preprocess(img):
    to_return = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    to_return = cv2.resize(to_return, (96,96), cv2.INTER_AREA)
    return to_return

#creates framestack or if given, adds framestack
def framestack(framestack,frame):
    if framestack:
        framestack.add(preprocess(frame))
    #else:
        #Create empty framestack here with preprocess
    pass



def main():
    env = make(game='SonicTheHedgehog-Genesis', state='LabyrinthZone.Act1')
    obs = env.reset()
    while True:
        obs, rew, done, info = env.step(env.action_space.sample())
        #env.render()
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
        grey_3_channel = cv2.cvtColor(preprocess(obs), cv2.COLOR_GRAY2BGR)
        grey_3_channel = cv2.resize(grey_3_channel, (320,224))
        numpy_horizontal_concat = np.concatenate((obs, grey_3_channel), axis=1)

        cv2.imshow('Sonic',numpy_horizontal_concat)
        cv2.waitKey(1)
        time.sleep(0.1)

        if done:
            obs = env.reset()



if __name__ == '__main__':
    main()
