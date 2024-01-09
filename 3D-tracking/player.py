import cv2
import numpy as np

from os import listdir
from os.path import isfile, join

FPS = 30

FRAMES_PATH_LEFT = 'frames_data_cam_1'
FRAMES_PATH_RIGHT = 'frames_data_cam_2'

FRAMES_LEFT = filter(isfile, map(lambda f: join(FRAMES_PATH_LEFT, f), listdir(FRAMES_PATH_LEFT)))
FRAMES_RIGHT = filter(isfile, map(lambda f: join(FRAMES_PATH_RIGHT, f), listdir(FRAMES_PATH_RIGHT)))

FRAMES = list(zip(FRAMES_LEFT, FRAMES_RIGHT))

iteration = 0

while True:
    left, right = FRAMES[iteration % len(FRAMES)]

    left = cv2.imread(left)
    right = cv2.imread(right)

    horizontal = np.concatenate((left, right), axis = 1)
    cv2.imshow('Frames', horizontal)

    if cv2.waitKey(1000 // FPS) == 13:
        break

    iteration += 1

cv2.destroyAllWindows()