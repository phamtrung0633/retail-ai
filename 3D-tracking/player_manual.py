import cv2
import numpy as np

from os import listdir
from os.path import isfile, join

FRAMES_PATH_LEFT = 'frames_data_cam_1'
FRAMES_PATH_RIGHT = 'frames_data_cam_2'

FRAMES_LEFT = filter(isfile, map(lambda f: join(FRAMES_PATH_LEFT, f), listdir(FRAMES_PATH_LEFT)))
FRAMES_RIGHT = filter(isfile, map(lambda f: join(FRAMES_PATH_RIGHT, f), listdir(FRAMES_PATH_RIGHT)))

FRAMES = list(zip(FRAMES_LEFT, FRAMES_RIGHT))

FPS = 1

FORWARD_KEY = '.'
BACKWARD_KEY = ','
END_KEY = '\r' # Enter

iteration = 0
playing = True

while True:
    left, right = FRAMES[iteration % len(FRAMES)]
    name = left
    left = cv2.imread(left)
    right = cv2.imread(right)

    horizontal = np.concatenate((left, right), axis = 1)
    cv2.putText(horizontal, f"{name}", (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2, cv2.LINE_AA)
    cv2.imshow("Frames", horizontal)

    key = cv2.waitKey(0)

    if key == ord(FORWARD_KEY):
        iteration += 1
    elif key == ord(BACKWARD_KEY):
        iteration -= 1
    elif key == ord(END_KEY):
        break

cv2.destroyAllWindows()
