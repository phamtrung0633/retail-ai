import json
from itertools import zip_longest

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

with open('poses_3d.json') as file:
    poses = json.load(file)

MIN_BOUNDS = {
    0: float('inf'),
    1: float('inf'),
    2: float('inf')
}

MAX_BOUNDS = {
    0: -float('inf'),
    1: -float('inf'),
    2: -float('inf')
}

NUM_POSES = max([len(poses[timestamp]) for timestamp in poses])
NUM_KPS = 17

for timestamp in poses:

    for pose in poses[timestamp]:
        kps = pose['points_3d']

        for axis in MIN_BOUNDS:
            MIN_BOUNDS[axis] = min(MIN_BOUNDS[axis], min(kps, key = lambda kp: kp[axis])[axis])
            MAX_BOUNDS[axis] = max(MAX_BOUNDS[axis], max(kps, key = lambda kp: kp[axis])[axis])

BBOX_MIN = np.fromiter(MIN_BOUNDS.values(), dtype = 'float')
BBOX_MAX = np.fromiter(MAX_BOUNDS.values(), dtype = 'float')

OFFSET = BBOX_MIN + 1/2 * (BBOX_MAX - BBOX_MIN)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_xlim3d(-3, 3)
ax.set_ylim3d(-3, 3)
ax.set_zlim3d(0, 5)

scatter = ax.scatter([], [], [], c='b', marker='o')

def animate_scatters(iteration, poses, scatter):
    timestamp = list(poses.keys())[iteration]
    points = []

    for pose in poses[timestamp]:
        kps = np.array(pose['points_3d']) - OFFSET
        points.append(kps)

    points = np.array(points).reshape(-1, 3)
    assert points.shape == (17 * len(poses[timestamp]), 3)

    scatter._offsets3d = (points[:, 0], points[:, 1], points[:, 2])
    ax.set_title(f'Timestamp: {timestamp}')

anim = FuncAnimation(fig, animate_scatters, len(poses), interval = 50, fargs = (poses, scatter), repeat = True)
plt.show()