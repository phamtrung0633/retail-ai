import json

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

NUM_KPS = 17

for timestamp in poses:
    kps = poses[timestamp][0]['points_3d']

    for axis in MIN_BOUNDS:
        MIN_BOUNDS[axis] = min(MIN_BOUNDS[axis], min(kps, key = lambda kp: kp[axis])[axis])

OFFSET = np.fromiter(MIN_BOUNDS.values(), dtype = 'float')

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_xlim3d(-3, 3)
ax.set_ylim3d(-3, 3)
ax.set_zlim3d(0, 5)

# scatter = ax.scatter([], [], [], c='g', marker='o')

# def update(index):
#         scatter._offsets3d = ([], [], [])
    
#         timestamp = list(poses.keys())[index]
#         points = []

#         for pose in poses[timestamp]:
#             kps = np.array(pose['points_3d']) * 1/500
#             points.append(kps)

#         plot_points = np.array(points).reshape(-1,3)
#         scatter._offsets3d = (plot_points[:,0], plot_points[:,1], plot_points[:,2])
    
#         # Set the plot title with the timestamp
#         ax.set_title(f'Timestamp: {timestamp}')

scatters = [ ax.scatter([], [], []) for kp in range(NUM_KPS)]

def animate_scatters(iteration, poses, scatters):
    timestamp = list(poses.keys())[iteration]
    pose = np.array(poses[timestamp][0]['points_3d'])

    for joint in range(NUM_KPS):
        # kp = pose[joint] * 1/1000
        kp = pose[joint] - OFFSET
        print(kp)
        scatters[joint]._offsets3d = (kp[0:1], kp[1:2], kp[2:])

    ax.set_title(f'Timestamp: {timestamp}')

    return scatters

# animation = FuncAnimation(fig, update, len(poses), interval=20, repeat=True)
anim = FuncAnimation(fig, animate_scatters, len(poses), interval = 50, fargs = (poses, scatters), repeat = True)
plt.show()