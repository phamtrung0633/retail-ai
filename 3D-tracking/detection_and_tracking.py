import sys
import time
from collections import defaultdict
import json
from torchreid.reid import metrics
import cv2
from transforms.hands import HandSegmentor
import os
import operator
import cvzone
import copy
import torch
import pickle
from ultralytics import YOLO
from pydantic import BaseModel
from bip_solver import GLPKSolver
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.optimize import linear_sum_assignment
import multiprocessing as mp
from camera import Camera, pose_matrix, normalize_intrinsic, get_Kr_inv, change_intrinsic
from calibration import Calibration
from embeddings.embedder import Embedder
from stream import Stream
from enum import Enum

# from reid import REID
# 
# # Reid
# reid = REID()

from LATransformer.model import LATransformerTest
from LATransformer.helpers import LATransformerForward

import timm

'''latreid_device = 'cuda' if torch.cuda.is_available() else 'cpu'

latreid_backbone = timm.create_model('vit_base_patch16_224', pretrained = True, num_classes = 751)
latreid_backbone = latreid_backbone.to(latreid_device)

latreid = LATransformerTest(latreid_backbone, lmbd = 8).to(latreid_device)
latreid.load_state_dict(torch.load('weights/latransformer_market1501.pth'), strict = False)
latreid.eval()'''

# Kernel for background subtractor
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
# Variables storing text settings for drawing on images
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.5
font_thickness = 2
font_color = (255, 0, 0)
font_color_event = (0, 255, 0)
font_color_event1 = (0, 0, 255)
line_type = cv2.LINE_AA
# Config data
delta_time_threshold = 2
# 2D correspondence config
w_2D = 0.4  # Weight of 2D correspondence
#alpha_2D = 500  # Threshold of 2D velocity
alpha_2D = 1000
lambda_a = 5  # Penalty rate of time interval
lambda_t = 10
# 3D correspondence config
w_3D = 0.6  # Weight of 3D correspondence
alpha_3D = 400  # Threshold of distance
thresh_c = 0.2  # Threshold of keypoint detection confidence
body_image_thresh = 0.6
#similarity_threshold = 6
similarity_threshold = 3.5
w_geometric_dist = 0.85
# Hand threshold for determining end of event
hand_thresh = 0.4
# Angle correspondence config
w_angle = 0.15
# Weights for vision and weight data
weight_v = 0.4
weight_w = 0.6
# THRESHOLD FOR DUPLICATE POSES
DUPLICATE_POSES_THRESHOLD = 40
# Confidence thresholds
UNSEEN_THRESHOLD = 0.1
HAND_IMAGE_THRESHOLD = 0.3
FOOT_JOINT_PROX_THRESHOLD = 0.5
HAND_MINIMUM_CONF = 0.4
# Constants
DET_UNASSIGNED = np.array([0, 0])
SHELF_CONSTANT = 2
TRIPLETS = [["LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST"], ["RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST"],
                ["LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE"], ["RIGHT_HIP", "RIGHT_KNEE", "RIGHT_ANKLE"],
                ["LEFT_SHOULDER", "LEFT_HIP", "LEFT_KNEE"], ["RIGHT_SHOULDER", "RIGHT_HIP", "RIGHT_KNEE"]]
MIN_NUM_FEATURES = 10
COLORS_BY_ID = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (120, 120, 120)]
#RESOLUTION = (640, 480)
RESOLUTION = (1920, 1080)
START_ITERATION = 0
EVALUATE_HAND_SEGMENT = False
KEYPOINTS_NUM = 17
UNASSIGNED = np.array([0, 0, 0])
KEYPOINTS_NAMES = ["NOSE", "LEFT_EYE", "RIGHT_EYE", "LEFT_EAR", "RIGHT_EAR",
                   "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW",
                   "LEFT_WRIST", "RIGHT_WRIST", "LEFT_HIP", "RIGHT_HIP", "LEFT_KNEE",
                   "RIGHT_KNEE", "LEFT_ANKLE", "RIGHT_ANKLE"]
VELOCITY_DATA_NUM = 20
EPSILON = 0.00001
MINIMUM_GROUP_WEIGHT_EVENT_THRESHOLD = 0.2
EVENT_START_THRESHOLD = 0.4
BOX_WIDTH = 160
BOX_HEIGHT = 160
HAND_BOX_WIDTH = 74
HAND_BOX_HEIGHT = 74
'''BOX_WIDTH = 80
BOX_HEIGHT = 80
HAND_BOX_WIDTH = 42
HAND_BOX_HEIGHT = 42'''
SHIFT_CONSTANT = 1.4
USE_MULTIPROCESS = True
# For testing sake, there's only exactly one shelf, this variable contains constant for the shelf
SHELF_DATA_TWO_CAM = np.array([[[1293.33, 51.67], [1453.33, 198.33], [1350, 885]],
                               [[1231.67, 5], [1500, 128.33], [1358.33, 785]]])
'''SHELF_DATA_TWO_CAM = np.array([[[469.29, 22.86], [542.86, 89.29], [494.29, 394.29]],
                               [[439.29, 2.14], [560.71, 57.14], [497.86, 349.29]]])'''
SHELF_PLANE_THRESHOLD = 250 # In this case is only = 10cm in real world scale due to calibration square size mistake
FOOT_JOINT_SHELF_THRESHOLD = 400 # In this case is only = 50cm in real world due to calibration square size mistake
PROXIMITY_EVENT_START_THRESHOLD_MULTIPROCESS = 3
PROXIMITY_EVENT_START_TIMEFRAME = 0.5
LEFT_WRIST_POS = 9
RIGHT_WRIST_POS = 10
LEFT_ELBOW_POS = 7
RIGHT_ELBOW_POS = 8
LEFT_FOOT_POS = 15
RIGHT_FOOT_POS = 16
MAX_ITERATIONS = 6000
USE_REPLAY = True
TEST_CALIBRATION = False
TEST_PROXIMITY = False
FRAMERATE = 15

CLEAR_CONF_THRESHOLD = 0.1 # Acceptable mean confidence from all camera views seeing a wrist (used to be hand_thresh)
CLEAR_LIMIT = 44 # Number of frames that a Proxicams_frames[camera_id][iterations] = {'filename': str(timestamp) + '.png', 'image': new_frame}mityEvent has to be confidently ended to be stopped

# Segmentor
segmentor = HandSegmentor((HAND_BOX_WIDTH, HAND_BOX_HEIGHT), (256, 256))

class ActionEnum(Enum):
    TAKE = 1
    PUT = 2


class Shelf:
    def __init__(self, data):
        self.top_left_point = data[0]
        self.top_right_point = data[1]
        self.bottom_right_point = data[2]

    def get_points(self):
        return np.array([self.top_left_point, self.top_right_point, self.bottom_right_point])


def compute_distance(qf, gf):
    distmat = metrics.compute_distance_matrix(qf, gf, 'euclidean')
    return distmat.numpy()


class ProximityEvent:
    def __init__(self, start_time, person_id):
        self.start_time = start_time
        self.person_id = person_id
        self.status = "active"
        self.hand_images = []
        self.end_time = None
        self.visualisation_attr = {}
        self.clear_count = 0

    def get_id(self):
        return str(self.get_person_id()) + '_' + str(self.get_start_time()) + '_' + str(self.get_end_time())

    def get_visualisation_attr(self):
        return self.visualisation_attr

    def get_event_id(self):
        return self.get_id()

    def get_person_id(self):
        return self.person_id

    def get_hand_images(self):
        return self.hand_images

    def get_start_time(self):
        return self.start_time

    def get_end_time(self):
        return self.end_time

    def set_visualisation_attr(self, camera_ids, timestamps):
        for index, id in enumerate(camera_ids):
            self.visualisation_attr[id] = timestamps[index]

    def set_end_time(self, official_end_time):
        self.end_time = official_end_time

    def add_hand_images(self, image):
        self.hand_images.append(image)

    def set_status(self, value):
        self.status = value

    def increment_clear_count(self):
        self.clear_count += 1

    def reset_clear_count(self):
        self.clear_count = 0

    def event_ended(self):
        return self.clear_count >= CLEAR_LIMIT

    def merge_event(self, other):
        self.start_time = other.get_start_time()
        self.hand_images = other.get_hand_images() + self.hand_images

    def delete_images_by_interval(self):
        del self.hand_images[1::2]


class ProximityEventGroup:
    def __init__(self, first_event, shelf_id=SHELF_CONSTANT):
        self.shelf_id = shelf_id
        self.proximity_events = [first_event]
        self.minimum_timestamp = first_event.get_start_time()
        self.maximum_timestamp = float('-inf')
        self.active_events_num = 1

    def get_events(self):
        return self.proximity_events

    def get_shelf_id(self):
        return self.shelf_id

    def add_event(self, event):
        self.proximity_events.append(event)
        self.active_events_num += 1

    def decrement_active_num(self):
        self.active_events_num -= 1

    def set_maximum_timestamp(self, value):
        if value > self.maximum_timestamp:
            self.maximum_timestamp = value

    def finished(self):
        return self.active_events_num == 0

    def get_minimum_timestamp(self):
        return self.minimum_timestamp

    def get_maximum_timestamp(self):
        return self.maximum_timestamp


class GetKeypointOpenPose(BaseModel):
    NOSE: int = 0
    NECK: int = 1
    RIGHT_SHOULDER: int = 2
    RIGHT_ELBOW: int = 3
    RIGHT_WRIST: int = 4
    LEFT_SHOULDER: int = 5
    LEFT_ELBOW: int = 6
    LEFT_WRIST: int = 7
    RIGHT_HIP: int = 8
    RIGHT_KNEE: int = 9
    RIGHT_ANKLE: int = 10
    LEFT_HIP: int = 11
    LEFT_KNEE: int = 12
    LEFT_ANKLE: int = 13
    RIGHT_EYE: int = 14
    LEFT_EYE: int = 15
    RIGHT_EAR: int = 16
    LEFT_EAR: int = 17


def get_index_from_key(name):
    string_to_value_mapping = {
        "NOSE": 0,
        "NECK": 1,
        "RIGHT_SHOULDER": 2,
        "RIGHT_ELBOW": 3,
        "RIGHT_WRIST": 4,
        "LEFT_SHOULDER": 5,
        "LEFT_ELBOW": 6,
        "LEFT_WRIST": 7,
        "RIGHT_HIP": 8,
        "RIGHT_KNEE": 9,
        "RIGHT_ANKLE": 10,
        "LEFT_HIP": 11,
        "LEFT_KNEE": 12,
        "LEFT_ANKLE": 13,
        "RIGHT_EYE": 14,
        "LEFT_EYE": 15,
        "RIGHT_EAR": 16,
        "LEFT_EAR": 17
    }
    return string_to_value_mapping[name]


class HumanPoseDetection():
    def __init__(self):
        self.model = self.load_model()
        self.warm_up()

    def warm_up(self):
        dummy_image = cv2.imread("images/reprojected_envi_1.png")
        self.predict(dummy_image)

    def load_model(self):
        model = YOLO('weights/yolov8x-pose.pt').to('cuda')
        return model

    def predict(self, image):
        results = self.model(image, verbose=False)
        return results


class WeightEvent:
    def __init__(self, start_time, event_id):
        self.id = event_id
        self.start_time = start_time
        self.start_value = None
        self.end_value = None
        self.end_time = float('inf')

    def get_id(self):
        return self.id

    def set_start_val(self, value):
        self.start_value = value

    def get_start_time(self):
        return self.start_time

    def get_end_time(self):
        return self.end_time

    def set_end_val(self, value):
        self.end_value = value

    def set_end_time(self, val):
        self.end_time = val

    def get_weight_change(self):
        return self.end_value - self.start_value


class InteractionEvent:
    def __init__(self, person_id, product, action_type: ActionEnum, start_time, end_time):
        self.person_id = person_id
        self.product = product
        self.action_type = action_type
        self.start_time = start_time
        self.end_time = end_time

    def get_person_id(self):
        return self.person_id

    def get_product(self):
        return self.product

    def get_action_type(self):
        return self.action_type

    def __str__(self):
        return f"Person {self.person_id} {self.action_type.name.lower()} product {self.product['sku']}"

def cross2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.cross(a, b)

def delete_directory(dir_name):
    for file in os.listdir(dir_name):
        file_path = os.path.join(dir_name, file)
        os.remove(file_path)

def get_plane_equation_shelf_points(data):
    P = data[0]
    Q = data[1]
    R = data[2]
    x1, y1, z1 = P
    x2, y2, z2 = Q
    x3, y3, z3 = R
    a1 = x2 - x1
    b1 = y2 - y1
    c1 = z2 - z1
    a2 = x3 - x1
    b2 = y3 - y1
    c2 = z3 - z1
    a = b1 * c2 - b2 * c1
    b = a2 * c1 - a1 * c2
    c = a1 * b2 - b1 * a2
    direction_vector = np.array([a, b, c])
    direction_vector = direction_vector / np.linalg.norm(direction_vector)
    P_ref = np.array([x1, y1, z1])
    d = np.dot(-P_ref, direction_vector)
    return direction_vector[0], direction_vector[1], direction_vector[2], d


def get_clipping_plane_type_one(P1, P2, N):
    V = P2 - P1
    N_new = cross2(V, N)
    direction_vector = N_new / np.linalg.norm(N_new)
    P_ref = P1
    a = direction_vector[0]
    b = direction_vector[1]
    c = direction_vector[2]
    d = np.dot(-P_ref, direction_vector)
    return a, b, c, d


# Clipping plane type two is a plane that is perpendicular to a type one plane, a shelf plane, and pass through a corner
def get_clipping_plane_type_two(N1, N2, Point):
    x1, y1, z1 = Point
    P_ref = Point
    normal = cross2(N1, N2)
    direction_vector = normal / np.linalg.norm(normal)
    a = direction_vector[0]
    b = direction_vector[1]
    c = direction_vector[2]
    d = np.dot(-P_ref, direction_vector)
    return a, b, c, d


def distance_to_plane(point, plane_equation):
    a, b, c, d = plane_equation
    X, Y, Z = point
    distance = (a * X + b * Y + c * Z + d) / np.sqrt(a ** 2 + b ** 2 + c ** 2)
    return abs(distance)


def plane_grid(normal, d):
    x, y = np.meshgrid(np.arange(-5, 5, 0.25), np.arange(-5, 5, 0.25))
    z = (-normal[0] * x - normal[1] * y - d) * 1. / normal[2]  # Solve for z using the plane equation
    return x, y, z


def visualise_3D(data, x_vis_shelf, y_vis_shelf, z_vis_shelf):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    num_frames = len(data)

    def animate(frame_num):
        ax.clear()  # Clear the previous frame
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')
        ax.set_title(f'Frame {frame_num}')
        ax.set_xlim(-2000, 5000)
        ax.set_ylim(-2000, 5000)
        ax.set_zlim(-2000, 6000)
        # Plot 3D coordinates for the current frame
        coordinates = data[frame_num]
        ax.plot_surface(x_vis_shelf, y_vis_shelf, z_vis_shelf, alpha=0.5)
        if len(coordinates[0]) > 0:
            ax.scatter(coordinates[:, 0], coordinates[:, 1], coordinates[:, 2], c='b', marker='o')

    # Set labels and title
    animation = FuncAnimation(fig, animate, frames=num_frames, interval=50)
    plt.show()


def is_point_between_planes(plane1_eq, plane2_eq, point):
    a1, b1, c1, d1 = plane1_eq
    a2, b2, c2, d2 = plane2_eq
    normal1 = np.array([a1, b1, c1])
    normal2 = np.array([a2, b2, c2])
    point_on_plane1 = np.array([-d1 / a1, 0, 0])
    point_on_plane2 = np.array([-d2 / a2, 0, 0])
    vector_to_plane1 = point_on_plane1 - point
    vector_to_plane2 = point_on_plane2 - point
    dot_product1 = np.dot(vector_to_plane1, normal1)
    dot_product2 = np.dot(vector_to_plane2, normal2)
    if dot_product1 == 0 or dot_product2 == 0:
        return True

    if np.dot(normal1, normal2) < 0:
        dot_product2 *= -1

    if np.sign(dot_product1) != np.sign(dot_product2):
        return True
    else:
        return False


def check_point_on_plane(point_to_check, plane_equation):
    tolerance = 1e-10
    return (np.dot(point_to_check, plane_equation[:3]) + plane_equation[3]) < tolerance


def estimate_velocity_3d(timestamps, positions):
    # Ensure consistent shapes
    timestamps = np.asarray(timestamps)
    positions = np.asarray(positions)
    if len(timestamps) != positions.shape[0]:
        raise ValueError("Timestamps and positions must have the same number of samples.")

    # Design matrix for linear regression in each dimension
    design_matrix = np.vstack([np.ones_like(timestamps), timestamps]).T

    # Estimate velocities for each dimension independently
    velocities = np.zeros(3)
    for i in range(3):
        params, _, _, _ = np.linalg.lstsq(design_matrix, positions[:, i], rcond=None)
        velocities[i] = params[1]  # Extract slope coefficient (velocity)

    return velocities


def get_velocity_at_this_timestamp_for_this_id_for_cur_timestamp(poses_3d_all_timestamps, timestamp_latest_pose,
                                                                 points_3d_latest_pose, id_latest_pose):
    """
    poses_3d_at_cur_timstamp, poses_3d_at_last_timstamp: numpy array of shape (1 x no of joints)
    """
    ## TODO: verify velocity estimation...
    #  3D velocity estimated via a linear least-square method

    # go from the second last index in the window delta time threshold to the second last occurence of the points 
    # 3d for the ID id_latest_pose
    velocity_t = []
    timestamp_tilde_frame = []
    points_3d_tilde_timestamp = []
    count = 0
    for index in range(len(poses_3d_all_timestamps) - 1, 0, -1):
        if count > VELOCITY_DATA_NUM:
            break
        this_timestamp = list(poses_3d_all_timestamps.keys())[index]
        if this_timestamp >= timestamp_latest_pose or all(
                value is None for value in poses_3d_all_timestamps[this_timestamp]):
            continue
        # iterate through to the current timestamp and append values for the IDs which are not already covered before
        for id_index in range(len(poses_3d_all_timestamps[this_timestamp])):
            if poses_3d_all_timestamps[this_timestamp][id_index]['id'] == id_latest_pose:
                count += 1
                points_3d_tilde_timestamp.append(
                    np.array(poses_3d_all_timestamps[this_timestamp][id_index]['points_3d']))
                timestamp_tilde_frame.append(this_timestamp)

    for k in range(len(points_3d_latest_pose)):
        joint_k_positions = []
        joint_k_timestamps = []
        for i in range(len(points_3d_tilde_timestamp)):
            if np.all(points_3d_tilde_timestamp[i][k] == UNASSIGNED):
                continue
            joint_k_positions.append(points_3d_tilde_timestamp[i][k])
            joint_k_timestamps.append(timestamp_tilde_frame[i])
        joint_k_positions.append(points_3d_latest_pose[k])
        joint_k_timestamps.append(timestamp_latest_pose)
        if len(joint_k_positions) <= 1:
            velocity_t.append(np.zeros(3))
        else:
            velocity_t.append(estimate_velocity_3d(joint_k_timestamps, joint_k_positions))

    return velocity_t


def get_latest_3D_poses_available_for_cur_timestamp(poses_3d_all_timestamps, timestamp_cur_frame,
                                                    delta_time_threshold):
    # Iterate through poses_3d_all_timestamps from the current timestamp to get the latest points 3D for IDs in
    # the window of the delta_time_threshold> Note that time window from the current timestamp and not from the 
    # timestamp when points 3d were estimated

    # [[{'id': calculated, 'points_3d': list of target joints, 'timestamp': , 'velocity': }], [{}], ]
    poses_3D_latest = []
    id_list = []

    for index in range(len(poses_3d_all_timestamps) - 1, -1, -1):
        this_timestamp = list(poses_3d_all_timestamps.keys())[index]
        # time window ends return the ID
        if (timestamp_cur_frame - this_timestamp) > delta_time_threshold:
            break
        # to get 3d pose at timestamp before the timestamp at the current frame
        if this_timestamp >= timestamp_cur_frame or all(
                value is None for value in poses_3d_all_timestamps[this_timestamp]):
            continue
        if all(value is not None for value in poses_3d_all_timestamps[this_timestamp]):
            # iterate through to the current timestamp and append values for the IDs which are not already covered before
            for id_index in range(len(poses_3d_all_timestamps[this_timestamp])):
                if poses_3d_all_timestamps[this_timestamp][id_index]['id'] not in id_list:
                    poses_3D_latest.append({'id': poses_3d_all_timestamps[this_timestamp][id_index]['id'],
                                            'points_3d': poses_3d_all_timestamps[this_timestamp][id_index]['points_3d'],
                                            'timestamp': this_timestamp,
                                            'detections': poses_3d_all_timestamps[this_timestamp][id_index][
                                                'detections'],
                                            'confidences': poses_3d_all_timestamps[this_timestamp][id_index][
                                                'confidences'],
                                            'timestamps_2d': poses_3d_all_timestamps[this_timestamp][id_index][
                                                'timestamps_2d'],
                                            'velocity': get_velocity_at_this_timestamp_for_this_id_for_cur_timestamp(
                                                poses_3d_all_timestamps,
                                                this_timestamp,
                                                poses_3d_all_timestamps[this_timestamp][id_index]['points_3d'],
                                                poses_3d_all_timestamps[this_timestamp][id_index]['id'])})
                    id_list.append(poses_3d_all_timestamps[this_timestamp][id_index]['id'])
        else:
            continue

    if len(poses_3D_latest) > 0:
        poses_3D_latest = sorted(poses_3D_latest, key=lambda i: int(i['id']), reverse=False)
    return poses_3D_latest


def calculate_perpendicular_distance(point, line_start, line_end):
    distance = np.linalg.norm(np.cross(line_end - line_start, line_start - point)) / np.linalg.norm(
        line_end - line_start)
    return distance


def extract_key_value_pairs_from_poses_2d_list(data, id, timestamp_cur_frame, dt_thresh):
    camera_id_covered_list = []
    result = []

    ## TODO: search the points in the delta time threshold window
    # Find the latest timestamp for each camera

    for index in range(len(data) - 1, 0, -1):

        this_timestamp = data[index]['timestamp']
        this_camera = data[index]['camera']
        # time window ends return the ID
        if (timestamp_cur_frame - this_timestamp) > dt_thresh:
            break
        if this_camera not in camera_id_covered_list:
            # iterate through to the current timestamp and append values for the IDs which are not already covered before
            for pose_index in range(len(data[index]['poses'])):
                if data[index]['poses'][pose_index]['id'] == id:
                    result.append({
                        'camera': this_camera,
                        'frame': data[index]['frame'],
                        'foreground_mask': data[index]['foreground_mask'],
                        'timestamp': this_timestamp,
                        'poses': data[index]['poses'][pose_index],
                        'image_wh': [RESOLUTION[0], RESOLUTION[1]]
                    })
                    camera_id_covered_list.append(this_camera)
                    break
        else:
            continue

    return result


def separate_lists_for_incremental_triangulation(data):
    result = {}
    for item in data:
        for key, value in item.items():
            if key not in result:
                result[key] = []
            result[key].append(value)
    return result


def compute_affinity_epipolar_constraint_with_pairs(detections_pairs, alpha_2D,
                                                    num_body_joints_detected_by_2d_pose_detector, calibration):
    Au_this_pair = 0
    # assuming D_i, D_j are each single matrix of 14x2
    D_L = np.array(detections_pairs[0]['points_2d'])
    D_R = np.array(detections_pairs[1]['points_2d'])
    scores_l = np.array(detections_pairs[0]['scores'])
    scores_r = np.array(detections_pairs[1]['scores'])
    cam_L_id = detections_pairs[0]['camera_id']
    cam_R_id = detections_pairs[1]['camera_id']
    epipolar_error = calibration.calc_epipolar_error([cam_L_id, cam_R_id], D_L, scores_l, D_R, scores_r)
    Au_this_pair = 1 - (epipolar_error /  (alpha_2D))
    #print(Au_this_pair)
    #print("Epipolar error to set threshold: ", epipolar_error)
    return Au_this_pair


def get_affinity_matrix_epipolar_constraint(Du, alpha_2D, calibration):
    # Step 1: Get all unmatched detections per camera for the current timestamp
    # Step 2: Generate pair of detections for all the detections of all cameras with
    # detections of every other cameras
    # Step 3: For each camera:
    #           for each pair of detection:
    #               for each body joint in the detection:
    #                   compute affinity matrix via epipolar contraint with the remaining detections in all other cameras

    Du_cam_wise_split = {}
    for entry in Du:
        camera_id = entry['camera_id']
        if camera_id not in Du_cam_wise_split:
            Du_cam_wise_split[camera_id] = []
        Du_cam_wise_split[camera_id].append(entry)

    num_entries = sum(len(entries) for entries in Du_cam_wise_split.values())
    Au = np.zeros((num_entries, num_entries), dtype=np.float32)

    # Create a dictionary to map each camera_id to an index
    camera_id_to_index = {camera_id: i for i, camera_id in enumerate(Du_cam_wise_split.keys())}
    # Iterate over each camera
    all_entries = []
    for camera_id, entries in Du_cam_wise_split.items():
        for i in range(len(entries)):
            all_entries.append([camera_id, entries[i]])

    for i in range(num_entries):
        for j in range(num_entries):
            if all_entries[j][0] != all_entries[i][0]:
                pair_ij = (all_entries[i][1], all_entries[j][1])
                pair_ji = (all_entries[j][1], all_entries[i][1])
                Au[i, j] = compute_affinity_epipolar_constraint_with_pairs(pair_ij,
                                                                           alpha_2D,
                                                                           KEYPOINTS_NUM,
                                                                           calibration)
                Au[j, i] = compute_affinity_epipolar_constraint_with_pairs(pair_ji,
                                                                           alpha_2D,
                                                                           KEYPOINTS_NUM,
                                                                           calibration)
    return Au


def check_joint_near_shelf(joint, object_plane_eq, left_plane_eq, right_plane_eq, top_plane_eq, timestamp, foot_joint=False):
    dist_from_shelf_plane = distance_to_plane(joint, object_plane_eq)
    if foot_joint:
        threshold = FOOT_JOINT_SHELF_THRESHOLD
    else:
        threshold = SHELF_PLANE_THRESHOLD
    if ((dist_from_shelf_plane < threshold) and
            (is_point_between_planes(left_plane_eq, right_plane_eq, joint))):
        '''if TEST_PROXIMITY:
            print(f"Joint near shelf with timestamp {timestamp}")'''
        return True
    elif (dist_from_shelf_plane < threshold) and TEST_PROXIMITY:
        #print("Joint near shelf plane but not inside")
        pass
    return False


# This function only draw the ID of a person on the image when the person is tracked not when initialized
def draw_id(top_left_point, bottom_right_point, track_id, tracking_visualization_frame):
    cv2.rectangle(tracking_visualization_frame, top_left_point, bottom_right_point, color=COLORS_BY_ID[track_id], thickness=3)
    cvzone.putTextRect(tracking_visualization_frame, f'ID: {int(track_id)}', (top_left_point[0] + 5, top_left_point[1] + 15), scale=2, thickness=2,
                       colorR=COLORS_BY_ID[track_id])

def draw_message(pos, message, image, color_font = font_color_event):
    cv2.putText(image, message, (int(pos[0]), int(pos[1])), font, font_scale, color_font, font_thickness, line_type,
                False)

def draw_id_2(data, image):
    for i, joint in enumerate(data):
        image = cv2.putText(image, "J", (int(joint[0]), int(joint[1])), font, font_scale,
                            font_color, font_thickness, line_type, False)
    return image

# Function to generate bounding box around a given point
def generate_bounding_box(x, y, hand=False):
    if hand:
        return x - HAND_BOX_WIDTH // 2, y - HAND_BOX_HEIGHT // 2, x + HAND_BOX_WIDTH // 2, y + HAND_BOX_HEIGHT // 2
    return x - BOX_WIDTH // 2, y - BOX_HEIGHT // 2, x + BOX_WIDTH // 2, y + BOX_HEIGHT // 2


# Special function used in mutliprocessing to extract features
def extract_features(images):
    f = reid.features(images)
    return f


def gather_weights(shared_list, lock, start_time, chronology = None) -> None:
    import serial.tools.list_ports
    WINDOW_LENGTH = 3
    SHARED_TIME = start_time
    CALIBRATION_WEIGHT = 1000
    PORT_NUMBER = 0
    BAUDRATE = 38400
    THRESHOLD = 1000
    id_num = 0

    weight_buffer = []
    current_event = None
    trigger_counter_start = 0
    potential_start_time = None
    potential_start_value = None
    def calculate_moving_variance(values):
        values = np.array(values)
        variance = np.var(values)
        return variance

    if not USE_REPLAY:
        def write_read(x):
            message = x + "\n"
            serialInst.write(message.encode('utf-8'))
            return None
        serialInst = serial.Serial()
        ports = serial.tools.list_ports.comports()

        portList = []
        for onePort in ports:
            portList.append(str(onePort))
            print(str(onePort))

        val = PORT_NUMBER

        for x in range(0, len(portList)):
            if portList[x].startswith("/dev/ttyUSB" + str(val)):
                portVal = "/dev/ttyUSB" + str(val)

        serialInst.baudrate = BAUDRATE
        serialInst.port = portVal
        serialInst.open()

        # Calibrate the scale
        while True:
            if serialInst.in_waiting:
                packet = serialInst.readline()
                print(packet.decode('utf'))
                time.sleep(5)
                num = str(CALIBRATION_WEIGHT)
                write_read(num)
                break

        while True:
            if serialInst.in_waiting:
                packet, time_packet = serialInst.readline(), time.time()
                value = float(packet.decode('utf')[:-2])
                if len(weight_buffer) < WINDOW_LENGTH:
                    weight_buffer.append([value, time_packet - SHARED_TIME])
                else:
                    del weight_buffer[0]
                    weight_buffer.append([value, time_packet - SHARED_TIME])
                    w = np.array(weight_buffer)
                    moving_variance = calculate_moving_variance(w[:, 0])
                    if moving_variance >= THRESHOLD and current_event is None:
                        if trigger_counter == 1:
                            current_event = WeightEvent(potential_start_time, id_num)
                            id_num += 1
                            current_event.set_start_val(potential_start_value)
                            potential_start_time = None
                            potential_start_value = None
                            trigger_counter = 0
                        elif trigger_counter == 0:
                            potential_start_time = weight_buffer[2][1]
                            potential_start_value = weight_buffer[1][0]
                            trigger_counter += 1

                    elif moving_variance < THRESHOLD and current_event is not None:
                        if trigger_counter == 1:
                            current_event.set_end_time(weight_buffer[1][1])
                            current_event.set_end_val(weight_buffer[1][0])
                            lock.acquire()
                            shared_list.append(current_event)
                            lock.release()
                            current_event = None
                            trigger_counter = 0
                        elif trigger_counter == 0:
                            trigger_counter += 1
                        potential_start_time = None
                        potential_start_value = None
                    else:
                        trigger_counter = 0
                        potential_start_time = None
                        potential_start_value = None
    else: # USE_REPLAY
        for event in chronology:
            timestamp, value = float(event) - SHARED_TIME, chronology[event]
            
            if len(weight_buffer) < WINDOW_LENGTH:
                weight_buffer.append([value, timestamp])
            else:
                del weight_buffer[0]
                weight_buffer.append([value, timestamp])
                w = np.array(weight_buffer)
                moving_variance = calculate_moving_variance(w[:, 0])
                if moving_variance >= THRESHOLD and current_event is None:
                    if trigger_counter_start == 1:
                        current_event = WeightEvent(weight_buffer[1][1], id_num)
                        id_num += 1
                        current_event.set_start_val(weight_buffer[0][0])
                        trigger_counter_start = 0
                        #print(w[:, 0])
                    elif trigger_counter_start == 0:
                        trigger_counter_start += 1
                elif moving_variance < THRESHOLD and current_event is not None:
                    current_event.set_end_time(weight_buffer[1][1])
                    current_event.set_end_val(weight_buffer[1][0])
                    lock.acquire()
                    shared_list.append(current_event)
                    lock.release()
                    current_event = None
                    trigger_counter_start = 0
                    #print(w[:, 0])
                else:
                    trigger_counter_start = 0


def affinity_score_avg_product(angle1, angle2, confidences1, confidences2):
    average_product = np.mean(confidences1) * np.mean(confidences2)
    angle_diff = np.abs(angle1 - angle2)
    affinity = (1.0 - (angle_diff / 180)) * average_product
    return affinity


def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


def process_proximity_detection_by_foot_joints(foot_joints, conf_foot_joints, object_plane_eq, left_plane_eq,
                                               right_plane_eq, top_plane_eq, id, timestamps, camera_ids, current_events,
                                               proximity_event_group, potential_proximity_events, position_wrist_cam1,
                                               position_wrist_cam2, frame1, frame2, joint_side):
    timestamp = timestamps[0]
    left_foot_near_shelf = check_joint_near_shelf(foot_joints[0], object_plane_eq, left_plane_eq, right_plane_eq,
                                                  top_plane_eq, timestamp, True)
    right_foot_near_shelf = check_joint_near_shelf(foot_joints[1], object_plane_eq, left_plane_eq, right_plane_eq,
                                                   top_plane_eq, timestamp, True)
    foot_prox_acceptance_near = ((left_foot_near_shelf and conf_foot_joints[0] > FOOT_JOINT_PROX_THRESHOLD) or
                                 (right_foot_near_shelf and conf_foot_joints[1] > FOOT_JOINT_PROX_THRESHOLD))
    foot_prox_acceptance_far = ((not left_foot_near_shelf and conf_foot_joints[0] > FOOT_JOINT_PROX_THRESHOLD) or
                                 (not right_foot_near_shelf and conf_foot_joints[1] > FOOT_JOINT_PROX_THRESHOLD))
    '''draw_message(position_wrist_cam1, str(round(distance_to_plane(foot_joints[0], object_plane_eq), 2)) + ' ' + str(round(distance_to_plane(foot_joints[1], object_plane_eq), 2)), frame1)
    draw_message(position_wrist_cam2, str(round(distance_to_plane(foot_joints[0], object_plane_eq), 2)) + ' ' + str(round(distance_to_plane(foot_joints[1], object_plane_eq), 2)), frame2)'''
    if joint_side == 'LEFT':
        id = str(id) + '_left'
    else:
        id = str(id) + '_right'
    if foot_prox_acceptance_near:
        if id not in current_events:
            if id not in potential_proximity_events:
                potential_proximity_events[id] = [timestamp]
            elif timestamp - potential_proximity_events[id][0] > PROXIMITY_EVENT_START_TIMEFRAME:
                del potential_proximity_events[id]
            elif len(potential_proximity_events[id]) < PROXIMITY_EVENT_START_THRESHOLD_MULTIPROCESS:
                potential_proximity_events[id].append(timestamp)
            else:
                current_events[id] = ProximityEvent(timestamp, id)
                # Add to group of proximity events if there exist, otherwise initialize a new group
                # Be careful as if a proximity event never get to finish - due to losing track, this won't work
                if proximity_event_group is None:
                    proximity_event_group = ProximityEventGroup(current_events[id])
                else:
                    proximity_event_group.add_event(current_events[id])
                del potential_proximity_events[id]
                if TEST_PROXIMITY:
                    draw_message(position_wrist_cam1, "Near", frame1)
                    draw_message(position_wrist_cam2, "Near", frame2)
    elif foot_prox_acceptance_far and id in current_events:
        # Potentially not necessary ?
        current_events[id].increment_clear_count()
        if current_events[id].event_ended():
            current_events[id].set_end_time(timestamp)
            current_events[id].set_status("completed")
            current_events[id].set_visualisation_attr(camera_ids, timestamps)
            if TEST_PROXIMITY:
                draw_message(position_wrist_cam1, "End", frame1)
                draw_message(position_wrist_cam2, "End", frame2)
            del current_events[id]

            proximity_event_group.decrement_active_num()
            if proximity_event_group.finished():
                proximity_event_group.set_maximum_timestamp(timestamp)
                return True, proximity_event_group, current_events, potential_proximity_events
    return False, proximity_event_group, current_events, potential_proximity_events


def process_proximity_detection(wrist, elbows, confs_elbow,
                                object_plane_eq, left_plane_eq, right_plane_eq, top_plane_eq,
                                id, timestamps, camera_ids, current_events, proximity_event_group, potential_proximity_events,
                                conf_wrist, position_wrist_cam1, position_wrist_cam2, frame1, frame2, fg_mask_1, fg_mask_2):
    '''draw_message(position_wrist_cam1, str(round(distance_to_plane(wrist, object_plane_eq), 2)), frame1, font_color_event1)
    draw_message(position_wrist_cam2, str(round(distance_to_plane(wrist, object_plane_eq), 2)), frame2, font_color_event1)'''
    timestamp = timestamps[0]
    wrist_near_shelf = check_joint_near_shelf(wrist, object_plane_eq, left_plane_eq, right_plane_eq, top_plane_eq, timestamp)
    if TEST_PROXIMITY:
        draw_message(position_wrist_cam1, str(distance_to_plane(wrist, object_plane_eq)), frame1)
        draw_message(position_wrist_cam2, str(distance_to_plane(wrist, object_plane_eq)), frame2)
    if wrist_near_shelf:
        if id not in current_events:
            if id not in potential_proximity_events:
                potential_proximity_events[id] = [timestamp]
                return False, proximity_event_group, current_events, potential_proximity_events
            elif timestamp - potential_proximity_events[id][0] > PROXIMITY_EVENT_START_TIMEFRAME:
                del potential_proximity_events[id]
                return False, proximity_event_group, current_events, potential_proximity_events
            elif len(potential_proximity_events[id]) < PROXIMITY_EVENT_START_THRESHOLD_MULTIPROCESS:
                potential_proximity_events[id].append(timestamp)
                return False, proximity_event_group, current_events, potential_proximity_events
            else:
                current_events[id] = ProximityEvent(timestamp, id)
                # Add to group of proximity events if there exist, otherwise initialize a new group
                # Be careful as if a proximity event never get to finish - due to losing track, this won't work
                if proximity_event_group is None:
                    proximity_event_group = ProximityEventGroup(current_events[id])
                else:
                    proximity_event_group.add_event(current_events[id])
                del potential_proximity_events[id]
                if TEST_PROXIMITY:
                    draw_message(position_wrist_cam1, "Near", frame1)
                    draw_message(position_wrist_cam2, "Near", frame2)
        current_events[id].reset_clear_count()
        if id in current_events:
            if conf_wrist[0] > HAND_IMAGE_THRESHOLD:
                if confs_elbow[0] < HAND_IMAGE_THRESHOLD:
                    hand_pos_cam_1 = position_wrist_cam1
                else:
                    elbow_to_hand_direction_vector = position_wrist_cam1 - elbows[0]
                    shift_vector = elbow_to_hand_direction_vector * SHIFT_CONSTANT
                    hand_pos_cam_1 = elbows[0] + shift_vector
                image1_x1, image1_y1, image1_x2, image1_y2 = generate_bounding_box(hand_pos_cam_1[0],
                                                                                   hand_pos_cam_1[1],
                                                                                   hand=True)
                image1_x1, image1_y1, image1_x2, image1_y2 = int(image1_x1), int(image1_y1), int(image1_x2), int(
                    image1_y2)
                if image1_x1 >= 0 and image1_y1 >= 0 and image1_y2 < RESOLUTION[1] and image1_x2 < RESOLUTION[0]:
                    fg_mask = fg_mask_1[image1_y1:image1_y2, image1_x1:image1_x2]
                    original_hand_image = frame1[image1_y1:image1_y2, image1_x1:image1_x2, :]
                    hand_image = original_hand_image.copy()
                    hand_image_bg_subtracted = cv2.bitwise_and(hand_image, hand_image, mask=fg_mask)
                    hand_image_bg_subtracted = segmentor.forward(hand_image_bg_subtracted)
                    current_events[id].add_hand_images(
                        [timestamp, original_hand_image.copy(), hand_image_bg_subtracted])
            if conf_wrist[1] > HAND_IMAGE_THRESHOLD:
                # Camera 2 image
                if confs_elbow[1] < HAND_IMAGE_THRESHOLD:
                    hand_pos_cam_2 = position_wrist_cam2
                else:
                    elbow_to_hand_direction_vector = position_wrist_cam2 - elbows[1]
                    shift_vector = elbow_to_hand_direction_vector * SHIFT_CONSTANT
                    hand_pos_cam_2 = elbows[1] + shift_vector
                image2_x1, image2_y1, image2_x2, image2_y2 = generate_bounding_box(hand_pos_cam_2[0],
                                                                                   hand_pos_cam_2[1],
                                                                                   hand=True)
                image2_x1, image2_y1, image2_x2, image2_y2 = int(image2_x1), int(image2_y1), int(image2_x2), int(
                    image2_y2)
                if image2_x1 >= 0 and image2_y1 >= 0 and image2_y2 < RESOLUTION[1] and image2_x2 < RESOLUTION[0]:
                    fg_mask = fg_mask_2[image2_y1:image2_y2, image2_x1:image2_x2]
                    original_hand_image = frame2[image2_y1:image2_y2, image2_x1:image2_x2, :]
                    hand_image = original_hand_image.copy()
                    hand_image_bg_subtracted = cv2.bitwise_and(hand_image, hand_image, mask=fg_mask)
                    hand_image_bg_subtracted = segmentor.forward(hand_image_bg_subtracted)
                    current_events[id].add_hand_images(
                        [timestamp, original_hand_image.copy(), hand_image_bg_subtracted])
    elif id in current_events:
        current_events[id].increment_clear_count()
        if current_events[id].event_ended():
            current_events[id].set_end_time(timestamp)
            current_events[id].set_status("completed")
            current_events[id].set_visualisation_attr(camera_ids, timestamps)
            if TEST_PROXIMITY:
                draw_message(position_wrist_cam1, "End", frame1)
                draw_message(position_wrist_cam2, "End", frame2)
            del current_events[id]
            proximity_event_group.decrement_active_num()
            if proximity_event_group.finished():
                proximity_event_group.set_maximum_timestamp(timestamp)
                return True, proximity_event_group, current_events, potential_proximity_events

    return False, proximity_event_group, current_events, potential_proximity_events


def analyze_shoppers(embedder, shared_events_list, EventsLock, events, shelf_id, minimum_timestamp, maximum_timestamp,
                     events_records) -> None:
    print("-------------------------------------------------------------------------------------------------------")
    relevant_events = []
    delete_pos = 0
    print(f"Proximity event group started at {minimum_timestamp} ended at {maximum_timestamp}")
    for i in range(len(shared_events_list)):
        #print(f"Event start at {shared_events_list[i].get_start_time()} end at {shared_events_list[i].get_end_time()} with weight change {shared_events_list[i].get_weight_change()}")
        if ((shared_events_list[i].get_end_time() >= minimum_timestamp) and
              (maximum_timestamp >= shared_events_list[i].get_start_time() + MINIMUM_GROUP_WEIGHT_EVENT_THRESHOLD)):
            relevant_events.append(shared_events_list[i])
        elif shared_events_list[i].get_start_time() > maximum_timestamp - MINIMUM_GROUP_WEIGHT_EVENT_THRESHOLD:
            delete_pos = i
            break
    if len(relevant_events) == 0:
        print("No event")
        return None
    else:
        print("The followings are events to be concerned")
        print("***************************************************")
    # Print results

    print("***************************************************")

    for weight_event in relevant_events:
        print(f"Official weights event starts at {weight_event.get_start_time()} ends at {weight_event.get_end_time()} with weight change {weight_event.get_weight_change()}")

    print("***************************************************")


    EventsLock.acquire()
    # Possibly can delete up to all events taken for this analysis
    del shared_events_list[:delete_pos]
    EventsLock.release()
    # Start analyzing
    shelf_id = 'shelf_' + str(shelf_id)
    product_list = embedder.get_products(shelf_id)

    print("Products are:")
    for prod in product_list:
        print(f"{prod['sku']}")


    # Find probability matrix between products and weight events
    A_prod_weight = np.zeros((len(product_list), len(relevant_events)))
    for i, weight_event in enumerate(relevant_events):
        weight_change = abs(weight_event.get_weight_change())
        sum_val = 0
        for j, product in enumerate(product_list):
            dist_between_prod_and_change = abs(weight_change - product['weight'])
            sum_val += 1 / (dist_between_prod_and_change + EPSILON)

        for j, product in enumerate(product_list):
            dist_between_prod_and_change = abs(weight_change - product['weight'])
            A_prod_weight[j, i] = (1 / (dist_between_prod_and_change + EPSILON)) / sum_val

    # Build affinity matrix between proximity events and actions with regard to items on the shelf
    N = len(events)
    N_W = len(relevant_events)
    M = len(product_list)
    A = np.zeros((N * N_W, M * N_W * 2))
    for i, proximity_event in enumerate(events):
        print(f"Proximity event of {proximity_event.get_person_id()} with starts at"
              f" {proximity_event.get_start_time()} ends at {proximity_event.get_end_time()}")
        vision_probabilities = []
        hand_images = proximity_event.get_hand_images()
        # Start gathering vision probabilities of this proximity event relative to each of the weight event in the group

        # Gather indices of weight events that have intersection - only those are given vision probabilities
        weight_events_for_this_prox_event_ids = []
        for z, weight_event in enumerate(relevant_events):
            intersection_start = max(proximity_event.get_start_time(), weight_event.get_start_time())
            intersection_end = min(proximity_event.get_end_time(), weight_event.get_end_time())
            intersection_length = max(0, intersection_end - intersection_start)
            if intersection_length != 0:
                weight_events_for_this_prox_event_ids.append(z)
        # Gather the closet weight event to each hand image
        hand_images_for_weight_events = {}
        if len(weight_events_for_this_prox_event_ids) > 0 and len(hand_images) > 0:
            for timestamp, image, image_masked in hand_images:
                nearest_weight_event = [None, float('inf')]
                for weight_event_index in range(N_W):
                    if weight_event_index in weight_events_for_this_prox_event_ids:
                        start_time = relevant_events[weight_event_index].get_start_time()
                        end_time = relevant_events[weight_event_index].get_end_time()
                        if abs(timestamp - start_time) < nearest_weight_event[1] and abs(timestamp - start_time) <= abs(timestamp - end_time):
                            nearest_weight_event = [weight_event_index, abs(timestamp - start_time)]
                        elif abs(timestamp - end_time) < nearest_weight_event[1] and abs(timestamp - end_time) <= abs(timestamp - start_time):
                            nearest_weight_event = [weight_event_index, abs(timestamp - end_time)]
                if nearest_weight_event[0] not in hand_images_for_weight_events:
                    hand_images_for_weight_events[nearest_weight_event[0]] = [[np.array(image_masked), timestamp]]
                else:
                    hand_images_for_weight_events[nearest_weight_event[0]].append([np.array(image_masked), timestamp])

            # Save images related to each proximity events and delete images if there are too many
            for index, key in enumerate(hand_images_for_weight_events.keys()):
                while len(hand_images_for_weight_events[key]) > 30:
                    del hand_images_for_weight_events[key][1::2]
                directory_path = f'hand_images/{proximity_event.get_person_id()}_{proximity_event.get_start_time()}_{proximity_event.get_end_time()}/{index}'
                os.makedirs(directory_path, exist_ok=True)
                for image_index, image_splitted in enumerate(hand_images_for_weight_events[key]):
                    file_path = os.path.join(directory_path, f'image_{image_splitted[1]}.png')
                    cv2.imwrite(file_path, image_splitted[0])

        # Gather vision probabilities for this proximity event relative to each weight event that is relevant
        for weight_event_index in range(N_W):
            if weight_event_index in weight_events_for_this_prox_event_ids:
                affinity_array = []
                if len(hand_images) == 0 or weight_event_index not in hand_images_for_weight_events:
                    for j in range(len(product_list)):
                        affinity_score = -1
                        affinity_array.append(affinity_score)
                    vision_probability = torch.softmax(torch.tensor(affinity_array).float(), 0).numpy()
                    vision_probabilities.append(vision_probability)
                    continue
                hand_images_this_weight_event = np.array(hand_images_for_weight_events[weight_event_index], dtype=object)[:, 0]
                top_k = embedder.search_many(shelf_id, hand_images_this_weight_event)
                top_k_mapping = {}
                for item_result in top_k:
                    top_k_mapping[item_result.fields['sku']] = item_result.score
                for j, product in enumerate(product_list):
                    if product['sku'] in top_k_mapping:
                        affinity_score = top_k_mapping[product['sku']]
                    else:
                        affinity_score = -1
                    affinity_array.append(affinity_score)
                vision_probability = torch.softmax(torch.tensor(affinity_array).float(), 0).numpy()
            else:
                vision_probability = None
            vision_probabilities.append(vision_probability)

        # Start adding data of this proximity event to the affinity matrix for action recognition
        for j, product in enumerate(product_list):
            for i_prox in range(N_W * i, N_W * i + N_W):
                if vision_probabilities[i_prox % N_W] is not None:
                    print(f"Vision probability for this event for weight event number {i_prox % N_W} is {vision_probabilities[i_prox % N_W]}")
                    for idx1 in range(N_W * j, N_W * j + N_W):
                        A[i_prox, idx1] += vision_probabilities[i_prox % N_W][j] * weight_v
                    for idx2 in range(((N_W * M) + N_W * j), ((N_W * M) + N_W * j + N_W)):
                        A[i_prox, idx2] += vision_probabilities[i_prox % N_W][j] * weight_v
            for z, weight_event in enumerate(relevant_events):
                row_associated_with_this_event = N_W * i + z
                intersection_start = max(proximity_event.get_start_time(), weight_event.get_start_time())
                intersection_end = min(proximity_event.get_end_time(), weight_event.get_end_time())
                intersection_length = max(0, intersection_end - intersection_start)
                union_length = ((proximity_event.get_end_time() - proximity_event.get_start_time()) +
                                (weight_event.get_end_time() - weight_event.get_start_time()))
                iou = intersection_length / union_length
                if weight_event.get_weight_change() <= 0:
                    # Take
                    for idx1 in range(N_W * j, N_W * j + N_W):
                        A[row_associated_with_this_event, idx1] += iou * A_prod_weight[j, z] * weight_w
                else:
                    # Put
                    for idx2 in range(((N_W * M) + N_W * j), ((N_W * M) + N_W * j + N_W)):
                        A[row_associated_with_this_event, idx2] += iou * A_prod_weight[j, z] * weight_w

    print("AFFINITY MATRIX BETWEEN PROXIMITY EVENTS AND POSSIBLE ACTIONS")
    print(A)

    indices_events, indices_actions = linear_sum_assignment(A, maximize=True)
    sorted_indices = np.argsort(-A[indices_events, indices_actions])
    items_event_recorded = []
    for event_index, action_index in zip(indices_events[sorted_indices], indices_actions[sorted_indices]):
        if len(items_event_recorded) == N_W:
            break
        event = events[int(event_index / N_W)]
        visualisation_attr = event.get_visualisation_attr()
        item_type_num = int(((action_index % (N_W * M)) / N_W))
        weight_event_type = int(event_index % N_W)
        if weight_event_type not in items_event_recorded:
            items_event_recorded.append(weight_event_type)
            # Take action
            if action_index < N_W * M:
                message = f"ID {event.get_person_id()} takes {product_list[item_type_num]['sku']}"
            # Put action
            else:
                message = f"ID {event.get_person_id()} return {product_list[item_type_num]['sku']}"
            # Add data for visualisation
            for key in visualisation_attr.keys():
                if key not in events_records:
                    events_records[key] = {}
                if visualisation_attr[key] not in events_records[key]:
                    events_records[key][visualisation_attr[key]] = [message]
                else:
                    events_records[key][visualisation_attr[key]].append(message)
            print(message)
        else:
            continue

def shifted_sigmoid(x):
  return 2 * (1 / (1 + np.exp(-x)) - 0.5)


def evaluate_hand_segment(elbows, position_wrist_cam1, position_wrist_cam2, frame1, frame2, fg_mask_1, fg_mask_2, timestamp, current_hand_images):
    if len(current_hand_images) > 50:
        del current_hand_images[1::2]
        current_images = np.array(current_hand_images)
        top_k = embedder.search_many("shelf_2", current_images[:, 0])
        print("*************************************************************************************")
        print("---------------------------------------------------------------------------------")
        print("Before background removal")
        for item_result in top_k:
            print(f"Similarity score to item {item_result.fields['sku']} is {item_result.score}")
        print("---------------------------------------------------------------------------------")
        top_k = embedder.search_many("shelf_2", current_images[:, 2])
        print("---------------------------------------------------------------------------------")
        print("After background removal")
        for item_result in top_k:
            print(f"Similarity score to item {item_result.fields['sku']} is {item_result.score}")
        print("---------------------------------------------------------------------------------")

    elbow_to_hand_direction_vector = position_wrist_cam1 - elbows[0]
    shift_vector = elbow_to_hand_direction_vector * SHIFT_CONSTANT
    hand_pos_cam_1 = elbows[0] + shift_vector
    image1_x1, image1_y1, image1_x2, image1_y2 = generate_bounding_box(hand_pos_cam_1[0],
                                                                        hand_pos_cam_1[1],
                                                                        hand=True)
    image1_x1, image1_y1, image1_x2, image1_y2 = int(image1_x1), int(image1_y1), int(image1_x2), int(
        image1_y2)
    if image1_x1 >= 0 and image1_y1 >= 0 and image1_y2 < RESOLUTION[1] and image1_x2 < RESOLUTION[0]:
        hand_image = frame1[image1_y1:image1_y2, image1_x1:image1_x2, :]
        masked_image = segmentor.forward(hand_image.copy())
        fg_mask = fg_mask_1[image1_y1:image1_y2, image1_x1:image1_x2]
        hand_image_bg_and_hand_subtracted = cv2.bitwise_and(masked_image, masked_image, mask=fg_mask)
        hand_image_bg_subtracted = cv2.bitwise_and(hand_image.copy(), hand_image.copy(), mask=fg_mask)
        cv2.imwrite(f"hand_images/{timestamp}_1.png", hand_image_bg_subtracted)
        current_hand_images.append([hand_image.copy(), hand_image_bg_and_hand_subtracted, hand_image_bg_subtracted])


    elbow_to_hand_direction_vector = position_wrist_cam2 - elbows[1]
    shift_vector = elbow_to_hand_direction_vector * SHIFT_CONSTANT
    hand_pos_cam_2 = elbows[1] + shift_vector
    image2_x1, image2_y1, image2_x2, image2_y2 = generate_bounding_box(hand_pos_cam_2[0],
                                                                        hand_pos_cam_2[1],
                                                                        hand=True)
    image2_x1, image2_y1, image2_x2, image2_y2 = int(image2_x1), int(image2_y1), int(image2_x2), int(
        image2_y2)
    if image2_x1 >= 0 and image2_y1 >= 0 and image2_y2 < RESOLUTION[1] and image2_x2 < RESOLUTION[0]:
        hand_image = frame2[image2_y1:image2_y2, image2_x1:image2_x2, :]
        masked_image = segmentor.forward(hand_image.copy())
        fg_mask = fg_mask_2[image2_y1:image2_y2, image2_x1:image2_x2]
        hand_image_bg_and_hand_subtracted = cv2.bitwise_and(masked_image, masked_image, mask=fg_mask)
        hand_image_bg_subtracted = cv2.bitwise_and(hand_image.copy(), hand_image.copy(), mask=fg_mask)
        cv2.imwrite(f"hand_images/{timestamp}_2.png", hand_image_bg_and_hand_subtracted)
        current_hand_images.append([hand_image.copy(), hand_image_bg_and_hand_subtracted, hand_image_bg_subtracted])


if __name__ == "__main__":
    # This contains data for visualisation
    your_data = []
    # Check if cuda is available and set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')
    # Start loading in two cameras' calibration data
    camera_matrix_l = np.load('calib_data/camera_matrix_l.npy')
    new_cam_l = np.load('calib_data/new_cam_l.npy')
    new_cam_r = np.load('calib_data/new_cam_r.npy')
    camera_matrix_r = np.load('calib_data/camera_matrix_r.npy')
    dist_l = np.load('calib_data/dist_l.npy')
    dist_r = np.load('calib_data/dist_r.npy')
    tvec_l = np.load("calib_data/tvec_l.npy")
    tvec_r = np.load("calib_data/tvec_r.npy")
    rotm_l = np.load("calib_data/rotm_l.npy")
    rotm_r = np.load("calib_data/rotm_r.npy")
    projection_matrix_l = np.load('calib_data/projection_matrix_l.npy')
    projection_matrix_r = np.load('calib_data/projection_matrix_r.npy')
    ''''# Normalized intrinsic matrices
    normalized_matrix_l = normalize_intrinsic(camera_matrix_l, RESOLUTION[0], RESOLUTION[1])
    normalized_matrix_r = normalize_intrinsic(camera_matrix_r, RESOLUTION[0], RESOLUTION[1])'''

    '''# Remove this line accordingly
    camera_matrix_l = change_intrinsic(camera_matrix_l, 1920, 1080, 640, 480)
    camera_matrix_r = change_intrinsic(camera_matrix_r, 1920, 1080, 640, 480)'''
    # Calibration object
    calibration = Calibration(cameras={
        0: Camera(camera_matrix_l, pose_matrix(rotm_l, tvec_l.flatten()), dist_l[0], get_Kr_inv(camera_matrix_l, rotm_l, tvec_l.flatten())),
        1: Camera(camera_matrix_r, pose_matrix(rotm_r, tvec_r.flatten()), dist_r[0], get_Kr_inv(camera_matrix_r, rotm_r, tvec_r.flatten()))
    })
    # Shelf fixed coordinates in two camera views - This needs to be changed whenever cameras' position changes
    shelf_cam_1 = Shelf(SHELF_DATA_TWO_CAM[0])
    shelf_cam_2 = Shelf(SHELF_DATA_TWO_CAM[1])
    # 3D location of shelf points
    shelf_points_3d = calibration.triangulate_complete_pose(
        np.array([shelf_cam_1.get_points(), shelf_cam_2.get_points()]), [0, 1], [[640, 480], [640, 480]])
    # Backgrounds extractor
    fgbg1 = cv2.createBackgroundSubtractorKNN()
    fgbg2 = cv2.createBackgroundSubtractorKNN()
    if TEST_CALIBRATION:
        # Shelf points reprojected
        envi_image_1 = cv2.imread("images/environmentLeft/1.png")
        envi_image_2 = cv2.imread("images/environmentRight/2.png")
        new_frame_1 = draw_id_2(calibration.project(np.array(shelf_points_3d), 0), envi_image_1)
        new_frame_2 = draw_id_2(calibration.project(np.array(shelf_points_3d), 1), envi_image_2)
        cv2.imwrite("images/reprojected_envi_1.png", new_frame_1)
        cv2.imwrite("images/reprojected_envi_2.png", new_frame_2)
    # Embedder used for product embedding
    if not TEST_PROXIMITY and not TEST_CALIBRATION:
        embedder = Embedder()
        embedder.initialise()
    # Get the object plane
    a, b, c, d = get_plane_equation_shelf_points(shelf_points_3d)
    object_plane_normal = np.array([a, b, c])
    object_plane_eq = np.array([a, b, c, d])
    x_vis, y_vis, z_vis = plane_grid(object_plane_normal, d)
    # Get the clipping planes
    a_right, b_right, c_right, d_right = (
        get_clipping_plane_type_one(shelf_points_3d[1], shelf_points_3d[2], object_plane_normal))
    right_clipping_plane_normal = np.array([a_right, b_right, c_right])
    right_plane_eq = np.array([a_right, b_right, c_right, d_right])
    check_point_on_plane(shelf_points_3d[1], right_plane_eq)
    check_point_on_plane(shelf_points_3d[2], right_plane_eq)
    x2_vis, y2_vis, z2_vis = plane_grid(right_clipping_plane_normal, d_right)

    a_top, b_top, c_top, d_top = get_clipping_plane_type_one(
        shelf_points_3d[0], shelf_points_3d[1], object_plane_normal)
    top_clipping_plane_normal = np.array([a_top, b_top, c_top])
    top_plane_eq = np.array([a_top, b_top, c_top, d_top])

    a_left, b_left, c_left, d_left = get_clipping_plane_type_two(object_plane_normal,
                                                                 top_clipping_plane_normal, shelf_points_3d[0])
    left_clipping_plane_normal = np.array([a_left, b_left, c_left])
    left_plane_eq = np.array([a_left, b_left, c_left, d_left])
    check_point_on_plane(shelf_points_3d[0], left_plane_eq)
    x4_vis, y4_vis, z4_vis = plane_grid(left_clipping_plane_normal, d_left)

    # Timer

    camera_start = time.time()
    
    SOURCE_1 = 3
    SOURCE_2 = 6

    if USE_REPLAY:
        with open('videos/chronology3.json') as file:
            chronology = json.load(file)

        camera_start = chronology['start']

        SOURCE_1 = 'videos/4.avi'
        SOURCE_2 = 'videos/5.avi'

        BAKES = {
             0: 'bakes/4.bake', # Bake for camera id 0
             1: 'bakes/5.bake',
        }
        #BAKES = None

        if BAKES:
            for camera in BAKES:
                with open(BAKES[camera], 'r') as handle:
                    BAKES[camera] = json.load(handle)


    if USE_MULTIPROCESS:
        cap = Stream(SOURCE_1, SOURCE_2, camera_start, RESOLUTION)
        cap.start()
    else:
        cap = cv2.VideoCapture(SOURCE_1)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('m', 'j', 'p', 'g'))
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, RESOLUTION[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION[1])

        cap2 = cv2.VideoCapture(SOURCE_2)
        cap2.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('m', 'j', 'p', 'g'))
        cap2.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
        cap2.set(cv2.CAP_PROP_FRAME_WIDTH, RESOLUTION[0])
        cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION[1])

    # Variables for storing shared weights data and locks
    EventsLock = mp.Lock()
    shared_events_list = mp.Manager().list()
    # Subprocess for weight sensor
    if not TEST_CALIBRATION and not TEST_PROXIMITY:
        if USE_REPLAY:
            weight_p = mp.Process(target=gather_weights, args=(shared_events_list, EventsLock, camera_start, chronology['weights']))
        else:
            weight_p = mp.Process(target=gather_weights, args=(shared_events_list, EventsLock, camera_start))
        weight_p.start()

    # Variables storing face images for re-identification, if an id's image hasn't been available for a while, delete
    images_by_id = dict()
    # Storing the invalid IDs that already got fix
    invalid_ids = []
    # Variable used to halt recording to start visualisation after a certain number of frames
    count = 0
    # Variables for storing visualization data
    output_dir_1 = "frames_data_cam_0"
    output_dir_2 = "frames_data_cam_1"
    output_dir_1_reprojected = "reprojected_frames_data_cam_0"
    output_dir_2_reprojected = "reprojected_frames_data_cam_1"
    delete_directory(output_dir_1)
    delete_directory(output_dir_2)
    delete_directory(output_dir_1_reprojected)
    delete_directory(output_dir_2_reprojected)
    cam_1_frames = {}
    cam_2_frames = {}
    cam_1_frames_reprojected = {}
    cam_2_frames_reprojected = {}
    # Data for poses along the timeline
    poses_2d_all_frames = []
    poses_3d_all_timestamps = defaultdict(list)
    unmatched_detections_all_frames = defaultdict(list)
    # World ltrb
    world_ltrb = calibration.compute_world_ltrb()
    # Iteration variables and ID variable for assigning new ID
    retrieve_iterations = 0
    new_id = -1
    iterations = 0
    new_id_last_update_timestamp = 0
    # Dictionary storing proximity events belong to different people ID
    current_events = {}
    # Proximity event group for the ONLY SHELF
    proximity_event_group = None
    # Dictionary storing records of events for visualisation
    events_records = {}
    cams_frames = [cam_1_frames, cam_2_frames]
    cams_frames_reprojected = [cam_1_frames_reprojected, cam_2_frames_reprojected]
    # Dictionary containing potential proximity events
    potential_proximity_events = {}
    local_feats_dict = {}

    lost_tracks = {}

    # Pose detector
    if BAKES is None:
        detector = HumanPoseDetection()
    if EVALUATE_HAND_SEGMENT:
        current_hand_images = []
    try:
        while True:
            camera_data = []
            if USE_MULTIPROCESS:
                res = cap.get()
                start = time.time()
                while not res:
                    res = cap.get()
                if retrieve_iterations >= START_ITERATION:
                    timestamp_1, img, timestamp_2, img2 = res
                else:
                    retrieve_iterations += 1
                    continue
            else:
                _, img = cap.read()
                _, img2 = cap2.read()
                timestamp_1 = time.time() - camera_start
                timestamp_2 = time.time() - camera_start

            if USE_REPLAY:
                timestamp_1, timestamp_2 = chronology['frames'][retrieve_iterations]
            else:
                pass
                #timestamp_1, timestamp_2 = timestamp_1 - camera_start, timestamp_2 - camera_start

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
            camera_data.append([img, timestamp_1])
            camera_data.append([img2, timestamp_2])
            # Start checking if an ID has been corrected
            # This way probably goes very wrong in the case of ID being swapped around and only meant to handle the case
            # where a new id is given to a person due to entering the scene back or being occluded

            '''for i in images_by_id:
                num_images = len(images_by_id[i])
                if num_images > 50 and i not in invalid_ids:
                    while num_images > 50:
                        del images_by_id[i][1::2]
                    #local_feats_dict[i] = extract_features(images_by_id[i])
                    local_feats_dict[i] = LATransformerForward(latreid, latreid_device, images_by_id[i]).reshape(len(images_by_id[i]), -1)
            if retrieve_iterations % 30 == 0:
                for track_id_1 in local_feats_dict.keys():
                    # h = local_feats_dict[track_id_1].shape[1]
                    # if local_feats_dict[track_id_1].shape[1] < MIN_NUM_FEATURES:
                    #     continue
                    similarity_scores = []
                    for track_id_2 in local_feats_dict.keys():
                        # if track_id_2 == track_id_1 or local_feats_dict[track_id_2].shape[1] < MIN_NUM_FEATURES or \
                        #         track_id_2 > track_id_1:
                        #     continue
                        if not (track_id_2 < track_id_1) or track_id_2 not in lost_tracks: # Only REID from larger id to smaller (track_id_1 -> track_id_2) and only if target is lost
                            continue
                        # Start checking if the ID belong to a different track by appearance, if yes, then change the id of both
                        # poses_2d_all_frames and poses_3d_all_times, and delete the images related to this "wrong id"
                        #tmp = np.mean(compute_distance(local_feats_dict[track_id_1], local_feats_dict[track_id_2]))
                        tmp = (local_feats_dict[track_id_1].mean(0) - local_feats_dict[track_id_2].mean(0)).norm().numpy()
                        similarity_scores.append([track_id_2, tmp])
                        print(f"Score between id {track_id_1} and {track_id_2} is {tmp}")
                    if similarity_scores:
                        similarity_scores.sort(key=operator.itemgetter(1))
                        for index in range(len(similarity_scores)):
                            if similarity_scores[index][1] < similarity_threshold:
                                print("Reid succesfulyyyyyyyy!!!!!!!!! from ID " + str(track_id_1) + " to ID " + str(
                                    similarity_scores[index][0]))
                                # This track id becomes invalid so that if the subprocess give embeddings for this id deny it
                                # and also delete images related to this id
                                real_id = similarity_scores[index][0]
                                invalid_ids.append(track_id_1)
                                del images_by_id[track_id_1]
                                # Change the wrong id to the real id in recent data of poses_2d_all_frames
                                for index_1 in range(len(poses_2d_all_frames) - 1, 0, -1):
                                    this_timestamp = poses_2d_all_frames[index_1]['timestamp']
                                    if (timestamp_2 - this_timestamp) > delta_time_threshold:
                                        break
                                    for pose_index in range(len(poses_2d_all_frames[index_1]['poses'])):
                                        if poses_2d_all_frames[index_1]['poses'][pose_index]['id'] == track_id_1:
                                            poses_2d_all_frames[index_1]['poses'][pose_index]['id'] = real_id
                                            break
                                # Change the wrong id to the real id in recent data of poses_3d_all_timestamps
                                for index_2 in range(len(poses_3d_all_timestamps) - 1, 0, -1):
                                    this_timestamp = list(poses_3d_all_timestamps.keys())[index_2]
                                    # time window ends return the ID
                                    if (timestamp_2 - this_timestamp) > delta_time_threshold:
                                        break
                                    # to get 3d pose at timestamp before the timestamp at the current frame
                                    if (this_timestamp >= timestamp_2) or all(value is None for value in poses_3d_all_timestamps[this_timestamp])\
                                            or poses_3d_all_timestamps[this_timestamp] is None:
                                        continue
                                    for id_index in range(len(poses_3d_all_timestamps[this_timestamp])):
                                        if poses_3d_all_timestamps[this_timestamp][id_index] is None:
                                            continue
                                        if poses_3d_all_timestamps[this_timestamp][id_index]['id'] == track_id_1:
                                            poses_3d_all_timestamps[this_timestamp][id_index]['id'] = real_id
                                            break

                                # Remove duplicate events
                                old_left = f"{str(track_id_1)}_left"
                                old_right = f"{str(track_id_1)}_right"
                                left = f"{str(real_id)}_left"
                                right = f"{str(real_id)}_right"

                                if old_left in current_events and left in current_events:
                                    if left in current_events:
                                        current_events[left].merge_event(current_events[old_left])
                                        del current_events[old_left]
                                        proximity_event_group.decrement_active_num()
                                    else:
                                        current_events[left] = current_events[old_left]
                                        del current_events[old_left]

                                if old_right in current_events and right in current_events:
                                    if right in current_events:
                                        current_events[right].merge_event(current_events[old_left])
                                        del current_events[old_right]
                                        proximity_event_group.decrement_active_num()
                                    else:
                                        current_events[right] = current_events[old_right]
                                        del current_events[old_right]

                                del lost_tracks[real_id] # Mark track matched to as no longer lost

                                break

                                #print("Not similar enough:", similarity_scores[index][1])
            for invalid_id in invalid_ids:
                del local_feats_dict[invalid_id]
            invalid_ids = []'''
            for camera_id, data in enumerate(camera_data):
                # List containing tracks and detections after retrieve_iterations += 1ungarian Algorithm
                indices_T = []
                indices_D = []
                frame, timestamp = data  # Get the frame (image) and timestamp for this camera_id
                if retrieve_iterations == 0:
                    cv2.imwrite(f"no_bake_{camera_id}.png", frame)
                # Extract background
                if camera_id == 0:
                    fg_mask = fgbg1.apply(frame)
                else:
                    fg_mask = fgbg2.apply(frame)
                fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
                # Get pose estimation data for this frame
                frame_resized = cv2.pyrDown(frame.copy())

                points_2d_cur_frames = []
                boxes_2d_cur_frames = []
                points_2d_scores_cur_frames = []
                points_2d_cur_frames_1 = []
                boxes_2d_cur_frames_1 = []
                points_2d_scores_cur_frames_1 = []
                if USE_REPLAY and BAKES:
                    poses_keypoints = np.array(BAKES[camera_id][str(timestamp)][0])
                    poses_bboxes = np.array(BAKES[camera_id][str(timestamp)][1])
                    poses_conf = np.array(BAKES[camera_id][str(timestamp)][2])
                    if len(poses_keypoints) == 0:
                        iterations += 1
                        poses_3d_all_timestamps[timestamp].append(None)
                        continue
                else:
                    # Get pose estimation data for this frame
                    poses_data_cur_frame = detector.predict(frame_resized)[0]
                    try:
                        poses_keypoints = poses_data_cur_frame.keypoints.xy.cpu().numpy()
                        poses_keypoints *= 2
                        poses_bboxes = poses_data_cur_frame.boxes.xywh.cpu().numpy()
                        poses_bboxes *= 2
                        poses_conf = poses_data_cur_frame.keypoints.conf.cpu().numpy()
                    except Exception:
                        iterations += 1
                        poses_3d_all_timestamps[timestamp].append(None)
                        continue

                    # Check to eliminate duplicate detections from yolo
                eliminated_pose_indices = []
                for i, pose_1 in enumerate(poses_keypoints):
                    if i in eliminated_pose_indices:
                        continue
                    distance_between_two_poses = 0
                    for j, pose_2 in enumerate(poses_keypoints[i + 1:]):
                        if i + j + 1 in eliminated_pose_indices:
                            continue
                        mask_1 = np.invert((pose_1 == DET_UNASSIGNED).all(1))
                        mask_2 = np.invert((pose_2 == DET_UNASSIGNED).all(1))
                        comb = np.bitwise_and(mask_1, mask_2)
                        distance_between_two_poses = np.linalg.norm(pose_2[comb] - pose_1[comb], axis=0).mean()
                        if distance_between_two_poses < DUPLICATE_POSES_THRESHOLD:
                            eliminated_pose_indices.append(i + j + 1)

                for poses_index in range(len(poses_keypoints)):
                    if poses_index not in eliminated_pose_indices:
                        boxes_2d_cur_frames.append(poses_bboxes[poses_index])
                        points_2d_cur_frames.append(poses_keypoints[poses_index])
                        points_2d_scores_cur_frames.append(poses_conf[poses_index])

                points_2d_cur_frames = np.array(points_2d_cur_frames)
                points_2d_scores_cur_frames = np.array(points_2d_scores_cur_frames)
                location_of_camera_center_cur_frame = calibration.cameras[camera_id].location
                poses_2d_all_frames.append({
                    'camera': camera_id,
                    'frame': frame,
                    'foreground_mask': fg_mask,
                    'timestamp': timestamp,
                    'poses': [{'id': -1, 'points_2d': points_2d_cur_frames[index], 'conf': points_2d_scores_cur_frames[index]} for index in
                              range(len(points_2d_cur_frames))],
                })
                poses_3D_latest = get_latest_3D_poses_available_for_cur_timestamp(poses_3d_all_timestamps, timestamp,
                                                                                  delta_time_threshold=delta_time_threshold)
                N_3d_poses_last_timestamp = len(poses_3D_latest)
                M_2d_poses_this_camera_frame = len(points_2d_cur_frames)
                Dt_c = points_2d_cur_frames # Shape (M poses on frame , no of body points , 2)
                Dt_boxes_c = np.array(boxes_2d_cur_frames)
                Dt_c_scores = points_2d_scores_cur_frames
                # Affinity matrix associating N current tracks and M detections
                A = np.zeros(
                    (N_3d_poses_last_timestamp, M_2d_poses_this_camera_frame))  # Cross-view association matrix shape N x M
                for i in range(N_3d_poses_last_timestamp):  # Iterate through prev N Target poses
                    x_t_tilde_tilde_c = poses_3D_latest[i]['detections'][camera_id]
                    delta_t_2d = timestamp - poses_3D_latest[i]['timestamps_2d'][camera_id]
                    confidences_last_2D = poses_3D_latest[i]['confidences'][camera_id]
                    # x_t_tilde_tilde_c = calibration.project(np.array(poses_3D_latest[i]['points_3d']), camera_id)
                    delta_t_3d = timestamp - poses_3D_latest[i]['timestamp']
                    for j in range(M_2d_poses_this_camera_frame):  # Iterate through M poses
                        # Each detection (Dj_tme will have k body points for every camera c
                        # x_t_c in image coordinate_c) in this frame
                        # x_t_c_norm scale normalized image coordinates
                        x_t_c_norm = Dt_c[j].copy()
                        '''
                        x_t_c_norm[:, 0] = x_t_c_norm[:, 0] / RESOLUTION[0]
                        x_t_c_norm[:, 1] = x_t_c_norm[:, 1] / RESOLUTION[1]'''
                        K_joints_detected_this_person = len(x_t_c_norm)
                        # Need to implement back_project
                        back_proj_x_t_c_to_ground = calibration.cameras[camera_id].back_project(x_t_c_norm,
                                                                                                z_worlds=np.zeros(
                                                                                                    K_joints_detected_this_person))
                        '''kps_rays = calibration.cameras[camera_id].unproject_uv_to_rays(x_t_c_norm)'''
                        affinity_geometric_dist = 0
                        affinity_geo_2d = 0
                        affinity_geo_3d = 0
                        affinity_angles = 0
                        for k in range(K_joints_detected_this_person):  # Iterate through K keypoints
                            target_joint = poses_3D_latest[i]['points_3d'][k]
                            # Calculating A2D between target's last updated joint K from the current camera
                            A_2D = 0
                            if Dt_c_scores[j][k] > UNSEEN_THRESHOLD and confidences_last_2D[k] > UNSEEN_THRESHOLD:
                                distance_2D = np.linalg.norm(x_t_c_norm[k] - x_t_tilde_tilde_c[k])  # Distance between joints
                                A_2D = w_2D * (1 - distance_2D / (alpha_2D * delta_t_2d)) * np.exp(
                                    -lambda_a * delta_t_2d)
                                # Normalize between -1 and 0
                                if A_2D < 0:
                                    A_2D = shifted_sigmoid(A_2D)
                                #print(f"Distance 2D is {distance_2D}, with velocity threshold estimated {alpha_2D * delta_t_2d} and time delta {delta_t_2d} and confidence {Dt_c_scores[j][k]}, {confidences_last_2D[k]}")

                            # Calculating A3D between predicted position in 3D space of the target's joint and the detection's joint projected into 3D
                            A_3D = 0
                            if (not np.all(target_joint == UNASSIGNED)) and (Dt_c_scores[j][k] > UNSEEN_THRESHOLD):
                                velocity_t_tilde = np.array(poses_3D_latest[i]['velocity'][k])
                                predicted_X_t = np.array(target_joint) + (velocity_t_tilde * delta_t_3d)
                                dl = calculate_perpendicular_distance(point=predicted_X_t,
                                                                      line_start=location_of_camera_center_cur_frame,
                                                                      line_end=back_proj_x_t_c_to_ground[k])
                                A_3D = w_3D * (1 - dl / alpha_3D) * np.exp(-lambda_a * delta_t_3d)
                                # Normalize between -1 and 0
                                if A_3D < 0:
                                    A_3D = shifted_sigmoid(A_3D)
                                #print(f"Distance 3D: {dl} for joint number {k}, confidence {Dt_c_scores[j][k]}")
                            # Add the affinity between the pair of target and detection in terms of this specific joint

                            affinity_geometric_dist += A_2D + A_3D
                            affinity_geo_2d += A_2D
                            affinity_geo_3d += A_3D

                        for triplet in TRIPLETS:
                            # Joints angle for this triplet from the last 2D detection of the track from this camera
                            first_joint_last_2D = x_t_tilde_tilde_c[get_index_from_key(triplet[0])]
                            second_joint_last_2D = x_t_tilde_tilde_c[get_index_from_key(triplet[1])]
                            third_joint_last_2D = x_t_tilde_tilde_c[get_index_from_key(triplet[2])]
                            joints_angle_last_2D = calculate_angle(first_joint_last_2D, second_joint_last_2D,
                                                                   third_joint_last_2D)
                            confidences_triplets = [confidences_last_2D[get_index_from_key(triplet[0])],
                                                    confidences_last_2D[get_index_from_key(triplet[1])],
                                                    confidences_last_2D[get_index_from_key(triplet[2])]]
                            # Joints angle for this triplet from the this detection
                            first_joint = x_t_c_norm[get_index_from_key(triplet[0])]
                            second_joint = x_t_c_norm[get_index_from_key(triplet[1])]
                            third_joint = x_t_c_norm[get_index_from_key(triplet[2])]
                            joints_angle = calculate_angle(first_joint, second_joint, third_joint)
                            confidences_dt_triplets = [Dt_c_scores[j][get_index_from_key(triplet[0])],
                                                    Dt_c_scores[j][get_index_from_key(triplet[1])],
                                                    Dt_c_scores[j][get_index_from_key(triplet[2])]]
                            # Calculate affinity
                            angle_affinity = affinity_score_avg_product(joints_angle_last_2D, joints_angle,
                                                                        confidences_triplets, confidences_dt_triplets)
                            affinity_angles += angle_affinity

                        # Combining affinity
                        A[i, j] += affinity_geometric_dist * w_geometric_dist + affinity_angles * w_angle
                        #print(f"Iteration: {retrieve_iterations : <5} Timestamp: {timestamp : <5} Camera: {camera_id : <4} Target: {i : <3}  <-> Pose: {j : <3} Affinity: {round(A[i, j], 5) : <12} A2D: {round(w_geometric_dist * affinity_geo_2d, 5) : <12} A3D: {round(w_geometric_dist * affinity_geo_3d, 5) : <12} Angle: {round(w_angle * affinity_angles, 5) : <7}")

                matched = set()
                # Hungarian algorithm able to assign detections to tracks based on Affinity matrix
                indices_T, indices_D = linear_sum_assignment(A, maximize=True)
                tracking_visualization_frame = frame.copy()
                for i, j in zip(indices_T, indices_D):
                    track_id = poses_3D_latest[i]['id']
                    poses_2d_all_frames[-1]['poses'][j]['id'] = track_id
                    #draw_id(poses_2d_all_frames[-1]['poses'][j], tracking_visualization_frame)
                    # Store images related to this track
                    x, y, w, h = Dt_boxes_c[j]
                    top_left_point = (int(x - w / 2), int(y - h / 2))
                    bottom_right_point = (int(x + w / 2), int(y + h / 2))
                    draw_id(top_left_point, bottom_right_point, track_id, tracking_visualization_frame)
                    confidence_bounding_box = Dt_c_scores[j].mean()
                    if confidence_bounding_box > body_image_thresh:
                        x1, x2, y1, y2 = int(x - w/2), int(x + w/2), int(y - h/2), int(y + h/2)
                        if x1 >= 0 and y1 >= 0 and y2 < RESOLUTION[1] and x2 < RESOLUTION[0]:
                            body_image = frame[y1:y2, x1:x2, :]
                            body_image = cv2.pyrDown(body_image.copy())
                            if track_id not in images_by_id:
                                images_by_id[track_id] = [body_image]
                            else:
                                images_by_id[track_id].append(body_image)
                    # Extract poses data from other camera
                    poses_2d_inc_rec_other_cam = extract_key_value_pairs_from_poses_2d_list(poses_2d_all_frames,
                                                                                            id=track_id,
                                                                                            timestamp_cur_frame=timestamp,
                                                                                            dt_thresh=delta_time_threshold)
                    # move following code in func extract_key_value_pairs_from_poses_2d_list to get *_inc_rec variables directly
                    # Get 2D poses of ID
                    dict_with_poses_for_n_cameras_for_latest_timeframe = separate_lists_for_incremental_triangulation(
                        poses_2d_inc_rec_other_cam)

                    points_2d_inc_rec = []

                    conf_2d_inc_rec = []

                    camera_ids_inc_rec = dict_with_poses_for_n_cameras_for_latest_timeframe['camera']
                    image_wh_inc_rec = dict_with_poses_for_n_cameras_for_latest_timeframe['image_wh']
                    timestamps_inc_rec = dict_with_poses_for_n_cameras_for_latest_timeframe['timestamp']
                    frames = dict_with_poses_for_n_cameras_for_latest_timeframe['frame']
                    fg_masks = dict_with_poses_for_n_cameras_for_latest_timeframe['foreground_mask']

                    for dict_index in range(len(dict_with_poses_for_n_cameras_for_latest_timeframe['poses'])):
                        points_2d_inc_rec.append(
                            dict_with_poses_for_n_cameras_for_latest_timeframe['poses'][dict_index]['points_2d'])
                        conf_2d_inc_rec.append(
                            dict_with_poses_for_n_cameras_for_latest_timeframe['poses'][dict_index]['conf'])

                    # Only take 2 camera at the moment
                    if len(points_2d_inc_rec) != 2:
                        continue
                    K_joints_detected_this_person = len(Dt_c[j])
                    Ti_t = []
                    for k in range(K_joints_detected_this_person):  # iterate through k points
                        # get all the 2d pose point from all the cameras where this target was detected last
                        # i.e. if current frame is from cam 1 then get last detected 2d pose of this target
                        # from all of the cameras. Do triangulation with all cameras with detected ID
                        if conf_2d_inc_rec[0][k] > UNSEEN_THRESHOLD and conf_2d_inc_rec[1][k] > UNSEEN_THRESHOLD:
                            _, Ti_k_t = calibration.linear_ls_triangulate_weighted(np.array(points_2d_inc_rec)[:, k, :],
                                                                                   camera_ids_inc_rec,
                                                                                   image_wh_inc_rec,
                                                                                   lambda_t,
                                                                                   timestamps_inc_rec)
                            Ti_t.append(Ti_k_t.tolist())
                        else:
                            '''delta_t = timestamp - poses_3D_latest[i]['timestamp']
                            target_joint = poses_3D_latest[i]['points_3d'][k]'''
                            Ti_t.append(UNASSIGNED.tolist())
                            # Store frames for visualizing tracking in 2D

                    if (TEST_CALIBRATION or TEST_PROXIMITY) and not EVALUATE_HAND_SEGMENT:
                        draw_id_2(calibration.project(np.array(Ti_t), camera_id), frame)
                    # Detection normalized
                    x_t_c_norm = Dt_c[j].copy()
                    '''
                    x_t_c_norm[:, 0] = x_t_c_norm[:, 0] / RESOLUTION[0]
                    x_t_c_norm[:, 1] = x_t_c_norm[:, 1] / RESOLUTION[1]
                    points_2d_inc_rec[0][:, 0] = points_2d_inc_rec[0][:, 0] / RESOLUTION[0]
                    points_2d_inc_rec[0][:, 1] = points_2d_inc_rec[0][:, 1] / RESOLUTION[1]
                    points_2d_inc_rec[1][:, 0] = points_2d_inc_rec[1][:, 0] / RESOLUTION[0]
                    points_2d_inc_rec[1][:, 1] = points_2d_inc_rec[1][:, 1] / RESOLUTION[1]'''
                    poses_3d_all_timestamps[timestamp].append({'id': poses_3D_latest[i]['id'],
                                                                'points_3d': Ti_t,
                                                                'camera_ID': camera_ids_inc_rec,
                                                                'detections': {
                                                                    camera_ids_inc_rec[0]: points_2d_inc_rec[0],
                                                                    camera_ids_inc_rec[1]: points_2d_inc_rec[1]
                                                                },
                                                                'timestamps_2d': {
                                                                    camera_ids_inc_rec[0]: timestamps_inc_rec[0],
                                                                    camera_ids_inc_rec[1]: timestamps_inc_rec[1]
                                                                },
                                                                'confidences': {
                                                                    camera_ids_inc_rec[0]: conf_2d_inc_rec[0],
                                                                    camera_ids_inc_rec[1]: conf_2d_inc_rec[1]
                                                                },
                                                                })

                    foot_joints = [Ti_t[LEFT_FOOT_POS], Ti_t[RIGHT_FOOT_POS]]
                    conf_foot_joints = [1 / 2 * (conf_2d_inc_rec[0][LEFT_FOOT_POS] + conf_2d_inc_rec[1][LEFT_FOOT_POS]),
                                        1 / 2 * (conf_2d_inc_rec[0][RIGHT_FOOT_POS] + conf_2d_inc_rec[1][
                                            RIGHT_FOOT_POS])]
                    elbows_left = [points_2d_inc_rec[0][LEFT_ELBOW_POS], points_2d_inc_rec[1][LEFT_ELBOW_POS]]
                    elbows_conf_left = [conf_2d_inc_rec[0][LEFT_ELBOW_POS], conf_2d_inc_rec[1][LEFT_ELBOW_POS]]
                    elbows_right = [points_2d_inc_rec[0][RIGHT_ELBOW_POS], points_2d_inc_rec[1][RIGHT_ELBOW_POS]]
                    elbows_conf_right = [conf_2d_inc_rec[0][RIGHT_ELBOW_POS], conf_2d_inc_rec[1][RIGHT_ELBOW_POS]]
                    left_wrist = Ti_t[LEFT_WRIST_POS]
                    conf_left_wrist = 1 / 2 * (conf_2d_inc_rec[0][LEFT_WRIST_POS] + conf_2d_inc_rec[1][LEFT_WRIST_POS])
                    confs_left_wrist = [conf_2d_inc_rec[0][LEFT_WRIST_POS], conf_2d_inc_rec[1][LEFT_WRIST_POS]]
                    right_wrist = Ti_t[RIGHT_WRIST_POS]
                    conf_right_wrist = 1 / 2 * (
                                conf_2d_inc_rec[0][RIGHT_WRIST_POS] + conf_2d_inc_rec[1][RIGHT_WRIST_POS])
                    confs_right_wrist = [conf_2d_inc_rec[0][RIGHT_WRIST_POS], conf_2d_inc_rec[1][RIGHT_WRIST_POS]]
                    person_id = poses_3D_latest[i]['id']

                    if EVALUATE_HAND_SEGMENT:
                        evaluate_hand_segment(elbows_left, points_2d_inc_rec[0][LEFT_WRIST_POS],
                                              points_2d_inc_rec[1][LEFT_WRIST_POS], frames[0], frames[1], fg_masks[0], fg_masks[1], timestamp, current_hand_images)
                        '''evaluate_hand_segment(elbows_right, points_2d_inc_rec[0][RIGHT_WRIST_POS],
                                              points_2d_inc_rec[1][RIGHT_WRIST_POS], frames[0], frames[1], timestamp, current_hand_images)'''

                    # Checking shelf proximity
                    if not TEST_CALIBRATION:
                        # Check shelf proximity for hands
                        group_finished, proximity_event_group, current_events, potential_proximity_events = process_proximity_detection(left_wrist, elbows_left,
                                                                    elbows_conf_left, object_plane_eq,
                                                                    left_plane_eq, right_plane_eq, top_plane_eq,
                                                                    str(person_id) + "_left",
                                                                    timestamps_inc_rec, camera_ids_inc_rec, current_events,
                                                                    proximity_event_group,
                                                                    potential_proximity_events,
                                                                    confs_left_wrist,
                                                                    points_2d_inc_rec[0][LEFT_WRIST_POS],
                                                                    points_2d_inc_rec[1][LEFT_WRIST_POS], frames[0],
                                                                    frames[1], fg_masks[0], fg_masks[1])
                        if group_finished and not TEST_PROXIMITY:
                            analyze_shoppers(embedder, shared_events_list, EventsLock,
                                            proximity_event_group.get_events(),
                                            proximity_event_group.get_shelf_id(),
                                            proximity_event_group.get_minimum_timestamp(),
                                            proximity_event_group.get_maximum_timestamp(),
                                            events_records)
                            proximity_event_group = None


                        group_finished, proximity_event_group, current_events, potential_proximity_events = process_proximity_detection(right_wrist, elbows_right,
                                                                    elbows_conf_right, object_plane_eq,
                                                                    left_plane_eq, right_plane_eq, top_plane_eq,
                                                                    str(person_id) + "_right",
                                                                    timestamps_inc_rec, camera_ids_inc_rec, current_events,
                                                                    proximity_event_group,
                                                                    potential_proximity_events,
                                                                    confs_right_wrist,
                                                                    points_2d_inc_rec[0][RIGHT_WRIST_POS],
                                                                    points_2d_inc_rec[1][RIGHT_WRIST_POS], frames[0],
                                                                    frames[1], fg_masks[0], fg_masks[1])

                        if group_finished and not TEST_PROXIMITY:
                            analyze_shoppers(embedder, shared_events_list, EventsLock,
                                             proximity_event_group.get_events(),
                                             proximity_event_group.get_shelf_id(),
                                             proximity_event_group.get_minimum_timestamp(),
                                             proximity_event_group.get_maximum_timestamp(),
                                             events_records)
                            proximity_event_group = None

                        # Use foot joints to try to start proximity events as well if both hands' confidence are bad
                        if (np.all(left_wrist == UNASSIGNED) and (str(person_id) + "_right") not in current_events):
                            group_finished, proximity_event_group, current_events, potential_proximity_events = (
                                process_proximity_detection_by_foot_joints(foot_joints, conf_foot_joints, object_plane_eq,
                                                                       left_plane_eq,
                                                                       right_plane_eq, top_plane_eq, person_id, timestamps_inc_rec,
                                                                       camera_ids_inc_rec,
                                                                       current_events,
                                                                       proximity_event_group,
                                                                       potential_proximity_events,
                                                                        points_2d_inc_rec[0][LEFT_WRIST_POS],
                                                                        points_2d_inc_rec[1][LEFT_WRIST_POS],
                                                                       frames[0], frames[1], 'LEFT'))
                            if group_finished and not TEST_PROXIMITY:
                                analyze_shoppers(embedder, shared_events_list, EventsLock,
                                                 proximity_event_group.get_events(),
                                                 proximity_event_group.get_shelf_id(),
                                                 proximity_event_group.get_minimum_timestamp(),
                                                 proximity_event_group.get_maximum_timestamp(),
                                                 events_records)
                                proximity_event_group = None
                        elif (np.all(right_wrist == UNASSIGNED) and (str(person_id) + "_left") not in current_events):
                            group_finished, proximity_event_group, current_events, potential_proximity_events = (
                                process_proximity_detection_by_foot_joints(foot_joints, conf_foot_joints,
                                                                           object_plane_eq,
                                                                           left_plane_eq,
                                                                           right_plane_eq, top_plane_eq, person_id,
                                                                           timestamps_inc_rec,
                                                                           camera_ids_inc_rec,
                                                                           current_events,
                                                                           proximity_event_group,
                                                                           potential_proximity_events,
                                                                           points_2d_inc_rec[0][LEFT_WRIST_POS],
                                                                           points_2d_inc_rec[1][LEFT_WRIST_POS],
                                                                           frames[0], frames[1], 'RIGHT'))
                            if group_finished and not TEST_PROXIMITY:
                                analyze_shoppers(embedder, shared_events_list, EventsLock,
                                                 proximity_event_group.get_events(),
                                                 proximity_event_group.get_shelf_id(),
                                                 proximity_event_group.get_minimum_timestamp(),
                                                 proximity_event_group.get_maximum_timestamp(),
                                                 events_records)
                                proximity_event_group = None

                        matched.add(person_id)

                if TEST_PROXIMITY or TEST_CALIBRATION:
                    if camera_id == 0:
                        filename = os.path.join(output_dir_1_reprojected, "{:.3f}".format(timestamp) + '.png')
                        cv2.imwrite(filename, frame)
                    else:
                        filename = os.path.join(output_dir_2_reprojected, "{:.3f}".format(timestamp) + '.png')
                        cv2.imwrite(filename, frame)
                if not TEST_CALIBRATION:
                    targets = {pose['id'] for pose in poses_3D_latest}
                    unmatched_targets = targets - matched

                    # TODO: Consider whether these targets should have been seen from this camera in the first place, i.e did they match with this camera last frame
                    # and thus should they have (likely) appeared in this subsequent camera frame
                    for target in unmatched_targets: # We don't know about the current status of two wrists of a target
                        left = f"{str(target)}_left"
                        right = f"{str(target)}_right"

                        if left in current_events:
                            current_events[left].reset_clear_count()

                        if right in current_events:
                            current_events[right].reset_clear_count()

                        if target not in lost_tracks:
                            lost_tracks[target] = 1
                        else:
                            lost_tracks[target] += 1



                # Store unmatched data
                for j in range(M_2d_poses_this_camera_frame):
                    if j not in indices_D:
                        unmatched_detections_all_frames[retrieve_iterations].append({'camera_id': camera_id,
                                                                                     'timestamp': timestamp,
                                                                                     'points_2d': Dt_c[j],
                                                                                     'frame': frame,
                                                                                     'foreground_mask': fg_mask,
                                                                                     'scores': Dt_c_scores[j],
                                                                                     'image_wh': [RESOLUTION[0],
                                                                                                  RESOLUTION[1]],
                                                                                     'poses_2d_all_frames_pos': len(
                                                                                         poses_2d_all_frames) - 1,
                                                                                     'pose_pos': j,
                                                                                     'bounding_box': Dt_boxes_c[j]})

                iterations += 1

                if iterations % len(calibration.cameras) == 0:
                    if unmatched_detections_all_frames[retrieve_iterations]:
                        unique_cameras_set_this_iter_with_unmatched_det = set(
                            item['camera_id'] for item in unmatched_detections_all_frames[retrieve_iterations])

                        num_cameras_this_iter_with_unmatched_det = len(unique_cameras_set_this_iter_with_unmatched_det)

                        if num_cameras_this_iter_with_unmatched_det > 1:
                            Au = get_affinity_matrix_epipolar_constraint(
                                unmatched_detections_all_frames[retrieve_iterations],
                                alpha_2D,
                                calibration)

                            # Apply epipolar constraint
                            solver = GLPKSolver(min_affinity=0, max_affinity=1)
                            clusters, sol_matrix = solver.solve(Au.astype(np.double), rtn_matrix=True)

                            # Target initialization from clusters
                            for Dcluster in clusters:
                                points_2d_this_cluster = []
                                camera_id_this_cluster = []
                                image_wh_this_cluster = []
                                scores_this_cluster = []
                                foreground_masks_this_cluster = []
                                frames_this_cluster = []
                                timestamps_this_cluster = []
                                if len(Dcluster) >= 2:
                                    new_id += 1
                                    for detection_index in Dcluster:
                                        points_2d_this_cluster.append(
                                            unmatched_detections_all_frames[retrieve_iterations][detection_index][
                                                'points_2d'])
                                        camera_id_this_cluster.append(
                                            unmatched_detections_all_frames[retrieve_iterations][detection_index][
                                                'camera_id'])

                                        image_wh_this_cluster.append(
                                            unmatched_detections_all_frames[retrieve_iterations][detection_index][
                                                'image_wh'])

                                        timestamps_this_cluster.append(
                                            unmatched_detections_all_frames[retrieve_iterations][detection_index][
                                                'timestamp'])

                                        foreground_masks_this_cluster.append(
                                            unmatched_detections_all_frames[retrieve_iterations][detection_index][
                                                'foreground_mask'])

                                        scores_this_cluster.append(
                                            unmatched_detections_all_frames[retrieve_iterations][detection_index]['scores'])
                                        # Change the ID of this detection to the new id
                                        pos_poses_all_frames = \
                                        unmatched_detections_all_frames[retrieve_iterations][detection_index][
                                            'poses_2d_all_frames_pos']

                                        pos_poses = unmatched_detections_all_frames[retrieve_iterations][detection_index][
                                            'pose_pos']

                                        frames_this_cluster = unmatched_detections_all_frames[retrieve_iterations][detection_index][
                                            'frame']

                                        poses_2d_all_frames[pos_poses_all_frames]['poses'][pos_poses]['id'] = new_id

                                        # Store images related to this new id
                                        x, y, w, h = \
                                        unmatched_detections_all_frames[retrieve_iterations][detection_index]['bounding_box']
                                        top_left_point = (int(x - w / 2), int(y - h / 2))
                                        bottom_right_point = (int(x + w / 2), int(y + h / 2))
                                        confidence_bounding_box = (
                                            unmatched_detections_all_frames[retrieve_iterations][detection_index]['scores'].mean())
                                        if confidence_bounding_box > body_image_thresh:
                                            x1, x2, y1, y2 = int(x - w / 2), int(x + w / 2), int(y - h / 2), int(
                                                y + h / 2)
                                            if x1 >= 0 and y1 >= 0 and y2 < RESOLUTION[1] and x2 < RESOLUTION[0]:
                                                body_image = frame[y1:y2, x1:x2, :]
                                                body_image = cv2.pyrDown(body_image.copy())
                                                if new_id not in images_by_id:
                                                    images_by_id[new_id] = [body_image]
                                                else:
                                                    images_by_id[new_id].append(body_image)

                                    # Overwriting the unmatched detection for the current timeframe with the indices
                                    # not present in the detection cluster
                                    Tnew_t = calibration.triangulate_complete_pose(points_2d_this_cluster,
                                                                                   camera_id_this_cluster,
                                                                                   image_wh_this_cluster)
                                    Tnew_t = Tnew_t.tolist()

                                    detections = defaultdict(list)
                                    confidences = defaultdict(list)
                                    for camera_index, detection, scores in zip(camera_id_this_cluster, points_2d_this_cluster, scores_this_cluster):
                                        x_t_c_norm = detection.copy()
                                        '''
                                        x_t_c_norm[:, 0] = x_t_c_norm[:, 0] / RESOLUTION[0]
                                        x_t_c_norm[:, 1] = x_t_c_norm[:, 1] / RESOLUTION[1]'''
                                        detections[camera_index] = x_t_c_norm
                                        confidences[camera_index] = scores

                                    for idx, (score_i, score_j) in enumerate(zip(*scores_this_cluster)):
                                        # Assuming only two point sets per cluster
                                        if (score_i < UNSEEN_THRESHOLD) or (score_j < UNSEEN_THRESHOLD):
                                            Tnew_t[idx] = UNASSIGNED.tolist()
                                    # Add the 3D points according to the ID
                                    poses_3d_all_timestamps[timestamp].append({'id': new_id,
                                                                               'points_3d': Tnew_t,
                                                                               'camera_ID': camera_id_this_cluster,
                                                                               'detections': detections,
                                                                               'timestamps_2d': timestamps_this_cluster,
                                                                               'confidences': confidences})
                                    if not TEST_CALIBRATION:
                                        # Check if hands are close to shelf
                                        foot_joints = [Tnew_t[LEFT_FOOT_POS], Tnew_t[RIGHT_FOOT_POS]]
                                        conf_foot_joints = [1/2 * (scores_this_cluster[0][LEFT_FOOT_POS] +
                                                           scores_this_cluster[1][LEFT_FOOT_POS]),
                                                            1/2 * (scores_this_cluster[0][RIGHT_FOOT_POS] +
                                                           scores_this_cluster[1][RIGHT_FOOT_POS])]
                                        elbows_left = [points_2d_this_cluster[0][LEFT_ELBOW_POS],
                                                       points_2d_this_cluster[1][LEFT_ELBOW_POS]]
                                        elbows_conf_left = [scores_this_cluster[0][LEFT_ELBOW_POS],
                                                            scores_this_cluster[1][LEFT_ELBOW_POS]]
                                        elbows_right = [points_2d_this_cluster[0][RIGHT_ELBOW_POS],
                                                        points_2d_this_cluster[1][RIGHT_ELBOW_POS]]
                                        elbows_conf_right = [scores_this_cluster[0][RIGHT_ELBOW_POS],
                                                             scores_this_cluster[1][RIGHT_ELBOW_POS]]
                                        left_wrist = Tnew_t[LEFT_WRIST_POS]
                                        conf_left_wrist = 1/2 * (scores_this_cluster[0][LEFT_WRIST_POS] +
                                                           scores_this_cluster[1][LEFT_WRIST_POS])
                                        confs_left_wrist = [scores_this_cluster[0][LEFT_WRIST_POS],
                                                            scores_this_cluster[1][LEFT_WRIST_POS]]
                                        right_wrist = Tnew_t[RIGHT_WRIST_POS]
                                        conf_right_wrist = 1/2 * (scores_this_cluster[0][RIGHT_WRIST_POS] +
                                                            scores_this_cluster[1][RIGHT_WRIST_POS])
                                        confs_right_wrist = [scores_this_cluster[0][RIGHT_WRIST_POS],
                                                             scores_this_cluster[1][RIGHT_WRIST_POS]]
                                        person_id = new_id
                                        # Check shelf proximity for hands

                                        group_finished, proximity_event_group, current_events, potential_proximity_events = process_proximity_detection(
                                                                                    left_wrist,
                                                                                    elbows_left,
                                                                                    elbows_conf_left,
                                                                                    object_plane_eq,
                                                                                    left_plane_eq,
                                                                                    right_plane_eq,
                                                                                    top_plane_eq,
                                                                                    str(person_id) + "_left",
                                                                                    timestamps_this_cluster,
                                                                                    camera_id_this_cluster,
                                                                                    current_events,
                                                                                    proximity_event_group,
                                                                                    potential_proximity_events,
                                                                                    confs_left_wrist,
                                                                                    points_2d_this_cluster[0][LEFT_WRIST_POS],
                                                                                    points_2d_this_cluster[1][LEFT_WRIST_POS],
                                                                                    frames_this_cluster[0],
                                                                                    frames_this_cluster[1],
                                                                                    foreground_masks_this_cluster[0],
                                                                                    foreground_masks_this_cluster[1])

                                        if group_finished and not TEST_PROXIMITY:
                                            analyze_shoppers(embedder, shared_events_list, EventsLock,
                                                             proximity_event_group.get_events(),
                                                             proximity_event_group.get_shelf_id(),
                                                             proximity_event_group.get_minimum_timestamp(),
                                                             proximity_event_group.get_maximum_timestamp(),
                                                             events_records)
                                            proximity_event_group = None


                                        group_finished, proximity_event_group, current_events, potential_proximity_events = process_proximity_detection(
                                                                                    right_wrist,
                                                                                    elbows_right,
                                                                                    elbows_conf_right,
                                                                                    object_plane_eq,
                                                                                    left_plane_eq,
                                                                                    right_plane_eq,
                                                                                    top_plane_eq,
                                                                                    str(person_id) + "_right",
                                                                                    timestamps_this_cluster,
                                                                                    camera_id_this_cluster,
                                                                                    current_events,
                                                                                    proximity_event_group,
                                                                                    potential_proximity_events,
                                                                                    confs_right_wrist,
                                                                                    points_2d_this_cluster[0][RIGHT_WRIST_POS],
                                                                                    points_2d_this_cluster[1][RIGHT_WRIST_POS],
                                                                                    frames_this_cluster[0],
                                                                                    frames_this_cluster[1],
                                                                                    foreground_masks_this_cluster[0],
                                                                                    foreground_masks_this_cluster[1])

                                        if group_finished and not TEST_PROXIMITY:
                                            analyze_shoppers(embedder, shared_events_list, EventsLock,
                                                             proximity_event_group.get_events(),
                                                             proximity_event_group.get_shelf_id(),
                                                             proximity_event_group.get_minimum_timestamp(),
                                                             proximity_event_group.get_maximum_timestamp(),
                                                             events_records)
                                            proximity_event_group = None

                                        if np.all(left_wrist == UNASSIGNED):
                                            group_finished ,proximity_event_group, current_events, potential_proximity_events = (
                                                process_proximity_detection_by_foot_joints(foot_joints,
                                                                                           conf_foot_joints,
                                                                                           object_plane_eq,
                                                                                           left_plane_eq,
                                                                                           right_plane_eq, top_plane_eq,
                                                                                           person_id, timestamps_this_cluster,
                                                                                           camera_id_this_cluster,
                                                                                           current_events,
                                                                                           proximity_event_group,
                                                                                           potential_proximity_events,
                                                                                           points_2d_this_cluster[0][
                                                                                               LEFT_WRIST_POS],
                                                                                           points_2d_this_cluster[1][
                                                                                               LEFT_WRIST_POS],
                                                                                           frames_this_cluster[0],
                                                                                           frames_this_cluster[1], 'LEFT'))
                                            if group_finished and not TEST_PROXIMITY:
                                                analyze_shoppers(embedder, shared_events_list, EventsLock,
                                                                 proximity_event_group.get_events(),
                                                                 proximity_event_group.get_shelf_id(),
                                                                 proximity_event_group.get_minimum_timestamp(),
                                                                 proximity_event_group.get_maximum_timestamp(),
                                                                 events_records)
                                                proximity_event_group = None
                                        elif (np.all(right_wrist == UNASSIGNED)):
                                            group_finished ,proximity_event_group, current_events, potential_proximity_events = (
                                                process_proximity_detection_by_foot_joints(foot_joints,
                                                                                           conf_foot_joints,
                                                                                           object_plane_eq,
                                                                                           left_plane_eq,
                                                                                           right_plane_eq, top_plane_eq,
                                                                                           person_id, timestamps_this_cluster,
                                                                                           camera_id_this_cluster,
                                                                                           current_events,
                                                                                           proximity_event_group,
                                                                                           potential_proximity_events,
                                                                                           points_2d_this_cluster[0][
                                                                                               LEFT_WRIST_POS],
                                                                                           points_2d_this_cluster[1][
                                                                                               LEFT_WRIST_POS],
                                                                                           frames_this_cluster[0],
                                                                                           frames_this_cluster[1], 'RIGHT'))
                                            if group_finished and not TEST_PROXIMITY:
                                                analyze_shoppers(embedder, shared_events_list, EventsLock,
                                                                 proximity_event_group.get_events(),
                                                                 proximity_event_group.get_shelf_id(),
                                                                 proximity_event_group.get_minimum_timestamp(),
                                                                 proximity_event_group.get_maximum_timestamp(),
                                                                 events_records)
                                                proximity_event_group = None

                                    print("New ID created:", new_id)

                tracking_visualization_frame = cv2.cvtColor(tracking_visualization_frame, cv2.COLOR_RGB2BGR)
                if camera_id == 0:
                    filename = os.path.join(output_dir_1, "{:.3f}".format(timestamp) + '.png')
                    cv2.imwrite(filename, tracking_visualization_frame)
                else:
                    filename = os.path.join(output_dir_2, "{:.3f}".format(timestamp) + '.png')
                    cv2.imwrite(filename, tracking_visualization_frame)
            # Keep storage size 50 max, and put images of the track into
            if len(poses_3d_all_timestamps.keys()) > 70:
                first_20_keys = list(poses_3d_all_timestamps.keys())[:20]
                for key in first_20_keys:
                    del poses_3d_all_timestamps[key]
            if len(poses_2d_all_frames) > 70:
                del poses_2d_all_frames[:20:]
            retrieve_iterations += 1
            if TEST_CALIBRATION or TEST_PROXIMITY:
                if retrieve_iterations > MAX_ITERATIONS:
                    break
    except KeyboardInterrupt:
        print("Start saving")
    # Save data for later visualisation of system
    with open('interaction_data.json', mode = 'w') as out:
        json.dump(events_records, out)
    print("Done")
    # Post-processing for visualization in matplotlib

    if USE_MULTIPROCESS:
        cap.kill()
        sys.exit()
    else:
        cap.release()
        cap2.release()
