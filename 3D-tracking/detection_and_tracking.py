import time
from collections import defaultdict, OrderedDict
import json
import cv2
import os
import operator
import copy
import torch
from ultralytics import YOLO
from pydantic import BaseModel
from bip_solver import GLPKSolver
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.optimize import linear_sum_assignment
from reid import REID
import multiprocessing as mp
from camera import Camera, pose_matrix, normalize_intrinsic
from calibration import Calibration

from openpose.body import Body

# Config data
delta_time_threshold = 0.4
# 2D correspondence config
w_2D = 0.4  # Weight of 2D correspondence
alpha_2D = 25  # Threshold of 2D velocity
lambda_a = 5  # Penalty rate of time interval
lambda_t = 10
# 3D correspondence confif
w_3D = 0.6  # Weight of 3D correspondence
alpha_3D = 0.1  # Threshold of distance
thresh_c = 0.2  # Threshold of keypoint detection confidence
face_thresh = 0.5
similarity_threshold = 500
# Constants
MIN_NUM_FEATURES = 10
RESOLUTION = (640, 480)
KEYPOINTS_NUM = 17
KEYPOINTS_NAMES = ["NOSE", "LEFT_EYE", "RIGHT_EYE", "LEFT_EAR", "RIGHT_EAR",
                   "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW",
                   "LEFT_WRIST", "RIGHT_WRIST", "LEFT_HIP", "RIGHT_HIP", "LEFT_KNEE",
                   "RIGHT_KNEE", "LEFT_ANKLE", "RIGHT_ANKLE"]
FACE_BOX_WIDTH = 80
FACE_BOX_HEIGHT = 80
# For testing sake, there's only exactly one shelf, this variable contains constant for the shelf
SHELF_DATA_TWO_CAM = np.array([[[258.20, 101.20], [401.29, 94.46], [403.69, 477.77]],
                               [[42.20, 81.20], [210.12,  84.85], [230.11, 478.57]]])

SHELF_PLANE_THRESHOLD = 40

USE_OPENPOSE = False
OPENPOSE_NUM_KPS = 18

class Shelf:
    def __init__(self, data):
        self.top_left_point = data[0]
        self.top_right_point = data[1]
        self.bottom_right_point = data[2]

    def get_points(self):
        return np.array([self.top_left_point, self.top_right_point, self.bottom_right_point])


class GetKeypoint(BaseModel):
    NOSE: int = 0
    LEFT_EYE: int = 1
    RIGHT_EYE: int = 2
    LEFT_EAR: int = 3
    RIGHT_EAR: int = 4
    LEFT_SHOULDER: int = 5
    RIGHT_SHOULDER: int = 6
    LEFT_ELBOW: int = 7
    RIGHT_ELBOW: int = 8
    LEFT_WRIST: int = 9
    RIGHT_WRIST: int = 10
    LEFT_HIP: int = 11
    RIGHT_HIP: int = 12
    LEFT_KNEE: int = 13
    RIGHT_KNEE: int = 14
    LEFT_ANKLE: int = 15
    RIGHT_ANKLE: int = 16


class HumanPoseDetection():
    def __init__(self):
        self.model = self.load_model()

    def load_model(self):
        model = YOLO('weights/yolov8l-pose.pt').to(device)
        return model

    def predict(self, image):
        results = self.model(image, verbose=False)
        return results


UNASSIGNED = np.array([0, 0, 0])


def cross2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.cross(a, b)


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
        if count > 20:
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
                                                    delta_time_threshold=0.2):
    # Iterate through poses_3d_all_timestamps from the current timestamp to get the latest points 3D for IDs in
    # the window of the delta_time_threshold> Note that time window from the current timestamp and not from the 
    # timestamp when points 3d were estimated

    # [[{'id': calculated, 'points_3d': list of target joints, 'timestamp': , 'velocity': }], [{}], ]
    poses_3D_latest = []
    id_list = []

    for index in range(len(poses_3d_all_timestamps) - 1, 0, -1):
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
                                            'detections': poses_3d_all_timestamps[this_timestamp][id_index]['detections'],
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


def extract_key_value_pairs_from_poses_2d_list(data, id, timestamp_cur_frame, dt_thresh=0.1):
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
    Au_this_pair = 1 - (
                (calibration.calc_epipolar_error([cam_L_id, cam_R_id], D_L, scores_l, D_R, scores_r)) / (8 * alpha_2D))
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


def check_hand_near_shelf(wrists, object_plane_eq, left_plane_eq, right_plane_eq):
    for wrist in wrists:
        dist_from_shelf_plane = distance_to_plane(wrist, object_plane_eq)
        if ((dist_from_shelf_plane < SHELF_PLANE_THRESHOLD) and
                (is_point_between_planes(left_plane_eq, right_plane_eq, wrist))):
            return True
    return False

# This function only draw the ID of a person on the image when the person is tracked not when initialized
def draw_id(data, image):
    # Variables storing text settings for drawing on images
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 2
    font_color = (255, 0, 0)
    line_type = cv2.LINE_AA
    id = str(data['id'])

    for i, joint in enumerate(data['points_2d']):
        if data['conf'][i] > 0.3:
            image = cv2.putText(image, id, (int(joint[0]), int(joint[1])), font, font_scale,
                                font_color, font_thickness, line_type, False)
    return image


# Function to generate bounding box around a given point
def generate_bounding_box(x, y):
    return x - FACE_BOX_WIDTH // 2, y - FACE_BOX_HEIGHT // 2, x + FACE_BOX_WIDTH // 2, y + FACE_BOX_HEIGHT // 2


# Special function used in mutliprocessing to extract features
def extract_features(feats, q, f_lock) -> None:
    from reid import REID
    reid = REID()
    l_dict = dict()
    while True:
        #Does this mean that always the latest image of an object will be the embedding of it ? Would it cause any cons ?
        if not q.empty():
            idx, cnt, img = q.get()
            if idx in l_dict.keys():
                if l_dict[idx][0] < cnt:
                    l_dict[idx] = [cnt, img]
                else:
                    continue
            else:
                l_dict[idx] = [cnt, img]
            f = reid.features(l_dict[idx][1])
            f_lock.acquire()
            feats[idx] = f
            f_lock.release()


if __name__ == "__main__":
    # This contains data for visualisation
    your_data = []
    # This is use to get keypoint by name
    get_keypoint = GetKeypoint()
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
    proj_rect_l = np.load('calib_data/projRectL.npy')
    rot_rect_l = np.load('calib_data/RotRectL.npy')
    proj_rect_r = np.load('calib_data/projRectR.npy')
    rot_rect_r = np.load('calib_data/RotRectR.npy')
    # Normalized intrinsic matrices
    normalized_matrix_l = normalize_intrinsic(camera_matrix_l, RESOLUTION[0], RESOLUTION[1])
    normalized_matrix_r = normalize_intrinsic(camera_matrix_r, RESOLUTION[0], RESOLUTION[1])
    # Calibration object
    calibration = Calibration(cameras={
        0: Camera(normalized_matrix_l, pose_matrix(rotm_l, tvec_l.flatten()), dist_l[0]),
        1: Camera(normalized_matrix_r, pose_matrix(rotm_r, tvec_r.flatten()), dist_r[0])
    })
    # Shelf fixed coordinates in two camera views - This needs to be changed whenever cameras' position changes
    shelf_cam_1 = Shelf(SHELF_DATA_TWO_CAM[0])
    shelf_cam_2 = Shelf(SHELF_DATA_TWO_CAM[1])
    # 3D location of shelf points
    shelf_points_3d = calibration.triangulate_complete_pose(np.array([shelf_cam_1.get_points(), shelf_cam_2.get_points()]), [0, 1], [[640,480], [640,480]])
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
    # Camera capture variables
    cap = cv2.VideoCapture(1)
    cap2 = cv2.VideoCapture(2)
    # Pose detector
    if USE_OPENPOSE:
        KEYPOINTS_NUM = OPENPOSE_NUM_KPS
        detector = Body('weights/body_pose_model.pth')
    else:
        detector = HumanPoseDetection()
    # Variable used to halt recording to start visualisation after a certain number of frames
    count = 0
    # Variables for storing visualization data
    output_dir_1 = "frames_data_cam_1"
    output_dir_2 = "frames_data_cam_2"
    cam_1_frames = {}
    cam_2_frames = {}
    # Data for poses along the timeline
    poses_2d_all_frames = []
    poses_3d_all_timestamps = defaultdict(list)
    unmatched_detections_all_frames = defaultdict(list)
    # World ltrb
    world_ltrb = calibration.compute_world_ltrb()
    # Timer
    camera_start = time.time()
    # Iteration variables and ID variable for assigning new ID
    retrieve_iterations = -1
    new_id = -1
    iterations = 0
    new_id_last_update_timestamp = 0
    # Variables storing face images for re-identification, if an id's image hasn't been available for a while, delete
    images_by_id = dict()
    # Variables storing shared data for re-identification subprocess
    FeatsLock = mp.Lock()
    shared_feats_dict = mp.Manager().dict()
    shared_images_queue = mp.Queue()
    # Subprocess running to generate embeddings for re-identification.
    extract_p = mp.Process(target=extract_features, args=(shared_feats_dict, shared_images_queue, FeatsLock,))
    extract_p.start()
    # REID object
    reid = REID()
    # Storing the invalid IDs that already got fix
    invalid_ids = []
    # Storing the ids' creation timestamp to prevent an old ID being reid to a new ID -> Loop
    id_timestamps = {}
    while True:
        FeatsLock.acquire()
        local_feats_dict = {}
        for key, value in shared_feats_dict.items():
            if key not in invalid_ids:
                local_feats_dict[key] = copy.deepcopy(value)
        FeatsLock.release()
        camera_data = []

        cap.grab()
        cap2.grab()
        timestamp_common = round(time.time() - camera_start, 2)
        _, img = cap.retrieve()
        camera_data.append([img, timestamp_common])
        _, img2 = cap2.retrieve()
        camera_data.append([img2, timestamp_common])
        cams_frames = [cam_1_frames, cam_2_frames]
        retrieve_iterations += 1
        # Start checking if an ID has been corrected
        # This way probably goes very wrong in the case of ID being swapped around and only meant to handle the case
        # where a new id is given to a person due to entering the scene back or being occluded
        for track_id_1 in local_feats_dict.keys():
            if local_feats_dict[track_id_1].shape[1] < MIN_NUM_FEATURES:
                continue
            similarity_scores = []
            for track_id_2 in local_feats_dict.keys():
                if track_id_2 == track_id_1 or local_feats_dict[track_id_2].shape[1] < MIN_NUM_FEATURES or id_timestamps[track_id_2] > id_timestamps[track_id_1]:
                    continue
                # Start checking if the ID belong to a different track by appearance, if yes, then change the id of both
                # poses_2d_all_frames and poses_3d_all_times, and delete the images related to this "wrong id"
                similarity_scores = []
                tmp = np.mean(reid.compute_distance(local_feats_dict[track_id_1], local_feats_dict[track_id_2]))
                similarity_scores.append([track_id_2, tmp])
            if similarity_scores:
                similarity_scores.sort(key=operator.itemgetter(1))
                for index in range(len(similarity_scores)):
                    if similarity_scores[index][1] < similarity_threshold:
                        print("Reid succesfulyyyyyyyy!!!!!!!!! from ID " + str(track_id_1) + " to ID " + str(similarity_scores[index][0]))
                        # This track id becomes invalid so that if the subprocess give embeddings for this id deny it
                        # and also delete images related to this id
                        real_id = similarity_scores[index][0]
                        invalid_ids.append(track_id_1)
                        del images_by_id[track_id_1]
                        # Change the wrong id to the real id in recent data of poses_2d_all_frames
                        for index_1 in range(len(poses_2d_all_frames) - 1, 0, -1):
                            this_timestamp = poses_2d_all_frames[index_1]['timestamp']
                            if (timestamp_common - this_timestamp) > delta_time_threshold:
                                break
                            for pose_index in range(len(poses_2d_all_frames[index_1]['poses'])):
                                if poses_2d_all_frames[index_1]['poses'][pose_index]['id'] == track_id_1:
                                    poses_2d_all_frames[index_1]['poses'][pose_index]['id'] = real_id
                                    break
                        # Change the wrong id to the real id in recent data of poses_3d_all_timestamps
                        for index_2 in range(len(poses_3d_all_timestamps) - 1, 0, -1):
                            this_timestamp = list(poses_3d_all_timestamps.keys())[index_2]
                            # time window ends return the ID
                            if (timestamp_common - this_timestamp) > delta_time_threshold:
                                break
                            # to get 3d pose at timestamp before the timestamp at the current frame
                            if this_timestamp >= timestamp_common or all(
                                    value is None for value in poses_3d_all_timestamps[this_timestamp]):
                                continue
                            for id_index in range(len(poses_3d_all_timestamps[this_timestamp])):
                                if poses_3d_all_timestamps[this_timestamp][id_index]['id'] == track_id_1:
                                    poses_3d_all_timestamps[this_timestamp][id_index]['id'] = real_id
                                    break
                        break
                    else:
                        print("Not similar enough:", similarity_scores[index][1])
        for camera_id, data in enumerate(camera_data):
            # List containing tracks and detections after Hungarian Algorithm
            indices_T = []
            indices_D = []
            frame, timestamp = data  # Get the frame (image) and timestamp for this camera_id

            # Get pose estimation data for this frame
            if USE_OPENPOSE:
                candidates, subsets = detector(frame)
                poses_keypoints = []
                poses_conf = []

                for n in range(len(subsets)):
                    poses_keypoints.append([])
                    poses_conf.append([])

                    for kp in range(OPENPOSE_NUM_KPS):
                        index = int(subsets[n][kp])

                        if index == -1: # KP not included
                            x, y, conf = 0, 0, 0
                        else:
                            x, y, conf = candidates[index][:3]

                        poses_keypoints[n].append([x, y])
                        poses_conf[n].append(conf)
                poses_keypoints = np.array(poses_keypoints)
                poses_conf = np.array(poses_conf)
            else:
                poses_data_cur_frame = detector.predict(frame)[0]
                try:
                    poses_keypoints = poses_data_cur_frame.keypoints.xy.cpu().numpy()
                    poses_conf = poses_data_cur_frame.keypoints.conf.cpu().numpy()
                except Exception:
                    continue
            poses_small = []
            poses_conf_small = []
            points_2d_cur_frames = []
            points_2d_scores_cur_frames = []

            if len(poses_keypoints) == 0:
                iterations += 1
                poses_3d_all_timestamps[timestamp].append(None)
                continue

            for poses_index in range(len(poses_keypoints)):
                '''
                poses_main = [poses_keypoints[poses_index][i] for i in range(len(poses_keypoints[poses_index])) if
                              i in [0, 7, 8, 9, 10, 13, 14, 15, 16]]
                conf_main = [poses_conf[poses_index][i] for i in range(len(poses_conf[poses_index])) if
                             i in [0, 7, 8, 9, 10, 13, 14, 15, 16]]'''
                poses_small.append(poses_keypoints[poses_index])
                poses_conf_small.append(poses_conf[poses_index])

            poses_small = np.array(poses_small)
            poses_conf_small = np.array(poses_conf_small)
            for poses_index in range(len(poses_small)):
                points_2d_cur_frames.append(poses_small[poses_index])
                points_2d_scores_cur_frames.append(poses_conf_small[poses_index])

            location_of_camera_center_cur_frame = calibration.cameras[camera_id].location
            poses_2d_all_frames.append({
                'camera': camera_id,
                'timestamp': timestamp,
                'poses': [{'id': -1, 'points_2d': poses_small[index], 'conf': poses_conf_small[index]} for index in range(len(poses_small))],
            })
            poses_3D_latest = get_latest_3D_poses_available_for_cur_timestamp(poses_3d_all_timestamps, timestamp,
                                                                              delta_time_threshold=delta_time_threshold)
            N_3d_poses_last_timestamp = len(poses_3D_latest)
            M_2d_poses_this_camera_frame = len(points_2d_cur_frames)
            Dt_c = np.array(points_2d_cur_frames)  # Shape (M poses on frame , no of body points , 2)
            Dt_c_scores = np.array(points_2d_scores_cur_frames)
            # Affinity matrix associating N current tracks and M detections
            A = np.zeros(
                (N_3d_poses_last_timestamp, M_2d_poses_this_camera_frame))  # Cross-view association matrix shape N x M
            for i in range(N_3d_poses_last_timestamp):  # Iterate through prev N Target poses
                if camera_id in poses_3D_latest[i]['detections']:
                    x_t_tilde_tilde_c = poses_3D_latest[i]['detections'][camera_id]
                else:
                    x_t_tilde_tilde_c = calibration.project(np.array(poses_3D_latest[i]['points_3d']), camera_id)
                delta_t = timestamp - poses_3D_latest[i]['timestamp']
                for j in range(M_2d_poses_this_camera_frame):  # Iterate through M poses
                    # Each detection (Dj_tme will have k body points for every camera c
                    # x_t_c in image coordinate_c) in this fras
                    # x_t_c_norm scale normalized image coordinates
                    x_t_c_norm = Dt_c[j].copy()
                    x_t_c_norm[:, 0] = x_t_c_norm[:, 0] / RESOLUTION[0]
                    x_t_c_norm[:, 1] = x_t_c_norm[:, 1] / RESOLUTION[1]
                    K_joints_detected_this_person = len(x_t_c_norm)
                    # Need to implement back_project
                    back_proj_x_t_c_to_ground = calibration.cameras[camera_id].back_project(x_t_c_norm,
                                                                                            z_worlds=np.zeros(
                                                                                                K_joints_detected_this_person))

                    for k in range(K_joints_detected_this_person):  # Iterate through K keypoints
                        target_joint = poses_3D_latest[i]['points_3d'][k]
                        if np.all(target_joint == UNASSIGNED):
                            continue
                        # Calculating A2D between target's last updated joint K from the current camera
                        distance_2D = np.linalg.norm(x_t_c_norm[k] - x_t_tilde_tilde_c[k])  # Distance between joints
                        A_2D = Dt_c_scores[j][k] * w_2D * (1 - distance_2D / (alpha_2D * delta_t)) * np.exp(
                            -lambda_a * delta_t)
                        # Calculating A3D between predicted position in 3D space of the target's joint and the detection's joint projected into 3D
                        velocity_t_tilde = np.array(poses_3D_latest[i]['velocity'][k])
                        predicted_X_t = np.array(target_joint) + (velocity_t_tilde * delta_t)
                        dl = calculate_perpendicular_distance(point=predicted_X_t,
                                                              line_start=location_of_camera_center_cur_frame,
                                                              line_end=back_proj_x_t_c_to_ground[k])
                        A_3D = Dt_c_scores[j][k] * w_3D * (1 - dl / alpha_3D) * np.exp(-lambda_a * delta_t)
                        # Add the affinity between the pair of target and detection in terms of this specific joint
                        A[i, j] += A_2D + A_3D

            # Hungarian algorithm able to assign detections to tracks based on Affinity matrix
            indices_T, indices_D = linear_sum_assignment(A, maximize=True)
            for i, j in zip(indices_T, indices_D):
                track_id = poses_3D_latest[i]['id']
                poses_2d_all_frames[-1]['poses'][j]['id'] = track_id
                # Store images related to this track
                if Dt_c_scores[j][0] > face_thresh:
                    nose_x, nose_y = Dt_c[j][0][0], Dt_c[j][0][1]
                    x1, y1, x2, y2 = generate_bounding_box(nose_x, nose_y)
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    if x1 >= 0 and x2 >= 0 and y2 < RESOLUTION[1] and x2 < RESOLUTION[0]:
                        face_image = frame[y1:y2, x1:x2, :]
                        if track_id not in images_by_id:
                            images_by_id[track_id] = [face_image]
                        else:
                            images_by_id[track_id].append(face_image)
                # Store frames for visualizing tracking in 2D
                new_frame = draw_id(poses_2d_all_frames[-1]['poses'][j], frame)
                cam_frames = cams_frames[camera_id]
                cam_frames[iterations] = {'filename': str(timestamp) + '.png', 'image': new_frame}
                # Extract poses data from other camera
                poses_2d_inc_rec_other_cam = extract_key_value_pairs_from_poses_2d_list(poses_2d_all_frames,
                                                                                        id=track_id,
                                                                                        timestamp_cur_frame=timestamp,
                                                                                        dt_thresh=delta_time_threshold)

                # move following code in func extract_key_value_pairs_from_poses_2d_list to get *_inc_rec variables directly
                # Get 2D poses of ID 
                dict_with_poses_for_n_cameras_for_latest_timeframe = separate_lists_for_incremental_triangulation(
                    poses_2d_inc_rec_other_cam)
                camera_ids_inc_rec = []

                image_wh_inc_rec = []

                timestamps_inc_rec = []

                points_2d_inc_rec = []

                conf_2d_inc_rec = []

                camera_ids_inc_rec = dict_with_poses_for_n_cameras_for_latest_timeframe['camera']
                image_wh_inc_rec = dict_with_poses_for_n_cameras_for_latest_timeframe['image_wh']
                timestamps_inc_rec = dict_with_poses_for_n_cameras_for_latest_timeframe['timestamp']

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
                    if conf_2d_inc_rec[0][k] > thresh_c and conf_2d_inc_rec[1][k] > thresh_c:
                        _, Ti_k_t = calibration.linear_ls_triangulate_weighted(np.array(points_2d_inc_rec)[:, k, :],
                                                                               camera_ids_inc_rec,
                                                                               image_wh_inc_rec,
                                                                               lambda_t,
                                                                               timestamps_inc_rec)
                        Ti_t.append(Ti_k_t.tolist())
                    else:
                        delta_t = timestamp - poses_3D_latest[i]['timestamp']
                        target_joint = poses_3D_latest[i]['points_3d'][k]

                        if not np.all(target_joint == UNASSIGNED):
                            velocity_t_tilde = np.array(poses_3D_latest[i]['velocity'][k])
                            Ti_t.append((np.array(target_joint) + (velocity_t_tilde * delta_t)).tolist())
                        else:
                            Ti_t.append(UNASSIGNED.tolist())
                # Detection normalized
                x_t_c_norm = Dt_c[j].copy()
                x_t_c_norm[:, 0] = x_t_c_norm[:, 0] / RESOLUTION[0]
                x_t_c_norm[:, 1] = x_t_c_norm[:, 1] / RESOLUTION[1]
                if i >= len(poses_3d_all_timestamps[timestamp]):
                    poses_3d_all_timestamps[timestamp].append({'id': poses_3D_latest[i]['id'],
                                                               'points_3d': Ti_t,
                                                               'camera_ID': [camera_id],
                                                               'detections': {
                                                                    camera_id: x_t_c_norm
                                                               }
                                                               })

                # If there exist an entry already overwrite as this would be contain updated timestamps
                # from all cameras for points 3D. 
                else:
                    poses_3d_all_timestamps[timestamp][i]['points_3d'] = Ti_t
                    poses_3d_all_timestamps[timestamp][i]['camera_ID'].append(camera_id)
                    poses_3d_all_timestamps[timestamp][i]['detections'][camera_id] = x_t_c_norm
                # Checking shelf proximity
                wrists = [Ti_t[3], Ti_t[4]]
                if check_hand_near_shelf(wrists, object_plane_eq, left_plane_eq, right_plane_eq):
                    #print("Person with ID " + str(poses_3D_latest[i]['id']) + " approaches the shelf!!")
                    pass

            for j in range(M_2d_poses_this_camera_frame):
                if j not in indices_D:
                    unmatched_detections_all_frames[retrieve_iterations].append({'camera_id': camera_id,
                                                                                 'points_2d': Dt_c[j],
                                                                                 'scores': Dt_c_scores[j],
                                                                                 'image_wh': [RESOLUTION[0], RESOLUTION[1]],
                                                                                 'poses_2d_all_frames_pos': len(poses_2d_all_frames) - 1,
                                                                                 'pose_pos': j})

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

                            if len(Dcluster) >= 2:
                                # logging.info(f'Inside cluster: {Dcluster} ')

                                # TODO: Adhoc Solution. Change in the future
                                # If there a new person detected within delta time threshold then probably
                                # this new person is belongs to the older id
                                if timestamp - new_id_last_update_timestamp > 0.2:  # This will be timestamp in the last camera
                                    new_id_last_update_timestamp = timestamp
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

                                    scores_this_cluster.append(
                                        unmatched_detections_all_frames[retrieve_iterations][detection_index]['scores'])
                                    # Change the ID of this detection to the new id
                                    pos_poses_all_frames = unmatched_detections_all_frames[retrieve_iterations][detection_index]['poses_2d_all_frames_pos']

                                    pos_poses = unmatched_detections_all_frames[retrieve_iterations][detection_index]['pose_pos']

                                    poses_2d_all_frames[pos_poses_all_frames]['poses'][pos_poses]['id'] = new_id

                                    # Store images related to this new id
                                    nose_score = unmatched_detections_all_frames[retrieve_iterations][detection_index]['scores'][0]
                                    nose_joint = unmatched_detections_all_frames[retrieve_iterations][detection_index]['points_2d'][0]
                                    if nose_score > face_thresh:
                                        nose_x, nose_y = nose_joint[0], nose_joint[1]
                                        x1, y1, x2, y2 = generate_bounding_box(nose_x, nose_y)
                                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                                        if x1 >= 0 and x2 >= 0 and y2 < RESOLUTION[1] and x2 < RESOLUTION[0]:
                                            face_image = frame[y1:y2, x1:x2, :]
                                            if new_id not in images_by_id:
                                                images_by_id[new_id] = [face_image]
                                            else:
                                                images_by_id[new_id].append(face_image)

                                # Overwriting the unmatched detection for the current timeframe with the indices
                                # not present in the detection cluster
                                Tnew_t = calibration.triangulate_complete_pose(points_2d_this_cluster,
                                                                               camera_id_this_cluster,
                                                                               image_wh_this_cluster)
                                Tnew_t = Tnew_t.tolist()

                                detections = defaultdict(list)

                                for camera_index, detection in zip(camera_id_this_cluster, points_2d_this_cluster):
                                    x_t_c_norm = detection.copy()
                                    x_t_c_norm[:, 0] = x_t_c_norm[:, 0] / RESOLUTION[0]
                                    x_t_c_norm[:, 1] = x_t_c_norm[:, 1] / RESOLUTION[1]
                                    detections[camera_index] = x_t_c_norm

                                for idx, (score_i, score_j) in enumerate(zip(*scores_this_cluster)):
                                    # Assuming only two point sets per cluster
                                    if (score_i < thresh_c) or (score_j < thresh_c):
                                        Tnew_t[idx] = UNASSIGNED.tolist()
                                # Add the 3D points according to the ID 
                                poses_3d_all_timestamps[timestamp].append({'id': new_id,
                                                                           'points_3d': Tnew_t,
                                                                           'camera_ID': camera_id_this_cluster,
                                                                           'detections': detections})
                                # Store id creation time
                                id_timestamps[new_id] = timestamp_common
                                # Check if hands are close to shelf
                                wrists = [Tnew_t[3], Tnew_t[4]]
                                if check_hand_near_shelf(wrists, object_plane_eq, left_plane_eq, right_plane_eq):
                                    #print("Newly created person with ID " + str(new_id) + " approaches the shelf!!")
                                    pass
                                print("New ID created:", new_id)
        # Keep storage size 50 max, and put images of the track into
        if len(poses_3d_all_timestamps.keys()) > 70:
            first_20_keys = list(poses_3d_all_timestamps.keys())[:20]
            for key in first_20_keys:
                del poses_3d_all_timestamps[key]
        if len(poses_2d_all_frames) > 70:
            del poses_2d_all_frames[:20:]
        for i in images_by_id:
            if len(images_by_id[i]) > 70:
                del images_by_id[i][:20:]
            shared_images_queue.put([i, iterations, images_by_id[i]])

    cap.release()
    cap2.release()
    cv2.destroyAllWindows()
    # Terminate the subprocess
    extract_p.terminate()
    extract_p.join()
    shared_images_queue.close()
    # Post processing for visualization in 2D images for tracking
    for i, cam_frames in enumerate(cams_frames):
        for key in cam_frames.keys():
            data = cam_frames[key]
            if i == 0:
                filename = os.path.join(output_dir_1, data['filename'])
                cv2.imwrite(filename, data['image'])
            else:
                filename = os.path.join(output_dir_2, data['filename'])
                cv2.imwrite(filename, data['image'])
    # Post-processing for visualization in matplotlib
    poses_1 = {}
    poses_2 = {}
    poses_3 = {}
    for key in poses_3d_all_timestamps.keys():
        for data in poses_3d_all_timestamps[key]:
            if data["id"] == 1:
                poses_1[key] = [data]
            elif data["id"] == 2:
                poses_2[key] = [data]
            elif data["id"] == 3:
                poses_3[key] = [data]
    for key in poses_3d_all_timestamps.keys():
        for index_j in range(len(poses_3d_all_timestamps[key])):
            poses_3d_all_timestamps[key][index_j]['detections'] = []

    converted_dict = dict(poses_3d_all_timestamps)
    with open("poses_3d.json", "w") as f:
        json.dump(converted_dict, f)

    with open("poses_3d1.json", "w") as f:
        json.dump(poses_1, f)

    with open("poses_3d2.json", "w") as f:
        json.dump(poses_2, f)

    with open("poses_3d3.json", "w") as f:
        json.dump(poses_3, f)

