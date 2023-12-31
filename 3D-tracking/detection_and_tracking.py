import numpy as np
import cv2
from ultralytics import YOLO
import torch
from pydantic import BaseModel
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import plotly.graph_objects as go
KEYPOINTS_NUM = 17
KEYPOINTS_NAMES = ["NOSE", "LEFT_EYE", "RIGHT_EYE", "LEFT_EAR", "RIGHT_EAR",
                   "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW",
                   "LEFT_WRIST", "RIGHT_WRIST", "LEFT_HIP", "RIGHT_HIP", "LEFT_KNEE",
                   "RIGHT_KNEE", "LEFT_ANKLE", "RIGHT_ANKLE"]
# For testing sake, there's only exactly one shelf, this variable contains constant for the shelf
SHELF_DATA_TWO_CAM = np.array([[[289.20, 106.20], [422.92, 86.35], [426.12, 334.27]],
              [[21.20, 87.20], [196.48, 99.14], [198.88, 367.06]]])

SHELF_PLANE_THRESHOLD = 20

class Shelf:
    def __init__(self, data):
        self.top_left_point = data[0]
        self.top_right_point = data[1]
        self.bottom_right_point = data[2]

    def get_points(self):
        return np.array([self.top_left_point, self.top_right_point, self.bottom_right_point])


class GetKeypoint(BaseModel):
    NOSE:           int = 0
    LEFT_EYE:       int = 1
    RIGHT_EYE:      int = 2
    LEFT_EAR:       int = 3
    RIGHT_EAR:      int = 4
    LEFT_SHOULDER:  int = 5
    RIGHT_SHOULDER: int = 6
    LEFT_ELBOW:     int = 7
    RIGHT_ELBOW:    int = 8
    LEFT_WRIST:     int = 9
    RIGHT_WRIST:    int = 10
    LEFT_HIP:       int = 11
    RIGHT_HIP:      int = 12
    LEFT_KNEE:      int = 13
    RIGHT_KNEE:     int = 14
    LEFT_ANKLE:     int = 15
    RIGHT_ANKLE:    int = 16


class HumanPoseDetection():
    def __init__(self):
        self.model = self.load_model()

    def load_model(self):
        model = YOLO('yolov8l-pose.pt').to(device)
        return model

    def predict(self, image):
        results = self.model(image, verbose=False)
        return results


def cross2(a:np.ndarray,b:np.ndarray)->np.ndarray:
    return np.cross(a,b)


def visualise_3D(data):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    num_frames = len(data)
    def animate(frame_num):
        ax.clear()  # Clear the previous frame
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')
        ax.set_title(f'Frame {frame_num}')
        ax.set_xlim(-5000, 5000)
        ax.set_ylim(-5000, 5000)
        ax.set_zlim(-5000, 5000)
        # Plot 3D coordinates for the current frame
        coordinates = data[frame_num]
        ax.scatter(coordinates[:, 0], coordinates[:, 1], coordinates[:, 2], c='b', marker='o')

    # Set labels and title
    animation = FuncAnimation(fig, animate, frames=num_frames, interval=50)
    plt.show()


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
    projection_matrix_l = np.load('calib_data/projection_matrix_l.npy')
    projection_matrix_r = np.load('calib_data/projection_matrix_r.npy')
    proj_rect_l = np.load('calib_data/projRectL.npy')
    rot_rect_l = np.load('calib_data/RotRectL.npy')
    proj_rect_r = np.load('calib_data/projRectR.npy')
    rot_rect_r = np.load('calib_data/RotRectR.npy')
    # Shelf fixed coordinates in two camera views - This needs to be changed whenever cameras' position changes
    shelf_cam_1 = Shelf(SHELF_DATA_TWO_CAM[0])
    shelf_cam_2 = Shelf(SHELF_DATA_TWO_CAM[1])
    # Camera capture variables
    cap = cv2.VideoCapture(1)
    cap2 = cv2.VideoCapture(2)
    # Pose detector
    detector = HumanPoseDetection()
    # Variable used to halt recording to start visualisation after a certain number of frames
    count = 0

    while True:
        cap.grab()
        cap2.grab()
        _, img = cap.retrieve()
        _, img2 = cap2.retrieve()
        # Detect poses from two camera feed
        pose_cam_1 = detector.predict(img)[0]
        pose_cam_2 = detector.predict(img2)[0]
        # Extract skeletons that contain pose data
        cam_1_poses = pose_cam_1.keypoints.xy.cpu().numpy()
        cam_1_poses_conf = pose_cam_1.keypoints.conf.cpu().numpy()
        cam_2_poses = pose_cam_2.keypoints.xy.cpu().numpy()
        cam_2_poses_conf = pose_cam_2.keypoints.conf.cpu().numpy()
        # Test for one human
        cam_1_human = cam_1_poses[0]
        cam_1_human_conf = cam_1_poses_conf[0]
        cam_2_human = cam_2_poses[0]
        cam_2_human_conf = cam_2_poses_conf[0]
        cam_1_human_undistorted = cv2.undistortPoints(cam_1_human, camera_matrix_l, dist_l, None, None,
                                                      camera_matrix_l).reshape(-1, 2)
        cam_2_human_undistorted = cv2.undistortPoints(cam_2_human, camera_matrix_r, dist_r, None, None,
                                                      camera_matrix_r).reshape(-1, 2)
        # Triangulate keypoints
        keypoints_triangulated = cv2.triangulatePoints(projection_matrix_l, projection_matrix_r,
                                                       cam_1_human_undistorted.transpose(),
                                                       cam_2_human_undistorted.transpose()).transpose()
        # Start gathering keypoints 3D only with enough confidence
        keypoints_from_stereo = {}
        for i in range(KEYPOINTS_NUM):
            if cam_1_human_conf[i] > 0.5 and cam_2_human_conf[i] > 0.5:
                x, y, z, w = keypoints_triangulated[i]
                if w == 0:
                    result = None
                else:
                    result = [x/w, y/w, z/w]
                keypoints_from_stereo[KEYPOINTS_NAMES[i]] = result
            else:
                keypoints_from_stereo[KEYPOINTS_NAMES[i]] = None
        # Store data for visualisation
        data_for_vis = []
        data_for_vis = np.array(data_for_vis)
        your_data.append(data_for_vis)
        if count == 200:
            break
        count += 1

    cap.release()
    cap2.release()
    cv2.destroyAllWindows()
    visualise_3D(your_data)

