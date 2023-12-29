import numpy as np
import cv2
from ultralytics import YOLO
import torch
from pydantic import BaseModel
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

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


def visualise_3D(your_data, x_vis_shelf, y_vis_shelf, z_vis_shelf, x2_vis, y2_vis, z2_vis, x4_vis, y4_vis, z4_vis):
    fig = plt.figure()
    ax = fig.add_subplot(211, projection='3d')
    ax2 = fig.add_subplot(212, projection='3d')
    ax2.plot_surface(x_vis_shelf, y_vis_shelf, z_vis_shelf, alpha=0.5)
    ax2.plot_surface(x2_vis, y2_vis, z2_vis, alpha=0.5)
    ax2.plot_surface(x4_vis, y4_vis, z4_vis, alpha=0.5)
    num_frames = len(your_data)
    def animate(frame_num):
        ax.clear()  # Clear the previous frame
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')
        ax.set_title(f'Frame {frame_num}')
        # Plot 3D coordinates for the current frame
        coordinates = your_data[frame_num]
        ax.scatter(coordinates[:, 0], coordinates[:, 1], coordinates[:, 2], c='b', marker='o')

    # Set labels and title
    animation = FuncAnimation(fig, animate, frames=num_frames, interval=50)
    plt.show()

def get_plane_equation_shelf_points(data):
    P = data[0]
    Q = data[1]
    R = data[2]
    x1, y1, z1, w1 = P
    x1 = x1/w1
    y1 = y1/w1
    z1 = z1/w1
    x2, y2, z2, w2 = Q
    x2 = x2 / w2
    y2 = y2 / w2
    z2 = z2 / w2
    x3, y3, z3, w3 = R
    x3 = x3 / w3
    y3 = y3 / w3
    z3 = z3 / w3
    a1 = x2 - x1
    b1 = y2 - y1
    c1 = z2 - z1
    a2 = x3 - x1
    b2 = y3 - y1
    c2 = z3 - z1
    a = b1 * c2 - b2 * c1
    b = a2 * c1 - a1 * c2
    c = a1 * b2 - b1 * a2
    d = (- a * x1 - b * y1 - c * z1)
    return a, b, c, d


# Type "one" clipping plane is a plane passing through two shelf corners and perpendicular to the shelf plane
def get_clipping_plane_type_one(P1, P2, N):
    x1, y1, z1, w1 = P1
    P1 = np.array([x1 / w1, y1 / w1, z1 / w1])
    x2, y2, z2, w2 = P2
    P2 = np.array([x2 / w2, y2 / w2, z2 / w2])
    V = P2 - P1
    N_new = np.cross(V, N)
    P_ref = P1
    a = N_new[0]
    b = N_new[1]
    c = N_new[2]
    d = -(a * P_ref[0] + b * P_ref[1] + c * P_ref[2])
    return a, b, c, d


# Clipping plane type two is a plane that is perpendicular to a type one plane, a shelf plane, and pass through a corner
def get_clipping_plane_type_two(N1, N2, Point):
    x1, y1, z1, w1 = Point
    P_ref = np.array([x1 / w1, y1 / w1, z1 / w1])
    normal = np.cross(N1, N2)
    a = normal[0]
    b = normal[1]
    c = normal[2]
    d = -(a * P_ref[0] + b * P_ref[1] + c * P_ref[2])
    return a, b, c, d

def distance_to_plane(point, plane_equation):
    a, b, c, d = plane_equation
    X, Y, Z = point
    distance = (a * X + b * Y + c * Z + d) / np.sqrt(a**2 + b**2 + c**2)
    return abs(distance)


def plane_grid(normal, d):
    x, y = np.meshgrid(np.arange(-5,5,0.25), np.arange(-5,5,0.25))
    z = (-normal[0] * x - normal[1] * y - d) * 1. / normal[2]  # Solve for z using the plane equation
    return x, y, z


def is_point_between_planes(plane1_eq, plane2_eq, point):
    a1, b1, c1, d1 = plane1_eq
    a2, b2, c2, d2 = plane2_eq
    normal1 = np.array([a1,b1,c1])
    normal2 = np.array([a2, b2, c2])
    point_on_plane1 = np.array([-d1/a1, 0, 0])
    point_on_plane2 = np.array([-d2/a2, 0 ,0])
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
    # Undistorted shelf coordinates
    shelf_cam1_points = cv2.undistortPoints(shelf_cam_1.get_points(), camera_matrix_l, dist_l, None, None,
                                            camera_matrix_l).reshape(-1, 2)
    shelf_cam2_points = cv2.undistortPoints(shelf_cam_2.get_points(), camera_matrix_r, dist_r, None, None,
                                            camera_matrix_r).reshape(-1, 2)
    # 3D location of shelf points
    shelf_points_3d = cv2.triangulatePoints(projection_matrix_l, projection_matrix_r,
                                            shelf_cam1_points.transpose(),
                                            shelf_cam2_points.transpose()).transpose()
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
    x2_vis, y2_vis, z2_vis = plane_grid(right_clipping_plane_normal, d_right)

    a_top, b_top, c_top, d_top = get_clipping_plane_type_one(
        shelf_points_3d[0], shelf_points_3d[1], object_plane_normal)
    top_clipping_plane_normal = np.array([a_top, b_top, c_top])
    top_plane_eq = np.array([a_top, b_top, c_top, d_top])

    a_left, b_left, c_left, d_left = get_clipping_plane_type_two(object_plane_normal,
                                                                 top_clipping_plane_normal, shelf_points_3d[0])
    left_clipping_plane_normal = np.array([a_left, b_left, c_left])
    left_plane_eq = np.array([a_left, b_left, c_left, d_left])
    x4_vis, y4_vis, z4_vis = plane_grid(left_clipping_plane_normal, d_left)

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
        for k in keypoints_from_stereo.keys():
            if keypoints_from_stereo[k] is not None:
                data_for_vis.append(keypoints_from_stereo[k])
                if k == "LEFT_WRIST" or k == "RIGHT_WRIST":
                    wrist = keypoints_from_stereo[k]
                    dist_from_shelf_plane = distance_to_plane(wrist, object_plane_eq)
                    if (dist_from_shelf_plane < SHELF_PLANE_THRESHOLD and
                            is_point_between_planes(left_plane_eq, right_plane_eq, wrist)):
                        print("Hand close to shelf, distance: ", dist_from_shelf_plane)


        data_for_vis = np.array(data_for_vis)
        your_data.append(data_for_vis)
        if count == 200:
            break
        count += 1

    cap.release()
    cap2.release()
    cv2.destroyAllWindows()
    visualise_3D(your_data, x_vis, y_vis, z_vis, x2_vis, y2_vis, z2_vis, x4_vis, y4_vis, z4_vis)

