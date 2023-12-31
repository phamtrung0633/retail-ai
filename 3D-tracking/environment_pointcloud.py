import numpy as np
import cv2

def get_keypoints_and_descriptors(imgL, imgR):
    """Use ORB detector and FLANN matcher to get keypoints, descritpors,
    and corresponding matches that will be good for computing
    homography.
    """
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(imgL, None)
    kp2, des2 = orb.detectAndCompute(imgR, None)

    ############## Using FLANN matcher ##############
    # Each keypoint of the first image is matched with a number of
    # keypoints from the second image. k=2 means keep the 2 best matches
    # for each keypoint (best matches = the ones with the smallest
    # distance measurement).
    FLANN_INDEX_LSH = 6
    index_params = dict(
        algorithm=FLANN_INDEX_LSH,
        table_number=6,  # 12
        key_size=12,  # 20
        multi_probe_level=1,
    )  # 2
    search_params = dict(checks=50)  # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    flann_match_pairs = flann.knnMatch(des1, des2, k=2)
    return kp1, des1, kp2, des2, flann_match_pairs


def lowes_ratio_test(matches, ratio_threshold=0.6):
    """Filter matches using the Lowe's ratio test.

    The ratio test checks if matches are ambiguous and should be
    removed by checking that the two distances are sufficiently
    different. If they are not, then the match at that keypoint is
    ignored.

    https://stackoverflow.com/questions/51197091/how-does-the-lowes-ratio-test-work
    """
    filtered_matches = []
    for match in matches:
        if len(match) == 2:
            m = match[0]
            n = match[1]
            if m.distance < ratio_threshold * n.distance:
                filtered_matches.append(m)
    return filtered_matches


def draw_matches(imgL, imgR, kp1, des1, kp2, des2, flann_match_pairs):
    """Draw the first 8 mathces between the left and right images."""
    # https://docs.opencv.org/4.2.0/d4/d5d/group__features2d__draw.html
    # https://docs.opencv.org/2.4/modules/features2d/doc/common_interfaces_of_descriptor_matchers.html
    img = cv2.drawMatches(
        imgL,
        kp1,
        imgR,
        kp2,
        flann_match_pairs[:8],
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    cv2.imshow("Matches", img)
    cv2.imwrite("ORB_FLANN_Matches.png", img)
    cv2.waitKey(0)


def compute_fundamental_matrix(matches, kp1, kp2, method=cv2.FM_RANSAC):
    """Use the set of good mathces to estimate the Fundamental Matrix.

    See  https://en.wikipedia.org/wiki/Eight-point_algorithm#The_normalized_eight-point_algorithm
    for more info.
    """
    pts1, pts2 = [], []
    fundamental_matrix, inliers = None, None
    for element in matches:
        if len(element) == 2:
            m = element[0]
            n = element[1]
            if m.distance < 0.7 * n.distance:
                pts1.append(kp1[m.queryIdx].pt)
                pts2.append(kp2[m.trainIdx].pt)
    if pts1 and pts2:
        # You can play with the Threshold and confidence values here
        # until you get something that gives you reasonable results. I
        # used the defaults
        fundamental_matrix, inliers = cv2.findFundamentalMat(
            np.float32(pts1),
            np.float32(pts2),
            method=method,
            ransacReprojThreshold=3,
            # confidence=0.99,
        )
    return fundamental_matrix, inliers, pts1, pts2


image_left = cv2.imread('images/environmentLeft/imageL.png')
image_right = cv2.imread('images/environmentRight/imageR.png')
# Undistort images
camera_matrix_l = np.load('calib_data/camera_matrix_l.npy')
camera_matrix_r = np.load('calib_data/camera_matrix_r.npy')
dist_l = np.load('calib_data/dist_l.npy')
dist_r = np.load('calib_data/dist_r.npy')
fundamental_mat = np.load('calib_data/fundamental_matrix.npy')
image_left = cv2.undistort(image_left, camera_matrix_l, dist_l, None, camera_matrix_l)
image_right = cv2.undistort(image_right, camera_matrix_r, dist_r, None, camera_matrix_r)
cv2.imshow("Undistorted image left:", image_left)
cv2.imshow("Undistorted image right:", image_right)
cv2.waitKey(0)

# Rectify images
kp1, des1, kp2, des2, flann_match_pairs = get_keypoints_and_descriptors(image_left, image_right)
good_matches = lowes_ratio_test(flann_match_pairs, 0.6)
draw_matches(image_left, image_right, kp1, des1, kp2, des2, good_matches)
F, I, points1, points2 = compute_fundamental_matrix(flann_match_pairs, kp1, kp2)
############## Stereo rectify uncalibrated ##############
h1, w1 = (640,480)
h2, w2 = (640, 480)
thresh = 0
_, H1, H2 = cv2.stereoRectifyUncalibrated(
    np.float32(points1), np.float32(points2), fundamental_mat, imgSize=(w1, h1), threshold=thresh,
)

imgL_undistorted = cv2.warpPerspective(image_left, H1, (w1, h1))
imgR_undistorted = cv2.warpPerspective(image_right, H2, (w2, h2))
cv2.imwrite("rectified_L.png", imgL_undistorted)
cv2.imwrite("rectified_R.png", imgR_undistorted)

