import cv2
import numpy as np
import glob
from tqdm import tqdm
import PIL.ExifTags
import PIL.Image
from matplotlib import pyplot as plt


def create_point_cloud_file(vertices, colors, filename):
    colors = colors.reshape(-1, 3)
    vertices = np.hstack([vertices.reshape(-1, 3), colors])

    ply_header = '''ply
		format ascii 1.0
		element vertex %(vert_num)d
		property float x
		property float y
		property float z
		property uchar red
		property uchar green
		property uchar blue
		end_header
		'''
    with open(filename, 'w') as f:
        f.write(ply_header % dict(vert_num=len(vertices)))
        np.savetxt(f, vertices, '%f %f %f %d %d %d')


def downsample_image(image, reduce_factor):
    for i in range(0, reduce_factor):
        # Check if image is color or grayscale
        if len(image.shape) > 2:
            row, col = image.shape[:2]
        else:
            row, col = image.shape

        image = cv2.pyrDown(image, dstsize=(col // 2, row // 2))
    return image


cv_file = cv2.FileStorage()
cv_file.open('calib_data/stereoMap.xml', cv2.FileStorage_READ)
stereoMapL_x = cv_file.getNode('stereoMapL_x').mat()
stereoMapL_y = cv_file.getNode('stereoMapL_y').mat()
stereoMapR_x = cv_file.getNode('stereoMapR_x').mat()
stereoMapR_y = cv_file.getNode('stereoMapR_y').mat()
camera_matrix_l = np.load('calib_data/camera_matrix_l.npy')
camera_matrix_r = np.load('calib_data/camera_matrix_r.npy')
dist_l = np.load('calib_data/dist_l.npy')
dist_r = np.load('calib_data/dist_r.npy')
Q = cv_file.getNode('q').mat()
image_left = cv2.imread('images/environmentLeft/imageL.png')
image_right = cv2.imread('images/environmentRight/imageR.png')
image_right = cv2.remap(image_right, stereoMapR_x, stereoMapR_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
image_left = cv2.remap(image_left, stereoMapL_x, stereoMapL_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
img_left_gray = cv2.cvtColor(image_left, cv2.COLOR_BGR2GRAY)
img_right_gray = cv2.cvtColor(image_right, cv2.COLOR_BGR2GRAY)
cv2.imshow("Rectified Left calibrated", img_left_gray)
cv2.imshow("Rectified Right calibrated", img_right_gray)
cv2.waitKey(0)

block_size = 11
min_disp = 0
max_disp = 80
num_disp = max_disp - min_disp  # Needs to be divisible by 16

# Create Block matching object.
stereo = cv2.StereoSGBM_create(minDisparity=min_disp,
                               numDisparities=num_disp,
                               blockSize=block_size,
                               uniquenessRatio=8,
                               speckleWindowSize=150,
                               speckleRange=2,
                               disp12MaxDiff=10,
                               P1=8 * 3 * block_size ** 2,  # 8*img_channels*block_size**2,
                               P2=32 * 3 * block_size ** 2)  # 32*img_channels*block_size**2)

disparity_map = stereo.compute(img_left_gray, img_right_gray)
plt.imshow(disparity_map, 'gray')
plt.show()
disparity_map = np.float32(np.divide(disparity_map, 16.0))
points3D = cv2.reprojectImageTo3D(disparity_map, Q, handleMissingValues=False)
colors = cv2.cvtColor(image_left, cv2.COLOR_BGR2RGB)
mask_map = disparity_map > disparity_map.min()
output_points = points3D[mask_map]
output_colors = colors[mask_map]
output_file = 'environment_pcl.ply'
# Generate point cloud file
create_point_cloud_file(output_points, output_colors, output_file)
