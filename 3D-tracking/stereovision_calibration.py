
import numpy as np
import cv2 as cv
import glob
import cv2
chessBoardSize = (8, 6)
frameSize = (640, 480)

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((chessBoardSize[0] * chessBoardSize[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessBoardSize[0], 0:chessBoardSize[1]].T.reshape(-1, 2)
objp = objp * 75
objpoints = []
imgpointsL = []
imgpointsR = []

imagesLeft = glob.glob('images/stereoLeft/*.png')
imagesRight = glob.glob('images/stereoRight/*.png')

for imgLeft, imgRight in zip(imagesLeft, imagesRight):
    imgL = cv.imread(imgLeft)
    imgR = cv.imread(imgRight)
    grayL = cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)
    grayR = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)
    retL, cornersL = cv.findChessboardCorners(grayL, chessBoardSize, None)
    retR, cornersR = cv.findChessboardCorners(grayR, chessBoardSize, None)
    print(retL, retR)
    if (retL is True) and (retR is True):

        objpoints.append(objp)
        cornersL = cv.cornerSubPix(grayL, cornersL, (11, 11), (-1, -1), criteria)
        imgpointsL.append(cornersL)
        cornersR = cv.cornerSubPix(grayR, cornersR, (11, 11), (-1, -1), criteria)
        imgpointsR.append(cornersR)
        cv.drawChessboardCorners(imgL, chessBoardSize, cornersL, retL)
        cv.imshow('img left', imgL)
        cv.drawChessboardCorners(imgR, chessBoardSize, cornersR, retR)
        cv.imshow('img right', imgR)
        cv.waitKey(1000)

cv.destroyAllWindows()

retL, cameraMatrixL, distL, rvecsL, tvecsL = cv.calibrateCamera(objpoints, imgpointsL, frameSize, None, None)
heightL, widthL, channelsL = imgL.shape
newCameraMatrixL, roi_L = cv.getOptimalNewCameraMatrix(cameraMatrixL, distL, (widthL, heightL), 1, (widthL, heightL))

retR, cameraMatrixR, distR, rvecsR, tvecsR = cv.calibrateCamera(objpoints, imgpointsR, frameSize, None, None)
heightR, widthR, channelsR = imgR.shape
newCameraMatrixR, roi_R = cv.getOptimalNewCameraMatrix(cameraMatrixR, distR, (widthR, heightR), 1, (widthR, heightR))

rotationMatrixLeft, _ = cv2.Rodrigues(rvecsL[0])
projectionMatrixLeft = np.hstack((rotationMatrixLeft, tvecsL[0]))
projectionMatrixLeft = cameraMatrixL @ projectionMatrixLeft

rotationMatrixRight, _ = cv2.Rodrigues(rvecsR[0])
projectionMatrixRight = np.hstack((rotationMatrixRight, tvecsR[0]))
projectionMatrixRight = cameraMatrixR @ projectionMatrixRight
np.save("calib_data/projection_matrix_l.npy", projectionMatrixLeft)
np.save("calib_data/projection_matrix_r.npy", projectionMatrixRight)
np.save("calib_data/camera_matrix_l.npy", cameraMatrixL)
np.save("calib_data/camera_matrix_r.npy", cameraMatrixR)
np.save("calib_data/dist_l.npy", distL)
np.save("calib_data/dist_r.npy", distR)
np.save("calib_data/new_cam_l.npy", newCameraMatrixL)
np.save("calib_data/new_cam_r.npy", newCameraMatrixR)
flags = 0
flags |= cv.CALIB_FIX_INTRINSIC
criteria_stereo = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
retStereo, newCameraMatrixL, distL, newCameraMatrixR, distR, rot, trans, essentialMatrix, fundamentalMatrix = cv.stereoCalibrate(objpoints, imgpointsL, imgpointsR, newCameraMatrixL, distL, newCameraMatrixR, distR, grayL.shape[::-1], criteria_stereo, flags)

rectifyScale = 1
rectL, rectR, projMatrixL, projMatrixR, Q, roi_L, roi_R= cv.stereoRectify(newCameraMatrixL, distL, newCameraMatrixR, distR, grayL.shape[::-1], rot, trans, rectifyScale,(0,0))
np.save("calib_data/projRectL.npy", projMatrixL)
np.save("calib_data/RotRectL.npy", retL)
np.save("calib_data/projRectR.npy", projMatrixR)
np.save("calib_data/RotRectR.npy", retR)

stereoMapL = cv.initUndistortRectifyMap(newCameraMatrixL, distL, rectL, projMatrixL, grayL.shape[::-1], cv.CV_16SC2)
stereoMapR = cv.initUndistortRectifyMap(newCameraMatrixR, distR, rectR, projMatrixR, grayR.shape[::-1], cv.CV_16SC2)

print("Saving parameters!")
cv_file = cv.FileStorage('calib_data/stereoMap.xml', cv.FILE_STORAGE_WRITE)

cv_file.write('stereoMapL_x',stereoMapL[0])
cv_file.write('stereoMapL_y',stereoMapL[1])
cv_file.write('stereoMapR_x',stereoMapR[0])
cv_file.write('stereoMapR_y',stereoMapR[1])

cv_file.release()



