import cv2
cap = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(2)

cam_1 = {}
cam_2 = {}
iteration = 0
while True:
    cap.grab()
    cap2.grab()

    _, img = cap.retrieve()
    _, img2 = cap2.retrieve()
    timestamp = str(round(time.time() - camera_start)
    cv2.imwrite("./data_record/cam1/" + iteration,)
    cam_2[iteration] = {'filename': str(timestamp) + '_cam2', 'image': img)

    iteration += 1

    break