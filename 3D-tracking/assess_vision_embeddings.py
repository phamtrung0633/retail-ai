import os.path
import time
import cv2
from detection_and_tracking import HumanPoseDetection
from stream import Stream

RECORD_FOOTAGES = False
TIMESTAMP_RESOLUTION = 3
RESOLUTION = (1920, 1080)
SOURCE = 0
detector = HumanPoseDetection()
START_TIME = None
END_TIME = None
output_dir = "frames_vision_assess"
BAKE = None
if RECORD_FOOTAGES:
    start = round(time.time(), TIMESTAMP_RESOLUTION)
    cap = cv2.VideoCapture(SOURCE)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('m', 'j', 'p', 'g'))
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, RESOLUTION[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION[1])
    while True:
        # print(f"Running is {running.value}")
        try:
            ret, frame = cap.read()
            timestamp1 = round(time.time() - start, TIMESTAMP_RESOLUTION)
            filepath = os.path.join(output_dir, f"{timestamp1}.png")
            cv2.imwrite(filepath, frame)
        except KeyboardInterrupt:
            break
else:
    for index, file in enumerate(sorted(os.listdir(output_dir), key=lambda s: float(s[:-4]))):
        timestamp = float(file[:-4])
        if START_TIME <= timestamp <= END_TIME:
            frame = cv2.imread(file)
            frame_resized = cv2.pyrDown(frame.copy())
            if not BAKE:
                poses_data_cur_frame = detector.predict(frame_resized)[0]
                try:
                    poses_keypoints = poses_data_cur_frame.keypoints.xy.cpu().numpy()
                    poses_keypoints *= 2
                    poses_bboxes = poses_data_cur_frame.boxes.xywh.cpu().numpy()
                    poses_bboxes *= 2
                    poses_conf = poses_data_cur_frame.keypoints.conf.cpu().numpy()
                except Exception:
                    continue
