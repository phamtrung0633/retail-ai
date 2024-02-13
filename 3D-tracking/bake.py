from ultralytics import YOLO
import numpy as np
import pickle
import torch
import json
import cv2
import os

from stream import Stream, STREAM_SENTINEL

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class HumanPoseDetection():
    def __init__(self):
        self.model = self.load_model()

    def load_model(self):
        model = YOLO('weights/yolov8x-pose.pt').to(device)
        return model

    def predict(self, image):
        results = self.model(image, verbose=False)
        return results


BAKE_FOLDER = "bakes"
USE_RESAMPLING = False

DET_UNASSIGNED = np.array([0, 0])
DUPLICATE_POSES_THRESHOLD = 40

if __name__ == '__main__':
    SOURCE_1 = 'videos/2.avi'
    SOURCE_2 = 'videos/3.avi'

    with open('videos/chronology2.json') as file:
        chronology = json.load(file)

        camera_start = chronology['start']

    cap = Stream(SOURCE_1, SOURCE_2, camera_start, (1920, 1080))
    cap.start()

    detector = HumanPoseDetection()


    left = {}
    right = {}

    try:
        retrieve_iterations = 0
        while True:

            res = cap.get()
            if res == STREAM_SENTINEL:
                break
            while not res:
                res = cap.get()

            timestamp_1, img, timestamp_2, img2 = res
            timestamp_1, timestamp_2 = chronology['frames'][retrieve_iterations]
            if USE_RESAMPLING:
                img = cv2.pyrDown(img)
                img2 = cv2.pyrDown(img2)

            print(f"Baking iteration {retrieve_iterations} (Frame: {retrieve_iterations})")

            out_l = detector.predict(img)[0]

            kps_l = out_l.keypoints.xy.cpu().numpy()
            bboxes_l = out_l.boxes.xywh.cpu().numpy()
            conf_l = out_l.keypoints.conf.cpu().numpy()



            out_r = detector.predict(img2)[0]

            kps_r = out_r.keypoints.xy.cpu().numpy()
            bboxes_r = out_r.boxes.xywh.cpu().numpy()
            conf_r = out_r.keypoints.conf.cpu().numpy()



            if USE_RESAMPLING:
                kps_l *= 2
                bboxes_l *= 2

                kps_r *= 2
                bboxes_r *= 2

            kps_l_trim = []
            bboxes_l_trim = []
            conf_l_trim = []

            for pose in range(len(kps_l)):
                kps_l_trim.append(kps_l[pose].tolist())
                bboxes_l_trim.append(bboxes_l[pose].tolist())
                conf_l_trim.append(conf_l[pose].tolist())

            kps_r_trim = []
            bboxes_r_trim = []
            conf_r_trim = []

            for pose in range(len(kps_r)):
                kps_r_trim.append(kps_r[pose].tolist())
                bboxes_r_trim.append(bboxes_r[pose].tolist())
                conf_r_trim.append(conf_r[pose].tolist())

            left[timestamp_1] = [kps_l_trim, bboxes_l_trim, conf_l_trim]
            right[timestamp_2] = [kps_r_trim, bboxes_r_trim, conf_r_trim]

            retrieve_iterations += 1
    except KeyboardInterrupt:
        pass
    print(retrieve_iterations)
    if not os.path.exists(BAKE_FOLDER):
        os.makedirs(BAKE_FOLDER, exist_ok=True)

    lpath = os.path.join(BAKE_FOLDER, SOURCE_1.split('/')[-1].split('.')[0] + ".bake")
    rpath = os.path.join(BAKE_FOLDER, SOURCE_2.split('/')[-1].split('.')[0] + '.bake')

    with open(lpath, 'w') as handle:
        json.dump(left, handle)

    with open(rpath, 'w') as handle:
        json.dump(right, handle)

    print(f"Baked '{SOURCE_1}' to '{lpath}'")
    print(f"Baked '{SOURCE_2}' to '{rpath}'")