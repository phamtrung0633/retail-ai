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
    SOURCE_1 = 'videos/0.avi'
    SOURCE_2 = 'videos/1.avi'

    with open('videos/chronology.json') as file:
        chronology = json.load(file)

        camera_start = chronology['start']

    cap = Stream(SOURCE_1, SOURCE_2, camera_start)
    cap.start()

    detector = HumanPoseDetection()
    retrieve_iterations = 0

    left = []
    right = []

    try:
        while True:
            res = cap.get()

            while not res:
                res = cap.get()

            if res == STREAM_SENTINEL:
                break

            if retrieve_iterations % 2 == 0:
                timestamp_1, img, timestamp_2, img2 = res

                if USE_RESAMPLING:
                    img = cv2.pyrDown(img)
                    img2 = cv2.pyrDown(img2)
            else:
                retrieve_iterations += 1
                continue

            print(f"Baking iteration {retrieve_iterations} (Frame: {retrieve_iterations // 2})")

            out_l = detector.predict(img)[0]
            
            kps_l = out_l.keypoints.xy.cpu().numpy()
            bboxes_l = out_l.boxes.xywh.cpu().numpy()
            conf_l = out_l.keypoints.conf.cpu().numpy()

            retained_l = []

            for i, pose_1 in enumerate(kps_l):
                dist = 0

                for j, pose_2 in enumerate(kps_l[i + 1:]):

                    mask_1 = np.invert((pose_1 == DET_UNASSIGNED).all(1))
                    mask_2 = np.invert((pose_2 == DET_UNASSIGNED).all(1))
                    comb = np.bitwise_and(mask_1, mask_2)

                    dist = np.linalg.norm(pose_2[comb] - pose_1[comb], axis = 0).mean()
                    print(f"\t[LEFT] Considering pose pair ({i}, {i + j + 1}), distance is {dist}")

                    if dist >= DUPLICATE_POSES_THRESHOLD:
                        retained_l.append(i + j + 1)

            out_r = detector.predict(img2)[0]
            
            kps_r = out_r.keypoints.xy.cpu().numpy()
            bboxes_r = out_r.boxes.xywh.cpu().numpy()
            conf_r = out_r.keypoints.conf.cpu().numpy()

            retained_r = []

            for i, pose_1 in enumerate(kps_r):
                dist = 0

                for j, pose_2 in enumerate(kps_r[i + 1:]):
                    mask_1 = np.invert((pose_1 == DET_UNASSIGNED).all(1))
                    mask_2 = np.invert((pose_2 == DET_UNASSIGNED).all(1))
                    comb = np.bitwise_and(mask_1, mask_2)

                    dist = np.linalg.norm(pose_2[comb] - pose_1[comb], axis = 0).mean()
                    print(f"\t[RIGHT] Considering pose pair ({i}, {i + j + 1}), distance is {dist}")

                    if dist >= DUPLICATE_POSES_THRESHOLD:
                        retained_r.append(i + j + 1)

            if USE_RESAMPLING:
                kps_l *= 2
                bboxes_l *= 2

                kps_r *= 2
                bboxes_r *= 2

            left.append((kps_l[retained_l].tolist(), bboxes_l[retained_l].tolist(), conf_l[retained_l].tolist()))
            right.append((kps_r[retained_r].tolist(), bboxes_r[retained_r].tolist(), conf_r[retained_r].tolist()))

            retrieve_iterations += 1
    except KeyboardInterrupt:
        pass

    if not os.path.exists(BAKE_FOLDER):
        os.makedirs(BAKE_FOLDER, exist_ok = True)

    lpath = os.path.join(BAKE_FOLDER, SOURCE_1.split('/')[-1].split('.')[0] + ".bake")
    rpath = os.path.join(BAKE_FOLDER, SOURCE_2.split('/')[-1].split('.')[0] + '.bake')

    with open(lpath, 'wb') as handle:
        pickle.dump(left, handle, protocol = pickle.HIGHEST_PROTOCOL)

    with open(rpath, 'wb') as handle:
        pickle.dump(right, handle, protocol = pickle.HIGHEST_PROTOCOL)

    print(f"Baked '{SOURCE_1}' to '{lpath}'")
    print(f"Baked '{SOURCE_2}' to '{rpath}'")