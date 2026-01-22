from ultralytics import YOLO
from collections import defaultdict

import os
import cv2
import torch
import shutil
import numpy as np

ungrouped = "bodyimages/ungrouped"
grouped = "bodyimages/grouped"

if not os.path.exists(ungrouped):
    os.makedirs(ungrouped, exist_ok = True)
else:
    shutil.rmtree(ungrouped)
    os.makedirs(ungrouped, exist_ok = True)

if not os.path.exists(grouped):
    os.makedirs(grouped, exist_ok = True)
else:
    shutil.rmtree(grouped)
    os.makedirs(grouped, exist_ok = True)

PATH = '1.avi'

DET_UNASSIGNED = np.array([0, 0])
DUPLICATE_POSES_THRESHOLD = 40

RESOLUTION = (640, 480)

SIMILARITY_THRESHOLD = 4.5
MAX_HIST = 7


device = 'cuda' if torch.cuda.is_available() else 'cpu'

class HumanPoseDetection():
    def __init__(self):
        self.model = self.load_model()
        self.warm_up()

    def warm_up(self):
        dummy_image = cv2.imread("reprojected_envi_1.png")
        self.predict(dummy_image)

    def load_model(self):
        model = YOLO('weights/yolov8x-pose.pt').to(device)
        return model

    def predict(self, image):
        results = self.model(image, verbose=False)
        return results

detector = HumanPoseDetection()
stream = cv2.VideoCapture(PATH)

from LATransformer.model import LATransformerTest
from LATransformer.helpers import LATransformerForward

import timm

latreid_device = 'cuda' if torch.cuda.is_available() else 'cpu'

latreid_backbone = timm.create_model('vit_base_patch16_224', pretrained = True, num_classes = 751)
latreid_backbone = latreid_backbone.to(latreid_device)

latreid = LATransformerTest(latreid_backbone, lmbd = 8).to(latreid_device)
latreid.load_state_dict(torch.load('weights/latransformer_market1501.pth'), strict = False)
latreid.eval()

bodyimages = defaultdict(list) 
iterations = 0

try:
    while True:
        ret, frame = stream.read()

        if not ret:
            break

        poses = detector.predict(frame)[0]

        bboxes = poses.boxes.xywh.cpu().numpy()

        confs = poses.keypoints.conf.cpu().numpy()
        kps = poses.keypoints.xy.cpu().numpy()

        eliminated = []

        for i, pose_1 in enumerate(kps):
            if i in eliminated:
                continue

            dist = 0

            for j, pose_2 in enumerate(kps[i + 1:]):
                if i + j + 1 in eliminated:
                    continue

                mask_1 = np.invert((pose_1 == DET_UNASSIGNED).all(1))
                mask_2 = np.invert((pose_2 == DET_UNASSIGNED).all(1))
                comb = np.bitwise_and(mask_1, mask_2)

                dist = np.linalg.norm(pose_2[comb] - pose_1[comb], axis = 0).mean()
                print(f"\tConsidering pose pair ({i}, {i + j + 1}), distance is {dist}")

                if dist < DUPLICATE_POSES_THRESHOLD:
                    eliminated.append(i + j + 1)
            
        for idx, bbox in enumerate(bboxes):
            if idx not in eliminated:
                x, y, w, h = bbox

                x1, x2, y1, y2 = int(x - w/2), int(x + w/2), int(y - h/2), int(y + h/2)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(RESOLUTION[0], x2), min(RESOLUTION[1], y2)

                bodyimages[iterations].append((frame[y1:y2, x1:x2, :]))

        print(f"Frame {iterations} complete, detected {len(bodyimages[iterations])} poses, eliminated {len(eliminated)} bad poses")


        iterations += 1
except KeyboardInterrupt:
    pass

print("Saving ungrouped images...", end = ' ')

for frame in bodyimages:
    for idx, image in enumerate(bodyimages[frame]):
        cv2.imwrite(f"{ungrouped}/{frame}_{idx}.png", image)

print("Complete!")

print("Running latreid on the bodyimages...")

groups = {}

try:
    for frame in bodyimages:
        print(f"Evaluating frame {frame} for reid")
        for idx, image in enumerate(bodyimages[frame]):
            similarity_scores = []

            if frame == 0: # Special case just make a new group for each body images
                groups[idx] = [image]
            else:
                embedding = LATransformerForward(latreid, latreid_device, [image])
                for group in groups:
                    group_embedding = LATransformerForward(latreid, latreid_device, groups[group]).reshape(len(groups[group]), -1)

                    similarity = (group_embedding.mean(0) - embedding).norm().numpy()
                    similarity_scores.append(similarity)
                    
                    print(f"\t\tDetection {idx} <-> Group {group} similarity is {similarity}")

                group_idx, score = min(enumerate(similarity_scores), key = lambda enum: enum[1])

                if score < SIMILARITY_THRESHOLD:
                    groups[group_idx].append(image)
                    groups[group_idx] = groups[group_idx][-MAX_HIST:]
                    print(f"\tDetection {idx} matched to Group {group_idx}!")
                else:
                    groups[len(groups)] = [image] # Make a new group
                    print(f"\tDetection {idx} too far from any group ({score} > {SIMILARITY_THRESHOLD}), Group {len(groups) - 1} created!")
except KeyboardInterrupt:
    pass

print("Latreid evaluation completed, saving images...", end = ' ')

for group in groups:
    folder = f"{grouped}/{group}"

    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok = True)
    else:
        shutil.rmtree(folder)
        os.makedirs(folder, exist_ok = True)

    for idx, image in enumerate(groups[group]):
        cv2.imwrite(f"{folder}/{idx}.png", image)


print(f"Complete! Saved the images for {len(groups)} groups")
