from detection_tracking import ObjectDetection, extract_features
import argparse, sys, multiprocessing as mp
from pydantic import BaseModel
import cv2
import numpy as np
from time import time

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

#This variable is used to get keypoint by its name
get_keypoint = GetKeypoint()

if __name__ == "__main__":
    FeatsLock = mp.Lock()
    shared_feats_dict = mp.Manager().dict()
    shared_images_queue = mp.Queue()
    extract_p = mp.Process(target=extract_features, args=(shared_feats_dict, shared_images_queue, FeatsLock,))
    extract_p.start()
    try:
        detector = ObjectDetection(shared_feats_dict, shared_images_queue, FeatsLock)
        cap = cv2.VideoCapture(detector.capture)
        assert cap.isOpened()
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_cnt = 0
        while True:
            start_time = time()
            _, img = cap.read()
            assert _
            results = detector.predict(img)
            frame_cnt, people_data = detector.track_detect(results, img, w, h, frame_cnt)
            print(people_data)
            end_time = time()
            fps = 1 / np.round(end_time - start_time, 2)
            print(fps)
            cv2.imshow('Image', img)
            if cv2.waitKey(1) == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
    except Exception as e:
        raise
    finally:
        extract_p.terminate()
        extract_p.join()
        shared_images_queue.close()