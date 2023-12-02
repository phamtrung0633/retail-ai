import copy
import operator
import argparse, sys, multiprocessing as mp
from ultralytics import YOLO
import super_gradients
import cv2
import cvzone
import math
from time import time, sleep
from deep_sort_realtime.deepsort_tracker import DeepSort
import torchreid
from time import time
import numpy as np
from reid import REID

#Extractor is used to extract embeddings for deepsort before hand
extractor = torchreid.utils.FeatureExtractor(
    model_name='osnet_x1_0',
    model_path='./weights/osnet_x1_0.pth.tar',
    device='cpu'
)

def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
    return color

def cv2_addBox(track_id, frame, x1, y1, x2, y2, line_thickness, text_thickness,text_scale):
    color = get_color(abs(track_id))
    cv2.rectangle(frame, (x1, y1), (x2, y2),color=color, thickness=line_thickness)
    cv2.putText(frame, str(track_id),(x1, y1+30), cv2.FONT_HERSHEY_PLAIN, text_scale, (0,0,255),thickness=text_thickness)

def get_FrameLabels(frame):
    text_scale = max(1, frame.shape[1] / 1600.)
    text_thickness = 1 if text_scale > 1.1 else 1
    line_thickness = max(1, int(frame.shape[1] / 500.))
    return text_scale, text_thickness, line_thickness
class ObjectDetection():
    def __init__(self, feats_dict_shared, images_queue_shared, feat_dict_lock):
        self.tracker = DeepSort(max_age = 5,
                           n_init=2,
                           nms_max_overlap=1.0,
                           max_cosine_distance=0.3,
                           nn_budget=None,
                           override_track_class=None,
                           embedder="mobilenet",
                           half=True,
                           bgr=True,
                           embedder_gpu=True,
                           embedder_model_name=None,
                           embedder_wts=None,
                           polygon=False,
                           today=None)
        self.feats_dict_shared = feats_dict_shared
        self.images_queue_shared = images_queue_shared
        self.feat_dict_lock = feat_dict_lock
        self.capture = 0
        self.model = self.load_model()
        self.final_fuse_id = dict()
        self.pose_by_id = dict()
        self.images_by_id = dict()
        self.exist_ids = set()
        self.threshold = 600
        self.reid = REID()

    def load_model(self):
        model = super_gradients.training.models.get("yolo_nas_pose_n", pretrained_weights="coco_pose")
        return model

    def predict(self, img):
        results = self.model.predict(img, conf=.55)
        return results

    def track_detect(self, results, img, width, height, frame_cnt):
        tracker = self.tracker
        detections = []
        embeds = []
        poses_of_persons = []
        for r in results:
            index = 0
            boxes = r.prediction.bboxes_xyxy
            poses = r.prediction.poses
            confidences = r.prediction.scores
            for box in boxes:
                conf = math.ceil(confidences[index] * 100) / 100
                x1, y1, x2, y2 = box
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                crop = img[y1:y2, x1:x2, :]
                features = extractor(crop)[0].cpu().numpy()
                w, h = x2 - x1, y2 - y1
                currentClass = "Person"
                embeds.append(features)
                poses_of_persons.append(poses[index])
                detections.append((([x1, y1, w, h]), conf, currentClass))
                index += 1
        tracks = tracker.update_tracks(detections, frame=img, embeds=embeds, poses_list=poses_of_persons)
        tmp_ids = []
        track_cnt = dict()
        ids_per_frame = []
        frame_cnt += 1
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            ltrb = track.to_ltrb()
            self.pose_by_id[track_id] = track.pose
            bbox = ltrb
            x1,y1,x2,y2 = bbox
            x1,y1,x2,y2 = int(x1), int(y1), int(x2), int(y2)
            if bbox[0] >= 0 and bbox[1] >= 0 and bbox[3] < height and bbox[2] < width:
                tmp_ids.append(track_id)
                track_cnt[track_id] = [x1, y1, x2, y2]
                if track_id not in self.images_by_id:
                    self.images_by_id[track_id] = [img[y1:y2, x1:x2, :]]
                else:
                    self.images_by_id[track_id].append(img[y1:y2, x1:x2, :])

        if len(tmp_ids) > 0:
            ids_per_frame.append(set(tmp_ids))

        for i in self.images_by_id:
            if len(self.images_by_id[i]) > 70:
                del self.images_by_id[i][:20:]
            self.images_queue_shared.put([i, frame_cnt, self.images_by_id[i]])

        self.feat_dict_lock.acquire()
        local_feats_dict = {}
        for key, value in self.feats_dict_shared.items():
            local_feats_dict[key] = copy.deepcopy(value)
        self.feat_dict_lock.release()
        min_num_features = 10
        for f in ids_per_frame:
            if f:
                if len(self.exist_ids) == 0:
                    for i in f:
                        self.final_fuse_id[i] = [i]
                        self.exist_ids = self.exist_ids | f
                else:
                    new_ids = f - self.exist_ids
                    for nid in new_ids:
                        print("Started collecting with NEW ids")
                        t = time()
                        if not (nid in local_feats_dict.keys()):
                            self.exist_ids.add(nid)
                            if nid in local_feats_dict.keys():
                                print("Not enough feats: {}, ID: {}".format(local_feats_dict[nid].shape[0], nid))
                            else:
                                print("New ID to be extracted: {}".format(nid))
                            continue
                        else:
                            pass

                    unpickable = []
                    for i in f:
                        for key,item in self.final_fuse_id.items():
                            if i in item:
                                unpickable += self.final_fuse_id[key]

                    for left_out_id in f & (self.exist_ids - set(unpickable)):
                        dis = []
                        if left_out_id not in local_feats_dict.keys() or local_feats_dict[left_out_id].shape[0] < min_num_features:
                            continue
                        for main_id in self.final_fuse_id.keys():
                            tmp = np.mean(self.reid.compute_distance(local_feats_dict[left_out_id], local_feats_dict[main_id]))
                            dis.append([main_id, tmp])
                        if dis:
                            dis.sort(key=operator.itemgetter(1))
                            for i in range(0, len(dis)):
                                if dis[i][1] < self.threshold:
                                    print("Real ID detected: ", dis[i][0])
                                    combined_id = dis[i][0]
                                    self.images_by_id[combined_id] += self.images_by_id[left_out_id]
                                    #Can you delete the images of the left out id after then ?? to save storage
                                    self.final_fuse_id[combined_id].append(left_out_id)
                                    break
                                else:
                                    print("New ID recorded: ", left_out_id)
                                    self.final_fuse_id[left_out_id] = [left_out_id]
                                    break
                        else:
                            self.final_fuse_id[left_out_id] = [left_out_id]
                            self.exist_ids.add(left_out_id)
        people_data = dict()
        for idx in self.final_fuse_id:
            for i in self.final_fuse_id[idx]:
                for current_ids in ids_per_frame:
                    for f in current_ids:
                        if str(i) == str(f) or str(idx) == str(f):
                            text_scale, text_thickness, line_thickness = get_FrameLabels(img)
                            detection_track = track_cnt[f]
                            people_data[idx] = [[detection_track[0], detection_track[1], detection_track[2], detection_track[3]], self.pose_by_id[f]]
                            cvzone.putTextRect(img, f'ID: {int(idx)}', (detection_track[0], detection_track[1]), scale=1, thickness=1,
                                               colorR=(0, 0, 255))
                            cv2_addBox(int(idx), img, detection_track[0], detection_track[1], detection_track[2], detection_track[3], line_thickness, text_thickness, text_scale)
        return frame_cnt, people_data

    def __call__(self):
        cap = cv2.VideoCapture(self.capture)
        assert cap.isOpened()
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_cnt = 0

        while True:
            start_time = time()
            _, img = cap.read()
            assert _
            results = self.predict(img)
            end_time = time()
            fps = 1 / np.round(end_time - start_time, 2)
            frame_cnt = self.track_detect(results, img, w, h, frame_cnt)
            cv2.imshow('Image', img)
            if cv2.waitKey(1) == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

def extract_features(feats, q, f_lock) -> None:
    from reid import REID
    reid = REID()
    l_dict = dict()
    while True:
        #Does this mean that always the latest image of an object will be the embedding of it ? Would it cause any cons ?
        if not q.empty():
            idx, cnt, img = q.get()
            if idx in l_dict.keys():
                if l_dict[idx][0] < cnt:
                    l_dict[idx] = [cnt, img]
                else:
                    continue
            else:
                l_dict[idx] = [cnt, img]
            f = reid.features(l_dict[idx][1])
            f_lock.acquire()
            feats[idx] = f
            f_lock.release()



if __name__ == "__main__":
    FeatsLock = mp.Lock()
    shared_feats_dict = mp.Manager().dict()
    shared_images_queue = mp.Queue()
    extract_p = mp.Process(target=extract_features, args=(shared_feats_dict, shared_images_queue, FeatsLock,))
    extract_p.start()
    try:
        detector = ObjectDetection(shared_feats_dict, shared_images_queue, FeatsLock)
        detector()
    except Exception as e:
        raise
    finally:
        extract_p.terminate()
        extract_p.join()
        shared_images_queue.close()



