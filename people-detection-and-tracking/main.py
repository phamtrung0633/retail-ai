from detection_tracking import ObjectDetection, extract_features, cv2_addBox, get_FrameLabels
import argparse, sys, multiprocessing as mp
from pydantic import BaseModel
import cv2
import numpy as np
from time import time
import math
import cvzone
from ultralytics import YOLO

#Global variables
person_details = {}
inventory = []
inventory_names = {}
object_mapper = {}
model = YOLO('./weights/yolov8n.pt')
model.fuse()
CLASS_NAME_DICT = model.model.names
class Person:
    def __init__(self):
        self.id = ""
        self.xtop = 0
        self.ytop = 0
        self.xbot = 0
        self.ybot = 0
        self.jointLocations = []

    def __str__(self):
        return f"Person: {self.xtop}, {self.ytop}, {self.xbot}, {self.ybot}, Joints are: {self.jointLocations}"

class PersonDetails:
    def __init__(self):
        self.id = ""
        self.pickedUpItems = set()
        self.Wallet = 0
        self.CartItems = set()

class Object:
    def __init__(self, type=-1, xtop=0, ytop=0, xbot=0, ybot=0):
        self.type = type
        self.xtop = xtop
        self.ytop = ytop
        self.xbot = xbot
        self.ybot = ybot

class ObjectInstance:
    def __init__(self):
        self.name = ''
        self.price = ''

class ObjectInventory:
    def __init__(self):
        self.name = ''
        self.price = ''
        self.quantity = 0

class Image:
    def __init__(self):
        self.number = ''
        self.location = ''
        self.objectList = []
        self.personList = []

class Sequence:
    def __init__(self):
        self.imageDataList = []

def FillStoreInventory(filepath):
    try:
        items = open(filepath, 'r')
    except IOError:
        print("Error: can\'t find file or read data ", filepath)
    count = 1
    for eachline in items:
        eachline = eachline.rstrip('\n')
        splitt = eachline.split(' ')
        if len(splitt) != 3:
            print("Check once for load error else ignore", eachline)
        newobject = ObjectInventory()
        newobject.name = splitt[0]
        newobject.price = int(splitt[1])
        newobject.quantity = int(splitt[2])
        inventory.append(newobject)
        inst = ObjectInstance()
        inst.name = newobject.name
        inst.price = newobject.price
        object_mapper[count] = inst
        count += 1
    return

def object_to_detail_mapper(object_number):
    if object_number in object_mapper.keys():
        return object_mapper[object_number]
    else:
        print(" ERROR in object mapping ", object_number)
    return


def calculateDistance(obj1, obj2):
    return math.sqrt(pow(obj1[0] - obj2[0], 2) + pow(obj1[1] - obj2[1], 2))

def calculateCentroid(obj):
    lis = []
    lis.append((float(obj.xtop) + float(obj.xbot))/2)
    lis.append((float(obj.ytop) + float(obj.ybot))/2)
    return lis

def get_bag_and_other_objects(objects):
    bags = []
    others = []
    for item in objects:
        if item.type == 24:
            bags.append(item)
        else:
            others.append(item)
    return bags,others

def get_closest_bag(objects,person):
    bags, others = get_bag_and_other_objects(objects)
    min_dist = 999999999
    closest_bag = Object()
    for item in bags:
        ans = calculateDistance(calculateCentroid(item), calculateCentroid(person))
        if ans < min_dist:
            min_dist = ans
            closest_bag = item
    others.append(closest_bag)
    return others

def sort_obj(obj):
    newlist = sorted(obj, key=lambda x: x.type)
    return newlist

def fill_in_missing_objects(modobjects, object_set):
    seenobjects= {}
    for obj in modobjects:
        if (obj.type)  not in seenobjects.keys():
            seenobjects[obj.type] = 1
    for i in object_set:
        if i not in seenobjects.keys():
            nullobj = Object()
            nullobj.type = i
            modobjects.append(nullobj)
            seenobjects[i] = 1
    return sort_obj(modobjects)

def generate_person_object_locations_in_frames(sequence):
    person_set = {}
    object_set = set()
    idx = 0
    if len(sequence.imageDataList) != 5:
        print("Maintain the length of the sequence as 5")
    for item in sequence.imageDataList:
        objects = item.objectList
        for o in objects:
            object_set.add(o.type)
    for item in sequence.imageDataList:
        idx += 1
        objects = item.objectList
        persons = item.personList
        for person in persons:
            #MAKING THE ASSUMPTION EACH PERSON MUST CARRY A "BAG"
            objectsmod = get_closest_bag(objects, person)
            objectsmod = fill_in_missing_objects(objectsmod, object_set)
            if person.id in sorted(person_set.keys()):
                lis = person_set[person.id]
                last = list[-1]
                prevIdx = last[0]
                while prevIdx != idx - 1:
                    per = Person()
                    per.id = person.name
                    lis.append((prevIdx + 1,[], per))
                    prevIdx += 1
                lis.append((idx, objectsmod, person))
                person_set[person.id] = lis
            else:
                lis = []
                prevIdx = 0
                while prevIdx != idx - 1:
                    per = Person()
                    per.id = person.id
                    lis.append((prevIdx + 1, [], per))
                    prevIdx += 1
                lis.append((idx, objectsmod, person))
                person_set[person.id] = lis

    for item in sorted(person_set.keys()):
        lis = person_set[item]
        last = lis[-1]
        prevIdx = last[0]
        while prevIdx < 5:
            per = Person()
            per.id = item
            lis.append((prevIdx + 1,[],per))
            prevIdx += 1

    for item in sorted(person_set.keys()):
        lis = person_set[item]
        newlis = []
        for it in lis:
            newlis.append((it[1], it[2]))
        person_set[item] = newlis
    return person_set, object_set

def check_activity(lwrist_obj, rwrist_obj, wrists_bag, wloc, threshold
                   , object_class_nums, movementThreshold, bagThreshold = 45):
    detected_activity = []
    length = 5
    for loop in range(2, length):
        for idx in object_class_nums:
            if lwrist_obj[idx][loop][0] != -1 and lwrist_obj[idx][loop - 1][0] != -1:
                if lwrist_obj[idx][loop][0] <= lwrist_obj[idx][loop - 1][0] and lwrist_obj[idx][loop][0] < threshold:
                    detected_activity.append(("picking", idx))
                elif ((lwrist_obj[idx][loop - 1][0] < threshold) and
                    (lwrist_obj[idx][loop][0] > lwrist_obj[idx][loop - 1][0])
                    and (calculateDistance(wloc[loop][0], wloc[loop - 1][0]) >= movementThreshold)):
                    detected_activity.append(("placing", idx))

            if rwrist_obj[idx][loop][0] != -1 and rwrist_obj[idx][loop - 1][0] != -1:
                if rwrist_obj[idx][loop][0] <= rwrist_obj[idx][loop - 1][0] and rwrist_obj[idx][loop][0] < threshold:
                    detected_activity.append(("picking", idx))
                elif ((rwrist_obj[idx][loop - 1][0] < threshold) and
                    (rwrist_obj[idx][loop][0] > rwrist_obj[idx][loop - 1][0])
                    and (calculateDistance(wloc[loop][1], wloc[loop - 1][1]) >= movementThreshold)):
                    detected_activity.append(("placing", idx))
        if wrists_bag[loop][0] != -1 and wrists_bag[loop][1] != -1:
            if wrists_bag[loop][0] <= wrists_bag[loop - 1][0] and wrists_bag[loop][0] <= bagThreshold:
                detected_activity.append(("cart", 24))
            if wrists_bag[loop][1] <= wrists_bag[loop - 1][1] and wrists_bag[loop][1] <= bagThreshold:
                detected_activity.append(("cart", 24))
    return detected_activity

def check_movement(obj , threshold):
    loop = 1
    dist_over_time = []
    dist_over_time.append(calculateCentroid(obj[0]))
    flag =False
    while loop < len(obj):
        dist_over_time.append(calculateCentroid(obj[loop]))
        if obj[loop].xtop !=0 and obj[loop].ytop !=0 and obj[loop-1].xtop !=0 and obj[loop].ytop !=0:
            if calculateDistance(calculateCentroid(obj[loop]) , calculateCentroid(obj[loop - 1])) >= threshold:
                flag = True
        loop += 1
    return flag

def check_wrist_movement(obj , threshold):
    loop = 1
    flag = False
    while loop < len(obj):
        if obj[loop][0][0] !=-1:
            if calculateDistance(obj[loop][0] , obj[loop - 1][0]) >= threshold:
                flag = True
        if obj[loop][1][0] !=-1:
            if calculateDistance(obj[loop][1] , obj[loop - 1][1]) >= threshold:
                flag = True
        loop += 1
    return flag

def activityRecognitionThreshold(sequence, threshold, movementThreshold):
    detected_activity = []
    person_set, object_set = generate_person_object_locations_in_frames(sequence)
    for person in sorted(person_set.keys()):
        person_data = person_set[person]
        obj_over_time_lwrist = {}
        obj_over_time_rwrist = {}
        obj_over_time_bag = {}
        obj = {}
        for i in object_set:
            obj_over_time_lwrist[i] = []
            obj_over_time_rwrist[i] = []
            obj_over_time_bag[i] = []
            obj[i] = []
        wrist_bag_over_time = []
        wrist_loc_over_time = []

        for item in person_data:
            bag, others = get_bag_and_other_objects(item[0])
            if len(bag) == 0:
                bag = Object()
                bag.type = 8
            else:
                bag = bag[0]

            per = item[1]
            if len(per.jointLocations) == 0:
                lwristloc = (-1, -1)
                rwristloc = (-1, -1)
                wrist_bag_over_time.append((-1, -1))
                wrist_loc_over_time.append((lwristloc, rwristloc))
            else:
                lwristloc = per.jointLocations[7]
                rwristloc = per.jointLocations[4]
                if calculateCentroid(bag) == (0, 0):
                    wrist_bag_over_time.append((-1, -1))
                else:
                    wrist_bag_over_time.append((calculateDistance(lwristloc,calculateCentroid(bag)),
                                                 calculateDistance(rwristloc,calculateCentroid(bag))))
                wrist_loc_over_time.append((lwristloc, rwristloc))

            for index in object_set:
                flag = 0
                for o in others:
                    if index == o.type:
                        obj[index].append(o)
                        obj_over_time_lwrist[index].append((calculateDistance(calculateCentroid(o) , lwristloc) , o))
                        obj_over_time_rwrist[index].append((calculateDistance(calculateCentroid(o), rwristloc), o))
                        obj_over_time_bag[index].append((calculateDistance(calculateCentroid(o), calculateCentroid(bag)), o))
                        flag = 1
                if flag == 0:
                    o = Object()
                    o.type = index
                    obj_over_time_lwrist[index].append((-1, o))
                    obj_over_time_rwrist[index].append((-1, o))
                    obj_over_time_bag[index].append((-1, o))

        det_activity = check_activity(obj_over_time_lwrist, obj_over_time_rwrist, obj_over_time_bag, wrist_bag_over_time,
                                      wrist_loc_over_time, object_set, threshold, movementThreshold)
        final_activity = []
        for item in det_activity:
            act = item[0]
            object_class = item[1]
            if act != "cart":
                if check_movement(obj[object_class], 10):
                    final_activity.append(item)
            else:
                if check_wrist_movement(wrist_loc_over_time, 10):
                    final_activity.append(item)
        detected_activity.append((person, final_activity))
    return detected_activity

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

def predict_objects(img):
    results_from_model = model(img, stream=True, verbose=False)
    return results_from_model

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
            curimage = Image()

            #Start detecting people along with their keypoints
            results = detector.predict(img)
            frame_cnt, people_data = detector.track_detect(results, img, w, h, frame_cnt)
            persons = []
            for key in people_data.keys():
                curperson = Person()
                curperson.id = key
                curperson.xtop = people_data[key][0][0]
                curperson.ytop = people_data[key][0][1]
                curperson.xbot = people_data[key][0][2]
                curperson.ybot = people_data[key][0][3]
                curperson.jointLocations = people_data[key][1]
                persons.append(curperson)
            curimage.personList = persons

            #Start detecting other objects in the scene along with their keypoints
            objects = []
            objects_result = predict_objects(img)
            for r in objects_result:
                boxes = r.boxes
                for box in boxes:
                    cls = int(box.cls[0])
                    if cls == 0:
                        continue
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    currentClass = CLASS_NAME_DICT[cls]
                    text_scale, text_thickness, line_thickness = get_FrameLabels(img)
                    cvzone.putTextRect(img, f'{currentClass} ID {cls}', (x1, x2), scale=1,
                                       thickness=1,
                                       colorR=(0, 0, 255))
                    cv2_addBox(int(cls), img, x1, y1, x2, y2, line_thickness, text_thickness, text_scale)
                    curobject = Object(cls, x1, y1, x2, y2)
                    objects.append(curobject)
            curimage.objectList = objects
            end_time = time()
            fps = 1 / np.round(end_time - start_time, 2)
            cv2.imshow('Image', img)
            if cv2.waitKey(5) == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
    except Exception as e:
        raise
    finally:
        extract_p.terminate()
        extract_p.join()
        shared_images_queue.close()