import time
from ultralytics import YOLO
import cv2
class HumanPoseDetection():
    def __init__(self):
        self.model = self.load_model()

    def load_model(self):
        model = YOLO('weights/yolov8l-pose.pt').to('cuda')
        return model

    def predict(self, image):
        results = self.model(image, verbose=False)
        return results

detector = HumanPoseDetection()
image = cv2.imread("images/stereoRight/imageL20.png")
detector.predict(image)
time.sleep(5)
start_time = time.time()
for i in range(30):
    image = cv2.imread("images/stereoRight/imageL20.png")
    detector.predict(image)
    end_time = time.time()
    print(round(end_time - start_time, 2))
    start_time = end_time