import ctypes
import time
import json
import sys

from multiprocessing import Process, Queue, Value
from queue import Empty

import cv2

TIMESTAMP_RESOLUTION = 3
MAX_FRAMES = 0 # Infinite
STREAM_SENTINEL = None, None, None, None
FRAMERATE = 15

RECORD_VIDEO = True

class Stream:

    def __init__(self, source, source2, camera_start, RESOLUTION):
        self.buffer = Queue(MAX_FRAMES)
        self.running = Value(ctypes.c_bool, True)
        self.resolution = RESOLUTION
        self.process = Process(target = self.run, args = (source, source2, self.running, self.buffer))
        self.camera_start = camera_start

    def start(self):
        self.process.start()

    def get(self):
        try:
            return self.buffer.get(False) # Don't block on empty
        except Empty:
            return None

    def run(self, source, source2, running, buffer):
        cap = cv2.VideoCapture(source)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('m', 'j', 'p', 'g'))
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])

        cap2 = cv2.VideoCapture(source2)
        cap2.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('m', 'j', 'p', 'g'))
        cap2.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
        cap2.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        lost_frames_num = 0
        while running.value:
            # print(f"Running is {running.value}")
            ret, frame = cap.read()
            timestamp1 = round(time.time() - self.camera_start, TIMESTAMP_RESOLUTION)
            ret2, frame2 = cap2.read()
            timestamp2 = round(time.time() - self.camera_start, TIMESTAMP_RESOLUTION)
            if not ret or not ret2: # No more readable frames
                break
            while self.buffer.qsize() > 20:
                continue
            buffer.put((timestamp1, frame, timestamp2, frame2)) # This will block if we can't consume fast enough and the buffer is not infinite

        print(f"Running is {running.value}")

    def stop(self):
        self.running.value = False

    def kill(self): # A bit messy, make termination cleaner
        if self.running.value:
            self.stop()

        self.process.terminate()
        del self.buffer

# def read(stream):
#     fourcc = cv2.VideoWriter_fourcc(*'MJPG')

#     if RECORD_VIDEO:
#         recorder = cv2.VideoWriter('videos/0.avi', fourcc, FRAMERATE, RESOLUTION)

#         chronology = []
#     else:
#         with open('videos/chronology.json') as file:
#             chronology = json.load(file)
#     try:
#         while True:
#             data = stream.get()

#             if data:
#                 time, frame = data
#                 if RECORD_VIDEO:
#                     recorder.write(frame)
#                     chronology.append(time)
#     except KeyboardInterrupt:
#         stream.kill()

#     if stream.running.value:
#         stream.kill()
    
#     cv2.destroyAllWindows()

#     recorder.release()

#     if RECORD_VIDEO:
#         with open('videos/chronology.json', mode = 'w') as file:
#             json.dump(chronology, file)

#     sys.exit()