import ctypes
import time
import json
import sys

from multiprocessing import Process, Queue, Value
from queue import Empty

import cv2

TIMESTAMP_RESOLUTION = 2
MAX_FRAMES = 0 # Infinite

FRAMERATE = 30
RESOLUTION = (640, 480)

RECORD_VIDEO = True

class Stream:

    def __init__(self, source, time):
        self.buffer = Queue(MAX_FRAMES)
        self.running = Value(ctypes.c_bool, True)
        self.start_time = time
        self.process = Process(target = self.run, args = (source, self.running, self.buffer))

    def start(self):
        self.process.start()

    def get(self):
        try:
            return self.buffer.get(False) # Don't block on empty
        except Empty:
            return None

    def run(self, source, running, buffer):
        cap = cv2.VideoCapture(source)

        while running.value:
            # print(f"Running is {running.value}")
            ret, frame = cap.read()

            if not ret: # No more readable frames
                break

            timestamp = round(time.time() - self.start_time, TIMESTAMP_RESOLUTION)
            buffer.put((timestamp, frame)) # This will block if we can't consume fast enough and the buffer is not infinite

        # print(f"Running is {running.value}")

    def kill(self): # A bit messy, make termination cleaner
        self.running.value = False

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