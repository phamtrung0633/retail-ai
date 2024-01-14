import ctypes
import time

from multiprocessing import Process, Queue, Value
from queue import Empty

from cv2 import VideoCapture

TIMESTAMP_RESOLUTION = 2
MAX_FRAMES = 0 # Infinite

class Stream:

    def __init__(self, source):
        self.buffer = Queue(MAX_FRAMES)
        self.running = Value(ctypes.c_bool, True)

        self.process = Process(target = self.run, args = (source, self.running, self.buffer))

    def start(self):
        self.process.start()

    def get(self):
        try:
            return self.buffer.get(False) # Don't block on empty
        except Empty:
            return None

    def run(self, source, running, buffer):
        cap = VideoCapture(source)

        while running.value:
            # print(f"Running is {running.value}")
            ret, frame = cap.read()

            if not ret: # No more readable frames
                break

            timestamp = round(time.time(), TIMESTAMP_RESOLUTION)
            buffer.put((timestamp, frame)) # This will block if we can't consume fast enough and the buffer is not infinite

        # print(f"Running is {running.value}")

    def kill(self): # A bit messy, make termination cleaner
        self.running.value = False

        self.process.terminate()
        del self.buffer

# def read(stream):
#     try:
#         while True:
#             data = stream.get()

#             if data:
#                 time, frame = data
#                 print(f"Read frame from {time}")
#                 imshow("Camera", frame)

#                 if waitKey(1) == 13:
#                     break    
#     except KeyboardInterrupt:
#         stream.kill()

#     stream.kill()
#     destroyAllWindows()