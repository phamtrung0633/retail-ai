import ctypes
import time

from multiprocessing import Process, Queue, Value
from queue import Empty

from cv2 import VideoCapture

TIMESTAMP_RESOLUTION = 2
MAX_FRAMES = 0 # Infinite

class Stream:

    def __init__(self, source, source2, time):
        self.buffer = Queue(MAX_FRAMES)
        self.running = Value(ctypes.c_bool, True)
        self.start_time = time
        self.process = Process(target = self.run, args = (source, source2, self.running, self.buffer))

    def start(self):
        self.process.start()

    def get(self):
        try:
            return self.buffer.get(False) # Don't block on empty
        except Empty:
            return None

    def run(self, source, source2, running, buffer):
        cap = VideoCapture(source)
        cap2 = VideoCapture(source2)
        while running.value:
            # print(f"Running is {running.value}")
            ret, frame = cap.read()
            timestamp1 = round(time.time() - self.start_time, TIMESTAMP_RESOLUTION)
            ret2, frame2 = cap2.read()
            timestamp2 = round(time.time() - self.start_time, TIMESTAMP_RESOLUTION)
            if not ret or not ret2: # No more readable frames
                break

            buffer.put((timestamp1, frame, timestamp2, frame2)) # This will block if we can't consume fast enough and the buffer is not infinite

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