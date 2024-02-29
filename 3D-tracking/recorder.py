import ctypes
import json
import time
import cv2
import os

from multiprocessing import Process, Queue, Value
from queue import Empty
from uuid import uuid4

from stream import Stream

TIMESTAMP_RESOLUTION = 3

FRAMERATE = 15
RESOLUTION = (640, 480)

MAX_WEIGHTS = 0
RECORD_WEIGHT = True
def gather_weights(running, buffer):
    import serial.tools.list_ports

    conn = serial.Serial()


    BAUDRATE = 38400

    def write_read(x):
        conn.write((x + "\n").encode('utf-8'))
        return None

    # comports = serial.tools.list_ports.comports()
    # ports = [str(port) for port in comports]

    port = "/dev/ttyUSB0"

    conn.baudrate = BAUDRATE
    conn.port = port
    conn.open()

    # Scale calibration

    while True:
        if conn.in_waiting:
            packet = conn.readline()
            print(packet.decode('utf'))
            break

    while running.value:
        if conn.in_waiting:
            packet, timestamp = conn.readline(), time.time()
            value = packet.decode('utf')[:-2]
            buffer.put((timestamp, value))

class Manifest:

    MANIFEST_FOLDER = 'recordings'

    def __init__(self, name = None, videos = None, chronfile = None, bake = None):
        self.name = name

        if not name:
            name = str(uuid4()).split()[1]

        self.videos = videos
        self.chronfile = chronfile

        if chronfile:
            with open(chronfile) as infile:
                self.chronology = json.load(infile)

        self.bake = bake

        if not bake:
            self.bake = {
                'exists': False
            }

    def streams(self):
        return Stream(
            self.pathto(self.videos['L']), # Source 1
            self.pathto(self.videos['R']), # Source 2
            self.chronology['start'] # Shared Start
        )

    def pathto(self, resource):
        return os.path.join(MANIFEST_FOLDER, self.name, resource)

    @staticmethod
    def load(path):
        with open(path) as infile:
            return Manifest(**json.load(infile))

class ManifestEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, Manifest):
            output = {
                'videos': obj.videos,
                'chronfile': obj.chronfile,
                'bake': {
                    'exists': obj.bake['exists'],
                    # 'range': obj.bake['range']
                }
            }

            if obj.bake.exists:
                output['bake']['file'] = obj.bake['file']

            return output

if __name__ == "__main__":
    start = round(time.time(), TIMESTAMP_RESOLUTION)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    recorders = [
        cv2.VideoWriter('videos/16.avi', fourcc, FRAMERATE, RESOLUTION),
        cv2.VideoWriter('videos/17.avi', fourcc, FRAMERATE, RESOLUTION)
    ]

    chronology = {
        'start': start,
        'weights': {},
        'frames': []
    }

    running = Value(ctypes.c_bool, True)
    if RECORD_WEIGHT:
        buffer = Queue(MAX_WEIGHTS)
        weights = Process(target = gather_weights, args = (running, buffer))
        weights.start()
    caps = Stream(3, 6, start, RESOLUTION)
    caps.start()

    try:
        while True:
            data = caps.get()

            if data:
                tl, fl, tr, fr = data
                
                chronology['frames'].append((tl, tr))
                recorders[0].write(fl)
                recorders[1].write(fr)
            if RECORD_WEIGHT:
                try:
                    t, w = buffer.get()
                    chronology['weights'][t] = w
                except Empty:
                    continue
    except KeyboardInterrupt:
        print("Saved recording!")

    running.value = False
    caps.stop()

    while not caps.buffer.empty():
        tl, fl, tr, fr = caps.get()

        chronology['frames'].append((tl, tr))
        recorders[0].write(fl)
        recorders[1].write(fr)
    if RECORD_WEIGHT:
        while not buffer.empty():
            t, w = buffer.get()

            chronology['weights'][t] = w

        weights.join()
        weights.terminate()

    caps.kill()
    for recorder in recorders:
        recorder.release()

    with open('videos/chronology9.json', mode = 'w') as out:
        json.dump(chronology, out)
