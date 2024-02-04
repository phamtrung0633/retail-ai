import ctypes
import json
import time
import cv2
from multiprocessing import Process, Queue, Value
from queue import Empty

from stream import Stream

TIMESTAMP_RESOLUTION = 3

FRAMERATE = 30
RESOLUTION = (1920, 1080)

MAX_WEIGHTS = 0
RECORD_WEIGHT = False
def gather_weights(running, buffer):
    import serial.tools.list_ports

    conn = serial.Serial()

    CALIBRATION_WEIGHT = 1000
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
            time.sleep(5)
            write_read(str(CALIBRATION_WEIGHT))
            break

    while running.value:
        if conn.in_waiting:
            packet, timestamp = conn.readline(), round(time.time(), TIMESTAMP_RESOLUTION)
            weight = float(packet.decode('utf')[:-2])
            buffer.put((timestamp, weight))

if __name__ == "__main__":
    start = round(time.time(), TIMESTAMP_RESOLUTION)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    recorders = [
        cv2.VideoWriter('videos/14.avi', fourcc, FRAMERATE, RESOLUTION),
        cv2.VideoWriter('videos/15.avi', fourcc, FRAMERATE, RESOLUTION)
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
    caps = Stream(2, 4, start, RESOLUTION)
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

    with open('videos/chronology7.json', mode = 'w') as out:
        json.dump(chronology, out)
