import ctypes
import json
import time

from multiprocessing import Process, Queue, Value
from queue import Empty

from stream import Stream

TIMESTAMP_RESOLUTION = 3

FRAMERATE = 30
RESOLUTION = (640, 480)

MAX_WEIGHTS = 0

def gather_weights(running, buffer):
    import serial.tools.list_ports

    conn = serial.Serial()

    CALIBRATION_WEIGHT = 1000
    PORT_NUMBER = 7
    BAUDRATE = 38400
    THRESHOLD = 200

    def write_read(x):
        conn.write((x + "\n").encode('utf-8'))
        time.sleep(0.05)
        
        return conn.readline()

    # comports = serial.tools.list_ports.comports()
    # ports = [str(port) for port in comports]

    port = f"COM{PORT_NUMBER}"

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

    recorders = [
        cv2.VideoWriter('videos/0.avi', fourcc, FRAMERATE, RESOLUTION),
        cv2.VideoWriter('videos/1.avi', fourcc, FRAMERATE, RESOLUTION)
    ]

    chronology = {
        'start': start
        'weights': {}
        'frames': []
    }

    running = Value(ctypes.c_bool, True)
    buffer = Queue(MAX_WEIGHTS)

    caps = Stream(0, 2, camera_start)
    weights = Process(target = gather_weights, args = (running, buffer))

    caps.start()
    weights.start()

    try:
        while True:
            data = caps.get()

            if data:
                tl, fl, tr, fr = data
                
                chronology['frames'].append((tl, tr))
                recorders[0].write(fl)
                recorders[1].write(fr)

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

    while not buffer.empty():
        t, w = buffer.get()

        chronology['weights'][t] = w

    weights.join()
    weights.terminate()

    caps.kill()

    for recorder in recorders:
        recorder.release()

    with open('videos/chronology.json', mode = 'w') as out:
        json.dump(chronology, out)
