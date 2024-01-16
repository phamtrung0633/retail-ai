import serial.tools.list_ports
import time
import numpy as np

serialInst = serial.Serial()

WINDOW_LENGTH = 3
SHARED_TIMER = time.time()
CALIBRATION_WEIGHT = 1000
PORT_NUMBER = 6
BAUDRATE = 38400
THRESHOLD = 200

class WeightEvent:
    def __init__(self, start_time):
        self.start_time = start_time
        self.start_value = None
        self.end_value = None
        self.end_time = float('inf')

    def set_start_val(self, value):
        self.start_value = value

    def set_end_val(self, value):
        self.end_value = value

    def set_end_time(self, val):
        self.end_time = val


    def get_weight_change(self):
        return self.end_value - self.start_value


def write_read(x):
    message = x + "\n"
    serialInst.write(message.encode('utf-8'))
    time.sleep(0.05)
    data = serialInst.readline()
    return data


def calculate_moving_variance(values):
    values = np.array(values)
    variance = np.var(values)
    return variance


ports = serial.tools.list_ports.comports()

portList = []
for onePort in ports:
    portList.append(str(onePort))
    print(str(onePort))

val = PORT_NUMBER

for x in range(0, len(portList)):
    if portList[x].startswith("COM" + str(val)):
        portVal = "COM" + str(val)

serialInst.baudrate = BAUDRATE
serialInst.port = portVal
serialInst.open()
count = 0

# Calibrate the scale
while True:
    if serialInst.in_waiting:
        packet = serialInst.readline()
        print(packet.decode('utf'))
        time.sleep(5)
        num = str(CALIBRATION_WEIGHT)
        value = write_read(num)
        break

events_shared_list = []
weight_buffer = []
current_event = None
while True:
    if serialInst.in_waiting:
        packet, time_packet = serialInst.readline(), time.time()
        print(packet.decode('utf'))
        weight_value = float(packet.decode('utf')[:-2])
        if len(weight_buffer) < WINDOW_LENGTH:
            weight_buffer.append(weight_value)
        else:
            del weight_buffer[0]
            weight_buffer.append(weight_value)
            moving_variance = calculate_moving_variance(weight_buffer)
            if moving_variance >= THRESHOLD and current_event is None:
                current_event = WeightEvent(time.time() - SHARED_TIMER)
                current_event.set_start_val(weight_buffer[2])
            elif moving_variance < THRESHOLD and current_event is not None:
                current_event.set_end_time(time.time() - SHARED_TIMER)
                current_event.set_end_val(weight_buffer[0])
                events_shared_list.append(current_event)
                current_event = None






