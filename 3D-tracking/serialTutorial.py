import serial.tools.list_ports
import time
import numpy as np
from embeddings.embedder import Embedder
from scipy.optimize import linear_sum_assignment
serialInst = serial.Serial()

WINDOW_LENGTH = 3
SHARED_TIMER = time.time()
CALIBRATION_WEIGHT = 1000
PORT_NUMBER = 6
BAUDRATE = 38400
THRESHOLD = 100

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
serialInst.port = "COM7"
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
weights_event_count = 0
while True:
    if weights_event_count > 2:
        break
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
                print(weight_buffer)
                current_event = WeightEvent(time.time() - SHARED_TIMER)
                current_event.set_start_val(weight_buffer[1])
            elif moving_variance < THRESHOLD and current_event is not None:
                print(weight_buffer)
                current_event.set_end_time(time.time() - SHARED_TIMER)
                current_event.set_end_val(weight_buffer[1])
                events_shared_list.append(current_event)
                weights_event_count += 1
                current_event = None

embedder = Embedder()
embedder.initialise()
product_list = embedder.get_products("shelf_1")
A_prod_weight = np.zeros((len(product_list), len(events_shared_list)))
for i, weight_event in enumerate(events_shared_list):
    weight_change = abs(weight_event.get_weight_change())
    sum_val = 0
    for j, product in enumerate(product_list):
        dist_between_prod_and_change = abs(weight_change - product['weight'])
        sum_val += 1 / dist_between_prod_and_change

    for j, product in enumerate(product_list):
        dist_between_prod_and_change = abs(weight_change - product['weight'])
        A_prod_weight[j, i] = (1 / dist_between_prod_and_change) / sum_val

indices_products, indices_events = linear_sum_assignment(A_prod_weight, maximize=True)
count = 2
for id_prod, id_event in zip(indices_products, indices_events):
    print("Weight change is:", events_shared_list[id_event].get_weight_change())
    if events_shared_list[id_event].get_weight_change() < 0:
        print(f"Event number {count} takes {product_list[id_prod]['sku']}")
    else:
        print(f"Event number {count} put back {product_list[id_prod]['sku']}")
    count -= 1

# Hungarian Algorithm only works 1 mapping to 1
