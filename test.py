import serial.tools.list_ports
import numpy as np
import time
import json
class WeightEvent:
    def __init__(self, start_time, event_id):
        self.id = event_id
        self.start_time = start_time
        self.start_value = None
        self.end_value = None
        self.end_time = float('inf')

    def get_id(self):
        return self.id

    def set_start_val(self, value):
        self.start_value = value

    def get_start_time(self):
        return self.start_time

    def get_end_time(self):
        return self.end_time

    def set_end_val(self, value):
        self.end_value = value

    def set_end_time(self, val):
        self.end_time = val

    def get_weight_change(self):
        return self.end_value - self.start_value

WINDOW_LENGTH = 3
PORT_NUMBER = 0
BAUDRATE = 38400
THRESHOLD = 1000
id_num = 0

weight_buffer_1 = []
weight_buffer_2 = []
weight_buffers = [weight_buffer_1, weight_buffer_2]
current_events = [None, None]
trigger_counters_start = [0, 0]

def calculate_moving_variance(values):
    values = np.array(values)
    variance = np.var(values)
    return variance


with open('3D-tracking/videos/chronology9.json') as file:
    chronology = json.load(file)
SHARED_TIME = chronology['start']
chronology = chronology['weights']

for event in chronology:
    timestamp, value_read = float(event) - SHARED_TIME, chronology[event]
    weight_sensor_num = int(value_read[0]) - 1
    value = float(value_read[1:])
    if len(weight_buffers[weight_sensor_num]) < WINDOW_LENGTH:
        weight_buffers[weight_sensor_num].append([value, timestamp])
    else:
        del weight_buffers[weight_sensor_num][0]
        weight_buffers[weight_sensor_num].append([value, timestamp])
        w = np.array(weight_buffers[weight_sensor_num])
        moving_variance = calculate_moving_variance(w[:, 0])
        if moving_variance >= THRESHOLD and current_events[weight_sensor_num] is None:
            if trigger_counters_start[weight_sensor_num] == 1:
                current_events[weight_sensor_num] = WeightEvent(weight_buffers[weight_sensor_num][1][1], id_num)
                id_num += 1
                current_events[weight_sensor_num].set_start_val(weight_buffers[weight_sensor_num][0][0])
                trigger_counters_start[weight_sensor_num] = 0
                #print(w[:, 0])
            elif trigger_counters_start[weight_sensor_num] == 0:
                trigger_counters_start[weight_sensor_num] += 1
        elif moving_variance < THRESHOLD and current_events[weight_sensor_num] is not None:
            current_events[weight_sensor_num].set_end_time(weight_buffers[weight_sensor_num][1][1])
            current_events[weight_sensor_num].set_end_val(weight_buffers[weight_sensor_num][1][0])
            print(f"Weight event starts from {current_events[weight_sensor_num].get_start_time()} ends at {current_events[weight_sensor_num].get_end_time()} with change {current_events[weight_sensor_num].get_weight_change()}")
            current_events[weight_sensor_num] = None
            trigger_counters_start[weight_sensor_num] = 0
            #print(w[:, 0])
        else:
            trigger_counters_start[weight_sensor_num] = 0