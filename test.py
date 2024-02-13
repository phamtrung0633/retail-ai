import serial.tools.list_ports
import time
WINDOW_LENGTH = 3
CALIBRATION_WEIGHT = 1000
PORT_NUMBER = 0
BAUDRATE = 38400
THRESHOLD = 1000
id_num = 0



def write_read(x):
    message = x + "\n"
    serialInst.write(message.encode('utf-8'))
    return None
serialInst = serial.Serial()
ports = serial.tools.list_ports.comports()

portList = []
for onePort in ports:
    portList.append(str(onePort))
    print(str(onePort))

val = PORT_NUMBER

for x in range(0, len(portList)):
    if portList[x].startswith("/dev/ttyUSB" + str(val)):
        portVal = "/dev/ttyUSB" + str(val)

serialInst.baudrate = BAUDRATE
serialInst.port = portVal
serialInst.open()

# Calibrate the scale
while True:
    if serialInst.in_waiting:
        packet = serialInst.readline()
        print(packet.decode('utf'))
        time.sleep(5)
        num = str(CALIBRATION_WEIGHT)
        write_read(num)
        break
count = 0
start_time = time.time()
while True:
    if count == 100:
        end_time = time.time()
        break
    if serialInst.in_waiting:
        packet = serialInst.readline()
        value = float(packet.decode('utf')[:-2])
        print(value)
        count += 1

print(count / (end_time - start_time))