import socket

s = socket.socket()
s.bind(('172.20.10.3', 6069))
s.listen(0)


def perform_calibration():
    # remotely set the calibration value
    print("Starting Calibration")
    calibration_data = float(input("Enter calibration weight value: "))
    return calibration_data


try:
    print("Server is up and running. Waiting for connections...")

    while True:
        print("Server Attempting to connect")
        client, addr = s.accept()
        try:
            print(f"Connected to {addr}")
            print(f"Calibration")

            calibration = perform_calibration()
            client.sendall(str(calibration).encode())
            print("Value sent to client:", calibration)

            # Open a file for storing received data
            with open('received_data.txt', 'a') as file:
                buffer = b''
                while True:

                    content = client.recv(1024)
                    if not content:
                        break

                    buffer += content

                    while b'\n' in buffer:

                        line, buffer = buffer.split(b'\n', 1)
                        line = line.strip()
                        if line:
                            decoded_data = line.decode('utf-8')
                            print(f"Received data: {decoded_data}")
                            # Store the received data into the file
                            file.write(decoded_data + '\n')
                            file.flush()  # Flush buffer as soon as written

        except Exception as e:
            print(f"Error during communication: {e}")
        finally:
            client.close()

except Exception as e:
    print(f"Error: {e}")
finally:
    s.close()