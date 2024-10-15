import socket

esp_ip = "192."
port = 80

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)


def send_to_esp(message):
    try:
        s.connect((esp_ip, port))

        message += "\n"
        s.sendall(message.encode())

        response = s.recv(1024).decode()
        print(f"Response from ESP32: {response}")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        s.close()


class HandDetectionApp:
    def __init__(self):
        send_to_esp("happy")


if __name__ == "__main__":
    HandDetectionApp()
