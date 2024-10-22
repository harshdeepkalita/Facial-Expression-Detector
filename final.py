import cv2
import mediapipe as mp
import math
import time
from collections import Counter
from threading import Thread
import socket
import tkinter as tk
import threading
from tkinter import messagebox

# Buffer settings
buffer_size = 10  # Number of frames to aggregate expressions
expression_buffer = []  # Buffer to hold recent expressions
cooldown_time = 5  # Cooldown period in seconds
last_command_time = 0  # Timestamp for when the last command was sent
busy = False  # Flag to indicate if ESP32 is processing a command

mp_face_mesh = mp.solutions.face_mesh
facemesh = mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5, min_tracking_confidence=0.5, refine_landmarks=True
)

# ESP32 IP and port configuration
esp_ip = "192.168.248.71"
port = 80


# Web server communication setup
def send_to_esp(message):
    global busy

    # Check if ESP32 is already processing a command
    if busy:
        print(f"ESP32 is busy processing another command: {message} will be ignored.")
        return

    # Set the busy flag to True before sending the command
    busy = True

    def send_message():
        global busy
        try:
            # Create a socket connection
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(5)  # Set a timeout to avoid indefinite waiting
            s.connect((esp_ip, port))

            # Prepare and send the message
            message_with_newline = message + "\n"
            s.sendall(message_with_newline.encode())

            # Receive the response from ESP32 (assuming ESP32 sends a response when done)
            response = s.recv(1024).decode()
            print(f"Response from ESP32: {response}")

            # Reset the busy flag after the command has been processed
            busy = False

        except socket.timeout:
            print("Connection timed out. ESP32 might not be responding.")
            busy = False  # Reset the busy flag in case of failure
        except ConnectionRefusedError:
            print(
                "Connection refused by ESP32. Make sure the ESP32 is running and reachable."
            )
            busy = False  # Reset the busy flag in case of failure
        except Exception as e:
            print(f"An error occurred: {e}")
            busy = False  # Reset the busy flag in case of failure
        finally:
            s.close()

    # Run the socket communication in a separate thread to avoid blocking
    Thread(target=send_message).start()


def calculate_left_eye_height(face_landmarks, inter_pupillary_distance):
    left_eye_top = face_landmarks.landmark[159]
    left_eye_bottom = face_landmarks.landmark[145]
    left_eye_height = abs(left_eye_top.y - left_eye_bottom.y)
    normalized_left_eye_height = left_eye_height / inter_pupillary_distance

    return normalized_left_eye_height


def calculate_mouth_metrics(face_landmarks, inter_pupillary_distance):
    w_left = face_landmarks.landmark[61]
    w_right = face_landmarks.landmark[291]
    dist_width = math.sqrt((w_left.x - w_right.x) ** 2 + (w_left.y - w_right.y) ** 2)

    h_top = face_landmarks.landmark[0]
    h_bottom = face_landmarks.landmark[17]
    dist_height = math.sqrt((h_top.x - h_bottom.x) ** 2 + (h_top.y - h_bottom.y) ** 2)

    normalized_width_mouth = dist_width / inter_pupillary_distance
    normalized_height_mouth = dist_height / inter_pupillary_distance

    return normalized_width_mouth, normalized_height_mouth


def calculate_cheek_metrics(face_landmarks, inter_pupillary_distance):
    left_cheek = face_landmarks.landmark[50]
    right_cheek = face_landmarks.landmark[280]
    dist_cheek = math.sqrt(
        (left_cheek.x - right_cheek.x) ** 2 + (left_cheek.y - right_cheek.y) ** 2
    )

    normalized_cheek_distance = dist_cheek / inter_pupillary_distance

    return normalized_cheek_distance


def calculate_eyebrow_metrics(face_landmarks, inter_pupillary_distance):
    left_eyebrow_inner = face_landmarks.landmark[107]
    right_eyebrow_inner = face_landmarks.landmark[336]

    eyebrow_inner_distance = math.sqrt(
        (left_eyebrow_inner.x - right_eyebrow_inner.x) ** 2
        + (left_eyebrow_inner.y - right_eyebrow_inner.y) ** 2
    )

    normalized_eyebrow_eye_dist = eyebrow_inner_distance / inter_pupillary_distance

    return normalized_eyebrow_eye_dist


# Since everyone's facial metrics are different, this function is designed to calculate and establish neutral facial metrics for each individual.
# This will be our baseline facial dimensions
def calibration(duration=2):
    cap = cv2.VideoCapture(1)
    neutral_widths = []
    neutral_heights = []
    neutral_cheeks = []
    neutral_eyebrows = []
    neutral_eye_heights = []
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or time.time() - start_time > duration:
            break
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img.flags.writeable = False
        results = facemesh.process(img)
        img.flags.writeable = True
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]

            eye_left = face_landmarks.landmark[33]
            eye_right = face_landmarks.landmark[263]
            inter_pupillary_distance = math.sqrt(
                (eye_left.x - eye_right.x) ** 2 + (eye_left.y - eye_right.y) ** 2
            )

            normalized_width_mouth, normalized_height_mouth = calculate_mouth_metrics(
                face_landmarks, inter_pupillary_distance
            )
            normalized_cheek_distance = calculate_cheek_metrics(
                face_landmarks, inter_pupillary_distance
            )
            normalized_eyebrows = calculate_eyebrow_metrics(
                face_landmarks, inter_pupillary_distance
            )
            normalized_eye_height = calculate_left_eye_height(
                face_landmarks, inter_pupillary_distance
            )

            neutral_widths.append(normalized_width_mouth)
            neutral_heights.append(normalized_height_mouth)
            neutral_cheeks.append(normalized_cheek_distance)
            neutral_eyebrows.append(normalized_eyebrows)
            neutral_eye_heights.append(normalized_eye_height)

            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.putText(
                frame,
                f"Calibrating...",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
            cv2.imshow("Calibration", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()

    return (
        sum(neutral_widths) / len(neutral_widths),
        sum(neutral_heights) / len(neutral_heights),
        sum(neutral_cheeks) / len(neutral_cheeks),
        sum(neutral_eyebrows) / len(neutral_eyebrows),
        sum(neutral_eye_heights) / len(neutral_eye_heights),
    )


# Initialize the counter and threshold for sad expression detection
sad_counter = 0
sad_threshold = 30
running = False


def inference():
    global sad_counter, last_command_time, expression_buffer, busy, running
    running = True
    (
        neutral_width,
        neutral_height,
        neutral_cheek,
        neutral_eyebrows,
        neutral_eye_height,
    ) = calibration(2)

    cap = cv2.VideoCapture(1)

    while cap.isOpened() and running:
        ret, frame = cap.read()
        if not ret:
            break
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img.flags.writeable = False
        results = facemesh.process(img)
        img.flags.writeable = True

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]

            eye_left = face_landmarks.landmark[33]
            eye_right = face_landmarks.landmark[263]
            inter_pupillary_distance = math.sqrt(
                (eye_left.x - eye_right.x) ** 2 + (eye_left.y - eye_right.y) ** 2
            )

            normalized_width_mouth, normalized_height_mouth = calculate_mouth_metrics(
                face_landmarks, inter_pupillary_distance
            )
            normalized_cheek_distance = calculate_cheek_metrics(
                face_landmarks, inter_pupillary_distance
            )
            normalized_eye_height = calculate_left_eye_height(
                face_landmarks, inter_pupillary_distance
            )

            expression = "Neutral"

            if (
                normalized_width_mouth > neutral_width * 1.1
                and normalized_cheek_distance > neutral_cheek * 1.02
            ):
                expression = "Smile"
                sad_counter = 0  # Reset the sad counter
            elif (
                normalized_height_mouth > neutral_height * 1.7
                and normalized_eye_height > neutral_eye_height * 1.03
            ):
                expression = "Surprise"
                sad_counter = 0  # Reset the sad counter
            elif normalized_eye_height < neutral_eye_height * 0.98:
                sad_counter += 1
                if sad_counter > sad_threshold:
                    expression = "Sad"
            else:
                expression = "Neutral"
                sad_counter = 0  # Reset the sad counter

            # Add the detected expression to the buffer
            expression_buffer.append(expression)

            # If buffer exceeds size, remove the oldest expression
            if len(expression_buffer) > buffer_size:
                expression_buffer.pop(0)

            # If enough time has passed and buffer is full, send the most common expression
            current_time = time.time()
            if current_time - last_command_time >= cooldown_time and not busy:
                # Get the most common expression from the buffer
                most_common_expression = Counter(expression_buffer).most_common(1)[0][0]

                # Send the most common expression to ESP32
                send_to_esp(most_common_expression)

                # Update the last command time and clear the buffer
                last_command_time = current_time
                expression_buffer.clear()

            # Display the expression on the frame
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.putText(
                frame,
                f"Expression: {expression}",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 0, 255),
                2,
            )
            cv2.imshow("Mouth Expression Inference", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


def start_inference():
    global running
    if not running:
        threading.Thread(target=inference).start()


def reset_program():
    global running, sad_counter
    running = False
    sad_counter = 0
    messagebox.showinfo("Reset", "The program has been reset.")


# Tkinter GUI setup
root = tk.Tk()
root.title("Facial Expression Detector")

# Create a label to display text
label = tk.Label(
    root, font=3, text=" Didn't get much time to make the GUI look pretty NGL :("
)
label.pack(pady=10)

start_button = tk.Button(
    root, text="Start Inference", width=10, height=5, command=start_inference
)
start_button.pack(pady=10)

reset_button = tk.Button(root, text="Reset", width=10, height=5, command=reset_program)
reset_button.pack(pady=10)

root.mainloop()
