
import cv2
import mediapipe as mp
import math
import time

mp_face_mesh = mp.solutions.face_mesh
facemesh  = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5,refine_landmarks = True)


def calculate_left_eye_height(face_landmarks, inter_pupillary_distance):
    
    left_eye_top = face_landmarks.landmark[159]
    left_eye_bottom = face_landmarks.landmark[145]  
    left_eye_height = abs(left_eye_top.y - left_eye_bottom.y)
    normalized_left_eye_height = left_eye_height / inter_pupillary_distance

    return normalized_left_eye_height


def calculate_mouth_metrics(face_landmarks, inter_pupillary_distance):
    w_left = face_landmarks.landmark[61]
    w_right = face_landmarks.landmark[291]
    dist_width = math.sqrt((w_left.x - w_right.x)**2 + (w_left.y - w_right.y)**2)

    h_top = face_landmarks.landmark[0]
    h_bottom = face_landmarks.landmark[17]
    dist_height = math.sqrt((h_top.x - h_bottom.x)**2 + (h_top.y - h_bottom.y)**2)

    normalized_width_mouth = dist_width / inter_pupillary_distance
    normalized_height_mouth = dist_height / inter_pupillary_distance

    return normalized_width_mouth, normalized_height_mouth

def calculate_cheek_metrics(face_landmarks, inter_pupillary_distance):
    left_cheek = face_landmarks.landmark[50]
    right_cheek = face_landmarks.landmark[280]
    dist_cheek = math.sqrt((left_cheek.x - right_cheek.x)**2 + (left_cheek.y - right_cheek.y)**2)

    normalized_cheek_distance = dist_cheek / inter_pupillary_distance

    return normalized_cheek_distance

def calculate_eyebrow_metrics(face_landmarks, inter_pupillary_distance):

    left_eyebrow_inner = face_landmarks.landmark[55]
    right_eyebrow_inner = face_landmarks.landmark[285] 

    eyebrow_inner_distance = math.sqrt((left_eyebrow_inner.x - right_eyebrow_inner.x)**2 + (left_eyebrow_inner.y - right_eyebrow_inner.y)**2)

    normalized_eyebrow_eye_dist = eyebrow_inner_distance / inter_pupillary_distance

    return normalized_eyebrow_eye_dist


# Since everyone's facial metrics are different, this function is designed to calculate and establish neutral facial metrics for each individual.
# This will be our baseline facial dimensions
def calibration(duration=10):
    cap = cv2.VideoCapture(0)
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
            inter_pupillary_distance = math.sqrt((eye_left.x - eye_right.x)**2 + (eye_left.y - eye_right.y)**2)

            normalized_width_mouth, normalized_height_mouth = calculate_mouth_metrics(face_landmarks, inter_pupillary_distance)
            normalized_cheek_distance = calculate_cheek_metrics(face_landmarks, inter_pupillary_distance)
            normalized_eyebrows = calculate_eyebrow_metrics(face_landmarks,inter_pupillary_distance)
            normalized_eye_height = calculate_left_eye_height(face_landmarks, inter_pupillary_distance)

            neutral_widths.append(normalized_width_mouth)
            neutral_heights.append(normalized_height_mouth)
            neutral_cheeks.append(normalized_cheek_distance)
            neutral_eyebrows.append(normalized_eyebrows)
            neutral_eye_heights.append(normalized_eye_height)

            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.putText(frame, f'Calibrating...', (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Calibration', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

    
    return sum(neutral_widths) / len(neutral_widths), sum(neutral_heights) / len(neutral_heights),sum(neutral_cheeks) / len(neutral_cheeks),sum(neutral_eyebrows) / len(neutral_eyebrows),sum(neutral_eye_heights) / len(neutral_eye_heights)


def inference():

    neutral_width, neutral_height,neutral_cheek,neutral_eyebrows,neutral_eye_height = calibration(10)

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
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
            inter_pupillary_distance = math.sqrt((eye_left.x - eye_right.x)**2 + (eye_left.y - eye_right.y)**2)

            normalized_width_mouth, normalized_height_mouth = calculate_mouth_metrics(face_landmarks, inter_pupillary_distance)
            normalized_cheek_distance = calculate_cheek_metrics(face_landmarks, inter_pupillary_distance)
            normalized_eyebrows = calculate_eyebrow_metrics(face_landmarks,inter_pupillary_distance)
            normalized_eye_height = calculate_left_eye_height(face_landmarks, inter_pupillary_distance)

            if normalized_width_mouth > neutral_width * 1.1 and normalized_cheek_distance > neutral_cheek * 1.02:
                expression = "Smile"
            elif normalized_height_mouth > neutral_height * 1.7 and normalized_eye_height > neutral_eye_height * 1.03:
                expression = "Surprise"
            elif normalized_eyebrows < neutral_eyebrows * 0.98:  # Slight decrease in eyebrow distance to detect sadness
                expression = "Sad"
            else:
                expression = "Neutral"

            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.putText(frame, f'Expression: {expression}', (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Mouth Expression Inference', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    inference()





