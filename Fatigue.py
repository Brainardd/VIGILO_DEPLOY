import cv2
import dlib
import math
from scipy.spatial import distance as dist

# EAR Calculation Function
def calculate_ear(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# MOR Calculation Function
def calculate_mor(landmarks):
    vertical = dist.euclidean(landmarks[51], landmarks[57])  # Vertical distance
    horizontal = dist.euclidean(landmarks[48], landmarks[54])  # Horizontal distance
    return vertical / horizontal if horizontal != 0 else 0

# Head Tilt Calculation Function
def calculate_tilt_angle(landmarks):
    nose = landmarks[30]
    chin = landmarks[8]
    dx = chin[0] - nose[0]
    dy = chin[1] - nose[1]
    angle_radians = math.atan2(dy, dx)
    return math.degrees(angle_radians)

# Dlib Initialization
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

# Fatigue Detection Parameters
EAR_THRESHOLD = 0.23
MOR_THRESHOLD = 0.45
TILT_THRESHOLD = 10

# Frame Processing Function
def process_frame(frame, closed_frames, mouth_open_counts, yawn_state):
    """
    Process a single frame to detect fatigue metrics.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    fatigue_status = "No Fatigue"
    metrics = {"EAR": "N/A", "MOR": "N/A", "Tilt Angle": "N/A", "Fatigue": fatigue_status}
    landmarks_to_draw = None

    if len(faces) > 0:
        for face in faces:
            shape = predictor(gray, face)
            landmarks = [(shape.part(i).x, shape.part(i).y) for i in range(68)]

            # EAR Calculation
            left_eye = landmarks[36:42]
            right_eye = landmarks[42:48]
            ear = (calculate_ear(left_eye) + calculate_ear(right_eye)) / 2.0
            metrics["EAR"] = ear

            # MOR Calculation
            mor = calculate_mor(landmarks)
            metrics["MOR"] = mor

            # Tilt Angle Calculation
            tilt_angle = calculate_tilt_angle(landmarks)
            metrics["Tilt Angle"] = tilt_angle

            # Fatigue Detection
            if ear < EAR_THRESHOLD or mor > MOR_THRESHOLD or abs(tilt_angle) > TILT_THRESHOLD:
                fatigue_status = "Fatigue Detected"
            metrics["Fatigue"] = fatigue_status

            landmarks_to_draw = landmarks

    return fatigue_status, metrics, landmarks_to_draw


# Updated Function to overlay metrics and facial outlines
def draw_metrics_on_frame(frame, metrics, fatigue_status, landmarks=None):
    """
    Draws fatigue detection metrics and highlights specific facial landmarks for visualization.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7

    if landmarks:
        # Highlight specific facial landmarks
        for (x, y) in landmarks[36:42] + landmarks[42:48] + landmarks[48:60]:  # Eyes and mouth
            cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)  # Small blue circles for eyes and mouth
        cv2.circle(frame, landmarks[30], 5, (0, 255, 0), -1)  # Larger green circle for nose tip
        cv2.circle(frame, landmarks[8], 5, (0, 255, 0), -1)   # Larger green circle for chin

    return frame
