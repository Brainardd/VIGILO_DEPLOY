import cv2
import dlib
import math
import mediapipe as mp
from scipy.spatial import distance as dist
from collections import deque
import time

# Hand Cover Function
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

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

# PERCLOS Calculation Function
def calculate_perclos(ear_values, threshold=0.23):

    if not ear_values:
        return 0.0
    closed_frames = sum(1 for ear in ear_values if ear < threshold)
    return (closed_frames / len(ear_values)) * 100

# Dlib Initialization
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

# Fatigue Detection Parameters
EAR_THRESHOLD = 0.23
MOR_THRESHOLD = 0.45
TILT_THRESHOLD = 10
PERCLOS_THRESHOLD = 70  # Percentage threshold for closed eyes
FOM_THRESHOLD = 5       # Number of mouth openings in a time window (e.g., 1 minute)

# Initialize tracking variables
ear_values = deque(maxlen=30)  # Store the last 30 EAR values for PERCLOS
mouth_open_counts = deque(maxlen=60)  # Store the time of mouth openings for FOM

# Frame Processing Function
def process_frame(frame, closed_frames, mouth_open_counts, yawn_state):
    """
    Process a single frame to detect fatigue metrics.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    fatigue_status = "No Fatigue"
    metrics = {"EAR": "N/A", "MOR": "N/A", "Tilt Angle": "N/A", "PERCLOS": "N/A", "FOM": "N/A", "Fatigue": fatigue_status}
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
            ear_values.append(ear)

            # PERCLOS Calculation
            perclos = calculate_perclos(ear_values, EAR_THRESHOLD)
            metrics["PERCLOS"] = perclos
            
            # MOR Calculation
            mor = calculate_mor(landmarks)
            metrics["MOR"] = mor
            
            # FOM Calculation
            if mor > MOR_THRESHOLD:
                current_time = time.time()
                if not mouth_open_counts or current_time - mouth_open_counts[-1] > 0.5:  # At least 0.5 seconds apart
                    mouth_open_counts.append(current_time)
            fom = len([t for t in mouth_open_counts if time.time() - t <= 60])  # Count within the last minute
            metrics["FOM"] = fom

            # Tilt Angle Calculation
            tilt_angle = calculate_tilt_angle(landmarks)
            metrics["Tilt Angle"] = tilt_angle

            # Fatigue Detection
            if (perclos > PERCLOS_THRESHOLD and
                fom > FOM_THRESHOLD and
                abs(tilt_angle) > TILT_THRESHOLD and
                mor > MOR_THRESHOLD and
                ear < EAR_THRESHOLD):
                fatigue_status = "Fatigue Detected"
            else:
                fatigue_status = "No Fatigue"

            landmarks_to_draw = landmarks
    else:
        # Default behavior when no face is detected
        metrics["Fatigue"] = "No Face Detected"

    return fatigue_status, metrics, landmarks_to_draw


def is_mouth_covered(landmarks, hand_landmarks, frame_shape):
    """
    Checks if the hand overlaps specifically with the mouth region.
    """
    h, w, _ = frame_shape
    if not landmarks or not hand_landmarks:
        return False  # Return False if landmarks are missing

    # Define the mouth bounding box
    mouth_x_min = min(landmarks[i][0] for i in range(48, 60))
    mouth_x_max = max(landmarks[i][0] for i in range(48, 60))
    mouth_y_min = min(landmarks[i][1] for i in range(48, 60))
    mouth_y_max = max(landmarks[i][1] for i in range(48, 60))

    # Add a slight padding to the mouth bounding box for tolerance
    padding = 10  # Adjust padding as needed
    mouth_x_min -= padding
    mouth_x_max += padding
    mouth_y_min -= padding
    mouth_y_max += padding

    # Check for overlap between the hand and mouth bounding box
    for hand in hand_landmarks:
        if not hand.landmark:
            continue  # Skip invalid hand data

        # Convert hand landmarks to image coordinates
        for landmark in hand.landmark:
            hand_x = int(landmark.x * w)
            hand_y = int(landmark.y * h)

            # Check if hand point is inside the mouth bounding box
            if mouth_x_min <= hand_x <= mouth_x_max and mouth_y_min <= hand_y <= mouth_y_max:
                return True

    return False




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
        
        # Overlay fatigue status and metrics
        cv2.putText(frame, f"Fatigue: {metrics['Fatigue']}", (20, 50), font, 0.8, (0, 0, 255), 2)
        cv2.putText(frame, f"EAR: {metrics['EAR']:.2f}" if metrics["EAR"] != "N/A" else "EAR: N/A", (20, 80), font, font_scale, (255, 255, 255), 1)
        cv2.putText(frame, f"MOR: {metrics['MOR']:.2f}" if metrics["MOR"] != "N/A" else "MOR: N/A", (20, 110), font, font_scale, (255, 255, 255), 1)
        cv2.putText(frame, f"Tilt: {metrics['Tilt Angle']:.2f} deg" if metrics["Tilt Angle"] != "N/A" else "Tilt: N/A", (20, 140), font, font_scale, (255, 255, 255), 1)
        cv2.putText(frame, f"PERCLOS: {metrics['PERCLOS']:.2f}%" if metrics["PERCLOS"] != "N/A" else "PERCLOS: N/A", (20, 170), font, font_scale, (255, 255, 255), 1)
        cv2.putText(frame, f"FOM: {metrics['FOM']}" if metrics["FOM"] != "N/A" else "FOM: N/A", (20, 200), font, font_scale, (255, 255, 255), 1)

        
    else:
        # Display "No Face Detected" if no landmarks are available
        cv2.putText(frame, "NO FACE DETECTED", (20, 50),
                    font, 1, (0, 0, 255), 2)
        
    if fatigue_status == "Hand Covering Mouth":
        cv2.putText(frame, "HAND COVERING MOUTH DETECTED!", (20, 300),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    return frame
