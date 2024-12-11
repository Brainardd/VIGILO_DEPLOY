import cv2
import dlib
import math
import numpy as np
from tensorflow.keras.models import load_model
import mediapipe as mp
from scipy.spatial import distance as dist
from collections import deque
import time
import librosa
import joblib
import sounddevice as sd

# Initialize audio variables
audio_sr = 16000  # Audio sampling rate
audio_buffer = deque(maxlen=audio_sr)  # Buffer to hold 1 second of audio

# Global variables
audio_chunk = None

# Audio callback function
def audio_callback(indata, frames, time, status):
    global audio_buffer
    if status:
        print(f"Audio callback status: {status}")
    # Append audio data to the buffer
    audio_buffer.extend(indata[:, 0])

# Start audio stream
audio_stream = sd.InputStream(callback=audio_callback, samplerate=audio_sr, channels=1)
audio_stream.start()
print("[INFO] Audio stream started...")

# Load the trained models
yawn_model = load_model('models/yawn_detection_model.h5')
audio_model = joblib.load("models/audio_classification_model.pkl")

# Dlib Initialization
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

# Mediapipe Initialization
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# Fatigue Detection Parameters
EAR_THRESHOLD = 0.23
MOR_THRESHOLD = 0.55
PERCLOS_THRESHOLD = 70
FOM_THRESHOLD = 5
YAWN_CONFIDENCE_THRESHOLD = 0.08

# Initialize tracking variables
ear_values = deque(maxlen=30)  # Store the last 30 EAR values
mouth_open_counts = deque(maxlen=60)
yawn_state = {"state": "Not Yawning", "confidence": 0.0}

# Define utility functions (same as in the provided code)
def calculate_ear(eye):  # EAR calculation
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def calculate_mor(landmarks):  # MOR calculation
    vertical = dist.euclidean(landmarks[51], landmarks[57])
    horizontal = dist.euclidean(landmarks[48], landmarks[54])
    return vertical / horizontal if horizontal != 0 else 0

def calculate_tilt_angle(landmarks):  # Tilt calculation
    nose = landmarks[30]
    chin = landmarks[8]
    dx = chin[0] - nose[0]
    dy = chin[1] - nose[1]
    angle_radians = math.atan2(dy, dx)
    return math.degrees(angle_radians)

def calculate_perclos(ear_values, threshold=0.23):  # PERCLOS calculation
    if not ear_values:
        return 0.0
    closed_frames = sum(1 for ear in ear_values if ear < threshold)
    return (closed_frames / len(ear_values)) * 100

def predict_live_audio(audio, sr):  # Predict audio fatigue
    if np.sqrt(np.mean(audio**2)) < 0.02:  # Silence threshold
        return "neutral"
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc.T, axis=0).reshape(1, -1)
    prediction = audio_model.predict(mfcc_mean)
    return prediction[0]

def process_frame(frame):  # Fatigue detection logic
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    fatigue_status = "No Fatigue"
    metrics = {
        "EAR": "N/A", "MOR": "N/A", "Tilt Angle": "N/A",
        "PERCLOS": "N/A", "FOM": "N/A", "Fatigue": fatigue_status,
        "Yawning": "N/A", "Yawning Confidence": "N/A",
        "Audio Fatigue": "No Audio",
    }

    if len(faces) > 0:
        for face in faces:
            shape = predictor(gray, face)
            landmarks = [(shape.part(i).x, shape.part(i).y) for i in range(68)]

            # Calculate fatigue metrics
            ear = calculate_ear(landmarks[36:42]) + calculate_ear(landmarks[42:48])
            metrics["EAR"] = ear
            ear_values.append(ear)
            metrics["PERCLOS"] = calculate_perclos(ear_values, EAR_THRESHOLD)
            metrics["MOR"] = calculate_mor(landmarks)
            metrics["Tilt Angle"] = calculate_tilt_angle(landmarks)

            # Audio fatigue detection
            if len(audio_buffer) >= audio_sr:
                audio_chunk = np.array(audio_buffer, dtype=np.float32)
                metrics["Audio Fatigue"] = predict_live_audio(audio_chunk, audio_sr)
            else:
                print("[DEBUG] Not enough audio data for processing.")

    return metrics

# Main loop to capture video and process frames
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("[ERROR] Unable to access the camera.")
    exit()

print("[INFO] Video stream started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Unable to read frame from the camera.")
        break

    # Process the current frame
    metrics = process_frame(frame)

    # Display the fatigue detection metrics on the frame
    for idx, (key, value) in enumerate(metrics.items()):
        cv2.putText(frame, f"{key}: {value}", (10, 30 + idx * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # Show the frame
    cv2.imshow("Fatigue Detection", frame)

    # Exit condition
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
audio_stream.stop()
print("[INFO] Fatigue detection terminated.")
