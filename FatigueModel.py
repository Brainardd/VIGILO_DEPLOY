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

# Load the trained models
yawn_model = load_model('models/yawn_detection_model.h5')
audio_model = joblib.load("models/audio_classification_model.pkl")

# Dlib Initialization
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

# Hand Cover Function
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# Fatigue Detection Parameters
EAR_THRESHOLD = 0.23
MOR_THRESHOLD = 0.55
MOR_THRESHOLD_MODEL = 10
TILT_THRESHOLD = 10
PERCLOS_THRESHOLD = 70  # Percentage threshold for closed eyes
FOM_THRESHOLD = 5       # Number of mouth openings in a time window (e.g., 1 minute)
YAWN_CONFIDENCE_THRESHOLD = 0.08

audio_buffer = deque(maxlen=16000)  # Buffer to hold 1 second of audio (assuming 16kHz)
sr = 16000  # Sampling rate for audio

# Initialize tracking variables
ear_values = deque(maxlen=30)  # Store the last 30 EAR values for PERCLOS
mouth_open_counts = deque(maxlen=60)  # Store the time of mouth openings for FOM
previous_mor = 0

def calculate_ear(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def calculate_mor(landmarks):
    vertical = dist.euclidean(landmarks[51], landmarks[57])
    horizontal = dist.euclidean(landmarks[48], landmarks[54])
    return vertical / horizontal if horizontal != 0 else 0

def calculate_perclos(ear_values, threshold=0.23):
    if not ear_values:
        return 0.0
    closed_frames = sum(1 for ear in ear_values if ear < threshold)
    return (closed_frames / len(ear_values)) * 100

def preprocess_mouth(mouth_region):
    mouth_resized = cv2.resize(mouth_region, (128, 128))
    mouth_normalized = mouth_resized / 255.0
    mouth_input = np.expand_dims(mouth_normalized, axis=0)
    return mouth_input

# Head Tilt Calculation Function
def calculate_tilt_angle(landmarks):
    nose = landmarks[30]
    chin = landmarks[8]
    dx = chin[0] - nose[0]
    dy = chin[1] - nose[1]
    angle_radians = math.atan2(dy, dx)
    return math.degrees(angle_radians)

def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    audio_buffer.extend(indata[:, 0])  # Add audio samples to the buffer

def predict_live_audio(audio):
    audio = np.array(audio)
    if np.sqrt(np.mean(audio**2)) < 0.03:  # Silence threshold
        return "neutral"
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc.T, axis=0).reshape(1, -1)
    prediction = audio_model.predict(mfcc_mean)
    return prediction[0]

def process_frame(frame, ear_values, mouth_open_counts, yawn_state, is_mouth_covered, audio_chunk=None):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    metrics = {"EAR": "N/A", "MOR": "N/A", "PERCLOS": "N/A", "FOM": "N/A", "Tilt Angle": "N/A", "Audio Fatigue": "N/A", "Yawning" : "N/A", "Yawning Confidence" : "N/A"}
    fatigue_status = "No Fatigue"
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
            mor = calculate_mor(landmarks) if landmarks else 0.0
            metrics["MOR"] = mor

            # FOM Calculation
            if mor > MOR_THRESHOLD:
                current_time = time.time()
                if not mouth_open_counts or current_time - mouth_open_counts[-1] > 0.5:
                    mouth_open_counts.append(current_time)
            fom = len([t for t in mouth_open_counts if time.time() - t <= 60])
            metrics["FOM"] = fom

            # Tilt Angle Calculation
            tilt_angle = calculate_tilt_angle(landmarks)
            metrics["Tilt Angle"] = tilt_angle

            # Check if the mouth is covered by a hand
            hand_landmarks_results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            hand_landmarks = hand_landmarks_results.multi_hand_landmarks

            is_hand_covering_mouth = is_mouth_covered(landmarks, hand_landmarks, frame.shape)

            # Detect Yawning with Model
            mouth_region, found = extract_mouth_region(frame)
            if mouth_region is not None and not is_hand_covering_mouth:
                mouth_input = preprocess_mouth(mouth_region)
                prediction = yawn_model.predict(mouth_input)
                yawn_confidence = prediction[0][0]
                if yawn_confidence > YAWN_CONFIDENCE_THRESHOLD and mor > MOR_THRESHOLD:
                    yawn_state["state"] = "Yawning"
                    yawn_state["confidence"] = yawn_confidence
                else:
                    yawn_state["state"] = "Not Yawning"
                    yawn_state["confidence"] = yawn_confidence
            elif is_hand_covering_mouth:
                print("Mouth is covered. Falling back to audio-based yawning detection.")
                if audio_chunk is not None:
                    try:
                        # Use audio to detect yawning
                        audio_fatigue = predict_live_audio(audio_chunk)
                        metrics["Audio Fatigue"] = audio_fatigue
                        if audio_fatigue == "yawning":
                            print("Audio confirmed yawning.")
                            yawn_state["state"] = "Yawning"
                            yawn_state["confidence"] = 1.0
                        else:
                            yawn_state["state"] = "Not Yawning"
                            yawn_state["confidence"] = 0.0
                    except Exception as e:
                        print(f"Error in audio fatigue detection: {e}")
                        yawn_state["state"] = "Error"
                        yawn_state["confidence"] = 0.0
                else:
                    yawn_state["state"] = "Mouth Covered"
                    yawn_state["confidence"] = 0.0

            metrics["Yawning"] = yawn_state["state"]
            metrics["Yawning Confidence"] = round(yawn_state["confidence"], 2)

            # Fatigue Detection Logic
            if yawn_state["state"] == "Yawning":
                if is_hand_covering_mouth:
                    # Disable MOR and FOM thresholds
                    if (
                        perclos > PERCLOS_THRESHOLD and
                        abs(tilt_angle) > TILT_THRESHOLD and
                        ear < EAR_THRESHOLD
                    ):
                        fatigue_status = "Fatigue Detected"
                    else:
                        fatigue_status = "Yawning (Metrics Low)"
                else:
                    # Use full metrics thresholds
                    if (
                        perclos > PERCLOS_THRESHOLD and
                        fom > FOM_THRESHOLD and
                        abs(tilt_angle) > TILT_THRESHOLD and
                        ear < EAR_THRESHOLD and
                        mor > MOR_THRESHOLD
                    ):
                        fatigue_status = "Fatigue Detected"
                    else:
                        fatigue_status = "No Fatigue"
            else:
                fatigue_status = "No Fatigue"

            metrics["Fatigue"] = fatigue_status
            landmarks_to_draw = landmarks
    else:
        metrics["Fatigue"] = "No Face Detected"

    return fatigue_status, metrics, landmarks_to_draw


def is_mouth_covered(landmarks, hand_landmarks, frame_shape):
    """
    Checks if the hand overlaps specifically with the mouth region.
    """
    h, w, _ = frame_shape
    if not landmarks or not hand_landmarks:
        return False

    # Define the mouth bounding box
    mouth_x_min = min(landmarks[i][0] for i in range(48, 60))
    mouth_x_max = max(landmarks[i][0] for i in range(48, 60))
    mouth_y_min = min(landmarks[i][1] for i in range(48, 60))
    mouth_y_max = max(landmarks[i][1] for i in range(48, 60))

    # Add padding
    padding = 10
    mouth_x_min -= padding
    mouth_x_max += padding
    mouth_y_min -= padding
    mouth_y_max += padding

    print("Mouth bounding box:", mouth_x_min, mouth_x_max, mouth_y_min, mouth_y_max)

    # Check for overlap between the hand and mouth bounding box
    for hand in hand_landmarks:
        for landmark in hand.landmark:
            hand_x = int(landmark.x * w)
            hand_y = int(landmark.y * h)
            if mouth_x_min <= hand_x <= mouth_x_max and mouth_y_min <= hand_y <= mouth_y_max:
                print("Hand point inside mouth bounding box.")
                return True

    return False

def detect_yawning(frame, landmarks, mouth_region, mor, hand_landmarks, audio_chunk=None, sr=None):
    """
    Detects yawning based on the mouth region and MOR.
    Handles cases where the mouth is covered, using audio as fallback.
    """
    try:
        # Check if the hand is covering the mouth
        if is_mouth_covered(landmarks, hand_landmarks, frame.shape):
            print("Hand detected covering the mouth.")
            if audio_chunk is not None and sr is not None:
                print(f"Audio chunk shape: {audio_chunk.shape}, Sampling rate: {sr}")
                print(f"Audio chunk preview: {audio_chunk[:10]}")
                try:
                    audio_fatigue = predict_live_audio(audio_chunk, sr)
                    if audio_fatigue == "yawning":
                        print("Audio detected yawning while mouth is covered.")
                        return "Yawning (Audio Based)", 1.0  # Assume high confidence for audio
                    else:
                        print("Audio did not detect yawning while mouth is covered.")
                        return "Not Yawning (Audio Based)", 0.0
                except Exception as e:
                    print(f"Error in audio yawning detection: {e}")
                    return "Error in Audio Detection", None
            else:
                print("Audio data not available while mouth is covered.")
                return "Mouth Covered, No Audio", None

        # If the mouth region is available, preprocess and predict
        if mouth_region is not None:
            try:
                mouth_input = preprocess_mouth(mouth_region)
                prediction = yawn_model.predict(mouth_input)
                confidence = prediction[0][0]

                if confidence > YAWN_CONFIDENCE_THRESHOLD and mor > MOR_THRESHOLD:
                    print(f"Yawning detected with confidence: {confidence}")
                    return "Yawning", confidence
                else:
                    print(f"Not yawning. Confidence: {confidence}")
                    return "Not Yawning", confidence
            except Exception as e:
                print(f"Error during model prediction: {e}")
                return "Error in Yawning Detection", None

        # If no mouth region is detected
        print("Mouth region not detected.")
        return "No Mouth Detected", None
    except Exception as e:
        print(f"Error in detect_yawning: {e}")
        return "Error", None

def extract_mouth_region(frame):
    """
    Extract the mouth region from a full face image.
    """
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        if len(faces) == 0:
            return None, False

        # Use the first detected face
        face = faces[0]
        landmarks = predictor(gray, face)

        # Get mouth landmarks (points 48 to 67)
        x_min = min(landmarks.part(i).x for i in range(48, 68))
        x_max = max(landmarks.part(i).x for i in range(48, 68))
        y_min = min(landmarks.part(i).y for i in range(48, 68))
        y_max = max(landmarks.part(i).y for i in range(48, 68))

        # Add padding
        padding = 10
        x_min = max(0, x_min - padding)
        x_max = min(frame.shape[1], x_max + padding)
        y_min = max(0, y_min - padding)
        y_max = min(frame.shape[0], y_max + padding)

        # Crop the mouth region
        mouth_region = frame[y_min:y_max, x_min:x_max]
        return mouth_region, True
    except Exception as e:
        print(f"Error extracting mouth region: {e}")
        return None, False
        
# Function to overlay metrics and facial outlines
def draw_metrics_on_frame(frame, metrics, fatigue_status, landmarks=None):
    """
    Draws fatigue detection metrics and highlights specific facial landmarks for visualization.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7

    # Draw fatigue status and metrics
    cv2.putText(frame, f"Fatigue Status: {metrics['Fatigue']}", (20, 50), font, 0.8, (0, 0, 255), 2)
    cv2.putText(frame, f"EAR: {metrics['EAR']:.2f}" if metrics["EAR"] != "N/A" else "EAR: N/A", (20, 80), font, font_scale, (255, 255, 255), 1)
    cv2.putText(frame, f"MOR: {metrics['MOR']:.2f}" if metrics["MOR"] != "N/A" else "MOR: N/A", (20, 110), font, font_scale, (255, 255, 255), 1)
    cv2.putText(frame, f"Tilt: {metrics['Tilt Angle']:.2f}°" if metrics["Tilt Angle"] != "N/A" else "Tilt: N/A", (20, 140), font, font_scale, (255, 255, 255), 1)
    cv2.putText(frame, f"PERCLOS: {metrics['PERCLOS']:.2f}%" if metrics["PERCLOS"] != "N/A" else "PERCLOS: N/A", (20, 170), font, font_scale, (255, 255, 255), 1)
    cv2.putText(frame, f"FOM: {metrics['FOM']}" if metrics["FOM"] != "N/A" else "FOM: N/A", (20, 200), font, font_scale, (255, 255, 255), 1)

    # Draw yawning metrics
    cv2.putText(frame, f"Yawning: {metrics['Yawning']}", (20, 230), font, font_scale, (255, 255, 0), 2)
    cv2.putText(frame, f"Yawning Confidence: {metrics['Yawning Confidence']:.2f}" if metrics["Yawning Confidence"] != "N/A" else "Yawning Confidence: N/A", (20, 260), font, font_scale, (255, 255, 0), 1)

    # Draw audio fatigue
    cv2.putText(frame, f"Audio Fatigue: {metrics['Audio Fatigue']}", (20, 290), font, font_scale, (0, 255, 255), 2)

    # Highlight specific facial landmarks if available
    if landmarks:
        for (x, y) in landmarks[36:42] + landmarks[42:48] + landmarks[48:60]:  # Eyes and mouth
            cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)  # Blue circles for eyes and mouth
        cv2.circle(frame, landmarks[30], 5, (0, 255, 0), -1)  # Green circle for nose tip
        cv2.circle(frame, landmarks[8], 5, (0, 255, 0), -1)   # Green circle for chin

    return frame

# Start video and audio streams
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("[ERROR] Unable to access the camera.")
    exit()

stream = sd.InputStream(callback=audio_callback, samplerate=sr, channels=1)
stream.start()

print("[INFO] Video and audio streams started. Press 'q' to quit.")

ear_values = deque(maxlen=30)
mouth_open_counts = deque()
yawn_state = {"state": "Not Yawning", "confidence": 0.0}  # Initialize yawn state

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Unable to read frame from the camera.")
        break

    # Process audio
    audio_chunk = list(audio_buffer)

    # Process the frame
    fatigue_status, metrics, landmarks_to_draw = process_frame(
        frame, ear_values, mouth_open_counts, yawn_state,
        is_mouth_covered,  # Ensure this is defined as a callable function
        audio_chunk
    )

    # Display the fatigue detection metrics on the frame
    frame = draw_metrics_on_frame(frame, metrics, fatigue_status, landmarks_to_draw)
    cv2.imshow("Fatigue Detection", frame)

    # Exit condition
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
stream.stop()
print("[INFO] Fatigue detection terminated.")