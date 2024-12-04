import cv2
import dlib
import math
import time
import csv
from collections import deque
from scipy.spatial import distance as dist
import sounddevice as sd
import soundfile as sf
import mediapipe as mp
import numpy as np
import os

# EAR Calculation Function
def calculate_ear(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# MOR Calculation Function
def calculate_mor(landmarks):
    vertical = dist.euclidean(landmarks[51], landmarks[57])  # Vertical distance
    horizontal = dist.euclidean(landmarks[48], landmarks[54])  # Horizontal distance
    mor = vertical / horizontal if horizontal != 0 else 0
    return mor

# Head Tilt Calculation Function
def calculate_tilt_angle(landmarks):
    nose = landmarks[30]  # Nose tip
    chin = landmarks[8]   # Chin
    dx = chin[0] - nose[0]
    dy = chin[1] - nose[1]
    angle_radians = math.atan2(dy, dx)
    angle_degrees = math.degrees(angle_radians)
    return angle_degrees

# Check if hand is covering the mouth
def is_mouth_covered(landmarks, hand_landmarks, frame_shape):
    h, w, _ = frame_shape
    # Define mouth bounding box based on landmarks
    mouth_x_min = min([landmarks[i][0] for i in range(48, 54)])
    mouth_x_max = max([landmarks[i][0] for i in range(48, 54)])
    mouth_y_min = min([landmarks[i][1] for i in range(51, 57)])
    mouth_y_max = max([landmarks[i][1] for i in range(51, 57)])

    mouth_box = (mouth_x_min, mouth_y_min, mouth_x_max, mouth_y_max)

    # Iterate through hand landmarks and check overlap
    for hand_landmark in hand_landmarks:
        hand_points = [
            (int(hand_landmark.landmark[i].x * w), int(hand_landmark.landmark[i].y * h))
            for i in range(21)
        ]

        # Calculate hand bounding box
        hand_x_min = min([point[0] for point in hand_points])
        hand_x_max = max([point[0] for point in hand_points])
        hand_y_min = min([point[1] for point in hand_points])
        hand_y_max = max([point[1] for point in hand_points])

        hand_box = (hand_x_min, hand_y_min, hand_x_max, hand_y_max)

        # Check for bounding box overlap
        if (
            hand_box[0] < mouth_box[2]
            and hand_box[2] > mouth_box[0]
            and hand_box[1] < mouth_box[3]
            and hand_box[3] > mouth_box[1]
        ):
            return True
    return False

# Mediapipe Hands Initialization
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# Initialize Dlib's face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("C:/Users/brain/Desktop/VIGILO_DEPLOY/models/shape_predictor_68_face_landmarks.dat") #ADJUST WHERE YOUR MODEL IS

# Create audio folder
audio_folder = "Audio"
os.makedirs(audio_folder, exist_ok=True)

# Fatigue Parameters
EAR_THRESHOLD = 0.23
MOR_THRESHOLD = 0.45
PERCLOS_THRESHOLD = 50  # Percentage
FOM_THRESHOLD = 20
TILT_THRESHOLD = 10     # Degrees
YAWN_THRESHOLD = 0.6    # Yawning Threshold
YAWN_DURATION_THRESHOLD = 1.25  # seconds
FATIGUE_DURATION = 3    # Seconds to trigger fatigue warning
TIME_WINDOW = 30        # Sliding window size (seconds)
FPS = 15                # Approximate frames per second
NO_FACE_DETECTED_THRESHOLD = 5  # Seconds before flagging "No Face Detected"

# Sliding windows for metrics
closed_frames = deque(maxlen=TIME_WINDOW * FPS)
mouth_open_counts = deque(maxlen=TIME_WINDOW * FPS)
tilt_durations = deque(maxlen=TIME_WINDOW * FPS)
yawn_in_progress = False
yawn_start_time = None
fatigue_start_time = None
last_face_detected_time = time.time()  # Initialize to current time

# Audio recording callback
def audio_callback(indata, frames, time, status):
    global audio_frames
    if status:
        print(status)
    audio_frames.append(indata.copy())

# Initialize CSV File
csv_file_path = "fatigue_metrics.csv"
file_exists = os.path.isfile(csv_file_path)

csv_file = open(csv_file_path, "a", newline="")
csv_writer = csv.writer(csv_file)

if not file_exists:
    csv_writer.writerow(["Timestamp", "EAR", "PERCLOS", "MOR", "FOM", "Tilt Angle", "Fatigue Detected"])

# Timer for logging per second
last_logged_time = time.time()

# Capture video from webcam
cap = cv2.VideoCapture(1)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        faces = detector(gray)

        fatigue_status = "No"
        hand_covering_mouth = False
        face_detected = len(faces) > 0  # Check if any face is detected

        if face_detected:
            last_face_detected_time = time.time()  # Update the last detected time

        # Detect hands covering mouth if face is detected
        hand_covering_mouth = False  # Initialize the flag
        if face_detected and results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )  # Draw hand landmarks on the frame

                # Extract hand bounding box
                h, w, _ = frame.shape
                hand_x_min = int(min(hand_landmarks.landmark, key=lambda lm: lm.x).x * w)
                hand_x_max = int(max(hand_landmarks.landmark, key=lambda lm: lm.x).x * w)
                hand_y_min = int(min(hand_landmarks.landmark, key=lambda lm: lm.y).y * h)
                hand_y_max = int(max(hand_landmarks.landmark, key=lambda lm: lm.y).y * h)

                # Draw bounding box
                cv2.rectangle(frame, (hand_x_min, hand_y_min), (hand_x_max, hand_y_max), (0, 255, 0), 2)

            for face in faces:
                shape = predictor(gray, face)
                landmarks = [(shape.part(i).x, shape.part(i).y) for i in range(68)]
                hand_covering_mouth = is_mouth_covered(landmarks, results.multi_hand_landmarks, frame.shape)

                # Add warning if hand is covering the mouth
                if hand_covering_mouth:
                    # Display warning to the user
                    cv2.putText(frame, "HAND COVERING MOUTH DETECTED!", 
                                (20, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        # Process each detected face for fatigue metrics
        for face in faces:
            shape = predictor(gray, face)
            landmarks = [(shape.part(i).x, shape.part(i).y) for i in range(68)]

            # EAR Calculation
            left_eye = landmarks[36:42]
            right_eye = landmarks[42:48]
            left_ear = calculate_ear(left_eye)
            right_ear = calculate_ear(right_eye)
            ear = (left_ear + right_ear) / 2.0
            closed = ear < EAR_THRESHOLD
            closed_frames.append(closed)

            # PERCLOS Calculation
            perclos = sum(closed_frames) / len(closed_frames) * 100 if closed_frames else 0

            # MOR Calculation
            if not hand_covering_mouth:
                mor = calculate_mor(landmarks)
            else:
                mor = mor if 'mor' in locals() else 0  # Freeze MOR when mouth is covered

            mouth_open = mor > MOR_THRESHOLD
            mouth_open_counts.append(mouth_open)

            # FOM Calculation
            fom = sum(mouth_open_counts)  # Frequency of open mouth events

            # Head Tilt Calculation
            tilt_angle = calculate_tilt_angle(landmarks)
            tilt_detected = abs(tilt_angle) > TILT_THRESHOLD
            tilt_durations.append(tilt_detected)

            # Sustained Tilt Duration
            sustained_tilt_duration = sum(tilt_durations) / FPS
            
            # Yawn Detection
            if mor > YAWN_THRESHOLD and not hand_covering_mouth:
                if not yawn_in_progress:
                    yawn_in_progress = True
                    yawn_start_time = time.time()
                    audio_frames = []
                    fs = 44100
                    audio_stream = sd.InputStream(samplerate=fs, channels=1, callback=audio_callback)
                    audio_stream.start()
            else:
                if yawn_in_progress:
                    yawn_in_progress = False
                    if audio_stream:
                        audio_stream.stop()
                        audio_stream.close()
                    if time.time() - yawn_start_time > YAWN_DURATION_THRESHOLD:
                        audio_data = np.concatenate(audio_frames, axis=0)
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        filename = os.path.join(audio_folder, f"Audio_{timestamp}.wav")
                        sf.write(filename, audio_data, fs)

            # Combine Fatigue Conditions
            if (perclos > PERCLOS_THRESHOLD and
                fom > FOM_THRESHOLD and
                sustained_tilt_duration > FATIGUE_DURATION and
                mouth_open and
                closed):
                fatigue_status = "Yes"

            # Visualize Metrics
            cv2.putText(frame, f"EAR: {ear:.2f}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"PERCLOS: {perclos:.2f}%", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, f"MOR: {mor:.2f}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"FOM: {fom} events", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            cv2.putText(frame, f"Tilt: {tilt_angle:.2f} deg", (20, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Fatigue: {fatigue_status}", (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if fatigue_status == "Yes" else (0, 255, 0), 2)

            # Draw Landmarks
            for (x, y) in left_eye + right_eye + landmarks[48:60]:  # Eyes and mouth
                cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)
            cv2.circle(frame, landmarks[30], 5, (0, 255, 0), -1)    # Nose tip
            cv2.circle(frame, landmarks[8], 5, (0, 255, 0), -1)     # Chin

        # Check for "No Face Detected"
        current_time = time.time()
        if (current_time - last_face_detected_time) > NO_FACE_DETECTED_THRESHOLD:
            fatigue_status = "No Face Detected"
            cv2.putText(frame, "NO FACE DETECTED!", (20, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            
        # Log Metrics to CSV
        current_time = time.time()
        if current_time - last_logged_time >= 1:  # Log every second
            if face_detected:  # Face is detected
                # Determine obstruction status as "Yes" or "No"
                obstruction_status = "Yes" if hand_covering_mouth else "No"

                # Determine the overall status
                if yawn_in_progress:  # Yawning detected
                    overall_status = "Yawning Detected"
                else:  # Fatigue or normal status
                    overall_status = fatigue_status

                # Write to CSV
                csv_writer.writerow([
                    time.strftime("%Y-%m-%d %H:%M:%S"),  # Timestamp
                    f"{ear:.2f}" if face_detected else "N/A",    # EAR metric or N/A
                    f"{perclos:.2f}" if face_detected else "N/A",  # PERCLOS metric or N/A
                    f"{mor:.2f}" if face_detected and not hand_covering_mouth else "Obstructed",  # MOR or Obstructed
                    fom if face_detected else "N/A",    # Frequency of mouth open
                    tilt_angle if face_detected else "N/A",  # Head tilt angle
                    overall_status,  # Fatigue, Normal, or Yawning
                    obstruction_status  # Obstruction status as Yes/No
                ])
                print(f"Logged to CSV: {time.strftime('%Y-%m-%d %H:%M:%S')}, Status: {overall_status}, Obstruction: {obstruction_status}")
            else:  # No face detected
                csv_writer.writerow([
                    time.strftime("%Y-%m-%d %H:%M:%S"),  # Timestamp
                    "N/A",  # EAR
                    "N/A",  # PERCLOS
                    "N/A",  # MOR
                    "N/A",  # FOM
                    "N/A",  # Tilt Angle
                    "No Face Detected",  # Status
                    "N/A"  # Obstruction
                ])
                print("Logged to CSV: No Face Detected")
            csv_file.flush()  # Ensure data is written immediately
            last_logged_time = current_time

        # Show frame
        cv2.imshow("ViGILO", frame)

        # Exit with 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    csv_file.close()  # Ensure the file is closed
    cap.release()
    cv2.destroyAllWindows()
