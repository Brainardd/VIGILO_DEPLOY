from flask import Flask, render_template, Response, jsonify, send_file
import cv2
import mediapipe as mp
import csv
import os
import time
from Fatigue import process_frame, draw_metrics_on_frame, is_mouth_covered, calculate_mor
from collections import deque

app = Flask(__name__)

# Initialize camera and variables
camera = cv2.VideoCapture(1)
closed_frames = deque(maxlen=10)
mouth_open_counts = deque(maxlen=10)
yawn_state = {"in_progress": False, "start_time": 0, "detected": 0}
last_frame_metrics = {"EAR": "N/A", "MOR": "N/A", "Tilt Angle": "N/A", "Fatigue": "No"}
log_file_path = "metrics_log.csv"  # File to store logged metrics
last_log_time = 0

# Write CSV header if the log file doesn't exist
if not os.path.exists(log_file_path):
    with open(log_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "EAR", "MOR", "Tilt Angle", "PERCLOS", "FOM", "Fatigue", "Obstructed"])

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# Global variable for the last valid MOR value
last_valid_mor = None

def gen_frames():
    global last_frame_metrics, last_valid_mor  # Make last_valid_mor global
    last_logged_time = 0  # Track the last time a log was written
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            try:
                # Process the frame for fatigue detection
                fatigue_status, metrics, landmarks = process_frame(frame, closed_frames, mouth_open_counts, yawn_state)

                # Hand detection
                results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                hand_covering_mouth = False

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        if not hand_landmarks or not hand_landmarks.landmark:
                            continue  # Skip if hand landmarks are invalid

                        mp.solutions.drawing_utils.draw_landmarks(
                            frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                        )
                    # Check if hand is covering the mouth
                    hand_covering_mouth = is_mouth_covered(landmarks, results.multi_hand_landmarks, frame.shape)

                # Update fatigue status if hand is covering mouth
                if hand_covering_mouth:
                    fatigue_status = "Hand Covering Mouth"
                    print("Hand covering mouth detected. Keeping MOR constant.")
                elif landmarks:
                    # If the hand is not covering the mouth, calculate and update the MOR
                    mor = calculate_mor(landmarks)
                    last_valid_mor = mor  # Store the latest valid MOR value

                # If hand is covering the mouth, use the last valid MOR value
                if hand_covering_mouth and last_valid_mor is not None:
                    mor = last_valid_mor
                elif not landmarks:
                    mor = 0.0  # Default MOR when no face is detected

                # Update metrics
                metrics["MOR"] = mor
                metrics["Fatigue"] = fatigue_status

                # Log metrics to CSV every second
                current_time = time.time()
                if current_time - last_logged_time >= 1:  # Log every second
                    face_detected = metrics["EAR"] != "N/A"  # Detect if a face is visible
                    obstruction_status = "Yes" if hand_covering_mouth else "No"  # Is the mouth obstructed?
                    
                    # Determine overall status
                    if yawn_state["in_progress"]:  # Yawning is ongoing
                        overall_status = "Yawning Detected"
                    else:
                        overall_status = fatigue_status

                    with open(log_file_path, mode='a', newline='') as file:
                        writer = csv.writer(file)
                        if face_detected:
                            # Log metrics when a face is detected
                            writer.writerow([
                                time.strftime("%Y-%m-%d %H:%M:%S"),  # Timestamp
                                f"{metrics['EAR']:.2f}",             # EAR
                                f"{metrics['PERCLOS']:.2f}",         # PERCLOS
                                f"{metrics['MOR']:.2f}" if not hand_covering_mouth else "Obstructed",  # MOR or Obstructed
                                metrics["FOM"],                      # FOM
                                metrics["Tilt Angle"],               # Tilt Angle
                                overall_status,                      # Fatigue, Normal, or Yawning
                                obstruction_status                   # Obstruction status
                            ])
                            print(f"Logged: {time.strftime('%Y-%m-%d %H:%M:%S')}, Status: {overall_status}, Obstruction: {obstruction_status}")
                        else:
                            # Log "No Face Detected"
                            writer.writerow([
                                time.strftime("%Y-%m-%d %H:%M:%S"),
                                "N/A", "N/A", "N/A", "N/A", "N/A",
                                "No Face Detected", "N/A"
                            ])
                            print("Logged: No Face Detected")
                    last_logged_time = current_time

                # Update metrics and overlay on the frame
                last_frame_metrics = metrics
                frame = draw_metrics_on_frame(frame, metrics, fatigue_status, landmarks)

            except Exception as e:
                # Log the error and continue processing
                print(f"Error processing frame: {e}")
                continue

            # Encode the frame for streaming
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            # Stream the frame
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/metrics')
def metrics():
    global last_frame_metrics
    return jsonify({"metrics": last_frame_metrics})

@app.route('/download_logs')
def download_logs():
    """
    Allow the user to download the current log file.
    """
    try:
        return send_file(log_file_path, as_attachment=True)
    except Exception as e:
        return jsonify({"error": f"Error downloading file: {e}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
