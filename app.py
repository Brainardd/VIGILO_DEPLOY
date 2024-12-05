from flask import Flask, render_template, Response, jsonify, request, send_file, session
import base64
import cv2
import csv
import os
import time
import numpy as np
import mediapipe as mp
from Fatigue import process_frame, draw_metrics_on_frame, is_mouth_covered, calculate_mor
from collections import deque
import uuid

app = Flask(__name__)
app.secret_key = "1234"  # Required for session management

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5
)
# Directory to store user-specific CSV files
CSV_DIR = "user_logs"
if not os.path.exists(CSV_DIR):
    os.makedirs(CSV_DIR)

# Initialize state variables
closed_frames = deque(maxlen=10)
mouth_open_counts = deque(maxlen=10)
yawn_state = {"in_progress": False, "start_time": 0, "detected": 0}
last_frame_metrics = {"EAR": "N/A", "MOR": "N/A", "Tilt Angle": "N/A", "Fatigue": "No"}
last_valid_mor = None  # To handle hand-over-mouth scenarios

# Write CSV header if the log file doesn't exist
log_file_path = "metrics_log.csv"
if not os.path.exists(log_file_path):
    with open(log_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "EAR", "MOR", "Tilt Angle", "PERCLOS", "FOM", "Fatigue", "Obstructed"])

@app.before_request
def assign_session_id():
    """
    Assign a unique session ID to each user if not already assigned.
    """
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())  # Generate a unique session ID
        session_csv_path = os.path.join(CSV_DIR, f"{session['session_id']}.csv")
        # Create a CSV file for this session
        with open(session_csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Timestamp", "EAR", "MOR", "Tilt Angle", "PERCLOS", "FOM", "Fatigue", "Hand Covering Mouth"])
        print(f"New session ID: {session['session_id']}, CSV Path: {session_csv_path}")

@app.route('/')
def index():
    print(f"Current session ID: {session.get('session_id')}")
    return render_template('index.html')


@app.route('/get_csv_filename', methods=['GET'])
def get_csv_filename():
    """
    Return the current session's CSV filename.
    """
    if 'session_id' in session:
        session_csv_path = os.path.join(CSV_DIR, f"{session['session_id']}.csv")
        if os.path.exists(session_csv_path):
            return jsonify({"csv_filename": session_csv_path})
        else:
            print("CSV file not found for session:", session['session_id'])
            return jsonify({"error": "CSV file does not exist"}), 404
    else:
        print("Session ID not found in request.")
        return jsonify({"error": "Session not initialized"}), 400
    
@app.route('/get_csv_updates', methods=['GET'])
def get_csv_updates():
    """
    Return the last 10 lines from the current session's CSV file.
    """
    if 'session_id' in session:
        session_csv_path = os.path.join(CSV_DIR, f"{session['session_id']}.csv")
        if os.path.exists(session_csv_path):
            try:
                with open(session_csv_path, 'r') as file:
                    lines = file.readlines()[-10:]  # Get the last 10 lines
                return jsonify({"lines": lines})
            except Exception as e:
                print(f"Error reading CSV file: {e}")
                return jsonify({"error": "Error reading CSV file"}), 500
        else:
            return jsonify({"error": "CSV file does not exist"}), 404
    else:
        return jsonify({"error": "Session not initialized"}), 400

@app.route('/process_frame', methods=['POST'])
def process_frame_endpoint():
    global last_frame_metrics, last_valid_mor

    data = request.get_json()
    frame_data = data['frame']

    # Decode the base64-encoded frame
    frame_bytes = base64.b64decode(frame_data.split(',')[1])
    np_arr = np.frombuffer(frame_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    try:
        # Process the frame for fatigue detection
        fatigue_status, metrics, landmarks = process_frame(frame, closed_frames, mouth_open_counts, yawn_state)

        # Hand detection with Mediapipe
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        hand_covering_mouth = False

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                if not hand_landmarks or not hand_landmarks.landmark:
                    continue
                # Check if the hand is covering the mouth
                hand_covering_mouth = is_mouth_covered(landmarks, results.multi_hand_landmarks, frame.shape)

        # Determine the MOR (Mouth Opening Ratio)
        if hand_covering_mouth:
            mor = last_valid_mor if last_valid_mor is not None else 0.0
        else:
            if landmarks:
                mor = calculate_mor(landmarks)
                last_valid_mor = mor
            else:
                mor = 0.0

        # Update fatigue metrics
        if mor > 0.5:
            fatigue_status = "Fatigue Detected While Yawning" if fatigue_status == "Fatigue Detected" else "Yawning Detected"
        elif fatigue_status == "Fatigue Detected":
            fatigue_status = "Fatigue Detected"
        else:
            fatigue_status = "No Fatigue"

        metrics["MOR"] = mor
        metrics["Fatigue"] = fatigue_status
        metrics["Hand Covering Mouth"] = "Yes" if hand_covering_mouth else "No"
        last_frame_metrics = metrics

        # Round all metrics to 2 decimal places
        for key, value in metrics.items():
            if isinstance(value, float):
                metrics[key] = round(value, 2)

        # Overlay metrics on the frame
        processed_frame = draw_metrics_on_frame(frame, metrics, fatigue_status, landmarks)

        # Save metrics to CSV
        session_csv_path = os.path.join(CSV_DIR, f"{session['session_id']}.csv")
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(session_csv_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                timestamp,
                metrics.get("EAR", "N/A"),
                metrics.get("MOR", "N/A"),
                metrics.get("Tilt Angle", "N/A"),
                metrics.get("PERCLOS", "N/A"),
                metrics.get("FOM", "N/A"),
                metrics.get("Fatigue", "N/A"),
                metrics.get("Hand Covering Mouth", "N/A")
            ])
            
            # Debug to ensure correct filename
        print(f"CSV filename sent to frontend: {session_csv_path}")

        # Encode the processed frame to Base64
        _, buffer = cv2.imencode('.jpg', processed_frame)
        frame_b64 = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            "metrics": metrics,
            "processed_frame": f"data:image/jpeg;base64,{frame_b64}",
            "csv_filename": session_csv_path  # Send filename back to frontend
        })

    except Exception as e:
        print(f"Error processing frame: {e}")
        return jsonify({"error": str(e)}), 500


    
@app.route('/download_logs', methods=['GET'])
def download_logs():
    """
    Allow the user to download their unique CSV file.
    """
    if 'session_id' in session:
        session_csv_path = os.path.join(CSV_DIR, f"{session['session_id']}.csv")
        if os.path.exists(session_csv_path):
            return send_file(session_csv_path, as_attachment=True, download_name=f"{session['session_id']}_logs.csv")
        else:
            return jsonify({"error": "CSV file does not exist"}), 404
    else:
        return jsonify({"error": "Session not initialized"}), 400



if __name__ == '__main__':
    app.run(debug=True)
