from flask import Flask, render_template, Response, jsonify, request, send_file, session
import base64
import cv2
import csv
import os
import time
import numpy as np
import mediapipe as mp
from Fatigue import process_frame, draw_metrics_on_frame, is_mouth_covered, calculate_mor, predict_live_audio
from collections import deque
import uuid
import sounddevice as sd

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
yawn_state = {"state": "Not Yawning", "confidence": 0.0}
last_frame_metrics = {"EAR": "N/A", "MOR": "N/A", "Tilt Angle": "N/A", "Fatigue": "No", "Audio Fatigue": "No"}
last_valid_mor = None  # To handle hand-over-mouth scenarios
last_csv_write_time = 0

# Write CSV header if the log file doesn't exist
log_file_path = "metrics_log.csv"
if not os.path.exists(log_file_path):
    with open(log_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "EAR", "MOR", "Tilt Angle", "PERCLOS", "FOM", "Fatigue", "Yawning", "Yawning Confidence"])

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
            writer.writerow(["Timestamp", "EAR", "MOR", "Tilt Angle", "PERCLOS", "FOM", "Fatigue", "Audio Fatigue", "Yawning Confidence", "Yawning"])
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

@app.route('/process_audio', methods=['POST'])
def process_audio_endpoint():
    try:
        # Get audio data from the frontend
        data = request.get_json()
        audio_base64 = data.get('audio', '')

        # Decode Base64 to bytes
        audio_bytes = base64.b64decode(audio_base64)
        audio_array = np.frombuffer(audio_bytes, dtype=np.float32)

        # Ensure sample rate compatibility
        sr = 16000
        
        if len(audio_array) == 0:
            raise ValueError("Audio data is empty or corrupted.")

        audio_fatigue_status = predict_live_audio(audio_array, sr)

        return jsonify({
            "status": "success",
            "audio_fatigue_status": audio_fatigue_status
        })
    except Exception as e:
        print(f"Error processing audio: {e}")
        return jsonify({"error": str(e), "status": "failure"}), 500

    
@app.route('/process_frame', methods=['POST'])
def process_frame_endpoint():
    global last_frame_metrics, last_csv_write_time, last_valid_mor

    try:
        # Decode incoming frame
        data = request.get_json()
        frame_data = data.get('frame', '')
        frame_bytes = base64.b64decode(frame_data.split(',')[1])
        np_arr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Fatigue detection
        fatigue_status, metrics, landmarks = process_frame(
            frame, closed_frames, mouth_open_counts, yawn_state
        )

        # MOR logic (only landmarks, no hand detection)
        try:
            if landmarks and len(landmarks) >= 68:
                mor = calculate_mor(landmarks)
                last_valid_mor = mor  # Update valid MOR
            else:
                mor = 0.0
                print("No valid landmarks detected. Setting MOR to 0.0")
        except Exception as mor_error:
            print(f"Error calculating MOR: {mor_error}")
            mor = last_valid_mor if last_valid_mor is not None else 0.0

        metrics["MOR"] = round(mor, 2)

        # Serialize metrics for JSON response
        serializable_metrics = {
            key: float(value) if isinstance(value, (np.float32, np.float64)) else value
            for key, value in metrics.items()
        }

        
        # CSV Logging
        session_csv_path = os.path.join(CSV_DIR, f"{session['session_id']}.csv")
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        if time.time() - last_csv_write_time >= 1:
            with open(session_csv_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                
                # Format numeric values to 2 decimal places
                writer.writerow([
                    timestamp,
                    round(serializable_metrics.get("EAR", "N/A"), 2) if isinstance(serializable_metrics.get("EAR"), (int, float)) else "N/A",
                    round(serializable_metrics["MOR"], 2) if isinstance(serializable_metrics["MOR"], (int, float)) else "N/A",
                    round(serializable_metrics.get("Tilt Angle", "N/A"), 2) if isinstance(serializable_metrics.get("Tilt Angle"), (int, float)) else "N/A",
                    round(serializable_metrics.get("PERCLOS", "N/A"), 2) if isinstance(serializable_metrics.get("PERCLOS"), (int, float)) else "N/A",
                    round(serializable_metrics.get("FOM", "N/A"), 2) if isinstance(serializable_metrics.get("FOM"), (int, float)) else "N/A",
                    serializable_metrics.get("Fatigue", "N/A"),
                    round(serializable_metrics.get("Yawning Confidence", "N/A"), 2) if isinstance(serializable_metrics.get("Yawning Confidence"), (int, float)) else "N/A",
                    serializable_metrics.get("Yawning", "N/A"),
                ])
            last_csv_write_time = time.time()

        # Frame overlay
        processed_frame = draw_metrics_on_frame(frame, serializable_metrics, fatigue_status, landmarks)
        _, buffer = cv2.imencode('.jpg', processed_frame)
        frame_b64 = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            "metrics": serializable_metrics,
            "processed_frame": f"data:image/jpeg;base64,{frame_b64}",
            "csv_filename": session_csv_path
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
    

@app.route('/current_metrics', methods=['GET'])
def get_current_metrics():
    """
    A route to fetch the current metrics being processed.
    """
    global last_frame_metrics

    try:
        # Convert all float32 values to float for JSON serialization
        converted_metrics = {
            key: float(value) if isinstance(value, (np.float32, np.float64)) else value
            for key, value in last_frame_metrics.items()
        }

        return jsonify({"current_metrics": converted_metrics})

    except Exception as e:
        print(f"Error fetching current metrics: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
