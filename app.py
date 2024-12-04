from flask import Flask, render_template, Response, jsonify
import cv2
from Fatigue import process_frame, draw_metrics_on_frame
from collections import deque

app = Flask(__name__)

# Initialize camera and variables
camera = cv2.VideoCapture(1)
closed_frames = deque(maxlen=10)
mouth_open_counts = deque(maxlen=10)
yawn_state = {"in_progress": False, "start_time": 0, "detected": 0}
last_frame_metrics = {"EAR": "N/A", "MOR": "N/A", "Tilt Angle": "N/A", "Fatigue": "No"}

# Generate video frames with fatigue detection
def gen_frames():
    global last_frame_metrics
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Process the frame for fatigue detection
            fatigue_status, metrics, landmarks = process_frame(frame, closed_frames, mouth_open_counts, yawn_state)
            last_frame_metrics = metrics  # Update global metrics

            # Overlay metrics and facial outlines
            frame = draw_metrics_on_frame(frame, metrics, fatigue_status, landmarks)

            # Encode the frame for streaming
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            # Stream the frame
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Route for the main web page
@app.route('/')
def index():
    return render_template('index.html')

# Route for the video feed
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Route to provide real-time metrics as JSON
@app.route('/metrics')
def metrics():
    global last_frame_metrics
    return jsonify({"metrics": last_frame_metrics})

if __name__ == '__main__':
    app.run(debug=True)
