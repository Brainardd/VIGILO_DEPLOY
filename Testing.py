import cv2
import dlib
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('models/yawn_detection_model.h5')

# Load dlib's face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

def extract_mouth_region(frame):
    """
    Extract the mouth region from a full face image.
    """
    if frame.dtype != 'uint8':  # Ensure frame is in the correct format
        frame = frame.astype('uint8')
    
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

def preprocess_mouth(mouth_region):
    """
    Preprocess the mouth region to make it compatible with the trained model.
    """
    mouth_resized = cv2.resize(mouth_region, (128, 128))
    mouth_normalized = mouth_resized / 255.0
    mouth_input = np.expand_dims(mouth_normalized, axis=0)
    return mouth_input

# Open webcam
cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Check frame properties
    print(f"Frame dtype: {frame.dtype}, shape: {frame.shape}")

    # Extract the mouth region
    mouth_region, found = extract_mouth_region(frame)
    if found and mouth_region is not None:
        # Preprocess the mouth region
        mouth_input = preprocess_mouth(mouth_region)

        # Predict yawning
        prediction = model.predict(mouth_input)
        label = "Yawning" if prediction[0] > 0.5 else "Not Yawning"
        confidence = prediction[0][0] if prediction[0] > 0.5 else 1 - prediction[0][0]

        # Display the label on the frame
        color = (0, 255, 0) if label == "Yawning" else (0, 0, 255)
        cv2.putText(frame, f"{label} ({confidence:.2f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Show the video feed
    cv2.imshow("Yawning Detection", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
