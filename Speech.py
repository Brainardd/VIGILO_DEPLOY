import librosa
import numpy as np
import joblib

# Load the trained audio model
audio_model = joblib.load("models/audio_classification_model.pkl")

def predict_live_audio(audio, sr, model):
    """
    Predicts the class of live audio input using the trained model.
    """
    if np.sqrt(np.mean(audio**2)) < 0.01:  # Silence threshold
        return "neutral"
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc.T, axis=0).reshape(1, -1)
    prediction = model.predict(mfcc_mean)
    return prediction[0]
