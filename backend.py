from fastapi import FastAPI, UploadFile, HTTPException
import librosa
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import binary_crossentropy
import tensorflow.keras.backend as K
from tensorflow.keras.regularizers import l2

app = FastAPI()

def focal_loss(alpha=0.25, gamma=2.0):
    def loss(y_true, y_pred):
        y_true = K.cast(y_true, dtype=K.floatx())
        bce = binary_crossentropy(y_true, y_pred)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        focal_term = K.pow(1 - p_t, gamma)
        return alpha * focal_term * bce
    return loss

# Load models
try:
    print("Loading custom model")
    model = load_model('model/cnn_40_optimized.keras', custom_objects={'loss': focal_loss(alpha=0.25, gamma=2.0)})
except:
    print("Failed to load model")

def pad_or_truncate(features, max_frames):
    """
    Pad atau truncate fitur agar memiliki jumlah frame yang konsisten.
    """
    if features.shape[1] < max_frames:
        pad_width = max_frames - features.shape[1]
        features = np.pad(features, ((0, 0), (0, pad_width)), mode='constant')
    elif features.shape[1] > max_frames:
        features = features[:, :max_frames]
    return features

def extract_mfcc(audio_path, n_mfcc=40, duration=3):
    """
    Ekstraksi MFCC dengan padding atau truncating.
    """
    y, sr = librosa.load(audio_path, sr=16000, duration=duration)
    hop_length = 512  # Default hop length
    max_frames = 100
    
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
    mfcc = pad_or_truncate(mfcc, max_frames)

    return mfcc

def process_audio_file(audio_path):
    """
    Process a single audio file to extract MFCC features.
    """
    if os.path.exists(audio_path):
        features = extract_mfcc(audio_path)
        return features[..., np.newaxis]  # Add channel dimension for model compatibility
    else:
        raise FileNotFoundError(f"Audio file '{audio_path}' not found.")

@app.post("/predict/")
async def predict(file: UploadFile):

    if not file.filename.endswith(".flac"):
        raise HTTPException(status_code=400, detail="Only FLAC files are supported.")
    
    try:
        # Save the uploaded file temporarily
        file_path = f"./temp_{file.filename}"
        with open(file_path, "wb") as f:
            f.write(file.file.read())

        # Process the audio file and extract features
        extracted_features = process_audio_file(file_path)

        # Preprocess the features for model input
        model_input = np.expand_dims(extracted_features, axis=0)

        # Make prediction
        prediction = model.predict(model_input)

        real_voice_prob = prediction[0][0]  # Probability of Real Human Voice

        # Define a threshold, e.g., 50%
        threshold = 0.5

        # Assign label based on threshold
        predicted_label = "Real Human Voice" if real_voice_prob >= threshold else "DeepFake AI Voice"

        # Clean up temporary file
        os.remove(file_path)

        return {
            "prediction": predicted_label# Return the formatted probability string
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
