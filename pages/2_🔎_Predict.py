import streamlit as st
import librosa
import librosa.display
import soundfile as sf
import matplotlib.pyplot as plt
import io
import pandas as pd
import os
import requests
import base64

#Title
st.title("Predict: Detect DeepFake Audio")

# Sidebar Instructions
st.sidebar.header("Page Instructions")
st.sidebar.info(
    """
    - Upload an **audio file (FLAC format only)** for analysis.
    - Once uploaded, you will see:
      - A **waveform visualization** of the audio.
      - An **audio player** to listen to the file.
    - Click the **"Predict"** button to analyze the file.
    - The model will determine if the audio is **Real Human Voice** or **DeepFake AI Voice**.
    - The result will include:
      - The **prediction label** (Real or DeepFake).
      - The **confidence probability** of the prediction.
    - Each prediction is **automatically saved** to the history page.
    """
)

# Sidebar Note
st.sidebar.header("Notes")
st.sidebar.info(
    """
    - Only **FLAC** files are supported.
    - Ensure the audio is **clear** for the best accuracy.
    - Predictions are based on **pre-trained AI models** and may not be 100% accurate.
    - You can view past predictions in the **History** page.
    """
)

#Initialize History
if "history" not in st.session_state:
    st.session_state.history = []

uploaded_file = st.file_uploader("Upload an audio file (FLAC)", type=["flac"])

if uploaded_file is not None:
    # Extract file extension
    file_extension = uploaded_file.name.split('.')[-1].lower()

    if file_extension == "flac":
        st.success(f"File '{uploaded_file.name}' uploaded successfully!")

        # Display waveform
        st.write(f"### {uploaded_file.name} waveform")
        y, sr = librosa.load(uploaded_file, sr=None)
        fig, ax = plt.subplots(figsize=(10, 4))
        librosa.display.waveshow(y, sr=sr, ax=ax, color="black")
        ax.axis('off')  # Disable x and y axis
        st.pyplot(fig)

        # Save waveform image to bytes
        img_buffer = io.BytesIO()
        fig.savefig(img_buffer, format="png")
        img_buffer.seek(0)
        img_str = base64.b64encode(img_buffer.read()).decode() 

        # Encode audio file in base64
        # Save audio as bytes
        audio_bytes = uploaded_file.getvalue()  # This correctly stores the raw bytes

        # Encode audio file in base64
        audio_str = base64.b64encode(audio_bytes).decode()

        # Play Audio
        st.audio(uploaded_file, format='audio/' + file_extension)

        # Predict Button
        if st.button("Predict"):
            st.write("Processing your request...")

            # Prepare audio and metadata
            audio_bytes = uploaded_file.read()
            files = {"file": (uploaded_file.name, audio_bytes, "audio/flac")}

            # Send request to backend
            backend_url = "http://localhost:8000/predict/" 
            try:
                response = requests.post(backend_url, files=files)
                if response.status_code == 200:
                    result = response.json()
                        
                    st.success(result["prediction"])

                    # Add to history
                    st.session_state.history.append({
                        "File": uploaded_file.name,
                        "Prediction": result["prediction"],
                        "Waveform": img_str,  # Store base64 waveform image
                        "Audio": audio_str  # Store base64 audio file  # Store base64 waveform image
                        })
                else:
                    st.error(f"Error from backend: {response.text}")
            except Exception as e:
                st.error(f"Failed to connect to backend: {e}")
