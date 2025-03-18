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
from PIL import Image
import io

#Page Title
st.title("Prediction History")

# Sidebar Instructions
st.sidebar.header("Page Instructions")
st.sidebar.info(
    """
    - This page displays the history of past predictions.
    - Each entry includes:
      - The uploaded audio file's **name**.
      - A **waveform image** representation of the audio.
      - The **prediction result** (Real or DeepFake).
      - The **prediction probability**.
      - An **audio player** to listen to the file.
    - You can **delete** an individual prediction by clicking the "Delete" button under it.
    """
)

# Display the current history
if "history" in st.session_state and st.session_state.history:
    for index, entry in enumerate(st.session_state.history[::-1]):  # Show latest predictions first
        st.write(f"### {entry['File']}")

        # Decode and display waveform image
        img_bytes = base64.b64decode(entry["Waveform"])
        img = Image.open(io.BytesIO(img_bytes))
        st.image(img, use_column_width=True)

        # Display Prediction and Probability
        st.write(f"**Prediction:** {entry['Prediction']}")

        # Play the audio
        audio_bytes = base64.b64decode(entry["Audio"])
        st.audio(audio_bytes, format="audio/flac")

        # Button to delete this prediction
        if st.button(f"Delete {entry['File']}", key=f"delete_{index}"):
            # Find the actual index in the original list (reverse the index)
            real_index = len(st.session_state.history) - 1 - index
            del st.session_state.history[real_index]  # Remove from history
            st.toast(f"✅ {entry['File']} deleted successfully!", icon="✅")
            st.rerun() 

        st.markdown("---")  # Separator line for clarity

else:
    st.info("No predictions have been made yet. Go to the **Predict** page to classify audio files.")
