import streamlit as st
import librosa
import librosa.display
import soundfile as sf
import matplotlib.pyplot as plt
import io
import pandas as pd
import os
import requests

#Initialize History
if "history" not in st.session_state:
    st.session_state.history = []

# Sidebar Instructions
st.sidebar.header("Page Instructions")
st.sidebar.info(
    """
    - This page provides an overview of the **DeepFake Voice Detection** app.
    - Learn how the system processes audio to detect **DeepFake AI voices**.
    - Navigate to the **Predict** page to upload and analyze an audio file.
    - View past predictions in the **History** page.
    """
)

# Welcome Section
st.title("🔊 DeepFake Voice Detection")
st.write(
    """
    Welcome to the **DeepFake Voice Detection** app! This tool allows you to upload an audio file in **FLAC** format and determine whether the voice is AI-generated or a real human voice.  
    The system uses **deep learning models** trained on real and synthetic audio to make accurate predictions.
    """
)

# How It Works Section
st.subheader("⚙️ How It Works")
st.markdown(
    """
    1️⃣ **Data Preprocessing** – The audio is cleaned and standardized.  
    2️⃣ **Feature Extraction** – Spectral features (**MFCCs**) are extracted from the sound.  
    3️⃣ **CNN Model Processing** – A **Convolutional Neural Network (CNN)** analyzes the features.  
    4️⃣ **Prediction Output** – The system determines whether the voice is **Real** or **DeepFake AI**.  
    """
)

# Quick Start Guide
st.subheader("🚀 Quick Start Guide")
st.markdown(
    """
    1. Go to the **Predict** page and upload an audio file.  
    2. Click the **Predict** button to analyze the file.  
    3. View the **Prediction Result**.  
    4. Check the **History** page to review past predictions.  
    """
)

