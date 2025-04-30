# thesis-deploy

# Deepfake Voice Detection App

This project is a web application designed to detect synthetic speech generated through Voice Conversion (VC), Text-To-Speech (TTS), and TTS-VC techniques. It combines the power of **Streamlit** for a user-friendly interface and **FastAPI** for fast and efficient backend processing.

# Requirements

You need to install all required packages listed in the requirements.txt to run this app.

`pip install -r requirements.txt`

# How to run

The app use streamlit for frontend and fastapi for backend

To run frontend, go into the frontend directory and run the following: `streamlit run 1_🏠_Main.py`

To run backend, go into the backend directory and run the following: `uvicorn backend:app --reload --host 0.0.0.0 --port 8000`
