import streamlit as st
import numpy as np
import os
from tensorflow.keras.models import load_model #type: ignore
from utils.preprocess import preprocess_audio
from utils.recorder import record_audio
from utils.denoise import denoise_audio
import shutil
import soundfile as sf
from utils.output import output

# Load your trained model
model = load_model(r'streamlit_app\models\cough_classifier.h5')

#recompile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Temporary directory for saving audio files
temp_dir = 'temp_audio'
if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)

# Set up the Streamlit app
st.title("Audio Classification AI")
st.write("Record an audio sample or upload an audio file to classify it.")

# Sidebar for navigation
option = st.sidebar.selectbox("Choose input method", ["Record Audio", "Upload Audio"])

# Function to handle audio processing
def process_audio(audio_path):
    denoised_segments, sr = denoise_audio(audio_path, temp_dir)
    for i, segment in enumerate(denoised_segments):
        segment_path = os.path.join(temp_dir, f'segment_{i+1}.wav')
        sf.write(segment_path, segment, sr)
        st.audio(segment_path, format='audio/wav')

    # Preprocess the denoised audio
    features = preprocess_audio(audio_path)

    # Make prediction
    prediction = model.predict(features)
    predicted_index = np.argmax(prediction, axis=1)
    print(prediction)
    output(prediction,predicted_index)

    # Clean up temporary files
    os.remove(audio_path)
    for segment_file in os.listdir(temp_dir):
        os.remove(os.path.join(temp_dir, segment_file))

# Record Audio
if option == "Record Audio":
    if st.button("Record Audio"):
        audio_path = os.path.join(temp_dir, 'recorded_audio.wav')
        record_audio(audio_path)
        process_audio(audio_path)

# Upload Audio
elif option == "Upload Audio":
    uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3"])
    
    if uploaded_file is not None:
        audio_path = os.path.join(temp_dir, uploaded_file.name)
        with open(audio_path, 'wb') as f:
            f.write(uploaded_file.read())
        process_audio(audio_path)

# Cleanup temporary directory on app exit
def cleanup_temp_dir():
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

cleanup_temp_dir()
