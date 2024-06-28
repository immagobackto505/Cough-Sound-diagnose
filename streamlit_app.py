import streamlit as st
import numpy as np
import tensorflow as tf
import librosa
import soundfile as sf
import io
import os
import traceback
from audio_recorder_streamlit import audio_recorder

# Function to load and return model
@st.cache_resource
def load_model(model_path):
    try:
        # Use custom_objects if you have any custom layers or objects
        custom_objects = {
            # 'MyCustomLayer': MyCustomLayer,
            # Add any other custom objects here
        }
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        st.success("Model loaded successfully.")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.text(traceback.format_exc())
        raise e
    return model

# Function to preprocess the input data
def preprocess_input(audio_data, num_mfcc=13, n_fft=2048, hop_length=512, expected_time_steps=120):
    # Save audio data to a temporary file
    with io.BytesIO(audio_data) as f:
        audio, sr = sf.read(f)
    
    # Extract MFCCs
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
    
    # Normalize MFCCs
    mfccs_normalized = mfccs.T  # Transpose to have shape (time_steps, num_mfcc)
    
    # Adjust number of time steps to match expected shape
    if mfccs_normalized.shape[0] < expected_time_steps:
        # Pad MFCCs if fewer time steps
        mfccs_normalized = np.pad(mfccs_normalized, ((0, expected_time_steps - mfccs_normalized.shape[0]), (0, 0)))
    elif mfccs_normalized.shape[0] > expected_time_steps:
        # Truncate MFCCs if more time steps
        mfccs_normalized = mfccs_normalized[:expected_time_steps, :]
    
    # Add an additional dimension for compatibility with Conv2D input shape
    mfccs_reshaped = mfccs_normalized[np.newaxis, ..., np.newaxis]  # Shape (1, expected_time_steps, num_mfcc, 1)
    
    return mfccs_reshaped

# Main function to run the Streamlit app
def main():
    st.title('Audio Classification App')
    st.write('Upload an audio file or record an audio clip and get predictions!')

    # File uploader
    uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3"])

    # Audio recorder
    st.write("Or record an audio clip:")
    audio_bytes = audio_recorder(
        energy_threshold=(-1.0, 1.0),
        pause_threshold=3.0,
    )

    audio_data = None

    if uploaded_file is not None:
        audio_data = uploaded_file.read()
        st.audio(audio_data, format='audio/wav')  # Display the audio file
    elif audio_bytes is not None:
        audio_data = audio_bytes
        st.audio(audio_data, format='audio/wav')  # Display the recorded audio

    if audio_data is not None:
        # Preprocess the audio data
        processed_data = preprocess_input(audio_data)

        # Print the current working directory for debugging
        st.write("Current working directory:", os.getcwd())
        
        # Load the model
        model_path = 'MODEL_CNN03.h5'  # Updated model path
        try:
            model = load_model(model_path)
            # Make prediction
            if processed_data is not None:
                prediction = model.predict(processed_data)
                st.write('Prediction:')
                st.write(prediction)  # Display the prediction results
        except Exception as e:
            st.error(f"Failed to load or predict using the model: {str(e)}")

if __name__ == '__main__':
    main()
