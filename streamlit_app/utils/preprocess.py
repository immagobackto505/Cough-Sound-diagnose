import numpy as np
import librosa
import soundfile as sf
import os
    
def preprocess_audio(audio_data, sr=22050):
    # Load audio file
    y, sr = librosa.load(audio_data, sr=sr)
    # Extract features (e.g., MFCCs)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    # print(f'\nmfcc:{mfccs.shape}\n')
    # Reshape to fit model input
    if mfccs.shape[1] < 87:
        print(f'\nbefore padding: {mfccs.shape}\n')
        mfccs = np.pad(mfccs, ((0, 0), (0, 87 - mfccs.shape[1])), mode='constant')
    elif mfccs.shape[1] > 87:
        print(f'\nbefore reshaping: {mfccs.shape}\n')
        mfccs = mfccs[:, :87]
    else:
        print(f'\nno padding or reshaping needed\n')
        pass
    print(f'\nafter: {mfccs.shape}\n')
    # Reshape to (87, 13, 1)
    mfccs = np.expand_dims(mfccs, axis=-1)
    
    # Add batch dimension to make the shape (1, 87, 13, 1)
    mfccs = np.expand_dims(mfccs, axis=0)
    print(f'\nmfcc reshaped:{mfccs.shape}\n')
    return mfccs




