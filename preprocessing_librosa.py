import preprocessing_librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

# Load an audio file
audio_file = "GD2.wav"
y, sr = preprocessing_librosa.load(audio_file, sr=22050)

# Plot waveform
plt.figure(figsize=(10, 4))
preprocessing_librosa.display.waveshow(y, sr=sr)
plt.title("Waveform")
plt.show()

# Compute MFCCs (Mel-Frequency Cepstral Coefficients)
mfccs = preprocessing_librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
print(mfccs.shape)
