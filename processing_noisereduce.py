import noisereduce as nr
import librosa

# Load audio
y, sr = librosa.load("Recording.wav", sr=None)

# Apply noise reduction
reduced_noise = nr.reduce_noise(y=y, sr=sr)

# Save the processed audio
import soundfile as sf
sf.write("denoised_audio.wav", reduced_noise, sr)

