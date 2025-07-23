# import torchaudio

# # (Optional but recommended)
# torchaudio.set_audio_backend("soundfile")

# waveform, sample_rate = torchaudio.load("data/audio/1.wav")
# print(waveform.shape, sample_rate)

# import torchaudio
# waveform, sr = torchaudio.load("data/audio_fixed/1.wav")
# print(waveform.shape, sr)


import torchaudio
print("torchaudio version:", torchaudio.__version__)
print("Available backends:", torchaudio.list_audio_backends())
