import torchaudio

class FeatureExtractor:
    def __init__(self, sample_rate=16000, num_mel_bins=80):
        self.sample_rate = sample_rate
        self.num_mel_bins = num_mel_bins
        self.transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=num_mel_bins,
            n_fft=400,
            hop_length=160,
            win_length=400
        )

    def __call__(self, waveform):
        return self.transform(waveform).squeeze(0).transpose(0, 1)  # shape: (T, D)
