# loader/dataset.py

import os
import torch
import torchaudio
import numpy as np
from torch.utils.data import Dataset
import json


class E2EDataset(Dataset):
    def __init__(self, cfg, feature_extractor, tokenizer):
        self.audio_dir = cfg["audio_dir"]
        self.label_dir = cfg["label_dir"]
        self.sample_rate = cfg.get("sample_rate", 16000)
        self.max_len = cfg.get("max_len", 60.0)
        self.hop_len = cfg.get("hop_len", self.max_len)  # no overlap if hop_len=max_len

        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer

        self.segments = self._build_segments()

    def _build_segments(self):
        segments = []
        for wav_name in os.listdir(self.audio_dir):
            if not wav_name.endswith(".wav"):
                continue

            audio_path = os.path.join(self.audio_dir, wav_name)
            utt_id = os.path.splitext(wav_name)[0]

            # Read the waveform to get sample rate and duration
            try:
                waveform, sample_rate = torchaudio.load(audio_path)
                duration = waveform.size(1) / sample_rate
            except Exception as e:
                print(f"Error loading {audio_path}: {e}")
                continue

            # Find matching JSON transcript
            transcript_path = os.path.join(self.label_dir, utt_id + ".json")
            if not os.path.exists(transcript_path):
                continue

            with open(transcript_path, "r") as f:
                data = json.load(f)
                utts = data["utterances"] if isinstance(data, dict) else data

            for utt in utts:
                segments.append({
                    "utt_id": utt_id,
                    "audio_path": audio_path,
                    "label_path": transcript_path,  # Add this so __getitem__ can load it
                    "start": utt["start"],
                    "end": utt["end"],
                    "text": utt["text"],
                    "speaker": utt["speaker"]
                })
        return segments


    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        seg = self.segments[idx]
        waveform, sr = torchaudio.load(seg["audio_path"])
        assert sr == self.sample_rate

        # Slice segment
        start_sample = int(seg["start"] * sr)
        end_sample = int(seg["end"] * sr)
        waveform = waveform[:, start_sample:end_sample]

        features = self.feature_extractor(waveform)  # (T, D)

        # Load JSON transcript & diar labels
        with open(seg["label_path"]) as f:
            full_turns = json.load(f)

        # Keep only relevant turns in segment
        turns = []
        for t in full_turns:
            if t["end"] < seg["start"] or t["start"] > seg["end"]:
                continue
            # Clip turn times to segment boundaries
            turns.append({
                "start": max(0, t["start"] - seg["start"]),
                "end": min(seg["end"] - seg["start"], t["end"] - seg["start"]),
                "speaker": t["speaker"],
                "text": t["text"]
            })

        # Tokenize all texts in this chunk
        transcript = " ".join([t["text"] for t in turns])
        asr_tokens = torch.tensor(self.tokenizer.encode(transcript), dtype=torch.long)

        # Diarization target
        T = features.size(0)
        frame_shift = 0.01  # 100fps
        diar_target = np.zeros((T,), dtype=np.int64)
        spk_ids = sorted({t["speaker"] for t in turns})
        spk2idx = {spk: i for i, spk in enumerate(spk_ids)}

        for t in turns:
            s = int(t["start"] / frame_shift)
            e = min(T, int(np.ceil(t["end"] / frame_shift)))
            diar_target[s:e] = spk2idx[t["speaker"]]

        return {
            "features": features,
            "asr_tokens": asr_tokens,
            "diar_target": torch.from_numpy(diar_target),
            "num_speakers": len(spk_ids)
        }
