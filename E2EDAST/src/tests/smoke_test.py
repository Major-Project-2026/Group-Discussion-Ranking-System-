# src/tests/smoke_test.py

import os, sys
# Ensure src/ is on Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

from loader.dataset import E2EDataset
from loader.feature_extractor import FeatureExtractor
from loader.tokenizer import CharTokenizer
from model import E2EDASTModel
from model.loss import DiarizationLoss, ASRLoss
from utils.helpers import collate_fn

def smoke_test():
    # 1) Build a minimal config for the model
    cfg = {
        "encoder": {
            "input_dim": 80,
            "d_model": 128,
            "num_layers": 2,
            "num_heads": 4,
            "dropout": 0.1
        },
        "decoder": {
            "vocab_size": None,  # to be filled after tokenizer
            "d_model": 128,
            "num_layers": 2,
            "num_heads": 4,
            "dropout": 0.1
        },
        "speaker_embedding": {"dim": 64},
        "max_num_speakers": 2
    }

    # 2) Create fake data
    # 1s of audio at 16kHz, produce ~100 frames @10ms frame shift
    waveform = torch.randn(1, 16000)
    fe = FeatureExtractor(sample_rate=16000, n_mels=80)
    features = fe(waveform)              # (T=100, D=80)

    # Fake transcript and tokenizer
    fake_transcripts = ["hello world"]  
    tk = CharTokenizer(transcripts=fake_transcripts)
    cfg["decoder"]["vocab_size"] = tk.vocab_size()

    tokens = torch.tensor(tk.encode("hello"), dtype=torch.long)  # (L_t=5)

    # Fake diarization target: alternate speaker every 20 frames
    T = features.size(0)
    diar_target = torch.zeros(T, dtype=torch.long)
    diar_target[20:40] = 1
    diar_target[60:80] = 1

    # Pack into DataLoader
    batch = [{
        "features": features,            # (T, D)
        "asr_tokens": tokens,            # (L,)
        "diar_target": diar_target,      # (T,)
        "num_speakers": 2
    }]
    loader = DataLoader(batch, batch_size=1, collate_fn=collate_fn)

    # 3) Instantiate model & losses
    model = E2EDASTModel(cfg)
    diar_loss_fn = DiarizationLoss(ignore_index=-100)
    asr_loss_fn  = ASRLoss(blank=tk.reserved_tokens.index("<pad>"))

    # 4) Forward and compute losses
    batch = next(iter(loader))
    feats     = batch["features"]      # (B=1, T, D)
    feat_lens = batch["feat_lens"]     # (B)
    toks      = batch["asr_tokens"]    # (B, L)
    tok_lens  = batch["tok_lens"]      # (B)
    diar_tgt  = batch["diar_target"]   # (B, T)

    out = model(feats)
    diar_logits = out["diar_logits"]   # (1, T, S=2)
    asr_logits  = out["asr_logits"]    # (1, T, V)

    # Diarization loss
    loss_d = diar_loss_fn(diar_logits, diar_tgt)

    # ASR loss (CTC): convert to (T, B, V)
    logp = nn.functional.log_softmax(asr_logits, dim=-1).transpose(0, 1)  # (T, B, V)
    # concatenate token labels
    labels_concat = torch.cat([toks[i, :tok_lens[i]] for i in range(len(tok_lens))])
    loss_a = asr_loss_fn(logp, labels_concat, feat_lens, tok_lens)

    print(f"Smoke test passed:\n  diar_loss = {loss_d.item():.4f}\n  asr_loss  = {loss_a.item():.4f}")

if __name__ == "__main__":
    smoke_test()
