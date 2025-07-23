# src/tests/overfit_test.py

import os, sys
# Ensure src/ is on Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from loader.feature_extractor import FeatureExtractor
from loader.tokenizer import CharTokenizer
from utils.helpers import collate_fn

from model import E2EDASTModel
from model.loss import DiarizationLoss, ASRLoss

def build_toy_loader():
    """
    Construct the same single-example DataLoader as in smoke_test,
    for overfitting.
    """
    # Fake 1s audio @16k -> ~100 frames
    waveform = torch.randn(1, 16000)
    fe = FeatureExtractor(sample_rate=16000, n_mels=80)
    features = fe(waveform)  # (T=100, D=80)

    # Fake tokenizer & transcript
    fake_transcripts = ["hello world"]
    tk = CharTokenizer(transcripts=fake_transcripts)
    tokens = torch.tensor(tk.encode("hello"), dtype=torch.long)  # (L=5,)

    # Fake diarization target: alternate speaker every 20 frames
    T = features.size(0)
    diar_target = torch.zeros(T, dtype=torch.long)
    diar_target[20:40] = 1
    diar_target[60:80] = 1

    # Single-item dataset
    batch = [{
        "features": features,           # (T, D)
        "asr_tokens": tokens,           # (L,)
        "diar_target": diar_target,     # (T,)
        "num_speakers": 2
    }]

    return DataLoader(batch, batch_size=1, collate_fn=collate_fn), tk

def overfit_test():
    # 1) Build DataLoader and tokenizer
    loader, tokenizer = build_toy_loader()

    # 2) Minimal config: must match smoke_test config
    cfg = {
        "encoder": {
            "input_dim": 80,
            "d_model": 128,
            "num_layers": 2,
            "num_heads": 4,
            "dropout": 0.1
        },
        "decoder": {
            "vocab_size": tokenizer.vocab_size(),
            "d_model": 128,
            "num_layers": 2,
            "num_heads": 4,
            "dropout": 0.1
        },
        "speaker_embedding": {"dim": 64},
        "max_num_speakers": 2
    }

    # 3) Model, losses, optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = E2EDASTModel(cfg).to(device)
    diar_loss_fn = DiarizationLoss(ignore_index=-100)
    asr_loss_fn  = ASRLoss(blank=tokenizer.reserved_tokens.index("<pad>"))
    optimizer    = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # 4) Overfit loop
    print("Starting overfit test...")
    for step in range(200):
        batch = next(iter(loader))
        feats     = batch["features"].to(device)      # (1, T, D)
        feat_lens = batch["feat_lens"]                # (1,)
        toks      = batch["asr_tokens"]               # (1, L)
        tok_lens  = batch["tok_lens"]                 # (1,)
        diar_tgt  = batch["diar_target"].to(device)   # (1, T)

        out = model(feats)
        diar_logits = out["diar_logits"]              # (1, T, S)
        asr_logits  = out["asr_logits"]               # (1, T, V)

        # Diarization loss
        loss_d = diar_loss_fn(diar_logits, diar_tgt)

        # ASR CTC loss: convert to (T, B, V)
        logp = nn.functional.log_softmax(asr_logits, dim=-1).transpose(0, 1)  # (T, B, V)
        # Concatenate token labels for CTC
        labels_concat = torch.cat([toks[0, :tok_lens[0]]])
        loss_a = asr_loss_fn(logp, labels_concat, feat_lens, tok_lens)

        loss = loss_d + loss_a
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 20 == 0 or step == 199:
            print(f"Step {step:3d}: diar_loss={loss_d.item():.4f}, asr_loss={loss_a.item():.4f}, total={loss.item():.4f}")

    print("Overfit test completed. Expect losses near zero.")

if __name__ == "__main__":
    overfit_test()
