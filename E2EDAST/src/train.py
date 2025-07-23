import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from loader.dataset import E2EDataset
from loader.feature_extractor import FeatureExtractor
from loader.tokenizer import CharTokenizer
from model import E2EDASTModel, DiarizationLoss, ASRLoss
from utils.helpers import collate_fn, set_seed, load_config
from utils.logger import Logger

def train():
    # Load configs
    cfg = load_config("configs/training_config.yaml")
    set_seed(cfg.get("seed", 42))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data loading
    tokenizer = CharTokenizer()
    feature_extractor = FeatureExtractor(
        sample_rate=cfg["data"]["sample_rate"],
        num_mel_bins=cfg["model"]["num_mel_bins"]
    )
    dataset = E2EDataset(cfg["data"], feature_extractor, tokenizer)
    loader = DataLoader(
        dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        collate_fn=collate_fn
    )

    # Model & Loss
    model = E2EDASTModel(cfg["model"]).to(device)
    diar_loss_fn = DiarizationLoss(ignore_index=-100)
    asr_loss_fn = ASRLoss(blank=tokenizer.char2idx["<pad>"])

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["optimizer"]["lr"],
        betas=tuple(cfg["optimizer"]["betas"]),
        weight_decay=cfg["optimizer"]["weight_decay"]
    )

    logger = Logger(log_dir="logs", experiment_name="e2edast")

    # Training loop
    model.train()
    accumulation_steps = cfg["training"].get("gradient_accumulation_steps", 1)

    for epoch in range(cfg["training"]["epochs"]):
        optimizer.zero_grad()
        for i, batch in enumerate(loader):
            feats = batch["features"].to(device)       # (B, T, D)
            toks = batch["asr_tokens"].to(device)      # (B, L)
            diar_tgt = batch["diar_target"].to(device) # (B, T)

            outputs = model(feats)
            diar_logits = outputs["diar_logits"]       # (B, T, S)
            asr_logits = outputs["asr_logits"]         # (B, T, V)

            # Diarization loss
            loss_diar = diar_loss_fn(diar_logits, diar_tgt)

            # CTC loss
            log_probs = nn.functional.log_softmax(asr_logits, dim=-1).transpose(0, 1)  # (T, B, V)
            input_lengths = torch.full((feats.size(0),), log_probs.size(0), dtype=torch.long)
            target_lengths = torch.tensor([len(tok) for tok in toks])
            toks_flat = torch.cat([tok[:l] for tok, l in zip(toks, target_lengths)], dim=0)

            loss_asr = asr_loss_fn(log_probs, toks_flat, input_lengths, target_lengths)

            loss = (
                cfg["loss_weights"]["diarization"] * loss_diar +
                cfg["loss_weights"]["asr"] * loss_asr
            ) / accumulation_steps

            loss.backward()

            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        print(f"Epoch {epoch+1}: diar_loss={loss_diar.item():.4f}, asr_loss={loss_asr.item():.4f}")
        logger.log_scalar("Loss/diar", loss_diar.item(), epoch)
        logger.log_scalar("Loss/asr",  loss_asr.item(),  epoch)

        if epoch % cfg["logging"].get("save_checkpoint_every", 1) == 0:
            os.makedirs(cfg["checkpoint_dir"], exist_ok=True)
            torch.save(model.state_dict(), os.path.join(cfg["checkpoint_dir"], f"e2edast_epoch{epoch+1}.pt"))

    logger.close()

if __name__ == "__main__":
    train()
