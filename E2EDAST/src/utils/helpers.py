import random
import numpy as np
import torch
import yaml
from torch.nn.utils.rnn import pad_sequence

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def pad_feats(feat_list, pad_value=0.0):
    lengths = torch.tensor([f.size(0) for f in feat_list], dtype=torch.long)
    padded = pad_sequence(feat_list, batch_first=True, padding_value=pad_value)
    return padded, lengths

def pad_tokens(token_list, pad_id=0):
    lengths = torch.tensor([t.size(0) for t in token_list], dtype=torch.long)
    padded = pad_sequence(token_list, batch_first=True, padding_value=pad_id)
    return padded, lengths

def pad_diar(diar_list, pad_value=0.0):
    padded = pad_sequence(diar_list, batch_first=True, padding_value=pad_value)
    return padded

def collate_fn(batch):
    feats = [b["features"] for b in batch]
    toks  = [b["asr_tokens"] for b in batch]
    diars = [b["diar_target"] for b in batch]
    spk_counts = torch.tensor([b["num_speakers"] for b in batch], dtype=torch.long)

    padded_feats, feat_lens = pad_feats(feats)
    padded_toks, tok_lens = pad_tokens(toks)
    padded_diars = pad_sequence(diars, batch_first=True, padding_value=-100)

    return {
        "features": padded_feats,
        "feat_lens": feat_lens,
        "asr_tokens": padded_toks,
        "tok_lens": tok_lens,
        "diar_target": padded_diars,
        "num_speakers": spk_counts,
    }
