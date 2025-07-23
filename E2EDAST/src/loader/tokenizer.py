import torch
from collections import defaultdict

class CharTokenizer:
    def __init__(self, transcripts=None, reserved_tokens=["<pad>", "<unk>", "<sos>", "<eos>"]):
        self.reserved_tokens = reserved_tokens
        self.char2idx = {token: idx for idx, token in enumerate(reserved_tokens)}
        self.idx2char = {idx: token for token, idx in self.char2idx.items()}
        self.vocab_built = False

        if transcripts:
            self.build_vocab(transcripts)

    def build_vocab(self, transcripts):
        char_set = set()
        for text in transcripts:
            char_set.update(text.lower().strip())

        # Remove reserved tokens from char_set (in case of collision)
        char_set = sorted(list(char_set - set(self.reserved_tokens)))

        for c in char_set:
            if c not in self.char2idx:
                idx = len(self.char2idx)
                self.char2idx[c] = idx
                self.idx2char[idx] = c

        self.vocab_built = True

    def encode(self, text):
        """
        Convert a string to a list of token IDs.
        """
        text = text.lower().strip()
        return [self.char2idx.get(c, self.char2idx["<unk>"]) for c in text]

    def decode(self, token_ids):
        """
        Convert a list of token IDs to a string.
        """
        chars = [self.idx2char.get(tid, "<unk>") for tid in token_ids]
        return "".join([c for c in chars if c not in self.reserved_tokens])

    def vocab_size(self):
        return len(self.char2idx)

    def save_vocab(self, filepath):
        import json
        with open(filepath, "w") as f:
            json.dump(self.char2idx, f)

    def load_vocab(self, filepath):
        import json
        with open(filepath, "r") as f:
            self.char2idx = json.load(f)
        self.idx2char = {idx: char for char, idx in self.char2idx.items()}
        self.vocab_built = True
