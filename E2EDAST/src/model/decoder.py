import torch
import torch.nn as nn

class TransformerDecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tgt, memory):
        """
        Args:
            tgt (Tensor): (B, T_tgt, D)
            memory (Tensor): (B, T_src, D)
        Returns:
            Tensor: (B, T_tgt, D)
        """
        tgt2, _ = self.self_attn(tgt, tgt, tgt)
        tgt = tgt + tgt2

        tgt2, _ = self.cross_attn(tgt, memory, memory)
        tgt = tgt + tgt2

        tgt = tgt + self.ffn(tgt)

        return self.norm(tgt)

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size=5000, d_model=256, num_layers=4, num_heads=4, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, 512, d_model))
        self.layers = nn.ModuleList([
            TransformerDecoderBlock(d_model, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.output_layer = nn.Linear(d_model, vocab_size)

    def forward(self, tgt_tokens, memory):
        """
        Args:
            tgt_tokens (Tensor): (B, T_tgt) - token ids
            memory (Tensor): (B, T_src, D)
        Returns:
            Tensor: (B, T_tgt, vocab_size)
        """
        B, T = tgt_tokens.size()
        x = self.embedding(tgt_tokens) + self.pos_embedding[:, :T, :]

        for layer in self.layers:
            x = layer(x, memory)

        logits = self.output_layer(x)
        return logits
