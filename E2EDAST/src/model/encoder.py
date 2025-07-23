import torch
import torch.nn as nn
import torchaudio

class ConformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.ffn1 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
        )
        self.ffn2 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: (B, T, D)
        residual = x
        x, _ = self.self_attn(x, x, x)
        x = residual + x

        x = x + self.ffn1(x)

        x_conv = x.transpose(1, 2)  # (B, D, T)
        x_conv = self.conv(x_conv)
        x = x + x_conv.transpose(1, 2)

        x = x + self.ffn2(x)

        return self.norm(x)


class ConformerEncoder(nn.Module):
    def __init__(self, input_dim=80, d_model=256, num_layers=6, num_heads=4, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.layers = nn.ModuleList([
            ConformerBlock(d_model, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        Args:
            x (Tensor): (B, T, input_dim) — e.g., log-mel spectrogram

        Returns:
            Tensor: (B, T, d_model)
        """
        x = self.input_proj(x)
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)
