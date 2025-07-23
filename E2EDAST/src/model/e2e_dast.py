# src/model/e2e_dast.py

import torch
import torch.nn as nn
from model.encoder import ConformerEncoder
from model.decoder import TransformerDecoder

class E2EDASTModel(nn.Module):
    def __init__(self, config):
        super(E2EDASTModel, self).__init__()
        
        # Encoder
        self.encoder = ConformerEncoder(
            input_dim=config['encoder']['input_dim'],
            d_model=config['encoder']['d_model'],
            num_layers=config['encoder']['num_layers'],
            num_heads=config['encoder']['num_heads'],
            dropout=config['encoder']['dropout'],
        )
        
        # Decoder (optional; used if tgt_tokens passed)
        self.decoder = TransformerDecoder(
            vocab_size=config['decoder']['vocab_size'],
            d_model=config['decoder']['d_model'],
            num_layers=config['decoder']['num_layers'],
            num_heads=config['decoder']['num_heads'],
            dropout=config['decoder']['dropout'],
        )
        
        # Speaker embedding projection
        self.speaker_embedding = nn.Linear(
            config['encoder']['d_model'], 
            config['speaker_embedding']['dim']
        )
        
        # ASR head: maps encoder output to vocab logits
        self.asr_head = nn.Linear(
            config['encoder']['d_model'], 
            config['decoder']['vocab_size']
        )
        
        # Diarization head: maps encoder output to speaker logits
        self.slidar_head = nn.Linear(
            config['encoder']['d_model'], 
            config.get('max_num_speakers', 4)
        )
        
        self.dropout = nn.Dropout(p=config['encoder'].get('dropout', 0.1))

    def forward(self, x, tgt_tokens=None):
        """
        Args:
            x (Tensor): (B, T, feat_dim)
            tgt_tokens (Tensor, optional): (B, T_tgt) input to decoder

        Returns:
            dict with:
              'diar_logits': (B, T, S)
              'asr_logits':  (B, T, V)
              'spk_embeds':  (B, T, spk_dim)
              'dec_out':     (B, T_tgt, V) or None
        """
        # Encode
        enc_out = self.encoder(x)             # (B, T, D)
        enc_out = self.dropout(enc_out)

        # Optional decoder output
        if tgt_tokens is not None:
            dec_out = self.decoder(tgt_tokens, enc_out)  
        else:
            dec_out = None

        # Speaker embeddings
        spk_embeds = self.speaker_embedding(enc_out)  

        # ASR logits
        asr_logits = self.asr_head(enc_out)  

        # Diarization logits  
        diar_logits = self.slidar_head(enc_out)  

        return {
            "diar_logits": diar_logits,
            "asr_logits": asr_logits,
            "spk_embeds": spk_embeds,
            "dec_out": dec_out
        }
