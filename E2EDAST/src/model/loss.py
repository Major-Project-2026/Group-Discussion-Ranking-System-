# src/model/loss.py

import torch
import torch.nn as nn

class DiarizationLoss(nn.Module):
    """
    Cross‐Entropy loss for speaker activity detection (frame‐level multi‐class).
    Expects:
      diar_logits: (B, T, S) raw logits over S speakers per frame
      diar_target: (B, T) integer labels in [0..S-1] or ignore_index for padded frames
    """
    def __init__(self, ignore_index=-100):
        super(DiarizationLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, diar_logits, diar_target):
        """
        Args:
            diar_logits: Tensor of shape (B, T, S)
            diar_target: LongTensor of shape (B, T)
        Returns:
            Scalar loss
        """
        # Permute to (B, S, T) for CrossEntropyLoss
        logits = diar_logits.permute(0, 2, 1)  # (B, S, T)
        loss = self.ce(logits, diar_target)    # compares against (B, T)
        return loss


class ASRLoss(nn.Module):
    """
    CTC loss for ASR branch.
    Expects:
      asr_logits: (T, B, V) log‐probabilities (time-major)
      asr_labels:  (sum L_i) concatenated label indices
      input_lengths: (B,) lengths of each input sequence in frames
      label_lengths: (B,) lengths of each label sequence
    """
    def __init__(self, blank=0, zero_infinity=True):
        super(ASRLoss, self).__init__()
        self.ctc_loss = nn.CTCLoss(blank=blank, zero_infinity=zero_infinity)

    def forward(self, asr_logits, asr_labels, input_lengths, label_lengths):
        """
        Args:
            asr_logits:     (T, B, V) raw logits or log‐probs
            asr_labels:     1D tensor of all target tokens concatenated
            input_lengths:  1D tensor of length B
            label_lengths:  1D tensor of length B
        Returns:
            Scalar CTC loss
        """
        # If logits are not log‐probs, convert
        log_probs = nn.functional.log_softmax(asr_logits, dim=2)
        loss = self.ctc_loss(
            log_probs,
            asr_labels,
            input_lengths,
            label_lengths
        )
        return loss


class ContrastiveLoss(nn.Module):
    """
    Simple pairwise contrastive loss for speaker embeddings.
    Expects:
      emb1, emb2: (B, D) embeddings
      label:      (B,) binary 1=same speaker, 0=different speaker
    """
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, emb1, emb2, label):
        """
        Args:
            emb1, emb2: FloatTensor (B, D)
            label: Bool or FloatTensor (B,) with 1.0 for positive pairs, 0.0 for negatives
        Returns:
            Scalar loss
        """
        # Euclidean distance per example
        distances = torch.norm(emb1 - emb2, p=2, dim=1)
        positive_loss = label * distances.pow(2)
        negative_loss = (1 - label) * torch.clamp(self.margin - distances, min=0).pow(2)
        loss = positive_loss + negative_loss
        return loss.mean()
