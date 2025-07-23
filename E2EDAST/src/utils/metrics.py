# src/utils/metrics.py

import torch
from jiwer import wer
from pyannote.metrics.diarization import DiarizationErrorRate

def compute_wer(reference: str, hypothesis: str) -> float:
    """
    Compute Word Error Rate (WER) between reference and hypothesis strings.
    Requires: pip install jiwer
    Args:
        reference: ground-truth transcript
        hypothesis: predicted transcript
    Returns:
        WER as a float (e.g. 0.12 for 12% error)
    """
    return wer(reference, hypothesis)

class DERMetric:
    """
    Wrapper around pyannote.metrics DiarizationErrorRate.
    Requires: pip install pyannote.metrics
    Usage:
        der_metric = DERMetric()
        der = der_metric(reference_rttm, hypothesis_rttm)
    """
    def __init__(self, collar=0.25, skip_overlap=False):
        """
        Args:
            collar: forgiveness collar in seconds
            skip_overlap: if True, exclude overlapping speech from scoring
        """
        self.der = DiarizationErrorRate(collar=collar, skip_overlap=skip_overlap)

    def __call__(self, reference_rttm: str, hypothesis_rttm: str) -> float:
        """
        Compute DER.
        Args:
            reference_rttm: path to reference RTTM file
            hypothesis_rttm: path to hypothesis RTTM file
        Returns:
            DER float (e.g. 0.15 for 15% error)
        """
        return self.der(reference_rttm, hypothesis_rttm).item()

# Example usage:
# from src.utils.metrics import compute_wer, DERMetric
#
# # WER
# ref = "hello world"
# hyp = "hallo world"
# print("WER:", compute_wer(ref, hyp))
#
# # DER
# der_metric = DERMetric(collar=0.25, skip_overlap=False)
# der_value = der_metric("test.ref.rttm", "test.hyp.rttm")
# print("DER:", der_value)
