"""Portable inference definitions; no research datasets or trained weights included."""

from __future__ import annotations

import torch

from torch import Tensor

from m1.heads import POLARITY_ORDER

VERSION = "M1_THREE_PAIR_POLARITY_UNIFORM_AFFIRMATIVE_DECISION_v1"

AFFIRMATIVE_INTERCEPT = 0.1

def decide(logits: Tensor) -> Tensor:
    """Return class IDs after a uniform AFFIRMATIVE intercept on FP32 [B,E,3].

    This function does not inspect text, labels, language, seed, or event
    geometry. Callers mask inactive slots to NONE after decoding.
    """
    if POLARITY_ORDER != ("NONE", "AFFIRMATIVE", "NEGATIVE"):
        raise ValueError("polarity class order differs")
    if (not isinstance(logits, Tensor) or logits.dtype != torch.float32
            or logits.ndim != 3 or logits.shape[-1] != 3):
        raise ValueError("three-class FP32 polarity logits required")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("nonfinite polarity logits")
    adjusted = logits.clone()
    adjusted[..., 1] += AFFIRMATIVE_INTERCEPT
    return adjusted.argmax(dim=-1)

def decide_polarity(logits: Tensor, event_mask: Tensor) -> Tensor:
    """Decode active events while preserving exact-zero inactive output."""
    if (not isinstance(event_mask, Tensor) or event_mask.dtype != torch.bool
            or not isinstance(logits, Tensor) or logits.ndim != 3
            or event_mask.shape != logits.shape[:2] or event_mask.device != logits.device):
        raise ValueError("aligned boolean event mask required")
    result = decide(logits)
    if not bool((logits.masked_select(~event_mask.unsqueeze(-1)) == 0).all()):
        raise ValueError("inactive polarity logits differ from exact zero")
    return result.masked_fill(~event_mask, 0)
