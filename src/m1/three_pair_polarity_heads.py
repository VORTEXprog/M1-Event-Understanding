"""Portable inference definitions; no research datasets or trained weights included."""

from __future__ import annotations

import math

import torch

from m1.heads import POLARITY_ORDER

MEMBER_COUNT = 3

def uniform_three_pair_log_probs(member_logits):
    """Exact amendment54 arithmetic, including normalization of each pair."""
    if (not isinstance(member_logits, torch.Tensor) or member_logits.ndim < 2
            or member_logits.shape[0] != MEMBER_COUNT or member_logits.shape[-1] != len(POLARITY_ORDER)
            or member_logits.dtype != torch.float32 or not bool(torch.isfinite(member_logits).all())):
        raise ValueError('three finite FP32 three-class pair score tensors required')
    result = torch.logsumexp(member_logits.log_softmax(-1), dim=0) - math.log(3)
    if not bool(torch.isfinite(result).all()):
        raise ValueError('three-pair log probabilities exceed the finite FP32 domain')
    return result
