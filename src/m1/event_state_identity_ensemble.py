"""Portable inference definitions; no research datasets or trained weights included."""

from __future__ import annotations

import torch

from m1.dataset import EVENT_ONTOLOGY

MEMBER_COUNT = 3

def uniform_event_state_log_probs(member_logits):
    """Stable uniform probability mean; no temperature, fitted weight or floor."""
    if (not isinstance(member_logits, torch.Tensor) or member_logits.ndim < 2
            or member_logits.shape[0] != MEMBER_COUNT or member_logits.shape[-1] != len(EVENT_ONTOLOGY)
            or member_logits.dtype != torch.float32 or member_logits.device.type != 'cpu'
            or not bool(torch.isfinite(member_logits).all())):
        raise ValueError('three finite FP32 seven-class member logit tensors required')
    scores = torch.logsumexp(member_logits.log_softmax(-1), dim=0) - member_logits.new_tensor(3.).log()
    if not bool(torch.isfinite(scores).all()):
        raise ValueError('ensemble log probabilities exceed the finite FP32 output domain')
    return scores
