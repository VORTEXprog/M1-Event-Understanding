"""Portable inference definitions; no research datasets or trained weights included."""

from __future__ import annotations

import torch

from torch import nn

import m1.event_sentence_mean_identity_v1 as base

from m1.dataset import EVENT_ONTOLOGY

EPS = 1e-5

class EventSentenceMeanNormalizedIdentityHead(base.EventSentenceMeanIdentityHead):
    """Same initialized affine layers, plus owned per-event hidden normalization."""

    def __init__(self, encoder_dimension: int = 2048, hidden_dimension: int = 128):
        super().__init__(encoder_dimension, hidden_dimension)
        # Preserve the control's affine initialization and RNG draws. Norm gain
        # and bias have deterministic standard initialization, not donor weights.
        self.network = nn.Sequential(self.network[0],
            nn.LayerNorm(self.hidden_dimension, eps=EPS, device='cpu', dtype=torch.float32),
            nn.GELU(), self.network[2])
        self.eval().requires_grad_(False)

def _shapes(dimension: int, hidden: int) -> dict:
    return {'network.0.weight': (hidden, 2 * dimension), 'network.0.bias': (hidden,),
        'network.1.weight': (hidden,), 'network.1.bias': (hidden,),
        'network.3.weight': (len(EVENT_ONTOLOGY), hidden), 'network.3.bias': (len(EVENT_ONTOLOGY),)}
