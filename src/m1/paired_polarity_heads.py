"""Portable inference definitions; no research datasets or trained weights included."""

from __future__ import annotations

import torch

from torch import nn

import m1.anchored_sequence_polarity_heads as anchored

import m1.normalized_semantic_heads as baseline

from m1.heads import PolarityHead

from m1.heads import POLARITY_ORDER

from m1.heads import _dimension

from m1.heads import _validate_module_input

from m1.heads import _validate_context

MEMBER_POLICY = {
    'ordered_members': ['baseline', 'anchored'],
    'baseline_architecture': baseline.VERSION,
    'anchored_architecture': anchored.VERSION,
    'aggregation': 'LOGSUMEXP_INDIVIDUAL_LOG_SOFTMAX_MINUS_LOG_TWO',
    'weights': [0.5, 0.5], 'fitted_parameters': 0,
    'baseline_context': 'ORIGINAL_SHARED_NORMALIZED_CONTEXT',
    'anchored_context': 'OWN_SEQUENCE_FIXED_CONTENT_ATTENTION',
    'inactive_output': 'EXACT_ZERO',
}

def uniform_pair_log_probs(member_logits):
    if (not isinstance(member_logits, torch.Tensor) or member_logits.ndim < 2
            or member_logits.shape[0] != 2 or member_logits.shape[-1] != len(POLARITY_ORDER)
            or member_logits.dtype != torch.float32 or not bool(torch.isfinite(member_logits).all())):
        raise ValueError('paired polarity requires two finite FP32 three-class score tensors')
    result = torch.logsumexp(member_logits.log_softmax(-1), dim=0) - member_logits.new_tensor(2.).log()
    if not bool(torch.isfinite(result).all()):
        raise ValueError('paired polarity output exceeds finite FP32 domain')
    return result

class PairedPolarityHead(nn.Module):
    def __init__(self, encoder_dimension=2048, hidden_dimension=128,
                 projection_dimension=128, gru_hidden_dimension=64):
        super().__init__()
        self.encoder_dimension, self.hidden_dimension = _dimension(encoder_dimension), _dimension(hidden_dimension)
        self.projection_dimension, self.gru_hidden_dimension = _dimension(projection_dimension), _dimension(gru_hidden_dimension)
        self.baseline = PolarityHead(encoder_dimension, hidden_dimension)
        self.anchored = anchored.AnchoredSequencePolarityHead(encoder_dimension, hidden_dimension,
                                                            projection_dimension, gru_hidden_dimension)

    def forward(self, inputs, context):
        _validate_module_input(self, inputs)
        _validate_context(inputs, context, self.hidden_dimension)
        scores = torch.stack((self.baseline(inputs, context), self.anchored(inputs, context)))
        return uniform_pair_log_probs(scores).masked_fill(~inputs.event_mask.unsqueeze(-1), 0)
