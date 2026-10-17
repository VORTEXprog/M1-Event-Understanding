"""Portable inference definitions; no research datasets or trained weights included."""

from __future__ import annotations

from dataclasses import replace

import math

import torch

from torch import Tensor

from torch import nn

from torch.nn import functional as F

from m1.event_anchor_argument_compatibility_v1 import AnchoredArgumentInputs

from m1.event_anchor_argument_compatibility_v1 import geometry_features

from m1.structured_joint_role_argument_v1 import StructuredArgumentPotentials

VERSION = 'M1_EVENT_CONDITIONED_REAL_PAIR_RESIDUAL_COMPONENT_v1'

STATE_KEYS = ('hidden.weight', 'hidden.bias', 'output.weight', 'output.bias')

SHAPES = dict(zip(STATE_KEYS, ((32, 104), (32,), (32, 32), (32,)), strict=True))

PARAMETERS = 4416

INITIALIZATION = 'ORDINARY_LINEAR104_TO32_THEN_LINEAR32_TO32_DRAWS; ZERO_FINAL_WEIGHT_AND_BIAS_LAST'

def _require(value, message):
    if not value:
        raise ValueError(message)

class EventConditionedPairResidual(nn.Module):
    """V'=V+FFN(cat(V,A,V*A,RELATIVE8)); A is the whole HEAD pair's V.

V' is [B,E,P,32], independent of the role axis. Existing Q[B,E,8,32] is
unchanged. Real unary adds dot(dV,Q)/sqrt(32). Both real factors are recomputed
from tanh(V'+Q) and the supplied native left/right matrices. No joint loss,
decoder, pair enumeration, role gate, NULL vector or candidate mask changes.
"""
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(104, 32, device='cpu', dtype=torch.float32)
        self.output = nn.Linear(32, 32, device='cpu', dtype=torch.float32)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def _validate_owned(self):
        parameters = dict(self.named_parameters())
        _require(set(self._modules) == {'hidden', 'output'} and not self._parameters
            and not self._buffers and type(self.hidden) is nn.Linear and type(self.output) is nn.Linear
            and set(parameters) == set(STATE_KEYS)
            and all(tuple(value.shape) == SHAPES[name] and value.device.type == 'cpu'
                and value.dtype == torch.float32 and bool(torch.isfinite(value).all())
                for name, value in parameters.items())
            and len({value.untyped_storage().data_ptr() for value in parameters.values()}) == 4
            and sum(value.numel() for value in parameters.values()) == PARAMETERS,
            'exact four independent finite CPU FP32 residual tensors required')

    def forward(self, inputs: AnchoredArgumentInputs, base: StructuredArgumentPotentials,
                *, factor_left_weight: Tensor, factor_right_weight: Tensor) -> StructuredArgumentPotentials:
        self._validate_owned()
        _require(type(inputs) is AnchoredArgumentInputs and type(base) is StructuredArgumentPotentials,
                 'explicit anchored inputs and original structured potentials required')
        inputs.validate()
        base.validate()
        _require(base.unary_logits.device.type == 'cpu' and base.pair_vectors is not None
            and base.query_vectors is not None
            and torch.equal(inputs.semantic.token_mask, base.token_mask)
            and torch.equal(inputs.semantic.event_mask, base.event_mask),
            'same CPU semantic masks and complete native V32/Q32 ports required')
        for weight in (factor_left_weight, factor_right_weight):
            _require(isinstance(weight, Tensor) and weight.shape == (32, 32)
                and weight.dtype == torch.float32 and weight.device.type == 'cpu'
                and bool(torch.isfinite(weight).all()),
                'supplied native factor matrix must be finite CPU FP32[32,32]')

        vectors, queries = base.pair_vectors, base.query_vectors
        starts, ends = inputs.event_unit_spans.unbind(-1)
        valid = base.event_mask[:, :, None] & base.pair_mask[:, None, :]
        matches = ((base.pair_indices[:, None, :, 0] == starts[..., None])
            & (base.pair_indices[:, None, :, 1] == ends[..., None]-1) & valid)
        _require(torch.equal(matches.sum(-1), base.event_mask.long()),
                 'each active whole half-open HEAD must identify exactly one complete pair')
        head_index = matches.long().argmax(-1)
        anchor = vectors.gather(2, head_index[..., None, None].expand(*head_index.shape, 1, 32))
        anchor = anchor.expand_as(vectors)
        geometry = geometry_features(inputs, base.pair_indices, base.pair_mask, 'RELATIVE')[:, :, :-1]
        features = torch.cat((vectors, anchor, vectors*anchor, geometry), -1)
        _require(bool(torch.isfinite(features).all()), 'event/pair product features overflowed')
        delta = self.output(F.gelu(self.hidden(features))).masked_fill(~valid[..., None], 0)
        _require(bool(torch.isfinite(delta).all()), 'event-conditioned pair residual is nonfinite')
        # The no-grad zero subcase preserves every original bit, including -0,
        # and does not reinterpret unrelated/potentially padded feature values.
        if not torch.is_grad_enabled() and not bool(delta.count_nonzero()):
            return replace(base)

        updated_vectors = torch.where(valid[..., None], vectors+delta, vectors)
        unary_delta = torch.einsum('bepk,berk->berp', delta, queries)/math.sqrt(32)
        role_valid = valid[:, :, None, :]
        real_unary = torch.where(role_valid, base.unary_logits[..., :-1]+unary_delta,
                                 base.unary_logits[..., :-1])
        feature = torch.tanh(updated_vectors[:, :, None, :, :]+queries[:, :, :, None, :])
        left = F.linear(feature, factor_left_weight)/math.sqrt(32)
        right = F.linear(feature, factor_right_weight)
        real_left = torch.where(role_valid[..., None], left, base.left_factors[..., :-1, :])
        real_right = torch.where(role_valid[..., None], right, base.right_factors[..., :-1, :])
        result = replace(base, unary_logits=torch.cat((real_unary, base.unary_logits[..., -1:]), -1),
            left_factors=torch.cat((real_left, base.left_factors[..., -1:, :]), -2),
            right_factors=torch.cat((real_right, base.right_factors[..., -1:, :]), -2),
            pair_vectors=updated_vectors)
        result.validate()
        return result
