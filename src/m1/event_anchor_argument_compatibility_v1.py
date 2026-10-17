"""Portable inference definitions; no research datasets or trained weights included."""

from __future__ import annotations

from collections.abc import Mapping

from dataclasses import dataclass

from dataclasses import replace

import torch

from torch import Tensor

from torch import nn

from torch.nn import functional as F

from m1.heads import SemanticHeadInputs

from m1.joint_argument_span_score_v1 import enumerate_unit_pairs

from m1.structured_joint_role_argument_v1 import StructuredArgumentPotentials

VERSION = 'M1_EVENT_ANCHOR_ARGUMENT_COMPATIBILITY_COMPONENT_v1'

MODES = ('ABSOLUTE_CONTROL', 'RELATIVE')

STATE_KEYS = ('null_vector', 'hidden.weight', 'hidden.bias', 'output.weight', 'output.bias')

SHAPES = dict(zip(STATE_KEYS, ((32,), (32, 104), (32,), (1, 32), (1,)), strict=True))

PARAMETERS = 3425

def _require(condition, message):
    if not condition:
        raise ValueError(message)

@dataclass(frozen=True)
class AnchoredArgumentInputs:
    """New input DTO; old SemanticHeadInputs ABI remains unchanged.

``event_unit_spans`` are upstream predicted LOC half-open unit intervals. Their
provenance belongs to the caller: no embedding-derived anchor or Gold fallback
is supplied here. Inactive event slots contain exactly(0,0).
"""
    semantic: SemanticHeadInputs
    event_unit_spans: Tensor             # int64[B,E,2], half-open

    def validate(self) -> None:
        _require(type(self.semantic) is SemanticHeadInputs, 'exact old SemanticHeadInputs required')
        semantic = self.semantic
        _require(isinstance(semantic.token_states, Tensor) and semantic.token_states.ndim == 3,
            'rank-three semantic token states required')
        semantic.validate(semantic.token_states.shape[-1])
        _require(semantic.token_states.device.type == 'cpu' and semantic.token_states.dtype == torch.float32,
            'anchored semantic ports require finite CPU FP32')
        mask = semantic.token_mask; lengths = mask.sum(-1)
        prefix = torch.arange(mask.shape[1], device=mask.device)[None, :] < lengths[:, None]
        _require(torch.equal(mask, prefix), 'anchor geometry requires a nonempty valid token prefix; holes denied')
        anchors = self.event_unit_spans
        _require(isinstance(anchors, Tensor) and anchors.dtype == torch.int64
            and anchors.device == mask.device and anchors.shape == (*semantic.event_mask.shape, 2),
            'event anchors must be int64 CPU[B,E,2] half-open unit intervals')
        start, end = anchors.unbind(-1); active = semantic.event_mask
        _require(not bool(((start < 0) | (start >= end) | (end > lengths[:, None])).masked_select(active).any())
            and not bool(anchors.masked_select(~active[..., None].expand_as(anchors)).count_nonzero()),
            'active event anchors must be in prefix bounds; inactive anchors must be exact zero')

def geometry_features(inputs: AnchoredArgumentInputs, pair_indices: Tensor,
        pair_mask: Tensor, mode: str) -> Tensor:
    """Continuous8 features[B,E,P+1,8]; inclusive pairs become half-open.

The last choice is NULL: only its last feature is1 for active events. All
inactive/padded features are exact zero. Normalization uses actual prefix n,
never padded tensor width. Both modes share exactly the same8-wide geometry.
"""
    _require(type(inputs) is AnchoredArgumentInputs and type(mode) is str and mode in MODES,
        'explicit anchored inputs and declared geometry mode required')
    inputs.validate(); semantic = inputs.semantic
    expected_pairs, expected_mask = enumerate_unit_pairs(semantic.token_mask)
    _require(isinstance(pair_indices, Tensor) and isinstance(pair_mask, Tensor)
        and pair_indices.dtype == torch.int64 and pair_mask.dtype == torch.bool
        and pair_indices.device == pair_mask.device == semantic.token_mask.device
        and torch.equal(pair_indices, expected_pairs) and torch.equal(pair_mask, expected_mask),
        'all complete original inclusive span pairs and exact padding mask required')
    b, e = semantic.event_mask.shape; p = pair_mask.shape[1]
    n = semantic.token_mask.sum(-1).to(torch.float32)[:, None, None]
    a0 = inputs.event_unit_spans[..., 0, None].to(torch.float32)
    a1 = inputs.event_unit_spans[..., 1, None].to(torch.float32)
    c0 = pair_indices[:, None, :, 0].to(torch.float32)
    c1 = (pair_indices[:, None, :, 1]+1).to(torch.float32)
    width, anchor_width = c1-c0, a1-a0
    zero = torch.zeros((b, e, p), dtype=torch.float32, device=n.device)
    if mode == 'RELATIVE':
        parts = ((c0-a0)/n, (c1-a1)/n, (a0-c1).clamp_min(0)/n,
            (c0-a1).clamp_min(0)/n, (torch.minimum(a1,c1)-torch.maximum(a0,c0)).clamp_min(0)/n,
            width/n, anchor_width/n, zero)
    else:
        parts = (a0/n, a1/n, c0/n, c1/n, width/n, anchor_width/n, 1/n, zero)
    real = torch.stack(torch.broadcast_tensors(*parts), dim=-1)
    valid = semantic.event_mask[:, :, None] & pair_mask[:, None, :]
    real = real.masked_fill(~valid[..., None], 0)
    null = torch.zeros((b, e, 1, 8), dtype=torch.float32, device=n.device)
    null[..., 0, 7] = semantic.event_mask.to(torch.float32)
    return torch.cat((real, null), dim=2)

class EventAnchorArgumentCompatibility(nn.Module):
    """Shared cat(V32,Q32,V*Q32,geometry8)→GELU32→scalar residual.

Exactly five new tensors/3425 parameters are owned. The learned NULL V starts
at zero; final Linear weight/bias start at zero after ordinary constructor
draws. In no_grad/inference a wholly zero residual returns original unary bits,
including signed zero. In gradient mode normal addition preserves learning;
valid IEEE negative zero may become positive zero, with numerical parity.
Invalid positions retain their original bits in both modes.
"""
    def __init__(self, mode: str = 'RELATIVE'):
        super().__init__()
        _require(type(mode) is str and mode in MODES, 'explicit ABSOLUTE_CONTROL or RELATIVE mode required')
        self.mode = mode
        self.null_vector = nn.Parameter(torch.zeros(32, dtype=torch.float32))
        self.hidden = nn.Linear(104, 32, device='cpu', dtype=torch.float32)
        self.output = nn.Linear(32, 1, device='cpu', dtype=torch.float32)
        nn.init.zeros_(self.output.weight); nn.init.zeros_(self.output.bias)

    def _validate_owned(self) -> None:
        parameters = dict(self.named_parameters())
        _require(self.mode in MODES and set(self._modules) == {'hidden', 'output'}
            and set(self._parameters) == {'null_vector'} and not self._buffers
            and type(self.hidden) is nn.Linear and type(self.output) is nn.Linear
            and set(parameters) == set(STATE_KEYS)
            and all(tuple(value.shape) == SHAPES[name] and value.device.type == 'cpu'
                and value.dtype == torch.float32 and bool(torch.isfinite(value).all())
                for name,value in parameters.items())
            and len({value.untyped_storage().data_ptr() for value in parameters.values()}) == 5
            and sum(value.numel() for value in parameters.values()) == PARAMETERS,
            'exact independent finite CPU FP32 five-tensor compatibility topology required')

    def forward(self, inputs: AnchoredArgumentInputs,
            base: StructuredArgumentPotentials) -> StructuredArgumentPotentials:
        self._validate_owned()
        _require(type(inputs) is AnchoredArgumentInputs and type(base) is StructuredArgumentPotentials,
            'new anchored DTO and existing structured potentials required')
        inputs.validate(); base.validate(); semantic = inputs.semantic
        _require(base.unary_logits.device.type == 'cpu' and base.pair_vectors is not None
            and base.query_vectors is not None and torch.equal(base.token_mask,semantic.token_mask)
            and torch.equal(base.event_mask,semantic.event_mask),
            'same semantic/output masks and existing complete32-wide V/Q ports required')
        geometry = geometry_features(inputs,base.pair_indices,base.pair_mask,self.mode)
        b,e,roles,choices = base.unary_logits.shape
        v = torch.cat((base.pair_vectors,self.null_vector[None,None,None,:].expand(b,e,1,32)),dim=2)
        v = v[:, :, None].expand(b,e,roles,choices,32)
        q = base.query_vectors[:, :, :, None].expand(b,e,roles,choices,32)
        features = torch.cat((v,q,v*q,geometry[:, :, None].expand(b,e,roles,choices,8)),dim=-1)
        _require(bool(torch.isfinite(features).all()), 'compatibility V/Q product overflowed')
        residual = self.output(F.gelu(self.hidden(features))).squeeze(-1)
        valid = base.event_mask[:, :, None, None] & torch.cat((base.pair_mask,
            torch.ones((b,1),dtype=torch.bool,device=base.unary_logits.device)),dim=-1)[:,None,None,:]
        residual = residual.masked_fill(~valid,0)
        _require(bool(torch.isfinite(residual).all()), 'compatibility residual is nonfinite')
        if not torch.is_grad_enabled() and not bool(residual.count_nonzero()):
            unary = base.unary_logits
        else:
            unary = torch.where(valid,base.unary_logits+residual,base.unary_logits)
        result = replace(base,unary_logits=unary)
        result.validate()
        return result

def model_from_state(state: Mapping[str,Tensor], *, mode: str) -> EventAnchorArgumentCompatibility:
    """Strict standalone component restore; independent CPU/eval/frozen.

Caller declares the geometry mode; tensor shapes cannot establish its semantics.
This factory owns no old65 parameters, deserialization, corpus or runtime route.
"""
    _require(type(mode) is str and mode in MODES and isinstance(state,Mapping)
        and set(state) == set(STATE_KEYS) and all(isinstance(value,Tensor)
            and tuple(value.shape) == SHAPES[name] and value.device.type == 'cpu'
            and value.dtype == torch.float32 and not value.requires_grad
            and bool(torch.isfinite(value).all()) for name,value in state.items()),
        'exact detached finite CPU FP32 five-tensor state and explicit mode required')
    with torch.random.fork_rng(devices=[]):
        model = EventAnchorArgumentCompatibility(mode)
    model.load_state_dict({name:value.detach().clone() for name,value in state.items()},strict=True)
    model.eval().requires_grad_(False); model._validate_owned()
    return model
