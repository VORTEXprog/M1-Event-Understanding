"""Portable inference definitions; no research datasets or trained weights included."""

from __future__ import annotations

from typing import Mapping

import torch

from torch import Tensor

from torch.nn import functional as F

def _state_factors(state: Mapping[str, Tensor]) -> dict[str, Tensor]:
    """Accept strict owner state or its four authenticated factor tensors."""
    if not isinstance(state, Mapping):
        raise ValueError('authenticated structured factor state required')
    # A full structured owner state has no prefix for its twelve appended keys.
    if 'base.role_pair_gate' in state:
        raise ValueError('structured factor state cannot be a nested base owner')
    selected = {}
    for key, shape in (('null_embedding', (32,)), ('factor_left.weight', (32, 32)),
                       ('factor_right.weight', (32, 32)), ('role_pair_gate', (28,))):
        value = state.get(key)
        if (type(value) is not Tensor or value.shape != shape or
                value.dtype != torch.float32 or value.device.type != 'cpu' or
                not bool(torch.isfinite(value).all())):
            raise ValueError('exact finite CPU FP32 structured factor state required')
        selected[key] = value.detach().contiguous()
    return selected

def reconstruct_factors(pair: Tensor, query: Tensor, state: Mapping[str, Tensor]) -> tuple[Tensor, Tensor]:
    """Pure FP32 arithmetic from saved pre-factor features and saved weights."""
    factors = _state_factors(state)
    if (pair.ndim != 2 or pair.shape[-1] != 32 or query.ndim != 3 or
            query.shape[1:] != (8, 32) or pair.dtype != torch.float32 or
            query.dtype != torch.float32 or pair.device.type != 'cpu' or
            query.device.type != 'cpu' or not bool(torch.isfinite(pair).all()) or
            not bool(torch.isfinite(query).all())):
        raise ValueError('exact saved pair/query feature geometry required')
    feature = torch.tanh(pair[None, None] + query[:, :, None, :])
    null = torch.tanh(factors['null_embedding'][None, None, None, :] + query[:, :, None, :])
    feature = torch.cat((feature, null), 2)
    left = F.linear(feature, factors['factor_left.weight']) / (32 ** .5)
    right = F.linear(feature, factors['factor_right.weight'])
    return left, right
