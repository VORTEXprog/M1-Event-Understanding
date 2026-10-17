"""Portable inference definitions; no research datasets or trained weights included."""

from __future__ import annotations

import torch

def _require(condition, message):
    if not condition:
        raise ValueError(message)

def _ordinary(value, shape):
    return (type(value) is torch.Tensor and tuple(value.shape) == shape
        and value.device.type == 'cpu' and value.dtype == torch.float32
        and value.layout == torch.strided and value.is_contiguous()
        and not value.is_inference() and not value.requires_grad
        and bool(torch.isfinite(value).all()))

def _validate_state_geometry(state, shapes):
    _require(type(state) is dict and set(state) == set(shapes), 'complete exact owned identity state required')
    _require(all(_ordinary(state[key], shape) for key, shape in shapes.items()),
        'independent ordinary finite contiguous CPU FP32 identity state required')
    pointers = [value.untyped_storage().data_ptr() for value in state.values()]
    _require(len(set(pointers)) == len(pointers), 'identity state tensors cannot share storage')
