"""Portable inference definitions; no research datasets or trained weights included."""

from __future__ import annotations

import hashlib

import json

from typing import Mapping

from torch import Tensor

class E2PLModelingError(ValueError):
    """A detector tensor, model, or fitting request violates frozen policy."""

def state_dict_sha256(state: Mapping[str, Tensor]) -> str:
    if not state:
        raise E2PLModelingError("detector state dict must be nonempty")
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name]
        if not isinstance(name, str) or not name or not isinstance(value, Tensor):
            raise E2PLModelingError("detector state dict entry differs")
        tensor = value.detach().cpu().contiguous()
        metadata = json.dumps(
            {"name": name, "dtype": str(tensor.dtype), "shape": list(tensor.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(metadata).to_bytes(8, "little"))
        digest.update(metadata)
        payload = tensor.numpy().tobytes(order="C")
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
    return digest.hexdigest()
