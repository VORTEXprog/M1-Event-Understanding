"""Portable inference definitions; no research datasets or trained weights included."""

from __future__ import annotations

from typing import Sequence

import torch

from torch import Tensor

PAD_BYTE_ID = 256

def utf8_byte_batch(surfaces: Sequence[str], maximum_bytes: int = 96) -> tuple[Tensor, Tensor]:
    """Encode exact raw mention strings with byte IDs 0..255 and PAD=256."""

    if not surfaces:
        raise ValueError("raw mention batch must be nonempty")
    encoded: list[bytes] = []
    for surface in surfaces:
        if not isinstance(surface, str) or not surface:
            raise ValueError("raw mention surface must be a nonempty string")
        value = surface.encode("utf-8")
        if not value or len(value) > maximum_bytes:
            raise ValueError("raw mention UTF-8 length exceeds the interface")
        encoded.append(value)
    width = max(map(len, encoded))
    tokens = torch.full((len(encoded), width), PAD_BYTE_ID, dtype=torch.int64)
    lengths = torch.tensor([len(value) for value in encoded], dtype=torch.int64)
    for row, value in enumerate(encoded):
        tokens[row, :len(value)] = torch.tensor(list(value), dtype=torch.int64)
    return tokens, lengths
