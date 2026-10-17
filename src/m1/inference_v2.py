"""Portable inference definitions; no research datasets or trained weights included."""

from __future__ import annotations

from dataclasses import dataclass

from typing import Sequence

import torch

from torch import Tensor

from torch.nn import functional as F

from m1.canonical_units import segment_text

from m1.heads import SemanticHeadInputs

@dataclass(frozen=True)
class ObservedSentence:
    sentence_id: str
    text: str
    language: str
    unit_states: Tensor
    event_spans: tuple[tuple[int, int], ...]

    def validate(self, dimension: int):
        if (not isinstance(self.sentence_id, str) or not self.sentence_id
                or not isinstance(self.text, str) or not self.text or self.language not in ("en", "ru")):
            raise ValueError("observed raw sentence identity differs")
        units = segment_text(self.text)
        if (self.unit_states.shape != (len(units), dimension) or self.unit_states.dtype != torch.float32
                or not torch.isfinite(self.unit_states).all() or self.unit_states.requires_grad):
            raise ValueError("observed raw feature geometry differs")
        if not isinstance(self.event_spans, tuple) or len(set(self.event_spans)) != len(self.event_spans):
            raise ValueError("observed span inventory must be a unique tuple")
        for start, end in self.event_spans:
            if (type(start) is not int or type(end) is not int or not 0 <= start < end <= len(self.text)
                    or not any(unit.char_start < end and unit.char_end > start for unit in units)):
                raise ValueError("observed predicate geometry invalid")

def observed_inputs(rows: Sequence[ObservedSentence], dimension: int) -> SemanticHeadInputs:
    if not rows or len({row.sentence_id for row in rows}) != len(rows):
        raise ValueError("observed sentence inventory differs")
    for row in rows:
        row.validate(dimension)
    device = rows[0].unit_states.device
    if any(row.unit_states.device != device for row in rows):
        raise ValueError("feature devices differ")
    width = max(len(row.unit_states) for row in rows)
    count = max(2, max(len(row.event_spans) for row in rows))
    tokens = torch.zeros(len(rows), width, dimension, device=device)
    events = torch.zeros(len(rows), count, dimension, device=device)
    token_mask = torch.zeros(len(rows), width, dtype=torch.bool, device=device)
    event_mask = torch.zeros(len(rows), count, dtype=torch.bool, device=device)
    for index, row in enumerate(rows):
        units = segment_text(row.text)
        normalized = F.layer_norm(row.unit_states, (dimension,), eps=1e-5)
        tokens[index, :len(units)] = normalized
        token_mask[index, :len(units)] = True
        for event_index, (start, end) in enumerate(row.event_spans):
            indices = [i for i, unit in enumerate(units) if unit.char_start < end and unit.char_end > start]
            events[index, event_index] = normalized[indices].mean(0)
            event_mask[index, event_index] = True
    return SemanticHeadInputs(tokens, events, token_mask, event_mask)
