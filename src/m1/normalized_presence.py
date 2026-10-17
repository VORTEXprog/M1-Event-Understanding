"""Portable inference definitions; no research datasets or trained weights included."""

from __future__ import annotations

import torch

from torch.nn import functional as F

from m1.heads import SharedContextAttention

from m1.heads import _validate_module_input

class NormalizedContextAttention(SharedContextAttention):
    """Identical learned Q/K/V projections; bounded cosine scores only change."""
    def forward(self, inputs):
        _validate_module_input(self, inputs)
        tokens = inputs.token_states.masked_fill(~inputs.token_mask.unsqueeze(-1), 0)
        events = inputs.event_states.masked_fill(~inputs.event_mask.unsqueeze(-1), 0)
        queries = F.normalize(self.query(events), dim=-1, eps=1e-8)
        keys = F.normalize(self.key(tokens), dim=-1, eps=1e-8)
        # Unit-vector inner products are in [-1,1]; sqrt(H) is fixed by the
        # unchanged width, not a selected temperature or semantic threshold.
        scores = torch.einsum("beh,bth->bet", queries, keys) * self.hidden_dimension ** .5
        scores = scores.masked_fill(~inputs.token_mask[:, None, :], torch.finfo(scores.dtype).min)
        attention = scores.softmax(-1).masked_fill(~inputs.event_mask.unsqueeze(-1), 0)
        return torch.einsum("bet,bth->beh", attention, self.value(tokens)), attention
