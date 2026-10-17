"""Portable inference definitions; no research datasets or trained weights included."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from torch import Tensor

from torch import nn

from torch.nn import functional as F

from m1.heads import ArgumentBindingHead

from m1.heads import SemanticHeadInputs

from m1.heads import SharedContextAttention

@dataclass(frozen=True)
class ArgumentExpertLogits:
    presence_logits: Tensor  # [B,E,8,2]: absent,present
    argument_start_logits: Tensor  # [B,E,8,T]: real text units only
    argument_end_logits: Tensor
    context_attention: Tensor
    event_mask: Tensor

class ArgumentExpert(nn.Module):
    def __init__(self, encoder_dimension: int = 2048, hidden_dimension: int = 128):
        super().__init__()
        self.encoder_dimension = encoder_dimension
        self.hidden_dimension = hidden_dimension
        self.context = SharedContextAttention(encoder_dimension, hidden_dimension)
        self.pointer = ArgumentBindingHead(encoder_dimension, hidden_dimension)
        # Reuse the historical pointer exactly but never optimize or decode its
        # discarded NONE branch. Presence is a separate binary decision.
        self.pointer.none.requires_grad_(False)
        self.presence_event = nn.Linear(encoder_dimension + hidden_dimension, hidden_dimension)
        self.presence_roles = nn.Parameter(torch.empty(8, hidden_dimension))
        nn.init.normal_(self.presence_roles, std=hidden_dimension ** -0.5)
        self.presence_output = nn.Linear(hidden_dimension, 2)

    def forward(self, inputs: SemanticHeadInputs) -> ArgumentExpertLogits:
        context, attention = self.context(inputs)
        start, end = self.pointer(inputs, context)
        event = self.presence_event(torch.cat((inputs.event_states, context), -1))
        queries = F.gelu(event.unsqueeze(2) + self.presence_roles[None, None])
        presence = self.presence_output(queries).masked_fill(~inputs.event_mask[:, :, None, None], 0)
        return ArgumentExpertLogits(presence, start[..., :-1], end[..., :-1], attention, inputs.event_mask)
