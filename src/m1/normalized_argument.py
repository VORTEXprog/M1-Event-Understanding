"""Portable inference definitions; no research datasets or trained weights included."""

from __future__ import annotations

import torch

from torch.nn import functional as F

from m1.argument_expert import ArgumentExpert

from m1.argument_expert import ArgumentExpertLogits

from m1.heads import _validate_module_input

from m1.heads import _validate_context

from m1.heads import _event_inputs

from m1.normalized_presence import NormalizedContextAttention

class NormalizedArgumentExpert(ArgumentExpert):
    # Inherit the constructor unchanged: identical parameters and RNG draws.
    def forward(self, inputs):
        context, attention = NormalizedContextAttention.forward(self.context, inputs)
        pointer = self.pointer
        _validate_module_input(pointer, inputs)
        _validate_context(inputs, context, self.hidden_dimension)
        event = pointer.event_query(_event_inputs(inputs, context))
        query = torch.tanh(F.layer_norm(event.unsqueeze(2) + pointer.role_queries[None, None],
                                       (self.hidden_dimension,), eps=1e-5))
        tokens = inputs.token_states.masked_fill(~inputs.token_mask.unsqueeze(-1), 0)
        start = torch.einsum("berh,bth->bert", pointer.start_query(query), pointer.start_key(tokens))
        end = torch.einsum("berh,bth->bert", pointer.end_query(query), pointer.end_key(tokens))
        valid = inputs.event_mask[:, :, None, None] & inputs.token_mask[:, None, None, :]
        scale, floor = self.hidden_dimension ** .5, torch.finfo(start.dtype).min
        start, end = (value.div(scale).masked_fill(~valid, floor) for value in (start, end))
        presence_event = self.presence_event(torch.cat((inputs.event_states, context), -1))
        presence = self.presence_output(F.gelu(presence_event.unsqueeze(2) + self.presence_roles[None, None]))
        presence = presence.masked_fill(~inputs.event_mask[:, :, None, None], 0)
        return ArgumentExpertLogits(presence, start, end, attention, inputs.event_mask)
