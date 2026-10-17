"""Portable inference definitions; no research datasets or trained weights included."""

from __future__ import annotations

import torch

from torch import nn

from torch.nn import functional as F

from torch.nn.utils.rnn import pack_padded_sequence

from torch.nn.utils.rnn import pad_packed_sequence

from torch.nn.utils.rnn import pad_sequence

from m1.normalized_argument import NormalizedArgumentExpert

from m1.normalized_presence import NormalizedContextAttention

from m1.argument_expert import ArgumentExpertLogits

from m1.heads import _validate_module_input

from m1.heads import _validate_context

from m1.heads import _event_inputs

class SequenceKeyArgumentExpert(NormalizedArgumentExpert):
    def __init__(self, encoder_dimension=2048, hidden_dimension=128):
        if (type(hidden_dimension) is not int or hidden_dimension <= 0
                or hidden_dimension % 2):
            raise ValueError("sequence pointer hidden dimension must be positive and even")
        # Complete the historical constructor first. All unaffected parameters
        # retain their exact same-seed initial values and RNG draws; no weights
        # from a trained checkpoint or localizer are used.
        super().__init__(encoder_dimension, hidden_dimension)
        self.key_projection = nn.Linear(encoder_dimension, hidden_dimension)
        self.key_encoder = nn.GRU(hidden_dimension, hidden_dimension // 2,
                                  batch_first=True, bidirectional=True)
        self.pointer.start_key = nn.Linear(hidden_dimension, hidden_dimension, bias=False)
        self.pointer.end_key = nn.Linear(hidden_dimension, hidden_dimension, bias=False)

    def encode_keys(self, inputs):
        _validate_module_input(self, inputs)
        lengths = inputs.token_mask.sum(1)
        tokens = inputs.token_states.masked_fill(~inputs.token_mask.unsqueeze(-1), 0)
        projected = F.gelu(self.key_projection(tokens))
        # SemanticHeadInputs permits holes. Compact valid units in original
        # order, encode only those units, then scatter to original coordinates.
        compact = pad_sequence(projected[inputs.token_mask].split(lengths.tolist()), batch_first=True)
        packed = pack_padded_sequence(compact, lengths.cpu(), batch_first=True, enforce_sorted=False)
        encoded, _ = self.key_encoder(packed)
        states, _ = pad_packed_sequence(encoded, batch_first=True, total_length=compact.shape[1])
        compact_mask = torch.arange(compact.shape[1], device=lengths.device)[None] < lengths[:, None]
        result = states.new_zeros((*inputs.token_mask.shape, self.hidden_dimension))
        result[inputs.token_mask] = F.layer_norm(states, (self.hidden_dimension,), eps=1e-5)[compact_mask]
        return result

    def forward(self, inputs):
        keys = self.encode_keys(inputs)
        context, attention = NormalizedContextAttention.forward(self.context, inputs)
        pointer = self.pointer
        _validate_context(inputs, context, self.hidden_dimension)
        event = pointer.event_query(_event_inputs(inputs, context))
        query = torch.tanh(F.layer_norm(event.unsqueeze(2) + pointer.role_queries[None, None],
                                       (self.hidden_dimension,), eps=1e-5))
        start = torch.einsum("berh,bth->bert", pointer.start_query(query), pointer.start_key(keys))
        end = torch.einsum("berh,bth->bert", pointer.end_query(query), pointer.end_key(keys))
        valid = inputs.event_mask[:, :, None, None] & inputs.token_mask[:, None, None, :]
        scale, floor = self.hidden_dimension ** .5, torch.finfo(start.dtype).min
        start, end = (value.div(scale).masked_fill(~valid, floor) for value in (start, end))
        presence_event = self.presence_event(torch.cat((inputs.event_states, context), -1))
        presence = self.presence_output(F.gelu(presence_event.unsqueeze(2) + self.presence_roles[None, None]))
        presence = presence.masked_fill(~inputs.event_mask[:, :, None, None], 0)
        return ArgumentExpertLogits(presence, start, end, attention, inputs.event_mask)
