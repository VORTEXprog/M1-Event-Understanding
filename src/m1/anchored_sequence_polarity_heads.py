"""Portable inference definitions; no research datasets or trained weights included."""

from __future__ import annotations

import torch

from torch import nn

from torch.nn import functional as F

from torch.nn.utils.rnn import pack_padded_sequence

from torch.nn.utils.rnn import pad_packed_sequence

from torch.nn.utils.rnn import pad_sequence

from m1.heads import POLARITY_ORDER

from m1.heads import _dimension

from m1.heads import _validate_module_input

from m1.heads import _validate_context

VERSION = "M1_POLARITY_OWNED_CONTENT_ANCHORED_SEQUENCE_HEADS_v1"

PROJECTION_DIMENSION, GRU_HIDDEN_DIMENSION = 128, 64

class AnchoredSequencePolarityHead(nn.Module):
    """Own projected units and bidirectional states; no cross-owner features."""

    def __init__(self, encoder_dimension=2048, hidden_dimension=128,
                 projection_dimension=PROJECTION_DIMENSION, gru_hidden_dimension=GRU_HIDDEN_DIMENSION):
        super().__init__()
        self.encoder_dimension = _dimension(encoder_dimension)
        self.hidden_dimension = _dimension(hidden_dimension)
        self.projection_dimension = _dimension(projection_dimension)
        self.gru_hidden_dimension = _dimension(gru_hidden_dimension)
        sequence_dimension = projection_dimension + 2 * gru_hidden_dimension
        self.projection = nn.Linear(encoder_dimension, projection_dimension)
        self.recurrent = nn.GRU(projection_dimension, gru_hidden_dimension,
                                batch_first=True, bidirectional=True)
        self.value = nn.Linear(sequence_dimension, hidden_dimension, bias=False)
        self.network = nn.Sequential(nn.Linear(encoder_dimension + hidden_dimension, hidden_dimension),
                                     nn.GELU(), nn.Linear(hidden_dimension, len(POLARITY_ORDER)))

    def sequence_states(self, inputs):
        _validate_module_input(self, inputs)
        # Inputs have already received the unchanged non-affine per-unit LN.
        # Pack selected units in original order; even a noncontiguous native
        # mask cannot insert a padding step into the recurrent computation.
        lengths = inputs.token_mask.sum(-1).tolist()
        projected = F.gelu(self.projection(inputs.token_states[inputs.token_mask]))
        padded = pad_sequence(projected.split(lengths), batch_first=True)
        packed = pack_padded_sequence(padded, lengths, batch_first=True, enforce_sorted=False)
        recurrent, _ = self.recurrent(packed)
        context, _ = pad_packed_sequence(recurrent, batch_first=True, total_length=padded.shape[1])
        valid = torch.arange(padded.shape[1], device=padded.device)[None, :] < torch.tensor(lengths, device=padded.device)[:, None]
        combined = torch.cat((padded, context), -1)
        result = combined.new_zeros((*inputs.token_mask.shape, combined.shape[-1]))
        result[inputs.token_mask] = combined[valid]
        return result

    def content_attention(self, inputs):
        """One dimension-derived soft native anchor; no learned Q/K or indices."""
        _validate_module_input(self, inputs)
        events = inputs.event_states.masked_fill(~inputs.event_mask.unsqueeze(-1), 0)
        tokens = inputs.token_states.masked_fill(~inputs.token_mask.unsqueeze(-1), 0)
        queries = F.normalize(events, dim=-1, eps=1e-8)
        keys = F.normalize(tokens, dim=-1, eps=1e-8)
        scores = torch.einsum("bed,btd->bet", queries, keys) * self.encoder_dimension ** .5
        scores = scores.masked_fill(~inputs.token_mask[:, None, :], torch.finfo(scores.dtype).min)
        return scores.softmax(-1).masked_fill(~inputs.event_mask[:, :, None], 0)

    def forward(self, inputs, context):
        _validate_module_input(self, inputs)
        _validate_context(inputs, context, self.hidden_dimension)
        states = self.sequence_states(inputs)
        events = inputs.event_states.masked_fill(~inputs.event_mask.unsqueeze(-1), 0)
        attention = self.content_attention(inputs)
        own_context = torch.einsum("bet,bth->beh", attention, self.value(states))
        logits = self.network(torch.cat((events, own_context), -1))
        return logits.masked_fill(~inputs.event_mask[:, :, None], 0)
