"""Portable inference definitions; no research datasets or trained weights included."""

from __future__ import annotations

import torch

from torch import nn

from torch.nn import functional as F

import m1.sequence_pointer as native

from m1.argument_expert import ArgumentExpertLogits

from m1.heads import _validate_context

from m1.heads import _event_inputs

from m1.normalized_presence import NormalizedContextAttention

VERSION = 'M1_ROLE_CONDITIONED_PRESENCE_SEQUENCE_KEY_ARGUMENT_v1'

FORMULA = 'NATIVE_GELU_Z_PLUS_ZERO_RESIDUAL_OF_L2_PROJECTED_ROLE_Z_ATTENDED_NATIVE_BIGRU_KEYS'

NEW_KEYS = ('presence_query.weight', 'presence_key.weight', 'presence_residual.weight')

NATIVE_KEYS = ('context.query.weight', 'context.key.weight', 'context.value.weight',
    'pointer.role_queries', 'pointer.event_query.weight', 'pointer.event_query.bias',
    'pointer.start_query.weight', 'pointer.end_query.weight', 'pointer.start_key.weight',
    'pointer.end_key.weight', 'pointer.none.weight', 'pointer.none.bias', 'presence_roles',
    'presence_event.weight', 'presence_event.bias', 'presence_output.weight', 'presence_output.bias',
    'key_projection.weight', 'key_projection.bias', 'key_encoder.weight_ih_l0',
    'key_encoder.weight_hh_l0', 'key_encoder.bias_ih_l0', 'key_encoder.bias_hh_l0',
    'key_encoder.weight_ih_l0_reverse', 'key_encoder.weight_hh_l0_reverse',
    'key_encoder.bias_ih_l0_reverse', 'key_encoder.bias_hh_l0_reverse')

def _dimensions(encoder, hidden):
    if (type(encoder) is not int or encoder <= 0 or type(hidden) is not int
            or hidden <= 0 or hidden % 2):
        raise ValueError('positive integer encoder and positive even hidden dimensions required')

def _shapes(encoder, hidden):
    _dimensions(encoder, hidden)
    shapes = {name: (hidden, encoder) for name in
              ('context.query.weight', 'context.key.weight', 'context.value.weight', 'key_projection.weight')}
    shapes.update({name: (hidden, hidden) for name in ('pointer.start_query.weight',
        'pointer.end_query.weight', 'pointer.start_key.weight', 'pointer.end_key.weight', *NEW_KEYS)})
    shapes.update({'pointer.role_queries': (8, hidden), 'presence_roles': (8, hidden),
        'pointer.event_query.weight': (hidden, encoder + hidden), 'pointer.event_query.bias': (hidden,),
        'pointer.none.weight': (2, hidden), 'pointer.none.bias': (2,),
        'presence_event.weight': (hidden, encoder + hidden), 'presence_event.bias': (hidden,),
        'presence_output.weight': (2, hidden), 'presence_output.bias': (2,), 'key_projection.bias': (hidden,)})
    for suffix in ('', '_reverse'):
        shapes.update({f'key_encoder.weight_ih_l0{suffix}': (3 * hidden // 2, hidden),
            f'key_encoder.weight_hh_l0{suffix}': (3 * hidden // 2, hidden // 2),
            f'key_encoder.bias_ih_l0{suffix}': (3 * hidden // 2,),
            f'key_encoder.bias_hh_l0{suffix}': (3 * hidden // 2,)})
    return shapes

class PresenceSequenceKeyArgumentExpert(native.SequenceKeyArgumentExpert):
    architecture = VERSION
    input_formula = FORMULA

    def __init__(self, encoder_dimension=2048, hidden_dimension=128):
        _dimensions(encoder_dimension, hidden_dimension)
        super().__init__(encoder_dimension, hidden_dimension)
        if any(p.device.type != 'cpu' or p.dtype != torch.float32 for p in self.parameters()):
            raise ValueError('native initialization requires CPU FP32 defaults')
        factory = {'device': 'cpu', 'dtype': torch.float32}
        self.presence_query = nn.Linear(hidden_dimension, hidden_dimension, bias=False, **factory)
        self.presence_key = nn.Linear(hidden_dimension, hidden_dimension, bias=False, **factory)
        self.presence_residual = nn.Linear(hidden_dimension, hidden_dimension, bias=False, **factory)
        nn.init.zeros_(self.presence_residual.weight)
        self.eval().requires_grad_(False)

    def enable_training(self):
        self.train().requires_grad_(True)
        self.pointer.none.requires_grad_(False)
        return self

    def presence_context(self, inputs, keys, preactivation):
        """One tensor query per event/role; invalid tokens/events are exact zero."""
        query = F.normalize(self.presence_query(F.layer_norm(preactivation,
            (self.hidden_dimension,), eps=1e-5)), dim=-1, eps=1e-8)
        key = F.normalize(self.presence_key(keys), dim=-1, eps=1e-8)
        scores = torch.einsum('berh,bth->bert', query, key) * self.hidden_dimension ** .5
        scores = scores.masked_fill(~inputs.token_mask[:, None, None, :], -torch.inf)
        attention = scores.softmax(-1).masked_fill(~inputs.event_mask[:, :, None, None], 0)
        attention = attention.masked_fill(~inputs.token_mask[:, None, None, :], 0)
        return torch.einsum('bert,bth->berh', attention, keys), attention

    def forward(self, inputs, *, ablate_presence_residual=False):
        if type(ablate_presence_residual) is not bool:
            raise ValueError('presence residual intervention requires a boolean flag')
        if any(p.device.type != 'cpu' or p.dtype != torch.float32 for p in self.parameters()):
            raise ValueError('presence-sequence forward requires CPU FP32 parameters')
        # Same native encoder, fresh GRU initial state, exactly once per call.
        keys = self.encode_keys(inputs)
        context, attention = NormalizedContextAttention.forward(self.context, inputs)
        pointer = self.pointer
        _validate_context(inputs, context, self.hidden_dimension)
        event = pointer.event_query(_event_inputs(inputs, context))
        query = torch.tanh(F.layer_norm(event.unsqueeze(2) + pointer.role_queries[None, None],
                                       (self.hidden_dimension,), eps=1e-5))
        start = torch.einsum('berh,bth->bert', pointer.start_query(query), pointer.start_key(keys))
        end = torch.einsum('berh,bth->bert', pointer.end_query(query), pointer.end_key(keys))
        valid = inputs.event_mask[:, :, None, None] & inputs.token_mask[:, None, None, :]
        scale, floor = self.hidden_dimension ** .5, torch.finfo(start.dtype).min
        start, end = (value.div(scale).masked_fill(~valid, floor) for value in (start, end))
        presence_event = self.presence_event(torch.cat((inputs.event_states, context), -1))
        preactivation = presence_event.unsqueeze(2) + self.presence_roles[None, None]
        features = F.gelu(preactivation)
        if not ablate_presence_residual:
            pooled, _ = self.presence_context(inputs, keys, preactivation)
            features = features + self.presence_residual(pooled)
        presence = self.presence_output(features).masked_fill(~inputs.event_mask[:, :, None, None], 0)
        return ArgumentExpertLogits(presence, start, end, attention, inputs.event_mask)
