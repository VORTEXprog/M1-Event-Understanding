"""Portable inference definitions; no research datasets or trained weights included."""

from __future__ import annotations

import torch

from torch import nn

from torch.nn.utils.rnn import pack_padded_sequence

from torch.nn.utils.rnn import pad_packed_sequence

from torch.nn.utils.rnn import pad_sequence

import m1.event_sentence_tail_identity_v1 as baseline

from m1.heads import SemanticHeadInputs

from m1.heads import _dimension

from m1.heads import _validate_module_input

VERSION = 'nmcm.m1.event_sentence_recurrent_identity/v1'

FORMULA = 'NATIVE_EVENT_TAIL_HIDDEN_NORMALIZED_MLP_PLUS_ZERO_INITIALIZED_EVENT_ATTENDED_BIGRU_RESIDUAL'

HIDDEN_DIMENSION = SEQUENCE_DIMENSION = ATTENTION_DIMENSION = 128

GRU_HIDDEN_DIMENSION = 64

EPS = 1e-5

def compact_features(*_args, **_kwargs):
    raise ValueError('recurrent identity requires full SemanticHeadInputs; compact features cannot carry a sequence')

class EventSentenceRecurrentIdentityHead(baseline.EventSentenceTailNormalizedIdentityHead):
    architecture = VERSION
    input_formula = FORMULA

    def __init__(self, encoder_dimension: int = 2048, hidden_dimension: int = 128):
        if _dimension(hidden_dimension) != HIDDEN_DIMENSION:
            raise ValueError('recurrent identity requires the unchanged 128 hidden coordinates')
        # Instantiate all six immutable tail MLP keys before any branch RNG draw.
        super().__init__(encoder_dimension, hidden_dimension)
        factory = {'device': 'cpu', 'dtype': torch.float32}
        self.token_normalization = nn.LayerNorm(self.encoder_dimension, eps=EPS, elementwise_affine=False, **factory)
        self.token_projection = nn.Linear(self.encoder_dimension, SEQUENCE_DIMENSION, **factory)
        self.token_activation = nn.GELU()
        self.token_encoder = nn.GRU(SEQUENCE_DIMENSION, GRU_HIDDEN_DIMENSION,
                                    batch_first=True, bidirectional=True, dropout=0, **factory)
        self.event_normalization = nn.LayerNorm(self.encoder_dimension, eps=EPS, elementwise_affine=False, **factory)
        self.event_query = nn.Linear(self.encoder_dimension, ATTENTION_DIMENSION, **factory)
        self.attention_key = nn.Linear(SEQUENCE_DIMENSION, ATTENTION_DIMENSION, **factory)
        self.attention_score = nn.Linear(ATTENTION_DIMENSION, 1, bias=False, **factory)
        self.residual_projection = nn.Linear(SEQUENCE_DIMENSION, HIDDEN_DIMENSION, bias=False, **factory)
        nn.init.zeros_(self.residual_projection.weight)
        self.eval().requires_grad_(False)

    def _validate(self, inputs: SemanticHeadInputs) -> None:
        if not isinstance(inputs, SemanticHeadInputs):
            raise ValueError('expected full SemanticHeadInputs, not semantic metadata')
        tensors = (inputs.token_states, inputs.event_states, inputs.token_mask, inputs.event_mask)
        if (any(not isinstance(value, torch.Tensor) or value.device.type != 'cpu' for value in tensors)
                or inputs.token_states.dtype != torch.float32 or inputs.event_states.dtype != torch.float32
                or any(p.dtype != torch.float32 or p.device.type != 'cpu' for p in self.parameters())):
            raise ValueError('recurrent identity requires native CPU FP32 inputs and every owned parameter')
        _validate_module_input(self, inputs)

    def features(self, *_args, **_kwargs):
        return compact_features()

    def forward_features(self, *_args, **_kwargs):
        return compact_features()

    def encode_sequence(self, inputs: SemanticHeadInputs) -> torch.Tensor:
        self._validate(inputs)
        lengths = inputs.token_mask.sum(-1).to(dtype=torch.int64, device='cpu')
        valid = [inputs.token_states[i, inputs.token_mask[i]] for i in range(len(lengths))]
        compact = pad_sequence(valid, batch_first=True)
        projected = self.token_activation(self.token_projection(self.token_normalization(compact)))
        packed = pack_padded_sequence(projected, lengths, batch_first=True, enforce_sorted=False)
        encoded, _ = self.token_encoder(packed)
        unpacked, _ = pad_packed_sequence(encoded, batch_first=True)
        scattered = inputs.token_states.new_zeros(*inputs.token_mask.shape, SEQUENCE_DIMENSION)
        for i, length in enumerate(lengths.tolist()):
            scattered[i, inputs.token_mask[i]] = unpacked[i, :length]
        if not bool(torch.isfinite(scattered).all()):
            raise ValueError('recurrent sequence encoder produced nonfinite states')
        return scattered

    def sequence_context(self, inputs: SemanticHeadInputs) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encode_sequence(inputs)
        events = inputs.event_states.masked_fill(~inputs.event_mask[..., None], 0)
        query = self.event_query(self.event_normalization(events))
        keys = self.attention_key(encoded)
        scores = self.attention_score(torch.tanh(query[:, :, None, :] + keys[:, None, :, :])).squeeze(-1)
        scores = scores.masked_fill(~inputs.token_mask[:, None, :], -torch.inf)
        attention = scores.softmax(-1).masked_fill(~inputs.token_mask[:, None, :], 0)
        attention = attention.masked_fill(~inputs.event_mask[..., None], 0)
        context = torch.einsum('bet,bth->beh', attention, encoded)
        context = context.masked_fill(~inputs.event_mask[..., None], 0)
        if not bool(torch.isfinite(context).all()) or not bool(torch.isfinite(attention).all()):
            raise ValueError('recurrent attention produced nonfinite context')
        return context, attention

    def forward(self, inputs: SemanticHeadInputs, *, ablate_sequence: bool = False,
                ablate_context: bool = False) -> torch.Tensor:
        self._validate(inputs)
        if type(ablate_sequence) is not bool or type(ablate_context) is not bool:
            raise ValueError('recurrent diagnostic interventions must be boolean')
        values = baseline.EventSentenceTailNormalizedIdentityHead.features(self, inputs)
        if ablate_context:
            values = values.clone()
            values[..., self.encoder_dimension:] = 0
        hidden = self.network[0](values)
        if not ablate_sequence and not ablate_context:
            context, _ = self.sequence_context(inputs)
            hidden = hidden + self.residual_projection(context)
        # Same native LayerNorm, GELU and seven-class affine output, in order.
        output = self.network[3](self.network[2](self.network[1](hidden)))
        output = output.masked_fill(~inputs.event_mask[..., None], 0)
        if not bool(torch.isfinite(output).all()):
            raise ValueError('recurrent identity readout produced nonfinite scores')
        return output

def _shapes(dimension: int, hidden: int = 128) -> dict:
    if _dimension(hidden) != HIDDEN_DIMENSION:
        raise ValueError('recurrent identity requires the unchanged 128 hidden coordinates')
    dimension = _dimension(dimension)
    shapes = baseline.normalized_base._shapes(dimension, hidden)
    shapes.update({'token_projection.weight': (128, dimension), 'token_projection.bias': (128,),
        'event_query.weight': (128, dimension), 'event_query.bias': (128,),
        'attention_key.weight': (128, 128), 'attention_key.bias': (128,),
        'attention_score.weight': (1, 128), 'residual_projection.weight': (128, 128)})
    for suffix in ('', '_reverse'):
        shapes.update({f'token_encoder.weight_ih_l0{suffix}': (192, 128),
            f'token_encoder.weight_hh_l0{suffix}': (192, 64),
            f'token_encoder.bias_ih_l0{suffix}': (192,), f'token_encoder.bias_hh_l0{suffix}': (192,)})
    return shapes

def _validate_state(state: dict, dimension: int, hidden: int = 128) -> None:
    shapes = _shapes(dimension, hidden)
    if (not isinstance(state, dict) or set(state) != set(shapes)
            or any(not isinstance(value, torch.Tensor) or value.shape != shapes[name]
                   or value.dtype != torch.float32 or value.device.type != 'cpu'
                   or not bool(torch.isfinite(value).all()) for name, value in state.items())):
        raise ValueError('exact finite CPU FP32 recurrent identity inventory required')
