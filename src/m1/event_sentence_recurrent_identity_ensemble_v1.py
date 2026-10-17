"""Portable inference definitions; no research datasets or trained weights included."""

from __future__ import annotations

import torch

from torch import nn

import m1.event_sentence_recurrent_identity_v1 as native

from m1.event_state_identity_ensemble import uniform_event_state_log_probs

from m1.heads import SemanticHeadInputs

VERSION = 'nmcm.m1.event_sentence_recurrent_identity_probability_ensemble/v1'

MEMBER_KEYS, OWNED_KEYS, PARAMETERS = 22, 66, 3472917

def _require(ok, reason):
    if not ok:
        raise ValueError(reason)

def _validate_members(model):
    _require(type(model) is SentenceRecurrentIdentityProbabilityEnsemble and set(model._modules) == {'members'}
             and type(model.members) is nn.ModuleList and len(model.members) == 3
             and len({id(m) for m in model.members}) == 3, 'three independent exact recurrent members required')
    shapes = native._shapes(model.encoder_dimension, model.hidden_dimension)
    _require(len(shapes) == MEMBER_KEYS and set(model.state_dict()) == {
        f'members.{index}.{name}' for index in range(3) for name in shapes}, 'exact 66-key recurrent ensemble required')
    storages = []
    modules = {'network', 'token_normalization', 'token_projection', 'token_activation', 'token_encoder',
               'event_normalization', 'event_query', 'attention_key', 'attention_score', 'residual_projection'}
    for member in model.members:
        _require(type(member) is native.EventSentenceRecurrentIdentityHead
                 and (member.encoder_dimension, member.hidden_dimension) == (model.encoder_dimension, 128)
                 and set(member._modules) == modules and not list(member.buffers()), 'exact stateless recurrent member type required')
        _require(type(member.network) is nn.Sequential and tuple(type(layer) for layer in member.network) ==
                 (nn.Linear, nn.LayerNorm, nn.GELU, nn.Linear)
                 and member.network[2].approximate == 'none' and member.network[1].eps == native.EPS
                 and member.network[1].normalized_shape == (128,) and member.network[1].elementwise_affine is True,
                 'native recurrent readout topology differs')
        for name in ('token_normalization', 'event_normalization'):
            layer = getattr(member, name)
            _require(type(layer) is nn.LayerNorm and layer.normalized_shape == (model.encoder_dimension,)
                     and layer.eps == native.EPS and layer.elementwise_affine is False, 'native nonaffine normalization differs')
        _require(type(member.token_activation) is nn.GELU and member.token_activation.approximate == 'none'
                 and type(member.token_encoder) is nn.GRU and member.token_encoder.input_size == 128
                 and member.token_encoder.hidden_size == 64 and member.token_encoder.num_layers == 1
                 and member.token_encoder.batch_first is True and member.token_encoder.bidirectional is True
                 and member.token_encoder.dropout == 0 and member.token_encoder.bias is True,
                 'native packed BiGRU topology differs')
        for name in ('token_projection', 'event_query', 'attention_key', 'attention_score', 'residual_projection'):
            _require(type(getattr(member, name)) is nn.Linear, 'native recurrent affine topology differs')
        state = member.state_dict()
        native._validate_state(state, model.encoder_dimension, 128)
        storages.extend(value.untyped_storage().data_ptr() for value in state.values())
    _require(len(storages) == OWNED_KEYS and len(set(storages)) == OWNED_KEYS,
             'recurrent ensemble parameters share storage')

class SentenceRecurrentIdentityProbabilityEnsemble(nn.Module):
    architecture = VERSION
    input_formula = native.FORMULA

    def __init__(self, members):
        super().__init__()
        members = list(members)
        _require(len(members) == 3 and len({id(m) for m in members}) == 3
                 and all(type(m) is native.EventSentenceRecurrentIdentityHead for m in members)
                 and len({(m.encoder_dimension, m.hidden_dimension) for m in members}) == 1,
                 'three independent same-architecture recurrent heads required')
        self.encoder_dimension, self.hidden_dimension = members[0].encoder_dimension, members[0].hidden_dimension
        self.members = nn.ModuleList(members)
        _validate_members(self)
        self.eval().requires_grad_(False)

    def features(self, *_args, **_kwargs):
        return native.compact_features()

    def forward_features(self, *_args, **_kwargs):
        return native.compact_features()

    def forward(self, inputs: SemanticHeadInputs, *, ablate_context=False, ablate_sequence=False):
        _require(type(ablate_context) is bool and type(ablate_sequence) is bool,
                 'recurrent diagnostic interventions must be boolean')
        _require(isinstance(inputs, SemanticHeadInputs), 'full native SemanticHeadInputs required')
        # The native members compute their own sequence, with fresh zero h0.
        logits = torch.stack([member(inputs, ablate_context=ablate_context, ablate_sequence=ablate_sequence)
                              for member in self.members])
        return uniform_event_state_log_probs(logits).masked_fill(~inputs.event_mask[..., None], 0)
