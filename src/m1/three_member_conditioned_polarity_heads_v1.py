"""Portable inference definitions; no research datasets or trained weights included."""

from __future__ import annotations

import hashlib

import json

from dataclasses import dataclass

import torch

from torch import nn

import m1.predicted_type_context_polarity_v5 as private

import m1.paired_polarity_heads as paired

import m1.three_pair_polarity_heads as three

from m1.heads import ArgumentBindingHead

from m1.heads import EventRelationsHead

from m1.heads import SemanticHeadInputs

from m1.heads import SemanticHeadLogits

from m1.heads import _event_inputs

from m1.normalized_presence import NormalizedContextAttention

VERSION = 'nmcm.m1.three_member_conditioned_polarity_heads/v1'



INPUT_BASIS = {
    'version': VERSION,
    'selected_nonpolarity': ['context', 'argument_binding', 'event_relations'],
    'member_count': 3,
    'member_layout': 'OWN_CONTEXT3_PLUS_FROZEN_PAIR19_PLUS_COPIED_CORRECTION4',
    'member_input_basis_sha256': private.INPUT_BASIS_SHA256,
    'borrowed_identity': 'DETACHED_ORIGINAL_NATIVE_SCORE7_ON_SAME_INPUTS_NOT_CORRECTED_UNARY',
    'member_correction': 'BASE_PAIR3_PLUS_LINEAR2058_128_GELU_EXACT_LINEAR128_3',
    'inner_pair': 'EQUAL_PROBABILITY_MEAN_BEFORE_CORRECTION',
    'outer_three': 'INDIVIDUAL_LOG_SOFTMAX_THEN_LOGSUMEXP_MINUS_LOG3',
    'nonpolarity': 'SEPARATELY_SELECTED18_TENSORS_AND_THEIR_OWN_CONTEXT',
    'inactive': 'EXACT_ZERO_POLARITY_AFTER_BOTH_AGGREGATIONS',
    'affirmative_offset': 'OUTSIDE_HEAD_EXISTING_DECODER_PLUS_POINT_ONE_ONLY_ONCE',
}

INPUT_BASIS_SHA256 = hashlib.sha256(json.dumps(INPUT_BASIS, sort_keys=True,
    separators=(',', ':')).encode()).hexdigest()

NONPOL_SHAPES = {key:shape for key,shape in private.BASE_SHAPES.items()
    if key.startswith(('context.', 'argument_binding.', 'event_relations.'))}

MEMBER_BASE_SHAPES = {**{key:shape for key,shape in private.BASE_SHAPES.items()
    if key.startswith('context.')},
    **{'pair.'+key.removeprefix('polarity.'):shape for key,shape in private.BASE_SHAPES.items()
       if key.startswith('polarity.')}}

MEMBER_SHAPES = {**MEMBER_BASE_SHAPES,
    **{key:private.STATE_SHAPES[key] for key in private.TRAINABLE_KEYS}}

MEMBER_FROZEN_SHAPES = MEMBER_BASE_SHAPES

STATE_SHAPES = {**NONPOL_SHAPES,
    **{f'members.{index}.{key}':shape for index in range(3)
       for key,shape in MEMBER_SHAPES.items()}}

PARAMETER_KEYS, PARAMETER_COUNT = 96, 7902112

def require(condition, message):
    if not condition:
        raise ValueError(message)

def _sha(value):
    return type(value) is str and len(value) == 64 and value != '0'*64 and set(value) <= set('0123456789abcdef')

def _validate_state(state, shapes):
    private.state_checks._validate_state_geometry(state, shapes)

def state_digest(state):
    _validate_state(state, STATE_SHAPES)
    return private.anchored_training.tensor_digest(state)

def _part_digest(state, shapes):
    _validate_state(state, shapes)
    return private.anchored_training.tensor_digest(state)

@dataclass(frozen=True)
class SourceBindings:
    """External source commitments to seal alongside any future dense96 archive.

    These are supplied state content commitments, without a claim about training
    provenance. A dense96 restore compares every frozen member22 tensor.
    """

    selected_nonpol_sha256: str
    provider_state_sha256: tuple[str, str, str]
    frozen_member22_sha256: tuple[str, str, str]
    original_private37_sha256: str
    original_identity66_sha256: str
    input_basis_sha256: str

def _validate_bindings(bindings):
    require(type(bindings) is SourceBindings and _sha(bindings.selected_nonpol_sha256)
        and type(bindings.provider_state_sha256) is tuple
        and type(bindings.frozen_member22_sha256) is tuple
        and len(bindings.provider_state_sha256) == len(bindings.frozen_member22_sha256) == 3
        and all(_sha(value) for value in bindings.provider_state_sha256 + bindings.frozen_member22_sha256)
        and _sha(bindings.original_private37_sha256)
        and _sha(bindings.original_identity66_sha256)
        and bindings.input_basis_sha256 == INPUT_BASIS_SHA256,
        'complete supplied component state commitments required')

def _frozen_member_state(state, index):
    prefix=f'members.{index}.'
    return {key:state[prefix+key] for key in MEMBER_FROZEN_SHAPES}

def _check_state_bindings(state, bindings):
    _validate_bindings(bindings)
    require(_part_digest({key:state[key] for key in NONPOL_SHAPES},NONPOL_SHAPES)
            ==bindings.selected_nonpol_sha256,
            'dense96 selected nonpolarity18 binding differs')
    for index, expected in enumerate(bindings.frozen_member22_sha256):
        require(_part_digest(_frozen_member_state(state,index),MEMBER_FROZEN_SHAPES)==expected,
                'dense96 frozen member22 binding differs')

class CorrectedPairMember(nn.Module):
    """The exact private41 pair/correction arithmetic with no extra nonpol head."""

    def __init__(self):
        super().__init__()
        self.context = NormalizedContextAttention(2048, 128)
        self.pair = paired.PairedPolarityHead(2048, 128, 128, 64)
        self.correction = nn.Sequential(nn.Linear(2058,128), nn.GELU(approximate='none'),
                                        nn.Linear(128,3))
        with torch.no_grad():
            self.correction[2].weight.zero_()
            self.correction[2].bias.zero_()

    def forward_with_features(self, inputs, original7):
        require(type(inputs) is SemanticHeadInputs,'native private41 member inputs required')
        inputs.validate(2048)
        require(all(type(value) is torch.Tensor
                    and private._tensor(value,value.shape,value.dtype)
                    for value in vars(inputs).values()),
                'ordinary detached private41 member inputs required')
        context, attention = self.context(inputs)
        baseline, anchor = self.pair.baseline, self.pair.anchored
        left_phi = baseline.network[1](baseline.network[0](_event_inputs(inputs,context)))
        left_scores = baseline.network[2](left_phi).masked_fill(~inputs.event_mask[...,None],0)
        states = anchor.sequence_states(inputs)
        events = inputs.event_states.masked_fill(~inputs.event_mask[...,None],0)
        own_attention = anchor.content_attention(inputs)
        own_context = torch.einsum('bet,bth->beh',own_attention,anchor.value(states))
        right_phi = anchor.network[1](anchor.network[0](torch.cat((events,own_context),-1)))
        right_scores = anchor.network[2](right_phi).masked_fill(~inputs.event_mask[...,None],0)
        base3 = paired.uniform_pair_log_probs(torch.stack((left_scores,right_scores))).masked_fill(
            ~inputs.event_mask[...,None],0)
        phi = torch.cat((left_phi,right_phi),-1).masked_fill(
            ~inputs.event_mask[...,None],0).contiguous()
        x = private.correction_inputs(phi,original7,base3,inputs.event_mask)
        corrected = (base3+self.correction(x)).masked_fill(~inputs.event_mask[...,None],0)
        return corrected,base3,phi,attention

class ThreeMemberConditionedPolarityHeads(nn.Module):
    architecture = VERSION
    input_basis_sha256 = INPUT_BASIS_SHA256

    def __init__(self):
        super().__init__()
        with torch.inference_mode(False), torch.random.fork_rng(devices=[]):
            self.context = NormalizedContextAttention(2048,128)
            self.argument_binding = ArgumentBindingHead(2048,128)
            self.event_relations = EventRelationsHead(2048,128)
            self.members = nn.ModuleList([CorrectedPairMember() for _ in range(3)])
        self.encoder_dimension, self.hidden_dimension = 2048,128
        self.eval().requires_grad_(False)

    def _forward_with_member_features(self, inputs, original7):
        validate_model(self)
        require(type(inputs) is SemanticHeadInputs,'native head inputs required')
        inputs.validate(2048)
        require(private._tensor(original7,(*inputs.event_mask.shape,7))
                and not bool(original7[~inputs.event_mask].any()),
            'borrowed detached original66 score7 on exactly these native inputs required')
        require(all(type(value) is torch.Tensor
            and private._tensor(value,value.shape,value.dtype)
            for value in vars(inputs).values()), 'ordinary detached CPU FP32/bool native inputs required')
        context, attention = self.context(inputs)
        starts, ends = self.argument_binding(inputs,context)
        relations, relation_mask = self.event_relations(inputs,context)
        members = [member.forward_with_features(inputs,original7) for member in self.members]
        scores = three.uniform_three_pair_log_probs(torch.stack([value[0] for value in members]))
        scores = scores.masked_fill(~inputs.event_mask[...,None],0)
        require(private._tensor(scores,(*inputs.event_mask.shape,3))
                and not bool(scores[~inputs.event_mask].any()),
                'finite unchanged three-member probability mean/inactive zeros required')
        output = SemanticHeadLogits(starts,ends,scores,relations,attention,
                                    inputs.event_mask,relation_mask)
        return output,tuple((value[1],value[2]) for value in members)

    def forward_with_member_features(self, inputs, original7):
        with torch.inference_mode(False), torch.no_grad():
            return self._forward_with_member_features(inputs,original7)

    def forward(self, inputs, original7):
        return self.forward_with_member_features(inputs,original7)[0]

def validate_model(model):
    require(type(model) is ThreeMemberConditionedPolarityHeads
            and set(model._modules) == {'context','argument_binding','event_relations','members'}
            and type(model.context) is NormalizedContextAttention
            and type(model.argument_binding) is ArgumentBindingHead
            and type(model.event_relations) is EventRelationsHead
            and type(model.members) is nn.ModuleList and len(model.members) == 3
            and all(type(member) is CorrectedPairMember and set(member._modules) ==
                    {'context','pair','correction'}
                    and type(member.context) is NormalizedContextAttention
                    and type(member.pair) is paired.PairedPolarityHead
                    and type(member.pair.baseline) is paired.PolarityHead
                    and type(member.pair.anchored) is paired.anchored.AnchoredSequencePolarityHead
                    and type(member.correction) is nn.Sequential
                    and set(member.correction._modules)=={'0','1','2'}
                    and type(member.correction[0]) is nn.Linear
                    and type(member.correction[1]) is nn.GELU
                    and member.correction[1].approximate=='none'
                    and type(member.correction[2]) is nn.Linear
                    and (member.correction[0].in_features,member.correction[0].out_features)==(2058,128)
                    and (member.correction[2].in_features,member.correction[2].out_features)==(128,3)
                    for member in model.members)
            and (model.encoder_dimension,model.hidden_dimension)==(2048,128)
            and all(not module.training for module in model.modules())
            and not list(model.buffers()), 'exact frozen native96 topology required')
    require(all(not parameter.requires_grad and parameter.grad is None
                for parameter in model.parameters()), 'native96 cannot learn here')
    for module in model.modules():
        require(not any(getattr(module,key) for key in
                    ('_forward_hooks','_forward_pre_hooks','_backward_hooks','_backward_pre_hooks'))
                and 'forward' not in vars(module), 'native96 cannot use hooks or overrides')
    state = dict(model.state_dict())
    _validate_state(state, STATE_SHAPES)
    require(len(state)==PARAMETER_KEYS and sum(p.numel() for p in model.parameters())==PARAMETER_COUNT,
            'exact native96/7902112 parameter inventory required')
    require(type(getattr(model,'source_bindings',None)) is SourceBindings,
            'native96 requires complete external source bindings')
    _check_state_bindings(state,model.source_bindings)

def model_from_state(state, *, architecture, input_basis_sha256, expected_state_sha256,
                     source_bindings):
    require(architecture==VERSION and input_basis_sha256==INPUT_BASIS_SHA256
            and _sha(expected_state_sha256), 'explicit unadmitted native96 basis/state commitment required')
    with torch.inference_mode(False):
        _validate_state(state,STATE_SHAPES)
        _check_state_bindings(state,source_bindings)
        require(state_digest(state)==expected_state_sha256,
                'complete native96 external state digest differs')
        with torch.random.fork_rng(devices=[]):
            model=ThreeMemberConditionedPolarityHeads()
            model.load_state_dict({key:value.detach().clone() for key,value in state.items()},strict=True)
        model.source_bindings=source_bindings
        validate_model(model)
        require(state_digest(dict(model.state_dict()))==expected_state_sha256
                and not ({p.untyped_storage().data_ptr() for p in model.parameters()}
                         & {v.untyped_storage().data_ptr() for v in state.values()}),
                'independent native96 exact restore differs')
        return model
