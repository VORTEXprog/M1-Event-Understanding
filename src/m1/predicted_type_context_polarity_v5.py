"""Portable inference definitions; no research datasets or trained weights included."""

from __future__ import annotations

import m1.identity_log_score_residual_v3 as state_checks
import m1.anchored_sequence_polarity_training as anchored_training

import hashlib

import json

import torch

import m1.paired_polarity_heads as paired

VERSION = 'nmcm.m1.predicted_type_context_polarity/v5'

INPUT_BASIS = {
    'version': VERSION+'/input/v1',
    'dimension': 2058,
    'phi': 'BASELINE_GELU128_THEN_ANCHORED_GELU128_SAME_PRIVATE_PAIR_PASS',
    'identity': 'SOFTMAX_DETACHED_ORIGINAL66_NATIVE_SCORE7_SAME_NORMALIZED_INPUTS_AND_CHAR_EVENTS',
    'identity_state_sha256': 'a7ed23842577bfd25d11c0008a0a9e527d21bfbccfbc4c86779650136b9a0b48',
    'private_pair_state_sha256': '4c20263304d11fb5f683d9e56064535879119eade3adcedb360f498bfa02d4a8',
    'order': ['phi256','p7','type_major_outer_p7_phi256','base_pair3'],
    'base_summary': 'EXACT_DETACHED_FROZEN_PRIVATE37_PAIR3_FROM_SAME_NATIVE_PASS_OR_ACCEPTED_CACHE',
    'new_raw_information': False,
    'correction': 'LINEAR2058_TO128_GELU_EXACT_LINEAR128_TO3_ZERO_OUTPUT_COLD',
    'base_score': json.loads(json.dumps(paired.MEMBER_POLICY)),
    'inactive': 'EXACT_ZERO_AFTER_NATIVE_READOUT',
}

INPUT_BASIS_SHA256 = hashlib.sha256(json.dumps(INPUT_BASIS,sort_keys=True,separators=(',',':')).encode()).hexdigest()

TRAINABLE_KEYS = frozenset(('correction.0.weight','correction.0.bias','correction.2.weight','correction.2.bias'))

BASE_SHAPES = {
    **{f'context.{n}.weight':(128,2048) for n in ('query','key','value')},
    'argument_binding.role_queries':(8,128),
    'argument_binding.event_query.weight':(128,2176),'argument_binding.event_query.bias':(128,),
    **{f'argument_binding.{n}.weight':(128,128) for n in ('start_query','end_query')},
    **{f'argument_binding.{n}.weight':(128,2048) for n in ('start_key','end_key')},
    'argument_binding.none.weight':(2,128),'argument_binding.none.bias':(2,),
    'event_relations.event_projection.weight':(128,2176),'event_relations.event_projection.bias':(128,),
    'event_relations.network.0.weight':(128,512),'event_relations.network.0.bias':(128,),
    'event_relations.network.2.weight':(3,128),'event_relations.network.2.bias':(3,),
    **{f'polarity.{n}.network.0.weight':(128,2176) for n in ('baseline','anchored')},
    **{f'polarity.{n}.network.0.bias':(128,) for n in ('baseline','anchored')},
    **{f'polarity.{n}.network.2.weight':(3,128) for n in ('baseline','anchored')},
    **{f'polarity.{n}.network.2.bias':(3,) for n in ('baseline','anchored')},
    'polarity.anchored.projection.weight':(128,2048),'polarity.anchored.projection.bias':(128,),
    'polarity.anchored.value.weight':(128,256),
    **{f'polarity.anchored.recurrent.weight_ih_l0{s}':(192,128) for s in ('','_reverse')},
    **{f'polarity.anchored.recurrent.weight_hh_l0{s}':(192,64) for s in ('','_reverse')},
    **{f'polarity.anchored.recurrent.bias_{n}_l0{s}':(192,) for n in ('ih','hh') for s in ('','_reverse')},
}

STATE_SHAPES = {**{'base.'+k:v for k,v in BASE_SHAPES.items()},
    'correction.0.weight':(128,2058),'correction.0.bias':(128,),
    'correction.2.weight':(3,128),'correction.2.bias':(3,)}

def require(condition,message):
    if not condition:raise ValueError(message)

def _tensor(value,shape,dtype=torch.float32):
    return (type(value) is torch.Tensor and tuple(value.shape)==tuple(shape)
        and value.dtype==dtype and value.device.type=='cpu' and value.is_contiguous()
        and not value.requires_grad and not value.is_inference() and bool(value.isfinite().all()))

def correction_inputs(phi,original7,base_scores,event_mask):
    require(type(event_mask) is torch.Tensor and event_mask.ndim==2 and event_mask.shape[0]>0
        and _tensor(event_mask,event_mask.shape,torch.bool),'ordinary native event mask required')
    shape=event_mask.shape
    require(_tensor(phi,(*shape,256)) and _tensor(original7,(*shape,7))
        and _tensor(base_scores,(*shape,3))
        and not bool(phi[~event_mask].any()) and not bool(original7[~event_mask].any())
        and not bool(base_scores[~event_mask].any()),'ordinary detached phi256/original7/frozen pair3 with inactive zeros required')
    p=original7.softmax(-1).masked_fill(~event_mask[...,None],0)
    interaction=(p[..., :,None]*phi[...,None,:]).flatten(-2)
    return torch.cat((phi,p,interaction,base_scores),-1).contiguous()
