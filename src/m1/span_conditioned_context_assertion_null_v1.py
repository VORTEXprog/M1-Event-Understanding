"""Portable inference definitions; no research datasets or trained weights included."""

from __future__ import annotations

import hashlib

import json

import math

import torch

from torch import nn

from torch.nn import functional as F

import m1.span_context_assertion_null_v1 as common

negative_log_likelihood,predict,event_gains = common.negative_log_likelihood,common.predict,common.event_gains

VERSION = 'M1_ALL_SPAN_CONDITIONED_FULL_CONTEXT_ASSERTION_NULL_COMPONENT_SOURCE_v1'

ENERGY_VERSION = common.ENERGY_VERSION

WIDTH = 32

STATE_SHAPES = {**common.STATE_SHAPES,
    'key_weight': (32,2048), 'key_bias': (32,),
    'query_weight': (32,32), 'query_bias': (32,)}

STATE_KEYS = tuple(STATE_SHAPES)

PARAMETERS = 332002

LAW = {**common.LAW, 'version': VERSION,
    'sentence_context': 'EVERY_SPAN_QUERY_ATTENDS_ALL_ORIGINAL_UNIT_RAW_KEYS_AND_GELU_VALUES',
    'memory_values': 'GELU_SHARED_CONTEXT_LINEAR2048_32_ON_EVERY_ORIGINAL_RAW_UNIT',
    'span_queries': 'SHARED_QUERY_LINEAR32_32_OF_UNCHANGED_DIRECT_RAW_SPAN32',
    'memory_keys': 'SHARED_KEY_LINEAR2048_32_OF_EVERY_ORIGINAL_RAW_UNIT',
    'attention': 'FP32_SOFTMAX_QUERY_KEY_TRANSPOSE_DIV_SQRT32_OVER_FULL_ORIGINAL_UNIT_AXIS',
    'attention_shape': 'ALL_SEVEN_BY_ALL_P_BY_FULL_ORIGINAL_U; P=U*(U+1)/2',
    'row_geometry': 'SINGLE_COMPLETE_ORIGINAL_ROW_NO_PADDING_MASK_HOLES_OR_SHORTLIST_PORT',
    'memory_bound': 'EXTERNAL_BOUNDED_ORIGINAL_ROW_ADMISSION; NO_INTERNAL_PRUNING_OR_WIDTH_CAP',
    'key_bias': 'SHARED_QUERY_DOT_KEY_BIAS_CANCELS_IN_EXACT_SOFTMAX; NO_NONZERO_INFLUENCE_CLAIM',
    'mechanism_attribution': 'OWNED_PAIE_STYLE_QUERY_CONTEXT_ADAPTATION_NOT_UPSTREAM_BART_OR_JOINT_ROLE_REPRODUCTION',
    'old_final_unit_zero_parity_claimed': False,
    'common_eight_state_coinitialization': 'EXTERNAL_ONLY_ALL_EIGHT_COMMON_NAMES_AND_BITS_PRESERVABLE',
    'parameter_tensors': 12, 'parameters': PARAMETERS}

require = common.require

digest = common.digest

tensor_digest = common.tensor_digest

_fp32 = common._fp32

exact = common.exact

TYPES = common.TYPES

SpanContextEventNullBank = common.SpanContextEventNullBank

validate_bank = common.validate_bank

def validate_state(state):
    require(type(state) is dict and tuple(state) == STATE_KEYS,
            'complete ordered twelve-tensor supplied state required')
    for key,shape in STATE_SHAPES.items():
        _fp32(state[key],shape,detached=True)
    require(len({value.untyped_storage().data_ptr() for value in state.values()}) == 12
        and sum(value.numel() for value in state.values()) == PARAMETERS,
        'complete disjoint332002 supplied state required')
    return state

def state_digest(state):
    validate_state(state)
    result = hashlib.sha256()
    for key,value in state.items():
        metadata = json.dumps([key,'FLOAT32',list(value.shape)],separators=(',',':')).encode()
        payload = value.numpy().tobytes()
        result.update(len(metadata).to_bytes(8,'little'));result.update(metadata)
        result.update(len(payload).to_bytes(8,'little'));result.update(payload)
    return result.hexdigest()

class SpanConditionedContextAssertionNullHead(nn.Module):
    """Clone externally supplied state; all7 TYPE evidence remains separate."""
    def __init__(self,state):
        validate_state(state)
        before = state_digest(state)
        super().__init__()
        with torch.inference_mode(False):
            for key,value in state.items():
                self.register_parameter(key,nn.Parameter(value.detach().clone(),requires_grad=False))
        self._learning = False
        self.eval()
        self._identities = tuple((key,id(value),value.untyped_storage().data_ptr())
            for key,value in self.named_parameters())
        self._frozen_sha256 = before
        require(not {value.untyped_storage().data_ptr() for value in self.parameters()}.intersection(
            value.untyped_storage().data_ptr() for value in state.values()),
            'supplied state storage cannot be reused')
        require(state_digest(state) == before,'constructor changed supplied state')

    def _state(self):
        return {key:value.detach() for key,value in self.named_parameters()}

    def validate(self):
        require(type(self) is SpanConditionedContextAssertionNullHead and not self._modules and not self._buffers
            and tuple(self._parameters) == STATE_KEYS and type(self._learning) is bool
            and self.training is self._learning
            and tuple((key,id(value),value.untyped_storage().data_ptr())
                for key,value in self.named_parameters()) == self._identities
            and all(value.requires_grad is self._learning for value in self.parameters()),
            'twelve-tensor component topology or explicit learning mode changed')
        current = state_digest(self._state())
        if not self._learning:
            require(current == self._frozen_sha256 and all(value.grad is None for value in self.parameters()),
                    'frozen supplied state changed outside explicit learning API')
        return current

    def set_learning(self,enabled):
        require(type(enabled) is bool,'explicit learning boolean required')
        self.validate()
        self.train(enabled).requires_grad_(enabled)
        self._learning = enabled
        for value in self.parameters():
            value.grad = None
        if not enabled:
            self._frozen_sha256 = state_digest(self._state())
        self.validate()
        return self

    def _representations(self,raw7,pairs):
        start,end = pairs.unbind(-1)
        # Identical project/gather/prefix-mean FP32 law to the common eight.
        first = F.linear(raw7,self.span_weight[:,:2048])
        last = F.linear(raw7,self.span_weight[:,2048:4096])
        projected_mean = F.linear(raw7,self.span_weight[:,4096:])
        prefix = F.pad(projected_mean.cumsum(1),(0,0,1,0))
        mean = (prefix[:,end]-prefix[:,start])/(end-start).to(torch.float32)[None,:,None]
        span = F.gelu(first[:,start]+last[:,end-1]+mean+self.span_bias)
        values = F.gelu(F.linear(raw7,self.context_weight,self.context_bias))
        keys = F.linear(raw7,self.key_weight,self.key_bias)
        queries = F.linear(span,self.query_weight,self.query_bias)
        attention = torch.softmax(torch.matmul(queries,keys.transpose(1,2))/math.sqrt(WIDTH),dim=-1)
        require(bool(attention.isfinite().all()),'complete span-conditioned attention is nonfinite')
        context = torch.matmul(attention,values)
        return span,context,attention

    def _arithmetic(self,raw7,type_scores,pairs):
        span,context,_ = self._representations(raw7,pairs)
        interaction = torch.cat((span,context,span*context),dim=-1)
        logits = F.linear(F.gelu(F.linear(interaction,self.joint_weight,self.joint_bias)),
                          self.decision_weight,self.decision_bias)
        return validate_bank(SpanContextEventNullBank(ENERGY_VERSION,
            F.log_softmax(type_scores,dim=-1).contiguous(),logits[:,:,0].t().contiguous(),
            logits[:,:,1].t().contiguous(),pairs,raw7.shape[1]))

    def forward(self,raw7,frozen_type_scores):
        before = self.validate()
        require(type(raw7) is torch.Tensor and raw7.ndim == 3 and raw7.shape[0] == 7
            and raw7.shape[1] > 0 and raw7.shape[2] == 2048,
            'whole original RAW[7,U,2048] required; no padding or mask port')
        _fp32(raw7,tuple(raw7.shape),detached=True)
        with torch.inference_mode(False):
            pairs = exact.complete_pairs(raw7.shape[1])
        _fp32(frozen_type_scores,(len(pairs),7),detached=True)
        require(not {raw7.untyped_storage().data_ptr(),frozen_type_scores.untyped_storage().data_ptr()}.intersection(
            value.untyped_storage().data_ptr() for value in self.parameters()),
            'supplied input/state alias rejected')
        input_bits = (tensor_digest(raw7),tensor_digest(frozen_type_scores))
        if self._learning:
            require(torch.is_grad_enabled() and not torch.is_inference_mode_enabled(),
                    'ordinary autograd required for explicit learning')
            result = self._arithmetic(raw7,frozen_type_scores,pairs)
        else:
            with torch.inference_mode(False),torch.no_grad():
                result = self._arithmetic(raw7,frozen_type_scores,pairs)
        require(input_bits == (tensor_digest(raw7),tensor_digest(frozen_type_scores))
            and self.validate() == before,'forward changed supplied input or state')
        return result
