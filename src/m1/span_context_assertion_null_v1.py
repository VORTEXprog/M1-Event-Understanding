"""Portable inference definitions; no research datasets or trained weights included."""

from __future__ import annotations

def align_frozen_type_scores(type_scores,end_major_pairs,unit_count):
    """Exact native14 geometry-only bijection; no HEAD/TYPE label selection."""
    require(type(unit_count) is int and unit_count>0,'complete positive original unit count required')
    count=unit_count*(unit_count+1)//2;_fp32(type_scores,(count,7),detached=True)
    expected=torch.tensor([(start,end)for end in range(1,unit_count+1)for start in range(end)],dtype=torch.int64,device='cpu')
    require(type(end_major_pairs) is torch.Tensor and end_major_pairs.device.type=='cpu'
        and end_major_pairs.dtype==torch.int64 and end_major_pairs.is_contiguous()
        and not end_major_pairs.is_inference() and torch.equal(end_major_pairs,expected),
        'exact complete native14 end-major interval geometry required')
    before=(tensor_digest(type_scores),tensor_digest(end_major_pairs))
    order=torch.tensor([end*(end-1)//2+start for start,end in exact.complete_pairs(unit_count).tolist()],dtype=torch.int64,device='cpu')
    result=type_scores[order].clone().contiguous()
    require(before==(tensor_digest(type_scores),tensor_digest(end_major_pairs)),'TYPE geometry join mutated supplied input')
    return result

from dataclasses import dataclass

import hashlib

import json

import torch

from torch import nn

from torch.nn import functional as F

import m1.typed_event_set_localization_loss_v1 as exact

from m1.semantic_prefix_conditional_identity_contract_v1 import TYPES

from m1.semantic_prefix_conditional_identity_contract_v1 import FORWARD_BASIS_SHA256

VERSION='M1_ALL_SPAN_CONTEXT_ASSERTION_NULL_COMPONENT_SOURCE_v1'

ENERGY_VERSION='M1_SPAN_CONTEXT_EVENT_MINUS_SPAN_NULL_ZERO_BASELINE_ENERGY_v1'

WIDTH=32

STATE_SHAPES={
    'span_weight':(32,6144),'span_bias':(32,),
    'context_weight':(32,2048),'context_bias':(32,),
    'joint_weight':(32,96),'joint_bias':(32,),
    'decision_weight':(2,32),'decision_bias':(2,),
}

STATE_KEYS=tuple(STATE_SHAPES)

PARAMETERS=265378

LAW={
    'version':VERSION,'energy_version':ENERGY_VERSION,'candidate_order':list(TYPES),
    'forward_basis_sha256':FORWARD_BASIS_SHA256,'RAW_width':2048,'hidden_width':WIDTH,
    'span_features':'DIRECT_RAW_START_END_MEAN_SHARED_LINEAR6144_32_GELU',
    'span_FP32_law':'PROJECT_ORIGINAL_UNITS_START_END_MEAN_SLICES_THEN_GATHER_AND_PREFIX_MEAN_THEN_ADD_BIAS',
    'sentence_context':'DIRECT_RAW_FINAL_ORIGINAL_CANONICAL_UNIT_SHARED_LINEAR2048_32_GELU',
    'interaction':'CONCAT_SPAN_CONTEXT_ELEMENTWISE_PRODUCT_96_SHARED_LINEAR32_GELU_LINEAR2',
    'decision_channels':['EVENT','SPAN_NULL'],
    'frozen_TYPE':'ALL_SEVEN_COMPLETE_SUPPLIED_FP32_LOGITS_LOGSOFTMAX_FIXED_COEFFICIENT_ONE_NO_GRAD',
    'TYPE_geometry_join':'NATIVE14_END_MAJOR_COMPLETE_INTERVALS_TO_COMPONENT_START_MAJOR_BIJECTION_NO_GOLD',
    'event_energy':'FROZEN_TYPE_LOGP_PLUS_EVENT_MINUS_SPAN_NULL',
    'absent_energy':'EMPTY_SET_AND_UNCOVERED_SINGLETON_UNIT_STEPS_EXACT_ZERO',
    'DP_adapter':'EXPLICIT_NEW_GAIN_BANK_TO_EXISTING_EXACT_SET_PRIMITIVE_WITH_ZERO_OUTSIDE',
    'objective':exact.OBJECTIVE,'decoder':exact.DECODER,
    'intervals':'ALL_START_MAJOR_HALFOPEN_WHOLE_UNIT_INTERVALS_NO_PRUNING',
    'maximum_HEAD_width':'FULL_ORIGINAL_ROW_U_NO_FIXED_CAP',
    'Gold_TYPE_or_HEAD_or_status_in_forward':False,
    'BIO_or_GRU_or_threshold_or_count_cap':False,
    'cross_event_transition_factors':False,
    'historical_NONE_energies_retagged':False,
    'parameter_tensors':8,'parameters':PARAMETERS,
    'initializer_or_optimizer_or_qualification_claimed':False,
}

def require(ok,message):
    if not ok:raise ValueError(message)

def digest(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,
        separators=(',',':'),allow_nan=False).encode()).hexdigest()

def tensor_digest(value):
    return hashlib.sha256(value.detach().contiguous().numpy().tobytes()).hexdigest()

def _fp32(value,shape,*,detached=False):
    require(type(value) is torch.Tensor and tuple(value.shape)==shape and value.dtype==torch.float32
        and value.device.type=='cpu' and value.is_contiguous() and not value.is_inference()
        and bool(value.isfinite().all()) and (not detached or not value.requires_grad),
        'complete ordinary finite CPU FP32 tensor with exact shape required')

def validate_state(state):
    require(type(state) is dict and tuple(state)==STATE_KEYS,'complete ordered new eight-tensor state required')
    for key,shape in STATE_SHAPES.items():_fp32(state[key],shape,detached=True)
    require(len({value.untyped_storage().data_ptr() for value in state.values()})==8
        and sum(value.numel() for value in state.values())==PARAMETERS,'complete disjoint265378 state required')
    return state

def state_digest(state):
    validate_state(state);result=hashlib.sha256()
    for key,value in state.items():
        metadata=json.dumps([key,'FLOAT32',list(value.shape)],separators=(',',':')).encode()
        payload=value.numpy().tobytes()
        result.update(len(metadata).to_bytes(8,'little'));result.update(metadata)
        result.update(len(payload).to_bytes(8,'little'));result.update(payload)
    return result.hexdigest()

@dataclass(frozen=True)
class SpanContextEventNullBank:
    """New one-row score semantics; no outside_logits or historical NONE port."""
    energy_version:str
    type_logp:torch.Tensor
    event_logits:torch.Tensor
    span_null_logits:torch.Tensor
    pair_indices:torch.Tensor
    unit_count:int

def validate_bank(bank):
    require(type(bank) is SpanContextEventNullBank and bank.energy_version==ENERGY_VERSION
        and type(bank.unit_count) is int and bank.unit_count>0,'explicit new span-NULL energy ABI required')
    count=bank.unit_count*(bank.unit_count+1)//2
    for value in (bank.type_logp,bank.event_logits,bank.span_null_logits):_fp32(value,(count,7))
    require(not bank.type_logp.requires_grad,'conditional TYPE evidence must stay frozen')
    pairs=bank.pair_indices
    require(type(pairs) is torch.Tensor and pairs.dtype==torch.int64 and pairs.device.type=='cpu'
        and pairs.is_contiguous() and not pairs.is_inference() and not pairs.requires_grad
        and torch.equal(pairs,exact.complete_pairs(bank.unit_count)),'every complete original interval exactly once required')
    require(bool(torch.allclose(bank.type_logp.logsumexp(-1),bank.type_logp.new_zeros(count),atol=2e-6,rtol=0)),
        'frozen conditional TYPE log-probability normalization differs')
    return bank

def event_gains(bank):
    validate_bank(bank)
    return (bank.type_logp+bank.event_logits-bank.span_null_logits).contiguous()

def to_zero_baseline_dp(bank):
    """Declared new gain representation; reuse only exact set arithmetic.

SPAN_NULL is already subtracted per event alternative. It is never added as
an uncovered-unit score or repeated for unselected overlapping intervals.
"""
    gains=event_gains(bank)
    return exact.validate_bank(exact.PackedEventSetScores(gains,
        torch.zeros(bank.unit_count,dtype=torch.float32,device='cpu'),bank.pair_indices,
        torch.tensor([0,len(gains)],dtype=torch.int64,device='cpu'),
        torch.tensor([0,bank.unit_count],dtype=torch.int64,device='cpu')))

def negative_log_likelihood(bank,gold_spans,*,global_row_count=None,reduction='mean'):
    """Whole HEAD/TYPE loss only: every proper subspan remains an alternative."""
    return exact.negative_log_likelihood(to_zero_baseline_dp(bank),(gold_spans,),
        global_row_count=global_row_count,reduction=reduction)

def predict(bank):
    """Complete unpruned joint typed-set decision; no annotation/count port."""
    return exact.decode_map(to_zero_baseline_dp(bank))[0]

class SpanContextAssertionNullHead(nn.Module):
    """Clone supplied new state only; frozen TYPE is a separate tensor input."""
    def __init__(self,state):
        validate_state(state);before=state_digest(state);super().__init__()
        with torch.inference_mode(False):
            for key,value in state.items():self.register_parameter(key,nn.Parameter(value.detach().clone(),requires_grad=False))
        self._learning=False;self.eval()
        self._identities=tuple((key,id(value),value.untyped_storage().data_ptr())for key,value in self.named_parameters())
        self._frozen_sha256=before
        require(not {value.untyped_storage().data_ptr()for value in self.parameters()}.intersection(
            value.untyped_storage().data_ptr()for value in state.values()),'supplied state storage cannot be reused')
        require(state_digest(state)==before,'constructor changed supplied state')

    def _state(self):return {key:value.detach()for key,value in self.named_parameters()}

    def validate(self):
        require(type(self) is SpanContextAssertionNullHead and not self._modules and not self._buffers
            and tuple(self._parameters)==STATE_KEYS and type(self._learning) is bool and self.training is self._learning
            and tuple((key,id(value),value.untyped_storage().data_ptr())for key,value in self.named_parameters())==self._identities
            and all(value.requires_grad is self._learning for value in self.parameters()),'new component topology/mode changed')
        current=state_digest(self._state())
        if not self._learning:require(current==self._frozen_sha256 and all(value.grad is None for value in self.parameters()),
            'frozen new state changed outside explicit learning API')
        return current

    def set_learning(self,enabled):
        require(type(enabled) is bool,'explicit component learning boolean required');self.validate()
        self.train(enabled).requires_grad_(enabled);self._learning=enabled
        for value in self.parameters():value.grad=None
        if not enabled:self._frozen_sha256=state_digest(self._state())
        self.validate();return self

    def _arithmetic(self,raw7,type_scores,pairs):
        start,end=pairs.unbind(-1)
        # Three shared projections precede interval pooling. This is the
        # declared FP32 computation law, not pool-first bit-equivalence.
        projected_start=F.linear(raw7,self.span_weight[:,:2048])
        projected_end=F.linear(raw7,self.span_weight[:,2048:4096])
        projected_mean=F.linear(raw7,self.span_weight[:,4096:])
        prefix=F.pad(projected_mean.cumsum(1),(0,0,1,0))
        mean=(prefix[:,end]-prefix[:,start])/(end-start).to(torch.float32)[None,:,None]
        span=F.gelu(projected_start[:,start]+projected_end[:,end-1]+mean+self.span_bias)
        context=F.gelu(F.linear(raw7[:,-1],self.context_weight,self.context_bias))[:,None,:].expand_as(span)
        interaction=torch.cat((span,context,span*context),dim=-1)
        logits=F.linear(F.gelu(F.linear(interaction,self.joint_weight,self.joint_bias)),
            self.decision_weight,self.decision_bias)
        return validate_bank(SpanContextEventNullBank(ENERGY_VERSION,F.log_softmax(type_scores,dim=-1).contiguous(),
            logits[:,:,0].t().contiguous(),logits[:,:,1].t().contiguous(),pairs,raw7.shape[1]))

    def forward(self,raw7,frozen_type_scores):
        before=self.validate()
        require(type(raw7) is torch.Tensor and raw7.ndim==3 and raw7.shape[0]==7
            and raw7.shape[1]>0 and raw7.shape[2]==2048,'whole original RAW[7,U,2048] required')
        _fp32(raw7,tuple(raw7.shape),detached=True)
        with torch.inference_mode(False):pairs=exact.complete_pairs(raw7.shape[1])
        _fp32(frozen_type_scores,(len(pairs),7),detached=True)
        require(not {raw7.untyped_storage().data_ptr(),frozen_type_scores.untyped_storage().data_ptr()}.intersection(
            value.untyped_storage().data_ptr()for value in self.parameters()),'supplied input/state alias rejected')
        input_bits=(tensor_digest(raw7),tensor_digest(frozen_type_scores))
        if self._learning:
            require(torch.is_grad_enabled() and not torch.is_inference_mode_enabled(),'ordinary autograd required for learning')
            result=self._arithmetic(raw7,frozen_type_scores,pairs)
        else:
            with torch.inference_mode(False),torch.no_grad():result=self._arithmetic(raw7,frozen_type_scores,pairs)
        require(input_bits==(tensor_digest(raw7),tensor_digest(frozen_type_scores)) and self.validate()==before,
            'forward changed supplied input or component state')
        return result
