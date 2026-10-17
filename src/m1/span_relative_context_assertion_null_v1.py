"""Portable inference definitions; no research datasets or trained weights included."""

from __future__ import annotations

import hashlib

import json

import math

import torch

from torch import nn

from torch.nn import functional as F

import m1.span_conditioned_context_assertion_null_v1 as baseline

VERSION = 'M1_ALL_SPAN_RELATIVE_DISTANCE31_CONTEXT_ASSERTION_NULL_COMPONENT_v1'

BIAS_KEY = 'relative_distance_bias'

BUCKETS = 31

LARGE_CUTS = (8,12,16,23,32,46,64,91)

STATE_SHAPES = {**baseline.STATE_SHAPES,BIAS_KEY:(BUCKETS,)}

STATE_KEYS = tuple(STATE_SHAPES)

PARAMETERS = baseline.PARAMETERS+BUCKETS

require,digest,tensor_digest,_fp32 = baseline.require,baseline.digest,baseline.tensor_digest,baseline._fp32

exact,TYPES,ENERGY_VERSION = baseline.exact,baseline.TYPES,baseline.ENERGY_VERSION

negative_log_likelihood,predict,event_gains = baseline.negative_log_likelihood,baseline.predict,baseline.event_gains

LAW = {**baseline.LAW,'version':VERSION,
    'attention':'UNCHANGED_FP32_QK_DIV_SQRT32_PLUS_LEARNED_RELATIVE_BUCKET_LOGIT_THEN_FULL_U_SOFTMAX',
    'relative_coordinate':'SIGNED_CANONICAL_UNIT_DISTANCE_FROM_EACH_ENUMERATED_HALFOPEN_SPAN_NEAREST_BOUNDARY; INSIDE_ZERO',
    'relative_buckets':{'inside':0,'left':list(range(1,16)),'right':list(range(16,31)),
        'T5_reference_buckets':32,'T5_max_distance':128,'exact_absolute_distances':list(range(8)),
        'integer_logarithmic_cuts':list(LARGE_CUTS),'unused_positive_zero_removed':True},
    'bias_sharing':'ONE_LEARNED31_VECTOR_SHARED_ACROSS_ALL7_CONTENT_BRANCHES_AND_ALL_SPANS',
    'bias_initialization':'EXACT_ZERO; NO_ADDITIONAL_RANDOM_DRAW; NO_INITIAL_NEAR_OR_FAR_PREFERENCE',
    'visibility':'ALL_ORIGINAL_UNITS_REMAIN; NO_PARSER_MASK_WINDOW_OR_SHORTLIST',
    'zero_bias_native12_full_score_parity_intended':True,
    'learned_locality_benefit_claimed':False,'parameter_tensors':13,'parameters':PARAMETERS}

def validate_state(state):
    require(type(state) is dict and tuple(state) == STATE_KEYS,'complete ordered relative13 state required')
    for key,shape in STATE_SHAPES.items(): _fp32(state[key],shape,detached=True)
    require(len({v.untyped_storage().data_ptr() for v in state.values()}) == 13
        and sum(v.numel() for v in state.values()) == PARAMETERS,'complete disjoint332033 state required')
    return state

def state_digest(state):
    validate_state(state); result = hashlib.sha256()
    for key,value in state.items():
        metadata = json.dumps([key,'FLOAT32',list(value.shape)],separators=(',',':')).encode(); payload = value.numpy().tobytes()
        result.update(len(metadata).to_bytes(8,'little')); result.update(metadata)
        result.update(len(payload).to_bytes(8,'little')); result.update(payload)
    return result.hexdigest()

def relative_buckets(pairs, unit_count):
    require(type(unit_count) is int and unit_count>0 and type(pairs) is torch.Tensor
        and pairs.device.type == 'cpu' and pairs.dtype == torch.int64
        and torch.equal(pairs,exact.complete_pairs(unit_count)), 'complete original all-span unit geometry required')
    start,end = pairs[:,0,None],pairs[:,1,None]; key = torch.arange(unit_count,dtype=torch.int64,device='cpu')[None,:]
    distance = torch.where(key<start,key-start,torch.where(key>=end,key-end+1,0))
    magnitude = distance.abs()
    large = 7+torch.bucketize(magnitude,torch.tensor(LARGE_CUTS,dtype=torch.int64,device='cpu'),right=True)
    magnitude_bucket = torch.where(magnitude<8,magnitude,large)
    return torch.where(distance>0,magnitude_bucket+15,magnitude_bucket).contiguous()

class RelativeSpanContextAssertionNullHead(baseline.SpanConditionedContextAssertionNullHead):
    """Reuse native forward/scoring ABI; override only topology and attention."""
    def __init__(self,state):
        validate_state(state); before = state_digest(state); nn.Module.__init__(self)
        with torch.inference_mode(False):
            for key,value in state.items(): self.register_parameter(key,nn.Parameter(value.detach().clone(),requires_grad=False))
        self._learning = False; self.eval(); self._frozen_sha256 = before
        self._identities = tuple((key,id(v),v.untyped_storage().data_ptr()) for key,v in self.named_parameters())
        require(not {v.untyped_storage().data_ptr() for v in self.parameters()}.intersection(
            v.untyped_storage().data_ptr() for v in state.values()) and state_digest(state) == before,
            'supplied complete13 storage/input must remain private')

    def validate(self):
        require(type(self) is RelativeSpanContextAssertionNullHead and not self._modules and not self._buffers
            and tuple(self._parameters) == STATE_KEYS and type(self._learning) is bool and self.training is self._learning
            and tuple((key,id(v),v.untyped_storage().data_ptr()) for key,v in self.named_parameters()) == self._identities
            and all(v.requires_grad is self._learning for v in self.parameters()),'relative13 topology or learning mode changed')
        current = state_digest(self._state())
        if not self._learning:
            require(current == self._frozen_sha256 and all(v.grad is None for v in self.parameters()),'frozen relative13 state changed')
        return current

    def set_learning(self,enabled):
        require(type(enabled) is bool,'explicit relative13 learning boolean required'); self.validate()
        self.train(enabled).requires_grad_(enabled); self._learning = enabled
        for value in self.parameters(): value.grad = None
        if not enabled: self._frozen_sha256 = state_digest(self._state())
        self.validate(); return self

    def _representations(self,raw7,pairs):
        # Same FP32 operation order as native12, through QK/sqrt32.
        start,end = pairs.unbind(-1)
        first = F.linear(raw7,self.span_weight[:,:2048]); last = F.linear(raw7,self.span_weight[:,2048:4096])
        projected_mean = F.linear(raw7,self.span_weight[:,4096:]); prefix = F.pad(projected_mean.cumsum(1),(0,0,1,0))
        mean = (prefix[:,end]-prefix[:,start])/(end-start).to(torch.float32)[None,:,None]
        span = F.gelu(first[:,start]+last[:,end-1]+mean+self.span_bias)
        values = F.gelu(F.linear(raw7,self.context_weight,self.context_bias))
        keys = F.linear(raw7,self.key_weight,self.key_bias); queries = F.linear(span,self.query_weight,self.query_bias)
        logits = torch.matmul(queries,keys.transpose(1,2))/math.sqrt(baseline.WIDTH)
        attention = torch.softmax(logits+self.relative_distance_bias[relative_buckets(pairs,raw7.shape[1])][None,:,:],dim=-1)
        require(bool(attention.isfinite().all()),'complete relative span attention is nonfinite')
        return span,torch.matmul(attention,values),attention

def export_state(owner):
    require(type(owner) is RelativeSpanContextAssertionNullHead,'exact complete relative13 owner required')
    before = owner.validate()
    with torch.inference_mode(False): state = {k:v.detach().clone().contiguous() for k,v in owner.named_parameters()}
    require(state_digest(state) == before,'complete relative13 export differs')
    return state,{'version':VERSION,'law_sha256':digest(LAW),'state_sha256':before,
        'parameter_tensors':13,'parameters':PARAMETERS,'checkpoint_or_runtime_authority':False}
