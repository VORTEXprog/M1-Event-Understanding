"""Portable inference definitions; no research datasets or trained weights included."""

from __future__ import annotations
from .source_basis import runtime_source_paths

from copy import deepcopy

import hashlib

from pathlib import Path

import torch

from torch import nn

from torch.nn import functional as F

import m1.event_conditioned_pair_owner_v1 as old

import m1.full_context_head_candidate_pooling_v1 as pool

import m1.cross_event_role_context_v1 as cross

VERSION = 'M1_FULL_CONTEXT_PARALLEL_BINDING_PRIVATE83_v1_UNADMITTED'

SOURCE = Path('src/m1/full_context_parallel_binding_owner_v1.py')

MODES = ('BASELINE','FULL_CONTEXT','FULL_CONTEXT_PARALLEL')

TRAINING_VERSION = 'M1_FULL_CONTEXT83_UNCHANGED_NATIVE_NLL_MARGIN_EXTERNAL_PROTOCOL_v1'

OBJECTIVE,DECODER = old.OBJECTIVE,old.DECODER

POOL_KEYS = tuple('context_pool.'+k for k in pool.STATE_KEYS)

CROSS_KEYS = tuple('cross_event.'+k for k in cross.STATE_KEYS)

NEW_KEYS = (*POOL_KEYS,*CROSS_KEYS)

FORWARD_SOURCES = runtime_source_paths()

IDENTITY_VERSION = 'M1_SUPPLIED_NATIVE74_ORIGIN_FOR_PRIVATE83_v1'

IDENTITY_KEYS = {'version','original_state74_sha256','basis74_sha256','native74_identity'}

INITIALIZATION = 'POOL_RECTANGULAR_IDENTITIES_OUTPUT_ZERO_CROSS_QKV_IDENTITIES_OUTPUT_ZERO_LITERAL_NO_RNG'

require,canonical,digest,state_dict_sha256 = old.require,old.canonical,old.digest,old.state_dict_sha256

structured,native,saved_port = old.parent.native,old,old.saved_port

def _pool_shapes(hidden):
    return pool.shape_map(hidden_dimension=hidden,span_dimension=32,attention_dimension=32)

def initial_components(hidden):
    return {**{'context_pool.'+k:v for k,v in pool.initial_state(hidden_dimension=hidden,
        span_dimension=32,attention_dimension=32).items()},
        **{'cross_event.'+k:v for k,v in cross.initial_state(hidden).items()}}

def shape_map(encoder,hidden):
    return {**{'native74.'+k:v for k,v in old.shape_map(encoder,hidden).items()},
        **{'context_pool.'+k:v for k,v in _pool_shapes(hidden).items()},
        **{'cross_event.'+k:v for k,v in cross.shape_map(hidden).items()}}

def validate_identity(identity):
    require(type(identity) is dict and set(identity) == IDENTITY_KEYS and identity['version'] == IDENTITY_VERSION
        and all(structured._sha(identity[k]) for k in ('original_state74_sha256','basis74_sha256')),
        'closed independent supplied74 state/basis origin for new83 required')
    old.validate_identity(identity['native74_identity'])
    return identity

def validate_state(state,encoder,hidden):
    shapes = shape_map(encoder,hidden)
    require(type(state) is dict and set(state) == set(shapes)
        and all(type(v) is torch.Tensor and tuple(v.shape) == shapes[k] and v.device.type == 'cpu'
            and v.dtype == torch.float32 and v.is_contiguous() and not v.requires_grad
            and not v.is_inference() and bool(v.isfinite().all()) for k,v in state.items())
        and len({v.untyped_storage().data_ptr() for v in state.values()}) == 83,
        'complete83 independent detached finite ordinary CPU FP32 tensors required')
    old.validate_state({k:state['native74.'+k] for k in old.shape_map(encoder,hidden)},encoder,hidden)

def trainable_keys(architecture_mode):
    require(type(architecture_mode) is str and architecture_mode in MODES,'explicit new83 architecture mode required')
    base = tuple('native74.'+k for k in old.TRAINABLE_KEYS)
    return (*base,*(POOL_KEYS if architecture_mode != 'BASELINE' else ()),
            *(CROSS_KEYS if architecture_mode == 'FULL_CONTEXT_PARALLEL' else ()))

def basis_metadata(identity,*,architecture_mode,source_pins=None):
    validate_identity(identity); origin = identity['native74_identity']; active = trainable_keys(architecture_mode)
    if source_pins is None:
        source_pins = {str(p):hashlib.sha256((old.ROOT/p).read_bytes()).hexdigest() for p in FORWARD_SOURCES}
    require(type(source_pins) is dict and set(source_pins) == set(map(str,FORWARD_SOURCES))
        and all(structured._sha(v) for v in source_pins.values()),'exact complete83/native/component Source closure required')
    parent_basis = old.basis_metadata(origin,architecture_mode='EVENT_PAIR74',
        source_pins={str(p):source_pins[str(p)] for p in old.FORWARD_SOURCES})
    require(digest(parent_basis) == identity['basis74_sha256'],'external original74 Source/backend basis differs')
    h = origin['hidden_dimension']
    return {'version':VERSION,'architecture_mode':architecture_mode,'parent_origin':deepcopy(identity),
        'native74_basis':parent_basis,'pool':pool.basis_metadata(hidden_dimension=h,span_dimension=32,
            attention_dimension=32),'pool_query_chunk_size':128,'cross_event':cross.basis_metadata(h),
        'assembly':'NATIVE_BASE_ONCE_Q0_PLUS_OPTIONAL_FOREIGN_DELTA_V0_PLUS_OPTIONAL_ALL_TOKEN_POOL_THEN_NATIVE_UNARY_NULL_LR_ANCHOR5_PAIR4',
        'zero_inference':'BYPASS_ONLY_EXACT_ZERO_ADDITION_UNDER_NO_GRAD_PRESERVE_NATIVE_BITS',
        'initialization':INITIALIZATION,'initial9_state_sha256':state_dict_sha256(initial_components(h)),
        'parameter_shapes':{k:list(v) for k,v in shape_map(origin['encoder_dimension'],h).items()},
        'trainable_keys':list(active),'objective_core':OBJECTIVE,'decoder':DECODER,'training_computation':TRAINING_VERSION,
        'source_pins':dict(source_pins),'runtime_admission_or_qualification_granted':False}

def _add(base,delta):
    require(base.shape == delta.shape and bool(delta.isfinite().all()),'finite same-shape new residual required')
    return base if not torch.is_grad_enabled() and not bool(delta.count_nonzero()) else base+delta

def _native_with_context(parent70,inputs,context_pool,cross_event):
    """Literal native65 assembly, with exactly two optional additive seams."""
    s = inputs.semantic; old_logits,states,ordered,native_query = parent70._base_components(s)
    b = s.token_states.shape[0]; e = s.event_states.shape[1]; dim = structured.SCORE_DIM
    query = torch.tanh(F.layer_norm(native_query+ordered,(parent70.hidden_dimension,),eps=1e-5))
    if cross_event is not None: query = _add(query,cross_event(query,s.event_mask))
    pairs,pair_mask = structured.enumerate_unit_pairs(s.token_mask); p = pairs.shape[1]; scorer = parent70.span_scorer
    repeated = states[:,None].expand(-1,e,-1,-1)
    vectors = scorer.pair_encoder[2](torch.tanh(scorer._pair_preactivation(repeated,s.token_mask,pairs)))
    extra = context_pool(pool.FullContextPoolingInputs(states,s.token_mask,s.event_mask,
        inputs.event_unit_spans,vectors,pair_mask))
    require(torch.equal(extra.pair_indices,pairs) and torch.equal(extra.pair_mask,pair_mask)
        and torch.equal(extra.event_mask,s.event_mask) and torch.equal(extra.token_mask,s.token_mask),
        'full-context pool must preserve original complete pair/event/token geometry')
    vectors = _add(vectors,extra.projected_correction)
    q = scorer.query_projection(query)
    residual = torch.einsum('bepk,berk->berp',vectors,q)/(dim**.5)
    starts,ends = pairs[...,0].clamp_min(0),pairs[...,1].clamp_min(0)
    start = old_logits.argument_start_logits.gather(-1,starts[:,None,None,:].expand(b,e,8,p))
    end = old_logits.argument_end_logits.gather(-1,ends[:,None,None,:].expand(b,e,8,p))
    valid = s.event_mask[:,:,None,None] & pair_mask[:,None,None,:]
    start,end = (torch.where(valid,v,torch.zeros_like(v)) for v in (start,end))
    odds = old_logits.presence_logits[...,1]-old_logits.presence_logits[...,0]
    pair_unary = odds[...,None]+start+end+residual
    pair_unary = torch.where(valid,pair_unary,torch.zeros_like(pair_unary))
    null = scorer.null_scorer(query).masked_fill(~s.event_mask[:,:,None,None],0)
    unary = torch.cat((pair_unary,null),-1)
    pair_features = torch.tanh(vectors[:,:,None,:,:]+q[:,:,:,None,:])
    null_features = torch.tanh(parent70.null_embedding[None,None,None,None,:]+q[:,:,:,None,:])
    features = torch.cat((pair_features,null_features),3)
    full = torch.cat((pair_mask,torch.ones((b,1),dtype=torch.bool,device=pair_mask.device)),1)
    features = features.masked_fill(~(s.event_mask[:,:,None,None] & full[:,None,None,:])[...,None],0)
    output = structured.StructuredArgumentPotentials(unary,parent70.factor_left(features)/(dim**.5),
        parent70.factor_right(features),parent70.role_pair_gate,pairs,pair_mask,s.event_mask.clone(),
        s.token_mask.clone(),old_logits.context_attention,vectors,q)
    output.validate(); return output

class FullContextParallelBindingExpert(nn.Module):
    architecture = VERSION

    def __init__(self,state,*,identity,architecture_mode,expected_state_sha256,expected_basis_sha256):
        super().__init__(); validate_identity(identity); origin = identity['native74_identity']
        self.encoder_dimension,self.hidden_dimension = origin['encoder_dimension'],origin['hidden_dimension']
        validate_state(state,self.encoder_dimension,self.hidden_dimension)
        require(structured._sha(expected_state_sha256) and state_dict_sha256(state) == expected_state_sha256,
            'external complete83 state bits required before construction')
        self._basis = basis_metadata(identity,architecture_mode=architecture_mode); self._basis_sha256 = digest(self._basis)
        require(self._basis_sha256 == expected_basis_sha256,'external complete83 basis differs before construction')
        self._identity = deepcopy(identity)
        values = {k:state['native74.'+k] for k in old.shape_map(self.encoder_dimension,self.hidden_dimension)}
        self.native74 = old.model_from_state(values,identity=origin,architecture_mode='EVENT_PAIR74',
            expected_state_sha256=state_dict_sha256(values),expected_basis_sha256=identity['basis74_sha256'])
        self.context_pool = pool.model_from_state({k:state['context_pool.'+k] for k in pool.STATE_KEYS},
            hidden_dimension=self.hidden_dimension,span_dimension=32,attention_dimension=32,query_chunk_size=128)
        self.cross_event = cross.model_from_state({k:state['cross_event.'+k] for k in cross.STATE_KEYS},hidden_dimension=self.hidden_dimension)
        self.eval().requires_grad_(False); self._validate_topology(); assert_frozen_unchanged(self)
        require(state_dict_sha256(dict(self.state_dict())) == expected_state_sha256
            and not {v.untyped_storage().data_ptr() for v in state.values()}.intersection(
                v.untyped_storage().data_ptr() for v in self.state_dict().values()),'complete83 restore must preserve bits with disjoint storage')

    @property
    def architecture_mode(self): return self._basis['architecture_mode']
    @property
    def base74(self): return self.native74
    @property
    def parent(self): return self.native74.parent
    @property
    def mode(self): return self.native74.mode

    def _validate_topology(self):
        validate_state(dict(self.state_dict()),self.encoder_dimension,self.hidden_dimension)
        require(set(self._modules) == {'native74','context_pool','cross_event'} and not self._parameters and not self._buffers
            and type(self.native74) is old.EventConditionedPairArgumentExpert
            and type(self.context_pool) is pool.FullContextHeadCandidatePooling
            and type(self.cross_event) is cross.CrossEventRoleContext
            and self.context_pool.query_chunk_size == self._basis['pool_query_chunk_size'] == 128
            and self.native74.architecture_mode == 'EVENT_PAIR74' and self.native74._identity == self._identity['native74_identity']
            and self.native74._basis == self._basis['native74_basis'] and digest(self._basis) == self._basis_sha256
            and self._basis['parent_origin'] == self._identity
            and self._basis['trainable_keys'] == list(trainable_keys(self.architecture_mode)), 'exact83 topology/mode/origin/Source basis required')
        self.native74._validate_topology(); self.context_pool._validate_owned(); self.cross_event._validate_owned()
        inactive = set(NEW_KEYS)-set(trainable_keys(self.architecture_mode)); initial = initial_components(self.hidden_dimension)
        require(all(torch.equal(self.state_dict()[k].view(torch.uint8),initial[k].view(torch.uint8)) for k in inactive),
            'every inactive new tensor must retain its literal identity/zero initialization bits')

    def forward(self,inputs):
        self._validate_topology()
        require(type(inputs) is old.parent.compatibility.AnchoredArgumentInputs,'same separate predicted HEAD/native semantic DTO required')
        inputs.validate()
        if self.architecture_mode == 'BASELINE': return self.native74(inputs)
        parent70 = self.native74.parent
        output = _native_with_context(parent70,inputs,self.context_pool,
            self.cross_event if self.architecture_mode == 'FULL_CONTEXT_PARALLEL' else None)
        output = parent70.anchor_compatibility(inputs,output)
        return self.native74.pair_residual(inputs,output,factor_left_weight=parent70.factor_left.weight,
            factor_right_weight=parent70.factor_right.weight)

    def enable_training(self):
        self.native74.enable_training(); self.train()
        for k,v in self.named_parameters(): v.grad = None; v.requires_grad_(k in trainable_keys(self.architecture_mode))
        assert_frozen_unchanged(self); return self

def assert_frozen_unchanged(net):
    require(type(net) is FullContextParallelBindingExpert,'exact new83 owner required')
    net._validate_topology(); return old.assert_frozen_unchanged(net.native74)

def model_from_state(state,**kwargs): return FullContextParallelBindingExpert(state,**kwargs)
