"""Portable inference definitions; no research datasets or trained weights included."""

from __future__ import annotations

from pathlib import Path

import torch

from torch.nn import functional as F

from torch.nn.utils.rnn import pack_padded_sequence

from torch.nn.utils.rnn import pad_packed_sequence

from torch.nn.utils.rnn import pad_sequence

from m1.tensor_state import state_dict_sha256

import m1.frozen_native_fresh_query_argument_v1 as prior

from m1.argument_expert import ArgumentExpertLogits

from m1.heads import _validate_context

from m1.heads import _event_inputs

from m1.heads import _validate_module_input

from m1.normalized_presence import NormalizedContextAttention

import m1.joint_role_context_coordination_v1 as coordination

from m1.sequence_pointer_training import state_digest

VERSION='M1_FROZEN_POINTER_JOINT_ROLE_CONTEXT_ARGUMENT_v1'

SOURCE=Path('src/m1/frozen_pointer_joint_role_context_argument_v1.py')

FORMULA='SHARED_C_JOINT_EQUALS_C_PLUS_ZERO_R_OF_FOUR_HEAD_EIGHT_ROLE_MHA_NONAFFINE_LN_Z_PLUS_C_THEN_FROZEN_AM90_POINTER31_PLUS_ZERO_QUERY_CORRECTIONS_OF_SHARED_ORDERED_ROLE_EVENT_POOL_SCORING_NATIVE_PLUS_ZERO_TOKEN_KEY_CORRECTIONS_AND_NATIVE_GELU_Z_PLUS_ZERO_OUTER_R_OF_ORDERED_CONTEXT_PLUS_GELU_ZERO_INNER_LINEAR_CONCAT_Z_ORDERED_CONTEXT'

COORDINATION_ATTENTION_HEADS=4

COORDINATION_NEW_KEYS=('joint_role_coordination.query_key_value.weight','joint_role_coordination.joint_role_residual.weight')

NATIVE_KEYS=prior.NATIVE_KEYS

FROZEN_KEYS=(*NATIVE_KEYS,'presence_query.weight','presence_key.weight',*prior.POINTER_NEW_KEYS)

ORDERED_PRESENCE_KEYS=('presence_projection.weight','presence_projection.bias',
    'presence_gru.weight_ih_l0','presence_gru.weight_hh_l0','presence_gru.bias_ih_l0','presence_gru.bias_hh_l0',
    'presence_gru.weight_ih_l0_reverse','presence_gru.weight_hh_l0_reverse',
    'presence_gru.bias_ih_l0_reverse','presence_gru.bias_hh_l0_reverse',
    'ordered_presence_query.weight','ordered_presence_key.weight','ordered_presence_value.weight','ordered_presence_residual.weight')

ORDERED_PRESENCE_MAP_KEYS=ORDERED_PRESENCE_KEYS[10:]

INNER_KEYS=('ordered_presence_inner.weight','ordered_presence_inner.bias')

ORIGINAL100_NEW_KEYS=(*ORDERED_PRESENCE_KEYS,'pointer_start_key_residual.weight','pointer_end_key_residual.weight')

ORDERED_PRESENCE_KEYS=(*ORDERED_PRESENCE_KEYS,*INNER_KEYS)

POINTER_KEY_NEW_KEYS=('pointer_start_key_residual.weight','pointer_end_key_residual.weight')

POINTER_QUERY_NEW_KEYS=('pointer_start_query_residual.weight','pointer_end_query_residual.weight')

POINTER_NEW_KEYS=(*POINTER_KEY_NEW_KEYS,*POINTER_QUERY_NEW_KEYS)

NEW_KEYS=(*ORIGINAL100_NEW_KEYS,*INNER_KEYS,*POINTER_QUERY_NEW_KEYS,*COORDINATION_NEW_KEYS)

_digest=prior._digest

def _shapes(encoder,hidden):
    coordination._dimensions(hidden,COORDINATION_ATTENTION_HEADS)
    shapes={name:shape for name,shape in prior._shapes(encoder,hidden).items() if name in FROZEN_KEYS}
    half=hidden//2
    shapes.update({'presence_projection.weight':(hidden,encoder),'presence_projection.bias':(hidden,)})
    for suffix in ('','_reverse'):
        shapes.update({'presence_gru.weight_ih_l0'+suffix:(3*half,hidden),
            'presence_gru.weight_hh_l0'+suffix:(3*half,half),
            'presence_gru.bias_ih_l0'+suffix:(3*half,), 'presence_gru.bias_hh_l0'+suffix:(3*half,)})
    shapes.update({name:(hidden,hidden) for name in ORDERED_PRESENCE_MAP_KEYS})
    shapes.update({name:(hidden,hidden) for name in POINTER_NEW_KEYS})
    shapes.update({'ordered_presence_inner.weight':(hidden,2*hidden),'ordered_presence_inner.bias':(hidden,)})
    shapes.update({COORDINATION_NEW_KEYS[0]:(3*hidden,hidden),COORDINATION_NEW_KEYS[1]:(hidden,hidden)})
    return shapes

def _validate_values(state,shapes,label):
    if (type(state) is not dict or set(state)!=set(shapes)
            or any(not isinstance(value,torch.Tensor) or tuple(value.shape)!=shapes[name]
                or value.device.type!='cpu' or value.dtype!=torch.float32 or value.requires_grad
                or not bool(torch.isfinite(value).all()) for name,value in state.items())):
        raise ValueError(label+' exact detached finite CPU FP32 inventory required')

def _validate_frozen(state,encoder,hidden):
    _validate_values(state,{name:shape for name,shape in _shapes(encoder,hidden).items() if name in FROZEN_KEYS},'frozen31')

def _validate_state(state,encoder,hidden):
    _validate_values(state,_shapes(encoder,hidden),'new53')

def _native(state):
    return {name:value for name,value in state.items() if name in NATIVE_KEYS}

def _verify_frozen(state,encoder,hidden,frozen_sha,native_sha,generic_sha):
    _validate_frozen(state,encoder,hidden)
    for sha in (frozen_sha,native_sha,generic_sha):_digest(sha)
    if state_digest(state)!=frozen_sha:raise ValueError('external frozen31 RAW digest differs')
    if state_digest(_native(state))!=native_sha:raise ValueError('external native27 RAW digest differs')
    if state_dict_sha256(_native(state))!=generic_sha:raise ValueError('external native27 GENERIC digest differs')

def frozen_state_digest(net):
    return state_digest({name:value for name,value in net.state_dict().items() if name in FROZEN_KEYS})

def native_state_digest(net):
    return state_digest(_native(net.state_dict()))

def assert_frozen_unchanged(net):
    if (type(net) is not FrozenPointerJointRoleContextArgumentExpert
            or frozen_state_digest(net)!=net.frozen_owner_state_sha256
            or native_state_digest(net)!=net.native_owner_state_sha256
            or state_dict_sha256(_native(net.state_dict()))!=net.native_generic_state_sha256
            or any(parameter.requires_grad or parameter.grad is not None
                for name,parameter in net.named_parameters() if name in FROZEN_KEYS)):
        raise ValueError('frozen31/native27 digest or gradient ownership changed')
    return net.frozen_owner_state_sha256

class FrozenPointerJointRoleContextArgumentExpert(prior.base.PresenceSequenceKeyArgumentExpert):
    architecture=VERSION
    input_formula=FORMULA

    def __init__(self,base_state,*,expected_frozen_state_sha256,expected_native_state_sha256,
                 expected_native_generic_sha256,encoder_dimension=2048,hidden_dimension=128):
        _verify_frozen(base_state,encoder_dimension,hidden_dimension,expected_frozen_state_sha256,
            expected_native_state_sha256,expected_native_generic_sha256)
        # Exact original native27/presence3/fresh2 draws before removing old R.
        super().__init__(encoder_dimension,hidden_dimension)
        factory={'device':'cpu','dtype':torch.float32}
        self.fresh_start_query=torch.nn.Linear(2*hidden_dimension,hidden_dimension,bias=False,**factory)
        self.fresh_end_query=torch.nn.Linear(2*hidden_dimension,hidden_dimension,bias=False,**factory)
        del self.presence_residual
        self.presence_projection=torch.nn.Linear(encoder_dimension,hidden_dimension,bias=True,**factory)
        self.presence_gru=torch.nn.GRU(hidden_dimension,hidden_dimension//2,num_layers=1,
            batch_first=True,bidirectional=True,bias=True,dropout=0,**factory)
        self.ordered_presence_query=torch.nn.Linear(hidden_dimension,hidden_dimension,bias=False,**factory)
        self.ordered_presence_key=torch.nn.Linear(hidden_dimension,hidden_dimension,bias=False,**factory)
        self.ordered_presence_value=torch.nn.Linear(hidden_dimension,hidden_dimension,bias=False,**factory)
        self.ordered_presence_residual=torch.nn.Linear(hidden_dimension,hidden_dimension,bias=False,**factory)
        torch.nn.init.zeros_(self.ordered_presence_residual.weight)
        # Preserve every Am95 draw before the two new standard Linear draws.
        self.pointer_start_key_residual=torch.nn.Linear(hidden_dimension,hidden_dimension,bias=False,**factory)
        self.pointer_end_key_residual=torch.nn.Linear(hidden_dimension,hidden_dimension,bias=False,**factory)
        torch.nn.init.zeros_(self.pointer_start_key_residual.weight)
        torch.nn.init.zeros_(self.pointer_end_key_residual.weight)
        # The only new draw occurs after every original Am100 draw, before copying frozen31.
        self.ordered_presence_inner=torch.nn.Linear(2*hidden_dimension,hidden_dimension,bias=True,**factory)
        # Preserve the exact Am101 draw consumption, then start at the Am100 subcase.
        torch.nn.init.zeros_(self.ordered_presence_inner.weight)
        torch.nn.init.zeros_(self.ordered_presence_inner.bias)
        # Append exactly two ordinary draws AFTER every original Am103 draw.
        self.pointer_start_query_residual=torch.nn.Linear(hidden_dimension,hidden_dimension,bias=False,**factory)
        self.pointer_end_query_residual=torch.nn.Linear(hidden_dimension,hidden_dimension,bias=False,**factory)
        torch.nn.init.zeros_(self.pointer_start_query_residual.weight)
        torch.nn.init.zeros_(self.pointer_end_query_residual.weight)
        # All original51 draws precede these two; frozen31 is copied only below.
        self.joint_role_coordination=coordination.JointRoleContextCoordination(
            hidden_dimension,COORDINATION_ATTENTION_HEADS)
        self.frozen_owner_state_sha256=expected_frozen_state_sha256
        self.native_owner_state_sha256=expected_native_state_sha256
        self.native_generic_state_sha256=expected_native_generic_sha256
        own=self.state_dict()
        with torch.no_grad():
            for name,value in base_state.items():own[name].copy_(value)
        self.eval().requires_grad_(False)
        self._validate_topology()
        assert_frozen_unchanged(self)
        if any(own[name].untyped_storage().data_ptr()==value.untyped_storage().data_ptr()
               for name,value in base_state.items()):raise ValueError('frozen31 storage must be independently owned')

    def _validate_topology(self):
        _validate_state(dict(self.state_dict()),self.encoder_dimension,self.hidden_dimension)
        if (type(self.joint_role_coordination) is not coordination.JointRoleContextCoordination
                or self.joint_role_coordination.hidden_dimension!=self.hidden_dimension
                or self.joint_role_coordination.attention_heads!=COORDINATION_ATTENTION_HEADS):
            raise ValueError('exact original joint-role primitive configuration required')
        self.joint_role_coordination._validate_owned()
        parameters=dict(self.named_parameters())
        if (set(parameters)!=set(_shapes(self.encoder_dimension,self.hidden_dimension))
                or len({p.untyped_storage().data_ptr() for p in parameters.values()})!=len(parameters)):
            raise ValueError('exact53 independently owned parameter storages required')

    def encode_presence(self,inputs):
        _validate_module_input(self,inputs)
        lengths=inputs.token_mask.sum(1)
        units=F.layer_norm(inputs.token_states,(self.encoder_dimension,),eps=1e-5)
        projected=F.gelu(self.presence_projection(units))
        compact=pad_sequence(projected[inputs.token_mask].split(lengths.tolist()),batch_first=True)
        packed=pack_padded_sequence(compact,lengths.cpu(),batch_first=True,enforce_sorted=False)
        # Omitted h0 is freshly zero on every call; there is no cross-turn state.
        encoded,_=self.presence_gru(packed)
        states,_=pad_packed_sequence(encoded,batch_first=True,total_length=compact.shape[1])
        compact_mask=torch.arange(compact.shape[1],device=lengths.device)[None]<lengths[:,None]
        result=states.new_zeros((*inputs.token_mask.shape,self.hidden_dimension))
        result[inputs.token_mask]=states[compact_mask]
        return result

    def ordered_presence_context(self,inputs,states,preactivation):
        query=F.normalize(self.ordered_presence_query(F.layer_norm(preactivation,
            (self.hidden_dimension,),eps=1e-5)),dim=-1,eps=1e-8)
        key=F.normalize(self.ordered_presence_key(F.layer_norm(states,
            (self.hidden_dimension,),eps=1e-5)),dim=-1,eps=1e-8)
        scores=torch.einsum('berh,bth->bert',query,key)*self.hidden_dimension**.5
        scores=scores.masked_fill(~inputs.token_mask[:,None,None,:],-torch.inf)
        attention=scores.softmax(-1).masked_fill(~inputs.event_mask[:,:,None,None],0)
        attention=attention.masked_fill(~inputs.token_mask[:,None,None,:],0)
        values=self.ordered_presence_value(states).masked_fill(~inputs.token_mask[:,:,None],0)
        pooled=torch.einsum('bert,bth->berh',attention,values)
        return pooled,attention

    def forward(self,inputs,*,ablate_presence_residual=False,ablate_pointer_residual=False,
                ablate_all_residuals=False,use_native_pointer=False,ablate_pointer_query_residual=False,
                ablate_joint_role_coordination=False):
        if any(type(value) is not bool for value in (ablate_presence_residual,ablate_pointer_residual,
                ablate_all_residuals,use_native_pointer,ablate_pointer_query_residual,ablate_joint_role_coordination)):
            raise ValueError('presence/key/query interventions require exact boolean flags')
        if any(p.device.type!='cpu' or p.dtype!=torch.float32 for p in self.parameters()):
            raise ValueError('ordered presence requires CPU FP32 parameters')
        keys=self.encode_keys(inputs)
        context,attention=NormalizedContextAttention.forward(self.context,inputs)
        _validate_context(inputs,context,self.hidden_dimension)
        pointer=self.pointer
        event=pointer.event_query(_event_inputs(inputs,context))
        query=torch.tanh(F.layer_norm(event.unsqueeze(2)+pointer.role_queries[None,None],
            (self.hidden_dimension,),eps=1e-5))
        z=self.presence_event(torch.cat((inputs.event_states,context),-1)).unsqueeze(2)+self.presence_roles[None,None]
        if use_native_pointer:start_query,end_query=pointer.start_query(query),pointer.end_query(query)
        else:
            pooled,_=self.presence_context(inputs,keys,z) # Frozen original Q/K and native-key values.
            combined=torch.cat((query,pooled),-1)
            start_query,end_query=self.fresh_start_query(combined),self.fresh_end_query(combined)
        presence_enabled=not (ablate_presence_residual or ablate_all_residuals)
        pointer_enabled=not (use_native_pointer or ablate_pointer_residual or ablate_all_residuals)
        query_enabled=pointer_enabled and not ablate_pointer_query_residual
        # Exactly one encoder and one ordered pool, shared when either loss needs them.
        states=self.encode_presence(inputs) if presence_enabled or pointer_enabled else None
        ordered=self.ordered_presence_context(inputs,states,z)[0] if presence_enabled or query_enabled else None
        if ordered is not None and not ablate_joint_role_coordination:
            ordered=self.joint_role_coordination(z,ordered,inputs.event_mask)
        if query_enabled:
            start_query=start_query+self.pointer_start_query_residual(ordered)
            end_query=end_query+self.pointer_end_query_residual(ordered)
        start=torch.einsum('berh,bth->bert',start_query,pointer.start_key(keys))
        end=torch.einsum('berh,bth->bert',end_query,pointer.end_key(keys))
        if pointer_enabled:
            start=start+torch.einsum('berh,bth->bert',start_query,self.pointer_start_key_residual(states))
            end=end+torch.einsum('berh,bth->bert',end_query,self.pointer_end_key_residual(states))
        valid=inputs.event_mask[:,:,None,None]&inputs.token_mask[:,None,None,:]
        scale,floor=self.hidden_dimension**.5,torch.finfo(start.dtype).min
        start,end=(value.div(scale).masked_fill(~valid,floor) for value in (start,end))
        features=F.gelu(z)
        if presence_enabled:
            fused=ordered+F.gelu(self.ordered_presence_inner(torch.cat((z,ordered),-1)))
            correction=self.ordered_presence_residual(fused).masked_fill(~inputs.event_mask[:,:,None,None],0)
            features=features+correction
        presence=self.presence_output(features).masked_fill(~inputs.event_mask[:,:,None,None],0)
        return ArgumentExpertLogits(presence,start,end,attention,inputs.event_mask)

    def enable_training(self):
        self.train().requires_grad_(False)
        for name,parameter in self.named_parameters():
            parameter.grad=None;parameter.requires_grad_(name in NEW_KEYS)
        assert_frozen_unchanged(self)
        return self
