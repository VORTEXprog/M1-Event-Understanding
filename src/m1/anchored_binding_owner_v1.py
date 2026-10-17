"""Portable inference definitions; no research datasets or trained weights included."""

from __future__ import annotations
from .source_basis import runtime_source_paths

import hashlib

import json

import math

from pathlib import Path

import torch

from torch import Tensor

from torch import nn

from m1.tensor_state import state_dict_sha256

import m1.structured_joint_role_argument_v1 as native

import m1.event_anchor_argument_compatibility_v1 as compatibility

import m1.binding_complete_path_margin_v1 as margin

VERSION='M1_ANCHORED_BINDING_PRIVATE70_v1_UNADMITTED'

SOURCE=Path('src/m1/anchored_binding_owner_v1.py')

TRAINING_VERSION='M1_ANCHORED70_FULL_POTENTIAL_NATIVE_NLL_AND_FIXED_BEAM_MARGIN1_v1'

OBJECTIVE='NATIVE_FP32_FULL_G_MARGINAL_NLL_PLUS_FIXED_BEAM_F64_ROLE_HAMMING_MARGIN_COEFFICIENT1'

DECODER=native.DECODER

EXTRA_KEYS=tuple('anchor_compatibility.'+name for name in compatibility.STATE_KEYS)

FROZEN_KEYS=native.FROZEN_KEYS

TRAINABLE_KEYS=(*native.TRAINABLE_KEYS,*EXTRA_KEYS)

FORWARD_SOURCES = runtime_source_paths()

ROOT=Path(__file__).resolve().parent

require=compatibility._require

def canonical(value):
    return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()

def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()

def shape_map(encoder,hidden):
    require(type(encoder) is int and encoder>0 and type(hidden) is int and hidden>0 and hidden%4==0,
        'explicit positive native dimensions and hidden multiple4 required')
    return {**native._shape_map(encoder,hidden),**{'anchor_compatibility.'+name:shape
        for name,shape in compatibility.SHAPES.items()}}

def validate_state(state,encoder,hidden):
    shapes=shape_map(encoder,hidden)
    require(type(state) is dict and set(state)==set(shapes) and all(isinstance(value,Tensor)
        and tuple(value.shape)==shapes[name] and value.device.type=='cpu' and value.dtype==torch.float32
        and value.is_contiguous() and not value.is_inference() and not value.requires_grad
        and bool(value.isfinite().all()) for name,value in state.items()),
        'exact70 ordinary detached finite contiguous CPU FP32 tensors required')
    require(len({value.untyped_storage().data_ptr() for value in state.values()})==70,
        'all70 state storages must be distinct; aliases cannot be normalized')
    if (encoder,hidden)==(2048,128):
        require(sum(math.prod(s) for s in shapes.values())==2436354,'production70 parameter geometry differs')

def dependency_hashes(net):
    return {'frozen_owner_state_sha256':net.base.frozen_owner_state_sha256,
        'native_owner_state_sha256':net.base.native_owner_state_sha256,
        'native_generic_state_sha256':net.base.native_generic_state_sha256}

def basis_metadata(mode,encoder,hidden,dependencies,*,source_pins=None):
    require(type(mode) is str and mode in compatibility.MODES and type(dependencies) is dict
        and set(dependencies)=={'frozen_owner_state_sha256','native_owner_state_sha256','native_generic_state_sha256'}
        and all(native._sha(value) for value in dependencies.values()),'explicit geometry mode and three native dependency hashes required')
    shapes=shape_map(encoder,hidden)
    if source_pins is None:
        source_pins={str(name):hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in FORWARD_SOURCES}
    require(type(source_pins) is dict and set(source_pins)==set(map(str,FORWARD_SOURCES))
        and all(native._sha(value) for value in source_pins.values()),'complete70 forward/loss basis SOURCE pins required')
    return {'version':VERSION,'geometry_mode':mode,'input':'EXPLICIT_PREDICTED_HALFOPEN_UNIT_ANCHORS_PLUS_UNCHANGED_SEMANTIC4',
        'native65':native.VERSION,'compatibility':compatibility.VERSION,'objective':OBJECTIVE,'decoder':DECODER,
        'training_computation':TRAINING_VERSION,'native_NLL':native.OBJECTIVE,'margin':margin.OBJECTIVE,
        'encoder_dimension':encoder,'hidden_dimension':hidden,'score_dimension':32,
        'role_order':list(native.ROLE_ORDER),'beam_width':8,'parameter_shapes':{k:list(v) for k,v in shapes.items()},
        'source_pins':dict(source_pins),'dependencies':dict(dependencies),
        'backend':{'torch_version':str(torch.__version__),'CPU_capability':torch.backends.cpu.get_cpu_capability(),
            'dtype':'float32','device':'cpu'},'runtime_admission_or_qualification_granted':False}

class AnchoredBindingArgumentExpert(nn.Module):
    architecture=VERSION
    _base_components=native.StructuredJointRoleArgumentExpert._base_components

    def __init__(self,base_state,*,mode,expected_frozen_state_sha256,expected_native_state_sha256,
            expected_native_generic_sha256,encoder_dimension=2048,hidden_dimension=128):
        super().__init__();shape_map(encoder_dimension,hidden_dimension)
        require(type(mode) is str and mode in compatibility.MODES,'explicit matched geometry mode required')
        self.encoder_dimension,self.hidden_dimension=encoder_dimension,hidden_dimension
        original=native.StructuredJointRoleArgumentExpert(base_state,
            expected_frozen_state_sha256=expected_frozen_state_sha256,
            expected_native_state_sha256=expected_native_state_sha256,
            expected_native_generic_sha256=expected_native_generic_sha256,
            encoder_dimension=encoder_dimension,hidden_dimension=hidden_dimension)
        self.base,self.span_scorer=original.base,original.span_scorer
        self.null_embedding=original.null_embedding
        self.factor_left,self.factor_right=original.factor_left,original.factor_right
        self.role_pair_gate=original.role_pair_gate
        self.anchor_compatibility=compatibility.EventAnchorArgumentCompatibility(mode)
        self._basis=basis_metadata(mode,encoder_dimension,hidden_dimension,dependency_hashes(self))
        self._basis_sha256=digest(self._basis)
        self.eval().requires_grad_(False);self._validate_topology();assert_frozen_unchanged(self)

    @property
    def mode(self):
        return self._basis['geometry_mode']

    def _validate_topology(self):
        validate_state(dict(self.state_dict()),self.encoder_dimension,self.hidden_dimension)
        require(type(self.anchor_compatibility) is compatibility.EventAnchorArgumentCompatibility
            and self.anchor_compatibility.mode==self.mode and digest(self._basis)==self._basis_sha256
            and self._basis['dependencies']==dependency_hashes(self)
            and set(dict(self.named_parameters()))==set(shape_map(self.encoder_dimension,self.hidden_dimension)),
            'exact70 topology and immutable declared mode/basis required')
        self.anchor_compatibility._validate_owned()

    def forward(self,inputs):
        require(type(inputs) is compatibility.AnchoredArgumentInputs,'explicit anchored input DTO required; old65 input denied')
        self._validate_topology();inputs.validate()
        require(inputs.semantic.token_states.shape[-1]==self.encoder_dimension,'declared encoder basis differs')
        old=native.StructuredJointRoleArgumentExpert.forward(self,inputs.semantic)
        return self.anchor_compatibility(inputs,old)

    def enable_training(self):
        self.base.enable_training();self.train()
        for name,parameter in self.named_parameters():
            parameter.grad=None;parameter.requires_grad_(name in TRAINABLE_KEYS)
        assert_frozen_unchanged(self);return self

def assert_frozen_unchanged(net):
    require(type(net) is AnchoredBindingArgumentExpert,'exact private70 owner required')
    return native.am106.assert_frozen_unchanged(net.base)

def model_from_state(state,*,mode,expected_state_sha256,expected_basis_sha256,
        expected_frozen_state_sha256,expected_native_state_sha256,expected_native_generic_sha256,
        encoder_dimension=2048,hidden_dimension=128):
    validate_state(state,encoder_dimension,hidden_dimension)
    require(native._sha(expected_state_sha256) and native._sha(expected_basis_sha256)
        and state_dict_sha256(state)==expected_state_sha256,'external full70 state identity required before construction')
    frozen={name[5:]:value for name,value in state.items() if name in FROZEN_KEYS}
    with torch.random.fork_rng(devices=[]):
        net=AnchoredBindingArgumentExpert(frozen,mode=mode,
            expected_frozen_state_sha256=expected_frozen_state_sha256,
            expected_native_state_sha256=expected_native_state_sha256,
            expected_native_generic_sha256=expected_native_generic_sha256,
            encoder_dimension=encoder_dimension,hidden_dimension=hidden_dimension)
    require(net._basis_sha256==expected_basis_sha256,'requested mode/current full70 forward basis differs')
    net.load_state_dict(state,strict=True);net.eval().requires_grad_(False)
    net._validate_topology();assert_frozen_unchanged(net)
    require(state_dict_sha256(dict(net.state_dict()))==expected_state_sha256
        and not {v.untyped_storage().data_ptr() for v in state.values()}.intersection(
            v.untyped_storage().data_ptr() for v in net.state_dict().values()),'fresh disjoint70 restore bits differ')
    return net
