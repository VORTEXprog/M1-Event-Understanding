"""Portable inference definitions; no research datasets or trained weights included."""

from __future__ import annotations
from .source_basis import runtime_source_paths

from copy import deepcopy

import hashlib

import math

from pathlib import Path

import torch

from torch import Tensor

from torch import nn

import m1.anchored_binding_owner_v1 as parent

import m1.event_conditioned_pair_residual_v1 as component

import m1.event_conditioned_pair_saved_port_v2 as saved_port

VERSION = 'M1_EVENT_CONDITIONED_PAIR_PRIVATE74_v1_UNADMITTED'

SOURCE = Path('src/m1/event_conditioned_pair_owner_v1.py')

TRAINING_VERSION = 'M1_PRIVATE74_UNCHANGED_NATIVE_FP32_NLL_FIXED_BEAM_F64_MARGIN1_v1'

OBJECTIVE, DECODER = parent.OBJECTIVE, parent.DECODER

RESIDUAL_KEYS = tuple('pair_residual.'+name for name in component.STATE_KEYS)

TRAINABLE_KEYS = (*('parent.'+name for name in parent.TRAINABLE_KEYS), *RESIDUAL_KEYS)

ARCHITECTURE_MODES = ('LEGACY70_CONTROL', 'EVENT_PAIR74')

FORWARD_SOURCES = runtime_source_paths()

IDENTITY_KEYS = {'version', 'mode', 'encoder_dimension', 'hidden_dimension', 'original_state70_sha256',
    'basis70_sha256', 'frozen_owner_state_sha256', 'native_owner_state_sha256', 'native_generic_state_sha256'}

IDENTITY_VERSION = 'M1_SUPPLIED_PRIVATE70_ORIGIN_FOR_PRIVATE74_v1'

ROOT, require, canonical, digest = parent.ROOT, parent.require, parent.canonical, parent.digest

state_dict_sha256 = parent.state_dict_sha256

def shape_map(encoder, hidden):
    return {**{'parent.'+name: shape for name, shape in parent.shape_map(encoder, hidden).items()},
        **{'pair_residual.'+name: shape for name, shape in component.SHAPES.items()}}

def validate_state(state, encoder, hidden):
    shapes = shape_map(encoder, hidden)
    require(type(state) is dict and set(state) == set(shapes)
        and all(type(value) is Tensor and tuple(value.shape) == shapes[name]
            and value.device.type == 'cpu' and value.dtype == torch.float32 and value.is_contiguous()
            and not value.is_inference() and not value.requires_grad and bool(value.isfinite().all())
            for name, value in state.items())
        and len({value.untyped_storage().data_ptr() for value in state.values()}) == 74,
        'complete disjoint ordinary detached finite CPU FP32 private74 state required')
    parent.validate_state({name: state['parent.'+name] for name in parent.shape_map(encoder, hidden)}, encoder, hidden)
    if (encoder, hidden) == (2048, 128):
        require(sum(math.prod(s) for s in shapes.values()) == 2440770, 'production74 geometry differs')

def validate_identity(identity):
    require(type(identity) is dict and set(identity) == IDENTITY_KEYS and identity['version'] == IDENTITY_VERSION
        and type(identity['mode']) is str and identity['mode'] in parent.compatibility.MODES
        and all(parent.native._sha(identity[k]) for k in ('original_state70_sha256', 'basis70_sha256',
            'frozen_owner_state_sha256', 'native_owner_state_sha256', 'native_generic_state_sha256')),
        'closed original70 identity/mode/basis/dependencies required')
    parent.shape_map(identity['encoder_dimension'], identity['hidden_dimension'])
    return identity

def trainable_keys(architecture_mode):
    require(type(architecture_mode) is str and architecture_mode in ARCHITECTURE_MODES, 'explicit74 architecture mode required')
    return TRAINABLE_KEYS if architecture_mode == 'EVENT_PAIR74' else TRAINABLE_KEYS[:-4]

def basis_metadata(identity, *, architecture_mode='EVENT_PAIR74', source_pins=None):
    validate_identity(identity)
    active = trainable_keys(architecture_mode)
    if source_pins is None: source_pins = {str(name): hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in FORWARD_SOURCES}
    require(type(source_pins) is dict and set(source_pins) == set(map(str, FORWARD_SOURCES))
        and all(parent.native._sha(v) for v in source_pins.values()), 'complete74 component/parent/saved-port Source basis required')
    dependencies = {key: identity[key] for key in ('frozen_owner_state_sha256', 'native_owner_state_sha256', 'native_generic_state_sha256')}
    old_basis = parent.basis_metadata(identity['mode'], identity['encoder_dimension'], identity['hidden_dimension'], dependencies,
        source_pins={str(name): source_pins[str(name)] for name in parent.FORWARD_SOURCES})
    require(digest(old_basis) == identity['basis70_sha256'], 'original70 current Source/backend/mode basis differs')
    return {'version': VERSION, 'architecture_mode': architecture_mode, 'trainable_keys': list(active),
        'parent_origin': deepcopy(identity), 'parent_basis': old_basis,
        'residual_component': component.VERSION, 'residual_initialization_recipe': component.INITIALIZATION,
        'saved_port': {'version': saved_port.VERSION, 'basis': saved_port.BASIS},
        'parameter_shapes': {k: list(v) for k, v in shape_map(identity['encoder_dimension'], identity['hidden_dimension']).items()},
        'objective': OBJECTIVE, 'decoder': DECODER, 'training_computation': TRAINING_VERSION,
        'source_pins': dict(source_pins), 'NULL': 'UNCHANGED_SHARED_PARENT_EMBEDDING_AND_NATIVE_FACTOR_PATH',
        'runtime_admission_or_qualification_granted': False}

class EventConditionedPairArgumentExpert(nn.Module):
    architecture = VERSION

    def __init__(self, state, *, identity, expected_state_sha256, expected_basis_sha256, architecture_mode='EVENT_PAIR74'):
        super().__init__(); validate_identity(identity)
        encoder, hidden = identity['encoder_dimension'], identity['hidden_dimension']; validate_state(state, encoder, hidden)
        require(parent.native._sha(expected_state_sha256) and parent.native._sha(expected_basis_sha256)
            and state_dict_sha256(state) == expected_state_sha256, 'external complete74 state identity required before construction')
        self._basis = basis_metadata(identity, architecture_mode=architecture_mode); self._basis_sha256 = digest(self._basis)
        require(self._basis_sha256 == expected_basis_sha256, 'requested complete74 basis differs before construction')
        self._identity = deepcopy(identity); self.encoder_dimension, self.hidden_dimension = encoder, hidden
        original = {name: state['parent.'+name] for name in parent.shape_map(encoder, hidden)}
        self.parent = parent.model_from_state(original, mode=identity['mode'],
            expected_state_sha256=state_dict_sha256(original), expected_basis_sha256=identity['basis70_sha256'],
            expected_frozen_state_sha256=identity['frozen_owner_state_sha256'],
            expected_native_state_sha256=identity['native_owner_state_sha256'], expected_native_generic_sha256=identity['native_generic_state_sha256'],
            encoder_dimension=encoder, hidden_dimension=hidden)
        with torch.random.fork_rng(devices=[]): self.pair_residual = component.EventConditionedPairResidual()
        self.pair_residual.load_state_dict({name: state['pair_residual.'+name] for name in component.STATE_KEYS}, strict=True)
        self._inactive_residual_sha256 = state_dict_sha256(dict(self.pair_residual.state_dict()))
        self.eval().requires_grad_(False); self._validate_topology(); assert_frozen_unchanged(self)
        require(state_dict_sha256(dict(self.state_dict())) == expected_state_sha256
            and not {v.untyped_storage().data_ptr() for v in state.values()}.intersection(
                v.untyped_storage().data_ptr() for v in self.state_dict().values()), 'restored74 bits/storage differ')

    @property
    def mode(self): return self._identity['mode']

    @property
    def architecture_mode(self): return self._basis['architecture_mode']

    def _validate_topology(self):
        validate_state(dict(self.state_dict()), self.encoder_dimension, self.hidden_dimension)
        require(set(self._modules) == {'parent', 'pair_residual'} and not self._parameters and not self._buffers
            and type(self.parent) is parent.AnchoredBindingArgumentExpert
            and type(self.pair_residual) is component.EventConditionedPairResidual
            and self.parent.mode == self.mode and self.parent._basis_sha256 == self._identity['basis70_sha256']
            and digest(self._basis) == self._basis_sha256 and self._basis['parent_origin'] == self._identity
            and self._basis['trainable_keys'] == list(trainable_keys(self.architecture_mode))
            and set(dict(self.named_parameters())) == set(shape_map(self.encoder_dimension, self.hidden_dimension)),
            'exact single-parent74 topology and immutable origin/basis required')
        self.parent._validate_topology(); self.pair_residual._validate_owned()
        if self.architecture_mode == 'LEGACY70_CONTROL':
            require(state_dict_sha256(dict(self.pair_residual.state_dict())) == self._inactive_residual_sha256,
                'unused CONTROL residual4 changed')

    def forward(self, inputs):
        self._validate_topology()
        base = self.parent(inputs)
        if self.architecture_mode == 'LEGACY70_CONTROL': return base
        return self.pair_residual(inputs, base, factor_left_weight=self.parent.factor_left.weight,
            factor_right_weight=self.parent.factor_right.weight)

    def enable_training(self):
        self.parent.enable_training(); self.train()
        for name, parameter in self.named_parameters():
            parameter.grad = None; parameter.requires_grad_(name in trainable_keys(self.architecture_mode))
        assert_frozen_unchanged(self); return self

def assert_frozen_unchanged(net):
    require(type(net) is EventConditionedPairArgumentExpert, 'exact private74 owner required')
    return parent.assert_frozen_unchanged(net.parent)

def model_from_state(state, *, identity, expected_state_sha256, expected_basis_sha256, architecture_mode='EVENT_PAIR74'):
    return EventConditionedPairArgumentExpert(state, identity=identity,
        expected_state_sha256=expected_state_sha256, expected_basis_sha256=expected_basis_sha256,
        architecture_mode=architecture_mode)
