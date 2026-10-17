"""Restore complete inference owners from caller-supplied FP32 state mappings.

Content commitments authenticate the supplied bytes. They do not certify a
training history, model quality, independent origins, or a qualification run.
"""
from __future__ import annotations
from collections.abc import Mapping
import torch

from . import semantic_prefix_conditional_identity_native_v1 as type_native
from . import semantic_prefix_conditional_identity_contract_v1 as type_contract
from . import span_relative_context_assertion_null_v1 as local
from . import event_sentence_recurrent_identity_ensemble_v1 as original
from . import event_sentence_recurrent_identity_v1 as member
from . import three_member_conditioned_polarity_heads_v1 as head
from . import full_context_parallel_binding_owner_v1 as full
from . import event_conditioned_pair_owner_v1 as pair
from . import anchored_binding_owner_v1 as anchor
from . import context_canonicalizer_local_joint_score_v1 as canonical
from .tensor_state import state_dict_sha256
from .sequence_pointer_training import state_digest as raw_digest

COMPONENT_NAMES = ('type14', 'localizer', 'original66', 'heads', 'binding')
BINDING_KINDS = ('native74', 'full_context83')


def _require(ok, reason):
    if not ok:
        raise ValueError(reason)


def _state(state, shapes):
    _require(isinstance(state, Mapping) and set(state) == set(shapes),
             'complete supplied state key inventory required')
    pointers = set()
    for key, shape in shapes.items():
        value = state[key]
        _require(type(value) is torch.Tensor and value.device.type == 'cpu'
            and value.dtype == torch.float32 and value.is_contiguous()
            and tuple(value.shape) == tuple(shape) and not value.requires_grad
            and not value.is_inference() and bool(value.isfinite().all()),
            'ordinary detached finite contiguous CPU FP32 state required')
        pointers.add(value.untyped_storage().data_ptr())
    _require(len(pointers) == len(state), 'supplied parameter aliases are invalid')
    return {key: state[key] for key in shapes}


def original66_from_state(state):
    shapes = {f'members.{i}.{key}': shape for i in range(3)
              for key, shape in member._shapes(2048, 128).items()}
    state = _state(state, shapes)
    with torch.random.fork_rng(devices=[]), torch.inference_mode(False):
        models = [member.EventSentenceRecurrentIdentityHead(2048, 128) for _ in range(3)]
        for i, model in enumerate(models):
            model.load_state_dict({key: state[f'members.{i}.{key}']
                                   for key in member._shapes(2048, 128)}, strict=True)
        result = original.SentenceRecurrentIdentityProbabilityEnsemble(models)
    _require(state_dict_sha256(dict(result.state_dict())) == state_dict_sha256(state),
             'original66 supplied state restore differs')
    return result


def head96_from_state(state, *, original66_state_sha256):
    state = _state(state, head.STATE_SHAPES)
    frozen = tuple(head._part_digest(head._frozen_member_state(state, i),
                                    head.MEMBER_FROZEN_SHAPES) for i in range(3))
    members = tuple(state_dict_sha256({key.removeprefix(f'members.{i}.'): value
        for key, value in state.items() if key.startswith(f'members.{i}.')}) for i in range(3))
    selected = head._part_digest({key: state[key] for key in head.NONPOL_SHAPES}, head.NONPOL_SHAPES)
    bindings = head.SourceBindings(selected, members, frozen, state_dict_sha256(state),
                                  original66_state_sha256, head.INPUT_BASIS_SHA256)
    return head.model_from_state(state, architecture=head.VERSION,
        input_basis_sha256=head.INPUT_BASIS_SHA256,
        expected_state_sha256=head.state_digest(state), source_bindings=bindings)


def binding_from_state(state, *, binding_kind='full_context83', binding_mode='FULL_CONTEXT',
                       geometry_mode='RELATIVE'):
    """Restore all 74 or 83 tensors, including every frozen dependency tensor."""
    _require(binding_kind in BINDING_KINDS, 'declared native74 or full_context83 required')
    shapes = full.shape_map(2048, 128) if binding_kind == 'full_context83' else pair.shape_map(2048, 128)
    state = _state(state, shapes)
    state74 = ({key: state['native74.' + key] for key in pair.shape_map(2048, 128)}
               if binding_kind == 'full_context83' else state)
    state70 = {key: state74['parent.' + key] for key in anchor.shape_map(2048, 128)}
    frozen = {key[5:]: value for key, value in state70.items() if key in anchor.FROZEN_KEYS}
    native = {key: value for key, value in frozen.items() if key in anchor.native.am106.NATIVE_KEYS}
    dependencies = {'frozen_owner_state_sha256': raw_digest(frozen),
        'native_owner_state_sha256': raw_digest(native),
        'native_generic_state_sha256': state_dict_sha256(native)}
    basis70 = anchor.basis_metadata(geometry_mode, 2048, 128, dependencies)
    identity74 = {'version': pair.IDENTITY_VERSION, 'mode': geometry_mode,
        'encoder_dimension': 2048, 'hidden_dimension': 128,
        'original_state70_sha256': state_dict_sha256(state70),
        'basis70_sha256': anchor.digest(basis70), **dependencies}
    basis74 = pair.basis_metadata(identity74, architecture_mode='EVENT_PAIR74')
    if binding_kind == 'native74':
        _require(binding_mode in ('EVENT_PAIR74', 'FULL_CONTEXT'), 'native74 mode must retain its pair residual')
        return pair.model_from_state(state74, identity=identity74, architecture_mode='EVENT_PAIR74',
            expected_state_sha256=state_dict_sha256(state74), expected_basis_sha256=pair.digest(basis74))
    _require(binding_mode in full.MODES, 'declared complete83 architecture mode required')
    identity83 = {'version': full.IDENTITY_VERSION,
        'original_state74_sha256': state_dict_sha256(state74),
        'basis74_sha256': pair.digest(basis74), 'native74_identity': identity74}
    basis83 = full.basis_metadata(identity83, architecture_mode=binding_mode)
    return full.model_from_state(state, identity=identity83, architecture_mode=binding_mode,
        expected_state_sha256=state_dict_sha256(state), expected_basis_sha256=full.digest(basis83))


def components_from_states(states, *, binding_kind='full_context83', binding_mode='FULL_CONTEXT',
                           type_architecture='shared_bigru16', expected_state_sha256=None):
    from .runtime import SuppliedComponents
    _require(isinstance(states, Mapping) and set(states) == set(COMPONENT_NAMES),
             'exact five component state groups required')
    if expected_state_sha256 is not None:
        _require(type(expected_state_sha256) is dict and set(expected_state_sha256) == set(COMPONENT_NAMES)
            and all(state_dict_sha256(dict(states[key])) == expected_state_sha256[key] for key in COMPONENT_NAMES),
            'externally expected complete component state digest differs')
    type_state = _state(states['type14'], type_contract.SHAPES[type_architecture])
    loc_state = _state(states['localizer'], local.STATE_SHAPES)
    type_owner = type_native.owner_from_state(type_state, type_architecture)
    loc_owner = local.RelativeSpanContextAssertionNullHead(loc_state)
    identity_owner = original66_from_state(states['original66'])
    head_owner = head96_from_state(states['heads'],
                                  original66_state_sha256=state_dict_sha256(dict(identity_owner.state_dict())))
    binding_owner = binding_from_state(states['binding'], binding_kind=binding_kind, binding_mode=binding_mode)
    return SuppliedComponents(type_owner, loc_owner, identity_owner, head_owner, binding_owner)


def export_component_states(models):
    """Return independent ordinary FP32 groups suitable for a state bundle."""
    models.validate()
    with torch.inference_mode(False):
        return {name: {key: value.detach().clone().contiguous()
                       for key, value in getattr(models, name).state_dict().items()}
                for name in COMPONENT_NAMES}


def canonicalizer_from_state(state):
    return canonical.model_from_state(_state(state, canonical.STATE_SHAPES))


def slots_from_state(state):
    from .slots import LearnedSlotAssignment
    shapes = {'network.0.weight': (64, 2048), 'network.0.bias': (64,),
              'network.2.weight': (2, 64), 'network.2.bias': (2,)}
    state = _state(state, shapes)
    with torch.random.fork_rng(devices=[]), torch.inference_mode(False):
        result = LearnedSlotAssignment(2048, 64)
        result.load_state_dict(state, strict=True)
        result.eval().requires_grad_(False)
    _require(state_dict_sha256(dict(result.state_dict())) == state_dict_sha256(state),
             'slots4 supplied state restore differs')
    return result


def experimental_components(seed=0, binding_kind='full_context83'):
    """Genuine random, untrained owners for executable examples and tests only.

    This is not a semantic extraction model with demonstrated accuracy.
    All complete states are generated on a private CPU stream.
    """
    _require(type(seed) is int and 0 <= seed < 2**63, 'nonnegative explicit integer seed required')
    generator = torch.Generator(device='cpu').manual_seed(seed)
    def random_state(shapes):
        return {key: (torch.randn(tuple(shape), generator=generator, dtype=torch.float32) * .01).contiguous()
                for key, shape in shapes.items()}
    type_state = random_state(type_contract.SHAPES['shared_bigru16'])
    loc_state = random_state(local.STATE_SHAPES)
    original_shapes = {f'members.{i}.{key}': shape for i in range(3)
                       for key, shape in member._shapes(2048, 128).items()}
    states = {'type14': type_state, 'localizer': loc_state,
              'original66': random_state(original_shapes), 'heads': random_state(head.STATE_SHAPES)}
    if binding_kind == 'full_context83':
        state = random_state(full.shape_map(2048, 128))
        for key, value in full.initial_components(128).items():
            if key.startswith('cross_event.'):
                state[key] = value
    else:
        _require(binding_kind == 'native74', 'declared binding kind required')
        state = random_state(pair.shape_map(2048, 128))
    states['binding'] = state
    return components_from_states(states, binding_kind=binding_kind)
