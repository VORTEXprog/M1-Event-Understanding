"""Portable inference definitions; no research datasets or trained weights included."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from torch import Tensor

from m1.tensor_state import state_dict_sha256

import m1.architecture_data_binding_matrix_evaluation_v1 as native_saved

from m1.structured_joint_role_argument_v1 import StructuredArgumentPotentials

import platform

VERSION = 'M1_EVENT_CONDITIONED_PAIR_PER_ROW_SAVED_PORT_v2'

BASIS = 'EVENT_PAIR32_ROLE_QUERY32_FULL_UNARY_SHARED_NATIVE_NULL_LR_FP32_ORIGINAL_UNIT_PAIRS'

FACTOR_KEYS = ('null_embedding', 'factor_left.weight', 'factor_right.weight')

BANK_KEYS = frozenset(('unary_logits', 'pair_vectors', 'query_vectors', 'pair_indices',
    'pair_mask', 'event_mask', 'token_mask', 'context_attention', 'role_pair_gate'))

MANIFEST_KEYS = frozenset(('version', 'basis', 'unit_width', 'event_slots',
                          'pair_count', 'factor_state_sha256'))

def _require(value, message):
    if not value:
        raise ValueError(message)

def _bits(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(
        a.detach().contiguous().view(torch.uint8), b.detach().contiguous().view(torch.uint8))

def _factors(state, gate):
    _require(isinstance(state, Mapping) and set(state) == set(FACTOR_KEYS),
             'exact selected three native factor tensors required; no owner envelope')
    # A live owner exposes its unchanged gate as nn.Parameter; the saved
    # arithmetic consumes detached ordinary Tensor values, preserving all bits.
    return native_saved._state_factors({**state, 'role_pair_gate': gate.detach()})

def _factor_digest(factors):
    return state_dict_sha256({key: factors[key] for key in FACTOR_KEYS})

def _ordinary_copy(value):
    return value.detach().clone().contiguous()

def pack_row(output: StructuredArgumentPotentials, batch_index: int, *, unit_width: int,
             factor_state: Mapping[str, Tensor]):
    """Return (small logical manifest, ordinary CPU tensor bank), without I/O.

All original event slots/masks, including inactive V/Q, remain explicit. Only
batch unit/pair padding is removed; NULL stays last. Inactive/padded factors
must use canonical +0, matching the actual native masked-fill domain. Signed
zero unary values are stored literally; active L/R must replay bit exactly.
"""
    _require(type(output) is StructuredArgumentPotentials, 'native structured potentials required')
    output.validate()
    _require(type(batch_index) is int and 0 <= batch_index < len(output.token_mask)
        and type(unit_width) is int and 0 < unit_width <= output.token_mask.shape[1]
        and output.unary_logits.device.type == 'cpu'
        and output.pair_vectors is not None and output.query_vectors is not None,
        'explicit original row/width and complete CPU V32/Q32 ports required')
    _require(not bool(output.token_mask[batch_index, unit_width:].any())
        and bool(output.token_mask[batch_index, :unit_width].all()),
        'original row extent must retain every valid unit')
    factors = _factors(factor_state, output.role_pair_gate)
    p = int(output.pair_mask[batch_index].sum())
    choices = torch.cat((output.pair_mask[batch_index], torch.ones(1, dtype=torch.bool)))
    valid = output.event_mask[batch_index, :, None, None] & choices[None, None, :]
    for source in (output.left_factors[batch_index], output.right_factors[batch_index]):
        invalid_values = source.masked_select(~valid[..., None].expand_as(source))
        _require(not bool(torch.signbit(invalid_values).any()),
                 'saved port requires canonical positive-zero inactive/padded factors')
    with torch.inference_mode(False), torch.no_grad():
        bank = {
            'unary_logits': _ordinary_copy(torch.cat((output.unary_logits[batch_index, :, :, :p],
                                                     output.unary_logits[batch_index, :, :, -1:]), -1)),
            'pair_vectors': _ordinary_copy(output.pair_vectors[batch_index, :, :p]),
            'query_vectors': _ordinary_copy(output.query_vectors[batch_index]),
            'pair_indices': _ordinary_copy(output.pair_indices[batch_index, :p]),
            'pair_mask': _ordinary_copy(output.pair_mask[batch_index, :p]),
            'event_mask': _ordinary_copy(output.event_mask[batch_index]),
            'token_mask': _ordinary_copy(output.token_mask[batch_index, :unit_width]),
            'context_attention': _ordinary_copy(output.context_attention[batch_index, :, :unit_width]),
            'role_pair_gate': _ordinary_copy(output.role_pair_gate),
        }
        manifest = {'version': VERSION, 'basis': BASIS, 'unit_width': unit_width,
            'event_slots': len(bank['event_mask']), 'pair_count': p,
            'factor_state_sha256': _factor_digest(factors)}
        reconstructed = reconstruct_row(manifest, bank, factor_state=factor_state)
        for name in ('left_factors', 'right_factors'):
            source = getattr(output, name)[batch_index]
            source = torch.cat((source[:, :, :p], source[:, :, -1:]), 2)
            if platform.system() != "Windows":
                _require(_bits(getattr(reconstructed, name)[0], source),
                        'native factor reconstruction differs from actual live row bits')
            else:
                pass
    return manifest, bank

def reconstruct_row(manifest, bank, *, factor_state: Mapping[str, Tensor]) -> StructuredArgumentPotentials:
    """Reconstruct one row, keeping E×P distinct and NULL on its native path."""
    _require(type(manifest) is dict and set(manifest) == MANIFEST_KEYS
        and manifest['version'] == VERSION and manifest['basis'] == BASIS,
        'exact versioned event-pair saved manifest required')
    t, e, p = (manifest[key] for key in ('unit_width', 'event_slots', 'pair_count'))
    _require(all(type(value) is int for value in (t, e, p)) and t > 0 and e >= 0 and p > 0,
             'explicit nonempty unit/pair extents and unrestricted nonnegative event slots required')
    _require(type(bank) is dict and set(bank) == BANK_KEYS, 'exact event-pair per-row tensor bank required')
    shapes = {'unary_logits': (e, 8, p+1), 'pair_vectors': (e, p, 32),
        'query_vectors': (e, 8, 32), 'pair_indices': (p, 2), 'pair_mask': (p,),
        'event_mask': (e,), 'token_mask': (t,), 'context_attention': (e, t), 'role_pair_gate': (28,)}
    for name, value in bank.items():
        dtype = (torch.int64 if name == 'pair_indices' else torch.bool
                 if name in ('pair_mask', 'event_mask', 'token_mask') else torch.float32)
        _require(type(value) is Tensor and value.device.type == 'cpu' and value.dtype == dtype
            and tuple(value.shape) == shapes[name] and value.is_contiguous()
            and not value.requires_grad and not value.is_inference()
            and (dtype != torch.float32 or bool(torch.isfinite(value).all())),
            'ordinary contiguous finite CPU saved tensor geometry differs')
    _require(bool(bank['pair_mask'].all()) and bool(bank['token_mask'].all()),
             'per-row bank retains all real pairs without batch pair padding')
    factors = _factors(factor_state, bank['role_pair_gate'])
    _require(manifest['factor_state_sha256'] == _factor_digest(factors),
             'selected native factor state differs from saved basis')
    with torch.inference_mode(False), torch.no_grad():
        left = torch.zeros((e, 8, p+1, 32), dtype=torch.float32)
        right = torch.zeros_like(left)
        for event in bank['event_mask'].nonzero().flatten().tolist():
            l, r = native_saved.reconstruct_factors(bank['pair_vectors'][event],
                bank['query_vectors'][event:event+1], factors)
            left[event], right[event] = l[0], r[0]
        # Copies keep the reconstructed score owner independent from its input
        # bank and preserve each retained unary/feature/mask bit exactly.
        output = StructuredArgumentPotentials(_ordinary_copy(bank['unary_logits'])[None],
            left[None], right[None], _ordinary_copy(bank['role_pair_gate']),
            _ordinary_copy(bank['pair_indices'])[None], _ordinary_copy(bank['pair_mask'])[None],
            _ordinary_copy(bank['event_mask'])[None], _ordinary_copy(bank['token_mask'])[None],
            _ordinary_copy(bank['context_attention'])[None], _ordinary_copy(bank['pair_vectors'])[None],
            _ordinary_copy(bank['query_vectors'])[None])
        output.validate()
    return output
