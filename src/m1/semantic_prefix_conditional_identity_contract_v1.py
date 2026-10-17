"""Portable inference definitions; no research datasets or trained weights included."""

from __future__ import annotations

from dataclasses import dataclass

from m1.prefix_type_conditioning_layout_v1 import TYPES

from m1.prefix_type_conditioning_layout_v1 import DEFINITIONS

from m1.prefix_type_conditioning_layout_v1 import RawRow

from m1.prefix_type_conditioning_layout_v1 import digest

from m1.prefix_type_conditioning_layout_v1 import require

from m1.prefix_type_conditioning_layout_v1 import meaningful_prefix

from m1.prefix_type_conditioning_layout_v1 import DONOR_REVISION

from m1.prefix_type_conditioning_layout_v1 import DONOR_WEIGHT_SHA256

from m1.prefix_type_conditioning_layout_v1 import TOKENIZER_SHA256

VERSION = 'M1_SEMANTIC_PREFIX_CONDITIONAL_IDENTITY_CONTRACT_v1'

FORWARD_VERSION = 'M1_SEMANTIC_PREFIX_CURRENT_TURN_RAW_PRODUCER_v1'

OWNER_VERSION = 'M1_SEMANTIC_PREFIX_CONDITIONAL_IDENTITY_NATIVE_OWNER_v1'

ARCHITECTURES = ('mean_context', 'shared_bigru16')

UNIT_SHAPES = {'unit_weight': (16, 2048), 'unit_bias': (16,)}

GRU_SHAPES = {f'recurrent_{field}{suffix}': shape
    for suffix in ('', '_reverse')
    for field, shape in (('weight_ih', (48, 16)), ('weight_hh', (48, 16)),
                        ('bias_ih', (48,)), ('bias_hh', (48,)))}

SHAPES = {
    'mean_context': {**UNIT_SHAPES, 'context_weight': (16, 80),
        'context_bias': (16,), 'type_weight': (7, 16), 'type_bias': (7,)},
    'shared_bigru16': {**UNIT_SHAPES, **GRU_SHAPES,
        'context_weight': (16, 160), 'context_bias': (16,),
        'type_weight': (7, 16), 'type_bias': (7,)}}

PARAMETERS = {'mean_context': 34199, 'shared_bigru16': 38743}

FORMULAS = {a: 'UNIT_LN2048_EPS1E_5_LINEAR16_GELU;'
    + ('SHARED_BIDIRECTIONAL_GRU16_ALL_UNITS;' if a == 'shared_bigru16' else '')
    + 'ALL_SPANS_START_END_MEAN_PRECEDING_MEAN_FOLLOWING_MEAN;'
    + ('LINEAR160_16' if a == 'shared_bigru16' else 'LINEAR80_16')
    + '_GELU_LINEAR16_7;CANDIDATE_K_COLUMN_K' for a in ARCHITECTURES}

FORWARD_RECIPE = {
    'version': FORWARD_VERSION, 'candidate_order': TYPES,
    'semantic_definitions': DEFINITIONS,
    'prefixes': tuple(meaningful_prefix(k) for k in range(7)),
    'donor_revision': DONOR_REVISION, 'donor_weight_sha256': DONOR_WEIGHT_SHA256,
    'tokenizer_sha256': TOKENIZER_SHA256, 'selected_layer_zero_based': 11,
    'representation': 'POST_BLOCK_RESIDUAL_BEFORE_FINAL_RMSNORM',
    'add_special_tokens': False, 'padding': 'RIGHT', 'one_appended_eos': 248044,
    'max_tokens_including_one_eos': 512, 'truncation': False,
    'unit_policy': 'unicode_category_lexical_offsets/v1', 'unicode_version': '15.0.0',
    'source_units': 'COMPLETE_CANONICAL_ORIGINAL_PYTHON_UNICODE_CODEPOINT_SPANS',
    'pool': 'RAW_FP32_OVERLAP_MEAN', 'normalization': 'NONE',
    'all_original_CHAR_units_and_suffix': True, 'donor_learning': False,
    'production_original66_basis': False, 'sample_dependent_basis': False}

FORWARD_BASIS_SHA256 = digest(FORWARD_RECIPE)

def _sha(value):
    return type(value) is str and len(value) == 64 and not set(value)-set('0123456789abcdef')

def readout_basis(architecture):
    require(architecture in ARCHITECTURES, 'declared native6/14 architecture required')
    return digest({'version': OWNER_VERSION, 'forward_basis_sha256': FORWARD_BASIS_SHA256,
        'architecture': architecture, 'formula': FORMULAS[architecture],
        'shapes': SHAPES[architecture], 'parameters': PARAMETERS[architecture],
        'candidate_order': TYPES, 'NONE_or_eventhood': False,
        'production_original66_basis': False, 'donor_learning': False})

def raw_sample_digest(rows):
    require(type(rows) is tuple and rows and all(type(r) is RawRow for r in rows)
        and len({r.sentence_id for r in rows}) == len(rows),
        'complete unique annotation-free raw rows required')
    return digest([(r.sentence_id, r.text, r.language, r.units) for r in rows])

def candidate_item_id(raw_sample_sha256, sentence_id, candidate):
    require(_sha(raw_sample_sha256) and type(sentence_id) is str and sentence_id
        and type(candidate) is int and 0 <= candidate < 7,
        'raw call commitment, sentence ID and native candidate required')
    return digest([FORWARD_BASIS_SHA256, raw_sample_sha256, sentence_id, candidate])

@dataclass(frozen=True)
class PrefixUnitRecord:
    item_id: str
    sentence_id: str
    candidate: int
    forward_basis_sha256: str
    original_char_units: tuple
    physical_item_id: str
    fp32_units_sha256: str
    fp32_units: object

def input_commitment(rows, records):
    require(type(records) is tuple and all(type(r) is PrefixUnitRecord for r in records),
        'native prefix records required')
    return digest({'version': VERSION, 'raw_sample_sha256': raw_sample_digest(rows),
        'forward_basis_sha256': FORWARD_BASIS_SHA256,
        'logical_records': [(r.item_id, r.physical_item_id, r.fp32_units_sha256) for r in records]})

@dataclass(frozen=True)
class DetectedCharHead:
    sentence_id: str
    char_start: int
    char_end: int
