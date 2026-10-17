"""Portable inference definitions; no research datasets or trained weights included."""

from __future__ import annotations

from dataclasses import dataclass

import hashlib

import json

import torch

from torch import nn

from m1.semantic_prefix_conditional_identity_contract_v1 import OWNER_VERSION

from m1.semantic_prefix_conditional_identity_contract_v1 import TYPES

from m1.semantic_prefix_conditional_identity_contract_v1 import ARCHITECTURES

from m1.semantic_prefix_conditional_identity_contract_v1 import SHAPES

from m1.semantic_prefix_conditional_identity_contract_v1 import GRU_SHAPES

from m1.semantic_prefix_conditional_identity_contract_v1 import PARAMETERS

from m1.semantic_prefix_conditional_identity_contract_v1 import FORMULAS

from m1.semantic_prefix_conditional_identity_contract_v1 import FORWARD_BASIS_SHA256

from m1.semantic_prefix_conditional_identity_contract_v1 import RawRow

from m1.semantic_prefix_conditional_identity_contract_v1 import PrefixUnitRecord

from m1.semantic_prefix_conditional_identity_contract_v1 import DetectedCharHead

from m1.semantic_prefix_conditional_identity_contract_v1 import candidate_item_id

from m1.semantic_prefix_conditional_identity_contract_v1 import raw_sample_digest

from m1.semantic_prefix_conditional_identity_contract_v1 import input_commitment

from m1.semantic_prefix_conditional_identity_contract_v1 import readout_basis

from m1.semantic_prefix_conditional_identity_contract_v1 import require

from m1.semantic_prefix_conditional_identity_contract_v1 import _sha

VERSION = OWNER_VERSION

def _tensor_version(x):
    return (id(x), x.untyped_storage().data_ptr(), x._version)

def _fp32_sha(x):
    return hashlib.sha256(x.detach().contiguous().numpy().tobytes()).hexdigest()

def _validate_state(state, architecture):
    require(architecture in ARCHITECTURES and type(state) is dict
        and tuple(state) == tuple(SHAPES[architecture]), 'closed ordered native6/14 state required')
    require(all(type(v) is torch.Tensor and tuple(v.shape) == SHAPES[architecture][k]
        and v.device.type == 'cpu' and v.dtype == torch.float32 and v.is_contiguous()
        and not v.requires_grad and not v.is_inference() and bool(v.isfinite().all())
        for k, v in state.items()), 'ordinary detached finite CPU FP32 supplied state required')
    require(len({v.untyped_storage().data_ptr() for v in state.values()}) == len(state)
        and sum(v.numel() for v in state.values()) == PARAMETERS[architecture],
        'native parameter tensors must be disjoint and complete')

def state_digest(state, architecture):
    _validate_state(state, architecture)
    h = hashlib.sha256()
    for key, value in state.items():
        metadata = json.dumps([key, 'FLOAT32', list(value.shape)], separators=(',', ':')).encode()
        h.update(len(metadata).to_bytes(8, 'little')); h.update(metadata)
        payload = value.numpy().tobytes()
        h.update(len(payload).to_bytes(8, 'little')); h.update(payload)
    return h.hexdigest()

def state_contract(architecture):
    require(architecture in ARCHITECTURES, 'declared native architecture required')
    return {'version': VERSION, 'architecture': architecture,
        'native_class': __name__ + '.SemanticPrefixConditionalIdentityOwner',
        'parameter_keys': list(SHAPES[architecture]), 'parameter_shapes': SHAPES[architecture],
        'parameter_tensors': len(SHAPES[architecture]), 'parameters': PARAMETERS[architecture],
        'forward_basis_sha256': FORWARD_BASIS_SHA256,
        'readout_basis_sha256': readout_basis(architecture), 'formula': FORMULAS[architecture],
        'candidate_order': TYPES, 'dtype': 'FLOAT32', 'device': 'cpu',
        'default_learning': False, 'random_initializer': False,
        'NONE_or_eventhood': False, 'old_native75_owner': False}

class SemanticPrefixConditionalIdentityOwner(nn.Module):
    """Exact native owner with supplied independent tensors and frozen inference.

    This constructor performs no random draw and uses no ambient seed. Real
    supplied states and real RAW calls require their external accepted scope.
    """
    def __init__(self, state, architecture, *, fixture_only_not_admissible=False):
        _validate_state(state, architecture)
        require(type(fixture_only_not_admissible) is bool, 'explicit fixture mode required')
        super().__init__()
        self.architecture = architecture
        self.forward_basis_sha256 = FORWARD_BASIS_SHA256
        self.input_basis_sha256 = readout_basis(architecture)
        self.fixture_only_not_admissible = fixture_only_not_admissible
        for key, value in state.items():
            self.register_parameter(key, nn.Parameter(value.detach().clone(), requires_grad=False))
        self.eval()
        self._owned_versions = tuple((k, *_tensor_version(v)) for k, v in self.named_parameters())
        self._frozen_state_sha256 = state_digest(self._detached_state(), architecture)
        require(not {v.untyped_storage().data_ptr() for v in self.parameters()}
            .intersection(v.untyped_storage().data_ptr() for v in state.values()),
            'native owner must clone every supplied tensor')

    def _detached_state(self):
        return {k: v.detach() for k, v in self.named_parameters()}

    def validate_frozen(self):
        require(type(self) is SemanticPrefixConditionalIdentityOwner
            and not self.training and not self._modules and not self._buffers
            and all(not p.requires_grad and p.grad is None for p in self.parameters())
            and self.forward_basis_sha256 == FORWARD_BASIS_SHA256
            and self.input_basis_sha256 == readout_basis(self.architecture),
            'exact frozen native owner and static basis required')
        state = self._detached_state()
        require(tuple((k, *_tensor_version(v)) for k, v in self.named_parameters()) == self._owned_versions
            and state_digest(state, self.architecture) == self._frozen_state_sha256,
            'frozen native owner state or pointer/version changed')
        return self._frozen_state_sha256

    def forward(self, raw7, pairs):
        self.validate_frozen()
        require(type(raw7) is torch.Tensor and raw7.ndim == 3 and raw7.shape[0] == 7
            and raw7.shape[1] > 0 and raw7.shape[2] == 2048 and raw7.device.type == 'cpu'
            and raw7.dtype == torch.float32 and not raw7.requires_grad and not raw7.is_inference()
            and bool(raw7.isfinite().all()), 'complete ordinary RAW2048 seven-candidate row required')
        require(type(pairs) is torch.Tensor and pairs.device.type == 'cpu' and pairs.dtype == torch.int64
            and torch.equal(pairs, _all_pairs(raw7.shape[1])), 'complete ordered original intervals required')
        before = (_tensor_version(raw7), _tensor_version(pairs))
        with torch.no_grad():
            result = self._forward_arithmetic(raw7, pairs)
        require(before == (_tensor_version(raw7), _tensor_version(pairs)), 'native forward mutated inputs')
        self.validate_frozen()
        require(tuple(result.shape) == (len(pairs), 7) and bool(result.isfinite().all()),
            'complete finite native candidate scores required')
        return result

    def _forward_arithmetic(self, raw7, pairs):
        from torch.nn import functional as F
        p = F.gelu(F.linear(F.layer_norm(raw7, (2048,), eps=1e-5), self.unit_weight, self.unit_bias))
        if self.architecture == 'shared_bigru16':
            from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence
            packed = pack_padded_sequence(p, (p.shape[1],)*7, batch_first=True, enforce_sorted=False)
            weights = [getattr(self, key) for key in GRU_SHAPES]
            h0 = torch.zeros((2, 7, 16), dtype=p.dtype, device=p.device)
            values, _ = torch._VF.gru(packed.data, packed.batch_sizes, h0,
                weights, True, 1, 0., False, True)
            encoded = PackedSequence(values, packed.batch_sizes, packed.sorted_indices, packed.unsorted_indices)
            p, _ = pad_packed_sequence(encoded, batch_first=True, total_length=p.shape[1])
        prefix = F.pad(p.cumsum(1), (0, 0, 1, 0))
        i, j = pairs.unbind(-1); width = raw7.shape[1]
        mean = (prefix[:, j]-prefix[:, i])/(j-i).to(torch.float32)[None, :, None]
        before = prefix[:, i]/i.clamp_min(1).to(torch.float32)[None, :, None]
        after = (prefix[:, width, None]-prefix[:, j])/(width-j).clamp_min(1).to(torch.float32)[None, :, None]
        z = torch.cat((p[:, i], p[:, j-1], mean, before, after), dim=-1)
        full = F.linear(F.gelu(F.linear(z, self.context_weight, self.context_bias)), self.type_weight, self.type_bias)
        # Every candidate passes the identical shared architecture; no label
        # selects its upstream representation or its target score column.
        return torch.stack([full[k, :, k] for k in range(7)], dim=-1)

def owner_from_state(state, architecture, *, fixture_only_not_admissible=False):
    return SemanticPrefixConditionalIdentityOwner(state, architecture,
        fixture_only_not_admissible=fixture_only_not_admissible)

def export_cpu_fp32_state(owner):
    require(type(owner) is SemanticPrefixConditionalIdentityOwner, 'exact native owner required')
    sha = owner.validate_frozen()
    state = {k: v.detach().clone() for k, v in owner.named_parameters()}
    require(state_digest(state, owner.architecture) == sha, 'export state digest differs')
    owner.validate_frozen()
    return state, {**state_contract(owner.architecture), 'state_sha256': sha,
        'fixture_only_not_admissible': owner.fixture_only_not_admissible}

@dataclass(frozen=True)
class PrefixReadoutInputs:
    rows: tuple[RawRow, ...]
    records: tuple[PrefixUnitRecord, ...]
    raw_sample_sha256: str
    input_sha256: str
    forward_basis_sha256: str
    fixture_only_not_admissible: bool
    tensor_versions: tuple

def authenticate_inputs(*, rows, records, raw_sample_sha256, input_sha256,
        fixture_only_not_admissible=False):
    require(type(fixture_only_not_admissible) is bool and _sha(raw_sample_sha256)
        and _sha(input_sha256) and raw_sample_digest(rows) == raw_sample_sha256,
        'explicit mode and annotation-free raw call commitment required')
    if fixture_only_not_admissible:
        require(all(r.sentence_id.startswith('invented/') for r in rows), 'invented SOURCE fixtures only')
    from m1.canonical_units import segment_text
    require(all(r.units == tuple((u.char_start, u.char_end) for u in segment_text(r.text)) for r in rows),
        'complete canonical original CHAR units including suffix required')
    require(type(records) is tuple and all(type(r) is PrefixUnitRecord for r in records)
        and [(r.sentence_id, r.candidate) for r in records]
            == [(row.sentence_id, k) for row in rows for k in range(7)]
        and input_commitment(rows, records) == input_sha256, 'complete ordered seven-candidate call required')
    by_sid = {r.sentence_id: r for r in rows}; versions = []
    for record in records:
        x = record.fp32_units; row = by_sid[record.sentence_id]
        item_id = candidate_item_id(raw_sample_sha256, record.sentence_id, record.candidate)
        require(record.forward_basis_sha256 == FORWARD_BASIS_SHA256
            and record.original_char_units == row.units and record.item_id == item_id
            and record.physical_item_id == item_id and _sha(record.fp32_units_sha256),
            'static recipe, per-call physical identity or full CHAR geometry differs')
        require(type(x) is torch.Tensor and tuple(x.shape) == (len(row.units), 2048)
            and x.device.type == 'cpu' and x.dtype == torch.float32 and x.is_contiguous()
            and not x.requires_grad and not x.is_inference() and bool(x.isfinite().all())
            and _fp32_sha(x) == record.fp32_units_sha256,
            'ordinary detached finite complete RAW2048 payload commitment required')
        versions.append(_tensor_version(x))
    require(len({v[1] for v in versions}) == len(versions), 'candidate payload storages must be disjoint')
    return PrefixReadoutInputs(rows, records, raw_sample_sha256, input_sha256,
        FORWARD_BASIS_SHA256, fixture_only_not_admissible, tuple(versions))

def _unchanged(inputs):
    require(type(inputs) is PrefixReadoutInputs and inputs.forward_basis_sha256 == FORWARD_BASIS_SHA256
        and raw_sample_digest(inputs.rows) == inputs.raw_sample_sha256
        and input_commitment(inputs.rows, inputs.records) == inputs.input_sha256
        and tuple(_tensor_version(r.fp32_units) for r in inputs.records) == inputs.tensor_versions
        and all(_fp32_sha(r.fp32_units) == r.fp32_units_sha256 for r in inputs.records),
        'authenticated raw call or prefix tensor mutation detected')

def _all_pairs(width):
    return torch.tensor([(i, j) for j in range(1, width+1) for i in range(j)], dtype=torch.int64)

@dataclass(frozen=True)
class DenseSpanCandidateScores:
    architecture: str
    raw_sample_sha256: str
    input_sha256: str
    readout_basis_sha256: str
    state_sha256: str
    rows: tuple[RawRow, ...]
    scores: object
    pairs: object
    row_pair_offsets: tuple[int, ...]
    fixture_only_not_admissible: bool
    tensor_versions: tuple

def score_complete_dense(inputs, owner):
    """All intervals times all seven candidates, independently of HEAD presence."""
    _unchanged(inputs)
    require(type(owner) is SemanticPrefixConditionalIdentityOwner
        and owner.fixture_only_not_admissible is inputs.fixture_only_not_admissible,
        'exact native owner and matching fixture mode required')
    sha = owner.validate_frozen()
    require(not {r.fp32_units.untyped_storage().data_ptr() for r in inputs.records}
        .intersection(p.untyped_storage().data_ptr() for p in owner.parameters()), 'RAW/state alias rejected')
    records = {(r.sentence_id, r.candidate): r for r in inputs.records}
    scores, geometries, offsets = [], [], [0]
    for row in inputs.rows:
        pairs = _all_pairs(len(row.units))
        raw7 = torch.stack([records[(row.sentence_id, k)].fp32_units for k in range(7)])
        scores.append(owner(raw7, pairs)); geometries.append(pairs)
        offsets.append(offsets[-1]+len(pairs))
    values = torch.cat(scores).contiguous(); pairs = torch.cat(geometries).contiguous()
    require(tuple(values.shape) == (offsets[-1], 7) and bool(values.isfinite().all()),
        'complete dense seven-candidate score bank required')
    _unchanged(inputs); require(owner.validate_frozen() == sha, 'state changed during dense scoring')
    return DenseSpanCandidateScores(owner.architecture, inputs.raw_sample_sha256, inputs.input_sha256,
        owner.input_basis_sha256, sha, inputs.rows, values, pairs, tuple(offsets),
        inputs.fixture_only_not_admissible, (_tensor_version(values), _tensor_version(pairs)))

@dataclass(frozen=True)
class DetectedHeadIdentity:
    heads: tuple[DetectedCharHead, ...]
    scores: object
    type_indices: object
    type_names: tuple[str, ...]
    readout_basis_sha256: str
    input_sha256: str
    state_sha256: str
    fixture_only_not_admissible: bool

def detected_head_readout(bank, heads):
    """Join the exact observed whole-CHAR HEAD tuple after full dense scoring.

    E0 is the empty supplied tuple. No Gold TYPE/count, NONE class, whitelist,
    shortlist, confidence threshold, padding or silent HEAD truncation enters.
    """
    require(type(bank) is DenseSpanCandidateScores and type(heads) is tuple
        and all(type(h) is DetectedCharHead for h in heads), 'native dense bank and observed HEAD tuple required')
    require(len({(h.sentence_id, h.char_start, h.char_end) for h in heads}) == len(heads),
        'detected HEAD event keys must be unique')
    require(bank.architecture in ARCHITECTURES and bank.readout_basis_sha256 == readout_basis(bank.architecture)
        and tuple(bank.scores.shape) == (bank.row_pair_offsets[-1], 7)
        and bank.scores.dtype == torch.float32 and bank.scores.device.type == 'cpu'
        and bool(bank.scores.isfinite().all()) and bank.pairs.dtype == torch.int64
        and bank.pairs.device.type == 'cpu'
        and (_tensor_version(bank.scores), _tensor_version(bank.pairs)) == bank.tensor_versions,
        'complete unchanged native score bank required')
    require(len(bank.row_pair_offsets) == len(bank.rows)+1 and bank.row_pair_offsets[0] == 0,
        'complete dense row offsets required')
    locations = {}
    for row_index, row in enumerate(bank.rows):
        start, end = bank.row_pair_offsets[row_index:row_index+2]
        expected = _all_pairs(len(row.units))
        require(end-start == len(expected) and torch.equal(bank.pairs[start:end], expected),
            'complete original interval geometry required')
        locations[row.sentence_id] = (row, start)
    selected = []
    for head in heads:
        require(type(head.sentence_id) is str and head.sentence_id in locations
            and type(head.char_start) is type(head.char_end) is int
            and 0 <= head.char_start < head.char_end, 'in-bounds detected CHAR HEAD required')
        row, offset = locations[head.sentence_id]
        starts = {s: i for i, (s, e) in enumerate(row.units)}
        ends = {e: i+1 for i, (s, e) in enumerate(row.units)}
        require(head.char_start in starts and head.char_end in ends,
            'detected HEAD must cover whole original CHAR units')
        i, j = starts[head.char_start], ends[head.char_end]
        require(i < j, 'detected HEAD interval is reversed')
        selected.append(offset + j*(j-1)//2 + i)
    scores = bank.scores[selected].clone() if selected else bank.scores.new_empty((0, 7))
    indices = scores.argmax(-1) if selected else torch.empty((0,), dtype=torch.int64)
    # torch.argmax returns the first index on ties: immutable native TYPE order.
    return DetectedHeadIdentity(heads, scores, indices, tuple(TYPES[i] for i in indices.tolist()),
        bank.readout_basis_sha256, bank.input_sha256, bank.state_sha256, bank.fixture_only_not_admissible)
