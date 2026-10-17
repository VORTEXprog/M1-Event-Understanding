"""Portable inference definitions; no research datasets or trained weights included."""

from __future__ import annotations

from dataclasses import dataclass

import hashlib

import torch

import m1.semantic_prefix_conditional_identity_native_v1 as native

import m1.semantic_prefix_conditional_identity_contract_v1 as contract

import m1.typed_event_set_localization_loss_v1 as joint

ARCHITECTURE = 'shared_bigru16'

require = contract.require

def _hash(value):
    return hashlib.sha256(value.detach().contiguous().numpy().tobytes()).hexdigest()

def _native_bank(bank):
    require(type(bank) is native.DenseSpanCandidateScores and bank.architecture == ARCHITECTURE
        and bank.readout_basis_sha256 == contract.readout_basis(ARCHITECTURE)
        and contract._sha(bank.state_sha256) and contract._sha(bank.input_sha256)
        and contract.raw_sample_digest(bank.rows) == bank.raw_sample_sha256
        and type(bank.row_pair_offsets) is tuple and len(bank.row_pair_offsets) == len(bank.rows)+1
        and bank.row_pair_offsets[0] == 0 and all(type(v) is int for v in bank.row_pair_offsets)
        and type(bank.scores) is torch.Tensor and bank.scores.device.type == 'cpu'
        and bank.scores.dtype == torch.float32 and bank.scores.is_contiguous()
        and not bank.scores.requires_grad and not bank.scores.is_inference()
        and tuple(bank.scores.shape) == (bank.row_pair_offsets[-1], 7) and bool(bank.scores.isfinite().all())
        and type(bank.pairs) is torch.Tensor and bank.pairs.device.type == 'cpu'
        and bank.pairs.dtype == torch.int64 and bank.pairs.is_contiguous()
        and not bank.pairs.is_inference() and tuple(bank.pairs.shape) == (len(bank.scores), 2)
        and bank.tensor_versions == (native._tensor_version(bank.scores), native._tensor_version(bank.pairs)),
        'complete unchanged native14 FP32/P7 END-major bank required')
    for row, start, end in zip(bank.rows, bank.row_pair_offsets[:-1], bank.row_pair_offsets[1:], strict=True):
        expected = native._all_pairs(len(row.units))
        require(end-start == len(expected) and torch.equal(bank.pairs[start:end], expected),
            'complete native14 END-major original half-open interval geometry required')
    return bank

@dataclass(frozen=True)
class FrozenTypeIntervals:
    """All logits with explicit START-major local unit pairs and row offsets."""
    rows: tuple
    logits: object
    pair_indices: object
    row_pair_offsets: tuple
    native_bank: native.DenseSpanCandidateScores
    native_scores_sha256: str
    tensor_versions: tuple

def align_complete_bank(bank):
    """A bijection preserves every FP32 score bit and every native TYPE column."""
    _native_bank(bank); before = _hash(bank.scores); values, pairs = [], []
    for row, begin, end in zip(bank.rows, bank.row_pair_offsets[:-1], bank.row_pair_offsets[1:], strict=True):
        canonical = joint.complete_pairs(len(row.units))
        start, stop = canonical.unbind(-1)
        index = stop*(stop-1)//2+start
        require(len(index) == end-begin and torch.equal(index.sort().values, torch.arange(len(index)))
            and torch.equal(bank.pairs[begin:end][index], canonical), 'exact END->START-major complete permutation required')
        values.append(bank.scores[begin:end][index].clone().contiguous()); pairs.append(canonical.clone())
    with torch.inference_mode(False):
        logits = torch.cat(values).contiguous(); geometry = torch.cat(pairs).contiguous()
    _native_bank(bank); require(_hash(bank.scores) == before, 'native TYPE bank changed during alignment')
    result = FrozenTypeIntervals(bank.rows, logits, geometry, bank.row_pair_offsets, bank, before,
        (native._tensor_version(logits), native._tensor_version(geometry)))
    validate_intervals(result); return result

def validate_intervals(value):
    require(type(value) is FrozenTypeIntervals, 'exact complete TYPE interval DTO required')
    bank = _native_bank(value.native_bank)
    require(value.rows == bank.rows and value.row_pair_offsets == bank.row_pair_offsets
        and value.native_scores_sha256 == _hash(bank.scores)
        and type(value.logits) is torch.Tensor and value.logits.dtype == torch.float32
        and value.logits.device.type == 'cpu' and value.logits.is_contiguous()
        and not value.logits.requires_grad and not value.logits.is_inference()
        and tuple(value.logits.shape) == tuple(bank.scores.shape) and bool(value.logits.isfinite().all())
        and type(value.pair_indices) is torch.Tensor and value.pair_indices.dtype == torch.int64
        and value.pair_indices.device.type == 'cpu' and value.pair_indices.is_contiguous()
        and not value.pair_indices.is_inference() and tuple(value.pair_indices.shape) == tuple(bank.pairs.shape)
        and value.tensor_versions == (native._tensor_version(value.logits), native._tensor_version(value.pair_indices))
        and not {value.logits.untyped_storage().data_ptr(), value.pair_indices.untyped_storage().data_ptr()}
            .intersection((bank.scores.untyped_storage().data_ptr(), bank.pairs.untyped_storage().data_ptr())),
        'unchanged complete disjoint aligned TYPE logits/geometry required')
    for row, begin, end in zip(value.rows, value.row_pair_offsets[:-1], value.row_pair_offsets[1:], strict=True):
        expected = joint.complete_pairs(len(row.units)); start, stop = expected.unbind(-1)
        index = stop*(stop-1)//2+start
        require(torch.equal(value.pair_indices[begin:end], expected)
            and torch.equal(value.logits[begin:end].view(torch.int32), bank.scores[begin:end][index].contiguous().view(torch.int32)),
            'every START-major interval and all seven original logit bits required')
    return value
