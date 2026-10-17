"""Portable inference definitions; no research datasets or trained weights included."""

from __future__ import annotations

from dataclasses import dataclass

import torch

import m1.canonicalizer_monotonic_edit_v1 as edit

from m1.fillers import PAD_BYTE_ID

MAXIMUM_TEXT_BYTES, CONTEXT_CHARACTERS = 4096, 32

MAXIMUM_INPUT_BYTES, MAXIMUM_OUTPUT_BYTES, MAXIMUM_CONTEXT_BYTES = 96, 64, 128

require = edit._require

@dataclass(frozen=True)
class ContextMention:
    text: str
    char_start: int
    char_end: int

@dataclass(frozen=True)
class MentionBatch:
    input_byte_ids: torch.Tensor
    input_lengths: torch.Tensor
    left_byte_ids: torch.Tensor
    left_lengths: torch.Tensor
    right_byte_ids: torch.Tensor
    right_lengths: torch.Tensor

def _pack(values):
    encoded = tuple(value.encode('utf8') for value in values)
    lengths = torch.tensor([len(value) for value in encoded], dtype=torch.int64, device='cpu')
    ids = torch.full((len(values), max(map(len, encoded))), PAD_BYTE_ID, dtype=torch.int64, device='cpu')
    for index, value in enumerate(encoded):
        if value: ids[index, :len(value)] = torch.tensor(list(value), dtype=torch.int64, device='cpu')
    return ids, lengths

def encode_mentions(mentions):
    require(type(mentions) in (tuple, list) and mentions and all(type(row) is ContextMention for row in mentions),
        'ordered nonempty exact raw ContextMention records required')
    surfaces, left, right = [], [], []
    for row in mentions:
        require(type(row.text) is str and type(row.char_start) is type(row.char_end) is int
            and 0 <= row.char_start < row.char_end <= len(row.text), 'exact Unicode character mention geometry required')
        try: raw = row.text.encode('utf8')
        except UnicodeEncodeError as exc: raise ValueError('valid raw UTF8 current text required') from exc
        require(len(raw) <= MAXIMUM_TEXT_BYTES, 'raw current text exceeds4096 UTF8 bytes')
        surface = row.text[row.char_start:row.char_end]
        require(0 < len(surface.encode('utf8')) <= MAXIMUM_INPUT_BYTES, 'nonempty verbatim mention cap96 required; no truncation')
        surfaces.append(surface)
        left.append(row.text[max(0, row.char_start - CONTEXT_CHARACTERS):row.char_start])
        right.append(row.text[row.char_end:row.char_end + CONTEXT_CHARACTERS])
    with torch.inference_mode(False):
        ids, lengths = _pack(surfaces); left_ids, left_lengths = _pack(left); right_ids, right_lengths = _pack(right)
    batch = MentionBatch(ids, lengths, left_ids, left_lengths, right_ids, right_lengths)
    validate_batch(batch)
    return batch

def _context_bytes(ids, lengths, count):
    require(type(ids) is type(lengths) is torch.Tensor and ids.dtype == lengths.dtype == torch.int64
        and ids.device.type == lengths.device.type == 'cpu' and ids.ndim == 2 and len(ids) == count
        and 0 <= ids.shape[1] <= MAXIMUM_CONTEXT_BYTES and lengths.shape == (count,)
        and bool((lengths >= 0).all()) and bool((lengths <= ids.shape[1]).all()), 'honest bounded context lengths0 required')
    mask = torch.arange(ids.shape[1])[None] < lengths[:, None]
    require(bool(((ids[mask] >= 0) & (ids[mask] < 256)).all()) and bool((ids[~mask] == PAD_BYTE_ID).all()),
        'context PAD exactly outside actual UTF8 bytes required')
    for row, length in zip(ids.tolist(), lengths.tolist(), strict=True):
        try: value = bytes(row[:length]).decode('utf8')
        except UnicodeDecodeError as exc: raise ValueError('valid context UTF8 bytes required') from exc
        require(len(value) <= CONTEXT_CHARACTERS, 'each context window is at most32 Unicode characters')

def validate_batch(batch):
    require(type(batch) is MentionBatch, 'exact typed MentionBatch required; no annotation dictionary ingress')
    edit._bytes(batch.input_byte_ids, batch.input_lengths, MAXIMUM_INPUT_BYTES)
    for ids, lengths in ((batch.left_byte_ids, batch.left_lengths), (batch.right_byte_ids, batch.right_lengths)):
        _context_bytes(ids, lengths, len(batch.input_byte_ids))
