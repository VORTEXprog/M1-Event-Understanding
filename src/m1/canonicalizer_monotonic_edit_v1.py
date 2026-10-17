"""Portable inference definitions; no research datasets or trained weights included."""

from __future__ import annotations

from dataclasses import dataclass

from functools import lru_cache

import torch

from torch import nn

from torch.nn import functional as F

from m1.canonicalizer import BOS_ID

from m1.canonicalizer import EOS_ID

from m1.canonicalizer import SEQUENCE_VOCABULARY_SIZE

from m1.canonicalizer import CanonicalDecode

from m1.constrained_canonicalizer import feasible_token_ids

from m1.constrained_canonicalizer import valid_symbol

from m1.fillers import PAD_BYTE_ID

from m1.fillers import utf8_byte_batch

MAXIMUM_INPUT_BYTES, MAXIMUM_OUTPUT_BYTES = 96, 64

COPY, DELETE, SUBSTITUTE, INSERT, STOP = range(5)

PARAMETER_KEYS, PARAMETER_COUNT = 24, 693125

@dataclass(frozen=True)
class EditLatticeLogProbabilities:
    action_log_probs: torch.Tensor  # [B,N+1,M+1,5]
    byte_log_probs: torch.Tensor    # [B,N+1,M+1,256]
    input_byte_ids: torch.Tensor    # [B,N], PAD exactly outside length
    input_lengths: torch.Tensor    # [B]
    target_byte_ids: torch.Tensor   # [B,M], teacher forcing/loss only
    target_lengths: torch.Tensor   # [B]
    node_mask: torch.Tensor         # [B,N+1,M+1], valid supervised lattice nodes

def _require(ok, reason):
    if not ok: raise ValueError(reason)

def _bytes(ids, lengths, cap, *, symbols=False):
    _require(type(ids) is torch.Tensor and type(lengths) is torch.Tensor
        and ids.dtype == lengths.dtype == torch.int64 and ids.device.type == lengths.device.type == 'cpu'
        and ids.ndim == 2 and ids.shape[0] > 0 and 0 < ids.shape[1] <= cap
        and lengths.shape == (len(ids),) and bool((lengths > 0).all())
        and bool((lengths <= ids.shape[1]).all()), 'nonempty bounded CPU byte/length tensors required')
    mask = torch.arange(ids.shape[1])[None] < lengths[:, None]
    _require(bool(((ids[mask] >= 0) & (ids[mask] < 256)).all())
        and bool((ids[~mask] == PAD_BYTE_ID).all()), 'PAD exactly outside bounded raw UTF8 bytes required')
    for row, length in zip(ids.tolist(), lengths.tolist(), strict=True):
        try: text = bytes(row[:length]).decode('utf-8')
        except UnicodeDecodeError as exc: raise ValueError('valid UTF8 byte sequence required') from exc
        if symbols: _require(valid_symbol(text), 'nonempty valid canonical symbol target required')
    return mask

@lru_cache(maxsize=8192)
def _syntax(prefix):
    """Pure structural cache, independent of model/input/target endpoint."""
    allowed = feasible_token_ids(prefix, MAXIMUM_OUTPUT_BYTES)
    return tuple(i for i in allowed if i < 256), prefix.startswith((b'entity/', b'object/')), EOS_ID in allowed

def _prefix_masks(prefixes):
    allowed = torch.zeros(len(prefixes), 256, dtype=torch.bool)
    complete, legal_stop = [], []
    for row, prefix in enumerate(prefixes):
        values, header, stop = _syntax(prefix)
        if values: allowed[row, list(values)] = True
        complete.append(header); legal_stop.append(stop)
    return allowed, torch.tensor(complete, dtype=torch.bool), torch.tensor(legal_stop, dtype=torch.bool)

def _masked_log_softmax(logits, mask):
    """All-absent conditional branches have zero mass and safe zero gradients."""
    any_valid = mask.any(-1, keepdim=True)
    first = torch.arange(mask.shape[-1]) == 0
    safe_mask = mask | (~any_valid & first)
    result = F.log_softmax(logits.masked_fill(~safe_mask, -torch.inf), dim=-1)
    return result.masked_fill(~mask, -torch.inf)

def _safe_logsumexp(values):
    finite = torch.isfinite(values).any(-1)
    safe = torch.where(finite[..., None], values, torch.zeros_like(values))
    return torch.where(finite, torch.logsumexp(safe, -1), torch.full_like(finite, -torch.inf, dtype=values.dtype))

def _groups(byte_allowed, prefix_complete, legal_stop, current_byte, cursor, lengths):
    consume = cursor < lengths
    at_body = consume & prefix_complete
    byte_possible = byte_allowed.any(-1)
    copy_allowed = byte_allowed.gather(-1, current_byte.clamp(0, 255)[..., None]).squeeze(-1)
    return torch.stack((at_body & copy_allowed, at_body, at_body & byte_possible,
        byte_possible, (cursor == lengths) & legal_stop), -1)

class MonotonicByteEditTransducer(nn.Module):
    """Fixed24-key candidate. Runtime state is local to each decode call."""
    maximum_input_bytes, maximum_output_bytes = MAXIMUM_INPUT_BYTES, MAXIMUM_OUTPUT_BYTES

    def __init__(self):
        super().__init__()
        factory = {'device': 'cpu', 'dtype': torch.float32}
        self.input_embedding = nn.Embedding(SEQUENCE_VOCABULARY_SIZE, 64, padding_idx=PAD_BYTE_ID, **factory)
        self.encoder = nn.GRU(64, 128, batch_first=True, bidirectional=True, **factory)
        self.prediction_embedding = nn.Embedding(SEQUENCE_VOCABULARY_SIZE, 64, padding_idx=PAD_BYTE_ID, **factory)
        self.prediction = nn.GRU(64, 256, batch_first=True, **factory)
        self.initial_prediction = nn.Linear(256, 256, **factory)
        self.action_join = nn.Linear(512, 128, **factory)
        self.action_output = nn.Linear(128, 5, **factory)
        self.source_byte = nn.Linear(256, 256, **factory)
        self.prediction_byte = nn.Linear(256, 256, **factory)
        _require(len(self.state_dict()) == PARAMETER_KEYS and sum(p.numel() for p in self.parameters()) == PARAMETER_COUNT,
            'declared fixed24 inventory differs')

    def _encode(self, ids, lengths):
        _bytes(ids, lengths, MAXIMUM_INPUT_BYTES)
        _require(all(p.device.type == 'cpu' and p.dtype == torch.float32 and bool(torch.isfinite(p).all())
            for p in self.parameters()), 'finite CPU FP32 owned parameters required')
        packed = nn.utils.rnn.pack_padded_sequence(self.input_embedding(ids), lengths,
            batch_first=True, enforce_sorted=False)
        values, hidden = self.encoder(packed)
        states, _ = nn.utils.rnn.pad_packed_sequence(values, batch_first=True, total_length=ids.shape[1])
        final = torch.cat((hidden[-2], hidden[-1]), -1)
        states = torch.cat((states, final[:, None]), 1)
        # Each ragged row has its own actual terminal state at its length.
        states = states.scatter(1, lengths[:, None, None].expand(-1, 1, 256), final[:, None])
        return states, torch.tanh(self.initial_prediction(final))[None]

    def forward(self, input_byte_ids, input_lengths, target_byte_ids, target_lengths):
        _bytes(target_byte_ids, target_lengths, MAXIMUM_OUTPUT_BYTES, symbols=True)
        _require(len(input_byte_ids) == len(target_byte_ids), 'input/target batch sizes differ')
        states, initial = self._encode(input_byte_ids, input_lengths)
        bos = torch.full((len(target_byte_ids), 1), BOS_ID, dtype=torch.int64)
        predictions, _ = self.prediction(self.prediction_embedding(torch.cat((bos, target_byte_ids), 1)), initial)
        # Exact real-algebra factoring: project onto units/prefixes once; only
        # addition/tanh and the five-output map occur at each pair of positions.
        source_join = F.linear(states, self.action_join.weight[:, :256])
        prediction_join = F.linear(predictions, self.action_join.weight[:, 256:], self.action_join.bias)
        action_logits = self.action_output(torch.tanh(source_join[:, :, None] + prediction_join[:, None]))
        byte_logits = self.source_byte(states)[:, :, None] + self.prediction_byte(predictions)[:, None]
        b, ni, mj = action_logits.shape[:3]
        prefix_values = []
        for row, length in zip(target_byte_ids.tolist(), target_lengths.tolist(), strict=True):
            raw = bytes(row[:length])
            prefix_values.extend(raw[:min(j, length)] for j in range(mj))
        byte_allowed, complete, legal_stop = _prefix_masks(prefix_values)
        byte_allowed = byte_allowed.reshape(b, 1, mj, 256).expand(-1, ni, -1, -1)
        complete, legal_stop = complete.reshape(b, 1, mj), legal_stop.reshape(b, 1, mj)
        cursors = torch.arange(ni)[None, :, None].expand(b, -1, mj)
        current = torch.cat((input_byte_ids, torch.full((b, 1), PAD_BYTE_ID, dtype=torch.int64)), 1)[:, :, None].expand(-1, -1, mj)
        groups = _groups(byte_allowed, complete, legal_stop, current, cursors, input_lengths[:, None, None])
        node_mask = (cursors <= input_lengths[:, None, None]) & (torch.arange(mj)[None, None] <= target_lengths[:, None, None])
        groups = groups & node_mask[..., None]
        action = _masked_log_softmax(action_logits, groups)
        byte = _masked_log_softmax(byte_logits, byte_allowed & node_mask[..., None])
        return EditLatticeLogProbabilities(action, byte, input_byte_ids, input_lengths,
            target_byte_ids, target_lengths, node_mask)

    @torch.no_grad()
    def greedy_decode(self, input_byte_ids, input_lengths):
        _require(not self.training, 'prototype decode requires eval mode')
        states, hidden = self._encode(input_byte_ids, input_lengths)
        bos = torch.full((len(input_byte_ids), 1), BOS_ID, dtype=torch.int64)
        _, hidden = self.prediction(self.prediction_embedding(bos), hidden)
        cursors = torch.zeros(len(input_byte_ids), dtype=torch.int64)
        finished = torch.zeros(len(input_byte_ids), dtype=torch.bool)
        outputs = [b'' for _ in input_byte_ids]
        rows = torch.arange(len(input_byte_ids))
        padded = torch.cat((input_byte_ids, torch.full((len(rows), 1), PAD_BYTE_ID, dtype=torch.int64)), 1)
        for _ in range(int(input_lengths.max()) + MAXIMUM_OUTPUT_BYTES + 1):
            source = states[rows, cursors]; pred = hidden.squeeze(0)
            action_logits = self.action_output(torch.tanh(self.action_join(torch.cat((source, pred), -1))))
            byte_logits = self.source_byte(source) + self.prediction_byte(pred)
            allowed, complete, legal_stop = _prefix_masks(outputs)
            current = padded[rows, cursors]
            groups = _groups(allowed, complete, legal_stop, current, cursors, input_lengths)
            action = _masked_log_softmax(action_logits, groups)
            byte = _masked_log_softmax(byte_logits, allowed)
            # COPY and SUBSTITUTE are identical next states when the byte is the
            # current raw byte. Sum them before fixed greedy transition choice.
            consume_scores = action[:, SUBSTITUTE, None] + byte
            copy_scores = torch.full_like(consume_scores, -torch.inf).scatter(1,
                current.clamp(0, 255)[:, None], action[:, COPY, None])
            consume_scores = _safe_logsumexp(torch.stack((consume_scores, copy_scores), -1))
            consume_value, consume_byte = consume_scores.max(-1)
            insert_value, insert_byte = (action[:, INSERT, None] + byte).max(-1)
            choices = torch.stack((consume_value, insert_value, action[:, DELETE], action[:, STOP]), -1).argmax(-1)
            emitted = ((choices == 0) | (choices == 1)) & ~finished
            selected = torch.where(choices == 0, consume_byte, insert_byte)
            for row in range(len(outputs)):
                if not bool(finished[row]) and bool(emitted[row]): outputs[row] += bytes((int(selected[row]),))
            cursors += (((choices == 0) | (choices == 2)) & ~finished).long()
            finished |= choices == 3
            _require(bool((cursors <= input_lengths).all()), 'neural edit cursor crossed raw input')
            if bool(finished.all()): break
            _, next_hidden = self.prediction(self.prediction_embedding(selected[:, None]), hidden)
            hidden = torch.where(emitted[None, :, None], next_hidden, hidden)
        _require(bool(finished.all()), 'bounded neural edit transitions failed to terminate')
        result = []
        for value in outputs:
            text = value.decode('utf-8')
            _require(len(value) <= MAXIMUM_OUTPUT_BYTES and valid_symbol(text), 'structural canonical boundary failed')
            result.append(CanonicalDecode(text, None, len(value)))
        return tuple(result)

    def decode_surfaces(self, surfaces):
        ids, lengths = utf8_byte_batch(surfaces, MAXIMUM_INPUT_BYTES)
        return self.greedy_decode(ids, lengths)

def _state_digest(state):
    from m1.tensor_state import state_dict_sha256
    return state_dict_sha256(state)
