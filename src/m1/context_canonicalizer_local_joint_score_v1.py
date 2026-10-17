"""Portable inference definitions; no research datasets or trained weights included."""

from __future__ import annotations

import torch

from torch import nn

from torch.nn import functional as F

import m1.canonicalizer_monotonic_edit_v1 as native

import m1.context_canonicalizer_edit_v1 as raw

import m1.source_reallocated_canonicalizer_owner_v1 as foundation

from m1.canonicalizer import BOS_ID

from m1.canonicalizer import CanonicalDecode

from m1.fillers import PAD_BYTE_ID

PARAMETER_KEYS, PARAMETER_COUNT, TRAINABLE_PARAMETERS = 28, 857994, 164869

BASE_KEYS = frozenset(foundation.STATE_SHAPES)

TRAINABLE_KEYS = frozenset({'local_hidden.weight', 'local_hidden.bias',
                          'local_output.weight', 'local_output.bias'})

STATE_SHAPES = {**foundation.STATE_SHAPES,
    'local_hidden.weight': (128, 1024), 'local_hidden.bias': (128,),
    'local_output.weight': (261, 128), 'local_output.bias': (261,)}

ContextMention, MentionBatch = raw.ContextMention, raw.MentionBatch

encode_mentions, validate_batch = raw.encode_mentions, raw.validate_batch

EditLatticeLogProbabilities, require = native.EditLatticeLogProbabilities, native._require

MAXIMUM_INPUT_BYTES, MAXIMUM_OUTPUT_BYTES = native.MAXIMUM_INPUT_BYTES, native.MAXIMUM_OUTPUT_BYTES

def validate_state(state):
    require(type(state) is dict and set(state) == set(STATE_SHAPES), 'complete exact local-score28 state required')
    require(all(type(value) is torch.Tensor and tuple(value.shape) == STATE_SHAPES[key]
        and value.dtype == torch.float32 and value.device.type == 'cpu' and value.layout == torch.strided
        and value.is_contiguous() and not value.requires_grad and not value.is_inference()
        and bool(torch.isfinite(value).all()) for key, value in state.items()),
        'ordinary finite contiguous detached CPU FP32 local-score28 shapes required')
    require(len({value.untyped_storage().data_ptr() for value in state.values()}) == PARAMETER_KEYS,
        'independent all28 tensor storage required')

def state_digest(state):
    validate_state(state)
    return native._state_digest(state)

def validate_model_learning_partition(model, *, learning):
    require(type(learning) is bool and type(model) is LocalJointScoreContextualMonotonicByteEditTransducer,
        'explicit bool/exact local-score contextual class required')
    validate_state(dict(model.state_dict())); parameters = dict(model.named_parameters())
    require(set(parameters) == BASE_KEYS | TRAINABLE_KEYS and not list(model.buffers())
        and all(not part.training for part in model.modules()), 'exact28/no buffers/all modules eval required')
    require(all(not parameters[key].requires_grad and parameters[key].grad is None for key in BASE_KEYS),
        'all24 original base tensors always frozen and gradient-free')
    require(all(parameters[key].requires_grad is learning and (parameters[key].grad is None or learning
        and parameters[key].grad.dtype == torch.float32 and parameters[key].grad.device.type == 'cpu'
        and bool(torch.isfinite(parameters[key].grad).all())) for key in TRAINABLE_KEYS),
        'only local scorer4 may learn with finite gradients')

def _float_geometry(value, shape):
    require(type(value) is torch.Tensor and tuple(value.shape) == shape
        and value.dtype == torch.float32 and value.device.type == 'cpu' and value.layout == torch.strided
        and bool(torch.isfinite(value).all()), 'finite CPU FP32 explicit local-score geometry required')

def _score_geometry(source, prefix, left, right, pairwise):
    require(type(pairwise) is bool and type(source) is type(prefix) is torch.Tensor
        and source.ndim == prefix.ndim == (3 if pairwise else 2) and len(source) > 0,
        'explicit bool and matching local source/prefix ranks required')
    count = len(source)
    if pairwise:
        require(0 < source.shape[1] <= MAXIMUM_INPUT_BYTES + 1
            and 0 < prefix.shape[1] <= MAXIMUM_OUTPUT_BYTES + 1, 'bounded complete lattice cursor/prefix axes required')
        _float_geometry(source, (count, source.shape[1], 256))
        _float_geometry(prefix, (count, prefix.shape[1], 256))
    else:
        _float_geometry(source, (count, 256)); _float_geometry(prefix, (count, 256))
    _float_geometry(left, (count, 256)); _float_geometry(right, (count, 256))

class LocalJointScoreContextualMonotonicByteEditTransducer(native.MonotonicByteEditTransducer):
    def __init__(self):
        with torch.inference_mode(False):
            super().__init__()
            self.local_hidden = nn.Linear(1024, 128, dtype=torch.float32, device='cpu')
            self.local_output = nn.Linear(128, 261, dtype=torch.float32, device='cpu')
            with torch.no_grad():
                self.local_output.weight.zero_(); self.local_output.bias.zero_()
        self.eval().requires_grad_(False)
        validate_model_learning_partition(self, learning=False)
        require(sum(value.numel() for value in self.parameters()) == PARAMETER_COUNT
            and sum(value.numel() for key, value in self.named_parameters() if key in TRAINABLE_KEYS) == TRAINABLE_PARAMETERS,
            'declared local-score28/857994 and scorer4/164869 inventory required')

    def encode_batch(self, batch):
        validate_batch(batch)
        validate_model_learning_partition(self, learning=self.local_hidden.weight.requires_grad)
        return super()._encode(batch.input_byte_ids, batch.input_lengths)

    def _context_summary(self, ids, lengths):
        result = torch.zeros(len(ids), 256, dtype=torch.float32, device='cpu')
        selected = (lengths > 0).nonzero().flatten()
        if not len(selected): return result
        packed = nn.utils.rnn.pack_padded_sequence(self.input_embedding(ids.index_select(0, selected)),
            lengths.index_select(0, selected), batch_first=True, enforce_sorted=False)
        _, hidden = self.encoder(packed)
        return result.index_copy(0, selected, torch.cat((hidden[-2], hidden[-1]), -1))

    def _contexts(self, batch):
        return (self._context_summary(batch.left_byte_ids, batch.left_lengths),
                self._context_summary(batch.right_byte_ids, batch.right_lengths),
                (batch.left_lengths > 0) | (batch.right_lengths > 0))

    def _local_residual(self, source, prefix, left, right, *, pairwise):
        """Linear1024 is split before lattice broadcast, not after concat."""
        _score_geometry(source, prefix, left, right, pairwise)
        weight = self.local_hidden.weight
        source_join = F.linear(source, weight[:, :256])
        prefix_join = F.linear(prefix, weight[:, 256:512])
        context_join = F.linear(torch.cat((left, right), -1), weight[:, 512:], self.local_hidden.bias)
        joined = (source_join[:, :, None] + prefix_join[:, None] + context_join[:, None, None]
            if pairwise else source_join + prefix_join + context_join)
        result = self.local_output(torch.tanh(joined))
        require(bool(torch.isfinite(result).all()), 'finite unbounded local action/byte residual required')
        return result[..., :5], result[..., 5:]

    def _raw_logits(self, source, prefix, left, right, active, *, pairwise):
        """One residual kernel; preserve the native arithmetic of each path."""
        _score_geometry(source, prefix, left, right, pairwise)
        require(type(pairwise) is bool and type(active) is torch.Tensor and active.dtype == torch.bool
            and active.device.type == 'cpu' and active.shape == (len(source),), 'honest raw nonempty-context mask required')
        if pairwise:
            source_join = F.linear(source, self.action_join.weight[:, :256])
            prefix_join = F.linear(prefix, self.action_join.weight[:, 256:], self.action_join.bias)
            action = self.action_output(torch.tanh(source_join[:, :, None] + prefix_join[:, None]))
            byte = self.source_byte(source)[:, :, None] + self.prediction_byte(prefix)[:, None]
        else:
            action = self.action_output(torch.tanh(self.action_join(torch.cat((source, prefix), -1))))
            byte = self.source_byte(source) + self.prediction_byte(prefix)
        if not bool(active.any()): return action, byte
        delta_action, delta_byte = self._local_residual(source, prefix, left, right, pairwise=pairwise)
        mask = active[:, None, None, None] if pairwise else active[:, None]
        return torch.where(mask, action + delta_action, action), torch.where(mask, byte + delta_byte, byte)

    def forward(self, batch, target_ids, target_lengths):
        validate_batch(batch)
        validate_model_learning_partition(self, learning=self.local_hidden.weight.requires_grad)
        native._bytes(target_ids, target_lengths, MAXIMUM_OUTPUT_BYTES, symbols=True)
        require(len(batch.input_byte_ids) == len(target_ids), 'input/target batch sizes differ')
        if not bool(((batch.left_lengths > 0) | (batch.right_lengths > 0)).any()):
            return super().forward(batch.input_byte_ids, batch.input_lengths, target_ids, target_lengths)
        states, initial = super()._encode(batch.input_byte_ids, batch.input_lengths)
        left, right, active = self._contexts(batch)
        bos = torch.full((len(target_ids), 1), BOS_ID, dtype=torch.int64)
        predictions, _ = self.prediction(self.prediction_embedding(torch.cat((bos, target_ids), 1)), initial)
        action_logits, byte_logits = self._raw_logits(states, predictions, left, right, active, pairwise=True)
        b, ni, mj = action_logits.shape[:3]; prefixes = []
        for row, length in zip(target_ids.tolist(), target_lengths.tolist(), strict=True):
            value = bytes(row[:length]); prefixes.extend(value[:min(j, length)] for j in range(mj))
        allowed, complete, legal_stop = native._prefix_masks(prefixes)
        allowed = allowed.reshape(b, 1, mj, 256).expand(-1, ni, -1, -1)
        complete, legal_stop = complete.reshape(b, 1, mj), legal_stop.reshape(b, 1, mj)
        cursors = torch.arange(ni)[None, :, None].expand(b, -1, mj)
        current = torch.cat((batch.input_byte_ids, torch.full((b, 1), PAD_BYTE_ID, dtype=torch.int64)), 1)[:, :, None].expand(-1, -1, mj)
        groups = native._groups(allowed, complete, legal_stop, current, cursors, batch.input_lengths[:, None, None])
        nodes = (cursors <= batch.input_lengths[:, None, None]) & (torch.arange(mj)[None, None] <= target_lengths[:, None, None])
        action = native._masked_log_softmax(action_logits, groups & nodes[..., None])
        byte = native._masked_log_softmax(byte_logits, allowed & nodes[..., None])
        return EditLatticeLogProbabilities(action, byte, batch.input_byte_ids, batch.input_lengths, target_ids, target_lengths, nodes)

    @torch.no_grad()
    def greedy_decode(self, batch):
        validate_batch(batch)
        validate_model_learning_partition(self, learning=self.local_hidden.weight.requires_grad)
        if not bool(((batch.left_lengths > 0) | (batch.right_lengths > 0)).any()):
            return super().greedy_decode(batch.input_byte_ids, batch.input_lengths)
        states, hidden = super()._encode(batch.input_byte_ids, batch.input_lengths)
        left, right, active = self._contexts(batch)
        bos = torch.full((len(batch.input_byte_ids), 1), BOS_ID, dtype=torch.int64)
        _, hidden = self.prediction(self.prediction_embedding(bos), hidden)
        cursors = torch.zeros(len(batch.input_byte_ids), dtype=torch.int64)
        finished = torch.zeros(len(batch.input_byte_ids), dtype=torch.bool)
        outputs = [b'' for _ in batch.input_byte_ids]; rows = torch.arange(len(batch.input_byte_ids))
        padded = torch.cat((batch.input_byte_ids, torch.full((len(rows), 1), PAD_BYTE_ID, dtype=torch.int64)), 1)
        for _ in range(int(batch.input_lengths.max()) + MAXIMUM_OUTPUT_BYTES + 1):
            source, prefix = states[rows, cursors], hidden.squeeze(0)
            logits, byte_logits = self._raw_logits(source, prefix, left, right, active, pairwise=False)
            allowed, complete, legal_stop = native._prefix_masks(outputs); current = padded[rows, cursors]
            groups = native._groups(allowed, complete, legal_stop, current, cursors, batch.input_lengths)
            action, byte = native._masked_log_softmax(logits, groups), native._masked_log_softmax(byte_logits, allowed)
            consume = action[:, native.SUBSTITUTE, None] + byte
            copy = torch.full_like(consume, -torch.inf).scatter(1, current.clamp(0, 255)[:, None], action[:, native.COPY, None])
            consume_value, consume_byte = native._safe_logsumexp(torch.stack((consume, copy), -1)).max(-1)
            insert_value, insert_byte = (action[:, native.INSERT, None] + byte).max(-1)
            choices = torch.stack((consume_value, insert_value, action[:, native.DELETE], action[:, native.STOP]), -1).argmax(-1)
            emitted = ((choices == 0) | (choices == 1)) & ~finished
            selected = torch.where(choices == 0, consume_byte, insert_byte)
            for row in range(len(outputs)):
                if not bool(finished[row]) and bool(emitted[row]): outputs[row] += bytes((int(selected[row]),))
            cursors += (((choices == 0) | (choices == 2)) & ~finished).long(); finished |= choices == 3
            require(bool((cursors <= batch.input_lengths).all()), 'neural edit cursor crossed raw mention')
            if bool(finished.all()): break
            _, next_hidden = self.prediction(self.prediction_embedding(selected[:, None]), hidden)
            hidden = torch.where(emitted[None, :, None], next_hidden, hidden)
        require(bool(finished.all()), 'bounded neural edit transitions failed to terminate')
        result = []
        for value in outputs:
            text = value.decode('utf8')
            require(len(value) <= MAXIMUM_OUTPUT_BYTES and native.valid_symbol(text), 'structural canonical boundary failed')
            result.append(CanonicalDecode(text, None, len(value)))
        return tuple(result)

    def decode_mentions(self, mentions):
        return self.greedy_decode(encode_mentions(mentions))

    def decode_surfaces(self, surfaces):
        require(type(surfaces) in (tuple, list) and all(type(value) is str for value in surfaces), 'raw surface sequence required')
        return self.decode_mentions(tuple(ContextMention(value, 0, len(value)) for value in surfaces))

def _partition(model, learning):
    require(type(learning) is bool, 'explicit bool learning partition required')
    model.eval().requires_grad_(False)
    for key, parameter in model.named_parameters():
        parameter.grad = None; parameter.requires_grad_(learning and key in TRAINABLE_KEYS)
    validate_model_learning_partition(model, learning=learning)
    return model

def model_from_state(state28):
    validate_state(state28)
    with torch.random.fork_rng(devices=[]), torch.inference_mode(False):
        model = LocalJointScoreContextualMonotonicByteEditTransducer(); model.load_state_dict(state28, strict=True)
    model = _partition(model, False)
    require(state_digest(dict(model.state_dict())) == state_digest(state28)
        and all(value.untyped_storage().data_ptr() != model.state_dict()[key].untyped_storage().data_ptr()
            for key, value in state28.items()), 'strict ordinary independent frozen28 restore required')
    return model
