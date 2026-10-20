"""CAN28 input128 inference, preserving the trained native FP32 arithmetic.

This additive domain accepts at most128 UTF8 mention bytes,64 output bytes,
4096 text bytes and32 Unicode characters of context on either side.
The original input96 canonicalizer and API remain unchanged.
"""
import torch
from torch import nn
from torch.nn import functional as F
from . import canonicalizer_monotonic_edit_v1 as native
from . import context_canonicalizer_edit_v1 as raw
from . import context_canonicalizer_local_joint_score_v1 as original
from .canonicalizer import BOS_ID, CanonicalDecode
from .fillers import PAD_BYTE_ID
MAXIMUM_TEXT_BYTES, CONTEXT_CHARACTERS =4096,32
MAXIMUM_INPUT_BYTES, MAXIMUM_OUTPUT_BYTES, MAXIMUM_CONTEXT_BYTES =128,64,128
PARAMETER_KEYS, PARAMETER_COUNT, TRAINABLE_PARAMETERS =28,857994,164869
BASE_KEYS, TRAINABLE_KEYS, STATE_SHAPES=original.BASE_KEYS,original.TRAINABLE_KEYS,dict(original.STATE_SHAPES)
ContextMention,MentionBatch=raw.ContextMention,raw.MentionBatch
EditLatticeLogProbabilities=native.EditLatticeLogProbabilities
require,_require,_bytes=native._require,native._require,native._bytes
_pack,_context_bytes=raw._pack,raw._context_bytes
_float_geometry=original._float_geometry
validate_state,state_digest=original.validate_state,original.state_digest


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
        require(0 < len(surface.encode('utf8')) <= MAXIMUM_INPUT_BYTES, 'nonempty verbatim mention cap128 required; no truncation')
        surfaces.append(surface)
        left.append(row.text[max(0, row.char_start - CONTEXT_CHARACTERS):row.char_start])
        right.append(row.text[row.char_end:row.char_end + CONTEXT_CHARACTERS])
    with torch.inference_mode(False):
        ids, lengths = _pack(surfaces); left_ids, left_lengths = _pack(left); right_ids, right_lengths = _pack(right)
    batch = MentionBatch(ids, lengths, left_ids, left_lengths, right_ids, right_lengths)
    validate_batch(batch)
    return batch

def validate_batch(batch):
    require(type(batch) is MentionBatch, 'exact typed MentionBatch required; no annotation dictionary ingress')
    native._bytes(batch.input_byte_ids, batch.input_lengths, MAXIMUM_INPUT_BYTES)
    for ids, lengths in ((batch.left_byte_ids, batch.left_lengths), (batch.right_byte_ids, batch.right_lengths)):
        _context_bytes(ids, lengths, len(batch.input_byte_ids))

def validate_model_learning_partition(model, *, learning):
    require(type(learning) is bool and type(model) is LocalJointScoreContextualMonotonicByteEditTransducer128,
        'explicit bool/exact local-score contextual class required')
    require(not any(callable(getattr(type(model), key, None)) for key in vars(model)),
        'immutable native root methods required before state or module enumeration')
    validate_state(dict(model.state_dict())); parameters = dict(model.named_parameters())
    require(set(parameters) == BASE_KEYS | TRAINABLE_KEYS and not list(model.buffers())
        and all(not part.training for part in model.modules()), 'exact28/no buffers/all modules eval required')
    require(all(not parameters[key].requires_grad and parameters[key].grad is None for key in BASE_KEYS),
        'all24 original base tensors always frozen and gradient-free')
    require(all(parameters[key].requires_grad is learning and (parameters[key].grad is None or learning
        and parameters[key].grad.dtype == torch.float32 and parameters[key].grad.device.type == 'cpu'
        and bool(torch.isfinite(parameters[key].grad).all())) for key in TRAINABLE_KEYS),
        'only local scorer4 may learn with finite gradients')

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

class MonotonicByteEditTransducer128(native.MonotonicByteEditTransducer):
    maximum_input_bytes, maximum_output_bytes = MAXIMUM_INPUT_BYTES, MAXIMUM_OUTPUT_BYTES

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

class LocalJointScoreContextualMonotonicByteEditTransducer128(MonotonicByteEditTransducer128):
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
        model = LocalJointScoreContextualMonotonicByteEditTransducer128(); model.load_state_dict(state28, strict=True)
    model = _partition(model, False)
    require(state_digest(dict(model.state_dict())) == state_digest(state28)
        and all(value.untyped_storage().data_ptr() != model.state_dict()[key].untyped_storage().data_ptr()
            for key, value in state28.items()), 'strict ordinary independent frozen28 restore required')
    return model

def validate_frozen(model):
    validate_model_learning_partition(model, learning=False)
    require(model.maximum_input_bytes == 128 and model.maximum_output_bytes == 64,
        'honest versioned input128/output64 owner required')
    require(all(not any(callable(getattr(type(part),key,None)) for key in vars(part))
        and getattr(part,'_compiled_call_impl',None) is None
        and not any(getattr(part,key,None) for key in ('_forward_hooks','_forward_pre_hooks',
            '_backward_hooks','_backward_pre_hooks')) for part in model.modules()),
        'immutable native methods and no inference hooks required')
    return state_digest(dict(model.state_dict()))
