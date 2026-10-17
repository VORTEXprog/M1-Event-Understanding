"""Portable inference definitions; no research datasets or trained weights included."""

from __future__ import annotations

from .heads import ROLE_ORDER

from dataclasses import dataclass

import math

from pathlib import Path

import torch

from torch import Tensor

from torch import nn

from torch.nn import functional as F

import m1.frozen_pointer_joint_role_context_argument_v1 as am106

from m1.argument_expert import ArgumentExpertLogits

from m1.heads import _event_inputs

from m1.heads import _validate_context

from m1.joint_argument_span_score_v1 import enumerate_unit_pairs

from m1.joint_argument_span_score_v2 import JointArgumentSpanScorer

from m1.normalized_presence import NormalizedContextAttention

VERSION = 'M1_STRUCTURED_JOINT_ROLE_ARGUMENT_v1'

SOURCE = Path('src/m1/structured_joint_role_argument_v1.py')

OBJECTIVE = 'UNWEIGHTED_NORMALIZED_EIGHT_ROLE_COMPLETE_PAIRED_GOLD_MARGINAL_NLL'

DECODER = 'NORMALIZED_EIGHT_ROLE_BEAM8_LEXICOGRAPHIC_APPROXIMATE_JOINT_MAP'

ROLE_COUNT, SCORE_DIM, BEAM_WIDTH, MAX_GOLD_ASSIGNMENTS = 8, 32, 8, 4

SCORER_KEYS = ('span_scorer.pair_encoder.0.weight', 'span_scorer.pair_encoder.0.bias',
    'span_scorer.pair_encoder.2.weight', 'span_scorer.pair_encoder.2.bias',
    'span_scorer.query_projection.weight', 'span_scorer.query_projection.bias',
    'span_scorer.null_scorer.weight', 'span_scorer.null_scorer.bias')

EXTRA_KEYS = (*SCORER_KEYS, 'null_embedding', 'factor_left.weight',
              'factor_right.weight', 'role_pair_gate')

FROZEN_KEYS = tuple('base.' + name for name in am106.FROZEN_KEYS)

TRAINABLE_KEYS = (*('base.' + name for name in am106.NEW_KEYS), *EXTRA_KEYS)

PAIR_ROLES = tuple((s, r) for s in range(ROLE_COUNT) for r in range(s + 1, ROLE_COUNT))

def _sha(value: object) -> bool:
    return type(value) is str and len(value) == 64 and value != '0' * 64 and not set(value) - set('0123456789abcdef')

def _shape_map(encoder: int, hidden: int) -> dict[str, tuple[int, ...]]:
    result = {'base.' + name: shape for name, shape in am106._shapes(encoder, hidden).items()}
    result.update({
        SCORER_KEYS[0]: (SCORE_DIM, 3 * hidden), SCORER_KEYS[1]: (SCORE_DIM,),
        SCORER_KEYS[2]: (SCORE_DIM, SCORE_DIM), SCORER_KEYS[3]: (SCORE_DIM,),
        SCORER_KEYS[4]: (SCORE_DIM, hidden), SCORER_KEYS[5]: (SCORE_DIM,),
        SCORER_KEYS[6]: (1, hidden), SCORER_KEYS[7]: (1,),
        'null_embedding': (SCORE_DIM,), 'factor_left.weight': (SCORE_DIM, SCORE_DIM),
        'factor_right.weight': (SCORE_DIM, SCORE_DIM), 'role_pair_gate': (len(PAIR_ROLES),),
    })
    return result

def _state_check(state: dict[str, Tensor], encoder: int, hidden: int) -> None:
    shapes = _shape_map(encoder, hidden)
    if (type(state) is not dict or set(state) != set(shapes)
            or any(not isinstance(value, Tensor) or tuple(value.shape) != shapes[name]
                or value.device.type != 'cpu' or value.dtype != torch.float32
                or value.requires_grad or not bool(torch.isfinite(value).all())
                for name, value in state.items())):
        raise ValueError('exact65 detached finite CPU FP32 state required')
    if encoder == 2048 and hidden == 128:
        if len(shapes) != 65 or sum(math.prod(shape) for shape in shapes.values()) != 2432929:
            raise ValueError('production structured owner geometry differs')

@dataclass(frozen=True)
class StructuredArgumentPotentials:
    unary_logits: Tensor                  # FP32[B,E,8,P+1], NULL last
    left_factors: Tensor                  # FP32[B,E,8,P+1,32]
    right_factors: Tensor                 # FP32[B,E,8,P+1,32]
    role_pair_gate: Tensor                # FP32[28], s<r order
    pair_indices: Tensor                  # int64[B,P,2], original unit coordinates
    pair_mask: Tensor                     # bool[B,P]
    event_mask: Tensor                    # bool[B,E]
    token_mask: Tensor                    # bool[B,T]
    context_attention: Tensor             # FP32[B,E,T]
    pair_vectors: Tensor | None = None     # FP32[B,E,P,32], shared across events
    query_vectors: Tensor | None = None    # FP32[B,E,8,32], before factor projections

    @property
    def null_index(self) -> int:
        return self.unary_logits.shape[-1] - 1

    def validate(self) -> None:
        unary = self.unary_logits
        if unary.ndim != 4 or unary.dtype != torch.float32 or unary.shape[2] != ROLE_COUNT:
            raise ValueError('structured unary must be FP32[B,E,8,P+1]')
        b, e, _, c = unary.shape
        if (self.left_factors.shape != (b, e, ROLE_COUNT, c, SCORE_DIM)
                or self.right_factors.shape != self.left_factors.shape
                or self.role_pair_gate.shape != (len(PAIR_ROLES),)
                or self.pair_indices.shape != (b, c - 1, 2)
                or self.pair_indices.dtype != torch.int64
                or self.pair_mask.shape != (b, c - 1) or self.pair_mask.dtype != torch.bool
                or self.event_mask.shape != (b, e) or self.event_mask.dtype != torch.bool
                or self.token_mask.ndim != 2 or self.token_mask.shape[0] != b
                or self.token_mask.dtype != torch.bool
                or self.context_attention.shape != (b, e, self.token_mask.shape[1])):
            raise ValueError('structured pair/factor/mask geometry differs')
        tensors = (self.left_factors, self.right_factors, self.role_pair_gate,
                   self.pair_indices, self.pair_mask, self.event_mask,
                   self.token_mask, self.context_attention)
        if any(value.device != unary.device for value in tensors):
            raise ValueError('structured tensors must share a device')
        if any(value.dtype != torch.float32 or not bool(torch.isfinite(value).all())
               for value in (unary, self.left_factors, self.right_factors,
                             self.role_pair_gate, self.context_attention)):
            raise ValueError('structured scores and factors must be finite FP32')
        if (self.pair_vectors is None) != (self.query_vectors is None):
            raise ValueError('structured saved-feature ports must be paired')
        if self.pair_vectors is not None:
            if (self.pair_vectors.shape != (b, e, c - 1, SCORE_DIM)
                    or self.query_vectors.shape != (b, e, ROLE_COUNT, SCORE_DIM)
                    or any(value.device != unary.device or value.dtype != torch.float32
                           or not bool(torch.isfinite(value).all())
                           for value in (self.pair_vectors, self.query_vectors))):
                raise ValueError('structured saved-feature geometry differs')
        pairs, mask = enumerate_unit_pairs(self.token_mask)
        if not torch.equal(pairs, self.pair_indices) or not torch.equal(mask, self.pair_mask):
            raise ValueError('every structural pair in original coordinates required')
        active = self.event_mask[:, :, None, None]
        valid = active & torch.cat((self.pair_mask, torch.ones((b, 1), dtype=torch.bool,
            device=unary.device)), -1)[:, None, None, :]
        if (bool(unary.masked_select(~valid.expand_as(unary)).count_nonzero())
                or bool(self.left_factors.masked_select(~valid[..., None].expand_as(self.left_factors)).count_nonzero())
                or bool(self.right_factors.masked_select(~valid[..., None].expand_as(self.right_factors)).count_nonzero())
                or bool(self.context_attention.masked_select(~(self.event_mask[:, :, None]
                    & self.token_mask[:, None, :])).count_nonzero())):
            raise ValueError('inactive/padded structured scores must be exact zero')

class StructuredJointRoleArgumentExpert(nn.Module):
    architecture = VERSION

    def __init__(self, base_state: dict[str, Tensor], *, expected_frozen_state_sha256: str,
                 expected_native_state_sha256: str, expected_native_generic_sha256: str,
                 encoder_dimension: int = 2048, hidden_dimension: int = 128):
        super().__init__()
        self.encoder_dimension, self.hidden_dimension = encoder_dimension, hidden_dimension
        self.base = am106.FrozenPointerJointRoleContextArgumentExpert(base_state,
            expected_frozen_state_sha256=expected_frozen_state_sha256,
            expected_native_state_sha256=expected_native_state_sha256,
            expected_native_generic_sha256=expected_native_generic_sha256,
            encoder_dimension=encoder_dimension, hidden_dimension=hidden_dimension)
        self.span_scorer = JointArgumentSpanScorer(hidden_dimension, SCORE_DIM, SCORE_DIM)
        self.null_embedding = nn.Parameter(torch.randn(SCORE_DIM, dtype=torch.float32) / math.sqrt(SCORE_DIM))
        self.factor_left = nn.Linear(SCORE_DIM, SCORE_DIM, bias=False, device='cpu', dtype=torch.float32)
        self.factor_right = nn.Linear(SCORE_DIM, SCORE_DIM, bias=False, device='cpu', dtype=torch.float32)
        self.role_pair_gate = nn.Parameter(torch.ones(len(PAIR_ROLES), dtype=torch.float32))
        # Final frozen31 copy follows every appended draw, exactly as plan114 requires.
        with torch.no_grad():
            current = self.base.state_dict()
            for name, value in base_state.items():
                current[name].copy_(value)
        self.eval().requires_grad_(False)
        self._validate_topology()
        assert_frozen_unchanged(self)

    def _validate_topology(self) -> None:
        _state_check(dict(self.state_dict()), self.encoder_dimension, self.hidden_dimension)
        params = dict(self.named_parameters())
        if (set(params) != set(_shape_map(self.encoder_dimension, self.hidden_dimension))
                or len({p.untyped_storage().data_ptr() for p in params.values()}) != len(params)):
            raise ValueError('exact65 independently owned parameter storages required')

    def _base_components(self, inputs):
        """Immutable Am106 default path, evaluated once; return owned hidden ports."""
        base = self.base
        if any(p.device.type != 'cpu' or p.dtype != torch.float32 for p in self.parameters()):
            raise ValueError('structured owner requires CPU FP32')
        keys = base.encode_keys(inputs)
        context, attention = NormalizedContextAttention.forward(base.context, inputs)
        _validate_context(inputs, context, base.hidden_dimension)
        pointer = base.pointer
        event = pointer.event_query(_event_inputs(inputs, context))
        native_query = torch.tanh(F.layer_norm(event.unsqueeze(2) + pointer.role_queries[None, None],
            (base.hidden_dimension,), eps=1e-5))
        z = base.presence_event(torch.cat((inputs.event_states, context), -1)).unsqueeze(2) + base.presence_roles[None, None]
        pooled, _ = base.presence_context(inputs, keys, z)
        combined = torch.cat((native_query, pooled), -1)
        start_query, end_query = base.fresh_start_query(combined), base.fresh_end_query(combined)
        states = base.encode_presence(inputs)
        ordered = base.ordered_presence_context(inputs, states, z)[0]
        ordered = base.joint_role_coordination(z, ordered, inputs.event_mask)
        start_query = start_query + base.pointer_start_query_residual(ordered)
        end_query = end_query + base.pointer_end_query_residual(ordered)
        start = torch.einsum('berh,bth->bert', start_query, pointer.start_key(keys))
        end = torch.einsum('berh,bth->bert', end_query, pointer.end_key(keys))
        start = start + torch.einsum('berh,bth->bert', start_query, base.pointer_start_key_residual(states))
        end = end + torch.einsum('berh,bth->bert', end_query, base.pointer_end_key_residual(states))
        valid = inputs.event_mask[:, :, None, None] & inputs.token_mask[:, None, None, :]
        floor = torch.finfo(start.dtype).min
        start, end = (value.div(base.hidden_dimension ** .5).masked_fill(~valid, floor) for value in (start, end))
        fused = ordered + F.gelu(base.ordered_presence_inner(torch.cat((z, ordered), -1)))
        correction = base.ordered_presence_residual(fused).masked_fill(~inputs.event_mask[:, :, None, None], 0)
        presence = base.presence_output(F.gelu(z) + correction).masked_fill(~inputs.event_mask[:, :, None, None], 0)
        return ArgumentExpertLogits(presence, start, end, attention, inputs.event_mask), states, ordered, native_query

    def forward(self, inputs) -> StructuredArgumentPotentials:
        old, states, ordered, native_query = self._base_components(inputs)
        b, t = inputs.token_states.shape[:2]
        e = inputs.event_states.shape[1]
        query = torch.tanh(F.layer_norm(native_query + ordered, (self.hidden_dimension,), eps=1e-5))
        pairs, pair_mask = enumerate_unit_pairs(inputs.token_mask)
        p = pairs.shape[1]
        repeated = states[:, None].expand(-1, e, -1, -1)
        hidden = torch.tanh(self.span_scorer._pair_preactivation(repeated, inputs.token_mask, pairs))
        vectors = self.span_scorer.pair_encoder[2](hidden)
        query_vectors = self.span_scorer.query_projection(query)
        residual = torch.einsum('bepk,berk->berp', vectors, query_vectors) / math.sqrt(SCORE_DIM)
        starts, ends = pairs[..., 0].clamp_min(0), pairs[..., 1].clamp_min(0)
        start_score = old.argument_start_logits.gather(-1, starts[:, None, None, :].expand(b, e, ROLE_COUNT, p))
        end_score = old.argument_end_logits.gather(-1, ends[:, None, None, :].expand(b, e, ROLE_COUNT, p))
        structural = inputs.event_mask[:, :, None, None] & pair_mask[:, None, None, :]
        start_score = torch.where(structural, start_score, torch.zeros_like(start_score))
        end_score = torch.where(structural, end_score, torch.zeros_like(end_score))
        presence_odds = old.presence_logits[..., 1] - old.presence_logits[..., 0]
        pair_unary = presence_odds[..., None] + start_score + end_score + residual
        pair_unary = torch.where(structural, pair_unary, torch.zeros_like(pair_unary))
        null_unary = self.span_scorer.null_scorer(query).masked_fill(~inputs.event_mask[:, :, None, None], 0)
        unary = torch.cat((pair_unary, null_unary), -1)
        pair_features = torch.tanh(vectors[:, :, None, :, :] + query_vectors[:, :, :, None, :])
        null_features = torch.tanh(self.null_embedding[None, None, None, None, :] + query_vectors[:, :, :, None, :])
        features = torch.cat((pair_features, null_features), 3)
        full_valid = torch.cat((pair_mask, torch.ones((b, 1), dtype=torch.bool, device=pair_mask.device)), 1)
        features = features.masked_fill(~(inputs.event_mask[:, :, None, None] & full_valid[:, None, None, :])[..., None], 0)
        left = self.factor_left(features) / math.sqrt(SCORE_DIM)
        right = self.factor_right(features)
        output = StructuredArgumentPotentials(unary, left, right, self.role_pair_gate,
            pairs, pair_mask, inputs.event_mask.clone(), inputs.token_mask.clone(), old.context_attention,
            vectors, query_vectors)
        output.validate()
        return output

    def enable_training(self):
        self.base.enable_training()
        self.train()
        for name, parameter in self.named_parameters():
            parameter.grad = None
            parameter.requires_grad_(name in TRAINABLE_KEYS)
        assert_frozen_unchanged(self)
        return self

def assert_frozen_unchanged(net: StructuredJointRoleArgumentExpert) -> str:
    if type(net) is not StructuredJointRoleArgumentExpert:
        raise ValueError('exact structured owner required')
    return am106.assert_frozen_unchanged(net.base)

@torch.no_grad()
def decode_beam8(output: StructuredArgumentPotentials) -> Tensor:
    """Fixed-width approximate whole-structure MAP; all structural candidates expand."""
    output.validate()
    unary, left, right, gates = output.unary_logits, output.left_factors, output.right_factors, output.role_pair_gate
    b, e, roles, c = unary.shape
    t = output.token_mask.shape[1]
    null = c - 1
    decisions = torch.full((b, e, roles, 2), t, dtype=torch.int64, device=unary.device)
    valid = torch.cat((output.pair_mask, torch.ones((b, 1), dtype=torch.bool,
        device=unary.device)), -1)
    for row in range(b):
        width = min(BEAM_WIDTH, int(valid[row].sum()))
        for event in range(e):
            if not output.event_mask[row, event]:
                continue
            beams: list[tuple[float, tuple[int, ...]]] = [(0.0, ())]
            for role in range(roles):
                expanded: list[tuple[float, tuple[int, ...]]] = []
                for prior_score, prefix in beams:
                    scores = unary[row, event, role]
                    if prefix:
                        context = sum(gates[s * (2 * ROLE_COUNT - s - 1) // 2 + role - s - 1]
                            * left[row, event, s, choice] for s, choice in enumerate(prefix))
                        scores = scores + right[row, event, role] @ context
                    log_probs = F.log_softmax(scores.masked_fill(~valid[row], -torch.inf), -1)
                    # Stable sort gives original pair-order then NULL ties.
                    indices = torch.argsort(log_probs, descending=True, stable=True)[:width]
                    expanded.extend((prior_score + float(log_probs[index]), (*prefix, int(index)))
                                    for index in indices)
                beams = sorted(expanded, key=lambda item: (-item[0], item[1]))[:BEAM_WIDTH]
            for role, choice in enumerate(beams[0][1]):
                if choice != null:
                    decisions[row, event, role] = output.pair_indices[row, choice]
    return decisions
