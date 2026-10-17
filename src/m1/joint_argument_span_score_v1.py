"""Portable inference definitions; no research datasets or trained weights included."""

from __future__ import annotations

from dataclasses import dataclass

from typing import Mapping

import torch

from torch import Tensor

from torch import nn

VERSION = "M1_JOINT_ARGUMENT_SPAN_SCORE_v1"

ROLE_COUNT = 8

def enumerate_unit_pairs(token_attention_mask: Tensor) -> tuple[Tensor, Tensor]:
    """All valid-rank start<=end pairs in original unit coordinates, no width cap."""
    if token_attention_mask.dtype != torch.bool or token_attention_mask.ndim != 2:
        raise ValueError("token_attention_mask must be bool[B,T]")
    valid_units = [row.nonzero(as_tuple=False).flatten() for row in token_attention_mask]
    counts = [len(units) * (len(units) + 1) // 2 for units in valid_units]
    batch, pair_count = len(valid_units), max(counts, default=0)
    pairs = torch.full((batch, pair_count, 2), -1, dtype=torch.long, device=token_attention_mask.device)
    pair_mask = torch.zeros((batch, pair_count), dtype=torch.bool, device=token_attention_mask.device)
    for index, units in enumerate(valid_units):
        ranks = torch.triu_indices(len(units), len(units), device=units.device)
        count = counts[index]
        pairs[index, :count] = units[ranks].transpose(0, 1)
        pair_mask[index, :count] = True
    return pairs, pair_mask

@dataclass(frozen=True)
class JointArgumentSpanLogits:
    """Joint score ABI; last score is NULL and inactive/padded scores are zero."""

    joint_logits: Tensor  # FP32[B,E,8,P+1]
    pair_indices: Tensor  # int64[B,P,2], original unit coordinates; padding -1
    pair_mask: Tensor  # bool[B,P]
    event_mask: Tensor  # bool[B,E]
    token_attention_mask: Tensor  # bool[B,T]

    @property
    def null_index(self) -> int:
        return self.joint_logits.shape[-1] - 1

    def validate(self) -> None:
        scores = self.joint_logits
        if scores.ndim != 4 or scores.dtype != torch.float32 or scores.shape[2] != ROLE_COUNT or scores.shape[-1] < 1:
            raise ValueError("joint_logits must be FP32[B,E,8,P+1]")
        batch, events, _, choices = scores.shape
        tensors = (self.pair_indices, self.pair_mask, self.event_mask, self.token_attention_mask)
        if any(tensor.device != scores.device for tensor in tensors):
            raise ValueError("joint ABI tensors must share a device")
        if self.pair_indices.dtype != torch.long or self.pair_indices.shape != (batch, choices - 1, 2):
            raise ValueError("pair_indices must be int64[B,P,2]")
        if self.pair_mask.dtype != torch.bool or self.pair_mask.shape != (batch, choices - 1):
            raise ValueError("pair_mask must be bool[B,P]")
        if self.event_mask.dtype != torch.bool or self.event_mask.shape != (batch, events):
            raise ValueError("event_mask must be bool[B,E]")
        if self.token_attention_mask.ndim != 2 or self.token_attention_mask.shape[0] != batch:
            raise ValueError("token_attention_mask must be bool[B,T]")
        expected_pairs, expected_mask = enumerate_unit_pairs(self.token_attention_mask)
        if not torch.equal(self.pair_indices, expected_pairs) or not torch.equal(self.pair_mask, expected_mask):
            raise ValueError("joint ABI must retain all legal original-coordinate pairs")
        if not bool(torch.isfinite(scores).all()):
            raise ValueError("joint ABI scores must be finite")
        active = self.event_mask[:, :, None, None]
        choices_valid = torch.cat((self.pair_mask, torch.ones((batch, 1), dtype=torch.bool, device=scores.device)), dim=-1)
        structural = active & choices_valid[:, None, None, :]
        if bool((scores.masked_select(~structural.expand_as(scores)) != 0).any()):
            raise ValueError("inactive events and padded pair scores must be exact zeros")

def select_joint_argument_spans(logits: JointArgumentSpanLogits) -> Tensor:
    """Joint MAP original unit endpoints [B,E,8,2]; NULL/inactive -> (-1,-1)."""
    logits.validate()
    scores = logits.joint_logits
    batch, events, roles, _ = scores.shape
    choices_valid = torch.cat((logits.pair_mask, torch.ones((batch, 1), dtype=torch.bool, device=scores.device)), dim=-1)
    choice = scores.masked_fill(~choices_valid[:, None, None, :], -torch.inf).argmax(dim=-1)
    null_pair = torch.full((batch, 1, 2), -1, dtype=torch.long, device=scores.device)
    coordinates = torch.cat((logits.pair_indices, null_pair), dim=1)
    selected = coordinates[:, None, None, :, :].expand(batch, events, roles, -1, 2).gather(
        3, choice[:, :, :, None, None].expand(-1, -1, -1, 1, 2)
    ).squeeze(3)
    return torch.where(logits.event_mask[:, :, None, None], selected, torch.full_like(selected, -1))

class JointArgumentSpanScorer(nn.Module):
    """Shared complete-pair FFN, projected role queries and learned explicit NULL.

    Inputs are already-owned event-conditioned states[B,E,T,H] and role
    queries[B,E,8,H]. Shared BTH states must be expanded explicitly by their
    owner. Pair encoding is shared across all roles, avoiding per-role FFNs.
    """

    def __init__(self, state_dim: int, hidden_dim: int, score_dim: int) -> None:
        super().__init__()
        if any(type(value) is not int or value <= 0 for value in (state_dim, hidden_dim, score_dim)):
            raise ValueError("state/hidden/score dimensions must be positive integers")
        self.state_dim, self.hidden_dim, self.score_dim = state_dim, hidden_dim, score_dim
        self.pair_encoder = nn.Sequential(nn.Linear(3 * state_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, score_dim))
        self.query_projection = nn.Linear(state_dim, score_dim)
        self.null_scorer = nn.Linear(state_dim, 1)
        self._scale = score_dim ** -0.5

    def metadata(self) -> dict[str, object]:
        return {"version": VERSION, "state_dim": self.state_dim, "hidden_dim": self.hidden_dim,
                "score_dim": self.score_dim, "role_count": ROLE_COUNT, "token_layout": "BETH",
                "span_evidence": "start_end_valid_unit_mean", "null": "last", "dtype": "float32"}

    def cpu_fp32_state(self) -> dict[str, Tensor]:
        if any(value.dtype != torch.float32 for value in self.state_dict().values()):
            raise ValueError("owned scorer parameters must be FP32")
        return {key: value.detach().cpu().clone() for key, value in self.state_dict().items()}

    def restore_cpu_fp32_state(self, state: Mapping[str, Tensor], metadata: Mapping[str, object]) -> None:
        """Strict source-level restore, with all checks before assigning weights."""
        expected = self.state_dict()
        if dict(metadata) != self.metadata() or set(state) != set(expected):
            raise ValueError("owned scorer metadata or state keys mismatch")
        if any(not isinstance(value, Tensor) or value.dtype != torch.float32 or value.device.type != "cpu"
               or value.shape != expected[key].shape or not bool(torch.isfinite(value).all())
               for key, value in state.items()):
            raise ValueError("owned scorer state must have exact finite CPU FP32 tensors")
        if any(value.dtype != torch.float32 for value in expected.values()):
            raise ValueError("restore destination must be FP32")
        self.load_state_dict(dict(state), strict=True)

    def _check_inputs(self, token_states: Tensor, role_queries: Tensor, event_mask: Tensor, token_attention_mask: Tensor) -> None:
        if token_states.ndim != 4 or token_states.shape[-1] != self.state_dim or token_states.dtype != torch.float32:
            raise ValueError("token_states must be FP32[B,E,T,H]")
        batch, events, units, _ = token_states.shape
        if role_queries.shape != (batch, events, ROLE_COUNT, self.state_dim) or role_queries.dtype != torch.float32:
            raise ValueError("role_queries must be FP32[B,E,8,H]")
        if event_mask.shape != (batch, events) or event_mask.dtype != torch.bool:
            raise ValueError("event_mask must be bool[B,E]")
        if token_attention_mask.shape != (batch, units) or token_attention_mask.dtype != torch.bool:
            raise ValueError("token_attention_mask must be bool[B,T]")
        if any(value.device != token_states.device for value in (role_queries, event_mask, token_attention_mask)):
            raise ValueError("inputs must share a device")
        if any(parameter.device != token_states.device or parameter.dtype != torch.float32 for parameter in self.parameters()):
            raise ValueError("owned parameters must share the FP32 input device")

    @staticmethod
    def _span_evidence(states: Tensor, attention: Tensor, pairs: Tensor) -> Tensor:
        batch, events, _, dimension = states.shape
        pair_count = pairs.shape[1]
        if pair_count == 0:
            return states.new_zeros((batch, events, 0, 3 * dimension))
        starts, ends = pairs[:, :, 0].clamp_min(0), pairs[:, :, 1].clamp_min(0)
        start_index = starts[:, None, :, None].expand(batch, events, pair_count, dimension)
        end_index = ends[:, None, :, None].expand(batch, events, pair_count, dimension)
        first, last = states.gather(2, start_index), states.gather(2, end_index)
        prefix = torch.cat((states.new_zeros((batch, events, 1, dimension)), states.cumsum(dim=2)), dim=2)
        summed = prefix.gather(2, end_index + 1) - prefix.gather(2, start_index)
        count_prefix = torch.cat((torch.zeros((batch, 1), dtype=torch.long, device=states.device), attention.long().cumsum(dim=1)), dim=1)
        widths = count_prefix.gather(1, ends + 1) - count_prefix.gather(1, starts)
        mean = summed / widths.clamp_min(1)[:, None, :, None].to(states.dtype)
        return torch.cat((first, last, mean), dim=-1)

    def forward(self, token_states: Tensor, role_queries: Tensor, event_mask: Tensor, token_attention_mask: Tensor) -> JointArgumentSpanLogits:
        self._check_inputs(token_states, role_queries, event_mask, token_attention_mask)
        pairs, pair_mask = enumerate_unit_pairs(token_attention_mask)
        token_active = event_mask[:, :, None, None] & token_attention_mask[:, None, :, None]
        states = torch.where(token_active, token_states, torch.zeros_like(token_states))
        queries = torch.where(event_mask[:, :, None, None], role_queries, torch.zeros_like(role_queries))
        if not bool(torch.isfinite(states).all()) or not bool(torch.isfinite(queries).all()):
            raise ValueError("active owned states and queries must be finite")
        evidence = self._span_evidence(states, token_attention_mask, pairs)
        pair_vectors = self.pair_encoder(evidence)  # [B,E,P,D], reused by eight roles
        query_vectors = self.query_projection(queries)
        pair_scores = torch.einsum("bepd,berd->berp", pair_vectors, query_vectors) * self._scale
        pair_scores = torch.where(event_mask[:, :, None, None] & pair_mask[:, None, None, :], pair_scores, torch.zeros_like(pair_scores))
        null_scores = torch.where(event_mask[:, :, None, None], self.null_scorer(queries), torch.zeros_like(queries[..., :1]))
        result = JointArgumentSpanLogits(torch.cat((pair_scores, null_scores), dim=-1), pairs, pair_mask, event_mask.clone(), token_attention_mask.clone())
        result.validate()
        return result
