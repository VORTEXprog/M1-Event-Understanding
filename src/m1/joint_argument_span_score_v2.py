"""Portable inference definitions; no research datasets or trained weights included."""

from __future__ import annotations

import torch

from torch import Tensor

from torch.nn import functional as F

from m1.joint_argument_span_score_v1 import JointArgumentSpanLogits

from m1.joint_argument_span_score_v1 import JointArgumentSpanScorer as _V1Scorer

from m1.joint_argument_span_score_v1 import enumerate_unit_pairs

VERSION = "M1_JOINT_ARGUMENT_SPAN_SCORE_v2"

class JointArgumentSpanScorer(_V1Scorer):
    """Factor linear work onto units and eight queries before full-pair tanh."""

    def metadata(self) -> dict[str, object]:
        return {**super().metadata(), "version": VERSION, "evaluation": "unit_linear_query_dual"}

    def copy_from_v1(self, scorer: _V1Scorer) -> None:
        if not isinstance(scorer, _V1Scorer) or scorer.metadata() != super().metadata():
            raise ValueError("copy requires compatible original v1 scorer metadata")
        self.restore_cpu_fp32_state(scorer.cpu_fp32_state(), self.metadata())

    def _pair_preactivation(self, states: Tensor, attention: Tensor, pairs: Tensor) -> Tensor:
        batch, events, _, _ = states.shape
        pair_count = pairs.shape[1]
        first_linear = self.pair_encoder[0]
        if pair_count == 0:
            # Preserve v1 empty-pair parameter gradients and disconnected input
            # semantics; the empty 3H tensor has no storage-sized pair features.
            return F.linear(states.new_zeros((batch, events, 0, 3 * self.state_dim)), first_linear.weight, first_linear.bias)
        start_weight, end_weight, mean_weight = first_linear.weight.split(self.state_dim, dim=1)
        starts, ends = pairs[:, :, 0].clamp_min(0), pairs[:, :, 1].clamp_min(0)
        start_index = starts[:, None, :, None].expand(batch, events, pair_count, self.hidden_dim)
        end_index = ends[:, None, :, None].expand(batch, events, pair_count, self.hidden_dim)
        first = F.linear(states, start_weight).gather(2, start_index)
        last = F.linear(states, end_weight).gather(2, end_index)
        projected_mean_units = F.linear(states, mean_weight)
        prefix = torch.cat((states.new_zeros((batch, events, 1, self.hidden_dim)), projected_mean_units.cumsum(dim=2)), dim=2)
        summed = prefix.gather(2, end_index + 1) - prefix.gather(2, start_index)
        count_prefix = torch.cat((torch.zeros((batch, 1), dtype=torch.long, device=states.device), attention.long().cumsum(dim=1)), dim=1)
        widths = count_prefix.gather(1, ends + 1) - count_prefix.gather(1, starts)
        mean = summed / widths.clamp_min(1)[:, None, :, None].to(states.dtype)
        return first + last + mean + first_linear.bias

    def forward(self, token_states: Tensor, role_queries: Tensor, event_mask: Tensor, token_attention_mask: Tensor) -> JointArgumentSpanLogits:
        self._check_inputs(token_states, role_queries, event_mask, token_attention_mask)
        pairs, pair_mask = enumerate_unit_pairs(token_attention_mask)
        token_active = event_mask[:, :, None, None] & token_attention_mask[:, None, :, None]
        states = torch.where(token_active, token_states, torch.zeros_like(token_states))
        queries = torch.where(event_mask[:, :, None, None], role_queries, torch.zeros_like(role_queries))
        if not bool(torch.isfinite(states).all()) or not bool(torch.isfinite(queries).all()):
            raise ValueError("active owned states and queries must be finite")
        hidden = torch.tanh(self._pair_preactivation(states, token_attention_mask, pairs))
        query_vectors = self.query_projection(queries)
        second_linear = self.pair_encoder[2]
        transformed_queries = F.linear(query_vectors, second_linear.weight.transpose(0, 1))
        bias_scores = (query_vectors * second_linear.bias).sum(dim=-1)
        pair_scores = (torch.einsum("beph,berh->berp", hidden, transformed_queries) + bias_scores[:, :, :, None]) * self._scale
        pair_scores = torch.where(event_mask[:, :, None, None] & pair_mask[:, None, None, :], pair_scores, torch.zeros_like(pair_scores))
        null_scores = torch.where(event_mask[:, :, None, None], self.null_scorer(queries), torch.zeros_like(queries[..., :1]))
        result = JointArgumentSpanLogits(torch.cat((pair_scores, null_scores), dim=-1), pairs, pair_mask, event_mask.clone(), token_attention_mask.clone())
        result.validate()
        return result
