"""Portable inference definitions; no research datasets or trained weights included."""

from __future__ import annotations

from collections.abc import Mapping

import math

import torch

from torch import Tensor

from torch import nn

from torch.nn import functional as F

VERSION = "M1_CROSS_EVENT_EIGHT_ROLE_CONTEXT_SOURCE_ONLY_v1"

SOURCE = "src/m1/cross_event_role_context_v1.py"

STATE_KEYS = ("query_weight", "key_weight", "value_weight", "output_weight")

ROLES, ATTENTION_HEADS = 8, 4

SHAPES = {key: (128, 128) for key in STATE_KEYS}

PARAMETERS = 65536

def _require(condition, message):
    if not condition:
        raise ValueError(message)

def shape_map(hidden_dimension: int = 128) -> dict[str, tuple[int, int]]:
    _require(type(hidden_dimension) is int and hidden_dimension > 0
             and hidden_dimension % ATTENTION_HEADS == 0,
             "positive hidden dimension divisible by four required")
    return {key: (hidden_dimension, hidden_dimension) for key in STATE_KEYS}

def initial_state(hidden_dimension: int = 128) -> dict[str, Tensor]:
    """Literal independent Q/K/V identity and zero output; no random draw."""
    shape_map(hidden_dimension)
    return {"query_weight": torch.eye(hidden_dimension, dtype=torch.float32, device="cpu"),
            "key_weight": torch.eye(hidden_dimension, dtype=torch.float32, device="cpu"),
            "value_weight": torch.eye(hidden_dimension, dtype=torch.float32, device="cpu"),
            "output_weight": torch.zeros(hidden_dimension, hidden_dimension, dtype=torch.float32, device="cpu")}

def basis_metadata(hidden_dimension: int = 128) -> dict:
    shapes = shape_map(hidden_dimension)
    return {"version": VERSION, "state_shapes": {key: list(shape) for key, shape in shapes.items()},
            "parameters": 4*hidden_dimension*hidden_dimension,
            "roles": ROLES, "attention_heads": ATTENTION_HEADS,
            "input": "SUPPLIED_POST_WITHIN_EVENT_COORDINATED_ROLE_QUERIES_CPU_FP32_B_E_8_H",
            "query_key_value": "SEPARATE_BIAS_FREE_LINEAR_H_H",
            "attention": "SCALED_DOT_PRODUCT_OVER_ALL_ROLES_OF_ALL_OTHER_ACTIVE_EVENTS",
            "own_event_keys": "ALL_EXCLUDED", "event_axis_positions": False,
            "empty_memory": "EXACT_GRAPH_CONNECTED_CANONICAL_POSITIVE_ZERO",
            "output": "DELTA_ONLY_SAME_B_E_8_H_BIAS_FREE_OUTPUT_LINEAR",
            "initialization": "Q_K_V_IDENTITY_OUTPUT_ZERO_LITERAL_NO_RNG",
            "within_event_coordination_changed": False, "event_count_cap": None,
            "hard_filler_uniqueness": False, "Gold_TYPE_or_lexical_inputs": False,
            "native_query_NULL_or_owner_integration_claimed": False,
            "TRAIN_or_runtime_admission": False, "paper_reproduction": False,
            "qualification_claimed": False}

class CrossEventRoleContext(nn.Module):
    """Return cross-event delta; every own-event memory key is masked out.

    Each active query sees all eight role states of every other active event.
    E0/E1 return zero for any finite owned weights; inactive and empty-memory
    outputs have positive-zero bits. Event permutations commute mathematically
    with this operation; floating-point reduction order can affect roundoff.
    """

    def __init__(self, hidden_dimension: int = 128):
        super().__init__()
        self.hidden_dimension = hidden_dimension
        for key, value in initial_state(hidden_dimension).items():
            self.register_parameter(key, nn.Parameter(value))
        self._validate_owned()

    def _validate_owned(self) -> None:
        parameters = dict(self.named_parameters())
        shapes = shape_map(self.hidden_dimension)
        _require(set(parameters) == set(STATE_KEYS) and not self._modules and not self._buffers
                 and len({value.untyped_storage().data_ptr() for value in parameters.values()}) == len(STATE_KEYS)
                 and all(tuple(value.shape) == shapes[key] and value.device.type == "cpu"
                         and value.dtype == torch.float32 and value.is_contiguous()
                         and not value.is_inference() and bool(torch.isfinite(value).all())
                         for key, value in parameters.items()),
                 "exact four independent finite ordinary CPU FP32 weights required")

    def forward(self, role_queries: Tensor, event_mask: Tensor) -> Tensor:
        self._validate_owned()
        h = self.hidden_dimension
        _require(isinstance(role_queries, Tensor) and role_queries.ndim == 4
                 and role_queries.shape[0] > 0 and role_queries.shape[2:] == (ROLES, h)
                 and role_queries.device.type == "cpu" and role_queries.dtype == torch.float32
                 and not role_queries.is_inference() and bool(torch.isfinite(role_queries).all())
                 and isinstance(event_mask, Tensor) and event_mask.dtype == torch.bool
                 and event_mask.device == role_queries.device and event_mask.shape == role_queries.shape[:2],
                 "finite ordinary CPU FP32[B,E,8,H] queries and Boolean[B,E] mask required")
        b, e = event_mask.shape
        companions = event_mask.sum(-1) > 1
        # No attention can exist in these rows. Bypass even Q/K products, so
        # arbitrary finite singleton weights cannot create an overflow/NaN.
        if not bool(companions.any()):
            zero = (role_queries * 0).sum()
            for parameter in self.parameters():
                zero = zero + (parameter * 0).sum()
            return (zero.expand_as(role_queries) + torch.zeros_like(role_queries))
        active = event_mask & companions[:, None]
        x = role_queries.masked_fill(~active[:, :, None, None], 0)
        width, d = e * ROLES, h // ATTENTION_HEADS
        projected = (F.linear(x, getattr(self, key)) for key in STATE_KEYS[:3])
        q, k, v = (value.reshape(b, width, ATTENTION_HEADS, d).transpose(1, 2)
                   for value in projected)
        _require(all(bool(torch.isfinite(value).all()) for value in (q, k, v)),
                 "finite Q/K/V projections required")
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d)
        _require(bool(torch.isfinite(scores).all()), "finite cross-event attention logits required")
        memory_events = torch.arange(e, device=event_mask.device).repeat_interleave(ROLES)
        other_event = memory_events[:, None] != memory_events[None, :]
        valid_roles = active.repeat_interleave(ROLES, dim=1)
        allowed = (valid_roles[:, None, :, None] & valid_roles[:, None, None, :]
                   & other_event[None, None, :, :])
        has_memory = allowed.any(-1, keepdim=True)
        # Empty rows become finite zeros BEFORE softmax; the final mask removes
        # their temporary uniform weights. No softmax is applied to all -inf.
        safe = torch.where(has_memory, scores.masked_fill(~allowed, -torch.inf), torch.zeros_like(scores))
        weights = F.softmax(safe, dim=-1).masked_fill(~allowed, 0)
        attended = torch.matmul(weights, v).transpose(1, 2).reshape(b, e, ROLES, h)
        delta = F.linear(attended, self.output_weight)
        _require(bool(torch.isfinite(delta).all()), "finite cross-event residual required")
        delta = delta.masked_fill(~active[:, :, None, None], 0)
        # IEEE -0 + +0 is +0. Addition preserves the derivative at a warm zero,
        # including the live first-step output-weight gradient.
        return delta + torch.zeros_like(delta)

def model_from_state(state: Mapping[str, Tensor], *, hidden_dimension: int = 128) -> CrossEventRoleContext:
    """Restore the four supplied weights without RNG or storage aliasing."""
    shapes = shape_map(hidden_dimension)
    _require(isinstance(state, Mapping) and set(state) == set(STATE_KEYS)
             and len({value.untyped_storage().data_ptr() for value in state.values() if isinstance(value, Tensor)}) == 4
             and all(isinstance(value, Tensor) and tuple(value.shape) == shapes[key]
                     and value.device.type == "cpu" and value.dtype == torch.float32
                     and value.is_contiguous() and not value.requires_grad and not value.is_inference()
                     and bool(torch.isfinite(value).all()) for key, value in state.items()),
             "four complete independent detached finite ordinary CPU FP32 state tensors required")
    net = CrossEventRoleContext(hidden_dimension)
    net.load_state_dict(dict(state))
    net.eval().requires_grad_(False)
    net._validate_owned()
    _require(not {value.untyped_storage().data_ptr() for value in state.values()}.intersection(
        value.untyped_storage().data_ptr() for value in net.state_dict().values()),
        "restored cross-event context may not alias supplied weights")
    return net
