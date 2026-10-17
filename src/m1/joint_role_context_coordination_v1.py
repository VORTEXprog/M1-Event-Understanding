"""Portable inference definitions; no research datasets or trained weights included."""

from __future__ import annotations

from collections.abc import Mapping

import math

import torch

from torch import Tensor

from torch import nn

from torch.nn import functional as F

VERSION = "M1_JOINT_ROLE_CONTEXT_COORDINATION_v1"

STATE_KEYS = ("query_key_value.weight", "joint_role_residual.weight")

ROLES = 8

NORMALIZATION_EPSILON = 1e-5

FORMULA = "C_PLUS_ZERO_R_OF_PER_EVENT_EIGHT_ROLE_MHA_NONAFFINE_LN_Z_PLUS_C"

def _dimensions(hidden_dimension: int, attention_heads: int) -> None:
    if (
        type(hidden_dimension) is not int or hidden_dimension <= 0
        or type(attention_heads) is not int or attention_heads <= 0
        or hidden_dimension % attention_heads
    ):
        raise ValueError("positive integer hidden size must be divisible by positive integer heads")

class JointRoleContextCoordination(nn.Module):
    """c + R(MHA(LN(z+c))) on [B,E,8,H], with exactly two bias-free weights.

    Constructor parameters train normally. The output projection is zeroed
    after both standard PyTorch draws; QKV draws are retained. Hidden state,
    role positions, schema restrictions, and persistent buffers are absent.
    """

    def __init__(self, hidden_dimension: int = 128, attention_heads: int = 4):
        super().__init__()
        _dimensions(hidden_dimension, attention_heads)
        self.hidden_dimension = hidden_dimension
        self.attention_heads = attention_heads
        self.query_key_value = nn.Linear(hidden_dimension, 3 * hidden_dimension, bias=False, device="cpu", dtype=torch.float32)
        self.joint_role_residual = nn.Linear(hidden_dimension, hidden_dimension, bias=False, device="cpu", dtype=torch.float32)
        nn.init.zeros_(self.joint_role_residual.weight)

    def _validate_owned(self) -> None:
        h = self.hidden_dimension
        if (
            set(self._modules) != {"query_key_value", "joint_role_residual"}
            or type(self.query_key_value) is not nn.Linear or type(self.joint_role_residual) is not nn.Linear
            or self.query_key_value.bias is not None or self.joint_role_residual.bias is not None
            or set(self.state_dict()) != set(STATE_KEYS)
            or tuple(self.query_key_value.weight.shape) != (3 * h, h)
            or tuple(self.joint_role_residual.weight.shape) != (h, h)
            or any(p.device.type != "cpu" or p.dtype != torch.float32 or not bool(torch.isfinite(p).all()) for p in self.parameters())
            or self.query_key_value.weight.untyped_storage().data_ptr() == self.joint_role_residual.weight.untyped_storage().data_ptr()
        ):
            raise ValueError("joint-role primitive requires its exact independent finite CPU FP32 two-weight topology")

    def forward(self, native_role_preactivation: Tensor, ordered_context: Tensor, event_mask: Tensor) -> Tensor:
        z, c = native_role_preactivation, ordered_context
        if (
            not all(isinstance(t, Tensor) for t in (z, c, event_mask))
            or z.ndim != 4 or z.shape != c.shape
            or z.shape[0] <= 0 or z.shape[1] <= 0 or z.shape[2:] != (ROLES, self.hidden_dimension)
            or event_mask.shape != z.shape[:2] or event_mask.dtype != torch.bool
            or z.dtype != torch.float32 or c.dtype != torch.float32
            or any(t.device.type != "cpu" for t in (z, c, event_mask))
            or not bool(torch.isfinite(z).all()) or not bool(torch.isfinite(c).all())
        ):
            raise ValueError("joint-role inputs require finite CPU FP32 [B,E,8,H] states and exact Boolean [B,E] mask")
        self._validate_owned()
        inactive = ~event_mask[:, :, None, None]
        z = z.masked_fill(inactive, 0)
        c = c.masked_fill(inactive, 0)
        fused = z + c
        if not bool(torch.isfinite(fused).all()):
            raise ValueError("joint-role input sum overflowed")
        normalized = F.layer_norm(fused, (self.hidden_dimension,), eps=NORMALIZATION_EPSILON)
        q, k, v = self.query_key_value(normalized).chunk(3, dim=-1)
        b, e, _, h = q.shape
        d = h // self.attention_heads
        q, k, v = (t.reshape(b, e, ROLES, self.attention_heads, d).transpose(2, 3) for t in (q, k, v))
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d)
        if not bool(torch.isfinite(scores).all()) or not bool(torch.isfinite(v).all()):
            raise ValueError("joint-role attention overflowed")
        weights = scores.softmax(dim=-1)
        attended = torch.matmul(weights, v).transpose(2, 3).reshape(b, e, ROLES, h)
        result = (c + self.joint_role_residual(attended)).masked_fill(inactive, 0)
        if not bool(torch.isfinite(result).all()):
            raise ValueError("joint-role output is nonfinite")
        return result

def model_from_state(
    state: Mapping[str, Tensor], *, hidden_dimension: int = 128, attention_heads: int = 4
) -> JointRoleContextCoordination:
    """Strict standalone two-weight restore; independent, CPU/eval/frozen.

    Dimensions are caller-declared, never inferred to accept a different owner.
    Constructor draws occur within a CPU RNG fork and do not perturb the caller.
    This factory does not deserialize files or own any m1 native parameters.
    """
    _dimensions(hidden_dimension, attention_heads)
    shapes = {STATE_KEYS[0]: (3 * hidden_dimension, hidden_dimension), STATE_KEYS[1]: (hidden_dimension, hidden_dimension)}
    if (
        not isinstance(state, Mapping) or set(state) != set(STATE_KEYS)
        or any(
            not isinstance(value, Tensor) or tuple(value.shape) != shapes[name]
            or value.device.type != "cpu" or value.dtype != torch.float32 or value.requires_grad
            or not bool(torch.isfinite(value).all())
            for name, value in state.items()
        )
    ):
        raise ValueError("joint-role restore requires exact detached finite CPU FP32 two-weight state")
    with torch.random.fork_rng(devices=[]):
        model = JointRoleContextCoordination(hidden_dimension, attention_heads)
    model.load_state_dict({name: value.detach().clone() for name, value in state.items()}, strict=True)
    model.eval().requires_grad_(False)
    model._validate_owned()
    return model
