"""Portable inference definitions; no research datasets or trained weights included."""

from __future__ import annotations

from collections.abc import Mapping

from dataclasses import dataclass

import math

import torch

from torch import Tensor

from torch import nn

from torch.nn import functional as F

from m1.joint_argument_span_score_v1 import enumerate_unit_pairs

VERSION = "M1_FULL_CONTEXT_HEAD_CANDIDATE_POOLING_v1_UNADMITTED"

STATE_KEYS = ("head_weight", "candidate_weight", "key_weight", "value_weight", "output_weight")

MAX_QUERY_CHUNK = 128

PARAMETERS = 14336

def require(condition, message):
    if not condition:
        raise ValueError(message)

def shape_map(hidden_dimension=128, span_dimension=32, attention_dimension=32):
    require(all(type(v) is int and v > 0 for v in
        (hidden_dimension, span_dimension, attention_dimension)), "positive integer dimensions required")
    h, d, r = hidden_dimension, span_dimension, attention_dimension
    return dict(zip(STATE_KEYS, ((r, h), (r, d), (r, h), (r, h), (d, r))))

def initial_state(hidden_dimension=128, span_dimension=32, attention_dimension=32):
    """Literal partial identities and zero final projection; no RNG draw."""
    shapes = shape_map(hidden_dimension, span_dimension, attention_dimension)
    return {k: (torch.zeros(*s, device="cpu", dtype=torch.float32) if k == "output_weight"
        else torch.eye(*s, device="cpu", dtype=torch.float32)) for k, s in shapes.items()}

def basis_metadata(hidden_dimension=128, span_dimension=32, attention_dimension=32):
    shapes = shape_map(hidden_dimension, span_dimension, attention_dimension)
    return {"version": VERSION, "state_shapes": {k: list(v) for k, v in shapes.items()},
        "query": "W_HEAD_MEAN_VALID_WHOLE_OWN_HEAD_PLUS_W_CANDIDATE_NATIVE_V0",
        "context": "ALL_VALID_ORIGINAL_ORDERED_UNITS_W_KEY_AND_W_VALUE",
        "attention": "MASKED_SOFTMAX_DOT_QUERY_KEY_DIV_SQRT_R_FP32",
        "returned": "W_OUT_WEIGHTED_CONTEXT_IN_NATIVE_V_SPACE",
        "native_seam": "V0_PLUS_CORRECTION_BEFORE_UNCHANGED_UNARY_U_L_R",
        "initialization": "LITERAL_PARTIAL_IDENTITIES_ZERO_W_OUT_NO_RNG",
        "initial_gradient": "W_OUT_LIVE_OTHER_FOUR_ZERO_UNTIL_W_OUT_CHANGES",
        "query_chunk_max": MAX_QUERY_CHUNK, "span_context_pruning": False,
        "singleton_HEAD_and_outside_context_dependency": True,
        "other_HEAD_mask_input": False, "cross_event_decoding": False,
        "TYPE_Gold_lexical_inputs": False, "SCPRG_reproduction": False,
        "pretrained_attention_input": False, "qualification": False}

@dataclass(frozen=True)
class FullContextPoolingInputs:
    unit_states: Tensor       # CPU FP32[B,T,H], unchanged ordered states
    token_mask: Tensor        # BOOL[B,T], original-coordinate holes allowed
    event_mask: Tensor        # BOOL[B,E], E>=0; inactive slots permitted
    event_unit_spans: Tensor  # I64[B,E,2], predicted half-open whole HEADs
    candidate_vectors: Tensor # CPU FP32[B,E,P,D], complete native V0
    pair_mask: Tensor         # BOOL[B,P], exact native complete enumeration

    def validate(self, hidden_dimension, span_dimension):
        x, tm, em, a, v, pm = (getattr(self, k) for k in self.__dataclass_fields__)
        require(all(isinstance(z, Tensor) and z.device.type == "cpu" for z in (x, tm, em, a, v, pm))
            and x.ndim == 3 and x.shape[0] > 0 and x.shape[1] > 0 and x.shape[2] == hidden_dimension
            and x.dtype == v.dtype == torch.float32 and bool(torch.isfinite(x).all())
            and bool(torch.isfinite(v).all()) and tm.dtype == em.dtype == pm.dtype == torch.bool
            and tm.shape == x.shape[:2] and bool(tm.any(-1).all())
            and em.ndim == 2 and em.shape[0] == x.shape[0]
            and a.dtype == torch.int64 and a.shape == (*em.shape, 2),
            "finite CPU FP32 states/candidates and original mask/HEAD geometry required")
        start, end = a.unbind(-1)
        width = x.shape[1]
        require(not bool(((start < 0) | (start >= end) | (end > width)).masked_select(em).any())
            and not bool(a.masked_select(~em[..., None].expand_as(a)).count_nonzero()),
            "active HEAD bounds or inactive zero intervals differ")
        require(bool((tm.gather(1, start.clamp(0, width-1))
            & tm.gather(1, (end-1).clamp(0, width-1))).masked_select(em).all()),
            "HEAD endpoints must be valid original units; interior holes remain masked")
        pairs, expected = enumerate_unit_pairs(tm)
        require(pm.shape == expected.shape and torch.equal(pm, expected)
            and v.shape == (*em.shape, pairs.shape[1], span_dimension),
            "complete native V0 and unchanged START-major pair mask required")
        return pairs

@dataclass(frozen=True)
class FullContextPoolingResult:
    projected_correction: Tensor  # FP32[B,E,P,D], caller adds to V0
    pair_indices: Tensor
    pair_mask: Tensor
    event_mask: Tensor
    token_mask: Tensor

class FullContextHeadCandidatePooling(nn.Module):
    def __init__(self, hidden_dimension=128, span_dimension=32, attention_dimension=32,
                 query_chunk_size=MAX_QUERY_CHUNK):
        super().__init__()
        require(type(query_chunk_size) is int and 1 <= query_chunk_size <= MAX_QUERY_CHUNK,
            "query chunks must contain 1..128 complete candidates")
        self.hidden_dimension, self.span_dimension = hidden_dimension, span_dimension
        self.attention_dimension, self.query_chunk_size = attention_dimension, query_chunk_size
        for k, value in initial_state(hidden_dimension, span_dimension, attention_dimension).items():
            setattr(self, k, nn.Parameter(value))

    def _validate_owned(self):
        shapes = shape_map(self.hidden_dimension, self.span_dimension, self.attention_dimension)
        state = dict(self.named_parameters())
        require(set(state) == set(shapes) and not self._modules and not self._buffers
            and len({p.untyped_storage().data_ptr() for p in state.values()}) == len(shapes)
            and all(p.shape == shapes[k] and p.device.type == "cpu" and p.dtype == torch.float32
                and bool(torch.isfinite(p).all()) for k, p in state.items()),
            "five independent finite CPU FP32 full-context weights required")
        require(type(self.query_chunk_size) is int and 1 <= self.query_chunk_size <= MAX_QUERY_CHUNK,
            "query chunk configuration differs")

    def forward(self, inputs: FullContextPoolingInputs) -> FullContextPoolingResult:
        self._validate_owned()
        require(type(inputs) is FullContextPoolingInputs, "explicit native states/V0/predicted HEAD inputs required")
        pairs = inputs.validate(self.hidden_dimension, self.span_dimension)
        x = inputs.unit_states.masked_fill(~inputs.token_mask[..., None], 0)
        t = torch.arange(x.shape[1], device=x.device)[None, None]
        start, end = (z[..., None] for z in inputs.event_unit_spans.unbind(-1))
        inside = (t >= start) & (t < end) & inputs.token_mask[:, None] & inputs.event_mask[..., None]
        heads = torch.einsum("bet,bth->beh", inside.float(), x)/inside.sum(-1).clamp_min(1)[..., None]
        q_head = F.linear(heads, self.head_weight)
        keys, values = F.linear(x, self.key_weight), F.linear(x, self.value_weight)
        active = inputs.event_mask[..., None] & inputs.pair_mask[:, None]
        chunks = []
        for left in range(0, pairs.shape[1], self.query_chunk_size):
            right = left+self.query_chunk_size
            valid = active[:, :, left:right]
            v0 = inputs.candidate_vectors[:, :, left:right].masked_fill(~valid[..., None], 0)
            query = q_head[:, :, None] + F.linear(v0, self.candidate_weight)
            logits = torch.einsum("bepr,btr->bept", query, keys)/math.sqrt(self.attention_dimension)
            require(bool(torch.isfinite(logits).all()), "finite full-context attention scores required")
            logits = logits.masked_fill(~inputs.token_mask[:, None, None], -torch.inf)
            mass = logits.softmax(-1)
            context = torch.einsum("bept,btr->bepr", mass, values)
            correction = F.linear(context, self.output_weight).masked_fill(~valid[..., None], 0)
            require(bool(torch.isfinite(correction).all()), "finite complete V-space correction required")
            chunks.append(correction)
        return FullContextPoolingResult(torch.cat(chunks, 2), pairs, inputs.pair_mask.clone(),
            inputs.event_mask.clone(), inputs.token_mask.clone())

def model_from_state(state: Mapping[str, Tensor], *, hidden_dimension=128,
                     span_dimension=32, attention_dimension=32, query_chunk_size=MAX_QUERY_CHUNK):
    shapes = shape_map(hidden_dimension, span_dimension, attention_dimension)
    require(isinstance(state, Mapping) and set(state) == set(shapes)
        and all(isinstance(p, Tensor) and p.shape == shapes[k] and p.device.type == "cpu"
            and p.dtype == torch.float32 and not p.requires_grad and not p.is_inference()
            and bool(torch.isfinite(p).all()) for k, p in state.items()),
        "exact detached finite CPU FP32 supplied full-context state required")
    model = FullContextHeadCandidatePooling(hidden_dimension, span_dimension,
        attention_dimension, query_chunk_size)
    model.load_state_dict({k: p.detach().clone() for k, p in state.items()}, strict=True)
    model.eval().requires_grad_(False)
    model._validate_owned()
    return model

def export_state(model):
    require(type(model) is FullContextHeadCandidatePooling, "genuine full-context component required")
    model._validate_owned()
    return {k: p.detach().clone() for k, p in model.state_dict().items()}
