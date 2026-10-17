"""Portable inference definitions; no research datasets or trained weights included."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from torch import Tensor

from torch import nn

ROLE_ORDER = (
    "agent", "source", "possessor", "recipient", "addressee", "context",
    "owner", "object",
)

POLARITY_ORDER = ("NONE", "AFFIRMATIVE", "NEGATIVE")

RELATION_ORDER = ("NONE", "BEFORE", "AFTER")

def _dimension(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("head dimensions must be positive integers")
    return value

@dataclass(frozen=True)
class SemanticHeadInputs:
    token_states: Tensor  # [B, T, D]
    event_states: Tensor  # [B, E, D]; spans are pooled by the upstream encoder
    token_mask: Tensor  # bool [B, T]
    event_mask: Tensor  # bool [B, E]; all-false supports a zero-event sentence

    def validate(self, encoder_dimension: int) -> None:
        tensors = (self.token_states, self.event_states, self.token_mask, self.event_mask)
        if not all(isinstance(value, Tensor) for value in tensors):
            raise ValueError("semantic head inputs must be tensors")
        if self.token_states.ndim != 3 or self.event_states.ndim != 3:
            raise ValueError("token/event states must have rank three")
        batch, tokens, dimension = self.token_states.shape
        if (
            batch <= 0 or tokens <= 0 or dimension != encoder_dimension
            or self.event_states.shape[0] != batch
            or self.event_states.shape[1] <= 0
            or self.event_states.shape[2] != dimension
            or self.token_mask.shape != (batch, tokens)
            or self.event_mask.shape != self.event_states.shape[:2]
        ):
            raise ValueError("semantic head input shapes differ")
        if self.token_mask.dtype != torch.bool or self.event_mask.dtype != torch.bool:
            raise ValueError("token/event masks must be boolean")
        if (
            not self.token_states.is_floating_point()
            or self.token_states.dtype != self.event_states.dtype
            or len({value.device for value in tensors}) != 1
        ):
            raise ValueError("semantic head input dtype/device differs")
        if not bool(self.token_mask.any(dim=1).all().item()):
            raise ValueError("each sentence needs at least one unmasked text token")
        if not all(bool(torch.isfinite(value).all().item()) for value in tensors[:2]):
            raise ValueError("semantic head input contains nonfinite values")

@dataclass(frozen=True)
class SemanticHeadLogits:
    argument_start_logits: Tensor  # [B, E, 8, T+1], final index T is NONE
    argument_end_logits: Tensor  # [B, E, 8, T+1], inclusive final token or NONE
    polarity_logits: Tensor  # [B, E, 3], POLARITY_ORDER
    relation_logits: Tensor  # [B, E, E, 3], RELATION_ORDER; ordered event pairs
    context_attention: Tensor  # [B, E, T], zero for inactive events/padding
    event_mask: Tensor  # [B, E]
    relation_mask: Tensor  # [B, E, E], excludes inactive and self edges

def _validate_module_input(module: nn.Module, inputs: SemanticHeadInputs) -> None:
    if not isinstance(inputs, SemanticHeadInputs):
        raise ValueError("expected SemanticHeadInputs, not semantic metadata")
    inputs.validate(module.encoder_dimension)
    parameter = next(module.parameters())
    if inputs.token_states.device != parameter.device or inputs.token_states.dtype != parameter.dtype:
        raise ValueError("input and head parameters must share dtype/device")

def _validate_context(inputs: SemanticHeadInputs, context: Tensor, hidden: int) -> None:
    if (
        not isinstance(context, Tensor)
        or context.shape != (*inputs.event_mask.shape, hidden)
        or context.dtype != inputs.token_states.dtype
        or context.device != inputs.token_states.device
        or not bool(torch.isfinite(context).all().item())
    ):
        raise ValueError("shared event context differs")

def _event_inputs(inputs: SemanticHeadInputs, context: Tensor) -> Tensor:
    value = torch.cat((inputs.event_states, context), dim=-1)
    return value.masked_fill(~inputs.event_mask.unsqueeze(-1), 0)

class SharedContextAttention(nn.Module):
    """Learned event queries over the sentence's unmasked token vectors."""

    def __init__(self, encoder_dimension: int, hidden_dimension: int = 64) -> None:
        super().__init__()
        self.encoder_dimension = _dimension(encoder_dimension)
        self.hidden_dimension = _dimension(hidden_dimension)
        self.query = nn.Linear(encoder_dimension, hidden_dimension, bias=False)
        self.key = nn.Linear(encoder_dimension, hidden_dimension, bias=False)
        self.value = nn.Linear(encoder_dimension, hidden_dimension, bias=False)

    def forward(self, inputs: SemanticHeadInputs) -> tuple[Tensor, Tensor]:
        _validate_module_input(self, inputs)
        tokens = inputs.token_states.masked_fill(~inputs.token_mask.unsqueeze(-1), 0)
        events = inputs.event_states.masked_fill(~inputs.event_mask.unsqueeze(-1), 0)
        scores = torch.einsum("beh,bth->bet", self.query(events), self.key(tokens))
        scores = scores / self.hidden_dimension ** 0.5
        scores = scores.masked_fill(~inputs.token_mask[:, None, :], torch.finfo(scores.dtype).min)
        attention = scores.softmax(dim=-1)
        attention = attention.masked_fill(~inputs.event_mask.unsqueeze(-1), 0)
        context = torch.einsum("bet,bth->beh", attention, self.value(tokens))
        return context, attention

class ArgumentBindingHead(nn.Module):
    """Eight learned role-conditioned start/end pointers with a NONE option."""

    def __init__(self, encoder_dimension: int, hidden_dimension: int = 64) -> None:
        super().__init__()
        self.encoder_dimension = _dimension(encoder_dimension)
        self.hidden_dimension = _dimension(hidden_dimension)
        self.event_query = nn.Linear(encoder_dimension + hidden_dimension, hidden_dimension)
        self.role_queries = nn.Parameter(torch.empty(len(ROLE_ORDER), hidden_dimension))
        nn.init.normal_(self.role_queries, std=hidden_dimension ** -0.5)
        self.start_query = nn.Linear(hidden_dimension, hidden_dimension, bias=False)
        self.end_query = nn.Linear(hidden_dimension, hidden_dimension, bias=False)
        self.start_key = nn.Linear(encoder_dimension, hidden_dimension, bias=False)
        self.end_key = nn.Linear(encoder_dimension, hidden_dimension, bias=False)
        self.none = nn.Linear(hidden_dimension, 2)

    def forward(self, inputs: SemanticHeadInputs, context: Tensor) -> tuple[Tensor, Tensor]:
        _validate_module_input(self, inputs)
        _validate_context(inputs, context, self.hidden_dimension)
        event = self.event_query(_event_inputs(inputs, context))
        queries = torch.tanh(event.unsqueeze(2) + self.role_queries[None, None, :, :])
        tokens = inputs.token_states.masked_fill(~inputs.token_mask.unsqueeze(-1), 0)
        start = torch.einsum("berh,bth->bert", self.start_query(queries), self.start_key(tokens))
        end = torch.einsum("berh,bth->bert", self.end_query(queries), self.end_key(tokens))
        scale = self.hidden_dimension ** 0.5
        valid = inputs.event_mask[:, :, None, None] & inputs.token_mask[:, None, None, :]
        floor = torch.finfo(start.dtype).min
        start = (start / scale).masked_fill(~valid, floor)
        end = (end / scale).masked_fill(~valid, floor)
        none = self.none(queries).masked_fill(~inputs.event_mask[:, :, None, None], 0)
        return torch.cat((start, none[..., :1]), -1), torch.cat((end, none[..., 1:]), -1)

class PolarityHead(nn.Module):
    """Event-local NONE/affirmative/negative logits; no predicate rule table."""

    def __init__(self, encoder_dimension: int, hidden_dimension: int = 64) -> None:
        super().__init__()
        self.encoder_dimension = _dimension(encoder_dimension)
        self.hidden_dimension = _dimension(hidden_dimension)
        self.network = nn.Sequential(
            nn.Linear(encoder_dimension + hidden_dimension, hidden_dimension),
            nn.GELU(), nn.Linear(hidden_dimension, len(POLARITY_ORDER)),
        )

    def forward(self, inputs: SemanticHeadInputs, context: Tensor) -> Tensor:
        _validate_module_input(self, inputs)
        _validate_context(inputs, context, self.hidden_dimension)
        return self.network(_event_inputs(inputs, context)).masked_fill(
            ~inputs.event_mask.unsqueeze(-1), 0,
        )

class EventRelationsHead(nn.Module):
    """Shared directed-pair classifier; row order itself is not semantic order."""

    def __init__(self, encoder_dimension: int, hidden_dimension: int = 64) -> None:
        super().__init__()
        self.encoder_dimension = _dimension(encoder_dimension)
        self.hidden_dimension = _dimension(hidden_dimension)
        self.event_projection = nn.Linear(encoder_dimension + hidden_dimension, hidden_dimension)
        self.network = nn.Sequential(
            nn.Linear(hidden_dimension * 4, hidden_dimension), nn.GELU(),
            nn.Linear(hidden_dimension, len(RELATION_ORDER)),
        )

    def forward(self, inputs: SemanticHeadInputs, context: Tensor) -> tuple[Tensor, Tensor]:
        _validate_module_input(self, inputs)
        _validate_context(inputs, context, self.hidden_dimension)
        event = torch.tanh(self.event_projection(_event_inputs(inputs, context)))
        count = event.shape[1]
        left = event.unsqueeze(2).expand(-1, -1, count, -1)
        right = event.unsqueeze(1).expand(-1, count, -1, -1)
        logits = self.network(torch.cat((left, right, left - right, left * right), dim=-1))
        mask = inputs.event_mask.unsqueeze(2) & inputs.event_mask.unsqueeze(1)
        mask = mask & ~torch.eye(count, dtype=torch.bool, device=event.device).unsqueeze(0)
        return logits.masked_fill(~mask.unsqueeze(-1), 0), mask

class SemanticHeads(nn.Module):
    """Compose independent factor heads with one explicitly shared context."""

    def __init__(self, encoder_dimension: int, hidden_dimension: int = 64) -> None:
        super().__init__()
        self.encoder_dimension = _dimension(encoder_dimension)
        self.hidden_dimension = _dimension(hidden_dimension)
        self.context = SharedContextAttention(encoder_dimension, hidden_dimension)
        self.argument_binding = ArgumentBindingHead(encoder_dimension, hidden_dimension)
        self.polarity = PolarityHead(encoder_dimension, hidden_dimension)
        self.event_relations = EventRelationsHead(encoder_dimension, hidden_dimension)

    def checkpoint_modules(self) -> dict[str, nn.Module]:
        """Each returned owner can save/restore its own standard state_dict."""
        return {
            "context": self.context, "argument_binding": self.argument_binding,
            "polarity": self.polarity, "event_relations": self.event_relations,
        }

    def forward(self, inputs: SemanticHeadInputs) -> SemanticHeadLogits:
        context, attention = self.context(inputs)
        start, end = self.argument_binding(inputs, context)
        polarity = self.polarity(inputs, context)
        relations, relation_mask = self.event_relations(inputs, context)
        return SemanticHeadLogits(
            argument_start_logits=start, argument_end_logits=end,
            polarity_logits=polarity, relation_logits=relations,
            context_attention=attention, event_mask=inputs.event_mask,
            relation_mask=relation_mask,
        )

__all__ = [
    "ROLE_ORDER", "POLARITY_ORDER", "RELATION_ORDER", "SemanticHeadInputs",
    "SemanticHeadLogits", "SharedContextAttention", "ArgumentBindingHead",
    "PolarityHead", "EventRelationsHead", "SemanticHeads",
]
