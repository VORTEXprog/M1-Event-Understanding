"""Explicit bounded two-slot compatibility operation."""
from __future__ import annotations
import torch
from torch import Tensor, nn

E_MAX = 2


class LearnedSlotAssignment(nn.Module):
    """Shared event-to-legacy-slot scores; weights are supplied separately."""
    def __init__(self, dimension: int, hidden: int = 64):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(dimension, hidden), nn.GELU(), nn.Linear(hidden, E_MAX))

    def forward(self, event_states: Tensor, event_mask: Tensor) -> Tensor:
        parameter = next(self.parameters())
        if (event_states.ndim != 3 or event_states.shape[1] != E_MAX
                or event_mask.shape != event_states.shape[:2] or event_mask.dtype != torch.bool
                or event_states.dtype != parameter.dtype or event_states.device != parameter.device
                or event_mask.device != event_states.device or not torch.isfinite(event_states).all()):
            raise ValueError('slot scorer expects finite label-free event tensors')
        return self.network(event_states.masked_fill(~event_mask.unsqueeze(-1), 0)).masked_fill(~event_mask.unsqueeze(-1), 0)


def assignment_permutation(scores: Tensor, event_mask: Tensor) -> Tensor:
    """Max-score two-slot assignment, compact singletons; no semantic routing."""
    if (scores.ndim != 3 or scores.shape[1:] != (E_MAX, E_MAX)
            or event_mask.shape != scores.shape[:2] or event_mask.dtype != torch.bool
            or scores.device != event_mask.device or not scores.is_floating_point()
            or not torch.isfinite(scores).all()):
        raise ValueError('slot assignment geometry differs')
    swap = (scores[:, 0, 1] + scores[:, 1, 0]) > (scores[:, 0, 0] + scores[:, 1, 1])
    count = event_mask.sum(-1)
    swap = torch.where(count == 1, event_mask[:, 1], swap & (count == 2))
    first = swap.long()
    return torch.stack((first, 1 - first), dim=-1)
