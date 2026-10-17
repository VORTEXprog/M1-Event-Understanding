"""Portable inference definitions; no research datasets or trained weights included."""

from __future__ import annotations

import torch

from torch import nn

from m1.dataset import EVENT_ONTOLOGY

from m1.heads import SemanticHeadInputs

from m1.heads import _dimension

from m1.heads import _validate_module_input

def sentence_means(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if (tokens.ndim != 3 or mask.shape != tokens.shape[:2] or mask.dtype != torch.bool
            or tokens.dtype != torch.float32 or tokens.device.type != 'cpu' or mask.device.type != 'cpu'
            or not bool(torch.isfinite(tokens).all()) or not bool(mask.any(-1).all())):
        raise ValueError('sentence mean requires finite CPU FP32 tokens and nonempty boolean masks')
    # Select valid units before reduction so padding cannot alter FP32 arithmetic.
    return torch.stack([tokens[i, mask[i]].mean(0) for i in range(len(tokens))])

class EventSentenceMeanIdentityHead(nn.Module):
    """One owned 2D→hidden→7 MLP; frozen by default and order equivariant."""

    def __init__(self, encoder_dimension: int = 2048, hidden_dimension: int = 128):
        super().__init__()
        self.encoder_dimension = _dimension(encoder_dimension)
        self.hidden_dimension = _dimension(hidden_dimension)
        self.network = nn.Sequential(
            nn.Linear(2 * self.encoder_dimension, self.hidden_dimension, device='cpu', dtype=torch.float32),
            nn.GELU(),
            nn.Linear(self.hidden_dimension, len(EVENT_ONTOLOGY), device='cpu', dtype=torch.float32))
        self.eval().requires_grad_(False)

    def features(self, inputs: SemanticHeadInputs) -> torch.Tensor:
        _validate_module_input(self, inputs)
        means = sentence_means(inputs.token_states, inputs.token_mask)
        return torch.cat((inputs.event_states, means[:, None, :].expand_as(inputs.event_states)), -1).masked_fill(
            ~inputs.event_mask.unsqueeze(-1), 0)

    def forward_features(self, values: torch.Tensor, mask: torch.Tensor, *, ablate_context: bool = False) -> torch.Tensor:
        if (values.ndim != 3 or values.shape[-1] != 2 * self.encoder_dimension
                or values.dtype != torch.float32 or values.device.type != 'cpu'
                or mask.shape != values.shape[:2] or mask.dtype != torch.bool or mask.device.type != 'cpu'
                or type(ablate_context) is not bool or not bool(torch.isfinite(values).all())
                or any(p.dtype != torch.float32 or p.device.type != 'cpu' for p in self.parameters())):
            raise ValueError('identity compact values require finite aligned CPU FP32 event/sentence features')
        if ablate_context:
            values = values.clone()
            values[..., self.encoder_dimension:] = 0
        output = self.network(values).masked_fill(~mask.unsqueeze(-1), 0)
        if not bool(torch.isfinite(output).all()):
            raise ValueError('identity readout produced nonfinite scores')
        return output

    def forward(self, inputs: SemanticHeadInputs, *, ablate_context: bool = False) -> torch.Tensor:
        return self.forward_features(self.features(inputs), inputs.event_mask, ablate_context=ablate_context)
