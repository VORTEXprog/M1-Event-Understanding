"""Portable inference definitions; no research datasets or trained weights included."""

from __future__ import annotations

import torch

import m1.event_sentence_mean_normalized_identity_v1 as normalized_base

from m1.heads import SemanticHeadInputs

from m1.heads import _validate_module_input

NORMALIZED_VERSION = "nmcm.m1.event_sentence_tail_normalized_identity/v1"

FORMULA = "CONCAT_NATIVE_EVENT_MEAN_AND_LAST_VALID_NORMALIZED_SENTENCE_CANONICAL_UNIT"

def sentence_tails(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Select each sentence's highest True index, including noncontiguous masks."""
    if (not isinstance(tokens, torch.Tensor) or not isinstance(mask, torch.Tensor)
            or tokens.ndim != 3 or not tokens.shape[0] or not tokens.shape[1]
            or mask.shape != tokens.shape[:2] or mask.dtype != torch.bool
            or tokens.dtype != torch.float32 or tokens.device.type != "cpu"
            or mask.device.type != "cpu" or not bool(torch.isfinite(tokens).all())
            or not bool(mask.any(-1).all())):
        raise ValueError("sentence tail requires finite CPU FP32 tokens and nonempty boolean masks")
    indices = torch.arange(tokens.shape[1], dtype=torch.int64).expand_as(mask)
    last = indices.masked_fill(~mask, -1).amax(-1)
    return tokens[torch.arange(len(tokens), dtype=torch.int64), last]

def _tail_features(module, inputs: SemanticHeadInputs) -> torch.Tensor:
    _validate_module_input(module, inputs)
    tails = sentence_tails(inputs.token_states, inputs.token_mask)
    values = torch.cat((inputs.event_states, tails[:, None, :].expand_as(inputs.event_states)), -1)
    return values.masked_fill(~inputs.event_mask.unsqueeze(-1), 0)

class EventSentenceTailNormalizedIdentityHead(normalized_base.EventSentenceMeanNormalizedIdentityHead):
    """Same fresh affine initialization with owned hidden LayerNorm before GELU."""

    architecture = NORMALIZED_VERSION
    input_formula = FORMULA

    def features(self, inputs: SemanticHeadInputs) -> torch.Tensor:
        return _tail_features(self, inputs)
