"""Portable inference definitions; no research datasets or trained weights included."""

from __future__ import annotations

from dataclasses import dataclass

BOS_ID = 257

EOS_ID = 258

SEQUENCE_VOCABULARY_SIZE = 259

@dataclass(frozen=True)
class CanonicalDecode:
    text: str | None
    error: str | None
    byte_count: int
