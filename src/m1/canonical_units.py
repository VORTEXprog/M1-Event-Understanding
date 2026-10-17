"""Portable inference definitions; no research datasets or trained weights included."""

from __future__ import annotations

import unicodedata

from dataclasses import dataclass

class E2PLTargetError(ValueError):
    """A frozen target, unit map, BIO stream, or L0 input is invalid."""

@dataclass(frozen=True, slots=True)
class CanonicalUnit:
    unit_index: int
    text: str
    char_start: int
    char_end: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.unit_index, bool)
            or not isinstance(self.unit_index, int)
            or self.unit_index < 0
            or not isinstance(self.text, str)
            or not self.text
            or isinstance(self.char_start, bool)
            or not isinstance(self.char_start, int)
            or isinstance(self.char_end, bool)
            or not isinstance(self.char_end, int)
            or self.char_start < 0
            or self.char_end <= self.char_start
        ):
            raise E2PLTargetError("canonical unit geometry is invalid")

    def to_dict(self) -> dict[str, object]:
        return {
            "unit_index": self.unit_index,
            "text": self.text,
            "char_start": self.char_start,
            "char_end": self.char_end,
        }

def _is_word_character(character: str) -> bool:
    return unicodedata.category(character)[:1] in {"L", "M", "N"} or character == "_"

def segment_text(text: str) -> tuple[CanonicalUnit, ...]:
    """Create the frozen language-agnostic lexical character-offset grid."""

    if not isinstance(text, str) or not text:
        raise E2PLTargetError("canonical segmentation requires nonempty raw text")
    units: list[CanonicalUnit] = []
    cursor = 0
    while cursor < len(text):
        if text[cursor].isspace():
            cursor += 1
            continue
        start = cursor
        if _is_word_character(text[cursor]):
            cursor += 1
            while cursor < len(text):
                if _is_word_character(text[cursor]):
                    cursor += 1
                    continue
                if (
                    text[cursor] in {"'", "’", "-"}
                    and cursor + 1 < len(text)
                    and _is_word_character(text[cursor - 1])
                    and _is_word_character(text[cursor + 1])
                ):
                    cursor += 1
                    continue
                break
        else:
            cursor += 1
        units.append(
            CanonicalUnit(
                unit_index=len(units),
                text=text[start:cursor],
                char_start=start,
                char_end=cursor,
            )
        )
    if not units:
        raise E2PLTargetError("canonical segmentation produced no units")
    return tuple(units)
