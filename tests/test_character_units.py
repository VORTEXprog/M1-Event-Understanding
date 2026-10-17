"""Character coordinates preserve Unicode text instead of UTF8 byte offsets."""
import pytest

from m1.canonical_units import CanonicalUnit, segment_text


def test_complete_grid_keeps_punctuation_combining_marks_apostrophes_and_suffix():
    text = "Participant_A co-op can't e\u0301. Участник_Б — 42!"
    units = segment_text(text)
    assert [unit.text for unit in units] == ["Participant_A", "co-op", "can't", "e\u0301", ".", "Участник_Б", "—", "42", "!"]
    assert [unit.unit_index for unit in units] == list(range(len(units)))
    assert all(text[unit.char_start:unit.char_end] == unit.text for unit in units)
    assert units[-1].char_end == len(text)
    assert all(left.char_end <= right.char_start for left, right in zip(units, units[1:]))


def test_offsets_are_codepoints_and_no_unicode_normalization_occurs():
    text = "e\u0301 é Участник_Б"
    units = segment_text(text)
    assert units[0].text == "e\u0301" and units[0].char_end == 2
    assert units[1].text == "é" and (units[1].char_start, units[1].char_end) == (3, 4)
    assert units[0].text != units[1].text
    assert len(text.encode("utf8")) > units[-1].char_end == len(text)
    assert units[-1].to_dict() == {"unit_index": 2, "text": "Участник_Б", "char_start": 5, "char_end": len(text)}


@pytest.mark.parametrize("text", ("", "  \t\n", None, 1))
def test_empty_or_nontext_input_is_rejected(text):
    with pytest.raises(ValueError):
        segment_text(text)


@pytest.mark.parametrize("arguments", (
    (False, "x", 0, 1), (-1, "x", 0, 1), (0, "", 0, 1),
    (0, "x", False, 1), (0, "x", 0, True), (0, "x", 2, 1), (0, "x", -1, 1),
))
def test_invalid_unit_geometry_and_boolean_coordinates_are_rejected(arguments):
    with pytest.raises(ValueError):
        CanonicalUnit(*arguments)
