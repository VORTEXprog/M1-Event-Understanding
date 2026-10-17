"""Portable inference definitions; no research datasets or trained weights included."""

from __future__ import annotations

import codecs

from m1.canonicalizer import EOS_ID

_PREFIXES = (b"entity/", b"object/")

def valid_symbol(text):
    return (isinstance(text, str) and "/" in text
            and text.split("/", 1)[0] in ("entity", "object")
            and bool(text.split("/", 1)[1]) and not any(c.isspace() for c in text))

def minimum_completion_bytes(prefix: bytes):
    """Minimum additional bytes to a legal symbol, or None for dead prefixes."""
    if not isinstance(prefix, bytes):
        raise ValueError("canonical prefix must be bytes")
    unfinished = [len(p) - len(prefix) + 1 for p in _PREFIXES if p.startswith(prefix)]
    if unfinished:
        return min(unfinished)
    header = next((p for p in _PREFIXES if prefix.startswith(p)), None)
    if header is None:
        return None
    suffix = prefix[len(header):]
    decoder = codecs.getincrementaldecoder("utf-8")("strict")
    try:
        complete = decoder.decode(suffix, final=False)
    except UnicodeDecodeError:
        return None
    if any(c.isspace() for c in complete):
        return None
    pending = decoder.getstate()[0]
    if not pending:
        return 0 if complete else 1
    lead = pending[0]
    if len(pending) >= 2:
        low, high = {0xE0: (0xA0, 0xBF), 0xED: (0x80, 0x9F),
                     0xF0: (0x90, 0xBF), 0xF4: (0x80, 0x8F)}.get(lead, (0x80, 0xBF))
        if not low <= pending[1] <= high:
            return None
    total = 2 if lead < 0xE0 else 3 if lead < 0xF0 else 4
    return total - len(pending)

def feasible_token_ids(prefix: bytes, maximum_output_bytes: int):
    if type(maximum_output_bytes) is not int or maximum_output_bytes < 8:
        raise ValueError("canonical output budget cannot encode its declared symbol type")
    minimum = minimum_completion_bytes(prefix)
    remaining = maximum_output_bytes - len(prefix)
    if minimum is None or remaining < minimum:
        raise ValueError("canonical prefix is outside the feasible output domain")
    allowed = []
    if remaining:
        for value in range(256):
            needed = minimum_completion_bytes(prefix + bytes((value,)))
            if needed is not None and needed <= remaining - 1:
                allowed.append(value)
    if minimum == 0:
        allowed.append(EOS_ID)
    if not allowed:
        raise ValueError("canonical output domain unexpectedly empty")
    return tuple(allowed)
