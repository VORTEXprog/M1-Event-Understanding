"""Portable inference definitions; no research datasets or trained weights included."""

from __future__ import annotations

import hashlib

def state_digest(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode())
        digest.update(value.detach().contiguous().numpy().tobytes())
    return digest.hexdigest()
