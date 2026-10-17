"""Portable inference definitions; no research datasets or trained weights included."""

from __future__ import annotations

import m1.presence_sequence_key_argument_v1 as base

NATIVE_KEYS = base.NATIVE_KEYS

POINTER_NEW_KEYS = ('fresh_start_query.weight', 'fresh_end_query.weight')

def _digest(value):
    if (type(value) is not str or len(value)!=64 or value=='0'*64
            or any(c not in '0123456789abcdef' for c in value)):
        raise ValueError('authenticated frozen native owner digest required')
    return value

def _shapes(encoder,hidden):
    return {**base._shapes(encoder,hidden),**{name:(hidden,2*hidden) for name in POINTER_NEW_KEYS}}
