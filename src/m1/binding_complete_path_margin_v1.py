"""Portable inference definitions; no research datasets or trained weights included."""

from __future__ import annotations

from pathlib import Path

SOURCE = Path('src/m1/binding_complete_path_margin_v1.py')

OBJECTIVE = 'COMPACT_FP32_GOLD_MARGINAL_NLL_PLUS_FIXED_B8_F64_BEST_GOLD_ROLE_HAMMING_MARGIN_COEFFICIENT1'
