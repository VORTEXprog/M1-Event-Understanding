"""Portable inference definitions; no research datasets or trained weights included."""

from __future__ import annotations

PARAMETER_KEYS, PARAMETER_COUNT = 24, 693125

STATE_SHAPES = {
    'input_embedding.weight': (259, 64),
    'encoder.weight_ih_l0': (384, 64), 'encoder.weight_hh_l0': (384, 128),
    'encoder.bias_ih_l0': (384,), 'encoder.bias_hh_l0': (384,),
    'encoder.weight_ih_l0_reverse': (384, 64), 'encoder.weight_hh_l0_reverse': (384, 128),
    'encoder.bias_ih_l0_reverse': (384,), 'encoder.bias_hh_l0_reverse': (384,),
    'prediction_embedding.weight': (259, 64),
    'prediction.weight_ih_l0': (768, 64), 'prediction.weight_hh_l0': (768, 256),
    'prediction.bias_ih_l0': (768,), 'prediction.bias_hh_l0': (768,),
    'initial_prediction.weight': (256, 256), 'initial_prediction.bias': (256,),
    'action_join.weight': (128, 512), 'action_join.bias': (128,),
    'action_output.weight': (5, 128), 'action_output.bias': (5,),
    'source_byte.weight': (256, 256), 'source_byte.bias': (256,),
    'prediction_byte.weight': (256, 256), 'prediction_byte.bias': (256,),
}
