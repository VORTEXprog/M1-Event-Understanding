"""Complete component inference on invented, untrained FP32 inputs.

This runs genuine model owners and decoders. It does not encode text and its
random features/weights do not provide meaningful language understanding.
Supply trained states and a compatible feature provider for a real experiment.
"""
import json

import torch

from m1.canonical_units import segment_text
from m1.factories import experimental_components
from m1.runtime import FeatureBatch, RawText, predict_features


def main():
    torch.set_num_threads(2)
    generator = torch.Generator(device="cpu").manual_seed(101)
    rows = (
        RawText("invented/en", "Participant_A opened a container.", "en"),
        RawText("invented/ru", "Участник_Б закрыл контейнер.", "ru"),
    )
    unprimed, prefixes = [], []
    for row in rows:
        count = len(segment_text(row.text))
        # Independent complete views: one unprimed and seven type-conditioned.
        unprimed.append(torch.randn((count, 2048), generator=generator))
        prefixes.append(tuple(torch.randn((count, 2048), generator=generator) for _ in range(7)))
    features = FeatureBatch(rows, tuple(unprimed), tuple(prefixes),
        {"invented_only": True, "pretrained": False, "feature_dimension": 2048})
    models = experimental_components(seed=7, binding_kind="full_context83")
    result = predict_features(features, models)
    result.validate()
    print(json.dumps({
        "model_status": "UNTRAINED_EXPERIMENTAL_FIXTURE",
        "features_are_synthetic": True,
        "semantic_accuracy_claimed": False,
        "qualification_claimed": False,
        "rows": len(rows),
        "type_score_shape": list(result.type_bank.logits.shape),
        "binding_saved_arrays_per_row": [len(bank) for _, bank in result.saved_binding_rows],
        "predicted_factors": result.factors,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
