"""Feature-interface demonstration, not a pretrained language understanding demo.

Both features and model states are invented CPU FP32 values. Predicted spans
have no semantic accuracy claim. Real use requires compatible independently
supplied features and trained states. No model download or network is used.
"""
import hashlib
import json

import torch

from m1 import semantic_prefix_conditional_identity_contract_v1 as contract
from m1 import semantic_prefix_conditional_identity_native_v1 as type_readout
from m1 import span_context_assertion_null_v1 as span_geometry
from m1 import span_relative_context_assertion_null_v1 as localization
from m1.canonical_units import segment_text


def main():
    torch.set_num_threads(2)
    generator = torch.Generator(device="cpu").manual_seed(41)
    texts = (
        ("en", "Participant_A opened the container."),
        ("ru", "Участник_Б закрыл контейнер."),
    )
    rows = tuple(
        contract.RawRow(f"invented/{index}", text, language,
            tuple((unit.char_start, unit.char_end) for unit in segment_text(text)))
        for index, (language, text) in enumerate(texts)
    )
    sample_digest = contract.raw_sample_digest(rows)
    records = []
    for row in rows:
        # Every type candidate has its own complete, independent feature view.
        # These values do not encode the text or implement a model provider.
        for candidate in range(7):
            features = torch.randn((len(row.units), 2048), generator=generator).contiguous()
            item_id = contract.candidate_item_id(sample_digest, row.sentence_id, candidate)
            records.append(contract.PrefixUnitRecord(
                item_id, row.sentence_id, candidate, contract.FORWARD_BASIS_SHA256,
                row.units, item_id, hashlib.sha256(features.numpy().tobytes()).hexdigest(), features,
            ))
    records = tuple(records)
    inputs = type_readout.authenticate_inputs(
        rows=rows, records=records, raw_sample_sha256=sample_digest,
        input_sha256=contract.input_commitment(rows, records), fixture_only_not_admissible=True,
    )
    type_state = {
        name: (torch.randn(shape, generator=generator) * .02).contiguous()
        for name, shape in contract.SHAPES["shared_bigru16"].items()
    }
    type_owner = type_readout.owner_from_state(type_state, "shared_bigru16", fixture_only_not_admissible=True)
    local_state = {
        name: (torch.randn(shape, generator=generator) * .01).contiguous()
        for name, shape in localization.STATE_SHAPES.items()
    }
    localizer = localization.RelativeSpanContextAssertionNullHead(local_state)
    typed = type_readout.score_complete_dense(inputs, type_owner)
    output = []
    for index, row in enumerate(rows):
        begin, end = typed.row_pair_offsets[index:index + 2]
        evidence = span_geometry.align_frozen_type_scores(typed.scores[begin:end], typed.pairs[begin:end], len(row.units))
        all_seven_views = torch.stack([record.fp32_units for record in records[index * 7:(index + 1) * 7]])
        scores = localizer(all_seven_views, evidence)
        decisions = localization.predict(scores)
        events = []
        for start, stop, kind in decisions["spans"]:
            left, right = row.units[start][0], row.units[stop - 1][1]
            events.append({"char_start": left, "char_end": right, "surface": row.text[left:right], "type": contract.TYPES[kind]})
        output.append({"sentence_id": row.sentence_id, "complete_candidate_spans": len(scores.pair_indices), "events": events})
    print(json.dumps({"fixture_only": True, "pretrained": False, "accuracy_claimed": False, "rows": output}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
