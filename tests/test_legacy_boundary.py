"""Genuine component predictions retain explicit legacy capacity failures."""
import pytest
import torch

from m1 import factories, runtime
from m1 import context_canonicalizer_local_joint_score_v1 as canonical
from m1.slots import assignment_permutation
from test_runtime import fixture


def test_real_e0_e1_e2_components_are_inside_explicit_legacy_domain():
    features, models, _ = fixture(last_text="alpha beta")
    result = runtime.predict_features(features, models)
    assert [len(row["events"]) for row in result.factors] == [0, 1, 2]
    report = runtime.require_legacy_compatible(result)
    assert report["compatible_with_legacy_boundaries"]
    assert report["consumer_capacity_failures"] == report["CAN_domain_failures"] == report["text_domain_failures"] == []
    assert report["legacy_dense_basis_parity_claimed"] is report["qualification_claimed"] is False


def test_real_e3_components_remain_complete_when_legacy_boundary_rejects():
    features, models, _ = fixture()
    result = runtime.predict_features(features, models)
    report = runtime.legacy_domain_report(result)
    assert report["consumer_capacity_failures"] == [{"sentence_id": "invented/three", "predicted_events": 3}]
    assert not report["compatible_with_legacy_boundaries"]
    with pytest.raises(runtime.LegacyDomainError) as caught:
        runtime.require_legacy_compatible(result)
    assert caught.value.component_result is result and caught.value.report == report
    assert len(result.factors[-1]["events"]) == 3
    assert int(result.saved_binding_rows[-1][1]["event_mask"].sum()) == 3
    result.validate()


def full_mention_case():
    _, _, states = fixture()
    # For one-unit sentences this genuine native presence head makes the sole
    # full-span candidate preferable to NULL for every role. Joint decoding and
    # all other score paths remain unchanged.
    keys = [key for key in states["binding"] if key.endswith("presence_output.bias")]
    assert len(keys) == 1
    states["binding"][keys[0]].copy_(torch.tensor([-20., 20.]))
    models = factories.components_from_states(states)
    texts = ("x" * 96, "я" * 49 + "x", "z" * 4096, "z" * 4097)
    rows = tuple(runtime.RawText(f"invented/bytes/{index}", text, "ru" if index == 1 else "en") for index, text in enumerate(texts))
    unprimed, prefixes = [], []
    for row in rows:
        base = (torch.arange(2048, dtype=torch.float32).reshape(1, 2048) % 17) / 17
        base[:, 0] = 2.
        unprimed.append(base.clone())
        views = []
        for candidate in range(7):
            value = base.clone()
            value[:, 0] = -2.
            value[:, 1] += candidate * .01
            views.append(value)
        prefixes.append(tuple(views))
    features = runtime.FeatureBatch(rows, tuple(unprimed), tuple(prefixes), {"invented_only": True})
    return runtime.predict_features(features, models)


def test_actual_96_99_byte_mentions_and_4096_4097_byte_texts_are_never_clipped():
    result = full_mention_case()
    assert [len(row["events"]) for row in result.factors] == [1, 1, 1, 1]
    for source, row in zip(result.features.rows, result.factors, strict=True):
        roles = row["events"][0]["roles"]
        assert roles and all(role["surface"] == source.text for role in roles)
        assert all(role["char_start"] == 0 and role["char_end"] == len(source.text) for role in roles)
    report = runtime.legacy_domain_report(result)
    assert not report["consumer_capacity_failures"]
    assert {item["UTF8_bytes"] for item in report["CAN_domain_failures"]} == {99, 4096, 4097}
    assert all(item["sentence_id"] != "invented/bytes/0" for item in report["CAN_domain_failures"])
    assert report["text_domain_failures"] == [{"sentence_id": "invented/bytes/3", "UTF8_bytes": 4097}]
    with pytest.raises(runtime.LegacyDomainError) as caught:
        runtime.require_legacy_compatible(result)
    assert caught.value.component_result is result
    assert len(result.factors) == 4 and len(result.features.rows[-1].text) == 4097
    result.validate()


def test_real_canonicalizer_28_two_private_restores_and_decode_parity_without_rng():
    supplied = {name: torch.zeros(shape, dtype=torch.float32) for name, shape in canonical.STATE_SHAPES.items()}
    before = torch.get_rng_state().clone()
    first, second = factories.canonicalizer_from_state(supplied), factories.canonicalizer_from_state(supplied)
    assert len(first.state_dict()) == len(second.state_dict()) == 28
    inventories = (supplied, dict(first.state_dict()), dict(second.state_dict()))
    pointers = [value.untyped_storage().data_ptr() for state in inventories for value in state.values()]
    assert len(set(pointers)) == len(pointers)
    surfaces = ("Participant_A", "object_1")
    assert first.decode_surfaces(surfaces) == second.decode_surfaces(surfaces)
    assert torch.equal(before, torch.get_rng_state())
    with pytest.raises(ValueError):
        first.decode_surfaces(("x" * 97,))


def test_real_slots4_supplied_state_mask_geometry_assignment_and_e3_rejection():
    supplied = {"network.0.weight": torch.zeros((64, 2048)), "network.0.bias": torch.zeros(64),
        "network.2.weight": torch.zeros((2, 64)), "network.2.bias": torch.zeros(2)}
    before = torch.get_rng_state().clone()
    model = factories.slots_from_state(supplied)
    assert len(model.state_dict()) == 4 and not model.training
    mask = torch.tensor([[False, False], [False, True], [True, True]])
    scores = model(torch.zeros((3, 2, 2048)), mask)
    assert not scores.any() and assignment_permutation(scores, mask).tolist() == [[0, 1], [1, 0], [0, 1]]
    changed = scores.clone()
    changed[2, 0, 1] = changed[2, 1, 0] = 2.
    assert assignment_permutation(changed, mask)[2].tolist() == [1, 0]
    with pytest.raises(ValueError):
        model(torch.zeros((1, 3, 2048)), torch.ones((1, 3), dtype=torch.bool))
    assert torch.equal(before, torch.get_rng_state())
