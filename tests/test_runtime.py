"""Genuine composed owners and exact decoders on invented feature matrices."""
from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from safetensors.torch import load_file, save_file

from m1 import checkpoint, factories, runtime
from m1.canonical_units import segment_text
from m1.tensor_state import state_dict_sha256


OWNER_NAMES = ("type14", "localizer", "original66", "heads", "binding")


def states_of(models):
    return {name: {key: value.detach().clone().contiguous()
        for key, value in getattr(models, name).state_dict().items()} for name in OWNER_NAMES}


def fixture(binding_kind="full_context83", *, last_text="alpha beta gamma"):
    original = factories.experimental_components(seed=73, binding_kind=binding_kind)
    states = states_of(original)
    states["type14"]["type_bias"][6] += 8.
    # A deliberately transparent numerical fixture: positive/negative feature
    # coordinate zero determines eventhood. It is not a learned text rule.
    for value in states["localizer"].values():
        value.zero_()
    states["localizer"]["span_weight"][0, 0] = -1.
    states["localizer"]["joint_weight"][0, 0] = 1.
    states["localizer"]["decision_weight"][0, 0] = 10.
    states["localizer"]["decision_bias"][0] = -1.
    models = factories.components_from_states(states, binding_kind=binding_kind)
    rows = (
        runtime.RawText("invented/zero", "quiet still", "en"),
        runtime.RawText("invented/one", "active", "en"),
        runtime.RawText("invented/three", last_text, "ru"),
    )
    unprimed, prefixes = [], []
    for index, row in enumerate(rows):
        width = len(segment_text(row.text))
        base = (torch.arange(width * 2048, dtype=torch.float32).reshape(width, 2048) % 17) / 17
        base[:, 0] = 2.
        unprimed.append(base.clone().contiguous())
        views = []
        for candidate in range(7):
            value = base.clone()
            value[:, 0] = 2. if index == 0 else -2.
            value[:, 1] += candidate * .01
            views.append(value.contiguous())
        prefixes.append(tuple(views))
    features = runtime.FeatureBatch(rows, tuple(unprimed), tuple(prefixes),
        {"invented_only": True, "pretrained": False, "feature_dimension": 2048})
    return features, models, states


@pytest.mark.parametrize("binding_kind", ("native74", "full_context83"))
def test_composed_real_owners_keep_e0_e1_e3_all_spans_and_global_rng(binding_kind):
    features, models, states = fixture(binding_kind)
    before = torch.get_rng_state().clone()
    original_states = states_of(models)
    original_features = [value.clone() for value in features.unprimed]
    result = runtime.predict_features(features, models)
    assert [len(row["events"]) for row in result.factors] == [0, 1, 3]
    assert [row["sentence_id"] for row in result.factors] == [row.sentence_id for row in features.rows]
    assert len(result.local_banks) == 3
    assert [len(bank.pair_indices) for bank in result.local_banks] == [3, 1, 6]
    assert result.type_bank.logits.shape == (10, 7)
    assert len(result.saved_binding_rows) == 3
    for row, factor in zip(features.rows, result.factors, strict=True):
        events = factor["events"]
        for event in events:
            assert 0 <= event["char_start"] < event["char_end"] <= len(row.text)
        for relation in factor["relations"]:
            source, target = relation["source_event_index"], relation["target_event_index"]
            assert 0 <= source < len(events) and 0 <= target < len(events) and source != target
    assert result.factors[-1]["events"][-1]["char_end"] == len(features.rows[-1].text)
    for index, (_, bank) in enumerate(result.saved_binding_rows):
        assert len(bank) == 9
        assert int(bank["event_mask"].sum()) == (0, 1, 3)[index]
    assert len(result.factor_state) == 3
    assert torch.equal(before, torch.get_rng_state())
    assert all(torch.equal(old, new) for old, new in zip(original_features, features.unprimed, strict=True))
    for name, state in original_states.items():
        assert all(torch.equal(value.view(torch.uint8), getattr(models, name).state_dict()[key].view(torch.uint8)) for key, value in state.items())
    result.validate()


@pytest.mark.parametrize("binding_kind", ("native74", "full_context83"))
def test_native_binding_scores_equal_direct_owner_and_physical_nine_array_replay(tmp_path, binding_kind):
    features, models, _ = fixture(binding_kind)
    result = runtime.predict_features(features, models)
    for batch in result.binding_batches:
        with torch.no_grad():
            direct = models.binding(batch.inputs)
        for name, expected in vars(direct).items():
            actual = getattr(batch.scores, name)
            if expected is None:
                assert actual is None
            else:
                assert torch.equal(expected.view(torch.uint8), actual.view(torch.uint8))
    reopened = []
    for index, (manifest, bank) in enumerate(result.saved_binding_rows):
        file = tmp_path / f"row_{index}.safetensors"
        save_file(bank, str(file))
        reopened.append((deepcopy(manifest), load_file(str(file))))
    factor_file = tmp_path / "factors.safetensors"
    save_file(result.factor_state, str(factor_file))
    factor_state = load_file(str(factor_file))
    assert all(torch.equal(value.view(torch.uint8), factor_state[name].view(torch.uint8)) for name, value in result.factor_state.items())
    proof = runtime.replay_saved_binding(result, tuple(reopened))
    assert proof["events"] == 4
    reopened[-1][1]["pair_vectors"][0, 0, 0] += .1
    with pytest.raises(ValueError):
        runtime.replay_saved_binding(result, tuple(reopened))


def test_two_supplied_restores_are_frozen_private_and_preserve_original_state_and_rng():
    _, original, states = fixture()
    before = torch.get_rng_state().clone()
    first = factories.components_from_states(states)
    second = factories.components_from_states(states)
    owner_pointers = []
    for model in (original, first, second):
        assert all(not getattr(model, name).training for name in OWNER_NAMES)
        assert all(not parameter.requires_grad and parameter.grad is None
            for name in OWNER_NAMES for parameter in getattr(model, name).parameters())
        owner_pointers.extend(parameter.untyped_storage().data_ptr()
            for name in OWNER_NAMES for parameter in getattr(model, name).parameters())
        for name, state in states.items():
            assert all(torch.equal(value.view(torch.uint8), getattr(model, name).state_dict()[key].view(torch.uint8)) for key, value in state.items())
    assert len(owner_pointers) == len(set(owner_pointers))
    supplied_pointers = {value.untyped_storage().data_ptr() for state in states.values() for value in state.values()}
    assert not supplied_pointers.intersection(owner_pointers)
    assert torch.equal(before, torch.get_rng_state())


@pytest.mark.parametrize("defect", ("missing_candidate", "wrong_width", "dtype", "nan", "raw_prefix_alias", "duplicate_sid"))
def test_bad_complete_feature_inventory_geometry_dtype_or_alias_is_rejected(defect):
    features, models, _ = fixture()
    prefixes = [list(row) for row in features.prefixes]
    if defect == "missing_candidate":
        prefixes[-1].pop()
    elif defect == "wrong_width":
        prefixes[-1][-1] = prefixes[-1][-1][:-1].clone()
    elif defect == "dtype":
        prefixes[-1][-1] = prefixes[-1][-1].double()
    elif defect == "nan":
        prefixes[-1][-1] = prefixes[-1][-1].clone()
        prefixes[-1][-1][-1, -1] = float("nan")
    elif defect == "raw_prefix_alias":
        prefixes[-1][-1] = features.unprimed[-1]
    with pytest.raises(ValueError):
        if defect == "duplicate_sid":
            features = replace(features, rows=(features.rows[0], features.rows[0], features.rows[2]))
        bad = replace(features, prefixes=tuple(tuple(row) for row in prefixes))
        runtime.predict_features(bad, models)


@pytest.mark.parametrize("defect", ("raw", "prefix", "scores", "factors", "owner"))
def test_retained_input_score_factor_and_owner_mutations_cannot_be_hidden(defect):
    features, models, _ = fixture()
    result = runtime.predict_features(features, models)
    if defect == "raw":
        features.unprimed[-1].data[-1, -1] += .1
    elif defect == "prefix":
        features.prefixes[-1][-1].data[-1, -1] += .1
    elif defect == "scores":
        result.saved_binding_rows[-1][1]["pair_vectors"].data[0, 0, 0] += .1
    elif defect == "factors":
        result.factors[-1]["events"].pop()
    else:
        next(models.binding.parameters()).data.flatten()[0] += .1
    with pytest.raises(ValueError):
        result.validate()


@pytest.mark.parametrize("group", OWNER_NAMES)
def test_missing_owner_tensor_is_rejected_by_complete_factory(group):
    _, _, states = fixture()
    states[group].pop(next(iter(states[group])))
    with pytest.raises(ValueError):
        factories.components_from_states(states)


def test_complete_bundle_restores_real_owners_and_identical_inference(tmp_path):
    features, models, states = fixture()
    expected = runtime.predict_features(features, models)
    metadata = {"invented_only": True, "binding_kind": "full_context83", "type_architecture": "shared_bigru16"}
    directory = tmp_path / "complete_components"
    receipt = checkpoint.save_state_bundle(directory, states, metadata=metadata)
    loaded = checkpoint.load_state_bundle(directory,
        expected_archive_sha256=receipt["archive_sha256"], expected_manifest_sha256=receipt["manifest_sha256"])
    restored = factories.components_from_states(loaded.states,
        binding_kind=loaded.metadata["binding_kind"], type_architecture=loaded.metadata["type_architecture"])
    actual = runtime.predict_features(features, restored)
    assert actual.factors == expected.factors
    assert torch.equal(actual.type_bank.logits.view(torch.uint8), expected.type_bank.logits.view(torch.uint8))
    for (_, left), (_, right) in zip(actual.saved_binding_rows, expected.saved_binding_rows, strict=True):
        assert set(left) == set(right)
        assert all(torch.equal(value.view(torch.uint8), right[name].view(torch.uint8)) for name, value in left.items())
    actual.validate()
    expected.validate()


def test_complete_factory_restore_in_fresh_process_without_test_fixture_imports(tmp_path):
    _, models, states = fixture()
    directory = tmp_path / "complete_components"
    receipt = checkpoint.save_state_bundle(directory, states, metadata={"invented_only": True})
    expected = {name: state_dict_sha256(dict(getattr(models, name).state_dict())) for name in OWNER_NAMES}
    script = '''import json,sys,torch
from m1.checkpoint import load_state_bundle
from m1.factories import components_from_states
from m1.tensor_state import state_dict_sha256
torch.set_num_threads(2)
before=torch.get_rng_state().clone()
loaded=load_state_bundle(sys.argv[1],expected_archive_sha256=sys.argv[2],expected_manifest_sha256=sys.argv[3])
models=components_from_states(loaded.states)
names=("type14","localizer","original66","heads","binding")
assert torch.equal(before,torch.get_rng_state())
assert all(not getattr(models,name).training for name in names)
print(json.dumps({name:state_dict_sha256(dict(getattr(models,name).state_dict())) for name in names}))
'''
    environment = dict(os.environ)
    environment.update(PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"),
        CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
    completed = subprocess.run([sys.executable, "-c", script, str(directory),
        receipt["archive_sha256"], receipt["manifest_sha256"]],
        cwd=tmp_path, env=environment, check=True, capture_output=True, text=True, timeout=60)
    assert json.loads(completed.stdout) == expected


def test_raw_provider_contract_captures_complete_features_once_before_real_inference():
    features, models, _ = fixture()
    calls = []

    class InventedProvider:
        def capture(self, rows):
            calls.append(rows)
            return features

    expected = runtime.predict_features(features, models)
    actual = runtime.predict_raw(features.rows, models=models, provider=InventedProvider())
    assert calls == [features.rows]
    assert actual.factors == expected.factors
    assert torch.equal(actual.type_bank.logits.view(torch.uint8), expected.type_bank.logits.view(torch.uint8))
    assert actual.provenance["feature_origin_verified"] is False
    assert actual.provenance["full_M1_qualification_claimed"] is False
    actual.validate()


@pytest.mark.parametrize("defect", ("annotation", "wrong_owner"))
def test_bad_raw_annotation_or_owner_is_rejected_before_provider_capture(defect):
    features, models, _ = fixture()
    calls = []

    class InventedProvider:
        def capture(self, rows):
            calls.append(rows)
            return features

    if defect == "annotation":
        object.__setattr__(features.rows[0], "Gold", [])
    else:
        next(models.binding.parameters()).data.flatten()[0] += .1
    with pytest.raises(ValueError):
        runtime.predict_raw(features.rows, models=models, provider=InventedProvider())
    assert calls == []


def test_provider_cannot_relabel_or_reorder_the_original_raw_inventory():
    features, models, _ = fixture()
    bad = replace(features, rows=(replace(features.rows[0], sentence_id="invented/other"), *features.rows[1:]))

    class InventedProvider:
        def capture(self, rows):
            return bad

    with pytest.raises(ValueError):
        runtime.predict_raw(features.rows, models=models, provider=InventedProvider())


@pytest.mark.parametrize("owner_name,method_name", (
    ("localizer", "validate"), ("binding", "_validate_topology"),
))
def test_instance_method_shadows_cannot_bypass_owner_validation(owner_name, method_name):
    features, models, _ = fixture()
    setattr(getattr(models, owner_name), method_name, lambda: None)
    with pytest.raises(ValueError):
        models.validate()
    with pytest.raises(ValueError):
        runtime.predict_features(features, models)
