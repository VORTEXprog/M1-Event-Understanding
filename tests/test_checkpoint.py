"""Portable data-only checkpoints: external digests, exact bits and safe paths."""
from copy import deepcopy
import hashlib
import json

import pytest
import torch

from m1 import checkpoint


def invented_states():
    return {
        "localizer": {
            "weight": (torch.arange(24, dtype=torch.float32).reshape(3, 8) / 17).contiguous(),
            "bias": torch.tensor([-0., 0., .25], dtype=torch.float32),
        },
        "binding": {"query.weight": torch.eye(4, dtype=torch.float32)},
    }


def save(directory, states=None, metadata=None):
    return checkpoint.save_state_bundle(
        directory, invented_states() if states is None else states,
        metadata={"invented_only": True, "pretrained": False} if metadata is None else metadata,
    )


def load(directory, receipt, **kwargs):
    return checkpoint.load_state_bundle(directory,
        expected_archive_sha256=receipt["archive_sha256"],
        expected_manifest_sha256=receipt["manifest_sha256"], **kwargs)


def test_roundtrip_two_private_loads_exact_signed_zero_bits_and_rng(tmp_path):
    supplied = invented_states()
    metadata = {"invented_only": True, "pretrained": False, "nested": {"values": [1, 2]}}
    expected = {group: {name: value.clone() for name, value in state.items()} for group, state in supplied.items()}
    expected_metadata = deepcopy(metadata)
    rng = torch.get_rng_state().clone()
    receipt = save(tmp_path / "bundle", supplied, metadata)
    supplied["localizer"]["weight"].add_(5)
    metadata["nested"]["values"].append(3)
    first = load(tmp_path / "bundle", receipt)
    second = load(tmp_path / "bundle", receipt)
    assert first.metadata == second.metadata == expected_metadata
    assert first.archive_sha256 == second.archive_sha256 == receipt["archive_sha256"]
    assert first.manifest_sha256 == second.manifest_sha256 == receipt["manifest_sha256"]
    assert set(first.states) == set(second.states) == set(expected)
    pointers = []
    for group, state in expected.items():
        assert set(first.states[group]) == set(second.states[group]) == set(state)
        for name, value in state.items():
            left, right = first.states[group][name], second.states[group][name]
            assert left.dtype == right.dtype == torch.float32
            assert left.device.type == right.device.type == "cpu"
            assert not left.requires_grad and not right.requires_grad
            assert not left.is_inference() and not right.is_inference()
            assert left.is_contiguous() and right.is_contiguous()
            assert torch.equal(left.view(torch.uint8), value.view(torch.uint8))
            assert torch.equal(right.view(torch.uint8), value.view(torch.uint8))
            pointers.extend((left.untyped_storage().data_ptr(), right.untyped_storage().data_ptr()))
    assert len(pointers) == len(set(pointers))
    assert torch.signbit(first.states["localizer"]["bias"][0])
    first.states["binding"]["query.weight"].add_(10)
    assert torch.equal(second.states["binding"]["query.weight"], expected["binding"]["query.weight"])
    assert torch.equal(rng, torch.get_rng_state())


def test_bundle_is_create_only_and_a_failed_overwrite_keeps_original_bytes(tmp_path):
    directory = tmp_path / "bundle"
    receipt = save(directory)
    before = {file.name: file.read_bytes() for file in directory.iterdir() if file.is_file()}
    with pytest.raises((ValueError, FileExistsError)):
        save(directory)
    after = {file.name: file.read_bytes() for file in directory.iterdir() if file.is_file()}
    assert after == before
    assert load(directory, receipt).metadata["invented_only"] is True


@pytest.mark.parametrize("field", ("archive_sha256", "manifest_sha256"))
def test_wrong_independent_expected_hash_is_rejected(tmp_path, field):
    directory = tmp_path / "bundle"
    receipt = save(directory)
    bad = {**receipt, field: "f" * 64}
    with pytest.raises(ValueError):
        load(directory, bad)


@pytest.mark.parametrize("kind", ("archive", "manifest"))
def test_changed_physical_bytes_cannot_authenticate_themselves(tmp_path, kind):
    directory = tmp_path / "bundle"
    receipt = save(directory)
    candidates = [file for file in directory.iterdir() if file.is_file() and (
        file.suffix == ".safetensors" if kind == "archive" else file.suffix == ".json")]
    assert len(candidates) == 1
    file = candidates[0]
    file.write_bytes(file.read_bytes() + b"changed")
    with pytest.raises(ValueError):
        load(directory, receipt)


def test_archive_size_limit_is_checked_for_complete_physical_archive(tmp_path):
    directory = tmp_path / "bundle"
    receipt = save(directory)
    archive = next(directory.glob("*.safetensors"))
    with pytest.raises(ValueError):
        load(directory, receipt, max_archive_bytes=archive.stat().st_size - 1)


def test_symlinked_bundle_directory_and_physical_leaves_are_rejected(tmp_path):
    directory = tmp_path / "bundle"
    receipt = save(directory)
    alias = tmp_path / "alias"
    try:
        alias.symlink_to(directory, target_is_directory=True)
    except OSError as e:
        import platform
        import pytest
        if platform.system() == "Windows":
            pytest.skip(f"Windows requires admin privileges to create symlinks: {e}")
        raise
    with pytest.raises(ValueError):
        load(alias, receipt)
    archive = next(directory.glob("*.safetensors"))
    moved = tmp_path / "archive.safetensors"
    archive.replace(moved)
    archive.symlink_to(moved)
    with pytest.raises(ValueError):
        load(directory, receipt)


@pytest.mark.parametrize("group", ("../escaped", "/absolute", "nested/path", "", ".", ".."))
def test_unsafe_group_names_are_rejected_without_creating_outside_files(tmp_path, group):
    with pytest.raises(ValueError):
        save(tmp_path / "bundle", {group: {"weight": torch.zeros(1)}})
    assert not (tmp_path / "escaped").exists()


@pytest.mark.parametrize("defect", ("nan", "dtype", "gradient", "inference", "metadata_nan", "metadata_object"))
def test_only_finite_detached_ordinary_fp32_and_json_metadata_are_admitted(tmp_path, defect):
    states, metadata = invented_states(), {"invented_only": True}
    if defect == "nan":
        states["localizer"]["bias"][0] = float("nan")
    elif defect == "dtype":
        states["localizer"]["bias"] = states["localizer"]["bias"].double()
    elif defect == "gradient":
        states["localizer"]["bias"].requires_grad_(True)
    elif defect == "inference":
        with torch.inference_mode():
            states["localizer"]["bias"] = states["localizer"]["bias"].clone()
    elif defect == "metadata_nan":
        metadata["bad"] = float("nan")
    else:
        metadata["bad"] = object()
    with pytest.raises((ValueError, TypeError)):
        save(tmp_path / "bundle", states, metadata)
