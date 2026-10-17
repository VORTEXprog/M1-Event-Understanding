"""Data-only, create-only FP32 state bundles with external SHA-256 commitments.

This codec does not execute model code or reconstruct owners. Restore owners
through their explicit constructors after loading. A manifest is written last;
an interrupted save may leave an incomplete directory, which cannot be loaded
or silently overwritten. Synchronize mutable model state before saving.
"""
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re

import torch
from safetensors.torch import load, save

FORMAT = "M1_FP32_STATE_BUNDLE_v1"
MAX_ARCHIVE_BYTES = 512 << 20
MAX_MANIFEST_BYTES = 1 << 20


def _require(value, message):
    if not value:
        raise ValueError(message)


def _json_value(value, depth=0):
    _require(depth <= 64, "metadata nesting is too deep")
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float:
        _require(math.isfinite(value), "metadata floats must be finite")
        return
    if type(value) is list:
        for item in value:
            _json_value(item, depth + 1)
        return
    if type(value) is dict:
        _require(all(type(key) is str for key in value), "metadata keys must be strings")
        for item in value.values():
            _json_value(item, depth + 1)
        return
    raise TypeError("metadata must contain only ordinary JSON values")


def _encode(value):
    _json_value(value)
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False, allow_nan=False) + "\n").encode("utf8")


def _directory(value):
    directory = Path(value).absolute()
    _require(".." not in directory.parts, "parent traversal is not allowed")
    _require(not any(path.is_symlink() for path in (directory, *directory.parents)),
             "symlinked bundle directories are not allowed")
    return directory


def _read(file, maximum):
    _require(file.is_file() and not file.is_symlink(), "ordinary bundle file required")
    _require(0 < file.stat().st_size <= maximum, "bundle file exceeds its size limit")
    descriptor = os.open(file, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(descriptor, "rb") as stream:
        payload = stream.read(maximum + 1)
    _require(0 < len(payload) <= maximum, "bundle file exceeds its size limit")
    return payload


def _write(file, payload):
    with file.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _sync_directory(directory):
    if hasattr(os, "O_DIRECTORY"):
        descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _digest(value):
    return hashlib.sha256(value).hexdigest()


def _expected(value):
    _require(type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
             "independent lowercase SHA-256 commitment required")


@dataclass(frozen=True)
class LoadedStateBundle:
    states: dict
    metadata: dict
    archive_sha256: str
    manifest_sha256: str


def save_state_bundle(directory, states, *, metadata):
    """Save finite ordinary CPU FP32 tensors, preserving native key order.

    Caller-supplied groups and tensors are copied. No directory is created until
    all values have been validated and serialization has completed.
    """
    directory = _directory(directory)
    _require(type(states) is dict and states, "nonempty named state groups required")
    _require(type(metadata) is dict, "JSON object metadata required")
    metadata = json.loads(_encode(metadata))
    flat, descriptors = {}, []
    with torch.inference_mode(False), torch.no_grad():
        for group, state in states.items():
            _require(type(group) is str and re.fullmatch(r"[a-z][a-z0-9_]*", group),
                     "safe state group name required")
            _require(type(state) is dict and state, "nonempty named tensor state required")
            for name, value in state.items():
                _require(type(name) is str and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]*", name),
                         "safe tensor name required")
                _require(type(value) is torch.Tensor and value.device.type == "cpu"
                         and value.dtype == torch.float32 and value.is_contiguous()
                         and not value.requires_grad and not value.is_inference()
                         and value.numel() > 0 and bool(value.isfinite().all()),
                         "finite detached ordinary contiguous CPU FP32 tensor required")
                flat[group + "::" + name] = value.clone()
                descriptors.append({"group": group, "name": name, "shape": list(value.shape)})
    archive = save(flat)
    _require(0 < len(archive) <= MAX_ARCHIVE_BYTES, "state archive exceeds size limit")
    manifest = _encode({"format": FORMAT, "archive_sha256": _digest(archive),
                        "metadata": metadata, "tensors": descriptors})
    _require(len(manifest) <= MAX_MANIFEST_BYTES, "state manifest exceeds size limit")
    _require(directory.parent.is_dir(), "bundle parent directory must already exist")
    directory.mkdir()  # Exclusive: an existing bundle is never overwritten.
    _write(directory / "weights.safetensors", archive)
    _write(directory / "manifest.json", manifest)
    _sync_directory(directory)
    _sync_directory(directory.parent)
    return {"archive_sha256": _digest(archive), "manifest_sha256": _digest(manifest)}


def load_state_bundle(directory, *, expected_archive_sha256,
                      expected_manifest_sha256, max_archive_bytes=MAX_ARCHIVE_BYTES):
    """Authenticate physical bytes before parsing, then return private tensors.

    Both commitments must come from a trusted channel independent of this
    bundle. A checksum contained in the manifest alone is insufficient.
    """
    _expected(expected_archive_sha256)
    _expected(expected_manifest_sha256)
    _require(type(max_archive_bytes) is int and 0 < max_archive_bytes <= MAX_ARCHIVE_BYTES,
             "bounded archive size limit required")
    directory = _directory(directory)
    _require(directory.is_dir(), "bundle directory required")
    _require({file.name for file in directory.iterdir()} == {"weights.safetensors", "manifest.json"},
             "exact complete bundle inventory required")
    manifest_bytes = _read(directory / "manifest.json", MAX_MANIFEST_BYTES)
    _require(_digest(manifest_bytes) == expected_manifest_sha256, "manifest SHA-256 mismatch")
    archive = _read(directory / "weights.safetensors", max_archive_bytes)
    _require(_digest(archive) == expected_archive_sha256, "archive SHA-256 mismatch")
    try:
        manifest = json.loads(manifest_bytes)
    except (ValueError, UnicodeDecodeError) as error:
        raise ValueError("invalid state manifest") from error
    _require(type(manifest) is dict and set(manifest) == {"format", "archive_sha256", "metadata", "tensors"}
             and manifest["format"] == FORMAT
             and manifest["archive_sha256"] == expected_archive_sha256,
             "closed matching state manifest required")
    _require(type(manifest["metadata"]) is dict, "JSON object metadata required")
    _json_value(manifest["metadata"])
    descriptors = manifest["tensors"]
    _require(type(descriptors) is list and descriptors, "complete tensor descriptor list required")
    keys, shapes = [], {}
    for item in descriptors:
        _require(type(item) is dict and set(item) == {"group", "name", "shape"}, "closed tensor descriptor required")
        group, name, shape = item["group"], item["name"], item["shape"]
        _require(type(group) is str and re.fullmatch(r"[a-z][a-z0-9_]*", group)
                 and type(name) is str and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]*", name)
                 and type(shape) is list and all(type(x) is int and x >= 0 for x in shape),
                 "safe tensor descriptor required")
        key = group + "::" + name
        _require(key not in shapes, "duplicate tensor descriptor")
        keys.append((key, group, name))
        shapes[key] = tuple(shape)
    try:
        with torch.inference_mode(False), torch.no_grad():
            loaded = load(archive)
            _require(set(loaded) == set(shapes), "complete exact archive tensor keys required")
            states = {}
            for key, group, name in keys:
                value = loaded[key]
                _require(value.dtype == torch.float32 and value.device.type == "cpu"
                         and tuple(value.shape) == shapes[key] and value.numel() > 0
                         and bool(value.isfinite().all()), "manifest/archive FP32 tensor mismatch")
                states.setdefault(group, {})[name] = value.contiguous().clone()
    except (RuntimeError, OverflowError) as error:
        raise ValueError("invalid FP32 state archive") from error
    return LoadedStateBundle(states, manifest["metadata"], expected_archive_sha256,
                             expected_manifest_sha256)
