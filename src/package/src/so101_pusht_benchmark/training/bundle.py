"""Minimal SafeTensors encoder/decoder with strict policy-state validation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import json
import struct
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray
import torch

from .artifacts import ArtifactError, ArtifactIndex, sha256_file
from .identity import BundleIdentity

_DTYPE_TO_NAME = {
    torch.bool: "BOOL",
    torch.uint8: "U8",
    torch.int8: "I8",
    torch.int16: "I16",
    torch.int32: "I32",
    torch.int64: "I64",
    torch.float16: "F16",
    torch.bfloat16: "BF16",
    torch.float32: "F32",
    torch.float64: "F64",
}
_NAME_TO_DTYPE = {name: dtype for dtype, name in _DTYPE_TO_NAME.items()}


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    value = tensor.detach().cpu().clone(memory_format=torch.contiguous_format)
    to_numpy = cast("Callable[[], NDArray[np.generic]]", value.reshape(-1).view(torch.uint8).numpy)
    array = to_numpy()
    return array.tobytes()


def save_bundle(path: Path, tensors: dict[str, torch.Tensor]) -> None:
    """Write the documented, non-executable SafeTensors wire format."""
    if not tensors or any(not key or key == "__metadata__" for key in tensors):
        raise ArtifactError("bundle tensor keys are invalid")
    header: dict[str, object] = {
        "__metadata__": {
            "format": "pt",
            "deployment_scope": "simulation_only",
        }
    }
    payload = bytearray()
    for key in sorted(tensors):
        tensor = tensors[key]
        dtype = _DTYPE_TO_NAME.get(tensor.dtype)
        if dtype is None:
            raise ArtifactError(f"unsupported tensor dtype: {tensor.dtype}")
        start = len(payload)
        payload.extend(_tensor_bytes(tensor))
        header[key] = {
            "dtype": dtype,
            "shape": list(tensor.shape),
            "data_offsets": [start, len(payload)],
        }
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    encoded += b" " * ((8 - len(encoded) % 8) % 8)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.write(struct.pack("<Q", len(encoded)))
        stream.write(encoded)
        stream.write(payload)


def _decode(path: Path) -> dict[str, torch.Tensor]:
    raw = path.read_bytes()
    if len(raw) < 10:
        raise ArtifactError("truncated SafeTensors bundle")
    header_size = struct.unpack("<Q", raw[:8])[0]
    if header_size > len(raw) - 8 or header_size > 100_000_000:
        raise ArtifactError("invalid SafeTensors header size")
    try:
        parsed: object = json.loads(raw[8 : 8 + header_size].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactError("invalid SafeTensors header") from exc
    if not isinstance(parsed, dict):
        raise ArtifactError("SafeTensors header must be an object")
    header = cast("dict[str, object]", parsed)
    payload = raw[8 + header_size :]
    tensors: dict[str, torch.Tensor] = {}
    occupied: list[tuple[int, int]] = []
    for key, raw_spec in header.items():
        if key == "__metadata__":
            continue
        if not isinstance(raw_spec, dict):
            raise ArtifactError("invalid SafeTensors tensor record")
        spec = cast("dict[str, object]", raw_spec)
        dtype_name, shape, offsets = spec.get("dtype"), spec.get("shape"), spec.get("data_offsets")
        if not isinstance(shape, list) or not isinstance(offsets, list):
            raise ArtifactError("invalid SafeTensors tensor metadata")
        raw_shape = cast("list[object]", shape)
        raw_offsets = cast("list[object]", offsets)
        if (
            not isinstance(dtype_name, str)
            or dtype_name not in _NAME_TO_DTYPE
            or not all(isinstance(item, int) and item >= 0 for item in raw_shape)
            or len(raw_offsets) != 2
            or not all(isinstance(item, int) for item in raw_offsets)
        ):
            raise ArtifactError("invalid SafeTensors tensor metadata")
        typed_offsets = cast("list[int]", raw_offsets)
        start, end = typed_offsets
        if start < 0 or end < start or end > len(payload):
            raise ArtifactError("SafeTensors offsets are out of range")
        if any(start < prior_end and prior_start < end for prior_start, prior_end in occupied):
            raise ArtifactError("SafeTensors tensor regions overlap")
        occupied.append((start, end))
        dtype = _NAME_TO_DTYPE[dtype_name]
        typed_shape = cast("list[int]", shape)
        count = 1
        for dimension in typed_shape:
            count *= dimension
        item_size = torch.empty((), dtype=dtype).element_size()
        if end - start != count * item_size:
            raise ArtifactError("SafeTensors tensor byte length mismatch")
        if count == 0:
            tensors[key] = torch.empty(typed_shape, dtype=dtype)
        else:
            storage = bytearray(payload[start:end])
            tensors[key] = (
                torch.frombuffer(storage, dtype=dtype, count=count).clone().reshape(typed_shape)
            )
    if not tensors:
        raise ArtifactError("SafeTensors bundle has no tensors")
    return tensors


@dataclass(frozen=True, slots=True)
class BundleExpectation:
    identity: BundleIdentity
    checkpoint_sha256: str


def load_bundle(
    path: Path,
    expected: dict[str, torch.Tensor],
    *,
    index: ArtifactIndex,
    artifact_id: str,
    expectation: BundleExpectation,
) -> dict[str, torch.Tensor]:
    """Verify digest, identity, tensors, shapes, and dtypes before policy loading."""
    anchored = index.verify(artifact_id, "bundle")
    if anchored != path.resolve():
        raise ArtifactError("bundle argument does not match anchored path")
    record = index.record(artifact_id)
    anchored_identity = BundleIdentity.from_dict(record.get("identity"))
    if anchored_identity != expectation.identity:
        raise ArtifactError("trusted bundle identity mismatch")
    manifest_path = index.verify(artifact_id, "manifest")
    try:
        manifest_raw: object = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError("bundle manifest is invalid") from exc
    if not isinstance(manifest_raw, dict):
        raise ArtifactError("bundle manifest is invalid")
    manifest = cast("dict[str, object]", manifest_raw)
    if (
        manifest.get("schema") != 1
        or BundleIdentity.from_dict(manifest.get("identity")) != expectation.identity
    ):
        raise ArtifactError("trusted bundle manifest identity mismatch")
    checkpoint = index.verify(artifact_id, "checkpoint")
    config = index.verify(artifact_id, "config")
    checkpoint_digest = sha256_file(checkpoint)
    if checkpoint_digest != expectation.checkpoint_sha256:
        raise ArtifactError("trusted bundle expected checkpoint mismatch")
    if manifest.get("source_checkpoint_sha256") != checkpoint_digest or manifest.get(
        "resolved_config_sha256"
    ) != sha256_file(config):
        raise ArtifactError("trusted bundle checkpoint/config identity mismatch")
    loaded = _decode(anchored)
    if set(loaded) != set(expected):
        raise ArtifactError("bundle keys do not exactly match trusted policy")
    for key, tensor in loaded.items():
        reference = expected[key]
        if tensor.shape != reference.shape:
            raise ArtifactError(f"bundle tensor shape mismatch: {key}")
        if tensor.dtype != reference.dtype:
            raise ArtifactError(f"bundle tensor dtype mismatch: {key}")
    return loaded
