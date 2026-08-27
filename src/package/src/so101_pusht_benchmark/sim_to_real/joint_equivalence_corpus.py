"""Parsing and member-integrity checks for joint-equivalence corpora."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import cast

from .joint_mapping import JOINT_ORDER
from .policy_types import FixtureApprovedSafetyPolicy, ProductionApprovedSafetyPolicy
from .rollout_codes import RolloutCode, RolloutViolation

CORPUS_SCHEMA = "so101-joint-equivalence-corpus-v2"
MEMBER_SCHEMA = "so101-joint-equivalence-member-v2"
SHA_HEX = frozenset("0123456789abcdef")
MAPPING_TOLERANCE = 1e-9
MIN_ISOLATED_DELTA_DEGREES = 5.0
MAX_OTHER_JOINT_DRIFT_DEGREES = 3.0
MIN_DISTINCT_POSE_DELTA_DEGREES = 1.0
VECTOR_SIZE = len(JOINT_ORDER)
MemberDocument = tuple[Mapping[str, object], Mapping[str, object]]
JointEquivalencePolicy = FixtureApprovedSafetyPolicy | ProductionApprovedSafetyPolicy


@dataclass(frozen=True, slots=True)
class JointMember:
    """One integrity-checked physical/simulator pose pair."""

    identifier: str
    split: str
    category: str
    isolated_joint: str | None
    degrees: tuple[float, ...]
    radians: tuple[float, ...]
    measured_xyz: tuple[float, ...]
    physical_timestamp: float
    simulator_timestamp: float


@dataclass(frozen=True, slots=True)
class ParsedJointCorpus:
    """Validated manifest material consumed by affine and FK verification."""

    digest: str
    origin: str
    claimed_order: tuple[str, ...]
    members: tuple[JointMember, ...]
    fit: tuple[JointMember, ...]
    held_out: tuple[JointMember, ...]
    member_hashes: tuple[str, ...]


def unproven(detail: str) -> RolloutViolation:
    """Build the single fail-closed error used by the corpus audit."""
    return RolloutViolation(RolloutCode.EQUIVALENCE_UNPROVEN, detail)


def mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise unproven(f"{label} must be a mapping")
    return cast("Mapping[str, object]", value)


def object_list(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise unproven(f"{label} must be a list")
    return cast("list[object]", value)


def text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise unproven(f"{label} must be non-empty text")
    return value


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise unproven(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise unproven(f"{label} must be finite")
    return result


def _vector(value: object, label: str, size: int = VECTOR_SIZE) -> tuple[float, ...]:
    items = object_list(value, label)
    if len(items) != size:
        raise unproven(f"{label} must contain {size} values")
    return tuple(_number(item, label) for item in items)


def digest(value: object, label: str) -> str:
    result = text(value, label)
    if len(result) != 64 or any(character not in SHA_HEX for character in result):
        raise unproven(f"{label} must be lowercase SHA-256")
    return result


def canonical_digest(document: Mapping[str, object], *, omit: str | None = None) -> str:
    """Hash canonical JSON, optionally excluding one self-declared field."""
    payload = {key: value for key, value in document.items() if key != omit}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _member_path(root: Path, value: object) -> Path:
    relative = Path(text(value, "member path"))
    if relative.is_absolute() or relative.parts[:1] != ("members",) or ".." in relative.parts:
        raise unproven("member path must stay below the corpus members directory")
    path = root / relative
    try:
        resolved = path.resolve(strict=True)
        members_root = (root / "members").resolve(strict=True)
    except OSError as exc:
        raise unproven(f"member evidence is absent: {relative}") from exc
    if resolved.parent != members_root or path.is_symlink() or not resolved.is_file():
        raise unproven("member path escapes the content-addressed corpus")
    return resolved


def _load_member(root: Path, entry: Mapping[str, object]) -> Mapping[str, object]:
    path = _member_path(root, entry.get("path"))
    expected = digest(entry.get("sha256"), "member sha256")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise unproven(f"member hash drift: {path.name}")
    try:
        return mapping(json.loads(raw), "member document")
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise unproven(f"member is not valid UTF-8 JSON: {path.name}") from exc


def _parse_member(
    entry: Mapping[str, object],
    document: Mapping[str, object],
    zero_counts: tuple[float, ...],
    degrees_per_count: tuple[float, ...],
    max_skew: float,
) -> JointMember:
    if document.get("schema") != MEMBER_SCHEMA:
        raise unproven("unsupported member schema")
    identifier = text(entry.get("id"), "member id")
    if text(entry.get("sample_id"), "manifest sample id") != identifier:
        raise unproven("member id/sample id binding drift")
    if document.get("sample_id") != identifier:
        raise unproven("member id/sample id binding drift")
    split = text(entry.get("split"), "member split")
    if split not in {"fit", "held_out"}:
        raise unproven("member split must be fit or held_out")
    category = text(entry.get("category"), "member category")
    if category not in {"baseline", "isolated", "task_plane"}:
        raise unproven("member category is unsupported")
    isolated_value = entry.get("isolated_joint")
    isolated = None if isolated_value is None else text(isolated_value, "isolated joint")
    physical = mapping(document.get("physical"), "physical evidence")
    simulator = mapping(document.get("simulator"), "simulator evidence")
    counts = _vector(physical.get("raw_encoder_counts"), "raw encoder counts")
    degrees = _vector(physical.get("joint_degrees"), "physical joint degrees")
    recomputed = tuple(
        (count - zero) * scale
        for count, zero, scale in zip(counts, zero_counts, degrees_per_count, strict=True)
    )
    if any(
        abs(declared - measured) > MAPPING_TOLERANCE
        for declared, measured in zip(degrees, recomputed, strict=True)
    ):
        raise unproven("physical degrees do not recompute from raw encoder evidence")
    radians = _vector(simulator.get("joint_radians"), "simulator joint radians")
    measured_xyz = _vector(physical.get("measured_tool_xyz_m"), "measured tool xyz", 3)
    physical_time = _number(physical.get("timestamp_s"), "physical timestamp")
    simulator_time = _number(simulator.get("timestamp_s"), "simulator timestamp")
    if abs(physical_time - simulator_time) > max_skew:
        raise unproven("paired member timestamps exceed approved sample skew")
    return JointMember(
        identifier,
        split,
        category,
        isolated,
        degrees,
        radians,
        measured_xyz,
        physical_time,
        simulator_time,
    )


def _inventories(members: Sequence[JointMember]) -> tuple[list[JointMember], list[JointMember]]:
    identifiers = [member.identifier for member in members]
    timestamps = [member.physical_timestamp for member in members]
    if len(set(identifiers)) != len(identifiers):
        raise unproven("member sample ids must be unique")
    if len(set(timestamps)) != len(timestamps) or timestamps != sorted(timestamps):
        raise unproven("physical timestamps must be unique and ordered")
    for index, member in enumerate(members):
        if any(
            max(abs(a - b) for a, b in zip(member.degrees, prior.degrees, strict=True))
            < MIN_DISTINCT_POSE_DELTA_DEGREES
            for prior in members[:index]
        ):
            raise unproven("duplicate or static corpus pose")
    fit = [member for member in members if member.split == "fit"]
    held_out = [member for member in members if member.split == "held_out"]
    if len(fit) < 2 * VECTOR_SIZE + 1 or len(held_out) < 2:
        raise unproven("fit and held-out multi-pose inventories are incomplete")
    baselines = [member for member in fit if member.category == "baseline"]
    if len(baselines) != 1:
        raise unproven("fit inventory needs exactly one baseline")
    baseline = baselines[0]
    isolated_deltas: dict[str, list[float]] = {joint: [] for joint in JOINT_ORDER}
    for member in fit:
        if member.category != "isolated":
            continue
        if member.isolated_joint not in JOINT_ORDER:
            raise unproven("isolated member declares an unknown physical joint")
        target = JOINT_ORDER.index(member.isolated_joint)
        deltas = tuple(
            value - base for value, base in zip(member.degrees, baseline.degrees, strict=True)
        )
        if abs(deltas[target]) < MIN_ISOLATED_DELTA_DEGREES:
            raise unproven("isolated joint span is insufficient")
        if any(
            abs(delta) > MAX_OTHER_JOINT_DRIFT_DEGREES
            for index, delta in enumerate(deltas)
            if index != target
        ):
            raise unproven("isolated member changes another physical joint")
        isolated_deltas[member.isolated_joint].append(deltas[target])
    if any(
        len(deltas) < 2
        or min(deltas) > -MIN_ISOLATED_DELTA_DEGREES
        or max(deltas) < MIN_ISOLATED_DELTA_DEGREES
        for deltas in isolated_deltas.values()
    ):
        raise unproven("each physical joint needs sufficient positive and negative span")
    task_fit = sum(member.category == "task_plane" for member in fit)
    task_held = sum(member.category == "task_plane" for member in held_out)
    if task_fit < 2 or task_held != len(held_out):
        raise unproven("task-plane fit and held-out combination poses are incomplete")
    return fit, held_out


def parse_joint_corpus(
    corpus: Mapping[str, object],
    member_documents: Sequence[MemberDocument],
    policy: JointEquivalencePolicy,
) -> ParsedJointCorpus:
    """Validate a manifest and parse all integrity-bound measured members."""
    if corpus.get("schema") != CORPUS_SCHEMA:
        raise unproven("boolean/claimed-residual corpus is not measured schema v2 evidence")
    if corpus.get("policy_digest") != policy.canonical_digest:
        raise unproven("corpus is not bound to the approved policy digest")
    claimed_digest = digest(corpus.get("corpus_digest"), "corpus digest")
    if claimed_digest != canonical_digest(corpus, omit="corpus_digest"):
        raise unproven("corpus manifest digest drift")
    origin = text(corpus.get("evidence_origin"), "evidence origin")
    if origin not in {"synthetic_test_fixture", "physical_read_only_capture"}:
        raise unproven("evidence origin must truthfully identify fixture or physical capture")
    order = tuple(
        text(value, "simulator joint order")
        for value in object_list(corpus.get("simulator_joint_order"), "simulator joint order")
    )
    if len(order) != VECTOR_SIZE:
        raise unproven("simulator joint order must contain five joints")
    calibration = mapping(corpus.get("encoder_calibration"), "encoder calibration")
    zero_counts = _vector(calibration.get("zero_counts"), "encoder zero counts")
    scales = _vector(calibration.get("degrees_per_count"), "encoder scales")
    if any(scale <= 0.0 for scale in scales):
        raise unproven("encoder scales must be positive")
    members = tuple(
        _parse_member(entry, document, zero_counts, scales, policy.timing.sample_max_skew_seconds)
        for entry, document in member_documents
    )
    fit, held_out = _inventories(members)
    hashes = tuple(digest(entry.get("sha256"), "member sha256") for entry, _ in member_documents)
    return ParsedJointCorpus(
        claimed_digest, origin, order, members, tuple(fit), tuple(held_out), hashes
    )


def load_joint_corpus_manifest(path: Path) -> tuple[Path, Mapping[str, object]]:
    """Load one manifest without granting its members or scope authority."""
    document_path = path / "corpus.json" if path.is_dir() else path
    if not document_path.is_file():
        raise unproven("genuine physical corpus is absent; no equivalence can be claimed")
    try:
        corpus = mapping(json.loads(document_path.read_text(encoding="utf-8")), "joint corpus")
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise unproven("joint corpus is not readable UTF-8 JSON") from exc
    return document_path, corpus


def load_joint_member_documents(
    document_path: Path, corpus: Mapping[str, object]
) -> list[MemberDocument]:
    """Verify every content-addressed member below an authenticated manifest."""
    entries = [
        mapping(value, "member manifest entry")
        for value in object_list(corpus.get("members"), "members")
    ]
    return [(entry, _load_member(document_path.parent, entry)) for entry in entries]


def load_joint_corpus_documents(path: Path) -> tuple[Mapping[str, object], list[MemberDocument]]:
    """Load one manifest and verify every referenced member's bytes."""
    document_path, corpus = load_joint_corpus_manifest(path)
    return corpus, load_joint_member_documents(document_path, corpus)
