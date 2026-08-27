"""Governed CLI for genuine guided camera/table/checkpoint registration capture."""

from __future__ import annotations

import argparse
from importlib import import_module
import json
from pathlib import Path
import sys
from typing import cast, Protocol

import numpy as np
from numpy.typing import NDArray

from so101_pusht_benchmark.hardware_profile import load_hardware_profile

from .camera_registration_capture import (
    CameraObservation,
    CaptureDependencies,
    OPENCV_VISION,
    RegistrationAuthority,
    RegistrationCamera,
    RegistrationCaptureRequest,
    run_guided_capture,
    utc_now,
)
from .camera_registration_target import Placement, TARGET_ASSET, validate_print_scale
from .live_capture_provider import LIVE_READ_PROVIDER_DIGEST, probe_device_identity
from .policy_approval import ProductionTrustStore, RsaPkcs1v15Sha256Anchor
from .read_only_authority import load_read_only_acquisition_authority
from .receipt_routing import CANONICAL_ROLLOUT_ROOT, locate_receipt_path, validate_receipt_identity
from .rollout_codes import RolloutViolation

Image = NDArray[np.uint8]


class _Capture(Protocol):
    def isOpened(self) -> bool: ...

    def get(self, property_id: int, /) -> float: ...

    def read(self) -> tuple[bool, Image]: ...

    def release(self) -> None: ...


class _Cv2(Protocol):
    CAP_PROP_FRAME_WIDTH: int
    CAP_PROP_FRAME_HEIGHT: int
    CAP_PROP_FPS: int

    def VideoCapture(self, path: str) -> _Capture: ...


class ReadOnlyRegistrationCamera:
    """Open, observe, read, and release an existing camera without setters."""

    def __init__(self, device: Path) -> None:
        self._device = device
        self._capture: _Capture | None = None

    def open(self) -> CameraObservation:
        if self._capture is not None:
            raise RuntimeError("camera is already open")
        cv2 = cast("_Cv2", import_module("cv2"))
        capture = cv2.VideoCapture(str(self._device))
        if not capture.isOpened():
            capture.release()
            raise RuntimeError("camera open failed")
        self._capture = capture
        try:
            return CameraObservation(
                int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                float(capture.get(cv2.CAP_PROP_FPS)),
            )
        except Exception:
            self._capture = None
            capture.release()
            raise

    def read(self) -> Image:
        capture = self._capture
        if capture is None:
            raise RuntimeError("camera is not open")
        success, frame = capture.read()
        if not success:
            raise RuntimeError("camera read failed")
        result = np.asarray(frame)
        if result.dtype != np.uint8 or result.ndim != 3 or result.shape[2] != 3:
            raise RuntimeError("camera returned an invalid BGR frame")
        return result

    def close(self) -> None:
        capture = self._capture
        if capture is None:
            return
        self._capture = None
        capture.release()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Capture an unsigned, resumable genuine camera/table/checkpoint registration corpus. "
            "This command is read-only toward hardware and never grants publication authority."
        )
    )
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--acquisition-authority", required=True, type=Path)
    parser.add_argument("--authority-signature", required=True, type=Path)
    parser.add_argument("--trust-anchor", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--measured-square-mm",
        required=True,
        type=float,
        help="Operator measurement of one printed 25 mm board square.",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Verify signed authority, current device/profile bindings, target scale, and route; do not open the camera.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an unsigned partial session after rehashing all prior records and PNG members.",
    )
    return parser


def _trust_store(path: Path) -> ProductionTrustStore:
    return ProductionTrustStore.from_owner_anchors((RsaPkcs1v15Sha256Anchor.from_pem_file(path),))


def _validate_output(path: Path) -> None:
    identity = validate_receipt_identity(locate_receipt_path(path), production=True)
    required = CANONICAL_ROLLOUT_ROOT / "camera/raw"
    if not identity.lexical.is_relative_to(required) or identity.lexical == required:
        raise ValueError(f"output session must be a child of {required}")


def _preflight(args: argparse.Namespace) -> RegistrationAuthority:
    validate_print_scale(float(args.measured_square_mm))
    _validate_output(cast("Path", args.output_dir))
    authority = load_read_only_acquisition_authority(
        cast("Path", args.acquisition_authority),
        signature_path=cast("Path", args.authority_signature),
        trust_store=_trust_store(cast("Path", args.trust_anchor)),
    )
    if authority.provider_digest != LIVE_READ_PROVIDER_DIGEST:
        raise ValueError("signed provider does not authorize the read-only camera adapter")
    profile_path = cast("Path", args.profile).resolve(strict=True)
    if profile_path != authority.profile_path:
        raise ValueError("signed profile path drift")
    profile = load_hardware_profile(profile_path)
    observed_device = probe_device_identity(profile.camera.device)
    if observed_device != authority.camera_device_digest:
        raise ValueError("signed camera device identity drift")
    if profile.camera.device.absolute() != authority.camera_device_path or (
        profile.camera.width,
        profile.camera.height,
        float(profile.camera.fps),
    ) != (authority.camera_width, authority.camera_height, authority.camera_fps):
        raise ValueError("signed camera profile drift")
    return RegistrationAuthority(
        authority.canonical_digest,
        authority.approved_by,
        authority.provider_digest,
        authority.profile_path,
        authority.profile_digest,
        authority.camera_device_path,
        authority.camera_device_digest,
        authority.calibration_digest,
        authority.camera_width,
        authority.camera_height,
        authority.camera_fps,
        (
            profile.camera.crop_x,
            profile.camera.crop_y,
            profile.camera.crop_size,
            profile.camera.crop_size,
        ),
        authority.camera,
    )


def preflight_registration_authority(args: argparse.Namespace) -> RegistrationAuthority:
    """Verify the signed camera authority for live or recorded capture."""
    return _preflight(args)


def _prompt(placement: Placement) -> None:
    print(f"\n[{placement.capture_id}] {placement.instruction}")
    response = input("Confirm target/table/object placement is exact, then type CAPTURE: ")
    if response != "CAPTURE":
        raise RuntimeError("operator aborted before capture")


def run(argv: list[str] | None = None, *, camera: RegistrationCamera | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        authority = _preflight(args)
        preflight = {
            "preflight": "passed",
            "camera_opened": False,
            "hardware_setters_called": False,
            "authority_digest": authority.authority_digest,
            "camera_device_digest": authority.camera_device_digest,
            "profile_digest": authority.profile_digest,
            "target": str(TARGET_ASSET),
            "publication_authority_granted": False,
        }
        if args.preflight_only:
            print(json.dumps(preflight, indent=2, sort_keys=True))
            return 0
        selected = camera or ReadOnlyRegistrationCamera(authority.camera_device_path)
        summary = run_guided_capture(
            RegistrationCaptureRequest(
                cast("Path", args.output_dir),
                float(args.measured_square_mm),
                bool(args.resume),
            ),
            authority,
            selected,
            CaptureDependencies(probe_device_identity, _prompt, OPENCV_VISION, utc_now),
        )
    except (OSError, RuntimeError, TypeError, ValueError, RolloutViolation) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def main() -> int:
    return run()
