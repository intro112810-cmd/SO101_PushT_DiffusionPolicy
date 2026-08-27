"""Pinned fixture verification and production trust-store interface."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from pathlib import Path
import subprocess
from typing import Protocol

from .policy_schema import SECTION_FIELDS, YamlMapping, mapping_value, text_value
from .policy_types import OwnerApproval

__all__ = ("ProductionTrustStore", "RsaPkcs1v15Sha256Anchor")
_PRODUCTION_RSA_SCHEME = "rsa-pkcs1v15-sha256-v1"
_FIXTURE_SCHEME = "rsa-pkcs1v15-sha256-test-fixture-v1"
_FIXTURE_SIGNER = "fixture-owner@example.invalid"
_COLLISION_FIXTURE_SIGNER = "collision-fixture-owner@example.invalid"
_FIXTURE_RSA_EXPONENT = 65537
_FIXTURE_RSA_MODULUS = int(
    "AC17AD335BF8EC44CF70AF2A4D442827CE52AA868A3198628DDE4D7EC289360751F8F901"
    "86A39EC128EE624BCE0CD72131FED7D26C060518E922D7F1196A6CBDD9F1AB59B2A9C822"
    "6420A2C17439AA8D27F681230D112D60A27A11DE5E7A580DC6D2A595BAECCB81D3D827D0"
    "713974FEB887EB0E242303B74052A275957EDABCFF707B199E30F2F44C2151E48BCADEA7"
    "9E0DC45C828961A3A8C24C77DE1090992CE534F47229D311794FA0AAD774DFC7A66D702F"
    "3BDCAAFDB043BB9F46CA75ED0377E0109E547C5CFC352D1BD94D5B2CD1EE17ED1A293808"
    "4FD2B8E2D506D26038E25EB1A221C1A942EF5D75FD0F40710AB9873C7E6056358B8110D1"
    "4F39C0AD",
    16,
)
_COLLISION_FIXTURE_RSA_MODULUS = int(
    "A2E7DF8443D61FE8865CD61AAF45F6B27E8E0F0D46270A2A304FE971630EBC3959D3EA0A"
    "C4E9BFA6927B16759F2E708A3D442D7B7F6B281DCC797AF8D4E7638508278937F40BD2E3"
    "7DAB6891A2EA68F2EA2BB9A754C28D5774D40F0F73283DB687A7291917158154B8A08C20"
    "8320D3496BA21F950A42F4F90F4BBD0799C2C1C6C81196D94D3A8AC052E890080886A355"
    "CF85241E3F49AAE74054765514F65081562EA363369BB98E97D3BCE2D90CCF42BD3FA1CE3"
    "FB3274B50292EFF3B0CBE59BBB5EBC8173C56AE560BC89AD492B16908CB19250CD1D9B2F9"
    "90F4BA5C13FC4C2930C7449E7B15607FE1AA6754DB4CFFAAFA0724F8F3566CA03699739AF"
    "5B209",
    16,
)
_FIXTURE_MODULI = {
    _FIXTURE_SIGNER: _FIXTURE_RSA_MODULUS,
    _COLLISION_FIXTURE_SIGNER: _COLLISION_FIXTURE_RSA_MODULUS,
}


class _ProductionApprovalAnchor(Protocol):
    def verify(self, signer_id: str, scheme: str, content: bytes, signature_hex: str) -> bool:
        """Verify one signature against an owner-controlled anchor."""
        ...


class RsaPkcs1v15Sha256Anchor:
    """Owner-controlled production RSA public-key verifier."""

    __slots__ = ("_public_key", "_signer_id")

    def __init__(self, public_key_pem: bytes) -> None:
        if not public_key_pem.startswith(b"-----BEGIN PUBLIC KEY-----\n"):
            raise ValueError("production trust anchor must be a public PEM key")
        self._public_key = public_key_pem
        self._signer_id = hashlib.sha256(public_key_pem).hexdigest()

    @classmethod
    def from_pem_file(cls, path: Path) -> RsaPkcs1v15Sha256Anchor:
        """Load public trust material from one regular non-symlink file."""
        if path.is_symlink() or not path.is_file():
            raise ValueError("production trust anchor must be a regular file")
        return cls(path.read_bytes())

    @property
    def signer_id(self) -> str:
        """Return the content identity used as the authority signer id."""
        return self._signer_id

    @staticmethod
    def _memory_pipe(content: bytes) -> int:
        read_descriptor, write_descriptor = os.pipe()
        try:
            view = memoryview(content)
            while view:
                written = os.write(write_descriptor, view)
                if written <= 0:
                    raise OSError("short trust material write")
                view = view[written:]
        except Exception:
            os.close(read_descriptor)
            raise
        finally:
            os.close(write_descriptor)
        return read_descriptor

    def verify(self, signer_id: str, scheme: str, content: bytes, signature_hex: str) -> bool:
        """Verify one detached signature using only public process-memory material."""
        if signer_id != self._signer_id or scheme != _PRODUCTION_RSA_SCHEME:
            return False
        try:
            signature = bytes.fromhex(signature_hex)
        except ValueError:
            return False
        public_fd = self._memory_pipe(self._public_key)
        signature_fd = self._memory_pipe(signature)
        try:
            result = subprocess.run(
                [
                    "openssl",
                    "dgst",
                    "-sha256",
                    "-verify",
                    f"/proc/self/fd/{public_fd}",
                    "-signature",
                    f"/proc/self/fd/{signature_fd}",
                ],
                input=content,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                pass_fds=(public_fd, signature_fd),
            )
        except OSError:
            return False
        else:
            return result.returncode == 0
        finally:
            os.close(signature_fd)
            os.close(public_fd)


class _StoreConstructionSeal(Protocol):
    """Opaque owner-governed construction token."""


class ProductionTrustStoreError(ValueError):
    """Production trust store construction was attempted without governance."""


_PRODUCTION_STORE_SEAL: _StoreConstructionSeal = object()


class ProductionTrustStore:
    """Sealed production anchor set supplied by an owner-controlled integration."""

    __slots__ = ("_anchors", "_marker")

    def __init__(
        self,
        construction_seal: _StoreConstructionSeal,
        anchors: tuple[_ProductionApprovalAnchor, ...],
    ) -> None:
        if construction_seal is not _PRODUCTION_STORE_SEAL or not anchors:
            raise ProductionTrustStoreError("production trust store is owner-governed")
        self._anchors = anchors
        self._marker = _PRODUCTION_STORE_SEAL

    @classmethod
    def from_owner_anchors(
        cls, anchors: tuple[_ProductionApprovalAnchor, ...]
    ) -> ProductionTrustStore:
        """Construct from out-of-band anchor implementations, never document keys."""
        return cls(_PRODUCTION_STORE_SEAL, anchors)

    def is_governed(self) -> bool:
        """Return true only for a sealed, nonempty owner anchor set."""
        return self._marker is _PRODUCTION_STORE_SEAL and bool(self._anchors)

    def verify(self, signer_id: str, scheme: str, content: bytes, signature_hex: str) -> bool:
        """Verify without accepting keys or anchors from policy content."""
        return any(
            anchor.verify(signer_id, scheme, content, signature_hex) for anchor in self._anchors
        )


def parse_approval_record(raw: YamlMapping) -> tuple[OwnerApproval, bytes]:
    value = mapping_value(raw["owner_approval"], "owner_approval", SECTION_FIELDS["owner_approval"])
    approval = OwnerApproval(
        text_value(value["scheme"], "owner_approval.scheme"),
        text_value(value["approval_id"], "owner_approval.approval_id"),
        text_value(value["signer_id"], "owner_approval.signer_id"),
        text_value(value["policy_digest"], "owner_approval.policy_digest"),
        text_value(value["binding_signature"], "owner_approval.binding_signature"),
    )
    signed: dict[str, str] = {
        "scheme": approval.scheme,
        "approval_id": approval.approval_id,
        "signer_id": approval.signer_id,
        "policy_digest": approval.policy_digest,
    }
    return approval, json.dumps(signed, sort_keys=True, separators=(",", ":")).encode()


def verify_fixture_approval(approval: OwnerApproval, signed: bytes) -> bool:
    try:
        signature = bytes.fromhex(approval.binding_signature)
        modulus = _FIXTURE_MODULI[approval.signer_id]
    except (KeyError, ValueError):
        return False
    size = (modulus.bit_length() + 7) // 8
    if approval.scheme != _FIXTURE_SCHEME or len(signature) != size:
        return False
    encoded = pow(
        int.from_bytes(signature, "big"),
        _FIXTURE_RSA_EXPONENT,
        modulus,
    ).to_bytes(size, "big")
    digest_info = (
        bytes.fromhex("3031300d060960864801650304020105000420") + hashlib.sha256(signed).digest()
    )
    expected = b"\x00\x01" + b"\xff" * (size - len(digest_info) - 3) + b"\x00" + digest_info
    return hmac.compare_digest(encoded, expected)
