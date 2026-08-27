"""Production RSA signing with private material confined to process memory."""

from __future__ import annotations

import os
import subprocess

__all__ = (
    "generate_rsa_private_key",
    "public_key_from_private",
    "rsa_pkcs1v15_sha256_sign",
)


def _run(
    command: list[str], *, payload: bytes | None = None, pass_fds: tuple[int, ...] = ()
) -> bytes:
    result = subprocess.run(
        command,
        input=payload,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
        pass_fds=pass_fds,
    )
    if result.returncode != 0:
        raise RuntimeError("OpenSSL production signing operation failed")
    return result.stdout


def _private_pipe(private_key: bytes) -> int:
    read_descriptor, write_descriptor = os.pipe()
    try:
        view = memoryview(private_key)
        while view:
            written = os.write(write_descriptor, view)
            if written <= 0:
                raise RuntimeError("private-key memory write failed")
            view = view[written:]
    except Exception:
        os.close(read_descriptor)
        raise
    finally:
        os.close(write_descriptor)
    return read_descriptor


def generate_rsa_private_key() -> bytes:
    """Generate a fresh 3072-bit private key and return it only to process memory."""
    return _run(
        [
            "openssl",
            "genpkey",
            "-algorithm",
            "RSA",
            "-pkeyopt",
            "rsa_keygen_bits:3072",
        ]
    )


def public_key_from_private(private_key: bytes) -> bytes:
    """Derive public PEM while passing private bytes through an anonymous pipe."""
    descriptor = _private_pipe(private_key)
    try:
        return _run(
            ["openssl", "pkey", "-in", f"/proc/self/fd/{descriptor}", "-pubout"],
            pass_fds=(descriptor,),
        )
    finally:
        os.close(descriptor)


def rsa_pkcs1v15_sha256_sign(private_key: bytes, content: bytes) -> bytes:
    """Create a detached PKCS#1 v1.5 SHA-256 signature without a key file."""
    descriptor = _private_pipe(private_key)
    try:
        return _run(
            [
                "openssl",
                "dgst",
                "-sha256",
                "-sign",
                f"/proc/self/fd/{descriptor}",
            ],
            payload=content,
            pass_fds=(descriptor,),
        )
    finally:
        os.close(descriptor)
