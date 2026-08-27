"""Historical signed source authorities remain usable only as identity templates."""

from datetime import datetime, timezone
from pathlib import Path
from typing import cast

import pytest

from so101_pusht_benchmark.sim_to_real import offline_authority_issuance as issuance
from so101_pusht_benchmark.sim_to_real.policy_approval import ProductionTrustStore
from so101_pusht_benchmark.sim_to_real.read_only_authority_types import (
    ProductionReadOnlyAcquisitionAuthority,
)


def test_historical_source_is_verified_at_its_signed_approval_time(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    authority = tmp_path / "authority.json"
    encoded = b'{"approved_at":"2026-08-25T15:32:40.000000Z"}'
    observed: dict[str, object] = {}
    sentinel = cast("ProductionReadOnlyAcquisitionAuthority", object())

    def fake_load(
        path: Path,
        *,
        signature_path: Path,
        trust_store: ProductionTrustStore,
        now: datetime,
    ) -> ProductionReadOnlyAcquisitionAuthority:
        observed.update(path=path, signature_path=signature_path, trust_store=trust_store, now=now)
        return sentinel

    def fake_read(_path: Path) -> bytes:
        return encoded

    monkeypatch.setattr(issuance, "_read_production", fake_read)
    monkeypatch.setattr(issuance, "load_read_only_acquisition_authority", fake_load)
    result = issuance.load_authenticated_source_template(
        authority,
        tmp_path / "authority.sig",
        cast("ProductionTrustStore", object()),
    )

    assert result is sentinel
    assert observed["now"] == datetime(2026, 8, 25, 15, 32, 40, tzinfo=timezone.utc)
