"""Tests for shared fine-tune data helpers (CoordStats)."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from jaguar_geo_assign.fine_tune.dataset import CoordStats


def test_coord_stats_json_round_trip(tmp_path: Path) -> None:
    """CoordStats.to_json and from_json must round-trip and clamp std devs."""

    stats = CoordStats(lat_mean=1.0, lat_std=0.0, lon_mean=-2.0, lon_std=0.0)
    path = tmp_path / "coord_stats.json"
    stats.to_json(path)
    loaded = CoordStats.from_json(path)
    assert loaded.lat_mean == pytest.approx(1.0)
    assert loaded.lon_mean == pytest.approx(-2.0)
    assert loaded.lat_std >= 1e-6 and loaded.lon_std >= 1e-6


def test_coord_stats_logs_when_clamping(caplog: pytest.LogCaptureFixture) -> None:
    """CoordStats must emit a WARNING log when standard deviations are clamped."""

    logger_name = "jaguar_geo_assign.fine_tune.dataset"
    caplog.set_level(logging.WARNING, logger=logger_name)

    _ = CoordStats(lat_mean=0.0, lat_std=0.0, lon_mean=0.0, lon_std=0.0)

    messages = [record.getMessage() for record in caplog.records if record.name == logger_name]
    assert any("CoordStats std devs clamped" in message for message in messages)
