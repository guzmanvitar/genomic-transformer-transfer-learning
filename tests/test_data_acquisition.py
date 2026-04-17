"""Tests for resilient dataset download behavior.

These tests guard the contract that ``download_with_retry`` must resume
partially downloaded files via HTTP Range requests, verify payload integrity
against a pinned checksum, retry only on transient network errors, and fail
fast on deterministic acquisition errors. Together they ensure the data
acquisition layer is both bandwidth-efficient and forensically reproducible.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from jaguar_geo_assign.data import acquisition
from jaguar_geo_assign.data.acquisition import AcquisitionError, DownloadAsset, download_with_retry


class _FakeResponse:
    def __init__(self, payload: bytes, status: int = 200) -> None:
        self.payload = payload
        self.status = status
        self._offset = 0

    def read(self, size: int = -1) -> bytes:
        if self._offset >= len(self.payload):
            return b""
        if size < 0:
            size = len(self.payload) - self._offset
        chunk = self.payload[self._offset : self._offset + size]
        self._offset += len(chunk)
        return chunk


class _FakeOpener:
    def __init__(self, responses: list[object]) -> None:
        self.responses = responses
        self.requests: list[object] = []

    def open(self, request, timeout: float = 30.0):  # noqa: ANN001
        del timeout
        self.requests.append(request)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def test_download_with_retry_resumes_partial_file_and_verifies_checksum(tmp_path: Path) -> None:
    """Resuming from a partial file issues a Range header and still passes checksum verification."""
    destination = tmp_path / "cat.vcf.gz"
    partial = destination.with_name(f"{destination.name}.part")
    partial.write_bytes(b"ACGT")
    payload = b"TTAA"
    checksum = hashlib.sha256(b"ACGTTTAA").hexdigest()
    opener = _FakeOpener([_FakeResponse(payload, status=206)])

    result = download_with_retry(
        DownloadAsset(url="https://example.test/cat.vcf.gz", destination=destination, checksum=checksum),
        opener=opener,
        sleep=lambda _: None,
    )

    assert result.resumed is True
    assert result.attempts == 1
    assert destination.read_bytes() == b"ACGTTTAA"
    assert opener.requests[0].headers["Range"] == "bytes=4-"


def test_download_with_retry_retries_after_transient_failure(tmp_path: Path) -> None:
    """Transient network errors (e.g. TimeoutError) are retried until the download succeeds."""
    destination = tmp_path / "cat.vcf.gz"
    opener = _FakeOpener([TimeoutError("try again"), _FakeResponse(b"content")])

    result = download_with_retry(
        DownloadAsset(url="https://example.test/cat.vcf.gz", destination=destination),
        opener=opener,
        retries=2,
        sleep=lambda _: None,
    )

    assert result.attempts == 2
    assert destination.read_bytes() == b"content"


def test_download_with_retry_fails_on_checksum_mismatch(tmp_path: Path) -> None:
    """A checksum mismatch aborts without retry and leaves no partial or final artifact behind."""
    destination = tmp_path / "cat.vcf.gz"
    opener = _FakeOpener([_FakeResponse(b"wrong")])

    with pytest.raises(AcquisitionError, match="Checksum mismatch"):
        download_with_retry(
            DownloadAsset(
                url="https://example.test/cat.vcf.gz",
                destination=destination,
                checksum=hashlib.sha256(b"expected").hexdigest(),
            ),
            opener=opener,
            retries=3,
            sleep=lambda _: None,
        )

    assert len(opener.requests) == 1
    assert not destination.exists()
    assert not destination.with_name(f"{destination.name}.part").exists()


def test_download_with_retry_does_not_retry_non_transient_acquisition_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Deterministic AcquisitionError is raised immediately without consuming retry budget."""
    destination = tmp_path / "cat.vcf.gz"
    attempts = 0

    def _fail_download_once(**_: object) -> int:
        nonlocal attempts
        attempts += 1
        raise AcquisitionError("deterministic failure")

    monkeypatch.setattr(acquisition, "_download_once", _fail_download_once)

    with pytest.raises(AcquisitionError, match="deterministic failure"):
        download_with_retry(
            DownloadAsset(url="https://example.test/cat.vcf.gz", destination=destination),
            retries=3,
            sleep=lambda _: None,
        )

    assert attempts == 1