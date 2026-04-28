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
    def __init__(
        self, payload: bytes, status: int = 200, headers: dict[str, str] | None = None
    ) -> None:
        self.payload = payload
        self.status = status
        self.headers = headers or {}
        self._offset = 0

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *args: object) -> None:
        pass

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
        DownloadAsset(
            url="https://example.test/cat.vcf.gz", destination=destination, checksum=checksum
        ),
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


def test_download_with_retry_head_size_mismatch(tmp_path: Path) -> None:
    """HEAD size mismatch raises AcquisitionError immediately without fetching body."""
    destination = tmp_path / "cat.vcf.gz"
    # First response for HEAD, second for GET (which should never happen)
    opener = _FakeOpener(
        [_FakeResponse(b"", headers={"Content-Length": "999"}), _FakeResponse(b"should not fetch")]
    )

    with pytest.raises(AcquisitionError, match="Size mismatch during HEAD pre-flight"):
        download_with_retry(
            DownloadAsset(
                url="https://example.test/cat.vcf.gz", destination=destination, expected_size=100
            ),
            opener=opener,
            retries=3,
            sleep=lambda _: None,
        )

    assert len(opener.requests) == 1
    assert getattr(opener.requests[0], "method", opener.requests[0].get_method()) == "HEAD"
    assert not destination.exists()


def test_download_with_retry_head_failure_falls_through(tmp_path: Path) -> None:
    """HEAD failure falls through to normal download."""
    destination = tmp_path / "cat.vcf.gz"
    opener = _FakeOpener([TimeoutError("HEAD failed"), _FakeResponse(b"content")])

    result = download_with_retry(
        DownloadAsset(
            url="https://example.test/cat.vcf.gz",
            destination=destination,
            expected_size=100,
            checksum=hashlib.sha256(b"content").hexdigest(),
        ),
        opener=opener,
        retries=1,
        sleep=lambda _: None,
    )

    assert len(opener.requests) == 2
    assert getattr(opener.requests[0], "method", opener.requests[0].get_method()) == "HEAD"
    assert getattr(opener.requests[1], "method", opener.requests[1].get_method()) == "GET"
    assert result.bytes_written == len(b"content")


def test_download_with_retry_mirror_fallback(tmp_path: Path) -> None:
    """Mirror fallback happens when primary SHA mismatches."""
    destination = tmp_path / "cat.vcf.gz"
    opener = _FakeOpener([_FakeResponse(b"wrong"), _FakeResponse(b"correct")])

    result = download_with_retry(
        DownloadAsset(
            url="https://example.test/primary.gz",
            destination=destination,
            mirror_url="https://example.test/mirror.gz",
            checksum=hashlib.sha256(b"correct").hexdigest(),
        ),
        opener=opener,
        retries=1,
        sleep=lambda _: None,
    )

    assert len(opener.requests) == 2
    assert (
        getattr(
            opener.requests[0],
            "full_url",
            getattr(
                opener.requests[0], "get_full_url", lambda: getattr(opener.requests[0], "url", "")
            )(),
        )
        == "https://example.test/primary.gz"
    )
    assert (
        getattr(
            opener.requests[1],
            "full_url",
            getattr(
                opener.requests[1], "get_full_url", lambda: getattr(opener.requests[1], "url", "")
            )(),
        )
        == "https://example.test/mirror.gz"
    )
    assert result.bytes_written == len(b"correct")
    assert destination.read_bytes() == b"correct"


def test_download_with_retry_md5_backward_compat(tmp_path: Path) -> None:
    """Backward compat for the 5 non-jaguar felids: md5 checksum_name round-trips identically."""
    destination = tmp_path / "cat.vcf.gz"
    content = b"legacy content"
    expected_md5 = hashlib.md5(content).hexdigest()
    opener = _FakeOpener([_FakeResponse(content)])

    result = download_with_retry(
        DownloadAsset(
            url="https://example.test/cat.vcf.gz",
            destination=destination,
            checksum=expected_md5,
            checksum_name="md5",
        ),
        opener=opener,
        retries=1,
        sleep=lambda _: None,
    )

    assert result.bytes_written == len(content)
    assert destination.read_bytes() == content
