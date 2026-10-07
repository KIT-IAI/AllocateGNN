"""Resumable, atomic, evidence-recording HTTP downloads."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import os
from pathlib import Path
import zipfile

import requests

from .hashing import sha256_file


class DownloadError(RuntimeError):
    pass


@dataclass(frozen=True)
class DownloadReceipt:
    url: str
    path: str
    bytes: int
    sha256: str
    resumed_from: int
    completed_at: str


def _remote_size(url: str, *, timeout: float) -> int | None:
    try:
        response = requests.head(url, allow_redirects=True, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException:
        return None
    value = response.headers.get("Content-Length")
    return int(value) if value and value.isdigit() else None


def download_file(
    url: str,
    target: Path | str,
    *,
    refresh: bool = False,
    timeout: float = 60.0,
    zip_probe: str | None = None,
) -> DownloadReceipt:
    destination = Path(target)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size > 0 and not refresh:
        return DownloadReceipt(
            url=url,
            path=str(destination),
            bytes=destination.stat().st_size,
            sha256=sha256_file(destination),
            resumed_from=destination.stat().st_size,
            completed_at=datetime.now(timezone.utc).isoformat(),
        )
    partial = destination.with_name(f"{destination.name}.part")
    if refresh:
        partial.unlink(missing_ok=True)
    resumed_from = partial.stat().st_size if partial.exists() else 0
    headers = {"Range": f"bytes={resumed_from}-"} if resumed_from else {}
    try:
        with requests.get(url, headers=headers, stream=True, timeout=timeout) as response:
            response.raise_for_status()
            append = resumed_from > 0 and response.status_code == 206
            if resumed_from and not append:
                resumed_from = 0
            mode = "ab" if append else "wb"
            with partial.open(mode) as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
                handle.flush()
                os.fsync(handle.fileno())
    except requests.RequestException as exc:
        raise DownloadError(f"download failed for {url}: {exc}; partial={partial}") from exc
    expected = _remote_size(url, timeout=timeout)
    if expected is not None and partial.stat().st_size != expected:
        raise DownloadError(
            f"download length mismatch for {url}: expected {expected}, got {partial.stat().st_size}"
        )
    if destination.suffix.lower() == ".zip" or zip_probe:
        try:
            with zipfile.ZipFile(partial) as archive:
                bad = archive.testzip()
                if bad:
                    raise DownloadError(f"ZIP CRC failed for member {bad}: {partial}")
                if zip_probe and zip_probe not in archive.namelist():
                    raise DownloadError(f"ZIP probe member missing: {zip_probe}")
        except zipfile.BadZipFile as exc:
            raise DownloadError(f"invalid ZIP archive: {partial}") from exc
    os.replace(partial, destination)
    return DownloadReceipt(
        url=url,
        path=str(destination),
        bytes=destination.stat().st_size,
        sha256=sha256_file(destination),
        resumed_from=resumed_from,
        completed_at=datetime.now(timezone.utc).isoformat(),
    )
