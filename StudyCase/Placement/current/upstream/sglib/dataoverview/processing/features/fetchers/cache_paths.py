"""FeatureExtractor 原始缓存的路径与原子写入工具。"""

from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path, PurePosixPath
import time

from sglib.core.infra.hashing import canonical_json_bytes as canonical_json


class CachePathError(ValueError):
    """缓存路径超出指定根目录。"""


def utc_now() -> str:
    """返回 UTC ISO-8601 时间。"""

    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class CacheAnchor:
    """把缓存相对路径约束在一个明确的根目录内。"""

    def __init__(self, root: str | os.PathLike, *, create: bool = False):
        self.root = Path(root).resolve()
        if create:
            self.root.mkdir(parents=True, exist_ok=True)
        if not self.root.is_dir():
            raise CachePathError(f"缓存根目录不存在: {self.root}")

    @staticmethod
    def normalise(relative: str) -> str:
        pure = PurePosixPath(relative)
        if pure.is_absolute() or not pure.parts or any(
            part in {"", ".", ".."} for part in pure.parts
        ):
            raise CachePathError(f"缓存相对路径无效: {relative!r}")
        return pure.as_posix()

    def resolve(self, relative: str, *, must_exist: bool = True) -> Path:
        normalised = self.normalise(relative)
        path = self.root.joinpath(*PurePosixPath(normalised).parts).resolve()
        try:
            path.relative_to(self.root)
        except ValueError as error:
            raise CachePathError(f"缓存路径越界: {relative!r}") from error
        if must_exist and not path.exists():
            raise FileNotFoundError(path)
        return path

    def relative(self, path: str | os.PathLike, *, must_exist: bool = True) -> str:
        candidate = Path(path).resolve()
        if must_exist and not candidate.exists():
            raise FileNotFoundError(candidate)
        try:
            relative = candidate.relative_to(self.root)
        except ValueError as error:
            raise CachePathError(f"路径不在缓存根目录内: {candidate}") from error
        return PurePosixPath(*relative.parts).as_posix()


def cache_file(
    cache_dir: str | os.PathLike,
    namespace: str,
    cache_key: str,
    filename: str,
) -> Path:
    """返回 `<cache>/<namespace>/<key>/<filename>`，并创建父目录。"""

    anchor = CacheAnchor(cache_dir, create=True)
    relative = "/".join((namespace, cache_key, filename))
    path = anchor.resolve(relative, must_exist=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def replace_file(partial: Path, target: Path) -> Path:
    """将同目录临时文件原子替换为正式文件。"""

    target.parent.mkdir(parents=True, exist_ok=True)
    _replace_with_bounded_permission_retry(partial, target)
    return target


def _atomic_write(path: Path, payload: bytes) -> Path:
    """原子写入字节，并复用 Windows 短暂占用重试。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.part")
    if partial.exists():
        partial.unlink()
    try:
        with partial.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        return replace_file(partial, path)
    finally:
        if partial.exists():
            partial.unlink()


def _replace_with_bounded_permission_retry(partial: Path, target: Path) -> None:
    """容忍 Windows 扫描器造成的短暂文件锁。"""

    for attempt in range(8):
        try:
            partial.replace(target)
            return
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(min(0.05 * (2**attempt), 1.0))
