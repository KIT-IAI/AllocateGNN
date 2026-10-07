from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping


Runner = Callable[["UnitContext", "Unit"], None]
DoneCheck = Callable[[], bool]


@dataclass(frozen=True)
class UnitContext:
    repo_root: Path
    config: Mapping[str, Any]
    refresh: bool = False


@dataclass(frozen=True)
class Unit:
    id: str
    kind: str
    country: str | None
    category: str
    phase: str
    step: str
    depends_on: tuple[str, ...]
    produces: tuple[Path, ...]
    run: Runner
    credential_env: str | None = None
    done_check: DoneCheck | None = None

    def done(self) -> bool:
        if self.done_check is not None:
            return bool(self.done_check())
        if not self.produces:
            return False
        for path in self.produces:
            if path.is_file() and path.stat().st_size > 0:
                continue
            if path.is_dir() and any(item.is_file() for item in path.rglob("*")):
                continue
            return False
        return True
