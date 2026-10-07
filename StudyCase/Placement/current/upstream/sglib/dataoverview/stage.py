"""Helpers for the numbered per-country DataOverview step files.

Each ``NN_*.py`` / ``NN_*.ipynb`` under ``casestudy/1_DataOverview/<dir>/``
runs a named subset of the unit DAG and then displays what it produced.
Earlier steps are never re-run implicitly: a step whose dependencies are not
``DONE`` fails closed and names the missing unit ids, so the numbered files
must be executed in order exactly like a sequential notebook chain.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

from sglib.core.infra.config import LoadedConfig, load_dataoverview_config
from sglib.core.infra.terms import load_country_profile

from .processing.config import CountryPipelineContext, build_context
from .processing.registry import build_registry, status, topological_order
from .processing.unit import Unit, UnitContext


class StageOrderError(RuntimeError):
    """Raised when a numbered step runs before its predecessors are DONE."""


@dataclass(frozen=True)
class StepReport:
    selected: tuple[str, ...]
    ran: tuple[str, ...]
    skipped: tuple[str, ...]


def stage_root(repo_root: Path | str) -> Path:
    return Path(repo_root).resolve() / "casestudy" / "1_DataOverview"


def load_country_configs(repo_root: Path | str) -> dict[str, LoadedConfig]:
    """Load every country that has a DataOverview overlay (same rule as the CLI)."""

    root = Path(repo_root).resolve()
    stage = stage_root(root)
    profile_root = root / "casestudy" / "config" / "countries"
    loaded: dict[str, LoadedConfig] = {}
    for profile_path in sorted(profile_root.glob("*.toml")):
        code = profile_path.stem.lower()
        profile = load_country_profile(profile_path)
        overlay = stage / profile.directory / f"{code}.toml"
        if overlay.is_file():
            loaded[code] = load_dataoverview_config(
                stage / "general" / "general.toml", profile_path, overlay
            )
    if not loaded:
        raise RuntimeError("no DataOverview country overlays were discovered")
    return loaded


def country_config(repo_root: Path | str, country: str) -> LoadedConfig:
    loaded = load_country_configs(repo_root)
    if country not in loaded:
        raise ValueError(f"unknown country {country!r}; discovered={sorted(loaded)}")
    return loaded[country]


def country_context(repo_root: Path | str, country: str) -> CountryPipelineContext:
    return build_context(Path(repo_root).resolve(), country_config(repo_root, country))


def results_root(repo_root: Path | str, country: str) -> Path:
    directory = str(country_config(repo_root, country).values["country"]["directory"])
    return Path(repo_root).resolve() / "results" / "1_DataOverview" / directory


def figures_root(repo_root: Path | str, country: str) -> Path:
    path = results_root(repo_root, country) / "figures"
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_units(repo_root: Path | str) -> tuple[dict[str, Unit], dict[str, dict]]:
    root = Path(repo_root).resolve()
    loaded = load_country_configs(root)
    configurations = {code: dict(item.values) for code, item in loaded.items()}
    return build_registry(root, configurations), configurations


def select_units(units: dict[str, Unit], country: str | None, members: Iterable[str]) -> list[str]:
    """Select units of ``country`` whose member id or step name is in ``members``.

    ``members`` accepts product/dataset ids (``"regions"``, ``"grid"``) and step
    names (``"download"`` selects every download unit of the country).  The
    country-neutral matrix unit is addressed with ``country=None``.
    """

    wanted = set(members)
    selected: list[str] = []
    for unit in units.values():
        if unit.country != country:
            continue
        parts = unit.id.split(".")
        member = parts[1] if len(parts) > 2 else ""
        if member in wanted or unit.step in wanted or unit.id in wanted:
            selected.append(unit.id)
    if not selected:
        raise ValueError(
            f"no DataOverview units match country={country!r} members={sorted(wanted)}"
        )
    return sorted(selected)


def _missing_predecessors(units: dict[str, Unit], selected: Iterable[str]) -> list[str]:
    chosen = set(selected)
    ordered = topological_order(units, chosen)
    return [unit.id for unit in ordered if unit.id not in chosen and not unit.done()]


def require_done(repo_root: Path | str, country: str | None, members: Iterable[str]) -> list[str]:
    """Fail closed unless every selected unit is DONE; return the unit ids."""

    units, _ = build_units(repo_root)
    selected = select_units(units, country, members)
    pending = [unit_id for unit_id in selected if not units[unit_id].done()]
    if pending:
        raise StageOrderError(
            "these units are not DONE; run the earlier numbered step first: "
            + ", ".join(pending)
        )
    return selected


_TRANSFORM_CONTRACTS = {
    "grid": "grid_metadata.json and grid_points.parquet present for every region with schema sg_grid_bundle_v2",
    "features": "features_receipt.json present and its grid hashes match the current grid bundles",
    "inventory": "data_inventory.json present and non-empty",
    "matrix": "data_matrix.md present and non-empty",
}


def skip_reason(repo_root: Path | str, unit: Unit, config: dict | None = None) -> list[str]:
    """Explain why a DONE unit is skipped: what is present and how that was established."""

    root = Path(repo_root).resolve()
    lines: list[str] = []
    member = unit.id.split(".")[1] if unit.id.count(".") == 2 else ""
    if unit.step == "download":
        for path in unit.produces:
            lines.append(f"already downloaded: {Path(path).resolve()}")
        spec = (config or {}).get("datasets", {}).get(member, {})
        protocol = spec.get("query_protocol")
        if protocol == "gee":
            receipt = next((Path(p) for p in unit.produces if Path(p).name == "acquisition_receipt.json"), None)
            document = None
            if receipt is not None and receipt.is_file():
                try:
                    document = json.loads(receipt.read_text(encoding="utf-8"))
                except (OSError, UnicodeError, json.JSONDecodeError):
                    document = None
            if document is None:
                lines.append("reason: output present but no acquisition receipt (provenance unverified)")
            elif document.get("reused"):
                lines.append(
                    "reason: GEE receipt says reused=true, i.e. the file pre-existed and was only "
                    "validated, never fetched by this code (refresh this unit to fetch)"
                )
            else:
                lines.append(f"reason: fetched by this code from GEE, receipt status={document.get('status')}")
        elif protocol:
            lines.append(f"reason: output present for query protocol {protocol}")
        else:
            receipt = root / "data" / "datasets" / "1_raw" / str(unit.country) / f"landing_{member}.json"
            if receipt.is_file():
                lines.append(f"reason: landed by this code, receipt {receipt}")
            else:
                lines.append(
                    "reason: output present but no landing receipt from this code "
                    "(provenance unverified; use --refresh to re-download)"
                )
        return lines
    contract = _TRANSFORM_CONTRACTS.get(member)
    if contract is None:
        contract = (
            "canonical/evidence contract satisfied (required columns or required status verified)"
            if unit.done_check is not None
            else "all declared outputs present and non-empty"
        )
    lines.append(f"reason: {contract}")
    for path in unit.produces:
        lines.append(f"output: {Path(path).resolve()}")
    return lines


def run_step(
    repo_root: Path | str,
    country: str | None,
    members: Iterable[str],
    *,
    refresh: bool = False,
) -> StepReport:
    """Run the selected units in DAG order, skipping units already DONE.

    Dependencies outside the selection must already be DONE; otherwise the
    step refuses to run so that the numbered files stay strictly sequential.
    """

    root = Path(repo_root).resolve()
    units, configurations = build_units(root)
    selected = select_units(units, country, members)
    missing = _missing_predecessors(units, selected)
    if missing:
        raise StageOrderError(
            "predecessor units are not DONE; run the earlier numbered step first: "
            + ", ".join(missing)
        )
    ran: list[str] = []
    skipped: list[str] = []
    for unit in topological_order(units, selected):
        if unit.id not in selected:
            continue
        state = status(unit)
        if state.startswith("BLOCKED") and not unit.done():
            raise RuntimeError(f"{unit.id}: {state}")
        if unit.done() and not refresh:
            print(f"SKIP {unit.id}")
            for line in skip_reason(root, unit, configurations.get(unit.country) if unit.country else None):
                print(f"     {line}")
            skipped.append(unit.id)
            continue
        print(f"RUN  {unit.id}")
        config = configurations.get(unit.country, {}) if unit.country else {}
        unit.run(UnitContext(repo_root=root, config=config, refresh=refresh), unit)
        if not unit.done():
            raise RuntimeError(
                f"{unit.id}: runner returned without satisfying its output contract"
            )
        print(f"PASS {unit.id}")
        ran.append(unit.id)
    return StepReport(selected=tuple(selected), ran=tuple(ran), skipped=tuple(skipped))


__all__ = [
    "StageOrderError",
    "StepReport",
    "build_units",
    "country_config",
    "country_context",
    "figures_root",
    "load_country_configs",
    "require_done",
    "results_root",
    "run_step",
    "select_units",
    "skip_reason",
    "stage_root",
]
