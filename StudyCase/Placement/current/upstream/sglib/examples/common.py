"""Persist explicit smoke inputs without consulting a formal result root."""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from sglib.core.infra.artifacts import atomic_json, atomic_npz
from sglib.core.infra.hashing import sha256_file, sha256_json


def check_smoke_output(
    repo_root: Path,
    output_root: Path,
    *,
    receipt_relative: str | None = None,
    receipt_schema: str | None = None,
    country: str | None = None,
    reuse: bool = False,
) -> Path | None:
    """Admit an empty isolated root or reuse fully verified nonformal artifacts.

    This check never creates, removes, or changes files. It runs before a
    workflow reads its source data or writes any fixture products.
    """
    root = Path(output_root).resolve()
    formal = (Path(repo_root).resolve() / "results").resolve()
    stages = ("1_DataOverview", "2_Generator", "3_Experiment", "4_Analysis", "5_Report")
    if root == formal or any(root.is_relative_to((formal / stage).resolve()) for stage in stages):
        raise ValueError("smoke output cannot use an original formal results root")
    for ancestor in (root, *root.parents):
        closure_roots = [ancestor / "_closures"]
        # The original results directory also contains writable auxiliary
        # roots, so its five formal stages are protected separately above.
        if ancestor != formal:
            closure_roots.extend(ancestor / stage / "_closures" for stage in stages)
        if any(path.is_dir() and any(path.iterdir()) for path in closure_roots):
            raise ValueError("smoke output cannot use a sealed results root or its descendants")
    if reuse and receipt_relative is not None:
        receipt = (root / receipt_relative).resolve()
        if not receipt.is_relative_to(root):
            raise ValueError("smoke receipt escapes its output root")
        if receipt.is_file():
            try:
                document = json.loads(receipt.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise ValueError("existing smoke receipt is invalid") from exc
            if (not isinstance(document, dict) or document.get("status") != "PASS"
                    or document.get("formal") is not False
                    or document.get("schema_version") != receipt_schema
                    or document.get("country") != country):
                raise ValueError("existing smoke receipt is not valid nonformal evidence for this example")
            from .verification import verify_smoke_products
            try:
                verify_smoke_products(Path(repo_root).resolve(), root, document, country=country)
            except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
                raise ValueError(f"existing smoke artifacts failed verification: {exc}") from exc
            return receipt
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise FileExistsError("new smoke computation requires an empty output root")
    return None


def write_fixture_inventory(handoff, output_root: Path):
    root = Path(output_root).resolve()
    directory = root / "inputs/fixture"
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, table in (("sources", handoff.regions_table), ("stations", handoff.stations_table)):
        path = directory / f"{name}.parquet"
        table.to_parquet(path, index=False)
        paths.append(path)
    for region in handoff.regions:
        region_root = directory / region.region
        region_root.mkdir(parents=True, exist_ok=True)
        grid_path = region_root / "grid.parquet"
        region.grid.to_parquet(grid_path, index=False)
        paths.extend([grid_path, atomic_json(dict(region.grid_metadata), region_root / "grid_metadata.json")])
        for name in ("landuse", "built_surface", "cuz_support", "ntl"):
            paths.append(atomic_npz(region_root / f"{name}.npz", **getattr(region, name)))
    artifacts = [{"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in paths]
    inventory = {**handoff.inventory, "schema_version": "sg_dataoverview_inventory_v1", "country": handoff.country, "formal": False, "artifacts": artifacts, "fingerprint": sha256_json(artifacts)}
    path = atomic_json(inventory, directory / "data_inventory.json")
    return replace(handoff, inventory=inventory), path


def publish_static_candidate_indexes(output_root: Path) -> Path:
    """Publish the explicit static-only subset used by a local smoke audit."""
    root = Path(output_root).resolve()
    entries = []
    for family in ("Uni", "GPM", "Equal"):
        document = json.loads((root / "candidates" / f"index_{family}.json").read_text(encoding="utf-8"))
        for entry in document["entries"]:
            path = root / entry["path"]
            if sha256_file(path) != entry["sha256"]:
                raise ValueError(f"smoke candidate hash differs: {path}")
            entries.append(entry)
    common = {"formal": False, "profile": "smoke", "families": ["Uni", "GPM", "Equal"]}
    atomic_json({**common, "schema_version": "sg_candidate_qa_index_v1", "entries": [entry for entry in entries if entry["qa_only"]]}, root / "candidates/candidate_qa_index.json")
    return atomic_json({**common, "schema_version": "sg_candidate_index_v1", "entries": [entry for entry in entries if not entry["qa_only"]]}, root / "candidates/candidate_index.json")
