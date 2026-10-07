"""TOML-driven static source landing."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import zipfile

from sglib.core.infra.artifacts import atomic_json
from sglib.core.infra.download import download_file
from sglib.core.infra.paths import portable_path

from .unit import Unit, UnitContext


def _raw_country_root(context: UnitContext, country: str) -> Path:
    return context.repo_root / "data" / "datasets" / "1_raw" / country


def output_for_file(context: UnitContext, country: str, file_spec: dict) -> Path:
    raw_root = _raw_country_root(context, country)
    return (raw_root / str(file_spec["filename"])).resolve()


def run_static_dataset(context: UnitContext, unit: Unit) -> None:
    if unit.country is None:
        raise ValueError("a source dataset must bind one country")
    dataset_id = unit.id.split(".")[1]
    dataset = context.config["datasets"][dataset_id]
    receipts: list[dict] = []
    for spec in dataset.get("files", []):
        destination = output_for_file(context, unit.country, spec)
        kind = str(spec.get("kind", "file"))
        if kind == "zip_extract":
            extract_root = _raw_country_root(context, unit.country) / str(spec["extract_root"])
            probe = extract_root / str(spec["zip_probe"]).split("/", 1)[-1]
            if probe.is_file() and not context.refresh:
                receipts.append({"url": spec["url"], "path": portable_path(probe, context.repo_root), "status": "present"})
                continue
            receipt = download_file(
                str(spec["url"]),
                destination,
                refresh=context.refresh,
                zip_probe=str(spec["zip_probe"]),
            )
            temporary = extract_root.with_name(f".{extract_root.name}.part")
            if temporary.exists():
                shutil.rmtree(temporary)
            temporary.mkdir(parents=True)
            with zipfile.ZipFile(destination) as archive:
                archive.extractall(temporary)
            landed = temporary / str(spec["extract_root"])
            if not landed.is_dir():
                raise RuntimeError(f"archive did not contain declared root: {spec['extract_root']}")
            if extract_root.exists():
                shutil.rmtree(extract_root)
            os.replace(landed, extract_root)
            shutil.rmtree(temporary, ignore_errors=True)
            destination.unlink(missing_ok=True)
        else:
            receipt = download_file(
                str(spec["url"]),
                destination,
                refresh=context.refresh,
                zip_probe=spec.get("zip_probe"),
            )
        receipts.append({**receipt.__dict__, "path": portable_path(receipt.path, context.repo_root)})
    manifest = _raw_country_root(context, unit.country) / f"landing_{dataset_id}.json"
    atomic_json({"schema_version": "sg_landing_receipt_v1", "dataset": unit.id, "files": receipts}, manifest)
