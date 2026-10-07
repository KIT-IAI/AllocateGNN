"""005 交付集合与内容核验；预期集合由现有科学配置展开。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from sglib.core.infra.hashing import sha256_file
from sglib.core.infra.paths import find_repo_root, resolve_case_path
from .handoff import oof_fold
from .materialize import validate_field
from .weighter.candidates import load_candidate_registry


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def checked_path(root: Path, record: dict) -> Path:
    path = (root / record["path"]).resolve()
    path.relative_to(root.resolve())
    if not path.is_file() or sha256_file(path) != record["sha256"]:
        raise ValueError(f"产物内容不符：{path}")
    if "bytes" in record and path.stat().st_size != record["bytes"]:
        raise ValueError(f"产物字节数不符：{path}")
    return path


def exact_set(observed: list, expected: set, label: str) -> None:
    if len(observed) != len(set(observed)) or set(observed) != expected:
        raise ValueError(f"{label} 集合不完整或重复：预期 {len(expected)}，实际 {len(observed)}；"
                         f"缺失 {list(expected - set(observed))[:5]}；额外 {list(set(observed) - expected)[:5]}")


def definitions(bundle) -> list[dict]:
    repo = find_repo_root(__file__)
    return [item for item in load_candidate_registry(
        resolve_case_path(repo / bundle.params["authorities"]["candidate_registry"])
    )["candidates"] if item["materialize"]]


def candidate_entries(root: Path, bundle, *, family_indexes: bool = False) -> list[dict]:
    specs = definitions(bundle)
    expected = {(d["label"], r, s) for d in specs for r in bundle.regions
                for s in (d["seed_policy"]["seeds"] or [None])}
    paths = ([root / "candidates" / f"index_{f}.json" for f in sorted({d["family"] for d in specs})]
             if family_indexes else [root / "candidates/candidate_index.json", root / "candidates/candidate_qa_index.json"])
    entries = [entry for p in paths for entry in read_json(p)["entries"]]
    exact_set([(e["label"], e["region"], e.get("seed")) for e in entries], expected, "candidate")
    lookup = {d["label"]: d for d in specs}
    for entry in entries:
        spec, region, seed = lookup[entry["label"]], entry["region"], entry.get("seed")
        if entry["family"] != spec["family"] or entry["qa_only"] != spec["qa_only"]:
            raise ValueError(f"candidate 族 / QA 身份不符：{entry}")
        if seed is not None:
            expected_fold = oof_fold(list(bundle.regions), seed, bundle.params["training"]["n_folds"], region)
            if entry.get("fold") != expected_fold or not entry.get("inference_node_id"):
                raise ValueError(f"OOF 来源不完整：{entry['label']}/{region}/{seed}")
        elif entry.get("fold") is not None:
            raise ValueError("静态场不得伪造 fold")
        with np.load(checked_path(root, entry), allow_pickle=False) as arrays:
            grid = bundle.grids[region][0]
            if not np.array_equal(arrays["grid_row"], np.arange(len(grid))):
                raise ValueError(f"candidate grid_row 不符：{region}")
            validate_field(arrays["data"], grid, bundle.source_regions[region], source_column=bundle.source_column)
    return sorted(entries, key=lambda e: (e["label"], e["region"], e.get("seed") or 0))


def audit_delivery(root: Path, bundle) -> dict:
    """完整性来自明确集合；禁用分支、静态别名与扫描支持分别计数。"""
    candidates = candidate_entries(root, bundle)
    formal = [e for e in candidates if not e["qa_only"]]
    coverage = {"formal": len(formal), "qa": len(candidates) - len(formal)}
    canonical = {}
    for component in ("assignments", "uniform", "gpm", "proximity", "public_activity"):
        entries = read_json(root / "static" / component / "index.json")["regions"]
        exact_set([e["region"] for e in entries], set(bundle.regions), component)
        for entry in entries:
            region = entry["region"]
            with np.load(checked_path(root, entry), allow_pickle=False) as arrays:
                if component == "assignments":
                    assignment = arrays["assignment"]
                    n = len(bundle.stations[region])
                    if assignment.shape != (len(bundle.grids[region][0]),) or np.any((assignment < 0) | (assignment >= n)):
                        raise ValueError(f"VD 范围不符：{region}")
                    if not np.array_equal(arrays["station_id"].astype(str), bundle.stations[region]["station_id"].astype(str).to_numpy()):
                        raise ValueError(f"VD target 顺序不符：{region}")
                    canonical[region] = assignment.copy()
                elif not np.isfinite(arrays["data"]).all():
                    raise ValueError(f"非有限静态场：{component}/{region}")
        coverage[component] = len(entries)
    for kind in ("idr_fixed", "idr_matched", "civd"):
        if kind == "civd" and not bundle.params["execution"].get("civd_enabled", True):
            coverage[kind] = {"produced": 0, "disabled": True}
            continue
        entries = read_json(root / kind / "index.json")["entries"]
        if kind == "idr_matched":
            expected = {(e["label"], e.get("seed") or 0, e["region"]) for e in formal}
            observed = [(e["candidate"], e["seed"], e["region"]) for e in entries]
        else:
            expected, observed = set(bundle.regions), [e["region"] for e in entries]
        exact_set(observed, expected, kind)
        for entry in entries:
            region = entry["region"]
            with np.load(checked_path(root, entry), allow_pickle=False) as arrays:
                assignment = arrays["assignment"]
                if assignment.shape != canonical[region].shape or np.any((assignment < 0) | (assignment >= len(bundle.stations[region]))):
                    raise ValueError(f"assignment 形状或范围不符：{kind}/{region}")
                if kind.startswith("idr"):
                    selected = arrays["raw_idr_assignment"] if entry["g0_pass"] and entry["g1_pass"] else canonical[region]
                    if not np.array_equal(assignment, selected) or entry["transport_budget"] != bundle.params["idr"]["b_tv"]:
                        raise ValueError(f"IDR 门控与最终 assignment 不符：{kind}/{region}")
        coverage[kind] = len(entries)
    for parameter, values in bundle.params["sweeps"].items():
        entries = read_json(root / "sweeps" / f"index_{parameter}.json")["entries"]
        seeds = [42] if parameter in {"lambda", "tau"} else bundle.params["training"]["seeds"]
        regions = [r for r in bundle.regions if parameter != "lambda" or oof_fold(list(bundle.regions), 42, bundle.params["training"]["n_folds"], r) == 1]
        expected = {(str(signal), float(v), int(s), r)
                    for signal in (("N", "P") if parameter == "lambda" else ("base",))
                    for v in values for s in seeds for r in regions}
        exact_set([(e["signal"], float(e["value"]), e["seed"], e["region"]) for e in entries], expected, parameter)
        for entry in entries:
            checked_path(root, entry)
            if entry["fold"] != oof_fold(list(bundle.regions), entry["seed"], bundle.params["training"]["n_folds"], entry["region"]):
                raise ValueError(f"扫描 OOF 不符：{parameter}")
        coverage[f"sweep_{parameter}"] = len(entries)
    return coverage
