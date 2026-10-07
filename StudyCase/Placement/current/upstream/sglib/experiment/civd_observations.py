"""Reproduce the four-country CIVD correction from frozen fields and assignments.

Commands: prepare, observe --country uk, analyze, seal. Each country can run in
its own process. Unrelated observations are inherited with their original
receipt, while allocator/T1 tables are recomputed and independently checked.
The formal historical results are never overwritten.
"""
from __future__ import annotations

import gc
import json
from pathlib import Path


import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from sglib.core.infra.artifacts import atomic_json
from sglib.core.infra.content_chain import (
    code_projection, derive_chain_commitment, derive_chain_receipt,
    verify_chain,
)
from sglib.core.infra.hashing import sha256_file, sha256_json
from sglib.experiment import allocator_observations, production, reconstruction, stage
from sglib.experiment.manifest import run_gate


COUNTRIES = {"uk": "1_UK", "au": "2_AU", "nl": "4_NL", "nz": "5_NZ"}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def checked_receipt(root):
    receipt = verify_chain(read_json(root / "receipt.json"))
    for relative, digest in receipt["outputs"].items():
        path = (root / relative).resolve()
        if not path.is_file():
            # Legacy planning-pool receipts use an artifact stem, as the registry does.
            legacy = [Path(str(path) + suffix) for suffix in (".npz", ".csv")]
            legacy = [candidate for candidate in legacy if candidate.is_file()]
            if len(legacy) != 1:
                raise ValueError(f"missing or ambiguous receipt artifact: {path}")
            path = legacy[0]
        if not path.is_relative_to(root.resolve()) or sha256_file(path) != digest:
            raise ValueError(f"receipt output mismatch: {path}")
    return receipt



def preserve_baselines(new, old, name):
    """Verify the recomputation and retain exact historical non-CIVD values."""
    if not len(new.columns):
        if len(old.columns):
            raise ValueError(f"lost baseline columns: {name}")
        return new
    mask = new.allocator.ne("CIVD")
    previous = old[old.allocator.ne("CIVD")].reset_index(drop=True)
    current = new.loc[mask, old.columns].reset_index(drop=True)
    # CSV represents empty strings and None by the same empty cell.
    for column in current.columns:
        if current[column].dtype.kind == "O":
            current[column] = current[column].map(lambda value: np.nan if value is None or isinstance(value, str) and value == "" else value)
    # GEOS polygon intersections differ by about 1e-9 square metres across OSes.
    rtol, atol = (1e-10, 1e-8) if name.startswith("T1/") else (1e-12, 1e-12)
    pd.testing.assert_frame_equal(current, previous, check_dtype=False,
                                  check_exact=False, rtol=rtol, atol=atol, obj=name)
    # This run changes CIVD. Original VD/IDR numeric evidence remains bit exact.
    for column in old.columns:
        new.loc[mask, column] = previous[column].to_numpy()
    return new


def check_equal_split(region, view, predicted):
    """Independent mask-and-sum oracle, deliberately not bincount."""
    labels = np.asarray(view.station_cluster)
    ordered = np.unique(labels)
    checks = []
    for field in region.fields:
        table = predicted[predicted.allocator.eq("CIVD") & predicted.candidate.eq(field.label)]
        table = table[table.seed.isna() if field.seed is None else table.seed.eq(field.seed)]
        if table.target_id.astype(str).tolist() != region.station_ids.astype(str).tolist():
            raise ValueError("CIVD prediction station order differs")
        expected = np.zeros(len(labels), float)
        for ordinal, label in enumerate(ordered):
            members = labels == label
            expected[members] = field.values[np.asarray(view.assignment) == ordinal].sum() / members.sum()
        actual = table.predicted.to_numpy(float)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-10)
        np.testing.assert_allclose(actual.sum(), field.values.sum(), rtol=1e-12, atol=1e-8)
        checks.append({"country": region.country, "region": region.region, "candidate": field.label,
            "seed": field.seed, "n_stations": len(labels), "n_clusters": len(ordered),
            "field_total": float(field.values.sum()), "prediction_total": float(actual.sum()),
            "max_oracle_difference": float(np.max(np.abs(actual - expected))), "status": "PASS"})
    return checks


def audit_country(ctx):
    """Keep an existing audit stable: its own recorded state precedes publication."""
    path = ctx.root / "audit.json"
    if path.exists():
        existing, current = read_json(path), production.coverage(ctx)
        for key in ("status", "expected", "produced"):
            if existing[key] != current[key] or current["status"] != "PASS":
                raise ValueError(f"stale Experiment audit: {ctx.country}")
        if ([u for u in existing["units"] if u["step"] != "audit"] !=
                [u for u in current["units"] if u["step"] != "audit"]):
            raise ValueError(f"Experiment audit unit coverage changed: {ctx.country}")
    else:
        production.run_audit(ctx, ctx.units[f"{ctx.country}.audit.audit"])
    gate = run_gate(ctx)
    if gate["status"] == "GENERATED":
        gate = run_gate(ctx)
    if gate["status"] != "PASS" or gate["unclaimed"]:
        raise ValueError(f"Experiment hash gate failed: {ctx.country}")


def observation_code(input_loader):
    """Identity of the observation producer and its explicit input implementation."""
    code = code_projection({"observe": allocator_observations.observe,
        "t1": production.t1_tables, "recompute": observe_country,
        "baseline_check": preserve_baselines, "independent_oracle": check_equal_split,
        "input_loader": input_loader})["code_sha256"]
    code = sha256_json({"rerun": code, "experiment": stage.numerical_code()})
    return code


def observe_country(ctx, *, input_loader, correction_id):
    repo, results, country = ctx.repo, ctx.results, ctx.country
    if ctx.read_only or any((results / "3_Experiment/_closures").glob("*")):
        raise ValueError("sealed Experiment root refuses CIVD observations")
    data, generator = production.data_handoff(ctx), production.generator_handoff(ctx)
    print(f"PASS inputs {country}: {len(generator.bundle.civd)} CIVD regions", flush=True)
    code = observation_code(input_loader)
    audits = []
    for name in ctx.loaded.values["regions"]:
        destination = ctx.root / "observations" / name
        if (destination / "receipt.json").exists():
            existing = checked_receipt(destination)
            if (existing.get("observations", {}).get("correction_id") != correction_id
                    or existing["commitment"]["code_sha256"] != code):
                raise ValueError("cannot resume an unrelated observation")
            print(f"SKIP verified corrected region {country}/{name}", flush=True)
            continue
        original = repo / "results/3_Experiment" / COUNTRIES[country] / "observations" / name
        prior = checked_receipt(original)
        params = prior["commitment"]["scientific_parameters"]
        _, _, _, vd, _, _, _, arrays = production.region_context(ctx, name)
        fields = tuple(f for f in generator.bundle.candidates if f.region == name and not f.qa_only)
        views = production.allocator_views(generator, name)
        if len(views["civd"]) != 1:
            raise ValueError(f"four-country run requires CIVD: {country}/{name}")
        if generator.inputs_receipt_sha256 != params["input_fingerprint"]:
            raise ValueError("frozen input fingerprint changed")
        if [(f.label, f.seed, f.fold) for f in fields] != [(f["label"], f["seed"], f["fold"]) for f in params["fields"]]:
            raise ValueError("field coordinates/order changed")
        inputs = {"inherited_observation_receipt": prior["receipt_sha256"],
                  "rebuilt_context": sha256_json({k: v.tolist() for k, v in arrays.items()})}
        civd_receipt = checked_receipt(results / "2_Generator" / COUNTRIES[country] / "civd")
        inputs["civd_extension_receipt"] = civd_receipt["receipt_sha256"]
        for i, field in enumerate(fields):
            role = f"field_{i}"
            if field.lineage["sha256"] != prior["commitment"]["inputs"][role]:
                raise ValueError(f"frozen field changed: {role}")
            inputs[role] = field.lineage["sha256"]
        for kind in ("vd", "fixed", "matched", "civd"):
            for i, view in enumerate(views[kind]):
                role = f"allocator_{kind}_{i}"
                before = prior["commitment"]["inputs"].get(role)
                if before is not None and before != view["sha256"]:
                    if kind != "civd" or view["object"].metadata.get("frozen_assignment_sha256") != before:
                        raise ValueError(f"frozen assignment changed: {role}")
                    frozen_path = repo / "results/2_Generator" / COUNTRIES[country] / "civd" / f"{name}.npz"
                    if sha256_file(frozen_path) != before:
                        raise ValueError("frozen CIVD archive digest changed")
                    with np.load(frozen_path, allow_pickle=False) as frozen:
                        for column in ("assignment", "grid_cluster", "station_cluster", "raw_labels", "probabilities"):
                            np.testing.assert_array_equal(frozen[column], getattr(view["object"], column))
                inputs[role] = view["sha256"]
        region = reconstruction.ReconstructionRegion(country, name, params["unit"], arrays["station_ids"],
            arrays["station_sources"], arrays["observed"], dict(zip(arrays["source_names"].tolist(), arrays["source_totals"].tolist(), strict=True)),
            arrays["grid_sources"], arrays["assignment"], fields, params["input_fingerprint"], arrays["station_xy"], arrays["grid_xy"],
            tuple(arrays["sources_wkt"].tolist()), params["map_selected"])
        outputs = {key: digest for key, digest in prior["outputs"].items() if not key.startswith(("allocator/", "T1/"))}
        for relative, digest in outputs.items():
            if sha256_file(destination / relative) != digest:
                raise ValueError(f"inherited output changed: {relative}")
        with threadpool_limits(limits=1):
            tables = allocator_observations.observe([region], [v["object"] for v in views["fixed"]],
                [v["object"] for v in views["matched"]], [v["object"] for v in views["civd"]], [vd])
            for table_name, table in tables.items():
                tables[table_name] = preserve_baselines(table, production.frame(original / "allocator" / f"{table_name}.csv"), table_name)
            region_audit = check_equal_split(region, views["civd"][0]["object"], tables["predictions"])
            rows = production.write_tables(destination, tables, outputs, "allocator")
            if params["t1"] is not None:
                t1, reference_digest = production.t1_tables(ctx, region, arrays, views, params["t1"])
                inputs["t1_reference"] = reference_digest
                for table_name in ("station_metrics", "region_metrics"):
                    t1[table_name] = preserve_baselines(t1[table_name], production.frame(original / "T1" / f"{table_name}.csv"), "T1/" + table_name)
                pd.testing.assert_frame_equal(t1["crosswalk"], production.frame(original / "T1/crosswalk.csv"), check_dtype=False)
                rows.update(production.write_tables(destination, t1, outputs, "T1"))
        revised = {**params, "correction_scope": "four_country_posthoc_bugfix_comparison",
            "correction_id": correction_id,
            "production_event": "structural_refactor_regeneration",
            "inherited_correction_id": params.get("correction_id"),
            "recomputed_families": ["allocator", "T1"] if params["t1"] is not None else ["allocator"],
            "station_prediction_rule": "equal_split_cluster_demand"}
        revised["allocators"] = [{"role": f"allocator_{kind}_{i}", "kind": kind,
            "candidate": v["candidate"], "seed": v["seed"], "gate": v["gate"]}
            for kind in ("vd", "fixed", "matched", "civd") for i, v in enumerate(views[kind])]
        commitment = derive_chain_commitment(prior["node_id"], inputs=inputs,
            scientific_parameters=revised, code_sha256=code)
        receipt = derive_chain_receipt(commitment, outputs=outputs, observations={
            "correction_id": correction_id, "backend": "local_cpu", "rows": rows,
            "inherited_receipt_sha256": prior["receipt_sha256"],
            "non_civd_validation": "allocator rtol=atol=1e-12; T1 geometry rtol=1e-10,atol=1e-8; original non-CIVD values retained",
            "equal_split_checks": region_audit})
        atomic_json(receipt, destination / "receipt.json")
        audits.extend(region_audit)
        print(f"PASS corrected {country}/{name}: {len(region_audit)} fields, {len(region.observed)} stations", flush=True)
    # Resume collects checks from the verified receipts too.
    audits = [row for name in ctx.loaded.values["regions"]
        for row in checked_receipt(ctx.root / "observations" / name)["observations"]["equal_split_checks"]]
    qa = results / "correction_checks"
    qa.mkdir(exist_ok=True)
    pd.DataFrame(audits).to_csv(qa / f"{country}_equal_split.csv", index=False, float_format="%.17g")
    audit_country(ctx)
    gc.collect()
