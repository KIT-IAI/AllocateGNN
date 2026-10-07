# -*- coding: utf-8 -*-
"""AU FY2024 站表与 firm capacity 还原(R9 解冻 #7,2026-08-11)。

判据(★)需要 F − G − X 的左边;UHC 图层给的是 A = F − G(N-1 剩余可接入
容量),故还原 F = A_2025 + G_FY2024(UHC extract_date=20240627 展望年 2025,
与 FY2024 负荷同口径——必须用 FY2024 而非 FY2009)。

口径不自造:清洗规则**直接复用**本包 derive.au 的 parse_year /
build_layer_index / match_bases(不复制任何清洗逻辑);为证明复用忠实,
开跑先做锚定——用同一份代码复算 FY2009,与我方冻结站表逐站比
peak_mw/energy_gwh/missing_rate(容差 = 落盘舍入位的存储精度),不符硬失败。

缺测门:006 年电量口径 0.10;本线年峰值口径 0.25(D-6 复核:20 座缺测
11.7–22.6% 的站缺的是连续区块,年峰低估中位 0.00%、总量期望偏差 0.043%;
整体排除反而从 Ref 标尺挖掉 12.2% 负荷——收回并打 peak_flagged)。

产物:datasets/2_derived/au/ledger/
    station_table_fy2024.csv / au_firm_capacity.csv / build_report.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import au as AU

from ...config import CountryPipelineContext


FY_MAIN = 2024
MISSING_THRESHOLD_006 = 0.10
MISSING_THRESHOLD = 0.25
#: 锚定容差 = 冻结 CSV 存储精度的半个 ulp(peak round3/energy round4/missing 6 位)
ANCHOR_TOL = {"peak_mw": 5e-4, "energy_gwh": 5e-5, "missing_rate": 5e-7}


def anchor_fy2009(context: CountryPipelineContext) -> dict:
    """用同一份代码复算 FY2009,与我方冻结站表逐站比对。不符即硬失败。"""
    frozen = pd.read_csv((context.derived_root / "station_table_fy2009.csv"), encoding="utf-8-sig").set_index("station")
    year = AU.parse_year(context, 2009)
    bases = year["bases"]

    checked, diffs = 0, []
    for base, b in bases.items():
        if base not in frozen.index:
            diffs.append(f"{base}: 冻结站表无此站")
            continue
        row = frozen.loc[base]
        for col, got in (("peak_mw", b["peak_mw"]),
                         ("energy_gwh", b["energy_gwh"]),
                         ("missing_rate", b["missing_rate"])):
            want = float(row[col])
            if np.isnan(want) and np.isnan(got):
                continue
            if abs(got - want) > ANCHOR_TOL[col] + 1e-9:
                diffs.append(f"{base}.{col}: 复算 {got!r} vs 冻结 {want!r}"
                             f"(超出存储精度容差 {ANCHOR_TOL[col]:g})")
        checked += 1

    rep = {"n_bases_recomputed": len(bases), "n_checked": checked,
           "n_diffs": len(diffs), "diffs": diffs[:20]}
    if diffs:
        raise AssertionError(
            "FY2009 锚定失败——清洗口径已漂移,前 20 条差异:\n" + "\n".join(diffs[:20]))
    print(f"  [锚定 OK] FY2009 复算 {len(bases)} 座基名站,三列存储精度内全一致")
    return rep


def build_station_table(context: CountryPipelineContext, fy: int) -> pd.DataFrame:
    """解析某财年 + 匹配位置图层 → 站表(schema 对齐 FY2009 可比子集)。"""
    year = AU.parse_year(context, fy)
    bases = year["bases"]
    norm2name, name2coord = AU.build_layer_index(context)
    match = AU.match_bases(bases, norm2name)

    rows = []
    for base, b in sorted(bases.items()):
        m = match[base]
        lon, lat = (name2coord.get(m["layer_name"], (np.nan, np.nan))
                    if m["layer_name"] else (np.nan, np.nan))
        matched = m["match_type"] != "none"
        mr = b["missing_rate"]
        usable = matched and mr <= MISSING_THRESHOLD
        rows.append({
            "station": base,
            "variants": ";".join(b["variants"]),
            "n_variants": len(b["variants"]),
            "status": ("usable" if usable else
                       "cleaned_out" if matched else "unlocated"),
            "peak_flagged": bool(usable and mr > MISSING_THRESHOLD_006),
            "match_type": m["match_type"],
            "matched_name": m["layer_name"],
            "similarity": m["similarity"],
            "lon_wgs84": lon,
            "lat_wgs84": lat,
            "energy_gwh": b["energy_gwh"],
            "peak_mw": b["peak_mw"],
            "missing_rate": b["missing_rate"],
            "n_valid_slots": b["n_valid"],
            "n_total_slots": b["n_total"],
            "n_neg_cells": b["n_neg_cells"],
            "merge_mode": b["merge_mode"],
        })
    df = pd.DataFrame(rows)
    # 元数据键随我方 parse_year(源 006 版含 zip_name/window,我方版键面不同)
    df.attrs["year_meta"] = {k: (sorted(year[k]) if isinstance(year[k], set)
                                 else year[k])
                             for k in ("fy", "n_csv_all", "n_empty_placeholder",
                                       "n_with_data_csv", "unit_values",
                                       "slots_set")}
    return df


def build_firm_capacity(context: CountryPipelineContext, tab: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """F = A_2025 + G_FY2024,逐站还原;并登记零裕度截断。"""
    uhc = json.loads((context.raw_root / "locations_sample/ausgrid_uhc_full_2025.geojson").read_text(encoding="utf-8"))
    cap = {}
    for f in uhc["features"]:
        p = f["properties"]
        cap[p["substation"]] = {
            "A_mva": float(p["available_capacity__load__at_n_"]),
            "kv_primary": float(p.get("voltage_level_primary") or np.nan),
            "extract_date": p.get("extract_date"),
        }

    ok = tab[(tab["status"] == "usable") & tab["matched_name"].notna()].copy()
    ok["A_mva"] = ok["matched_name"].map(lambda n: cap.get(n, {}).get("A_mva", np.nan))
    ok["kv_primary"] = ok["matched_name"].map(
        lambda n: cap.get(n, {}).get("kv_primary", np.nan))
    n_no_cap = int(ok["A_mva"].isna().sum())
    ok = ok.dropna(subset=["A_mva"]).copy()

    ok["G_fy2024_mw"] = ok["peak_mw"]
    ok["F_mva"] = ok["A_mva"] + ok["G_fy2024_mw"]
    ok["A_zero_truncated"] = ok["A_mva"] == 0.0   # 疑似截断:F 被低估为 G 本身

    out = ok[["station", "matched_name", "lon_wgs84", "lat_wgs84", "kv_primary",
              "G_fy2024_mw", "A_mva", "F_mva", "A_zero_truncated",
              "peak_flagged", "missing_rate"]].reset_index(drop=True)

    rep = {
        "n_stations": int(len(out)),
        "n_usable_without_capacity": n_no_cap,
        "n_peak_flagged": int(out["peak_flagged"].sum()),
        "n_A_zero_truncated": int(out["A_zero_truncated"].sum()),
        "sum_G_fy2024_mw": round(float(out["G_fy2024_mw"].sum()), 1),
        "sum_A_mva": round(float(out["A_mva"].sum()), 1),
        "sum_F_mva": round(float(out["F_mva"].sum()), 1),
        "utilisation_G_over_F": round(float(out["G_fy2024_mw"].sum()
                                            / out["F_mva"].sum()), 4),
        "F_ge_G_share": round(float((out["F_mva"] >= out["G_fy2024_mw"]).mean()), 4),
    }
    return out, rep


def build_ledger(context: CountryPipelineContext, force: bool = False, skip_anchor: bool = False) -> list[Path]:
    """FY2024 台账再生入口;幂等(三产物齐且非 force 则跳过)。"""
    outs = [(context.derived_root / "ledger") / "station_table_fy2024.csv",
            (context.derived_root / "ledger") / "au_firm_capacity.csv",
            (context.derived_root / "ledger") / "build_report.json"]
    if all(p.is_file() for p in outs) and not force:
        print(f"台账产物已存在,跳过(force=True 可重算):{(context.derived_root / "ledger").name}/")
        return outs
    (context.derived_root / "ledger").mkdir(parents=True, exist_ok=True)
    report: dict = {"reuses": "sglib/dataoverview/processing/derive/au/au.py(parse_year/"
                              "build_layer_index/match_bases)"}

    if not skip_anchor:
        print("【锚定】同一份代码复算 FY2009,与冻结站表比对")
        report["anchor_fy2009"] = anchor_fy2009(context)
    else:
        print("⚠ 已跳过锚定——产物不得用于正式结果")
        report["anchor_fy2009"] = "SKIPPED"

    print(f"【解析】FY{FY_MAIN}")
    tab = build_station_table(context, FY_MAIN)
    meta = tab.attrs["year_meta"]
    vc = tab["status"].value_counts().to_dict()
    print(f"  基名站 {len(tab)};状态分布 {vc}")
    tab.to_csv(outs[0], index=False, encoding="utf-8-sig")
    report["fy2024"] = {**meta, "n_bases": len(tab), "status_counts": vc,
                        "match_types": tab["match_type"].value_counts().to_dict()}

    print("【还原】F = A_2025 + G_FY2024")
    fc, rep = build_firm_capacity(context, tab)
    print(f"  逐站 {rep['n_stations']} 座;ΣG {rep['sum_G_fy2024_mw']} MW | "
          f"ΣF {rep['sum_F_mva']} MVA | 利用率 {rep['utilisation_G_over_F']:.3f} | "
          f"A=0 疑似截断 {rep['n_A_zero_truncated']} | "
          f"peak_flagged {rep['n_peak_flagged']}")
    fc.to_csv(outs[1], index=False, encoding="utf-8-sig")
    report["firm_capacity"] = rep

    outs[2].write_text(json.dumps(report, ensure_ascii=False, indent=1),
                       encoding="utf-8")
    print(f"→ {(context.derived_root / "ledger")}")
    return outs
