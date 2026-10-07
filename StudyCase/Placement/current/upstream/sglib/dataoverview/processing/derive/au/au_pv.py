# -*- coding: utf-8 -*-
"""AU PV 净负荷筛查(A4)再生函数库(R9 解冻 #6,2026-08-11 补迁生成链)。

对两个财年逐站统计三项 PV 污染指标并数字规则分层(禁手写),产出
`datasets/2_derived/au/a4_pv_screening.json` —— 站表 `pv_level_fy2009` 列与
010 显著性 pv_excl 分支的唯一合法来源。`au_registry/a4_pv_screening.json`
自此降级为纯对照登记件(与 a3 同级,代码不再消费)。

指标(15-min MW 净负荷,时段按列序定位,AEST/AEDT 本地时):
1. 负值占比 neg_frac = 负值单元格 / 有效(非缺测)单元格——PV 倒送铁证;
2. 日间/夜间比 day_night_ratio = mean(10:00–14:00) / mean(00:00–04:00);
3. 中午凹陷度 midday_dip = 1 − mean(年均日曲线 11:00–14:00) / mean(肩部),
   肩部 = 08:00–10:00 ∪ 16:00–18:00。

分层(阈值登记于产物 JSON,数字规则生成):
- suspected:neg_frac > 1% 或 day_night_ratio < 1.0;
- mild:非 suspected 且(neg_frac > 0.1% 或 ratio < 1.15 或 dip > 0.10);
- clean:其余。

两年时间列标签不同但列序一致,按**列序**定位时段窗(免标签解析);
占位空站(数值全空)剔除并计数。中文筛查报告段不迁(叙事归 4_Analysis)。
"""
from __future__ import annotations

import io
import json
import warnings
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd

from ...config import CountryPipelineContext


META_COL_NAMES = ("year", "zone substation", "date", "unit")
SLOTS_PER_DAY = 96  # 15 分钟 × 96;列序即时段序(段末刻标签,首列 = 00:00–00:15)

# 时段窗(0 基列下标,含首不含尾)
NIGHT_SLOTS = slice(0, 16)     # 00:00–04:00
DAY_SLOTS = slice(40, 56)      # 10:00–14:00
MIDDAY_SLOTS = slice(44, 56)   # 11:00–14:00
SHOULDER_AM = slice(32, 40)    # 08:00–10:00
SHOULDER_PM = slice(64, 72)    # 16:00–18:00

# 分层阈值(全部登记进产物)
NEG_FRAC_SUSPECT = 0.01
RATIO_SUSPECT = 1.0
NEG_FRAC_MILD = 0.001
RATIO_MILD = 1.15
DIP_MILD = 0.10

THRESH_EARLY_PCT = 0.10     # 判定条款:早年疑似污染站占比 < 10%


def station_metrics(df: pd.DataFrame, station: str) -> dict | None:
    """单站三指标;占位空站(数值全空)返回 None。"""
    norm_cols = [c.strip().lower() for c in df.columns]
    meta_idx = {i for i, c in enumerate(norm_cols) if c in META_COL_NAMES}
    time_cols = [df.columns[i] for i in range(len(df.columns)) if i not in meta_idx]
    if len(time_cols) != SLOTS_PER_DAY:
        raise ValueError(f"{station}: 时间列数 {len(time_cols)} ≠ {SLOTS_PER_DAY}")

    values = df[time_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    valid = ~np.isnan(values)
    n_valid = int(valid.sum())
    if n_valid == 0:
        return None  # 占位空站,剔除并计数

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # 全 NaN 切片的 nanmean 告警
        neg_frac = float((values < 0).sum()) / n_valid
        day_mean = float(np.nanmean(values[:, DAY_SLOTS]))
        night_mean = float(np.nanmean(values[:, NIGHT_SLOTS]))
        ratio = day_mean / night_mean if night_mean > 0 else float("nan")
        profile = np.nanmean(values, axis=0)  # 年均日曲线(96 点)
        midday = float(np.nanmean(profile[MIDDAY_SLOTS]))
        shoulder = float(np.nanmean(np.concatenate(
            [profile[SHOULDER_AM], profile[SHOULDER_PM]])))
        dip = 1.0 - midday / shoulder if shoulder > 0 else float("nan")

    # 分层(NaN 指标不触发对应条件)
    if neg_frac > NEG_FRAC_SUSPECT or (not np.isnan(ratio) and ratio < RATIO_SUSPECT):
        level = "suspected"
    elif (neg_frac > NEG_FRAC_MILD
          or (not np.isnan(ratio) and ratio < RATIO_MILD)
          or (not np.isnan(dip) and dip > DIP_MILD)):
        level = "mild"
    else:
        level = "clean"

    return {
        "station": station,
        "neg_frac": neg_frac,
        "day_night_ratio": ratio,
        "midday_dip": dip,
        "n_valid_cells": n_valid,
        "n_neg_cells": int((values < 0).sum()),
        "level": level,
    }


def analyse_year(context: CountryPipelineContext, fy: int) -> dict:
    """解析一个财年 zip,返回逐站指标与年级汇总。"""
    zp = (context.raw_root / "ausgrid_zs") / f"ausgrid_zs_fy{fy}.zip"
    if not zp.exists():
        raise FileNotFoundError(
            f"{zp} 不存在——请先运行 1_DataOverview/2_AU notebook 的下载步")

    stations: list[dict] = []
    n_empty = 0
    with zipfile.ZipFile(zp) as zf:
        for member in zf.namelist():
            if member.endswith("/") or not member.lower().endswith(".csv"):
                continue
            name = PurePosixPath(member).stem
            name = name.rsplit(" FY", 1)[0]  # 去财年后缀,保留电压变体(站的唯一标识)
            with zf.open(member) as fh:
                df = pd.read_csv(io.TextIOWrapper(fh, encoding="utf-8-sig"))
            rec = station_metrics(df, name)
            if rec is None:
                n_empty += 1
            else:
                stations.append(rec)

    def frac_of(level: str) -> float:
        return sum(1 for s in stations if s["level"] == level) / len(stations)

    ratios = np.array([s["day_night_ratio"] for s in stations], dtype=float)
    dips = np.array([s["midday_dip"] for s in stations], dtype=float)
    neg_fracs = np.array([s["neg_frac"] for s in stations], dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        summary = {
            "fy": fy,
            "n_stations_with_data": len(stations),
            "n_empty_placeholder": n_empty,
            "n_suspected": sum(1 for s in stations if s["level"] == "suspected"),
            "n_suspected_neg_evidence": sum(
                1 for s in stations
                if s["level"] == "suspected" and s["neg_frac"] > NEG_FRAC_SUSPECT),
            "n_suspected_ratio_only": sum(
                1 for s in stations
                if s["level"] == "suspected" and s["neg_frac"] <= NEG_FRAC_SUSPECT),
            "n_mild": sum(1 for s in stations if s["level"] == "mild"),
            "n_clean": sum(1 for s in stations if s["level"] == "clean"),
            "frac_suspected": frac_of("suspected"),
            "frac_mild": frac_of("mild"),
            "frac_clean": frac_of("clean"),
            "n_any_negative": sum(1 for s in stations if s["n_neg_cells"] > 0),
            "neg_frac_median": float(np.nanmedian(neg_fracs)),
            "neg_frac_p90": float(np.nanpercentile(neg_fracs, 90)),
            "ratio_median": float(np.nanmedian(ratios)),
            "ratio_p10": float(np.nanpercentile(ratios, 10)),
            "n_ratio_lt_1": int(np.nansum(ratios < RATIO_SUSPECT)),
            "n_ratio_lt_115": int(np.nansum(ratios < RATIO_MILD)),
            "dip_median": float(np.nanmedian(dips)),
            "dip_p90": float(np.nanpercentile(dips, 90)),
            "n_dip_gt_010": int(np.nansum(dips > DIP_MILD)),
        }
    # 疑似站按证据强度排序:负值占比降序,其次日夜比升序
    suspected = sorted((s for s in stations if s["level"] == "suspected"),
                       key=lambda s: (-s["neg_frac"],
                                      s["day_night_ratio"] if not np.isnan(s["day_night_ratio"]) else 9e9))
    summary["suspected_stations"] = suspected
    summary["stations"] = stations
    return summary


def sanitize(obj):
    """JSON 落盘前把 NaN/inf 换成 None(标准 JSON 不含 NaN 字面量)。"""
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize(v) for v in obj]
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def screen_pv(context: CountryPipelineContext, years: tuple[int, int] = (2009, 2024),
              out_path: Path | None = None, force: bool = False) -> Path:
    """两财年 PV 筛查再生入口;幂等(产物已存在且非 force 则跳过)。"""
    out_path = Path(out_path) if out_path is not None else context.derived_root / "a4_pv_screening.json"
    if out_path.is_file() and not force:
        print(f"PV 筛查产物已存在,跳过(force=True 可重算):{out_path.name}")
        return out_path

    y1, y2 = (analyse_year(context, fy) for fy in years)  # y1 = 早年,y2 = 近年
    for y in (y1, y2):
        print(f"FY{y['fy']}:有数据站 {y['n_stations_with_data']},疑似 {y['n_suspected']}"
              f"({y['frac_suspected']:.2%}),轻度 {y['n_mild']}({y['frac_mild']:.2%})"
              f",负值站 {y['n_any_negative']},日夜比中位 {y['ratio_median']:.3f}")

    # —— 年份策略与判定:数字规则生成,禁手写 ——
    early_ok = bool(y1["frac_suspected"] < THRESH_EARLY_PCT)
    recent_ok = bool(y2["frac_suspected"] < THRESH_EARLY_PCT)
    if early_ok and recent_ok:
        recommended = "dual"
        rationale = (f"两年疑似污染占比均 < {THRESH_EARLY_PCT:.0%}"
                     f"（FY{y1['fy']}={y1['frac_suspected']:.2%}，FY{y2['fy']}={y2['frac_suspected']:.2%}），"
                     f"可做双年份稳健性分析。")
    elif early_ok:
        recommended = "early_years"
        rationale = (f"早年 FY{y1['fy']} 疑似污染占比 {y1['frac_suspected']:.2%} < {THRESH_EARLY_PCT:.0%}，"
                     f"而近年 FY{y2['fy']} 达 {y2['frac_suspected']:.2%}（≥ {THRESH_EARLY_PCT:.0%}）——"
                     f"主分析用早年（PV 渗透前），近年仅作 gross 校正后的稳健性附检。")
    else:
        recommended = "recent_with_gross_correction"
        rationale = (f"早年 FY{y1['fy']} 疑似污染占比已达 {y1['frac_suspected']:.2%}（≥ {THRESH_EARLY_PCT:.0%}），"
                     f"早年方案不成立——改用近年 + Haghdadi 2018 gross load 校正。")
    strategy = {
        "recommended_strategy": recommended,
        "early_ok": early_ok,
        "recent_ok": recent_ok,
        "rationale": rationale,
    }
    verdict = {
        "pv_suspected_early_lt_10pct": early_ok,
        "frac_suspected_early": y1["frac_suspected"],
        "frac_suspected_recent": y2["frac_suspected"],
        "recommended_strategy": recommended,
        "risk_note": ("全部条款通过，无需登记风险。" if early_ok else
                      f"早年疑似污染占比 {y1['frac_suspected']:.2%} 未过 10% 线，年份策略降级处理"),
    }

    def year_payload(y: dict) -> dict:
        keep = {k: v for k, v in y.items() if k not in ("stations", "suspected_stations")}
        keep["suspected_stations"] = [s["station"] for s in y["suspected_stations"]]
        keep["stations"] = y["stations"]  # 逐站全量指标
        return keep

    payload = sanitize({
        "generated_by": "sglib/dataoverview/processing/derive/au/au_pv.py",
        "years": list(years),
        "thresholds": {
            "neg_frac_suspect": NEG_FRAC_SUSPECT,
            "ratio_suspect": RATIO_SUSPECT,
            "neg_frac_mild": NEG_FRAC_MILD,
            "ratio_mild": RATIO_MILD,
            "dip_mild": DIP_MILD,
            "early_frac_suspected_lt": THRESH_EARLY_PCT,
        },
        "windows": {
            "night": "00:00-04:00", "day": "10:00-14:00",
            "midday": "11:00-14:00", "shoulder": "08:00-10:00 + 16:00-18:00",
        },
        "strategy": strategy,
        "verdict": verdict,
        f"fy{y1['fy']}": year_payload(y1),
        f"fy{y2['fy']}": year_payload(y2),
    })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已写出:{out_path}")
    return out_path
