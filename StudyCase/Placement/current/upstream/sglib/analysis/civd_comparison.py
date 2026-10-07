"""Descriptive, post-hoc CIVD correction comparisons from regional C3 products.

This consumer does not change registered contrasts or re-average seed rows.
Call ``build_comparison(repo, results)`` after all four corrected C3 tables exist.
The original UK/AU tables remain under ``repo / 'results'`` for the audit trail.
"""

from datetime import datetime
import inspect
from itertools import product
import json
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from sglib.analysis.config import COUNTRIES, DIRECTORIES, load_analysis_config
from sglib.core.infra.artifacts import atomic_json
from sglib.core.infra.paths import portable_path
from sglib.core.infra.content_chain import code_projection, derive_chain_commitment, derive_chain_receipt, verify_chain
from sglib.core.infra.hashing import sha256_file


ALLOCATORS = ("VD", "IDR-fixed", "IDR-matched", "CIVD")
METRICS = ("rmse", "mae", "wape", "predictive_r2", "corr")
ERROR_METRICS = ("rmse", "mae", "wape")
DISPLAY_CANDIDATES = ("Uni", "GPM", "EqualGrid", "MLP", "GNN")
SCOPE = "posthoc_correction_comparison"
BOOTSTRAP_SEED = 20260914
BOOTSTRAP_REPETITIONS = 10000
PDF_METADATA_TIME = "2026-09-14T00:00:00+00:00"
COUNTRY_NAMES = {"uk": "英国 UK", "au": "澳大利亚 AU", "nl": "荷兰 NL", "nz": "新西兰 NZ"}
KEYS = ["country", "region", "candidate", "allocator", "metric"]


def _read_country(repo, results, country):
    config = load_analysis_config(repo, country)
    spec = config.specification
    path = results / "4_Analysis" / DIRECTORIES[country] / "C3/region_metrics.csv"
    frame = pd.read_csv(path, dtype={"country": str, "region": str}, float_precision="round_trip")
    required = set(KEYS) | {"value", "status", "unit", "realization_count", "n_targets"}
    if not required.issubset(frame.columns):
        raise ValueError(f"{country}: C3 region_metrics is missing {sorted(required - set(frame.columns))}")
    frame = frame[frame.allocator.isin(ALLOCATORS)].copy()
    if frame.empty or set(frame.country) != {country} or frame.duplicated(KEYS).any():
        raise ValueError(f"{country}: empty, duplicate or mixed-country C3 regional coordinates")
    candidates = [candidate for candidate in spec["method_order"]
                  if "direct_reference" not in spec["methods"][candidate].get("evidence", [])]
    if set(candidates) != set(frame.candidate) or set(frame.metric) != set(METRICS):
        raise ValueError(f"{country}: unregistered or omitted expected candidate, or incomplete metric inventory")
    expected = pd.MultiIndex.from_tuples(list(product([country], spec["regions"], candidates, ALLOCATORS, METRICS)), names=KEYS)
    actual = pd.MultiIndex.from_frame(frame[KEYS])
    if len(expected.difference(actual)) or len(actual.difference(expected)):
        raise ValueError(f"{country}: incomplete regional allocator coordinates; all four allocators are required")
    expected_seeds = frame.candidate.map(lambda candidate: len(spec["methods"][candidate]["seeds"]))
    if not frame.realization_count.eq(expected_seeds).all():
        raise ValueError(f"{country}: regional seed reduction differs from the C3 registration")
    if not frame.status.isin(["VALID", "METRIC_NOT_ASSESSABLE"]).all():
        raise ValueError(f"{country}: unexpected regional metric status")
    if not np.isfinite(frame.loc[frame.status.eq("VALID"), "value"].to_numpy(float)).all():
        raise ValueError(f"{country}: a VALID metric contains non-finite values")
    if frame.loc[frame.status.ne("VALID"), "value"].notna().any():
        raise ValueError(f"{country}: an invalid metric carries a numeric value")
    for metric, group in frame.groupby("metric", sort=False):
        unit = spec["unit"] if metric in ("rmse", "mae") else "dimensionless"
        if set(group.unit) != {unit}:
            raise ValueError(f"{country}/{metric}: mixed or unexpected units")
        if metric in ERROR_METRICS and (group.value.dropna() < 0).any():
            raise ValueError(f"{country}/{metric}: negative error metric")
    if (frame.n_targets <= 0).any() or frame.groupby("region").n_targets.nunique().gt(1).any():
        raise ValueError(f"{country}: station counts differ across regional method coordinates")
    frame["candidate_order"] = frame.candidate.map({candidate: i for i, candidate in enumerate(candidates)})
    frame["metric_order"] = frame.metric.map({metric: i for i, metric in enumerate(METRICS)})
    frame["allocator_order"] = frame.allocator.map({allocator: i for i, allocator in enumerate(ALLOCATORS)})
    frame["region_order"] = frame.region.map({region: i for i, region in enumerate(spec["regions"])})
    frame = frame.sort_values(["candidate_order", "metric_order", "allocator_order", "region_order"], kind="stable")
    return frame, spec, path


def _country_summary(frame):
    rows = []
    for keys, group in frame.groupby(["country", "candidate", "metric", "allocator"], sort=False):
        valid = group[group.status.eq("VALID")]
        values = valid.value.to_numpy(float)
        n_valid = len(values)
        rows.append({**dict(zip(["country", "candidate", "metric", "allocator"], keys, strict=True)),
            "candidate_order": int(group.candidate_order.iloc[0]), "metric_order": int(group.metric_order.iloc[0]),
            "allocator_order": int(group.allocator_order.iloc[0]), "unit": group.unit.iloc[0],
            "n_regions": len(group), "n_valid": n_valid, "n_not_assessable": len(group)-n_valid,
            "mean": float(values.mean()) if n_valid else None, "median": float(np.median(values)) if n_valid else None,
            "q25": float(np.quantile(values, .25)) if n_valid else None,
            "q75": float(np.quantile(values, .75)) if n_valid else None,
            "minimum": float(values.min()) if n_valid else None, "maximum": float(values.max()) if n_valid else None,
            "std_across_regions": float(values.std(ddof=1)) if n_valid > 1 else None,
            "realizations_per_region": int(group.realization_count.iloc[0]),
            "status": "VALID" if n_valid == len(group) else "PARTIAL_VALID" if n_valid else "METRIC_NOT_ASSESSABLE",
            "direction": "lower_is_better" if keys[2] in ERROR_METRICS else "higher_is_better",
            "scope": SCOPE, "regional_weighting": "equal_regions_after_within_region_seed_mean"})
    return pd.DataFrame(rows)


def _bootstrap(civd, baseline, metric):
    """Resample paired regions; the same deterministic draws are used in each cell."""
    count = len(civd)
    empty = {"delta_ci_low": None, "delta_ci_high": None, "relative_ci_low_pct": None,
             "relative_ci_high_pct": None, "zero_denominator_draws": 0,
             "interval_status": "METRIC_NOT_ASSESSABLE"}
    if not count:
        return empty
    indices = np.random.default_rng(BOOTSTRAP_SEED).integers(0, count, size=(BOOTSTRAP_REPETITIONS, count))
    delta = civd-baseline
    draws = delta[indices].mean(axis=1)
    low, high = np.quantile(draws, [.025, .975], method="linear")
    result = {**empty, "delta_ci_low": float(low), "delta_ci_high": float(high),
              "interval_status": "DEGENERATE_ONE_REGION" if count == 1 else "DESCRIPTIVE_PERCENTILE_95"}
    if metric in ERROR_METRICS:
        denominators = baseline[indices].sum(axis=1)
        positive = denominators > 0
        result["zero_denominator_draws"] = int((~positive).sum())
        if positive.any() and baseline.sum() > 0:
            relative_draws = 100 * delta[indices].sum(axis=1)[positive] / denominators[positive]
            low, high = np.quantile(relative_draws, [.025, .975], method="linear")
            result.update(relative_ci_low_pct=float(low), relative_ci_high_pct=float(high))
    return result


def _paired(frame):
    regional, summary = [], []
    index = ["country", "candidate", "metric", "region"]
    civd = frame[frame.allocator.eq("CIVD")]
    for baseline in ALLOCATORS[:-1]:
        joined = civd.merge(frame[frame.allocator.eq(baseline)], on=index, suffixes=("_civd", "_baseline"), validate="one_to_one")
        for keys, group in joined.groupby(index[:-1], sort=False):
            group = group.sort_values("region_order_civd", kind="stable")
            country, candidate, metric = keys
            valid = group.status_civd.eq("VALID") & group.status_baseline.eq("VALID")
            x = group.loc[valid, "value_civd"].to_numpy(float)
            y = group.loc[valid, "value_baseline"].to_numpy(float)
            delta = x-y
            tolerance = 1e-12 + 1e-12*np.maximum(np.abs(x), np.abs(y))
            better = delta < -tolerance if metric in ERROR_METRICS else delta > tolerance
            worse = delta > tolerance if metric in ERROR_METRICS else delta < -tolerance
            relative_reason = "" if metric in ERROR_METRICS and len(y) and y.sum() > 0 else (
                "RELATIVE_PERCENT_NOT_DEFINED_FOR_SIGNED_SCORE" if metric not in ERROR_METRICS else "ZERO_BASELINE_OR_NO_VALID_PAIRS")
            summary.append({"country": country, "candidate": candidate, "metric": metric, "baseline": baseline,
                "candidate_order": int(group.candidate_order_civd.iloc[0]), "metric_order": int(group.metric_order_civd.iloc[0]),
                "baseline_order": ALLOCATORS.index(baseline), "unit": group.unit_civd.iloc[0],
                "n_regions": len(group), "n_pairs": len(x), "n_excluded": int((~valid).sum()),
                "civd_mean": float(x.mean()) if len(x) else None, "baseline_mean": float(y.mean()) if len(y) else None,
                "mean_delta": float(delta.mean()) if len(delta) else None,
                "relative_mean_pct": float(100*delta.sum()/y.sum()) if not relative_reason else None,
                "relative_reason": relative_reason, "n_improved": int(better.sum()),
                "n_tied": int((~(better | worse)).sum()), "n_worse": int(worse.sum()),
                **_bootstrap(x, y, metric), "bootstrap_seed": BOOTSTRAP_SEED, "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
                "direction": "lower_is_better" if metric in ERROR_METRICS else "higher_is_better",
                "status": "VALID" if len(x) == len(group) else "PARTIAL_VALID" if len(x) else "METRIC_NOT_ASSESSABLE",
                "scope": SCOPE, "inference": "descriptive_paired_region_bootstrap_no_preregistered_test"})
            for row in group.itertuples():
                assessable = row.status_civd == row.status_baseline == "VALID"
                difference = float(row.value_civd-row.value_baseline) if assessable else None
                tol = 1e-12 + 1e-12*max(abs(row.value_civd), abs(row.value_baseline)) if assessable else 0.
                signed = difference if metric in ERROR_METRICS else -difference if assessable else None
                classification = ("not_assessable" if not assessable else "tied" if abs(difference) <= tol else
                                  "improved" if signed < 0 else "worse")
                regional.append({"country": country, "candidate": candidate, "metric": metric, "baseline": baseline,
                    "region": row.region, "unit": row.unit_civd, "civd_value": row.value_civd,
                    "baseline_value": row.value_baseline, "civd_status": row.status_civd,
                    "baseline_status": row.status_baseline, "delta": difference,
                    "relative_pct": float(100*difference/row.value_baseline) if assessable and metric in ERROR_METRICS and row.value_baseline > 0 else None,
                    "classification": classification, "tie_tolerance": tol if assessable else None,
                    "realizations_per_region": row.realization_count_civd, "n_targets": row.n_targets_civd,
                    "status": "VALID" if assessable else "METRIC_NOT_ASSESSABLE", "scope": SCOPE})
    paired = pd.DataFrame(summary).sort_values(["country", "candidate_order", "metric_order", "baseline_order"], kind="stable")
    return pd.DataFrame(regional), paired.reset_index(drop=True)


def _historical_comparison(repo, corrected):
    rows, sources = [], {}
    for country in ("uk", "au"):
        path = repo / "results/4_Analysis" / DIRECTORIES[country] / "C3/region_metrics.csv"
        parent_path = path.parent / "receipt.json"
        parent = verify_chain(json.loads(parent_path.read_text(encoding="utf-8")))
        if parent["outputs"].get(path.name) != sha256_file(path):
            raise ValueError(f"{country}: prior C3 table differs from its receipt")
        prior_valid = parent["commitment"]["scientific_parameters"].get("correction_scope") == "four_country_posthoc_bugfix_comparison"
        old = pd.read_csv(path, dtype={"country": str, "region": str}, float_precision="round_trip")
        old = old[old.allocator.eq("CIVD") & old.metric.isin(["rmse", "mae"])]
        new = corrected[corrected.country.eq(country) & corrected.allocator.eq("CIVD") & corrected.metric.isin(["rmse", "mae"])]
        joined = new.merge(old, on=KEYS, how="outer", suffixes=("_new", "_old"), validate="one_to_one", indicator=True)
        if not joined._merge.eq("both").all() or not (joined.status_old.eq("VALID") & joined.status_new.eq("VALID")).all():
            raise ValueError(f"{country}: original and corrected CIVD error coordinates do not match")
        if not joined.unit_old.eq(joined.unit_new).all():
            raise ValueError(f"{country}: units differ between original and corrected CIVD results")
        for (candidate, metric), group in joined.groupby(["candidate", "metric"], sort=False):
            before, after = float(group.value_old.mean()), float(group.value_new.mean())
            rows.append({"country": country, "candidate": candidate, "metric": metric, "unit": group.unit_new.iloc[0],
                "n_regions": len(group), "prior_implementation_mean": before,
                "old_invalid_implementation_mean": None if prior_valid else before, "corrected_mean": after,
                "correction_delta": after-before, "correction_relative_pct": 100*(after-before)/before if before else None,
                "old_implementation_valid": prior_valid, "scope": SCOPE,
                "withdrawal_reason": None if prior_valid else "CLUSTER_IDS_WERE_USED_AS_STATION_IDS_AND_STEP4_WAS_OMITTED"})
        sources[country] = {"path": portable_path(path, repo), "sha256": sha256_file(path),
            "receipt_sha256": parent["receipt_sha256"],
            "implementation_status": "prior_corrected" if prior_valid else "historical_missing_step4"}
    return pd.DataFrame(rows), sources


def _plot(summary, specs, destination):
    figure = Figure(figsize=(12.4, 8.4))
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 2).ravel()
    colors = ("#8C959F", "#3973AC", "#42A68D", "#D65347")
    for axis, country in zip(axes, COUNTRIES, strict=True):
        selected = summary[summary.country.eq(country) & summary.metric.eq("rmse")]
        present = set(selected.candidate)
        candidates = [c for c in specs[country]["method_order"] if c in DISPLAY_CANDIDATES and c in present]
        if not candidates:
            raise ValueError(f"{country}: no registered base candidate is available for the RMSE figure")
        positions = np.arange(len(candidates))
        for offset, (allocator, color) in enumerate(zip(ALLOCATORS, colors, strict=True)):
            values = selected[selected.allocator.eq(allocator)].set_index("candidate").loc[candidates, "mean"].to_numpy(float)
            axis.bar(positions + (offset-1.5)*.19, values, width=.18, label=allocator, color=color)
        axis.set_xticks(positions, candidates)
        axis.set_ylabel(f"Regional mean RMSE ({specs[country]['unit']})")
        axis.set_title(f"{country.upper()} | {len(specs[country]['regions'])} regions", loc="left", fontweight="bold")
        axis.set_ylim(bottom=0)
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=.18)
        axis.set_axisbelow(True)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.subplots_adjust(left=.09, right=.985, bottom=.13, top=.84, hspace=.32, wspace=.25)
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .935), ncol=4, frameon=False)
    figure.suptitle("CIVD Step 4 correction: four-country RMSE", fontsize=17, y=.985)
    figure.supxlabel("Registered base candidates; equal region weights after within-region seed means.\n"
                     "Separate native-unit axes; complete candidate comparisons are provided in CSV.", fontsize=10, y=.025)
    figure.savefig(destination / "four_country_rmse.png", dpi=220)
    figure.savefig(destination / "four_country_rmse.pdf", metadata={"Title": "CIVD Step 4 correction: four-country RMSE",
        "Subject": SCOPE, "CreationDate": datetime.fromisoformat(PDF_METADATA_TIME),
        "ModDate": datetime.fromisoformat(PDF_METADATA_TIME)})
    figure.clear()


def _number(value, signed=False):
    return "不可评估" if pd.isna(value) else format(float(value), "+.4g" if signed else ".4g")


def _write_explanation(destination, country_summary, paired, before_after, specs):
    readme = """# CIVD 修正后的四国对照

本目录提供事后修正对照（post-hoc correction comparison），标识为 `posthoc_correction_comparison`。输入为各国 C3 区域指标（regional metrics）；原有预注册检验（preregistered tests）与报告注册保持独立。

- `country_summary.csv`：全部注册候选在四种分配器（allocator）上的区域均值、中位数、四分位数与有效区域数量，完整保留五项指标。
- `paired_comparisons.csv`：CIVD 分别相对 VD、IDR-fixed、IDR-matched 的区域配对差值（paired difference）、相对变化、改善／持平／劣化数量及 95% 自助置信区间（bootstrap confidence interval）。
- `region_comparisons.csv`：每个配对区域的原始数值、差值、判读、种子数量与站点数量。
- `implementation_before_after.csv`：英国与澳大利亚继承实现和本次实现的 RMSE／MAE 数值；旧实现是否已修正由其原回执确定。
- `four_country_rmse.png`／`.pdf`：注册顺序中的基础候选；其余变体的完整结果保存在 CSV。
- `provenance.json`：输入文件路径与散列值（hash）、统计口径和输出清单。
- `receipt.json`：标准内容链收据（content-chain receipt），绑定四国 C3 收据、输入 CSV 散列、自定义函数的代码投影（code projection）以及全部输出文件。

随机种子（random seed）已由 C3 在每个区域内部平均。本模块直接读取这些区域指标，以区域作为统计单位并等权汇总；站点数量与种子数量不作为区域权重。四国原生单位分别保留，不汇总跨国绝对误差。

差值始终定义为 CIVD − baseline。均方根误差（root mean squared error, RMSE）、平均绝对误差（mean absolute error, MAE）与加权绝对百分比误差（weighted absolute percentage error, WAPE）越低越好；预测决定系数（predictive R²）与相关系数（correlation）越高越好。相对变化为 `100 × Σ(CIVD − baseline) / Σbaseline`，不平均各区域百分比。R²／相关系数属于带符号评分，不赋予误差下降百分比；零基准和不可评估指标以空值及原因保留。

配对自助法（paired bootstrap）在同一国家、候选和指标内部同步重采样两种方法的区域，固定随机种子 20260914，共 10000 次，报告 2.5%／97.5% 分位点。有效配对集合、排除数量与零分母抽样次数均明确记录；仅有一个区域时标记退化区间。区间属于事后描述性分析，未调整多重比较，也未构成空间相关修正后的预注册推断。没有生成 p 值或显著性排名。

区域持平的数值容差（numerical tolerance）为 `1e-12 + 1e-12 × max(abs(CIVD), abs(baseline))`。CIVD Step 4 将每个簇的需求均分给成员站点；这一规则不定义唯一站点空间足迹（station footprint），因此 CIVD 的 T1 交并比（intersection over union, IoU）不可评估。

PDF 创建与修改时间固定为 2026-09-14 00:00:00 UTC，作为可重现构建元数据（reproducible build metadata），不表示实际执行时间。
"""
    (destination / "README.md").write_text(readme, encoding="utf-8")
    prior_valid = bool(before_after.old_implementation_valid.all())
    provenance_note = ("继承输入已完成历史 CIVD Step 4 更正。本表记录新代码与上次正式实现的结果差异；旧实现保持有效，历史错误更正证据继续保留在原封存中。"
        if prior_valid else "继承的英国／澳大利亚旧实现缺失 Step 4 簇内均分。其错误数值用于影响追溯，不作为 CIVD 方法结论。")
    lines = ["# CIVD 四国对照与实现复核", "", provenance_note, "",
        "下表展示基础候选的均方根误差（RMSE）和平均绝对误差（MAE）区域均值。", "",
        "| 国家 | 候选 | 指标 | 单位 | 继承实现 | 本次实现 |", "|---|---|---|---|---:|---:|"]
    for row in before_after.itertuples():
        if row.candidate in DISPLAY_CANDIDATES:
            lines.append(f"| {row.country.upper()} | {row.candidate} | {row.metric.upper()} | {row.unit} | {_number(row.prior_implementation_mean)} | {_number(row.corrected_mean)} |")
    lines += ["", "修正后逐国逐候选结果：每格为 CIVD 相对列名基准的区域均值相对变化（relative change, %）；负值表示误差降低，正值表示误差升高。完整差值、置信区间、有效配对数量和区域改善数量保存在 paired_comparisons.csv。", ""]
    for country in COUNTRIES:
        lines += [f"## {COUNTRY_NAMES[country]}", "",
            f"区域数量：{len(specs[country]['regions'])}；RMSE／MAE 原生单位：{specs[country]['unit']}。", "",
            "| 候选 | RMSE vs VD | RMSE vs IDR-fixed | RMSE vs IDR-matched | MAE vs VD | MAE vs IDR-fixed | MAE vs IDR-matched |",
            "|---|---:|---:|---:|---:|---:|---:|"]
        selected = paired[paired.country.eq(country)].set_index(["candidate", "metric", "baseline"])
        present = set(country_summary.loc[country_summary.country.eq(country), "candidate"])
        for candidate in specs[country]["method_order"]:
            if candidate not in present:
                continue
            values = [selected.loc[(candidate, metric, baseline), "relative_mean_pct"]
                      for metric in ("rmse", "mae") for baseline in ALLOCATORS[:-1]]
            lines.append("| " + " | ".join([candidate, *[_number(value, signed=True) for value in values]]) + " |")
        lines.append("")
    lines += ["边界信息：区域平均 RMSE 是各区域 RMSE 的算术均值，不等于把全国站点合并后计算的 RMSE。相关性高不保证绝对误差低；方法判读以同国、同候选、同指标的有效配对为依据。", ""]
    (destination / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def build_comparison(repo, results):
    """Read all four corrected C3 products and write ``results/civd_comparison``."""
    repo, results = Path(repo).resolve(), Path(results).resolve()
    if any((results / "4_Analysis/_closures").glob("*")):
        raise ValueError("sealed Analysis root refuses CIVD comparisons")
    if results == repo / "results":
        raise ValueError("Keep corrected results separate from repo/results until original UK/AU values are archived")
    destination = results / "civd_comparison"
    if destination.exists() and any(destination.iterdir()):
        raise ValueError(f"will not overwrite existing CIVD comparison outputs: {destination}")
    frames, specs, sources, inputs = [], {}, {}, {}
    for country in COUNTRIES:
        frame, spec, path = _read_country(repo, results, country)
        frames.append(frame)
        specs[country] = spec
        digest = sha256_file(path)
        receipt_path = path.parent / "receipt.json"
        parent = verify_chain(json.loads(receipt_path.read_text(encoding="utf-8")))
        if parent.get("outputs", {}).get("region_metrics.csv") != digest:
            raise ValueError(f"{country}: corrected C3 region_metrics differs from its content-chain receipt")
        sources[country] = {"path": portable_path(path, repo), "sha256": digest,
                            "receipt_path": portable_path(receipt_path, repo), "receipt_sha256": parent["receipt_sha256"]}
        inputs[f"analysis:{country}:C3"] = parent["receipt_sha256"]
        inputs[f"analysis:{country}:C3:region_metrics.csv"] = digest
    corrected = pd.concat(frames, ignore_index=True)
    country_summary = _country_summary(corrected)
    regional, paired = _paired(corrected)
    before_after, historical_sources = _historical_comparison(repo, corrected)
    for country, source in historical_sources.items():
        inputs[f"historical:{country}:C3:region_metrics.csv"] = source["sha256"]
    projection = code_projection({name: fn for name, fn in globals().items()
                                  if inspect.isfunction(fn) and fn.__module__ == __name__})
    parameters = {"scope": SCOPE, "countries": list(COUNTRIES), "directories": DIRECTORIES,
        "country_names": COUNTRY_NAMES, "allocators": list(ALLOCATORS), "metrics": list(METRICS),
        "error_metrics": list(ERROR_METRICS), "display_candidates": list(DISPLAY_CANDIDATES), "identity_keys": KEYS,
        "bootstrap_seed": BOOTSTRAP_SEED, "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
        "pdf_metadata_time": PDF_METADATA_TIME,
        "specifications": {cc: {key: specs[cc][key] for key in ("unit", "regions", "method_order", "methods")} for cc in COUNTRIES}}
    commitment = derive_chain_commitment("Analysis.posthoc.CIVD_four_country_comparison", inputs=inputs,
        scientific_parameters=parameters, code_sha256=projection["code_sha256"])
    destination.mkdir(parents=True, exist_ok=True)
    outputs = {"country_summary.csv": country_summary, "paired_comparisons.csv": paired,
               "region_comparisons.csv": regional, "implementation_before_after.csv": before_after}
    for filename, table in outputs.items():
        table.to_csv(destination / filename, index=False, float_format="%.17g", lineterminator="\n")
    _plot(country_summary, specs, destination)
    _write_explanation(destination, country_summary, paired, before_after, specs)
    provenance = {"scope": SCOPE, "corrected_sources": sources, "previous_implementation_sources": historical_sources,
        "implementation_sha256": sha256_file(Path(__file__)), "code_projection": projection, "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_repetitions": BOOTSTRAP_REPETITIONS, "confidence_level": .95,
        "resampling_unit": "paired_region_within_country", "seed_reduction": "already_averaged_within_region_in_C3",
        "regional_weighting": "equal", "cross_country_absolute_pooling": False, "multiple_testing_adjustment": "none_descriptive_only",
        "relative_percent_metrics": list(ERROR_METRICS), "delta_definition": "CIVD_minus_baseline",
        "pdf_metadata_time": PDF_METADATA_TIME,
        "figure_candidates": {cc: [c for c in specs[cc]["method_order"] if c in DISPLAY_CANDIDATES and c in set(frames[i].candidate)]
                              for i, cc in enumerate(COUNTRIES)},
        "outputs": {path.name: sha256_file(path) for path in sorted(destination.iterdir()) if path.is_file() and path.name != "provenance.json"}}
    atomic_json(provenance, destination / "provenance.json")
    output_hashes = {**provenance["outputs"], "provenance.json": sha256_file(destination / "provenance.json")}
    receipt = derive_chain_receipt(commitment, outputs=output_hashes,
        observations={"scope": SCOPE, "code_projection": projection, "backend": "local_cpu"})
    atomic_json(receipt, destination / "receipt.json")
    return {"status": "PASS", "destination": portable_path(destination, repo), "scope": SCOPE,
            "rows": {filename: len(table) for filename, table in outputs.items()}, "provenance": provenance,
            "receipt_sha256": receipt["receipt_sha256"]}
