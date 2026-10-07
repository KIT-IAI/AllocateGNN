"""006 四国描述性装配、覆盖规范化与综合表；禁止跨国推断。"""

from __future__ import annotations

import numpy as np
import pandas as pd


COUNTRIES = ("uk", "au", "nl", "nz")


def _c1_differences(core):
    regional = core["C1"]["region_metrics"]
    wide = regional[regional.metric.eq("rmse")].pivot(index="region", columns="candidate", values="value")
    rows = []
    for contrast in core["C1"]["contrasts"].itertuples():
        values = wide[contrast.left] - wide[contrast.right]
        rows.extend({"region": region, "claim_id": "C1", "contrast_id": contrast.contrast_id,
                     "value": value, "unit": contrast.unit} for region, value in values.items())
    return pd.DataFrame(rows)


def _with_section(frames):
    rows = []
    for section, frame in frames:
        if frame is None or not len(frame):
            continue
        table = frame.copy(); table.insert(0, "section", section); rows.append(table)
    return pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame(columns=["section"])


def _target_reconciliation(country, context):
    """逐区域对账 target：非空、粒度比自洽、源/目标总量守恒；不与历史计数比较。"""
    table = context[["country", "region", "n_source", "n_target", "granularity_ratio", "source_total", "target_total"]].copy()
    if not table.country.eq(country).all() or table.region.duplicated().any():
        raise ValueError(f"{country}: 区域上下文的国家或区域标识不唯一")
    n_source, n_target = table.n_source.astype(float), table.n_target.astype(float)
    source_total, target_total = table.source_total.astype(float), table.target_total.astype(float)
    table["region_nonempty"] = n_source.gt(0) & n_target.gt(0)
    table["granularity_ratio_consistent"] = np.isclose(table.granularity_ratio.astype(float), n_target / n_source, rtol=1e-12, atol=0.0)
    table["mass_reconciled"] = (np.isfinite(source_total) & source_total.gt(0)
                                & np.isclose(source_total, target_total, rtol=1e-9, atol=1e-9))
    table["country_target_count_from_regions"] = int(n_target.sum())
    return table


def assemble(country_data):
    if tuple(country_data) != COUNTRIES:
        raise ValueError("综合装配国家顺序必须固定为 UK/AU/NL/NZ")
    inventories, evidence, coordinates, resolutions, effects = [], [], [], [], []
    target_counts, connection_maps = [], []
    for country, data in country_data.items():
        core, support, defense = data["core"], data["support"], data["defense"]
        inv = core["C1"]["shared_inventory"].copy(); inv["country"] = country; inventories.append(inv)
        ev = support["claim_evidence_status"].copy()
        evidence.append(ev)
        coord = support["claim_coordinate_audit"].copy(); coord["country"] = country; coordinates.append(coord)
        res = support["inference_resolution_audit"].copy(); res["country"] = country; resolutions.append(res)
        difference_tables = [_c1_differences(core)]
        for claim in ("C2", "C3", "C4", "C5"):
            table = core[claim]["region_differences"].copy(); table["claim_id"] = claim; difference_tables.append(table)
        diff = pd.concat(difference_tables, ignore_index=True, sort=False); diff["country"] = country
        context_table = data["context"].copy()
        uncontexted = sorted(set(diff.region) - set(context_table.region))
        if uncontexted:
            raise ValueError(f"{country}: 效应区域缺少区域上下文: {uncontexted}")
        context = context_table[["country", "region", "n_source", "n_target", "granularity_ratio", "capacity_basis", "unit"]]
        effects.append(diff.merge(context, on=["country", "region"], how="left", validate="many_to_one"))
        target_counts.append(_target_reconciliation(country, context_table))
        mapping = defense["C4_connection_map"].copy(); mapping["country"] = country; connection_maps.append(mapping)
    evidence_table = pd.concat(evidence, ignore_index=True, sort=False)
    if evidence_table.duplicated(["country", "claim_id"]).any() or len(evidence_table) != 24:
        raise ValueError("四国×六主张证据矩阵不完整")
    target_table = pd.concat(target_counts, ignore_index=True, sort=False)
    checks = ["region_nonempty", "granularity_ratio_consistent", "mass_reconciled"]
    failed = target_table.loc[~target_table[checks].all(axis=1), ["country", "region"]]
    if len(failed):
        raise ValueError(f"区域上下文未通过 target 对账: {failed.to_dict(orient='records')}")
    engineering = pd.DataFrame([
        {"engineering_question": "是否使用学习模型", "evidence": "C1,C2-E08,C4", "allowed_action": "按国家和任务比较已登记效应与区间", "boundary": "不生成跨国赢家或等效结论"},
        {"engineering_question": "是否加入 N/P 辅助信息", "evidence": "C2", "allowed_action": "按基座、信号和 operator 报告已验证方向", "boundary": "回顾对齐不是部署门；P/NP 不外推至规划"},
        {"engineering_question": "是否改变 VD 边界", "evidence": "C3", "allowed_action": "保留完整程序效果和回退比例", "boundary": "规划 allocator 槽是结构重复"},
        {"engineering_question": "用什么尺度评价", "evidence": "C5", "allowed_action": "读取任务核尺度与支持域", "boundary": "不从曲线挑选最优半径"},
        {"engineering_question": "上游改善是否传至任务", "evidence": "C4", "allowed_action": "逐任务读取效应向量", "boundary": "不合成混合量纲总分"},
        {"engineering_question": "接入信息是否充分", "evidence": "C6", "allowed_action": "在预算实现域读取条件上界曲线", "boundary": "无 planner tolerance 和外部预算，不给部署通过"},
    ])
    inheritance = pd.DataFrame([
        {"topic": "C1/C2/C4 historical UK/AU", "current_protocol": "006 unified four-country HPC", "comparability": "reproduction_not_independent_confirmation", "status": "REVISED_SCOPE"},
        {"topic": "C2 old sigma threshold", "current_protocol": "mean_term_plus_exact_identity", "comparability": "retrospective_diagnostic_only", "status": "REVISED_INTERPRETATION"},
        {"topic": "C3 allocator in planning", "current_protocol": "canonical_VD_only", "comparability": "old_extra_slots_structural_duplicates", "status": "CORRECTED_MATRIX"},
        {"topic": "Connection operator", "current_protocol": "fixed_raw_grid_candidate_neighbourhood", "comparability": "legacy_station_operator_not_numeric_substitute", "status": "VERSIONED_REPLACEMENT"},
        {"topic": "C6 guarantee", "current_protocol": "retrospective_LOO_budget_and_normalized_curves", "comparability": "not_external_or_deployment_guarantee", "status": "CONDITIONAL_ONLY"},
    ])
    coverage = pd.DataFrame([
        {"claim_id": claim, "experiments": experiments, "figure_ids": figures, "table_ids": tables,
         "cross_country_inference": False, "status_source": "claim_evidence_status"}
        for claim, experiments, figures, tables in [
            ("C1", "C1-E01,E04,E05", "F-C1-01..03", "T-C1-01,T-INV-01"),
            ("C2", "C2-E01..06,E08", "F-C2-01..03", "T-C2-01..03,T-INV-01"),
            ("C3", "C3-E01..05", "F-C3-01..03", "T-C3-01..02"),
            ("C4", "C4-E01..05", "F-C4-01..03", "T-C4-01..03"),
            ("C5", "C5-E01..05", "F-C5-01..03", "T-C5-01..02"),
            ("C6", "C6-E01..05", "F-C6-01..03", "T-C6-01..02"),
        ]])
    limitations = pd.DataFrame([
        {"item": "UK_GBP_TNUoS", "status": "MISSING_REQUIRED", "reason": "VERSIONED_TARIFF_AND_CONVERSION_CONTRACT_NOT_PROVIDED"},
        {"item": "planner_tolerance", "status": "NOT_PROVIDED", "reason": "NORMALIZED_CURVES_ONLY"},
        {"item": "external_error_budget", "status": "NOT_PROVIDED", "reason": "RETROSPECTIVE_LOO_BUDGET_ONLY"},
        {"item": "MILP", "status": "NOT_AUTHORIZED", "reason": "EXPLICIT_SCOPE_BOUNDARY"},
        {"item": "legacy_station_connection_operator", "status": "NOT_EVALUATED", "reason": "NOT_REUSED_AS_CURRENT_PROTOCOL"},
    ])
    return {
        "T_INV_01": pd.concat(inventories, ignore_index=True, sort=False),
        "claim_evidence_status": evidence_table,
        "claim_coordinate_audit": pd.concat(coordinates, ignore_index=True, sort=False),
        "inference_resolution_audit": pd.concat(resolutions, ignore_index=True, sort=False),
        "region_effect_context": pd.concat(effects, ignore_index=True, sort=False),
        "target_count_audit": target_table,
        "coverage_corrections": pd.DataFrame(columns=["country", "artifact", "field", "before", "after", "reason", "source_receipt_immutable"]),
        "connection_map": pd.concat(connection_maps, ignore_index=True, sort=False),
        "engineering_choices": engineering,
        "inheritance_revision": inheritance,
        "claim_experiment_figure_coverage": coverage,
        "limitations_registry": limitations,
    }
