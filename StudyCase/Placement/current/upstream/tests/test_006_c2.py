"""C2 的直接交互与九主 / 四次级边界。"""

import numpy as np
import pandas as pd
import pytest

from sglib.analysis.c2 import analyze, contrast_definitions

pytestmark = pytest.mark.consume


def fixture():
    definitions = contrast_definitions()
    methods = sorted({m for d in definitions for m in d["coefficients"]})
    regions = [f"r{i}" for i in range(9)]
    spec = {"country": "nz", "regions": regions, "unit": "MVA",
            "methods": {m: {"seeds": [42, 123, 456] if m.startswith(("GNN", "MLP")) else [None]} for m in methods},
            "queen_adjacency": np.zeros((9, 9), bool), "blocks": [[i] for i in range(9)]}
    def level(method):
        base = 10. if method.startswith("GPM") else 9. if method.startswith("MLP") else 8.
        correction = (-2. if "post" in method else -1. if "add" in method else 0.)
        return base + correction * (2 if method.startswith("GNN") else 1)
    table = pd.DataFrame([{"country": "nz", "region": r, "candidate": m, "metric": "rmse", "allocator": "VD",
        "value": level(m), "status": "VALID", "unit": "MVA", "realization_count": len(spec["methods"][m]["seeds"])} for r in regions for m in methods])
    return table, spec, definitions


def test_c2_nine_primary_and_four_secondary_with_direct_interactions():
    table, spec, definitions = fixture()
    result = analyze(table, spec, definitions)
    contrast = result["contrasts"]
    assert len(contrast) == 13 and contrast.primary.sum() == 9
    assert len(result["inference_resolution_audit"]) == 18
    assert contrast.iloc[0].effect == -2.
    assert contrast.iloc[6].effect == 2.
    assert contrast.iloc[:9].relative_pct.isna().all()
    assert contrast.iloc[9:].p.isna().all() and contrast.iloc[9:].holm_p.isna().all()
    assert not contrast.iloc[9:].claim_supported.any()
    assert contrast.iloc[9:].relative_pct.notna().all()
    assert contrast.iloc[0].holm_p == pytest.approx(9 * 2 / 512)


def test_c2_missing_or_invalid_rmse_never_silently_changes_region_set():
    table, spec, definitions = fixture()
    with pytest.raises(ValueError, match="缺失"):
        analyze(table.iloc[1:], spec, definitions)
    table.loc[0, "status"] = "METRIC_NOT_ASSESSABLE"
    with pytest.raises(ValueError, match="不可评价"):
        analyze(table, spec, definitions)
