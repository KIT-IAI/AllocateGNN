\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
   
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon


def min_attainable_p(n: int) -> float:
\
\
\
\
\
       
    if n < 1:
        return 1.0
                                        
    return min(1.0, 2.0 ** (1 - n))


def bootstrap_ci(
    values: Sequence[float],
    n_boot: int = 20000,
    seed: int = 0,
    alpha: float = 0.05,
) -> Tuple[float, float]:
                                         
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), (n_boot, len(v)))
    means = v[idx].mean(axis=1)
    lo, hi = np.percentile(means, [alpha / 2 * 100, (1 - alpha / 2) * 100])
    return float(lo), float(hi)


def holm(pvalues: Dict[str, float]) -> Dict[str, float]:
                                                
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    out: Dict[str, float] = {}
    running = 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, (m - i) * p)
        running = max(running, adj)         
        out[k] = running
    return out


def paired_compare(
    df: pd.DataFrame,
    value_col: str,
    method_col: str = "method",
    unit_col: str = "region",
    pairs: Sequence[Tuple[str, str]] = (),
    n_boot: int = 20000,
    seed: int = 0,
    alpha: float = 0.05,
) -> pd.DataFrame:
\
\
\
\
\
\
\
\
\
\
\
\
       
    wide = df.pivot(index=unit_col, columns=method_col, values=value_col)

    raw: Dict[str, float] = {}
    rows: List[dict] = []
    for a, b in pairs:
        if a not in wide.columns or b not in wide.columns:
            continue
        sub = wide[[a, b]].dropna()
        n = len(sub)
        if n < 2:
            continue

        diff = sub[a].to_numpy(float) - sub[b].to_numpy(float)              
        improve = diff / sub[a].to_numpy(float) * 100.0                         

        try:
            _, p = wilcoxon(sub[a], sub[b], alternative="two-sided", method="exact")
        except ValueError:
            p = 1.0

        key = f"{a}_vs_{b}"
        raw[key] = float(p)
        lo, hi = bootstrap_ci(improve, n_boot=n_boot, seed=seed, alpha=alpha)

        rows.append({
            "pair": key,
            "n": n,
            "n_B_better": int((diff > 0).sum()),
            "median_diff": float(np.median(diff)),
            "mean_diff": float(diff.mean()),
            "improve_pct_mean": float(improve.mean()),
            "improve_pct_median": float(np.median(improve)),
            "ci_lo_pct": lo,
            "ci_hi_pct": hi,
            "p_wilcoxon": float(p),
            "min_attainable_p": min_attainable_p(n),
        })

    if not rows:
        return pd.DataFrame()

    adj = holm(raw)
    out = pd.DataFrame(rows)
    out["p_holm"] = out["pair"].map(adj)
    out["significant"] = out["p_holm"] < alpha
                                         
    out["underpowered"] = out["min_attainable_p"] >= alpha
    return out


def friedman_test(
    df: pd.DataFrame,
    value_col: str,
    methods: Sequence[str],
    method_col: str = "method",
    unit_col: str = "region",
) -> Dict[str, float]:
                         
    wide = df.pivot(index=unit_col, columns=method_col, values=value_col)
    cols = [m for m in methods if m in wide.columns]
    sub = wide[cols].dropna()
    if len(cols) < 3 or len(sub) < 3:
        return {}
    stat, p = friedmanchisquare(*[sub[c].to_numpy(float) for c in cols])
    return {"friedman_chi2": float(stat), "friedman_p": float(p), "n": int(len(sub))}
