"""006 通用区域配对推断：区域主检验、Queen 诊断与固定 block 敏感性。"""

import hashlib
import numpy as np


def holm(pvalues):
    p = np.asarray(pvalues, dtype=float)
    if p.ndim != 1 or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Holm 输入必须为完整 family 的合法 p")
    order = np.argsort(p, kind="stable")
    adjusted = np.empty_like(p)
    adjusted[order] = np.minimum(1., np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    return adjusted


def sign_flip(diffs):
    d = np.asarray(diffs, dtype=float)
    if d.ndim != 1 or not len(d) or len(d) > 20 or not np.isfinite(d).all():
        raise ValueError("符号翻转单位无效")
    signs = 2*((np.arange(2**len(d))[:, None] >> np.arange(len(d))) & 1)-1
    tolerance = 1e-12*max(1., np.abs(d).sum())
    p = float(np.mean(np.abs(signs @ d) >= abs(d.sum())-tolerance))
    nonzero = int(np.count_nonzero(d))
    return {"p": p, "n": len(d), "nonzero": nonzero,
            "p_min": min(1., 2./2**nonzero) if nonzero else 1.}


def greedy_blocks(adjacency):
    adjacency = np.asarray(adjacency, bool)
    used, blocks = set(), []
    for i in range(len(adjacency)):
        if i in used:
            continue
        j = next((j for j in range(i+1, len(adjacency)) if j not in used and adjacency[i, j]), None)
        block = [i] if j is None else [i, j]
        blocks.append(block)
        used.update(block)
    return blocks


def _bootstrap(a, b, groups, *, seed, repetitions):
    sums_a = np.array([a[g].sum() for g in groups])
    sums_b = np.array([b[g].sum() for g in groups])
    sizes = np.array([len(g) for g in groups])
    indices = np.random.Generator(np.random.PCG64(seed)).integers(len(groups), size=(repetitions, len(groups)))
    aa, bb = sums_a[indices].sum(axis=1), sums_b[indices].sum(axis=1)
    native = (aa-bb)/sizes[indices].sum(axis=1)
    relative = 100.*(aa-bb)/np.where(bb != 0, bb, 1.)
    relative_ok = bool(b.sum() != 0 and np.all(bb != 0))
    return {"native_ci": np.quantile(native, [.025, .975], method="linear").tolist(),
            "relative_ci": np.quantile(relative, [.025, .975], method="linear").tolist() if relative_ok else None,
            "zero_denominator_draws": int(np.sum(bb == 0)),
            "indices_sha256": hashlib.sha256(indices.astype('<i8').tobytes()).hexdigest()}


def moran(diffs, adjacency, *, seed=20260906, permutations=999):
    d, adj = np.asarray(diffs, float), np.asarray(adjacency, bool)
    keep = adj.sum(axis=1) > 0
    result = {"islands": int((~keep).sum()), "n": int(keep.sum()), "I": None, "p": None, "alarm": False}
    if keep.sum() < 3:
        return {**result, "reason": "NO_ASSESSABLE_QUEEN_NEIGHBOURHOOD"}
    w = adj[np.ix_(keep, keep)].astype(float)
    w /= w.sum(axis=1, keepdims=True)
    z = d[keep]-d[keep].mean()
    denominator = float(z @ z)
    if denominator == 0:
        return {**result, "reason": "CONSTANT_PAIRED_DIFFERENCE"}
    def statistic(v):
        return float(v @ w @ v / denominator)
    value, expectation = statistic(z), -1./(len(z)-1)
    rng = np.random.Generator(np.random.PCG64(seed))
    extreme = sum(abs(statistic(rng.permutation(z))-expectation) >= abs(value-expectation)-1e-12 for _ in range(permutations))
    p = (extreme+1)/(permutations+1)
    return {**result, "I": value, "p": p, "alarm": p <= .05, "reason": ""}


def compare(a, b, adjacency, blocks, *, seed=20260906, repetitions=10000):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape or a.ndim != 1 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("配对指标形状或有限性不符")
    d = a-b
    main = _bootstrap(a, b, [[i] for i in range(len(a))], seed=seed, repetitions=repetitions)
    sensitive = _bootstrap(a, b, blocks, seed=seed, repetitions=repetitions)
    block_p = sign_flip([d[g].sum() for g in blocks])
    point = float(d.mean())
    lo, hi = sensitive["native_ci"]
    support = bool((point < 0 and hi < 0) or (point > 0 and lo > 0))
    diagnostic = moran(d, adjacency, seed=seed)
    percentage_valid = b != 0
    return {"effect": point, "relative_pct": float(100*d.sum()/b.sum()) if b.sum() else None,
            "relative_reason": "ZERO_ORIGINAL_BASELINE_SUM" if not b.sum() else "ZERO_BOOTSTRAP_BASELINE_SUM" if main["zero_denominator_draws"] else "",
            "median_region_pct": float(np.median(100*d[percentage_valid]/b[percentage_valid])) if percentage_valid.any() else None,
            "n_pct_valid": int(percentage_valid.sum()), "n": len(d), "improved": int((d < 0).sum()),
            "main": main, "sign_flip": sign_flip(d), "block": sensitive, "block_sign_flip": block_p,
            "block_support": support, "moran": diagnostic,
            "descriptive_only": bool(diagnostic["alarm"] and not support)}
