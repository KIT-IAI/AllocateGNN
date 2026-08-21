# Task-scale lu5 paper release

This directory is the sole numerical authority for the manuscript
"Task- and scale-matched evaluation of spatial allocation proxies for network
planning" scoped by manuscript archive SHA-256
`F7C1B9330909D7952E794D8184D39C0354CA69C5D336CD69D65CAC6285A6021B`.

## Reproducibility boundary

The retained artifacts reproduce manuscript tables, statistics, and the data
consumed by the two quantitative figures from derived frozen CSVs.  They do
not reproduce GNN training or held-out field generation.  The original lu5
held-out fields and checkpoints were not retained, and the later 013 release
uses a different data/generator snapshot.  It is therefore not substituted.

The frozen CSVs were originally generated in an internal research branch at
commit `b6f6f68158c389a843536ab4b7c898f8f53dc759` and were frozen into the
authoritative development repository on 2026-08-12. The public package was
exported from development commit `3e360e0`; no internal repository is required
at runtime.

## Included results

Only the twelve files declared in `manifest.json` are part of this release.
They cover the British diagnostic panel, both countries' four planning tasks,
station-centred scale sweeps, and the connection-bound audit.

The following historical artifacts are intentionally excluded because they do
not occur in the scoped manuscript: AU vintage decomposition, AU diagnostic
controls, grid-centred scale sweeps, and the earlier criterion, saturation,
dose-response, LORO, and MLP experiments.

## Frozen-summary exception

The LU-sharp alpha range (1.23--10.07; median 2.54, n=16) appears in the
manuscript, but its per-region lu5 artifact is unavailable.  The values are
retained in `model_contract.json` with status `frozen_summary_only`.  No tool in
this release claims to recompute them.

## Commands

```powershell
python -m SpatialPlacement.reproduce_paper_numbers --verify
```

The command reads no development result directory, checkpoint, held-out field,
or historical 10-feature graph.
