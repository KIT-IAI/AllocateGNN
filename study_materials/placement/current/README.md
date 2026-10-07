# Current numerical evidence

This is the public subset of `fixedload_20260925_r2` for **Task- and Scale-Matched
Evaluation of Spatial Demand Allocation for Power Grid Planning**.

- `frozen/`: unchanged regional results, statistics, claim ledger and provenance.
- `code/upstream_release_69c4008732ed.zip`: exact archived scientific source.
- `manifest.json`: SHA-256 and byte size of every included evidence file.

The subset omits the two candidate-level Parquet files in the original frozen
manifest. `frozen/SHA256SUMS` is retained unchanged and is not a completeness
claim for this subset. Use this directory's `manifest.json` for public verification.

From the repository root, run:

```bash
python -m SpatialPlacement.reproduce_paper_numbers --verify
```

For commands, source provenance and the distinction between statistical
reproduction and recalculation from geographic inputs, see the
[current reproduction guide](../../../StudyCase/Placement/current/README.md).

The MIT code license applies to the source code. These derived research outputs
do not grant a license to redistribute their underlying provider datasets.
