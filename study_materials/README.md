# Study Materials

Supporting materials referenced as "available in the study materials" in
*Mechanism-Dependent Antagonism of Auxiliary Information in Substation-Level
Load Disaggregation for Distribution Network Planning* (Applied Energy,
manuscript APEN-D-26-11322). Each item below corresponds to a specific
statement in the paper or in the response letter.

## Contents

### `placement/`

Manuscript-scoped derived numerical evidence for *"Task- and scale-matched
evaluation of spatial allocation proxies for network planning"*. It contains
twelve allowlisted CSVs, hashes and schemas, case facts, protocol and model
contracts, and a reading-to-source ledger. Reproduce and verify the reported
numbers with:

```powershell
python -m StudyCase.Placement.scripts.reproduce_paper_numbers --verify
```

This is a derived-result reproduction package, not an end-to-end training
archive; see `placement/PROVENANCE.md` for the precise boundary.

### `environment-lock.yml`
Full machine-readable environment lock (conda export, 221 pinned
`package=version=build` entries) referenced in the Data Availability
statement. The software versions quoted in the paper (Python 3.13,
PyTorch 2.7.1 + CUDA 12.6, PyTorch Geometric 2.6.1, pandapower 3.5.4)
are taken from this file.

### `manifests/`
SHA-256 manifests of all frozen input files, one per study area
(referenced in the Data Availability statement). Each entry records the
relative path, byte size, and SHA-256 hash of one input file, so that
data obtained from the original providers (see the data-source table in
the paper) can be verified against the exact inputs used in the study.

| File | Coverage |
|---|---|
| `manifest_britain_v1.json` | 36 files (substations, regions, 16 regional grid-point and NTL freezes, graph caches) |
| `manifest_australia_v1.json` | 24 files (regions, grids, features, NTL, demand) |
| `manifest_germany_v1.json` | 7 files (regions, substations, grid points, NTL, assembled features, network distances) |

The raw datasets themselves are not redistributed here; they are
available from the providers listed in the paper's data-source table
under the licenses recorded there.

### `statistics/`
Region-level paired bootstrap confidence intervals (B = 10^4),
referenced in the statistical-protocol description: the bootstrap is
specified for confidence intervals only and is not used to derive
p-values.

| File | Coverage |
|---|---|
| `robust_tests.csv` | Britain: all 27 comparison-metric cells (9 planned comparisons x 3 metrics, 16 regions) with exact sign-flip permutation p-values, bootstrap CIs (`ci_lo`/`ci_hi`, `boot_B = 10000`), Moran's I diagnostics, and spatial block-permutation p-values |
| `au_hypothesis_tests.csv` | Australia: the same test family over 12 regions (exact 2^12 sign-flip enumeration, bootstrap CIs) |

### `frozen_artifacts/`
Frozen measurement of the softmax off-support gradient pull on the
trained dual-prior models (three seeds, 255 sources), referenced in the
response on loss-implementation details.

| File | Coverage |
|---|---|
| `kl_gradient_decomposition.json` | Per-signal (NTL / proximity) off-support gradient shares (median 1.9 %, p90 23 %, max 77 %), gated-pool edge counts (~8.6e4 edges per seed), and the loss-implementation cross-checks |
| `per_source_ratios.csv` | Per-source ratios behind the JSON summary (2 signals x 255 sources) |
