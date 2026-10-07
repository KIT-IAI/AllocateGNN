# Current paper: source and reproduction

**Task- and Scale-Matched Evaluation of Spatial Demand Allocation for Power Grid Planning**

This directory provides the source used by the revised paper and a portable
entry point for its published numerical evidence. Run the commands below from
the AllocateGNN repository root unless a different directory is specified.

## Versions

| Component | Version |
|---|---|
| Frozen paper results | `fixedload_20260925_r2` |
| Scientific pipeline release | `refactor_20260922_r2` |
| Scientific source commit | `69c4008732ed83dfb876020504baacbfa526d1aa` |
| Exported public engineering source | `333c32c2919f7e334c724e19b5331ce2c7e081a8` |
| Manuscript source at export | `9a4e17104f1f68c576bc9f3614d227612a768a57` |

The later engineering commit adds smoke-artifact verification and removes
dependencies on private documentation. It preserves the scientific modules
used by the paper. `SOURCE.json` records the exported Git files and the
paper scripts. Four figure-related scripts have recorded portability adaptations for input paths and exported acronym definitions. Their original and public hashes are listed separately. The original scientific source archive is also provided at
`study_materials/placement/current/code/upstream_release_69c4008732ed.zip`.
Its SHA-256 is `a8ef11fad856f8bbb549eeffb7782211f29181bfea9ba59ad50db1d7efa9f150`.

## Numerical reproduction

Python 3.11 or later is required. The original development environment used
Python 3.13. No GPU, private workspace or GIS files are needed for this command:

```bash
pip install -r StudyCase/Placement/current/requirements-current.txt
python -m SpatialPlacement.reproduce_paper_numbers --verify
```

The adapter imports the paper's statistical functions and the exported `sglib`
kernels. It recomputes regional task contrasts, the four-task Holm adjustment,
per-seed contrasts, control-panel associations, scale curves, conditional-bound
summaries, connection endpoints and the T25 sensitivity summary. It compares
the recomputed results with the frozen statistics and regenerates the claim
ledger. The supplied regional task CSVs and connection outputs are the inputs
to these calculations.

The output directory is `results/placement_current_reproduction/`. The default
entry point is shared by `StudyCase/Placement/000_reproduce_all.py` and the
numbered British and Australian case scripts.

The published subset has its own `study_materials/placement/current/manifest.json`.
The original `frozen/SHA256SUMS` is preserved for provenance and still lists two
candidate-level Parquet files omitted from this public subset.

The original exports from individual substation and planning records, the
GIS-derived Voronoi radius metadata and historical nine-member C4 comparison
remain frozen inputs. Their underlying geographic records are not reconstructed
by the lightweight command. The contemporary four-task statistics are recomputed.

## Figures from the public subset

```bash
python StudyCase/Placement/current/reproduce_figures.py --output results/placement_current_figures
```

This regenerates the pipeline diagram, scale curves and conditional-bound
curves with the paper's figure generators and portable input paths. The standalone acronym mapping is exported from the manuscript. The map generator
`scripts/make_fig123_maps.py` is included but needs the external geographic
inputs. LaTeX captions retain the manuscript's acronym macros.

## Full source pipeline

`upstream/` contains the complete public engineering checkout: `sglib/`,
`casestudy/`, source-data registration, configurations, examples and tests.
Its [README](upstream/README.md), [data guide](upstream/data/README.md) and
[data registration](upstream/data/metadata.toml) describe acquisition and training.
Install its full dependencies and execute its numbered stages from `upstream/`.
NL, NZ and DE support is preserved as part of that source export. The current
paper evaluates only Britain and Australia.

The generic upstream C4 connection experiment uses capacity-scaled loads.
To obtain this paper's results, use `scripts/paper2_fixed_load_connection.py`
with fixed 100, 300 and 500 MW loads, followed by `scripts/paper2_statistics.py`.
The latter imports the archived scientific kernels rather than a later installed
version. Reconstruction, siting and sizing reuse the frozen upstream task outputs.

Given the original external backup layout, the author-side commands are:

```bash
python StudyCase/Placement/current/scripts/paper2_fixed_load_connection.py --backup /path/to/backup --run-id new_fixedload_run
python StudyCase/Placement/current/scripts/paper2_statistics.py --backup /path/to/backup --run-id new_fixedload_run
python StudyCase/Placement/current/scripts/paper2_ledger.py --root /path/to/backup/recompute/new_fixedload_run --out /path/to/backup/recompute/new_fixedload_run/ledger
```

The external backup must provide `inputs/base/`, `inputs/release_r2/`,
`inputs/connection_pricing/`, the archived source under `code/`, and the original
active-input freeze receipt under `manifests/`. The consumed input hashes are
recorded in `study_materials/placement/current/frozen/inputs_manifest.csv` and
the statistical metadata. `close_active_inputs.py` preserves the original
author-side sealing procedure and additionally requires its source-copy receipts
and legacy Git history. It is not a public data downloader.

Raw and derived geographic inputs, trained checkpoints and private research
materials are not redistributed. Australian NSW location data are subject to
a data-sharing agreement. Their coordinates cannot be supplied as part of this
source release. A new download or retraining run is a new experiment until it
has been checked against the frozen input and output identities.

## Historical reproduction

```bash
python -m SpatialPlacement.reproduce_legacy_paper_numbers --verify
```

This command uses the earlier lu5 results retained directly under
`study_materials/placement/`. Those results and the old `SpatialPlacement/core/`
and `pipeline/` utilities remain available for historical compatibility. They
are not the numerical authority for the current paper.
