> **Historical lu5 release.** This material describes an earlier manuscript.
> Use [current/](current/README.md) for the revised paper and its default reproduction command.

# Manuscript-scoped numerical package

- `results/`: the twelve allowlisted derived CSVs used by the manuscript.
- `case_facts/`: register facts reported in the experimental setup.
- `manuscript_readings.csv`: paper-only reading-to-source contract.
- `protocol.json`: fixed evaluation settings and conventions.
- `model_contract.json`: lu5 model and fold disclosure, including missing-artifact status.
- `manifest.json`: hashes, row counts, and schemas for every result CSV.
- `PROVENANCE.md`: origin, exclusions, and reproducibility boundary.

Run `python -m SpatialPlacement.reproduce_legacy_paper_numbers --verify`
from the repository root. Development repositories are not required.
