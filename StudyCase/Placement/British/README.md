# Britain case

Britain is the primary 16-region case. The manuscript evaluates peak
reconstruction, substation siting, rule-based sizing, and a stylised
large-demand connection proxy. The public release includes derived result
tables and method code, but not raw registers, lu5 held-out fields, or model
checkpoints.

See `study_materials/placement/case_facts/register_summary.json` for the
reported register facts and `protocol.json` for fixed settings.

## Execution order

1. `001_reconstruction.py` - peak reconstruction and per-seed results.
2. `002_siting_sizing.py` - p-median siting and both matching protocols.
3. `003_connection.py` - fixed-site connection-cost error.
4. `004_scale_analysis.py` - station-centred geographic-support sweep.
5. `005_insensitivity_audit.py` - connection-bound audit.
6. `006_diagnostic_panel.py` - nine-distribution RQ1 diagnostic panel.

Run a script from the repository root, for example:

```powershell
python StudyCase/Placement/British/002_siting_sizing.py
```
