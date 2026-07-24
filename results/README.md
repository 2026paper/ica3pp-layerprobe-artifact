# Packaged results

Only result directories explicitly approved after `FINAL_FREEZE_READY` are
placed here. Frozen files are copied without numerical changes. If a text
metadata file contains a local absolute path or account identifier, the
packager substitutes a neutral token and records original and packaged hashes
in `../PROVENANCE_TRANSFORMATIONS.csv`.

`RESULTS_INVENTORY.csv` identifies each directory's scientific role, evidence
class, acceptance marker, and source-tree binding. Timings are
single-workstation observations; correctness and deterministic work-count
claims have separate verification gates.

A recovered formal result may retain its earlier `DO_NOT_USE` marker beside
`RECOVERY_OVERRIDE.json` and `FORMAL_PROVENANCE.json`. This is intentional:
the historical wrapper failure is not erased, and the recovery record states
the independent checks used to accept the immutable numerical payload.
