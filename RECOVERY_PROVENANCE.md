# Formal-result recovery provenance

Two accepted timing directories require different interpretations. Their
machine-readable records are retained inside the corresponding result
directories.

## Method ladder

`review_method_ladder_n10_local_20260724c_retry17_20260724T163927256Z_e393119e`
has `FORMAL_PROVENANCE.json` status `recovered_formal_result` and an explicit
`RECOVERY_OVERRIDE.json`. The target completed all 60/60 rows and ten semantic
groups, but no native `CLEAN_RUN.json` survived the wrapper-tail interruption.
The recovery proves completion, saved-output consistency, semantic agreement,
and a match to the retained code/config fingerprint at recovery time. It must
not be described as proving wrapper-certified zero interference or
wrapper-observed pre/post source identity throughout the timing interval.

## Parallel scaling

`parallel_scaling_n10_local_20260724c` also has
`FORMAL_PROVENANCE.json` status `recovered_formal_result`, but its recovery
class is narrower. A native `CLEAN_RUN.json` was successfully written after
runner exit code 0, with zero foreign Python processes and unchanged source
inputs. The old wrapper then failed only while creating a secondary sidecar
because an expected property was absent. The package therefore retains:

- the native clean-run record under `formal_isolation_records/`;
- the original `DO_NOT_USE` marker;
- `RECOVERY_OVERRIDE.json`;
- the recovered formal-provenance certificate.

The recovery overrides only the wrapper's post-success sidecar failure. It
does not erase or reinterpret the historical marker.
