# LayerProbe anonymous reproducibility package

This offline package accompanies an anonymous ICA3PP submission. It contains
the implementation, frozen configurations, accepted single-workstation
evidence, figure-generation code, and reproducibility instructions. It does
not contain author identity, a public repository URL, or manuscript source.

## Scope

LayerProbe evaluates contract-scoped exact reuse over a finite braking-task
family and a second finite-state grid transfer audit. Reported timing evidence
is single-workstation evidence; it is not a multi-machine or cross-platform
performance claim. Computational-agent results are not human-subject or
learning-effect evidence.

## Environment

Create the pinned Conda environment:

```powershell
conda env create -f environment\environment.yml
conda activate layerprobe-artifact
```

Python 3.12 is required. See `environment/SYSTEM.md` and
`THIRD_PARTY_NOTICES.md`.

## Minimal reproduction

From the package root:

```powershell
.\scripts\reproduce_minimal.ps1 -PythonExe python
```

This runs the test suite and the small semantic smoke gates. It writes only to
a new directory under `reproduction_runs/`. Smoke results validate the
implementation but do not replace the frozen full-domain evidence.

## Verify packaged evidence

```powershell
.\scripts\verify_package.ps1
.\scripts\verify_recovery_evidence.ps1
.\scripts\02_verify_frozen_outputs.ps1 -PythonExe python
```

The first command checks the package manifest and hashes. The second reads and
rehashes the retained, sanitized recovery evidence. The third recomputes the
project-specific frozen-output gates against the paired frozen source snapshot.

## Full reproduction

Full single-host reproduction is compute-intensive and must be run without
unrelated Python workloads:

```powershell
.\scripts\reproduce_full.ps1 `
  -PythonExe python `
  -ResultTag reproduction
```

The full runner never overwrites an existing result directory. New timings
will naturally vary by machine and load; semantic digests and deterministic
work counts are the primary cross-run checks.

Validate the full-run entry point without starting the compute-intensive queue:

```powershell
.\scripts\reproduce_full.ps1 -PythonExe python -ValidateOnly
```

## Figures

Figure-generation sources are under `experiments/`; accepted rendered figures
are under `figures/`. The package does not redistribute the LNCS template or
manuscript source. Figure commands and their accepted inputs are documented in
`figures/README.md` and the result inventory.

Regenerate the final four figures without manuscript sources:

```powershell
.\scripts\03_regenerate_figures_and_paper.ps1 `
  -FiguresOnly `
  -PythonExe <python-with-matplotlib-pillow-pypdf> `
  -PrimaryRunDir results\parallel_scaling_n10_local_20260724c `
  -MethodLadderDir results\review_method_ladder_n10_local_20260724c_retry17_20260724T163927256Z_e393119e `
  -DeliberativeDir results\deliberative_policy_n10_local_20260724c `
  -SchedulerDir results\scheduler_sensitivity_n10_local_20260724c_retry2_20260724T182652510Z_956ac959
```

## Provenance and licensing

`MANIFEST.json` records every packaged file, role, byte count, and SHA-256.
`PROVENANCE_TRANSFORMATIONS.csv` records any metadata-only local-identifier
redaction with both original and packaged hashes. `SHA256SUMS.txt` covers all
payload files plus the manifest and transformation ledger.

Original software is MIT-licensed. Original configurations, result data, and
artifact documentation are CC BY 4.0-licensed. See `LICENSE`,
`licenses/LICENSE-CODE-MIT.txt`, and
`licenses/LICENSE-DATA-DOCS-CC-BY-4.0.txt`.
