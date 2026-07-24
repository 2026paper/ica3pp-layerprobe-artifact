# Formal timing isolation wrapper

`run_isolated_formal.ps1` is the required launcher for formal single-host
wall-clock runs. It does not make a busy workstation look clean: it first
requires a continuous 30-second interval with no detected Python-family
workload (including versioned names such as `python3.exe`, `python312.exe`, or
`python3.12.exe`). During the run, it polls the complete runner process tree.
If another Python-family workload appears, the wrapper records the foreign
process and stops only the identity-checked PIDs in its own runner tree.

No Python-family process is whitelisted. Read-only admission probes, test
runners, short-lived helper scripts, and long-running workloads are all
treated as competing work unless they belong to the wrapper's
identity-checked target process tree. The record is evidence of no *detected
foreign Python workload*, not a claim that every possible non-Python
background service was absent.

Example:

```powershell
.\scripts\run_isolated_formal.ps1 `
  -PythonExe $pythonExe `
  -ScriptPath experiments\deadline_runner.py `
  -ArgumentList @(
    "--config", "experiments\deadline_profile_review_8c32g.json",
    "--mode", "paper",
    "--only", "method_ladder",
    "--output", "results\review_method_ladder_n10_clean_20260724",
    "--ignore-freeze"
  ) `
  -OutputLabel method_ladder_n10 `
  -ExpectedOutputDir results\review_method_ladder_n10_clean_20260724 `
  -InputFiles @(
    "experiments\deadline_profile_review_8c32g.json",
    "src\layerprobe\__init__.py",
    "src\layerprobe\cli.py",
    "src\layerprobe\evaluator.py",
    "src\layerprobe\mechanics.py",
    "src\layerprobe\model.py",
    "src\layerprobe\workloads.py",
    "scripts\run_isolated_formal.ps1",
    "scripts\queue_formal_singlehost.ps1"
  ) `
  -PreflightStableSeconds 30 `
  -PollSeconds 1
```

Every invocation creates a non-overwriting directory below
`artifact_runs/formal_isolation/`. A clean isolation interval produces
`CLEAN_RUN.json` only when the target exits zero; runtime interference
produces `CONTAMINATED.json`; a failed preflight produces `REFUSED.json`; and
a nonzero target exit produces `PROGRAM_FAILED.json`. The companion
single-host queue preserves every contaminated directory, chooses a fresh
non-overwriting output suffix, and retries after the next clean launch
window. Standard output,
standard error, the invoked script, declared input files, and the result tree
are hashed. The source script and every declared input are hashed immediately
before launch and again after the target process tree has exited. A difference
or missing post-run input produces `SOURCE_INPUTS_CHANGED.json`, marks any
partial result `DO_NOT_USE`, and cannot produce `CLEAN_RUN.json`. A clean
record stores both binding snapshots and an explicit unchanged flag. The
single-host queue declares `src/layerprobe/cli.py` and both formal-control
PowerShell scripts for every formal role; the cost profile additionally
declares `pyproject.toml`.

A successful result directory receives `FORMAL_PROVENANCE.json`, which binds
it to the clean isolation record, the exact ordered target arguments, both
source/input snapshots, and the result tree. A contaminated, changed-input,
failed, or wrapper-failed partial output receives a `DO_NOT_USE_*.json`
marker.

Exit codes are:

- `0`: uncontaminated execution with a successful target;
- any other target-program exit code: clean interval, but failed target and
  therefore inadmissible output;
- `20`: preflight refusal because Python was already running;
- `21`: contamination detected after launch;
- `22`: source script or declared input changed during execution;
- `99`: wrapper or provenance-recording failure.

The default admissible path remains a result directory with
`FORMAL_PROVENANCE.json`, a matching result-tree hash, and a referenced
`CLEAN_RUN.json` with `runner_exit_code: 0`, no ignored or foreign Python
process, at least 30 stable preflight seconds, and identical pre/post source
bindings.

Two already-completed targets affected by old wrapper-tail defects have a
separate, explicit recovery path. Run
`scripts/recover_completed_formal_results.ps1 -VerifyOnly` to revalidate their
fixed row counts, progress and summary ledgers, semantic checks, target-code
fingerprint, result-tree hash, wrapper logs, and failure records. The generated
`RECOVERY_OVERRIDE.json` and `FORMAL_PROVENANCE.json` use schema 2 and the
status `recovered_formal_result`; they are never labeled `clean_run`. The
method-ladder certificate explicitly does not claim a native runner exit code,
zero foreign processes, or wrapper-observed pre/post source identity. The
parallel-scaling certificate binds an original native `CLEAN_RUN`, the
post-success `expected_output_dir` sidecar exception, and the original
`DO_NOT_USE` marker, which remains on disk. Release and figure gates accept
only these exact, hash-bound certificates and recovery classes.

The wrapper uses exact `Stop-Process -Id` calls for its own recorded process
tree and never uses a name-wide kill or `taskkill /T`.
