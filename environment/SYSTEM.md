# LayerProbe single-workstation environment

This document records the reference environment and the minimum tooling for the
artifact entry points. It does not claim cross-machine performance
reproducibility.

## Reference workstation

The reported single-host runs were produced on:

- Windows 10 Pro 10.0.19045;
- Intel Xeon E5-2640 v3 at 2.60 GHz;
- 8 physical cores and 16 logical processors;
- approximately 31.8 GiB RAM;
- SATA storage;
- NVIDIA Quadro M4000, which is not used by LayerProbe;
- Python 3.12.7;
- Tectonic 0.16.9 for the LNCS manuscript.

The formal physical-core comparison stops at 8 workers. Results at 12 or 16
workers, where present, are SMT-throughput diagnostics rather than 12-core or
16-core scaling claims.

## Creating the software environment

From the repository root:

```powershell
conda env create -f environment\environment.yml
conda activate layerprobe-artifact
.\scripts\00_check_environment.ps1
```

The scripts set `PYTHONPATH` from their own location, so an editable package
installation is not required. A specific interpreter can be selected without
changing a script:

```powershell
.\scripts\00_check_environment.ps1 `
  -PythonExe <path-to-python.exe>
```

The same override is accepted by every artifact entry point. Alternatively,
set `LAYERPROBE_PYTHON`. Tectonic may be selected with `-TectonicExe` or
`LAYERPROBE_TECTONIC`.

## Required system resources

- PowerShell on Windows;
- Python 3.12 or newer with the packages in `environment.yml`;
- Tectonic for paper regeneration;
- Times New Roman installed by the operating system;
- at least 5 GiB free space for safe reproduction products;
- a new output directory for every invocation.

Times New Roman is not redistributed because it is a proprietary system font.
The figure scripts fail clearly if it is unavailable.

## Timing hygiene

Only isolated, matched runs should be used for timing claims. During a formal
timing run, pause synchronization software and other CPU-heavy work, leave at
least 8 GiB memory available, and do not run the smoke, figure, or verification
entry points concurrently. Correctness jobs may use concurrency for throughput,
but their elapsed times are not performance evidence.

## Scope

This is a single-workstation artifact. It contains no multi-node launcher and
makes no multi-machine scalability claim. The reproducibility entry points
cover environment inspection, a quick semantic smoke test, read-only
verification of frozen outputs, and staged figure/manuscript rebuilding.
