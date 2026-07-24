# Grid transfer audit

This experiment is a second finite-state structural transfer case for the
LayerProbe contract. It is not a real-world benchmark and does not support a
cross-platform performance claim.

## Frozen domain

- 1,296 two-dimensional grid mechanisms
- state: `x`, `y`, `heading`, and logical `step`
- actions: forward, left turn, right turn, and wait
- three obstacle layouts and four finite horizons
- four deterministic agents with explicit memory
- 18 observation-only presentations:
  `x precision (exact/coarse/hidden)` ×
  `y precision (exact/coarse/hidden)` ×
  `delay (immediate/one-step)`

Only mechanisms whose goal is reachable within the declared horizon enter the
trace census. The report preserves both the declared and reachable counts.

## Evidence paths

For every reachable mechanism, the audit compares:

1. flat replay;
2. complete-key LayerProbe replay with
   `(world state, pre-ingest memory, observation)`;
3. an independently implemented naive interpreter; and
4. drop-state, drop-memory, and drop-observation cache mutants.

The complete key and all three mutants are replayed in canonical and reverse
presentation order. Passing requires zero complete-key/oracle differences and
an end-to-end semantic-trace witness for every mutant in both orders.

## Commands

Run tests first after the workstation is free:

```powershell
$pythonExe = (Get-Command python).Source
& $pythonExe -m pytest -q tests\test_grid_transfer_audit.py
```

Small smoke run:

```powershell
$pythonExe = (Get-Command python).Source
& $pythonExe experiments\grid_transfer_audit.py `
  --output results\smoke_grid_transfer_20260724 `
  --limit 16 --workers 4 --chunk-size 8
```

Full frozen domain:

```powershell
$pythonExe = (Get-Command python).Source
& $pythonExe experiments\grid_transfer_audit.py `
  --output results\grid_transfer_audit_20260724 `
  --workers 8 --chunk-size 32
```

Resume an interrupted run by repeating the full command with `--resume`.
Checkpoints are accepted only when the domain and source fingerprints match.

## Outputs

- `manifest.json`: host, command, source fingerprints, domain digest, and file
  hashes after completion
- `domain_manifest.json`: the exact mechanism, presentation, agent, key, and
  replay-order census used by the run
- `summary.json`: domain counts, trace/signature digests, work counters,
  cache-hit rate, and claim boundary
- `semantic_gate.json`: all pass/fail conditions
- `weak_key_witnesses.json`: expanded first replay witness for every weak key
  and order
- `mechanisms.csv`: per-mechanism verification, exactness, cache, and witness
  ledger
- `digest_ledger.csv`: per-mechanism trace/signature hashes
- `chunks/*.json`: atomic resumable checkpoints
