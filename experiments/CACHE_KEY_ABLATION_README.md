# Cache-key component ablation

This experiment tests whether any one component can be removed from
LayerProbe's semantic-step cache key on the frozen braking domain.

It is a separate fault-injection audit. It does not modify or monkeypatch
`src`, and it must not be described as an independent semantic checker or as a
universal proof that the key is minimal.

## Variants

| Variant | Key |
|---|---|
| `full` | `(world state, agent memory, observation)` |
| `drop_state` | `(agent memory, observation)` |
| `drop_memory` | `(world state, observation)` |
| `drop_observation` | `(world state, agent memory)` |

The order-independent census first enumerates true deterministic contexts. A
projected-key class is unsafe if it maps to more than one true output
`(action, next state, next memory, status)`. Actual weak-cache replay is then
performed in both canonical and reverse presentation order.

## Smoke test

From the repository root:

```powershell
$paperPython = (Get-Command python).Source
$env:PYTHONPATH = "$PWD\src"
& $paperPython experiments\cache_key_ablation.py `
  --mode smoke `
  --workers 8 `
  --output results\cache_key_ablation_smoke_20260723_xeon
```

Smoke output validates the runner only and is not paper evidence.

## Full frozen-domain run

Run only when the workstation has been reserved for the experiment:

```powershell
$paperPython = (Get-Command python).Source
$env:PYTHONPATH = "$PWD\src"
& $paperPython experiments\cache_key_ablation.py `
  --mode paper `
  --workers 8 `
  --output results\cache_key_ablation_full_24624_20260724_xeon
```

The CLI defaults to 8 workers and accepts up to 16. Eight workers are the
physical-core configuration; 16 workers must be labelled SMT/logical-worker
throughput if used.

Resume an interrupted run with exactly the same code, config, and options:

```powershell
& $paperPython experiments\cache_key_ablation.py `
  --mode paper `
  --workers 8 `
  --output results\cache_key_ablation_full_24624_20260724_xeon `
  --resume
```

Each process-pool task owns one kernel. Results are committed in atomic chunks
under `chunks/`; a resume reuses only chunks whose plan and fingerprint match.

## Outputs

- `summary.json`: exact aggregate collision and replay counts.
- `ablation_summary.csv`: table-ready rows for each variant and order.
- `counterexamples.json`: first complete collision, trace mismatch, and
  nontermination witnesses.
- `SUMMARY.md`: compact human-readable summary.
- `plan.json`, `progress.json`, `run_manifest.json`: plan, recovery state,
  hashes, machine provenance, and gates.
- `chunks/*.json`: atomic kernel-result chunks used for recovery and audit.

The full-key control must have zero unsafe classes, zero trace mismatches, zero
candidate-signature mismatches, zero unsafe cache hits, and zero
nontermination guards in both orders. Finalization fails closed otherwise.

## Paper interpretation

Permitted:

> Each of the three key components is empirically necessary, or
> component-wise irreducible, on the frozen finite domain.

Not permitted:

- “The key is universally minimal.”
- “This is an independent formal proof.”
- “Every other simulator requires the same key.”
