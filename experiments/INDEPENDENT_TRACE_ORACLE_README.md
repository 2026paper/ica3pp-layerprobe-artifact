# Independent trace oracle

This experiment is a second, deliberately separated implementation of the
frozen braking semantics. Its purpose is to detect common-mode errors that
cannot be excluded merely by comparing two evaluators which share the same
transition, observation, and agent-policy functions.

## Separation boundary

The reference side:

- imports only immutable data types from `layerprobe.model`;
- generates the frozen 24,624-kernel grid and 18 presentations locally from
  `independent_trace_oracle_config.json`;
- independently implements validation, transition, terminal status,
  observation/delay state, all four declared agents, trace construction, and
  six-bit candidate signatures;
- never calls a function or reads a constant from `layerprobe.mechanics`,
  `layerprobe.evaluator`, or `layerprobe.workloads`.

The system-under-test adapter is a separate section of the script. It loads the
existing mechanics and evaluator modules dynamically and compares:

1. the valid-mechanism decision;
2. every observable trace for all four agents and every selected presentation,
   against both the flat simulator and the complete-key memoized trace path;
3. direct candidate signatures reconstructed from those traces;
4. candidate signatures emitted by the factorized implementation.

Agreement is evidence from an independent differential implementation. It is
not a formal proof and must not be described as one.

## Mutation smoke

Six fixed semantic faults test whether the oracle can reject plausible unsafe
implementations:

1. an observation-only semantic cache key;
2. a cache shared across agents without agent identity in its scope;
3. delayed presentations returning the current rather than previous display;
4. coarse values rounded to nearest rather than floored;
5. a presentation mode changing braking dynamics;
6. an exclusive rather than inclusive goal end.

The mutation smoke stops each mutant at its first deterministic witness and
records the kernel, presentation, agent, first differing trace step, and both
trace hashes. Mutants are not run over the full domain.

## Commands

Run tests:

```powershell
# Run from the repository root.
$oraclePython = (Get-Command python).Source
$env:PYTHONPATH = "$PWD\src"
& $oraclePython -m pytest -p no:cacheprovider `
  tests\test_independent_trace_oracle.py
```

Run the small smoke (default 8 workers):

```powershell
& $oraclePython experiments\independent_trace_oracle.py `
  --config experiments\independent_trace_oracle_config.json `
  --output results\independent_trace_oracle_smoke_20260723 `
  --smoke `
  --workers 8
```

The script defaults to smoke mode. A future complete run requires the explicit
`--full-domain` flag. Worker counts above 16 are rejected.

```powershell
& $oraclePython experiments\independent_trace_oracle.py `
  --config experiments\independent_trace_oracle_config.json `
  --output results\independent_trace_oracle_full_20260724 `
  --full-domain `
  --workers 8
```

Do not run timing studies concurrently with this experiment. The oracle is a
correctness experiment, not a performance benchmark.
