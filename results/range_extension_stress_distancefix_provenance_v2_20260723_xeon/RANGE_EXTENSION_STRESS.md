# Range-extension braking stress test

This is a frozen numeric range extension within the braking mechanism family.
It is neither a second domain nor a matched timing benchmark.

- Requested kernels: 432
- Valid kernels: 272
- Presentations: 18
- Candidates: 4,896
- Exact candidate-signature agreement: True
- Exact minimum-suite agreement: True

## Work accounting

| Quantity | Flat | LayerProbe-P8 | Reduction |
|---|---:|---:|---:|
| Graph builds | 7,776 | 432 | 94.444% |
| Policy calls | 120,096 | 87,310 | 27.300% |
| Transition calls | 120,096 | 87,310 | 27.300% |

Elapsed times in `summary.json` are diagnostic only and are not performance evidence.

## Gates

- factorized_graph_builds_equal_requested_kernels: PASS
- flat_graph_builds_equal_requested_kernels_times_presentations: PASS
- identical_candidate_signatures: PASS
- identical_minimum_suite: PASS
- identical_valid_kernel_names: PASS
