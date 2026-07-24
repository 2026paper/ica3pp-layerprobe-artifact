# Mechanism-parameter robustness heterogeneity

Overall: **PASS**

## Frozen question

Does presentation-robust computational distinguishability exhibit materially heterogeneous behavior across pre-specified mechanism-parameter strata in the frozen braking domain?

This is a complete saved-output census for the declared finite braking domain. It does not call the simulator and is not evidence about human learning, diagnostic accuracy, causality, or cross-domain behavior.

## Primary friction × start-speed strata

| Friction | Speed band | Requested | Valid | Validity | Robust nonzero | Mean robust pairs | Mean gap | Delay delta |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | low | 2736 | 2069 | 75.62% | 81.44% | 2.630 | 2.365 | -0.034 |
| 0 | medium | 2736 | 1927 | 70.43% | 95.17% | 3.179 | 1.821 | -0.237 |
| 0 | high | 2736 | 1098 | 40.13% | 96.99% | 3.270 | 1.730 | -0.236 |
| 1 | low | 2736 | 246 | 8.99% | 86.59% | 2.854 | 2.915 | -0.278 |
| 1 | medium | 2736 | 1604 | 58.63% | 92.14% | 3.132 | 2.807 | -0.257 |
| 1 | high | 2736 | 1671 | 61.07% | 94.79% | 3.309 | 2.652 | -0.306 |
| 2 | low | 2736 | 12 | 0.44% | 75.00% | 2.250 | 3.000 | -0.167 |
| 2 | medium | 2736 | 531 | 19.41% | 73.45% | 2.384 | 3.458 | -0.495 |
| 2 | high | 2736 | 1386 | 50.66% | 89.18% | 3.177 | 2.773 | -0.357 |

## Global reconciliation

- Requested/valid mechanisms: 24,624/10,544.
- All-presentation robust-nonzero mechanisms: 9,494.
- All-presentation robust-full mechanisms: 0.
- Aggregate matched delay delta: -0.240473781824 pairs.

## Frozen main-text promotion rule

- Robust-nonzero range: 23.548 percentage points (threshold 10.0; PASS).
- Smallest primary cell: 12 valid mechanisms (minimum 100; FAIL).
- Placement decision: `appendix_only`.

Secondary thresholds are descriptive support only and cannot replace the primary threshold. Complete marginal results are in `marginal_parameter_effects.csv`.

## Acceptance gates

- all frozen input hashes match: **PASS**
- independently reconstructed requested grid: **PASS**
- frozen presentation design: **PASS**
- candidate-table row count: **PASS**
- valid-kernel count: **PASS**
- each valid kernel has every presentation: **PASS**
- primary cells partition requested and valid kernels: **PASS**
- all-18 robust-nonzero reconciliation: **PASS**
- all-18 robust-full reconciliation: **PASS**
- aggregate delay reconciliation: **PASS**
- friction-zero reference/friction-blind negative control: **PASS**

## Provenance

- Spec SHA-256: `77bd5c1838e9c6555285f9493f85f56b5b3e77686d7e607444b664eae586bdb3`
- Script SHA-256: `b3f8823844469087a584c7594df3691d9d800d93a240e77c28ef3ee6038cf1d2`
- Candidate table SHA-256: `40e547e5ee06b03bc19e6f0cde3145dfbdbe442963fe0bd875261ec8c47cd162`

No adaptive bins, additional interactions, best-cell search, per-cell cover search, p-values, or sampled confidence intervals were used.
