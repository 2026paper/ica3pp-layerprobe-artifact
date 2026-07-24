# Declared-agent sensitivity analysis

This saved-output analysis covers 10,544 valid mechanisms and 189,792 mechanism-presentation candidates.
It does not re-run the simulator and does not constitute human learning, diagnostic, or communication-effect evidence.

## Delay effect by declared agent pair

| Pair | Candidate separation | all-18 robust kernels | Immediate rate | Delayed rate | Delta |
|---|---:|---:|---:|---:|---:|
| reference vs instant_stop | 66.67% | 1855 | 69.67% | 63.67% | -0.0600 |
| reference vs speed_only | 95.38% | 8993 | 96.82% | 93.94% | -0.0288 |
| reference vs friction_blind | 36.17% | 2168 | 35.83% | 36.51% | +0.0067 |
| instant_stop vs speed_only | 93.22% | 8609 | 94.29% | 92.15% | -0.0214 |
| instant_stop vs friction_blind | 73.71% | 3486 | 76.54% | 70.89% | -0.0566 |
| speed_only vs friction_blind | 88.08% | 6998 | 92.11% | 84.06% | -0.0805 |

## Leave-one-agent-out robustness

| Omitted agent | Retained pairs | Delay delta | all-18 robust full | all-18 robust suite |
|---|---:|---:|---:|---:|
| reference | 3 | -0.1584 | 1362 | 1 |
| instant_stop | 3 | -0.1026 | 404 | 1 |
| speed_only | 3 | -0.1098 | 168 | 1 |
| friction_blind | 3 | -0.1102 | 925 | 1 |

## Construct-validity negative control

The reference and friction-blind agents are definitionally equivalent when friction is zero. Their separation bit must therefore remain zero.

| Friction | Separated candidates | Total candidates | Rate |
|---:|---:|---:|---:|
| 0 | 0 | 91692 | 0.000% |
| 1 | 41611 | 63378 | 65.655% |
| 2 | 27036 | 34722 | 77.864% |

## Interpretation boundary

- Pair decomposition tests whether the aggregate delay result is dominated by one declared pair.
- Leave-one-agent-out results test directional sensitivity to removing one hand-written agent; they do not establish population robustness.
- A positive pair-level delta is not evidence that delay helps people. It means only that this deterministic pair diverged more often.
