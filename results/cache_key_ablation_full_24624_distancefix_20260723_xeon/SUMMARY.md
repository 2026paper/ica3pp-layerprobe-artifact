# Cache-key ablation summary

- Status: `complete`
- Selected kernels: 24,624
- Valid kernels: 10,544
- Oracle semantic contexts: 3,382,177
- Full-key control: PASS

| Variant | Unsafe key classes | Affected valid kernels | Canonical trace mismatches | Reverse trace mismatches | Canonical signature mismatches | Reverse signature mismatches |
|---|---:|---:|---:|---:|---:|---:|
| full | 0 | 0 | 0 | 0 | 0 | 0 |
| drop_state | 389,642 | 10,025 | 231,493 | 229,742 | 14,637 | 19,192 |
| drop_memory | 45,727 | 7,054 | 2,264 | 3,036 | 1,238 | 1,433 |
| drop_observation | 271,784 | 10,544 | 354,772 | 291,930 | 91,075 | 91,089 |

The collision census is computed from oracle contexts and therefore does not depend on which presentation is replayed first. Fault replay is reported separately for canonical and reverse orders.

This experiment establishes component-wise necessity only on the selected finite domain and declared agents. It is not a universal minimal-key theorem or an independent semantic proof.
