# Distance-sentinel repair impact audit

## Result

**PASS_distance_sentinel_impact_audit.** The audit joined every frozen full-domain candidate
one-to-one: 189,792 / 189,792.
The repair changed 2,986 candidate masks
(1.573%) across
1,579 / 10,544 valid
kernels (14.975%). The remaining
186,806 masks were identical.

The changed candidates are confined to presentations whose distance channel is
visible (exact or coarse). Presentations with `distance_mode=hidden` have
exactly 0 candidate changes.
This is the key sentinel-specific negative control.

The valid-kernel set is unchanged at 10,544. Across
all valid kernels, independent-oracle trace steps changed from
3,381,840 to 3,382,177,
a signed delta of +337. Trace hashes changed for
7,890 kernels; only
390 kernels changed trace length.

## Correctness closure

The corrected independent oracle reports 0 aggregate
mismatches and detects 7 of
7 frozen semantic mutants. The complete-key cache
control has zero signature mismatches, zero trace mismatches, zero bit flips,
zero unsafe hits, and zero nontermination guards in both
`canonical` and `reverse` presentation orders.

The range-extension stress evidence preserves the same
272 valid kernels and
4,896 candidates, and the corrected run passes every
saved Flat/LayerProbe exactness gate. Its saved summaries do not contain raw
per-candidate masks, so the exact 189,792-candidate change census above is
deliberately restricted to the primary frozen domain.

## Manuscript-safe statement

> Repairing the signed-distance/missing-value collision preserved the 10,544
> valid-kernel set and changed 2,986 of 189,792 candidate signatures (1.573%)
> across 1,579 kernels. No signature changed when the distance channel was
> hidden. On the corrected outputs, the independent oracle had zero validity,
> trace, or candidate mismatches, detected all seven semantic mutants, and the
> full cache key remained exact in both presentation orders.

This is a semantic sensitivity and correctness audit. It does not compare or
emit old/new wall-clock measurements and does not support claims about human
learning, communication effectiveness, or users.

## Output tables

- `candidate_change_by_presentation.csv`: all 18 presentation conditions,
  including zero-change rows.
- `pair_flip_counts.csv`: directional bit flips for all six declared agent
  pairs.
- `mask_transitions.csv`: the complete changed-mask transition census.
- `kernel_trace_impact.csv.gz`: all 24,624 kernels, including invalid rows.
- `input_manifest.json`: frozen input paths, sizes, and verified SHA-256 values.

## Gates

| Gate | Status |
|---|---|
| frozen_input_hashes_match | PASS |
| communication_pair_bit_order_frozen | PASS |
| candidate_key_bijection_189792 | PASS |
| candidate_presentation_lattice_complete | PASS |
| valid_kernel_set_consistent_10544 | PASS |
| candidate_cartesian_closure | PASS |
| candidate_to_oracle_hash_closure_old | PASS |
| candidate_to_oracle_hash_closure_new | PASS |
| kernel_check_aggregate_hash_closure_old | PASS |
| kernel_check_aggregate_hash_closure_new | PASS |
| new_oracle_zero_mismatches | PASS |
| new_oracle_mutants_7_of_7 | PASS |
| trace_steps_plus_337 | PASS |
| trace_steps_cross_source_cache_closure | PASS |
| candidate_change_closure_2986 | PASS |
| changed_kernel_closure_1579 | PASS |
| hidden_distance_candidate_changes_zero | PASS |
| presentation_change_distribution_closure | PASS |
| mask_transition_distribution_closure | PASS |
| pair_flip_distribution_closure | PASS |
| trace_step_delta_distribution_closure | PASS |
| kernel_trace_hash_change_closure | PASS |
| kernel_candidate_hash_change_closure | PASS |
| nonzero_trace_step_delta_kernel_closure | PASS |
| full_key_canonical_zero_old | PASS |
| full_key_reverse_zero_old | PASS |
| full_key_canonical_zero_new | PASS |
| full_key_reverse_zero_new | PASS |
| range_extension_valid_set_consistent | PASS |
| range_extension_internal_exactness_pre_and_corrected | PASS |
| enhanced_verifier_provenance_v3_61_of_61 | PASS |
| output_artifact_hashes_recorded | PASS |
