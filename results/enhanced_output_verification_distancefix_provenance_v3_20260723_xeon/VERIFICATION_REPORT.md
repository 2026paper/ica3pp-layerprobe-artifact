# Enhanced evidence read-only audit

Overall: **PASS**

This verifier did not import or call the simulator. It recomputed the saved structural, arithmetic, hash, and claim-boundary checks.

| Evidence bundle | Check | Status |
|---|---|---:|
| independent_trace_oracle | required artifacts exist | **PASS** |
| independent_trace_oracle | full-domain status and mode | **PASS** |
| independent_trace_oracle | frozen full-domain totals and zero mismatches | **PASS** |
| independent_trace_oracle | no comparison witness exists | **PASS** |
| independent_trace_oracle | JSONL rows independently reproduce summary counts | **PASS** |
| independent_trace_oracle | every JSONL row has exact agreement | **PASS** |
| independent_trace_oracle | JSONL rows independently reproduce aggregate hashes | **PASS** |
| independent_trace_oracle | all aggregate digests are SHA-256 | **PASS** |
| independent_trace_oracle | mutant artifact matches embedded summary | **PASS** |
| independent_trace_oracle | all seven frozen semantic mutants are detected with witnesses | **PASS** |
| independent_trace_oracle | oracle static isolation boundary | **PASS** |
| independent_trace_oracle | oracle code/config/SUT hashes still match recorded run | **PASS** |
| independent_trace_oracle | recorded separation contract has no false fields | **PASS** |
| independent_trace_oracle | metadata cardinalities match frozen comparison | **PASS** |
| independent_trace_oracle | oracle claim boundary is explicit | **PASS** |
| cache_key_ablation | required artifacts exist | **PASS** |
| cache_key_ablation | chunk directory exists | **PASS** |
| cache_key_ablation | manifest embeds the exact frozen plan | **PASS** |
| cache_key_ablation | cache plan covers the complete frozen domain | **PASS** |
| cache_key_ablation | complete-grid selection hash | **PASS** |
| cache_key_ablation | cache code/config/source/run fingerprints | **PASS** |
| cache_key_ablation | cache run is fully complete | **PASS** |
| cache_key_ablation | run fingerprint is consistent across top-level artifacts | **PASS** |
| cache_key_ablation | chunk sequence and job identities are complete and unique | **PASS** |
| cache_key_ablation | manifest chunk hash | **PASS** |
| cache_key_ablation | manifest artifact hashes | **PASS** |
| cache_key_ablation | chunk-derived full-domain cardinalities | **PASS** |
| cache_key_ablation | chunk-derived collision census matches summary | **PASS** |
| cache_key_ablation | chunk-derived fault replay matches summary | **PASS** |
| cache_key_ablation | complete cache key is an exact zero-error control | **PASS** |
| cache_key_ablation | each deleted component fails in both replay orders with witnesses | **PASS** |
| cache_key_ablation | counterexamples are the earliest witnesses stored in chunks | **PASS** |
| cache_key_ablation | cache acceptance gates | **PASS** |
| cache_key_ablation | manifest repeats cache acceptance gates | **PASS** |
| cache_key_ablation | cache CSV exactly matches recomputed summary | **PASS** |
| agent_sensitivity | required artifacts exist | **PASS** |
| agent_sensitivity | required artifacts exist | **PASS** |
| agent_sensitivity | communication producer provenance | **PASS** |
| agent_sensitivity | saved-output full-domain cardinalities | **PASS** |
| agent_sensitivity | agent-analysis source hashes | **PASS** |
| agent_sensitivity | compressed signature table shape | **PASS** |
| agent_sensitivity | pair-level delay and robustness statistics recomputed | **PASS** |
| agent_sensitivity | leave-one-agent-out statistics and suites recomputed | **PASS** |
| agent_sensitivity | construct negative-control statistics recomputed | **PASS** |
| agent_sensitivity | friction-zero construct negative control | **PASS** |
| agent_sensitivity | non-zero-friction construct separation counts | **PASS** |
| agent_sensitivity | agent sensitivity gates recomputed | **PASS** |
| agent_sensitivity | pair_delay_sensitivity.csv exactly matches recomputation | **PASS** |
| agent_sensitivity | leave_one_agent_out.csv exactly matches recomputation | **PASS** |
| agent_sensitivity | construct_negative_control.csv exactly matches recomputation | **PASS** |
| agent_sensitivity | agent-analysis interpretation boundary is explicit | **PASS** |
| range_extension_stress | required artifacts exist | **PASS** |
| range_extension_stress | pre-frozen config bytes and hashes | **PASS** |
| range_extension_stress | stress script hash | **PASS** |
| range_extension_stress | stress core-source provenance | **PASS** |
| range_extension_stress | predeclared range-extension grid | **PASS** |
| range_extension_stress | range-extension status, exactness gates, and no mismatch | **PASS** |
| range_extension_stress | stress candidate exactness and expected cardinality | **PASS** |
| range_extension_stress | stress work-accounting invariants | **PASS** |
| range_extension_stress | stress valid-kernel name list | **PASS** |
| range_extension_stress | diagnostic timing is explicitly excluded from performance evidence | **PASS** |

Scope boundaries: the oracle is independent differential evidence on a finite domain, cache-key necessity is component-wise on that domain, agent sensitivity is not human-effect evidence, and stress-run elapsed times are diagnostic only.
