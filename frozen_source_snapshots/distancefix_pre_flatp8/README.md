# Frozen source paired with the 2026-07-23 distance-fix evidence

This directory is a byte-for-byte source snapshot for the saved full-domain
result families produced before the later `Flat-P8` comparator was added. It is
kept so that provenance verification does not incorrectly compare historical
results with a newer source tree.

The snapshot is evidence-specific. It is not the current implementation and
must not replace the repository source tree.

Verified provenance values:

- deadline-runner code fingerprint:
  `1607cab3b931d8fd324de8d7446e3cfb1eb5b9653f35de0e7743a8762cbc233a`
- enhanced-evidence `src/` tree hash:
  `308a79a70edb6559f630ad9119ca0c8f87fecac7887fb3d6f1e5794d068e3ded`
- evaluator SHA-256:
  `c54155c207c48dec4ee1bb5653dcbbc53aabfa7aa2d5e50d3152ad989a30c5ea`

`scripts/02_verify_frozen_outputs.ps1` passes this snapshot explicitly to the
read-only enhanced verifier. New experiments use the current repository source
and record their own fingerprints.
