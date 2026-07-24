# Anonymization checklist

- [x] Package contains no author name, affiliation, email, or acknowledgement.
- [x] Package contains no public repository/archive URL.
- [x] Package excludes both Chinese and English manuscript sources.
- [x] Package excludes absolute local paths, Windows account names, and host
  identifiers.
- [x] Package excludes other-conference process traces and artifacts.
- [x] Package excludes virtual environments, caches, bytecode, general logs,
  raw profiler binaries, downloaded papers, private images, and venue
  templates. The only log exceptions are sanitized, manifest-listed machine
  evidence required to revalidate the two recovered formal results.
- [x] Only explicitly accepted result directories are included.
- [x] Rejected, partial, contaminated, and smoke-only results are excluded.
- [x] A historical `DO_NOT_USE` marker is retained only when paired with an
  explicit recovered-result provenance record and recovery override.
- [x] Metadata-only redactions preserve original and packaged hashes in
  `PROVENANCE_TRANSFORMATIONS.csv`.
- [x] Frozen configurations and the source snapshot paired with historical
  evidence are included.
- [x] Minimal and full reproduction commands write to new directories.
- [x] Dual licensing separates software from configurations/data/docs.
- [x] `MANIFEST.json` and `SHA256SUMS.txt` validate successfully.
- [x] Final text and filename scan reports zero blocking identity/path hits.
- [x] No upload, Git push, email, or remote-repository access was performed.

Before a camera-ready release, replace anonymous attribution wording with the
final author attribution and re-run the package builder.
