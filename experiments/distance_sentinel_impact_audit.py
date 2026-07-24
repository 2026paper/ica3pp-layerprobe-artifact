"""Read-only impact audit for the visible-distance sentinel repair.

The audit compares frozen saved outputs.  It never imports or executes the
simulator and deliberately excludes old/new wall-clock fields.  The primary
comparison is a key-by-key join over all 189,792 full-domain candidates.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPEC = Path(__file__).with_name(
    "distance_sentinel_impact_audit_spec.json"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "results" / "distance_sentinel_impact_audit_20260724_xeon"
)
AUDIT_SCRIPT = Path(__file__).resolve()

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
EXPECTED_CANDIDATE_FIELDS = (
    "kernel",
    "presentation",
    "signature_mask",
    "pairs_separated",
)
KERNEL_CHECK_HASH_FIELDS = (
    "oracle_trace_sha256",
    "flat_trace_sha256",
    "factorized_trace_sha256",
    "oracle_candidate_sha256",
    "sut_candidate_sha256",
    "factorized_candidate_sha256",
)
MISMATCH_FIELDS = (
    "validity_mismatch_count",
    "factorized_validity_mismatch_count",
    "flat_trace_mismatch_count",
    "factorized_trace_mismatch_count",
    "direct_candidate_mismatch_count",
    "factorized_candidate_mismatch_count",
)
FULL_KEY_ZERO_FIELDS = (
    "candidate_signature_mismatches",
    "trace_mismatches",
    "signature_bit_flips",
    "unsafe_cache_hits",
    "nontermination_guards",
)


class AuditError(RuntimeError):
    """Raised when a frozen input is structurally unusable."""


@dataclass(frozen=True)
class CandidateData:
    masks: dict[tuple[str, str], int]
    pairs_separated: dict[tuple[str, str], int]
    kernels: frozenset[str]
    presentations: frozenset[str]


@dataclass(frozen=True)
class KernelCheckData:
    ordered: tuple[dict[str, Any], ...]
    by_kernel: dict[str, dict[str, Any]]
    oracle_valid: frozenset[str]
    sut_valid: frozenset[str]
    factorized_valid: frozenset[str]
    recomputed_hashes: dict[str, str]


@dataclass(frozen=True)
class Gate:
    name: str
    passed: bool
    observed: Any
    expected: Any

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "observed": self.observed,
            "expected": self.expected,
        }


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def update_aggregate_hash(
    digest: Any,
    label: str,
    item_digest: str,
) -> None:
    """Match the independent oracle's aggregate-hash framing."""

    digest.update(label.encode("utf-8"))
    digest.update(b"\0")
    digest.update(item_digest.encode("ascii"))
    digest.update(b"\n")


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AuditError(f"cannot read JSON input {path}: {exc}") from exc
    require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def resolve_inside_project(relative_path: str) -> Path:
    candidate = (PROJECT_ROOT / relative_path).resolve()
    try:
        candidate.relative_to(PROJECT_ROOT.resolve())
    except ValueError as exc:
        raise AuditError(f"input escapes project root: {relative_path}") from exc
    return candidate


def validate_spec(spec: Mapping[str, Any]) -> None:
    require(spec.get("schema_version") == 1, "unsupported spec schema")
    expected = spec.get("expected")
    require(isinstance(expected, dict), "spec expected block is missing")
    inputs = spec.get("inputs")
    require(isinstance(inputs, list) and inputs, "spec inputs are missing")
    ids: set[str] = set()
    for item in inputs:
        require(isinstance(item, dict), "each frozen input must be an object")
        identifier = item.get("id")
        require(
            isinstance(identifier, str) and identifier not in ids,
            f"duplicate or invalid frozen input id: {identifier}",
        )
        ids.add(identifier)
        require(
            isinstance(item.get("path"), str),
            f"input {identifier} has no path",
        )
        require(
            isinstance(item.get("sha256"), str)
            and SHA256_RE.fullmatch(item["sha256"]) is not None,
            f"input {identifier} has an invalid SHA-256",
        )

    pairs = spec.get("model_pairs")
    require(
        isinstance(pairs, list)
        and len(pairs) == 6
        and all(isinstance(pair, list) and len(pair) == 2 for pair in pairs),
        "model-pair bit order must contain six pairs",
    )
    presentations = spec.get("presentations")
    require(
        isinstance(presentations, list)
        and len(presentations) == expected["presentations"],
        "presentation lattice does not match the frozen count",
    )
    presentation_names = [item["name"] for item in presentations]
    require(
        len(presentation_names) == len(set(presentation_names)),
        "presentation names are not unique",
    )

    by_presentation = spec.get("expected_candidate_changes_by_presentation")
    require(
        isinstance(by_presentation, dict)
        and set(by_presentation) == set(presentation_names),
        "expected per-presentation change distribution is incomplete",
    )
    require(
        sum(int(value) for value in by_presentation.values())
        == expected["changed_candidates"],
        "per-presentation candidate changes do not close",
    )
    hidden_changes = sum(
        int(by_presentation[item["name"]])
        for item in presentations
        if item["distance_mode"] == "hidden"
    )
    require(
        hidden_changes == expected["hidden_distance_changed_candidates"],
        "hidden-distance change count does not close",
    )

    transitions = spec.get("expected_mask_transitions")
    require(isinstance(transitions, list), "expected mask transitions missing")
    require(
        sum(int(item["count"]) for item in transitions)
        == expected["changed_candidates"],
        "mask-transition counts do not close",
    )
    directional_from_transitions: dict[str, dict[str, int]] = {
        f"{left}__{right}": {"zero_to_one": 0, "one_to_zero": 0}
        for left, right in pairs
    }
    for item in transitions:
        old_mask = int(item["old_mask"])
        new_mask = int(item["new_mask"])
        count = int(item["count"])
        require(old_mask != new_mask, "mask transition must be a change")
        for index, (left, right) in enumerate(pairs):
            old_bit = (old_mask >> index) & 1
            new_bit = (new_mask >> index) & 1
            key = f"{left}__{right}"
            if old_bit == 0 and new_bit == 1:
                directional_from_transitions[key]["zero_to_one"] += count
            elif old_bit == 1 and new_bit == 0:
                directional_from_transitions[key]["one_to_zero"] += count
    require(
        directional_from_transitions == spec.get("expected_pair_flips"),
        "pair-flip totals do not close against mask transitions",
    )

    trace_distribution = spec.get("expected_trace_step_delta_distribution")
    require(
        isinstance(trace_distribution, dict),
        "trace-step delta distribution missing",
    )
    require(
        sum(int(value) for value in trace_distribution.values())
        == expected["valid_kernels"],
        "trace-step delta distribution does not cover the valid set",
    )
    require(
        sum(int(delta) * int(count) for delta, count in trace_distribution.items())
        == expected["trace_step_delta"],
        "trace-step delta distribution does not sum to the frozen delta",
    )
    require(
        sum(
            int(count)
            for delta, count in trace_distribution.items()
            if int(delta) != 0
        )
        == expected["nonzero_trace_step_delta_kernels"],
        "nonzero trace-step delta kernel count does not close",
    )


def verify_frozen_inputs(
    spec: Mapping[str, Any],
) -> tuple[dict[str, Path], dict[str, Any]]:
    paths: dict[str, Path] = {}
    files: list[dict[str, Any]] = []
    for item in spec["inputs"]:
        path = resolve_inside_project(item["path"])
        require(path.is_file(), f"missing frozen input: {path}")
        observed = sha256_file(path)
        expected = item["sha256"]
        require(
            observed == expected,
            f"frozen input hash mismatch for {item['id']}: "
            f"{observed} != {expected}",
        )
        paths[item["id"]] = path
        files.append(
            {
                "id": item["id"],
                "path": item["path"],
                "role": item["role"],
                "size_bytes": path.stat().st_size,
                "expected_sha256": expected,
                "observed_sha256": observed,
                "verified": True,
            }
        )
    manifest = {
        "schema_version": 1,
        "audit_id": spec["audit_id"],
        "frozen_input_count": len(files),
        "all_frozen_hashes_match": True,
        "spec_path": str(DEFAULT_SPEC.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "spec_sha256": sha256_file(DEFAULT_SPEC),
        "audit_script_path": str(AUDIT_SCRIPT.relative_to(PROJECT_ROOT)).replace(
            "\\", "/"
        ),
        "audit_script_sha256": sha256_file(AUDIT_SCRIPT),
        "inputs": files,
    }
    return paths, manifest


def load_candidates(path: Path, pair_count: int) -> CandidateData:
    masks: dict[tuple[str, str], int] = {}
    pairs_separated: dict[tuple[str, str], int] = {}
    kernels: set[str] = set()
    presentations: set[str] = set()
    try:
        with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            require(
                tuple(reader.fieldnames or ()) == EXPECTED_CANDIDATE_FIELDS,
                f"unexpected candidate fields in {path}",
            )
            for line_number, row in enumerate(reader, start=2):
                kernel = row["kernel"]
                presentation = row["presentation"]
                key = (kernel, presentation)
                require(key not in masks, f"duplicate candidate key at {path}:{line_number}")
                try:
                    mask = int(row["signature_mask"])
                    separated = int(row["pairs_separated"])
                except ValueError as exc:
                    raise AuditError(
                        f"non-integer candidate value at {path}:{line_number}"
                    ) from exc
                require(
                    0 <= mask < (1 << pair_count),
                    f"candidate mask out of range at {path}:{line_number}",
                )
                require(
                    separated == mask.bit_count(),
                    f"pairs_separated mismatch at {path}:{line_number}",
                )
                masks[key] = mask
                pairs_separated[key] = separated
                kernels.add(kernel)
                presentations.add(presentation)
    except (OSError, gzip.BadGzipFile, UnicodeDecodeError) as exc:
        raise AuditError(f"cannot read candidate input {path}: {exc}") from exc
    return CandidateData(
        masks=masks,
        pairs_separated=pairs_separated,
        kernels=frozenset(kernels),
        presentations=frozenset(presentations),
    )


def expected_kernel_names(count: int) -> tuple[str, ...]:
    return tuple(f"brake_{index:04d}" for index in range(count))


def load_kernel_checks(
    path: Path,
    expected_count: int,
    summary: Mapping[str, Any],
) -> KernelCheckData:
    ordered: list[dict[str, Any]] = []
    by_kernel: dict[str, dict[str, Any]] = {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise AuditError(
                        f"invalid JSONL record at {path}:{line_number}"
                    ) from exc
                require(
                    isinstance(record, dict),
                    f"kernel check is not an object at {path}:{line_number}",
                )
                kernel = record.get("kernel")
                require(
                    isinstance(kernel, str) and kernel not in by_kernel,
                    f"duplicate or invalid kernel at {path}:{line_number}",
                )
                for field in KERNEL_CHECK_HASH_FIELDS:
                    require(
                        isinstance(record.get(field), str)
                        and SHA256_RE.fullmatch(record[field]) is not None,
                        f"invalid {field} at {path}:{line_number}",
                    )
                ordered.append(record)
                by_kernel[kernel] = record
    except (OSError, UnicodeDecodeError) as exc:
        raise AuditError(f"cannot read kernel checks {path}: {exc}") from exc

    names = expected_kernel_names(expected_count)
    require(len(ordered) == expected_count, f"kernel-check row count mismatch in {path}")
    require(
        tuple(record["kernel"] for record in ordered) == names,
        f"kernel-check order/domain mismatch in {path}",
    )

    oracle_valid = frozenset(
        record["kernel"] for record in ordered if record["oracle_valid"]
    )
    sut_valid = frozenset(
        record["kernel"] for record in ordered if record["sut_valid"]
    )
    factorized_valid = frozenset(
        record["kernel"] for record in ordered if record["factorized_valid"]
    )

    hashes = recompute_kernel_check_hashes(ordered)
    recorded_hashes = summary["comparison"]["hashes"]
    require(
        hashes == recorded_hashes,
        f"kernel-check aggregate hash mismatch in {path}",
    )
    recomputed_counts = recompute_kernel_check_counts(ordered)
    recorded_counts = summary["comparison"]["counts"]
    for field, observed in recomputed_counts.items():
        require(
            int(recorded_counts[field]) == observed,
            f"kernel-check aggregate count mismatch for {field} in {path}",
        )
    require(
        int(recorded_counts["requested_kernels"]) == expected_count,
        f"requested-kernel count mismatch in {path}",
    )
    return KernelCheckData(
        ordered=tuple(ordered),
        by_kernel=by_kernel,
        oracle_valid=oracle_valid,
        sut_valid=sut_valid,
        factorized_valid=factorized_valid,
        recomputed_hashes=hashes,
    )


def recompute_kernel_check_counts(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    summed_fields = (
        *MISMATCH_FIELDS,
        "trace_cases",
        "flat_trace_comparisons",
        "factorized_trace_comparisons",
        "candidate_comparisons",
        "oracle_trace_steps",
        "flat_trace_steps",
        "factorized_trace_steps",
    )
    result = {
        "processed_kernels": len(records),
        "oracle_valid_kernels": sum(bool(record["oracle_valid"]) for record in records),
        "sut_valid_kernels": sum(bool(record["sut_valid"]) for record in records),
        "factorized_valid_kernels": sum(
            bool(record["factorized_valid"]) for record in records
        ),
    }
    for field in summed_fields:
        result[field] = sum(int(record[field]) for record in records)
    return result


def recompute_kernel_check_hashes(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    digests = {
        "oracle_valid_kernel_sha256": hashlib.sha256(),
        "sut_valid_kernel_sha256": hashlib.sha256(),
        "factorized_valid_kernel_sha256": hashlib.sha256(),
        "oracle_trace_sha256": hashlib.sha256(),
        "flat_trace_sha256": hashlib.sha256(),
        "factorized_trace_sha256": hashlib.sha256(),
        "oracle_candidate_sha256": hashlib.sha256(),
        "sut_candidate_sha256": hashlib.sha256(),
        "factorized_candidate_sha256": hashlib.sha256(),
    }
    for record in records:
        kernel = record["kernel"]
        for prefix, valid_field in (
            ("oracle", "oracle_valid"),
            ("sut", "sut_valid"),
            ("factorized", "factorized_valid"),
        ):
            if record[valid_field]:
                digests[f"{prefix}_valid_kernel_sha256"].update(
                    (kernel + "\n").encode("utf-8")
                )
        for prefix in ("oracle", "flat", "factorized"):
            update_aggregate_hash(
                digests[f"{prefix}_trace_sha256"],
                kernel,
                record[f"{prefix}_trace_sha256"],
            )
        for prefix in ("oracle", "sut", "factorized"):
            update_aggregate_hash(
                digests[f"{prefix}_candidate_sha256"],
                kernel,
                record[f"{prefix}_candidate_sha256"],
            )
    return {name: digest.hexdigest() for name, digest in digests.items()}


def candidate_hashes_by_kernel(
    data: CandidateData,
    presentation_order: Sequence[str],
) -> dict[str, str]:
    result: dict[str, str] = {}
    for kernel in sorted(data.kernels):
        digest = hashlib.sha256()
        for presentation in presentation_order:
            key = (kernel, presentation)
            require(key in data.masks, f"candidate lattice hole at {key}")
            update_aggregate_hash(
                digest,
                f"{kernel}::{presentation}",
                sha256_value(data.masks[key]),
            )
        result[kernel] = digest.hexdigest()
    return result


def summarize_candidate_changes(
    old_masks: Mapping[tuple[str, str], int],
    new_masks: Mapping[tuple[str, str], int],
    presentations: Sequence[Mapping[str, Any]],
    pairs: Sequence[Sequence[str]],
) -> dict[str, Any]:
    require(set(old_masks) == set(new_masks), "candidate key sets are not identical")
    presentation_map = {item["name"]: item for item in presentations}
    changed_by_presentation: Counter[str] = Counter()
    bit_flips_by_presentation: Counter[str] = Counter()
    transitions: Counter[tuple[int, int]] = Counter()
    directional: dict[str, dict[str, int]] = {
        f"{left}__{right}": {"zero_to_one": 0, "one_to_zero": 0}
        for left, right in pairs
    }
    old_pair_counts: Counter[str] = Counter()
    new_pair_counts: Counter[str] = Counter()
    changed_by_kernel: Counter[str] = Counter()
    bit_flips_by_kernel: Counter[str] = Counter()

    for key in sorted(old_masks):
        kernel, presentation = key
        require(
            presentation in presentation_map,
            f"unknown presentation in candidate key: {presentation}",
        )
        old_mask = int(old_masks[key])
        new_mask = int(new_masks[key])
        for index, (left, right) in enumerate(pairs):
            pair_key = f"{left}__{right}"
            old_bit = (old_mask >> index) & 1
            new_bit = (new_mask >> index) & 1
            old_pair_counts[pair_key] += old_bit
            new_pair_counts[pair_key] += new_bit
            if old_bit == 0 and new_bit == 1:
                directional[pair_key]["zero_to_one"] += 1
            elif old_bit == 1 and new_bit == 0:
                directional[pair_key]["one_to_zero"] += 1
        if old_mask == new_mask:
            continue
        flips = (old_mask ^ new_mask).bit_count()
        changed_by_presentation[presentation] += 1
        bit_flips_by_presentation[presentation] += flips
        transitions[(old_mask, new_mask)] += 1
        changed_by_kernel[kernel] += 1
        bit_flips_by_kernel[kernel] += flips

    hidden_distance_changes = sum(
        changed_by_presentation[item["name"]]
        for item in presentations
        if item["distance_mode"] == "hidden"
    )
    changed_candidates = sum(changed_by_presentation.values())
    total_pair_bit_flips = sum(
        values["zero_to_one"] + values["one_to_zero"]
        for values in directional.values()
    )
    return {
        "changed_by_presentation": dict(changed_by_presentation),
        "bit_flips_by_presentation": dict(bit_flips_by_presentation),
        "transitions": dict(transitions),
        "directional_pair_flips": directional,
        "old_pair_counts": dict(old_pair_counts),
        "new_pair_counts": dict(new_pair_counts),
        "changed_by_kernel": dict(changed_by_kernel),
        "bit_flips_by_kernel": dict(bit_flips_by_kernel),
        "changed_candidates": changed_candidates,
        "unchanged_candidates": len(old_masks) - changed_candidates,
        "changed_kernels": len(changed_by_kernel),
        "hidden_distance_changes": hidden_distance_changes,
        "total_pair_bit_flips": total_pair_bit_flips,
    }


def full_key_order_is_zero(cache_summary: Mapping[str, Any], order: str) -> bool:
    row = cache_summary["fault_replay"]["full"][order]
    return (
        all(int(row[field]) == 0 for field in FULL_KEY_ZERO_FIELDS)
        and row["candidate_signature_digest"]
        == row["oracle_candidate_signature_digest"]
        and row["trace_digest"] == row["oracle_trace_digest"]
    )


def validate_range_summary(
    summary: Mapping[str, Any],
    expected: Mapping[str, int],
) -> bool:
    if summary.get("status") != "PASS" or summary.get("first_mismatch") is not None:
        return False
    if int(summary.get("requested_kernels", -1)) != expected["range_requested_kernels"]:
        return False
    if int(summary.get("presentations", -1)) != expected["presentations"]:
        return False
    if not all(bool(value) for value in summary.get("gates", {}).values()):
        return False
    results = summary.get("results", {})
    if set(results) != {"Flat", "LayerProbe-P8"}:
        return False
    for row in results.values():
        if int(row.get("valid_kernels", -1)) != expected["range_valid_kernels"]:
            return False
        if int(row.get("candidates", -1)) != expected["range_candidates"]:
            return False
    return (
        results["Flat"]["candidate_signature_sha256"]
        == results["LayerProbe-P8"]["candidate_signature_sha256"]
    )


def read_nonempty_lines(path: Path) -> tuple[str, ...]:
    return tuple(
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def add_gate(
    gates: list[Gate],
    name: str,
    observed: Any,
    expected: Any,
) -> None:
    gates.append(
        Gate(
            name=name,
            passed=observed == expected,
            observed=observed,
            expected=expected,
        )
    )


def write_csv(
    path: Path,
    fieldnames: Sequence[str],
    rows: Iterable[Mapping[str, Any]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def write_deterministic_gzip_csv(
    path: Path,
    fieldnames: Sequence[str],
    rows: Iterable[Mapping[str, Any]],
) -> None:
    with path.open("wb") as raw:
        with gzip.GzipFile(
            filename="",
            fileobj=raw,
            mode="wb",
            mtime=0,
        ) as compressed:
            with io.TextIOWrapper(
                compressed,
                encoding="utf-8",
                newline="",
            ) as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=fieldnames,
                    lineterminator="\n",
                )
                writer.writeheader()
                writer.writerows(rows)


def make_presentation_rows(
    impact: Mapping[str, Any],
    presentations: Sequence[Mapping[str, Any]],
    valid_kernels: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in presentations:
        name = item["name"]
        changed = int(impact["changed_by_presentation"].get(name, 0))
        rows.append(
            {
                "presentation": name,
                "speed_mode": item["speed_mode"],
                "distance_mode": item["distance_mode"],
                "delay": item["delay"],
                "candidates_compared": valid_kernels,
                "changed_candidates": changed,
                "unchanged_candidates": valid_kernels - changed,
                "change_rate": f"{changed / valid_kernels:.12f}",
                "pair_bit_flips": int(
                    impact["bit_flips_by_presentation"].get(name, 0)
                ),
            }
        )
    return rows


def make_pair_rows(
    impact: Mapping[str, Any],
    pairs: Sequence[Sequence[str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, (left, right) in enumerate(pairs):
        key = f"{left}__{right}"
        zero_to_one = impact["directional_pair_flips"][key]["zero_to_one"]
        one_to_zero = impact["directional_pair_flips"][key]["one_to_zero"]
        rows.append(
            {
                "pair_index": index,
                "agent_a": left,
                "agent_b": right,
                "old_separated_candidates": impact["old_pair_counts"].get(key, 0),
                "new_separated_candidates": impact["new_pair_counts"].get(key, 0),
                "zero_to_one": zero_to_one,
                "one_to_zero": one_to_zero,
                "total_bit_flips": zero_to_one + one_to_zero,
                "net_new_minus_old": zero_to_one - one_to_zero,
            }
        )
    return rows


def make_transition_rows(
    impact: Mapping[str, Any],
) -> list[dict[str, Any]]:
    changed = int(impact["changed_candidates"])
    rows: list[dict[str, Any]] = []
    for (old_mask, new_mask), count in sorted(impact["transitions"].items()):
        lost = (old_mask & ~new_mask).bit_count()
        gained = (new_mask & ~old_mask).bit_count()
        rows.append(
            {
                "old_mask": old_mask,
                "new_mask": new_mask,
                "old_pairs_separated": old_mask.bit_count(),
                "new_pairs_separated": new_mask.bit_count(),
                "lost_pair_bits": lost,
                "gained_pair_bits": gained,
                "pair_bit_flips": (old_mask ^ new_mask).bit_count(),
                "candidate_count": count,
                "share_of_changed_candidates": f"{count / changed:.12f}",
            }
        )
    return rows


def make_kernel_rows(
    old_checks: KernelCheckData,
    new_checks: KernelCheckData,
    impact: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for old_record, new_record in zip(
        old_checks.ordered,
        new_checks.ordered,
        strict=True,
    ):
        kernel = old_record["kernel"]
        require(kernel == new_record["kernel"], "kernel-check rows lost alignment")
        rows.append(
            {
                "kernel": kernel,
                "old_valid": int(bool(old_record["oracle_valid"])),
                "new_valid": int(bool(new_record["oracle_valid"])),
                "old_oracle_trace_steps": int(old_record["oracle_trace_steps"]),
                "new_oracle_trace_steps": int(new_record["oracle_trace_steps"]),
                "trace_step_delta": int(new_record["oracle_trace_steps"])
                - int(old_record["oracle_trace_steps"]),
                "trace_hash_changed": int(
                    old_record["oracle_trace_sha256"]
                    != new_record["oracle_trace_sha256"]
                ),
                "candidate_hash_changed": int(
                    old_record["oracle_candidate_sha256"]
                    != new_record["oracle_candidate_sha256"]
                ),
                "changed_candidates": int(
                    impact["changed_by_kernel"].get(kernel, 0)
                ),
                "pair_bit_flips": int(
                    impact["bit_flips_by_kernel"].get(kernel, 0)
                ),
                "old_oracle_trace_sha256": old_record["oracle_trace_sha256"],
                "new_oracle_trace_sha256": new_record["oracle_trace_sha256"],
                "old_oracle_candidate_sha256": old_record[
                    "oracle_candidate_sha256"
                ],
                "new_oracle_candidate_sha256": new_record[
                    "oracle_candidate_sha256"
                ],
            }
        )
    return rows


def build_report(summary: Mapping[str, Any]) -> str:
    comparison = summary["candidate_impact"]
    trace = summary["trace_impact"]
    oracle = summary["corrected_oracle"]
    cache = summary["full_key_control"]
    range_stress = summary["range_extension"]
    gate_rows = "\n".join(
        f"| {item['name']} | {'PASS' if item['passed'] else 'FAIL'} |"
        for item in summary["gates"]
    )
    return f"""# Distance-sentinel repair impact audit

## Result

**{summary["status"]}.** The audit joined every frozen full-domain candidate
one-to-one: {comparison["candidates_compared"]:,} / {comparison["candidates_compared"]:,}.
The repair changed {comparison["changed_candidates"]:,} candidate masks
({comparison["changed_candidate_rate"]:.3%}) across
{comparison["changed_kernels"]:,} / {comparison["valid_kernels"]:,} valid
kernels ({comparison["changed_kernel_rate"]:.3%}). The remaining
{comparison["unchanged_candidates"]:,} masks were identical.

The changed candidates are confined to presentations whose distance channel is
visible (exact or coarse). Presentations with `distance_mode=hidden` have
exactly {comparison["hidden_distance_changed_candidates"]} candidate changes.
This is the key sentinel-specific negative control.

The valid-kernel set is unchanged at {comparison["valid_kernels"]:,}. Across
all valid kernels, independent-oracle trace steps changed from
{trace["old_oracle_trace_steps"]:,} to {trace["new_oracle_trace_steps"]:,},
a signed delta of {trace["trace_step_delta"]:+,}. Trace hashes changed for
{trace["trace_hash_changed_kernels"]:,} kernels; only
{trace["nonzero_trace_step_delta_kernels"]:,} kernels changed trace length.

## Correctness closure

The corrected independent oracle reports {oracle["mismatch_total"]} aggregate
mismatches and detects {oracle["mutants_detected"]} of
{oracle["mutants_total"]} frozen semantic mutants. The complete-key cache
control has zero signature mismatches, zero trace mismatches, zero bit flips,
zero unsafe hits, and zero nontermination guards in both
`{cache["orders"][0]}` and `{cache["orders"][1]}` presentation orders.

The range-extension stress evidence preserves the same
{range_stress["valid_kernels"]} valid kernels and
{range_stress["candidates"]:,} candidates, and the corrected run passes every
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
{gate_rows}
"""


def run_audit(spec_path: Path, output: Path) -> dict[str, Any]:
    spec = load_json(spec_path)
    validate_spec(spec)
    require(
        spec_path.resolve() == DEFAULT_SPEC.resolve(),
        "this frozen audit accepts only the checked-in default spec",
    )
    paths, input_manifest = verify_frozen_inputs(spec)
    expected = spec["expected"]
    pairs = spec["model_pairs"]
    presentations = spec["presentations"]
    presentation_order = [item["name"] for item in presentations]

    old_communication_summary = load_json(paths["old_communication_summary"])
    new_communication_summary = load_json(paths["new_communication_summary"])
    old_oracle_summary = load_json(paths["old_oracle_summary"])
    new_oracle_summary = load_json(paths["new_oracle_summary"])
    old_cache_summary = load_json(paths["old_cache_summary"])
    new_cache_summary = load_json(paths["new_cache_summary"])
    old_range_summary = load_json(paths["old_range_summary"])
    new_range_summary = load_json(paths["new_range_summary"])
    enhanced_report = load_json(paths["new_enhanced_verifier_report"])

    old_candidates = load_candidates(
        paths["old_communication_candidates"],
        len(pairs),
    )
    new_candidates = load_candidates(
        paths["new_communication_candidates"],
        len(pairs),
    )
    old_checks = load_kernel_checks(
        paths["old_oracle_kernel_checks"],
        expected["requested_kernels"],
        old_oracle_summary,
    )
    new_checks = load_kernel_checks(
        paths["new_oracle_kernel_checks"],
        expected["requested_kernels"],
        new_oracle_summary,
    )
    impact = summarize_candidate_changes(
        old_candidates.masks,
        new_candidates.masks,
        presentations,
        pairs,
    )

    gates: list[Gate] = []
    add_gate(gates, "frozen_input_hashes_match", True, True)
    add_gate(
        gates,
        "communication_pair_bit_order_frozen",
        [
            old_communication_summary["model_pairs"],
            new_communication_summary["model_pairs"],
        ],
        [pairs, pairs],
    )
    add_gate(
        gates,
        "candidate_key_bijection_189792",
        {
            "old": len(old_candidates.masks),
            "new": len(new_candidates.masks),
            "same_keys": set(old_candidates.masks) == set(new_candidates.masks),
        },
        {
            "old": expected["candidates"],
            "new": expected["candidates"],
            "same_keys": True,
        },
    )
    add_gate(
        gates,
        "candidate_presentation_lattice_complete",
        {
            "old": sorted(old_candidates.presentations),
            "new": sorted(new_candidates.presentations),
        },
        {
            "old": sorted(presentation_order),
            "new": sorted(presentation_order),
        },
    )
    add_gate(
        gates,
        "valid_kernel_set_consistent_10544",
        {
            "old_oracle": len(old_checks.oracle_valid),
            "old_sut": len(old_checks.sut_valid),
            "old_factorized": len(old_checks.factorized_valid),
            "new_oracle": len(new_checks.oracle_valid),
            "new_sut": len(new_checks.sut_valid),
            "new_factorized": len(new_checks.factorized_valid),
            "candidate_old": len(old_candidates.kernels),
            "candidate_new": len(new_candidates.kernels),
            "all_sets_equal": len(
                {
                    old_checks.oracle_valid,
                    old_checks.sut_valid,
                    old_checks.factorized_valid,
                    new_checks.oracle_valid,
                    new_checks.sut_valid,
                    new_checks.factorized_valid,
                    old_candidates.kernels,
                    new_candidates.kernels,
                }
            )
            == 1,
        },
        {
            "old_oracle": expected["valid_kernels"],
            "old_sut": expected["valid_kernels"],
            "old_factorized": expected["valid_kernels"],
            "new_oracle": expected["valid_kernels"],
            "new_sut": expected["valid_kernels"],
            "new_factorized": expected["valid_kernels"],
            "candidate_old": expected["valid_kernels"],
            "candidate_new": expected["valid_kernels"],
            "all_sets_equal": True,
        },
    )
    expected_cartesian = {
        (kernel, presentation)
        for kernel in old_checks.oracle_valid
        for presentation in presentation_order
    }
    add_gate(
        gates,
        "candidate_cartesian_closure",
        {
            "expected_product": len(expected_cartesian),
            "old_exact": set(old_candidates.masks) == expected_cartesian,
            "new_exact": set(new_candidates.masks) == expected_cartesian,
        },
        {
            "expected_product": expected["candidates"],
            "old_exact": True,
            "new_exact": True,
        },
    )

    old_candidate_hashes = candidate_hashes_by_kernel(
        old_candidates,
        presentation_order,
    )
    new_candidate_hashes = candidate_hashes_by_kernel(
        new_candidates,
        presentation_order,
    )
    old_hash_mismatches = sum(
        digest != old_checks.by_kernel[kernel]["oracle_candidate_sha256"]
        for kernel, digest in old_candidate_hashes.items()
    )
    new_hash_mismatches = sum(
        digest != new_checks.by_kernel[kernel]["oracle_candidate_sha256"]
        for kernel, digest in new_candidate_hashes.items()
    )
    add_gate(gates, "candidate_to_oracle_hash_closure_old", old_hash_mismatches, 0)
    add_gate(gates, "candidate_to_oracle_hash_closure_new", new_hash_mismatches, 0)
    add_gate(
        gates,
        "kernel_check_aggregate_hash_closure_old",
        old_checks.recomputed_hashes,
        old_oracle_summary["comparison"]["hashes"],
    )
    add_gate(
        gates,
        "kernel_check_aggregate_hash_closure_new",
        new_checks.recomputed_hashes,
        new_oracle_summary["comparison"]["hashes"],
    )

    new_counts = new_oracle_summary["comparison"]["counts"]
    new_mismatch_total = sum(int(new_counts[field]) for field in MISMATCH_FIELDS)
    add_gate(
        gates,
        "new_oracle_zero_mismatches",
        new_mismatch_total,
        expected["new_oracle_mismatch_total"],
    )
    mutant = new_oracle_summary["mutant_smoke"]
    add_gate(
        gates,
        "new_oracle_mutants_7_of_7",
        {
            "total": mutant["mutants_total"],
            "detected": mutant["mutants_detected"],
            "all_detected": mutant["all_detected"],
            "undetected": mutant["undetected_mutants"],
        },
        {
            "total": expected["new_mutants_total"],
            "detected": expected["new_mutants_detected"],
            "all_detected": True,
            "undetected": [],
        },
    )

    old_steps = int(old_oracle_summary["comparison"]["counts"]["oracle_trace_steps"])
    new_steps = int(new_oracle_summary["comparison"]["counts"]["oracle_trace_steps"])
    add_gate(
        gates,
        "trace_steps_plus_337",
        {
            "old": old_steps,
            "new": new_steps,
            "delta": new_steps - old_steps,
        },
        {
            "old": expected["old_oracle_trace_steps"],
            "new": expected["new_oracle_trace_steps"],
            "delta": expected["trace_step_delta"],
        },
    )
    add_gate(
        gates,
        "trace_steps_cross_source_cache_closure",
        {
            "old_oracle": old_steps,
            "old_cache": int(old_cache_summary["oracle_contexts"]),
            "new_oracle": new_steps,
            "new_cache": int(new_cache_summary["oracle_contexts"]),
        },
        {
            "old_oracle": expected["old_oracle_trace_steps"],
            "old_cache": expected["old_oracle_trace_steps"],
            "new_oracle": expected["new_oracle_trace_steps"],
            "new_cache": expected["new_oracle_trace_steps"],
        },
    )

    add_gate(
        gates,
        "candidate_change_closure_2986",
        impact["changed_candidates"],
        expected["changed_candidates"],
    )
    add_gate(
        gates,
        "changed_kernel_closure_1579",
        impact["changed_kernels"],
        expected["changed_kernels"],
    )
    add_gate(
        gates,
        "hidden_distance_candidate_changes_zero",
        impact["hidden_distance_changes"],
        expected["hidden_distance_changed_candidates"],
    )
    add_gate(
        gates,
        "presentation_change_distribution_closure",
        {
            name: int(impact["changed_by_presentation"].get(name, 0))
            for name in presentation_order
        },
        spec["expected_candidate_changes_by_presentation"],
    )
    observed_transitions = [
        {"old_mask": old, "new_mask": new, "count": count}
        for (old, new), count in sorted(impact["transitions"].items())
    ]
    add_gate(
        gates,
        "mask_transition_distribution_closure",
        observed_transitions,
        spec["expected_mask_transitions"],
    )
    add_gate(
        gates,
        "pair_flip_distribution_closure",
        impact["directional_pair_flips"],
        spec["expected_pair_flips"],
    )

    kernel_rows = make_kernel_rows(old_checks, new_checks, impact)
    valid_kernel_rows = [row for row in kernel_rows if row["old_valid"]]
    trace_delta_distribution = Counter(
        int(row["trace_step_delta"]) for row in valid_kernel_rows
    )
    observed_delta_distribution = {
        str(delta): count for delta, count in sorted(trace_delta_distribution.items())
    }
    add_gate(
        gates,
        "trace_step_delta_distribution_closure",
        observed_delta_distribution,
        spec["expected_trace_step_delta_distribution"],
    )
    trace_hash_changed = sum(
        int(row["trace_hash_changed"]) for row in valid_kernel_rows
    )
    candidate_hash_changed = sum(
        int(row["candidate_hash_changed"]) for row in valid_kernel_rows
    )
    nonzero_step_delta = sum(
        int(row["trace_step_delta"]) != 0 for row in valid_kernel_rows
    )
    add_gate(
        gates,
        "kernel_trace_hash_change_closure",
        trace_hash_changed,
        expected["trace_hash_changed_kernels"],
    )
    add_gate(
        gates,
        "kernel_candidate_hash_change_closure",
        {
            "oracle_hash_changed": candidate_hash_changed,
            "candidate_join_changed": impact["changed_kernels"],
        },
        {
            "oracle_hash_changed": expected["changed_kernels"],
            "candidate_join_changed": expected["changed_kernels"],
        },
    )
    add_gate(
        gates,
        "nonzero_trace_step_delta_kernel_closure",
        nonzero_step_delta,
        expected["nonzero_trace_step_delta_kernels"],
    )

    for label, cache in (("old", old_cache_summary), ("new", new_cache_summary)):
        add_gate(
            gates,
            f"full_key_canonical_zero_{label}",
            full_key_order_is_zero(cache, "canonical"),
            True,
        )
        add_gate(
            gates,
            f"full_key_reverse_zero_{label}",
            full_key_order_is_zero(cache, "reverse"),
            True,
        )

    old_range_names = read_nonempty_lines(paths["old_range_valid_names"])
    new_range_names = read_nonempty_lines(paths["new_range_valid_names"])
    add_gate(
        gates,
        "range_extension_valid_set_consistent",
        {
            "old": len(old_range_names),
            "new": len(new_range_names),
            "same_ordered_names": old_range_names == new_range_names,
        },
        {
            "old": expected["range_valid_kernels"],
            "new": expected["range_valid_kernels"],
            "same_ordered_names": True,
        },
    )
    add_gate(
        gates,
        "range_extension_internal_exactness_pre_and_corrected",
        {
            "old": validate_range_summary(old_range_summary, expected),
            "new": validate_range_summary(new_range_summary, expected),
        },
        {"old": True, "new": True},
    )

    enhanced_summary = enhanced_report["summary"]
    enhanced_checks_all_pass = all(
        item.get("status") == "PASS" for item in enhanced_report["checks"]
    )
    add_gate(
        gates,
        "enhanced_verifier_provenance_v3_61_of_61",
        {
            "overall": enhanced_report["overall"],
            "passed": enhanced_summary["passed"],
            "failed": enhanced_summary["failed"],
            "total": enhanced_summary["total"],
            "all_check_rows_pass": enhanced_checks_all_pass,
            "mode": enhanced_report["verifier_mode"],
        },
        {
            "overall": "PASS",
            "passed": expected["enhanced_verifier_passed"],
            "failed": expected["enhanced_verifier_failed"],
            "total": expected["enhanced_verifier_total"],
            "all_check_rows_pass": True,
            "mode": "read_only_inputs_no_simulator_import_or_execution",
        },
    )

    require(not output.exists(), f"output already exists: {output}")
    output.mkdir(parents=True)
    write_json(output / "input_manifest.json", input_manifest)
    presentation_rows = make_presentation_rows(
        impact,
        presentations,
        expected["valid_kernels"],
    )
    pair_rows = make_pair_rows(impact, pairs)
    transition_rows = make_transition_rows(impact)
    write_csv(
        output / "candidate_change_by_presentation.csv",
        (
            "presentation",
            "speed_mode",
            "distance_mode",
            "delay",
            "candidates_compared",
            "changed_candidates",
            "unchanged_candidates",
            "change_rate",
            "pair_bit_flips",
        ),
        presentation_rows,
    )
    write_csv(
        output / "pair_flip_counts.csv",
        (
            "pair_index",
            "agent_a",
            "agent_b",
            "old_separated_candidates",
            "new_separated_candidates",
            "zero_to_one",
            "one_to_zero",
            "total_bit_flips",
            "net_new_minus_old",
        ),
        pair_rows,
    )
    write_csv(
        output / "mask_transitions.csv",
        (
            "old_mask",
            "new_mask",
            "old_pairs_separated",
            "new_pairs_separated",
            "lost_pair_bits",
            "gained_pair_bits",
            "pair_bit_flips",
            "candidate_count",
            "share_of_changed_candidates",
        ),
        transition_rows,
    )
    write_deterministic_gzip_csv(
        output / "kernel_trace_impact.csv.gz",
        (
            "kernel",
            "old_valid",
            "new_valid",
            "old_oracle_trace_steps",
            "new_oracle_trace_steps",
            "trace_step_delta",
            "trace_hash_changed",
            "candidate_hash_changed",
            "changed_candidates",
            "pair_bit_flips",
            "old_oracle_trace_sha256",
            "new_oracle_trace_sha256",
            "old_oracle_candidate_sha256",
            "new_oracle_candidate_sha256",
        ),
        kernel_rows,
    )

    hashed_output_names = (
        "candidate_change_by_presentation.csv",
        "pair_flip_counts.csv",
        "mask_transitions.csv",
        "kernel_trace_impact.csv.gz",
        "input_manifest.json",
    )
    output_artifacts = {
        name: {
            "size_bytes": (output / name).stat().st_size,
            "sha256": sha256_file(output / name),
        }
        for name in hashed_output_names
    }
    add_gate(
        gates,
        "output_artifact_hashes_recorded",
        {
            "files": sorted(output_artifacts),
            "valid_sha256": all(
                SHA256_RE.fullmatch(item["sha256"]) is not None
                for item in output_artifacts.values()
            ),
        },
        {"files": sorted(hashed_output_names), "valid_sha256": True},
    )

    all_passed = all(gate.passed for gate in gates)
    summary: dict[str, Any] = {
        "schema_version": 1,
        "audit_id": spec["audit_id"],
        "status": "PASS_distance_sentinel_impact_audit"
        if all_passed
        else "FAIL_distance_sentinel_impact_audit",
        "analysis_policy": {
            "saved_outputs_only": True,
            "simulator_imported_or_executed": False,
            "old_new_wall_clock_compared_or_emitted": False,
        },
        "candidate_impact": {
            "requested_kernels": expected["requested_kernels"],
            "valid_kernels": expected["valid_kernels"],
            "presentations": expected["presentations"],
            "candidates_compared": len(old_candidates.masks),
            "changed_candidates": impact["changed_candidates"],
            "unchanged_candidates": impact["unchanged_candidates"],
            "changed_candidate_rate": impact["changed_candidates"]
            / len(old_candidates.masks),
            "changed_kernels": impact["changed_kernels"],
            "changed_kernel_rate": impact["changed_kernels"]
            / expected["valid_kernels"],
            "hidden_distance_changed_candidates": impact[
                "hidden_distance_changes"
            ],
            "pair_bit_flips": impact["total_pair_bit_flips"],
            "mask_transition_types": len(impact["transitions"]),
            "affected_presentations": sum(
                count > 0 for count in impact["changed_by_presentation"].values()
            ),
        },
        "trace_impact": {
            "old_oracle_trace_steps": old_steps,
            "new_oracle_trace_steps": new_steps,
            "trace_step_delta": new_steps - old_steps,
            "trace_hash_changed_kernels": trace_hash_changed,
            "candidate_hash_changed_kernels": candidate_hash_changed,
            "nonzero_trace_step_delta_kernels": nonzero_step_delta,
            "trace_step_delta_distribution": observed_delta_distribution,
        },
        "corrected_oracle": {
            "trace_cases": int(new_counts["trace_cases"]),
            "candidate_comparisons": int(new_counts["candidate_comparisons"]),
            "mismatch_total": new_mismatch_total,
            "mutants_total": int(mutant["mutants_total"]),
            "mutants_detected": int(mutant["mutants_detected"]),
            "valid_kernel_sha256": new_oracle_summary["comparison"]["hashes"][
                "oracle_valid_kernel_sha256"
            ],
            "trace_sha256": new_oracle_summary["comparison"]["hashes"][
                "oracle_trace_sha256"
            ],
            "candidate_sha256": new_oracle_summary["comparison"]["hashes"][
                "oracle_candidate_sha256"
            ],
        },
        "full_key_control": {
            "orders": ["canonical", "reverse"],
            "old_both_zero": all(
                full_key_order_is_zero(old_cache_summary, order)
                for order in ("canonical", "reverse")
            ),
            "new_both_zero": all(
                full_key_order_is_zero(new_cache_summary, order)
                for order in ("canonical", "reverse")
            ),
        },
        "range_extension": {
            "requested_kernels": expected["range_requested_kernels"],
            "valid_kernels": expected["range_valid_kernels"],
            "candidates": expected["range_candidates"],
            "valid_set_unchanged": old_range_names == new_range_names,
            "old_candidate_signature_sha256": old_range_summary["results"][
                "Flat"
            ]["candidate_signature_sha256"],
            "new_candidate_signature_sha256": new_range_summary["results"][
                "Flat"
            ]["candidate_signature_sha256"],
            "corrected_internal_gates_all_pass": validate_range_summary(
                new_range_summary,
                expected,
            ),
        },
        "enhanced_verifier": {
            "provenance_version": "v3",
            "overall": enhanced_report["overall"],
            "passed": enhanced_summary["passed"],
            "failed": enhanced_summary["failed"],
            "total": enhanced_summary["total"],
        },
        "output_artifacts": output_artifacts,
        "gates": [gate.as_dict() for gate in gates],
        "gate_summary": {
            "passed": sum(gate.passed for gate in gates),
            "failed": sum(not gate.passed for gate in gates),
            "total": len(gates),
        },
        "claim_boundary": spec["claim_boundary"],
    }
    write_json(output / "summary.json", summary)
    (output / "DISTANCE_SENTINEL_IMPACT_AUDIT.md").write_text(
        build_report(summary),
        encoding="utf-8",
    )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit the saved-output impact of the distance-sentinel repair."
    )
    parser.add_argument(
        "--spec",
        type=Path,
        default=DEFAULT_SPEC,
        help="Checked-in frozen audit spec.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="New output directory; existing directories are rejected.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        summary = run_audit(args.spec.resolve(), args.output.resolve())
    except AuditError as exc:
        print(f"AUDIT ERROR: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": summary["status"],
                "output": str(args.output.resolve()),
                "candidates_compared": summary["candidate_impact"][
                    "candidates_compared"
                ],
                "changed_candidates": summary["candidate_impact"][
                    "changed_candidates"
                ],
                "changed_kernels": summary["candidate_impact"][
                    "changed_kernels"
                ],
                "hidden_distance_changes": summary["candidate_impact"][
                    "hidden_distance_changed_candidates"
                ],
                "trace_step_delta": summary["trace_impact"]["trace_step_delta"],
                "gates": summary["gate_summary"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if summary["gate_summary"]["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
