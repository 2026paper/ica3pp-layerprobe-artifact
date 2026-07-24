"""Strict read-only audit for the four enhanced LayerProbe evidence bundles.

The verifier never imports or calls the simulator.  It independently reads the
saved artifacts, recomputes their structural and arithmetic invariants, checks
recorded hashes, and writes a JSON/Markdown audit report to a new directory.
Any failed check produces a non-zero exit status.
"""

from __future__ import annotations

import argparse
import ast
import csv
import gzip
import hashlib
import json
import math
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from itertools import combinations, product
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "results"
SOURCE_ROOT = PROJECT_ROOT / "src"

DEFAULT_ORACLE_DIR = (
    RESULTS_ROOT / "independent_trace_oracle_full_24624_20260723_xeon"
)
DEFAULT_CACHE_DIR = (
    RESULTS_ROOT / "cache_key_ablation_full_24624_20260723_xeon"
)
DEFAULT_AGENT_DIR = (
    RESULTS_ROOT / "agent_sensitivity_full_24624_20260723_xeon"
)
DEFAULT_STRESS_DIR = RESULTS_ROOT / "range_extension_stress_20260723_xeon"
DEFAULT_COMMUNICATION_DIR = (
    RESULTS_ROOT / "communication_full_24624_20260723_xeon"
)

ORACLE_SCRIPT = PROJECT_ROOT / "experiments" / "independent_trace_oracle.py"
ORACLE_CONFIG = (
    PROJECT_ROOT / "experiments" / "independent_trace_oracle_config.json"
)
CACHE_SCRIPT = PROJECT_ROOT / "experiments" / "cache_key_ablation.py"
CACHE_CONFIG = (
    PROJECT_ROOT / "experiments" / "cache_key_ablation_profile_8c32g.json"
)
AGENT_SCRIPT = PROJECT_ROOT / "experiments" / "agent_sensitivity_analysis.py"
COMMUNICATION_SCRIPT = PROJECT_ROOT / "experiments" / "communication_analysis.py"
STRESS_SCRIPT = PROJECT_ROOT / "experiments" / "range_extension_stress.py"
STRESS_CONFIG = (
    PROJECT_ROOT / "experiments" / "range_extension_stress_config.json"
)

EXPECTED_AGENTS = (
    "reference",
    "instant_stop",
    "speed_only",
    "friction_blind",
)
EXPECTED_PAIRS = tuple(combinations(EXPECTED_AGENTS, 2))
EXPECTED_MUTANTS = (
    "cache_key_observation_only",
    "cache_scope_cross_agent",
    "delay_returns_current",
    "coarse_rounds_to_nearest",
    "presentation_intervenes_on_dynamics",
    "goal_end_is_exclusive",
    "signed_distance_missing_collision",
)
EXPECTED_CACHE_VARIANTS = (
    "full",
    "drop_state",
    "drop_memory",
    "drop_observation",
)
EXPECTED_CACHE_ORDERS = ("canonical", "reverse")
EXPECTED_PRIMARY_KERNELS = 24_624
EXPECTED_VALID_KERNELS = 10_544
EXPECTED_PRESENTATIONS = 18
EXPECTED_TRACES = 759_168
EXPECTED_CANDIDATES = 189_792
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
KERNEL_RE = re.compile(r"^brake_(\d+)$")
STRESS_KERNEL_RE = re.compile(r"^stress_brake_(\d+)$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_source_tree(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*.py") if item.is_file()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        data = path.read_bytes()
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


def stable_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def stable_digest(value: Any) -> str:
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def is_sha256(value: Any) -> bool:
    return isinstance(value, str) and bool(SHA256_RE.fullmatch(value))


def update_labeled_hash(
    digest: Any,
    label: str,
    item_digest: str,
) -> None:
    digest.update(label.encode("utf-8"))
    digest.update(b"\0")
    digest.update(item_digest.encode("ascii"))
    digest.update(b"\n")


def close_json_array_hash(digest: Any, item_count: int) -> str:
    del item_count
    digest.update(b"]")
    return digest.hexdigest()


def start_json_array_hash() -> Any:
    digest = hashlib.sha256()
    digest.update(b"[")
    return digest


def append_json_array_item(digest: Any, item: Any, item_count: int) -> None:
    if item_count:
        digest.update(b",")
    digest.update(stable_json(item).encode("utf-8"))


def values_match(left: Any, right: Any, *, tolerance: float = 1e-12) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(
            float(left),
            float(right),
            rel_tol=0.0,
            abs_tol=tolerance,
        )
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(
            values_match(left[key], right[key], tolerance=tolerance)
            for key in left
        )
    if isinstance(left, Sequence) and isinstance(right, Sequence):
        if isinstance(left, (str, bytes)) or isinstance(right, (str, bytes)):
            return left == right
        return len(left) == len(right) and all(
            values_match(a, b, tolerance=tolerance)
            for a, b in zip(left, right)
        )
    return left == right


def compact_detail(value: Any, limit: int = 500) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value if len(value) <= limit else value[: limit - 3] + "..."
    encoded = json.dumps(value, ensure_ascii=False, default=str)
    return value if len(encoded) <= limit else encoded[: limit - 3] + "..."


@dataclass
class Audit:
    checks: list[dict[str, Any]] = field(default_factory=list)

    def add(
        self,
        group: str,
        name: str,
        passed: bool,
        *,
        observed: Any = None,
        expected: Any = None,
        detail: Any = None,
    ) -> None:
        row: dict[str, Any] = {
            "group": group,
            "check": name,
            "status": "PASS" if passed else "FAIL",
        }
        if observed is not None:
            row["observed"] = compact_detail(observed)
        if expected is not None:
            row["expected"] = compact_detail(expected)
        if detail is not None:
            row["detail"] = compact_detail(detail)
        self.checks.append(row)

    def equal(
        self,
        group: str,
        name: str,
        observed: Any,
        expected: Any,
        *,
        tolerance: float = 1e-12,
        detail: Any = None,
    ) -> None:
        self.add(
            group,
            name,
            values_match(observed, expected, tolerance=tolerance),
            observed=observed,
            expected=expected,
            detail=detail,
        )

    @property
    def passed(self) -> bool:
        return bool(self.checks) and all(
            row["status"] == "PASS" for row in self.checks
        )


def require_files(
    audit: Audit,
    group: str,
    directory: Path,
    names: Sequence[str],
) -> bool:
    missing = [name for name in names if not (directory / name).is_file()]
    audit.add(
        group,
        "required artifacts exist",
        not missing,
        observed={"directory": str(directory), "missing": missing},
        expected={"missing": []},
    )
    return not missing


def inspect_oracle_isolation(script_path: Path) -> tuple[bool, dict[str, Any]]:
    source = script_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(script_path))
    layerprobe_imports: list[str] = []
    non_model_imports: list[str] = []
    imported_model_names: list[str] = []
    forbidden_calls: list[dict[str, Any]] = []
    expected_model_names = {
        "Action",
        "AgentMemory",
        "DisplayMemory",
        "KernelSpec",
        "Observation",
        "PresentationSpec",
        "Trace",
        "WorldState",
    }

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("layerprobe"):
                    layerprobe_imports.append(alias.name)
                    if alias.name != "layerprobe.model":
                        non_model_imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module.startswith("layerprobe"):
                layerprobe_imports.append(node.module)
                if node.module != "layerprobe.model":
                    non_model_imports.append(node.module)
                else:
                    imported_model_names.extend(alias.name for alias in node.names)

    oracle_function_names: list[str] = []
    forbidden_call_names = {
        "load_system_under_test",
        "import_module",
        "simulate_flat",
        "run_flat",
        "run_factorized",
        "_memoized_agent_traces",
    }
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not node.name.startswith("oracle_"):
            continue
        oracle_function_names.append(node.name)
        for child in ast.walk(node):
            if not isinstance(child, ast.Call):
                continue
            called: str | None = None
            if isinstance(child.func, ast.Name):
                called = child.func.id
            elif isinstance(child.func, ast.Attribute):
                called = child.func.attr
            if called in forbidden_call_names:
                forbidden_calls.append(
                    {
                        "function": node.name,
                        "call": called,
                        "line": child.lineno,
                    }
                )

    adapter_functions = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in {"load_system_under_test", "compare_kernel_task"}
    }
    passed = (
        not non_model_imports
        and set(imported_model_names) == expected_model_names
        and not forbidden_calls
        and {"load_system_under_test", "compare_kernel_task"}
        <= adapter_functions
        and bool(oracle_function_names)
    )
    return passed, {
        "layerprobe_imports": sorted(set(layerprobe_imports)),
        "model_names": sorted(set(imported_model_names)),
        "oracle_function_count": len(oracle_function_names),
        "forbidden_oracle_calls": forbidden_calls,
        "adapter_functions": sorted(adapter_functions),
    }


def verify_oracle(
    audit: Audit,
    directory: Path,
    script_path: Path,
    config_path: Path,
) -> None:
    group = "independent_trace_oracle"
    required = (
        "summary.json",
        "metadata.json",
        "mutant_smoke.json",
        "kernel_checks.jsonl",
    )
    if not require_files(audit, group, directory, required):
        return

    summary = load_json(directory / "summary.json")
    metadata = load_json(directory / "metadata.json")
    mutants = load_json(directory / "mutant_smoke.json")
    comparison = summary.get("comparison", {})
    counts = comparison.get("counts", {})
    hashes = comparison.get("hashes", {})

    audit.equal(
        group,
        "full-domain status and mode",
        (summary.get("status"), summary.get("mode")),
        ("PASS_independent_trace_oracle_full_domain", "full_domain"),
    )
    expected_counts = {
        "requested_kernels": EXPECTED_PRIMARY_KERNELS,
        "processed_kernels": EXPECTED_PRIMARY_KERNELS,
        "oracle_valid_kernels": EXPECTED_VALID_KERNELS,
        "sut_valid_kernels": EXPECTED_VALID_KERNELS,
        "factorized_valid_kernels": EXPECTED_VALID_KERNELS,
        "validity_mismatch_count": 0,
        "factorized_validity_mismatch_count": 0,
        "trace_cases": EXPECTED_TRACES,
        "flat_trace_comparisons": EXPECTED_TRACES,
        "factorized_trace_comparisons": EXPECTED_TRACES,
        "flat_trace_mismatch_count": 0,
        "factorized_trace_mismatch_count": 0,
        "candidate_comparisons": EXPECTED_CANDIDATES,
        "direct_candidate_mismatch_count": 0,
        "factorized_candidate_mismatch_count": 0,
        "oracle_trace_steps": 3_382_177,
        "flat_trace_steps": 3_382_177,
        "factorized_trace_steps": 3_382_177,
    }
    audit.equal(
        group,
        "frozen full-domain totals and zero mismatches",
        counts,
        expected_counts,
    )
    audit.equal(
        group,
        "no comparison witness exists",
        comparison.get("first_witness"),
        None,
    )

    record_sum_fields = tuple(
        key
        for key in expected_counts
        if key not in {"requested_kernels", "processed_kernels"}
        and not key.endswith("_valid_kernels")
    )
    recomputed_counts = {
        "requested_kernels": EXPECTED_PRIMARY_KERNELS,
        "processed_kernels": 0,
        "oracle_valid_kernels": 0,
        "sut_valid_kernels": 0,
        "factorized_valid_kernels": 0,
        **{key: 0 for key in record_sum_fields},
    }
    valid_hashers = {
        "oracle_valid_kernel_sha256": hashlib.sha256(),
        "sut_valid_kernel_sha256": hashlib.sha256(),
        "factorized_valid_kernel_sha256": hashlib.sha256(),
    }
    aggregate_hashers = {
        "oracle_trace_sha256": hashlib.sha256(),
        "flat_trace_sha256": hashlib.sha256(),
        "factorized_trace_sha256": hashlib.sha256(),
        "oracle_candidate_sha256": hashlib.sha256(),
        "sut_candidate_sha256": hashlib.sha256(),
        "factorized_candidate_sha256": hashlib.sha256(),
    }
    row_errors: list[str] = []
    seen_kernels: set[str] = set()
    with (directory / "kernel_checks.jsonl").open(
        "r", encoding="utf-8"
    ) as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                row_errors.append(f"blank line at {line_number}")
                continue
            record = json.loads(line)
            kernel = str(record.get("kernel"))
            expected_kernel = f"brake_{line_number - 1:04d}"
            if kernel != expected_kernel:
                row_errors.append(
                    f"line {line_number}: {kernel!r} != {expected_kernel!r}"
                )
            if kernel in seen_kernels:
                row_errors.append(f"duplicate kernel {kernel}")
            seen_kernels.add(kernel)
            recomputed_counts["processed_kernels"] += 1
            for key in record_sum_fields:
                recomputed_counts[key] += int(record[key])

            validity_items = (
                (
                    "oracle_valid",
                    "oracle_valid_kernels",
                    "oracle_valid_kernel_sha256",
                ),
                ("sut_valid", "sut_valid_kernels", "sut_valid_kernel_sha256"),
                (
                    "factorized_valid",
                    "factorized_valid_kernels",
                    "factorized_valid_kernel_sha256",
                ),
            )
            for flag, count_name, digest_name in validity_items:
                if record[flag]:
                    recomputed_counts[count_name] += 1
                    valid_hashers[digest_name].update(
                        (kernel + "\n").encode("utf-8")
                    )

            digest_items = (
                ("oracle_trace_sha256", "oracle_trace_sha256"),
                ("flat_trace_sha256", "flat_trace_sha256"),
                ("factorized_trace_sha256", "factorized_trace_sha256"),
                ("oracle_candidate_sha256", "oracle_candidate_sha256"),
                ("sut_candidate_sha256", "sut_candidate_sha256"),
                (
                    "factorized_candidate_sha256",
                    "factorized_candidate_sha256",
                ),
            )
            for record_key, aggregate_key in digest_items:
                item_digest = record.get(record_key)
                if not is_sha256(item_digest):
                    row_errors.append(
                        f"{kernel}: malformed digest {record_key}"
                    )
                    continue
                update_labeled_hash(
                    aggregate_hashers[aggregate_key],
                    kernel,
                    item_digest,
                )

            if not (
                record["oracle_valid"]
                == record["sut_valid"]
                == record["factorized_valid"]
            ):
                row_errors.append(f"{kernel}: validity disagreement")
            if any(
                int(record[key])
                for key in (
                    "validity_mismatch_count",
                    "factorized_validity_mismatch_count",
                    "flat_trace_mismatch_count",
                    "factorized_trace_mismatch_count",
                    "direct_candidate_mismatch_count",
                    "factorized_candidate_mismatch_count",
                )
            ):
                row_errors.append(f"{kernel}: non-zero mismatch")
            if record.get("first_witness") is not None:
                row_errors.append(f"{kernel}: unexpected witness")
            if not (
                record["oracle_trace_sha256"]
                == record["flat_trace_sha256"]
                == record["factorized_trace_sha256"]
            ):
                row_errors.append(f"{kernel}: trace digest disagreement")
            if not (
                record["oracle_candidate_sha256"]
                == record["sut_candidate_sha256"]
                == record["factorized_candidate_sha256"]
            ):
                row_errors.append(f"{kernel}: candidate digest disagreement")

    recomputed_hashes = {
        key: digest.hexdigest() for key, digest in valid_hashers.items()
    }
    recomputed_hashes.update(
        {key: digest.hexdigest() for key, digest in aggregate_hashers.items()}
    )
    audit.equal(
        group,
        "JSONL rows independently reproduce summary counts",
        recomputed_counts,
        counts,
    )
    audit.add(
        group,
        "every JSONL row has exact agreement",
        not row_errors,
        observed={
            "rows": recomputed_counts["processed_kernels"],
            "unique_kernels": len(seen_kernels),
            "error_count": len(row_errors),
            "examples": row_errors[:10],
        },
        expected={
            "rows": EXPECTED_PRIMARY_KERNELS,
            "unique_kernels": EXPECTED_PRIMARY_KERNELS,
            "error_count": 0,
        },
    )
    audit.equal(
        group,
        "JSONL rows independently reproduce aggregate hashes",
        recomputed_hashes,
        hashes,
    )
    audit.add(
        group,
        "all aggregate digests are SHA-256",
        set(hashes) == set(recomputed_hashes)
        and all(is_sha256(value) for value in hashes.values()),
        observed=hashes,
    )

    audit.equal(
        group,
        "mutant artifact matches embedded summary",
        mutants,
        summary.get("mutant_smoke"),
    )
    mutant_rows = mutants.get("mutants", [])
    mutant_names = tuple(row.get("mutant") for row in mutant_rows)
    mutant_witness_errors = [
        row.get("mutant")
        for row in mutant_rows
        if not row.get("detected")
        or not isinstance(row.get("first_witness"), dict)
        or not is_sha256(
            (row.get("first_witness") or {}).get("oracle_trace_sha256")
        )
        or not is_sha256(
            (row.get("first_witness") or {}).get("observed_trace_sha256")
        )
    ]
    audit.add(
        group,
        "all seven frozen semantic mutants are detected with witnesses",
        mutants.get("mutants_total") == 7
        and mutants.get("mutants_detected") == 7
        and mutants.get("all_detected") is True
        and mutants.get("undetected_mutants") == []
        and mutant_names == EXPECTED_MUTANTS
        and not mutant_witness_errors,
        observed={
            "mutants_total": mutants.get("mutants_total"),
            "mutants_detected": mutants.get("mutants_detected"),
            "names": mutant_names,
            "bad_witnesses": mutant_witness_errors,
        },
        expected={
            "mutants_total": 7,
            "mutants_detected": 7,
            "names": EXPECTED_MUTANTS,
            "bad_witnesses": [],
        },
    )

    isolation_pass, isolation_detail = inspect_oracle_isolation(script_path)
    audit.add(
        group,
        "oracle static isolation boundary",
        isolation_pass,
        observed=isolation_detail,
        expected=(
            "oracle_* functions import only layerprobe.model data types and "
            "do not call SUT adapters"
        ),
    )
    expected_sut_hashes = {
        "model": sha256_file(SOURCE_ROOT / "layerprobe" / "model.py"),
        "mechanics": sha256_file(
            SOURCE_ROOT / "layerprobe" / "mechanics.py"
        ),
        "evaluator": sha256_file(
            SOURCE_ROOT / "layerprobe" / "evaluator.py"
        ),
    }
    recorded_file_hashes = metadata.get("hashes", {})
    actual_file_hashes = {
        "script_sha256": sha256_file(script_path),
        "config_sha256": sha256_file(config_path),
        "system_under_test": expected_sut_hashes,
    }
    audit.equal(
        group,
        "oracle code/config/SUT hashes still match recorded run",
        recorded_file_hashes,
        actual_file_hashes,
    )
    separation = metadata.get("separation_contract", {})
    audit.add(
        group,
        "recorded separation contract has no false fields",
        bool(separation) and all(value is True for value in separation.values()),
        observed=separation,
    )
    audit.equal(
        group,
        "metadata cardinalities match frozen comparison",
        {
            "status": metadata.get("status"),
            "mode": metadata.get("mode"),
            "kernel_count": metadata.get("kernel_count"),
            "presentation_count": metadata.get("presentation_count"),
            "agent_count": metadata.get("agent_count"),
            "trace_comparisons_planned_upper_bound": metadata.get(
                "trace_comparisons_planned_upper_bound"
            ),
        },
        {
            "status": "PASS_independent_trace_oracle_full_domain",
            "mode": "full_domain",
            "kernel_count": EXPECTED_PRIMARY_KERNELS,
            "presentation_count": EXPECTED_PRESENTATIONS,
            "agent_count": len(EXPECTED_AGENTS),
            "trace_comparisons_planned_upper_bound": (
                EXPECTED_PRIMARY_KERNELS
                * EXPECTED_PRESENTATIONS
                * len(EXPECTED_AGENTS)
            ),
        },
    )
    claim_boundary = str(summary.get("claim_boundary", "")).lower()
    audit.add(
        group,
        "oracle claim boundary is explicit",
        "finite domain" in claim_boundary
        and "not a formal proof" in claim_boundary
        and "not evidence about human" in claim_boundary,
        observed=summary.get("claim_boundary"),
    )


def choose_earlier_witness(
    incumbent: dict[str, Any] | None,
    candidate: dict[str, Any],
) -> dict[str, Any]:
    if incumbent is None:
        return candidate
    if tuple(candidate["rank"]) < tuple(incumbent["rank"]):
        return candidate
    return incumbent


def merge_witness_map(
    target: dict[str, Any],
    source: Mapping[str, Any],
    *,
    nested: bool,
) -> None:
    for variant, value in source.items():
        if not nested:
            target[variant] = choose_earlier_witness(
                target.get(variant), value
            )
            continue
        target_orders = target.setdefault(variant, {})
        for order, witness in value.items():
            target_orders[order] = choose_earlier_witness(
                target_orders.get(order), witness
            )


def empty_collision_totals() -> dict[str, int]:
    return {
        "semantic_contexts": 0,
        "unique_projected_keys": 0,
        "unsafe_key_classes": 0,
        "contexts_in_unsafe_classes": 0,
        "distinct_output_variants_in_unsafe_classes": 0,
        "conflicting_context_pairs": 0,
        "affected_kernel_agent_scopes": 0,
        "affected_valid_kernels": 0,
    }


def empty_replay_totals() -> dict[str, Any]:
    return {
        "traces": 0,
        "trace_mismatches": 0,
        "candidates": 0,
        "candidate_signature_mismatches": 0,
        "signature_bit_flips": 0,
        "cache_hits": 0,
        "cache_misses": 0,
        "unsafe_cache_hits": 0,
        "nontermination_guards": 0,
    }


def cache_expected_summary_rows(
    summary: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for variant in EXPECTED_CACHE_VARIANTS:
        census = summary["collision_census"][variant]
        for order in EXPECTED_CACHE_ORDERS:
            replay = summary["fault_replay"][variant][order]
            rows.append(
                {
                    "variant": variant,
                    "order": order,
                    "unsafe_key_classes": census["unsafe_key_classes"],
                    "contexts_in_unsafe_classes": census[
                        "contexts_in_unsafe_classes"
                    ],
                    "affected_kernel_agent_scopes": census[
                        "affected_kernel_agent_scopes"
                    ],
                    "affected_valid_kernels": census[
                        "affected_valid_kernels"
                    ],
                    "traces": replay["traces"],
                    "trace_mismatches": replay["trace_mismatches"],
                    "candidates": replay["candidates"],
                    "candidate_signature_mismatches": replay[
                        "candidate_signature_mismatches"
                    ],
                    "signature_bit_flips": replay["signature_bit_flips"],
                    "unsafe_cache_hits": replay["unsafe_cache_hits"],
                    "nontermination_guards": replay[
                        "nontermination_guards"
                    ],
                }
            )
    return rows


def coerce_csv_value(text: str, exemplar: Any) -> Any:
    if isinstance(exemplar, bool):
        if text not in {"True", "False", "true", "false"}:
            raise ValueError(f"not a boolean: {text!r}")
        return text.lower() == "true"
    if isinstance(exemplar, int):
        return int(text)
    if isinstance(exemplar, float):
        return float(text)
    if exemplar is None:
        return None if text == "" else text
    return text


def read_and_compare_csv(
    path: Path,
    expected_rows: Sequence[Mapping[str, Any]],
) -> tuple[bool, dict[str, Any]]:
    if not expected_rows:
        return False, {"error": "expected rows are empty"}
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    expected_fields = list(expected_rows[0])
    errors: list[str] = []
    if len(rows) != len(expected_rows):
        errors.append(f"row count {len(rows)} != {len(expected_rows)}")
    for index, (row, expected) in enumerate(
        zip(rows, expected_rows), start=2
    ):
        if list(row) != expected_fields:
            errors.append(f"line {index}: header/field order mismatch")
            continue
        for key, target in expected.items():
            try:
                observed = coerce_csv_value(row[key], target)
            except (TypeError, ValueError) as exc:
                errors.append(f"line {index}/{key}: {exc}")
                continue
            if not values_match(observed, target):
                errors.append(
                    f"line {index}/{key}: {observed!r} != {target!r}"
                )
    return not errors, {
        "rows": len(rows),
        "expected_rows": len(expected_rows),
        "error_count": len(errors),
        "examples": errors[:10],
    }


def verify_cache(
    audit: Audit,
    directory: Path,
    script_path: Path,
    config_path: Path,
) -> None:
    group = "cache_key_ablation"
    required = (
        "plan.json",
        "progress.json",
        "run_manifest.json",
        "summary.json",
        "counterexamples.json",
        "ablation_summary.csv",
        "SUMMARY.md",
    )
    if not require_files(audit, group, directory, required):
        return
    chunks_dir = directory / "chunks"
    audit.add(
        group,
        "chunk directory exists",
        chunks_dir.is_dir(),
        observed=str(chunks_dir),
    )
    if not chunks_dir.is_dir():
        return

    plan = load_json(directory / "plan.json")
    progress = load_json(directory / "progress.json")
    manifest = load_json(directory / "run_manifest.json")
    summary = load_json(directory / "summary.json")
    counterexamples = load_json(directory / "counterexamples.json")

    plan_projection = {key: manifest.get(key) for key in plan}
    audit.equal(
        group,
        "manifest embeds the exact frozen plan",
        plan_projection,
        plan,
    )
    expected_plan_core = {
        "schema_version": 1,
        "mode": "paper",
        "paper_evidence_eligible": True,
        "selection": "complete_grid",
        "frozen_grid_size": EXPECTED_PRIMARY_KERNELS,
        "selected_kernel_count": EXPECTED_PRIMARY_KERNELS,
        "variants": list(EXPECTED_CACHE_VARIANTS),
        "orders": list(EXPECTED_CACHE_ORDERS),
    }
    observed_plan_core = {key: plan.get(key) for key in expected_plan_core}
    audit.equal(
        group,
        "cache plan covers the complete frozen domain",
        observed_plan_core,
        expected_plan_core,
    )
    expected_indices_hash = stable_digest(list(range(EXPECTED_PRIMARY_KERNELS)))
    audit.equal(
        group,
        "complete-grid selection hash",
        plan.get("selected_indices_sha256"),
        expected_indices_hash,
    )

    actual_fingerprints = {
        "script_sha256": sha256_file(script_path),
        "config_sha256": sha256_file(config_path),
        "core_source_sha256": sha256_source_tree(SOURCE_ROOT),
    }
    actual_fingerprints["experiment_fingerprint"] = stable_digest(
        actual_fingerprints
    )
    actual_fingerprints["run_fingerprint"] = stable_digest(
        {
            "experiment_fingerprint": actual_fingerprints[
                "experiment_fingerprint"
            ],
            "mode": plan["mode"],
            "workers": plan["workers"],
            "chunk_size": plan["chunk_size"],
            "selected_indices": list(range(EXPECTED_PRIMARY_KERNELS)),
            "variants": plan["variants"],
            "orders": plan["orders"],
            "minimum_step_guard": plan["minimum_step_guard"],
            "horizon_guard_multiplier": plan[
                "horizon_guard_multiplier"
            ],
        }
    )
    audit.equal(
        group,
        "cache code/config/source/run fingerprints",
        plan.get("fingerprints"),
        actual_fingerprints,
    )

    expected_total_chunks = math.ceil(
        EXPECTED_PRIMARY_KERNELS / int(plan["chunk_size"])
    )
    completion_projection = {
        "manifest_status": manifest.get("status"),
        "manifest_completed_chunks": manifest.get("completed_chunks"),
        "manifest_total_chunks": manifest.get("total_chunks"),
        "progress_status": progress.get("status"),
        "progress_completed_chunks": progress.get("completed_chunks"),
        "progress_total_chunks": progress.get("total_chunks"),
        "progress_completed_jobs": progress.get("completed_jobs"),
        "progress_planned_jobs": progress.get("planned_jobs"),
    }
    expected_completion = {
        "manifest_status": "complete",
        "manifest_completed_chunks": expected_total_chunks,
        "manifest_total_chunks": expected_total_chunks,
        "progress_status": "completed",
        "progress_completed_chunks": expected_total_chunks,
        "progress_total_chunks": expected_total_chunks,
        "progress_completed_jobs": EXPECTED_PRIMARY_KERNELS,
        "progress_planned_jobs": EXPECTED_PRIMARY_KERNELS,
    }
    audit.equal(
        group,
        "cache run is fully complete",
        completion_projection,
        expected_completion,
    )
    run_fingerprint = actual_fingerprints["run_fingerprint"]
    audit.add(
        group,
        "run fingerprint is consistent across top-level artifacts",
        plan.get("fingerprints", {}).get("run_fingerprint")
        == progress.get("run_fingerprint")
        == manifest.get("fingerprints", {}).get("run_fingerprint")
        == summary.get("run_fingerprint")
        == counterexamples.get("run_fingerprint")
        == run_fingerprint,
        observed={
            "plan": plan.get("fingerprints", {}).get("run_fingerprint"),
            "progress": progress.get("run_fingerprint"),
            "manifest": manifest.get("fingerprints", {}).get(
                "run_fingerprint"
            ),
            "summary": summary.get("run_fingerprint"),
            "counterexamples": counterexamples.get("run_fingerprint"),
            "recomputed": run_fingerprint,
        },
    )

    collision_totals = {
        variant: empty_collision_totals()
        for variant in EXPECTED_CACHE_VARIANTS
    }
    replay_totals = {
        variant: {
            order: empty_replay_totals()
            for order in EXPECTED_CACHE_ORDERS
        }
        for variant in EXPECTED_CACHE_VARIANTS
    }
    digest_fields = (
        "trace_digest",
        "candidate_signature_digest",
        "oracle_trace_digest",
        "oracle_candidate_signature_digest",
    )
    digest_hashers = {
        (variant, order, digest_field): start_json_array_hash()
        for variant in EXPECTED_CACHE_VARIANTS
        for order in EXPECTED_CACHE_ORDERS
        for digest_field in digest_fields
    }
    digest_item_counts = {key: 0 for key in digest_hashers}
    collision_witnesses: dict[str, Any] = {}
    trace_witnesses: dict[str, Any] = {}
    nontermination_witnesses: dict[str, Any] = {}
    chunk_errors: list[str] = []
    chunk_digest = hashlib.sha256()
    total_jobs = 0
    valid_jobs = 0
    oracle_contexts = 0
    seen_job_ids: set[str] = set()
    seen_positions: set[int] = set()
    chunk_paths = sorted(chunks_dir.glob("chunk_*.json"))

    for chunk_index, path in enumerate(chunk_paths):
        raw = path.read_bytes()
        name_bytes = path.name.encode("utf-8")
        chunk_digest.update(len(name_bytes).to_bytes(4, "big"))
        chunk_digest.update(name_bytes)
        chunk_digest.update(len(raw).to_bytes(8, "big"))
        chunk_digest.update(raw)
        payload = json.loads(raw.decode("utf-8"))
        start = chunk_index * int(plan["chunk_size"])
        end = min(
            start + int(plan["chunk_size"]) - 1,
            EXPECTED_PRIMARY_KERNELS - 1,
        )
        expected_name = f"chunk_{chunk_index:05d}_{start:05d}_{end:05d}.json"
        if path.name != expected_name:
            chunk_errors.append(f"{path.name}: expected {expected_name}")
        if payload.get("chunk_index") != chunk_index:
            chunk_errors.append(f"{path.name}: wrong chunk_index")
        if payload.get("selection_start") != start:
            chunk_errors.append(f"{path.name}: wrong selection_start")
        if payload.get("selection_end") != end:
            chunk_errors.append(f"{path.name}: wrong selection_end")
        if payload.get("run_fingerprint") != run_fingerprint:
            chunk_errors.append(f"{path.name}: wrong run fingerprint")
        jobs = payload.get("jobs", [])
        if len(jobs) != end - start + 1:
            chunk_errors.append(f"{path.name}: wrong job count")
        merge_witness_map(
            collision_witnesses,
            payload.get("first_collision_witnesses", {}),
            nested=False,
        )
        merge_witness_map(
            trace_witnesses,
            payload.get("first_trace_mismatch_witnesses", {}),
            nested=True,
        )
        merge_witness_map(
            nontermination_witnesses,
            payload.get("first_nontermination_witnesses", {}),
            nested=True,
        )

        for offset, job in enumerate(jobs):
            expected_position = start + offset
            expected_job_id = f"kernel-{expected_position:05d}"
            expected_kernel = f"brake_{expected_position:04d}"
            total_jobs += 1
            job_id = str(job.get("job_id"))
            position = int(job.get("selection_position", -1))
            if job_id in seen_job_ids:
                chunk_errors.append(f"duplicate job ID {job_id}")
            seen_job_ids.add(job_id)
            if position in seen_positions:
                chunk_errors.append(f"duplicate selection position {position}")
            seen_positions.add(position)
            if (
                job_id != expected_job_id
                or position != expected_position
                or job.get("kernel_index") != expected_position
                or job.get("kernel_name") != expected_kernel
            ):
                chunk_errors.append(
                    f"{path.name}/{offset}: job identity mismatch"
                )
            if not job.get("valid"):
                continue
            valid_jobs += 1
            oracle_contexts += int(job["oracle_contexts"])
            for variant in EXPECTED_CACHE_VARIANTS:
                collision = job["collision"][variant]
                for key in collision_totals[variant]:
                    if key == "affected_valid_kernels":
                        continue
                    collision_totals[variant][key] += int(collision[key])
                collision_totals[variant][
                    "affected_valid_kernels"
                ] += int(bool(collision["affected_kernel"]))
                for order in EXPECTED_CACHE_ORDERS:
                    replay = job["replay"][variant][order]
                    for key in replay_totals[variant][order]:
                        replay_totals[variant][order][key] += int(replay[key])
                    for digest_field in digest_fields:
                        digest_key = (variant, order, digest_field)
                        item_count = digest_item_counts[digest_key]
                        append_json_array_item(
                            digest_hashers[digest_key],
                            [job_id, replay[digest_field]],
                            item_count,
                        )
                        digest_item_counts[digest_key] = item_count + 1

    for digest_key, digest in digest_hashers.items():
        variant, order, digest_field = digest_key
        replay_totals[variant][order][digest_field] = close_json_array_hash(
            digest, digest_item_counts[digest_key]
        )

    audit.add(
        group,
        "chunk sequence and job identities are complete and unique",
        len(chunk_paths) == expected_total_chunks
        and total_jobs == EXPECTED_PRIMARY_KERNELS
        and len(seen_job_ids) == EXPECTED_PRIMARY_KERNELS
        and seen_positions == set(range(EXPECTED_PRIMARY_KERNELS))
        and not chunk_errors,
        observed={
            "chunks": len(chunk_paths),
            "jobs": total_jobs,
            "unique_job_ids": len(seen_job_ids),
            "unique_positions": len(seen_positions),
            "error_count": len(chunk_errors),
            "examples": chunk_errors[:10],
        },
        expected={
            "chunks": expected_total_chunks,
            "jobs": EXPECTED_PRIMARY_KERNELS,
            "unique_job_ids": EXPECTED_PRIMARY_KERNELS,
            "unique_positions": EXPECTED_PRIMARY_KERNELS,
            "error_count": 0,
        },
    )
    audit.equal(
        group,
        "manifest chunk hash",
        manifest.get("chunks_sha256"),
        chunk_digest.hexdigest(),
    )

    artifact_hash_errors: dict[str, dict[str, Any]] = {}
    expected_artifact_names = {
        "plan.json",
        "progress.json",
        "summary.json",
        "counterexamples.json",
        "ablation_summary.csv",
        "SUMMARY.md",
    }
    recorded_artifacts = manifest.get("artifacts", {})
    for name in sorted(set(recorded_artifacts) | expected_artifact_names):
        path = directory / name
        actual = sha256_file(path) if path.is_file() else None
        expected = recorded_artifacts.get(name)
        if actual != expected:
            artifact_hash_errors[name] = {
                "recorded": expected,
                "actual": actual,
            }
    audit.add(
        group,
        "manifest artifact hashes",
        set(recorded_artifacts) == expected_artifact_names
        and not artifact_hash_errors,
        observed={
            "artifact_names": sorted(recorded_artifacts),
            "hash_errors": artifact_hash_errors,
        },
        expected={
            "artifact_names": sorted(expected_artifact_names),
            "hash_errors": {},
        },
    )

    audit.equal(
        group,
        "chunk-derived full-domain cardinalities",
        {
            "selected_kernel_count": total_jobs,
            "valid_kernel_count": valid_jobs,
            "invalid_kernel_count": total_jobs - valid_jobs,
            "presentations": summary.get("presentations"),
            "agents": summary.get("agents"),
            "oracle_contexts": oracle_contexts,
        },
        {
            "selected_kernel_count": EXPECTED_PRIMARY_KERNELS,
            "valid_kernel_count": EXPECTED_VALID_KERNELS,
            "invalid_kernel_count": (
                EXPECTED_PRIMARY_KERNELS - EXPECTED_VALID_KERNELS
            ),
            "presentations": EXPECTED_PRESENTATIONS,
            "agents": len(EXPECTED_AGENTS),
            "oracle_contexts": summary.get("oracle_contexts"),
        },
    )
    audit.equal(
        group,
        "chunk-derived collision census matches summary",
        collision_totals,
        summary.get("collision_census"),
    )
    audit.equal(
        group,
        "chunk-derived fault replay matches summary",
        replay_totals,
        summary.get("fault_replay"),
    )

    full_census = summary["collision_census"]["full"]
    full_orders = summary["fault_replay"]["full"]
    full_errors: list[str] = []
    for key in (
        "unsafe_key_classes",
        "contexts_in_unsafe_classes",
        "distinct_output_variants_in_unsafe_classes",
        "conflicting_context_pairs",
        "affected_kernel_agent_scopes",
        "affected_valid_kernels",
    ):
        if int(full_census[key]) != 0:
            full_errors.append(f"collision_census.{key}={full_census[key]}")
    for order in EXPECTED_CACHE_ORDERS:
        replay = full_orders[order]
        if replay["traces"] != EXPECTED_TRACES:
            full_errors.append(f"{order}.traces={replay['traces']}")
        if replay["candidates"] != EXPECTED_CANDIDATES:
            full_errors.append(f"{order}.candidates={replay['candidates']}")
        for key in (
            "trace_mismatches",
            "candidate_signature_mismatches",
            "signature_bit_flips",
            "unsafe_cache_hits",
            "nontermination_guards",
        ):
            if int(replay[key]) != 0:
                full_errors.append(f"{order}.{key}={replay[key]}")
        if replay["trace_digest"] != replay["oracle_trace_digest"]:
            full_errors.append(f"{order}.trace_digest")
        if (
            replay["candidate_signature_digest"]
            != replay["oracle_candidate_signature_digest"]
        ):
            full_errors.append(f"{order}.candidate_signature_digest")
    if (
        full_orders["canonical"]["trace_digest"]
        != full_orders["reverse"]["trace_digest"]
    ):
        full_errors.append("canonical/reverse trace digest")
    if (
        full_orders["canonical"]["candidate_signature_digest"]
        != full_orders["reverse"]["candidate_signature_digest"]
    ):
        full_errors.append("canonical/reverse candidate digest")
    audit.add(
        group,
        "complete cache key is an exact zero-error control",
        not full_errors,
        observed={"errors": full_errors},
        expected={"errors": []},
    )

    deletion_errors: list[str] = []
    collision_map = counterexamples.get("collision_witnesses", {})
    trace_map = counterexamples.get("trace_mismatch_witnesses", {})
    for variant in EXPECTED_CACHE_VARIANTS[1:]:
        census = summary["collision_census"][variant]
        if int(census["unsafe_key_classes"]) <= 0:
            deletion_errors.append(f"{variant}: no unsafe key class")
        if int(census["affected_valid_kernels"]) <= 0:
            deletion_errors.append(f"{variant}: no affected kernel")
        collision_witness = collision_map.get(variant)
        if not isinstance(collision_witness, dict):
            deletion_errors.append(f"{variant}: missing collision witness")
        else:
            if collision_witness.get("variant") != variant:
                deletion_errors.append(
                    f"{variant}: collision witness variant mismatch"
                )
            first = collision_witness.get("first_context", {}).get("output")
            second = collision_witness.get("second_context", {}).get("output")
            if first == second:
                deletion_errors.append(
                    f"{variant}: collision witness outputs do not differ"
                )
        for order in EXPECTED_CACHE_ORDERS:
            replay = summary["fault_replay"][variant][order]
            if int(replay["trace_mismatches"]) <= 0:
                deletion_errors.append(
                    f"{variant}/{order}: no trace mismatch"
                )
            if int(replay["unsafe_cache_hits"]) <= 0:
                deletion_errors.append(
                    f"{variant}/{order}: no unsafe cache hit"
                )
            witness = trace_map.get(variant, {}).get(order)
            if not isinstance(witness, dict):
                deletion_errors.append(
                    f"{variant}/{order}: missing trace witness"
                )
            elif (
                witness.get("variant") != variant
                or witness.get("order") != order
            ):
                deletion_errors.append(
                    f"{variant}/{order}: malformed trace witness"
                )
            elif (
                witness.get("oracle_step") == witness.get("weak_step")
                and witness.get("oracle_trace_length")
                == witness.get("weak_trace_length")
            ):
                deletion_errors.append(
                    f"{variant}/{order}: witness shows no difference"
                )
    audit.add(
        group,
        "each deleted component fails in both replay orders with witnesses",
        not deletion_errors,
        observed={"errors": deletion_errors},
        expected={"errors": []},
    )
    audit.equal(
        group,
        "counterexamples are the earliest witnesses stored in chunks",
        {
            "collision_witnesses": collision_witnesses,
            "trace_mismatch_witnesses": trace_witnesses,
            "nontermination_witnesses": nontermination_witnesses,
        },
        {
            "collision_witnesses": counterexamples.get(
                "collision_witnesses"
            ),
            "trace_mismatch_witnesses": counterexamples.get(
                "trace_mismatch_witnesses"
            ),
            "nontermination_witnesses": counterexamples.get(
                "nontermination_witnesses"
            ),
        },
    )
    expected_gates = {
        "full_key_control_pass": True,
        "component_necessity_on_selected_domain": {
            variant: True for variant in EXPECTED_CACHE_VARIANTS[1:]
        },
        "end_to_end_trace_failure_observed": {
            variant: {order: True for order in EXPECTED_CACHE_ORDERS}
            for variant in EXPECTED_CACHE_VARIANTS[1:]
        },
        "all_component_necessity_witnesses_present": True,
    }
    audit.equal(
        group,
        "cache acceptance gates",
        summary.get("gates"),
        expected_gates,
    )
    audit.equal(
        group,
        "manifest repeats cache acceptance gates",
        manifest.get("gates"),
        expected_gates,
    )
    csv_pass, csv_detail = read_and_compare_csv(
        directory / "ablation_summary.csv",
        cache_expected_summary_rows(summary),
    )
    audit.add(
        group,
        "cache CSV exactly matches recomputed summary",
        csv_pass,
        observed=csv_detail,
    )


def minimum_cover(
    signatures: Iterable[tuple[str, int]],
    target_mask: int,
) -> tuple[str, ...] | None:
    representatives: dict[int, str] = {}
    for candidate, raw_mask in sorted(signatures):
        mask = raw_mask & target_mask
        if mask:
            representatives.setdefault(mask, candidate)
    dp: dict[int, tuple[str, ...]] = {0: ()}
    for mask, candidate in sorted(
        representatives.items(), key=lambda item: item[1]
    ):
        for covered, suite in list(dp.items()):
            combined = covered | mask
            proposal = suite + (candidate,)
            incumbent = dp.get(combined)
            if incumbent is None or (len(proposal), proposal) < (
                len(incumbent),
                incumbent,
            ):
                dp[combined] = proposal
    return dp.get(target_mask)


def direction(value: float, tolerance: float = 1e-15) -> str:
    if value < -tolerance:
        return "negative"
    if value > tolerance:
        return "positive"
    return "zero"


def kernel_friction_from_frozen_name(name: str) -> int:
    match = KERNEL_RE.fullmatch(name)
    if match is None:
        raise ValueError(f"invalid frozen kernel name: {name}")
    index = int(match.group(1))
    if not 0 <= index < EXPECTED_PRIMARY_KERNELS:
        raise ValueError(f"kernel index outside frozen domain: {name}")
    return index % 3


def recompute_agent_rows(
    communication_summary: Mapping[str, Any],
    signature_path: Path,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    model_pairs = tuple(
        tuple(pair) for pair in communication_summary["model_pairs"]
    )
    if model_pairs != EXPECTED_PAIRS:
        raise ValueError("communication model-pair order is not frozen")
    presentation_rows = communication_summary["presentation_conditions"]
    presentation_meta = {
        row["presentation"]: {
            "speed_mode": row["speed_mode"],
            "distance_mode": row["distance_mode"],
            "delay": int(row["delay"]),
        }
        for row in presentation_rows
    }
    if len(presentation_meta) != EXPECTED_PRESENTATIONS:
        raise ValueError("presentation family is not exhaustive")

    by_kernel: dict[str, dict[str, int]] = defaultdict(dict)
    rows_seen = 0
    signature_errors: list[str] = []
    with gzip.open(
        signature_path, "rt", newline="", encoding="utf-8"
    ) as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != [
            "kernel",
            "presentation",
            "signature_mask",
            "pairs_separated",
        ]:
            raise ValueError("unexpected candidate-signature CSV schema")
        for line_number, row in enumerate(reader, start=2):
            kernel = row["kernel"]
            presentation = row["presentation"]
            if presentation not in presentation_meta:
                signature_errors.append(
                    f"line {line_number}: unknown presentation"
                )
                continue
            if presentation in by_kernel[kernel]:
                signature_errors.append(
                    f"line {line_number}: duplicate candidate"
                )
                continue
            mask = int(row["signature_mask"])
            if not 0 <= mask < 64:
                signature_errors.append(
                    f"line {line_number}: mask outside six-bit universe"
                )
            if int(row["pairs_separated"]) != mask.bit_count():
                signature_errors.append(
                    f"line {line_number}: bit-count mismatch"
                )
            kernel_friction_from_frozen_name(kernel)
            by_kernel[kernel][presentation] = mask
            rows_seen += 1
    for kernel, signatures in by_kernel.items():
        if set(signatures) != set(presentation_meta):
            signature_errors.append(f"{kernel}: incomplete presentation family")
    if signature_errors:
        raise ValueError("; ".join(signature_errors[:10]))

    cell_pairs: list[tuple[str, str]] = []
    for speed_mode in ("exact", "coarse", "hidden"):
        for distance_mode in ("exact", "coarse", "hidden"):
            immediate = [
                name
                for name, meta in presentation_meta.items()
                if meta
                == {
                    "speed_mode": speed_mode,
                    "distance_mode": distance_mode,
                    "delay": 0,
                }
            ]
            delayed = [
                name
                for name, meta in presentation_meta.items()
                if meta
                == {
                    "speed_mode": speed_mode,
                    "distance_mode": distance_mode,
                    "delay": 1,
                }
            ]
            if len(immediate) != 1 or len(delayed) != 1:
                raise ValueError("delay-paired presentation cells are malformed")
            cell_pairs.append((immediate[0], delayed[0]))

    pair_immediate: dict[int, list[int]] = defaultdict(list)
    pair_delayed: dict[int, list[int]] = defaultdict(list)
    pair_deltas: dict[int, list[int]] = defaultdict(list)
    for signatures in by_kernel.values():
        for immediate, delayed in cell_pairs:
            before_mask = signatures[immediate]
            after_mask = signatures[delayed]
            for bit_index in range(len(model_pairs)):
                before = int(bool(before_mask & (1 << bit_index)))
                after = int(bool(after_mask & (1 << bit_index)))
                pair_immediate[bit_index].append(before)
                pair_delayed[bit_index].append(after)
                pair_deltas[bit_index].append(after - before)

    pair_rows: list[dict[str, Any]] = []
    for bit_index, pair in enumerate(model_pairs):
        bit = 1 << bit_index
        candidate_values = [
            int(bool(mask & bit))
            for signatures in by_kernel.values()
            for mask in signatures.values()
        ]
        robust_values = []
        for signatures in by_kernel.values():
            robust_mask = 63
            for mask in signatures.values():
                robust_mask &= mask
            robust_values.append(int(bool(robust_mask & bit)))
        deltas = pair_deltas[bit_index]
        immediate_values = pair_immediate[bit_index]
        delayed_values = pair_delayed[bit_index]
        mean_delta = statistics.fmean(deltas)
        pair_rows.append(
            {
                "pair_index": bit_index,
                "left_agent": pair[0],
                "right_agent": pair[1],
                "pair": f"{pair[0]} vs {pair[1]}",
                "candidate_count": len(candidate_values),
                "candidate_separated_count": sum(candidate_values),
                "candidate_separation_rate": statistics.fmean(
                    candidate_values
                ),
                "robust_kernel_count": sum(robust_values),
                "robust_kernel_rate": statistics.fmean(robust_values),
                "mechanism_presentation_bases": len(deltas),
                "immediate_rate": statistics.fmean(immediate_values),
                "delayed_rate": statistics.fmean(delayed_values),
                "delayed_minus_immediate_rate": mean_delta,
                "improved_count": sum(delta > 0 for delta in deltas),
                "degraded_count": sum(delta < 0 for delta in deltas),
                "same_count": sum(delta == 0 for delta in deltas),
                "direction": direction(mean_delta),
            }
        )

    construct_pair_index = model_pairs.index(
        ("reference", "friction_blind")
    )
    construct_bit = 1 << construct_pair_index
    construct_rows: list[dict[str, Any]] = []
    for friction in (0, 1, 2):
        kernels = [
            name
            for name in by_kernel
            if kernel_friction_from_frozen_name(name) == friction
        ]
        masks = [
            by_kernel[kernel][presentation]
            for kernel in kernels
            for presentation in presentation_meta
        ]
        separated = sum(bool(mask & construct_bit) for mask in masks)
        construct_rows.append(
            {
                "friction": friction,
                "valid_kernels": len(kernels),
                "candidate_count": len(masks),
                "separated_candidates": separated,
                "separation_rate": separated / len(masks),
                "expected_equivalent": friction == 0,
                "negative_control_pass": friction != 0 or separated == 0,
            }
        )

    all_presentations = tuple(sorted(presentation_meta))
    leave_rows: list[dict[str, Any]] = []
    for omitted_agent in EXPECTED_AGENTS:
        retained_indices = tuple(
            index
            for index, pair in enumerate(model_pairs)
            if omitted_agent not in pair
        )
        target_mask = sum(1 << index for index in retained_indices)
        robust_signatures: dict[str, int] = {}
        union_signatures: dict[str, int] = {}
        ordinary_signatures: list[tuple[str, int]] = []
        delay_deltas: list[int] = []
        for kernel, signatures in by_kernel.items():
            robust_mask = target_mask
            union_mask = 0
            for presentation in all_presentations:
                subset_mask = signatures[presentation] & target_mask
                robust_mask &= subset_mask
                union_mask |= subset_mask
                ordinary_signatures.append(
                    (f"{kernel}::{presentation}", subset_mask)
                )
            robust_signatures[kernel] = robust_mask
            union_signatures[kernel] = union_mask
            for immediate, delayed in cell_pairs:
                before = signatures[immediate] & target_mask
                after = signatures[delayed] & target_mask
                delay_deltas.append(after.bit_count() - before.bit_count())
        robust_suite = minimum_cover(
            robust_signatures.items(), target_mask
        )
        union_suite = minimum_cover(union_signatures.items(), target_mask)
        ordinary_suite = minimum_cover(ordinary_signatures, target_mask)
        mean_delta = statistics.fmean(delay_deltas)
        leave_rows.append(
            {
                "omitted_agent": omitted_agent,
                "retained_pair_count": len(retained_indices),
                "retained_pairs": " | ".join(
                    f"{model_pairs[index][0]}__{model_pairs[index][1]}"
                    for index in retained_indices
                ),
                "target_mask": target_mask,
                "mean_pair_delta_delayed_minus_immediate": mean_delta,
                "delay_direction": direction(mean_delta),
                "robust_nonzero_kernels": sum(
                    mask != 0 for mask in robust_signatures.values()
                ),
                "robust_full_kernels": sum(
                    mask == target_mask
                    for mask in robust_signatures.values()
                ),
                "robust_mean_pairs": statistics.fmean(
                    mask.bit_count()
                    for mask in robust_signatures.values()
                ),
                "robust_minimum_suite_size": (
                    None if robust_suite is None else len(robust_suite)
                ),
                "robust_minimum_suite": " | ".join(robust_suite or ()),
                "union_minimum_suite_size": (
                    None if union_suite is None else len(union_suite)
                ),
                "union_minimum_suite": " | ".join(union_suite or ()),
                "ordinary_minimum_suite_size": (
                    None if ordinary_suite is None else len(ordinary_suite)
                ),
                "ordinary_minimum_suite": " | ".join(
                    ordinary_suite or ()
                ),
            }
        )

    diagnostics = {
        "rows": rows_seen,
        "kernels": len(by_kernel),
        "presentations": len(presentation_meta),
        "model_pairs": model_pairs,
    }
    return pair_rows, leave_rows, construct_rows, diagnostics


def verify_agent(
    audit: Audit,
    directory: Path,
    communication_dir: Path,
    script_path: Path,
) -> None:
    group = "agent_sensitivity"
    required = (
        "summary.json",
        "pair_delay_sensitivity.csv",
        "leave_one_agent_out.csv",
        "construct_negative_control.csv",
        "AGENT_SENSITIVITY.md",
    )
    if not require_files(audit, group, directory, required):
        return
    if not require_files(
        audit,
        group,
        communication_dir,
        ("summary.json", "candidate_signatures.csv.gz"),
    ):
        return
    summary = load_json(directory / "summary.json")
    communication_summary_path = communication_dir / "summary.json"
    signature_path = communication_dir / "candidate_signatures.csv.gz"
    communication_summary = load_json(communication_summary_path)
    audit.equal(
        group,
        "communication producer provenance",
        {
            "script_sha256": communication_summary.get("script_sha256"),
            "core_source_sha256": communication_summary.get(
                "core_source_sha256"
            ),
        },
        {
            "script_sha256": sha256_file(COMMUNICATION_SCRIPT),
            "core_source_sha256": sha256_source_tree(SOURCE_ROOT),
        },
    )

    audit.equal(
        group,
        "saved-output full-domain cardinalities",
        {
            "valid_kernels": summary.get("valid_kernels"),
            "presentations": summary.get("presentations"),
            "candidates": summary.get("candidates"),
            "agents": tuple(summary.get("agents", [])),
            "model_pairs": tuple(
                tuple(pair) for pair in summary.get("model_pairs", [])
            ),
        },
        {
            "valid_kernels": EXPECTED_VALID_KERNELS,
            "presentations": EXPECTED_PRESENTATIONS,
            "candidates": EXPECTED_CANDIDATES,
            "agents": EXPECTED_AGENTS,
            "model_pairs": EXPECTED_PAIRS,
        },
    )
    expected_source_hashes = {
        "summary.json": sha256_file(communication_summary_path),
        "candidate_signatures.csv.gz": sha256_file(signature_path),
        "script": sha256_file(script_path),
    }
    audit.equal(
        group,
        "agent-analysis source hashes",
        summary.get("source_sha256"),
        expected_source_hashes,
    )

    pair_rows, leave_rows, construct_rows, diagnostics = (
        recompute_agent_rows(communication_summary, signature_path)
    )
    audit.equal(
        group,
        "compressed signature table shape",
        diagnostics,
        {
            "rows": EXPECTED_CANDIDATES,
            "kernels": EXPECTED_VALID_KERNELS,
            "presentations": EXPECTED_PRESENTATIONS,
            "model_pairs": EXPECTED_PAIRS,
        },
    )
    audit.equal(
        group,
        "pair-level delay and robustness statistics recomputed",
        summary.get("pair_delay_sensitivity"),
        pair_rows,
    )
    audit.equal(
        group,
        "leave-one-agent-out statistics and suites recomputed",
        summary.get("leave_one_agent_out"),
        leave_rows,
    )
    audit.equal(
        group,
        "construct negative-control statistics recomputed",
        summary.get("construct_negative_control"),
        construct_rows,
    )

    friction_zero = next(
        row for row in construct_rows if row["friction"] == 0
    )
    audit.equal(
        group,
        "friction-zero construct negative control",
        friction_zero,
        {
            "friction": 0,
            "valid_kernels": 5_094,
            "candidate_count": 91_692,
            "separated_candidates": 0,
            "separation_rate": 0.0,
            "expected_equivalent": True,
            "negative_control_pass": True,
        },
    )
    audit.equal(
        group,
        "non-zero-friction construct separation counts",
        [
            (
                row["friction"],
                row["valid_kernels"],
                row["candidate_count"],
                row["separated_candidates"],
            )
            for row in construct_rows
            if row["friction"] != 0
        ],
        [
            (1, 3_521, 63_378, 41_611),
            (2, 1_929, 34_722, 27_036),
        ],
    )

    expected_gates = {
        "all_leave_one_out_delay_directions_negative": all(
            row["delay_direction"] == "negative" for row in leave_rows
        ),
        "pair_direction_counts": {
            label: sum(row["direction"] == label for row in pair_rows)
            for label in ("negative", "zero", "positive")
        },
        "all_leave_one_out_robust_suites_exist": all(
            row["robust_minimum_suite_size"] is not None
            for row in leave_rows
        ),
        "friction_zero_negative_control_pass": all(
            row["negative_control_pass"] for row in construct_rows
        ),
    }
    audit.equal(
        group,
        "agent sensitivity gates recomputed",
        summary.get("gates"),
        expected_gates,
    )
    for filename, expected_rows in (
        ("pair_delay_sensitivity.csv", pair_rows),
        ("leave_one_agent_out.csv", leave_rows),
        ("construct_negative_control.csv", construct_rows),
    ):
        passed, detail = read_and_compare_csv(
            directory / filename, expected_rows
        )
        audit.add(
            group,
            f"{filename} exactly matches recomputation",
            passed,
            observed=detail,
        )
    status_and_boundary = (
        str(summary.get("status", "")).lower()
        + " "
        + str(summary.get("interpretation", "")).lower()
    )
    audit.add(
        group,
        "agent-analysis interpretation boundary is explicit",
        "saved_output" in status_and_boundary
        and "finite-domain" in status_and_boundary
        and "not evidence about human" in status_and_boundary,
        observed={
            "status": summary.get("status"),
            "interpretation": summary.get("interpretation"),
        },
    )


def nested_keys(value: Any, prefix: str = "") -> Iterator[str]:
    if isinstance(value, Mapping):
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            yield path
            yield from nested_keys(item, path)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, item in enumerate(value):
            yield from nested_keys(item, f"{prefix}[{index}]")


def verify_stress(
    audit: Audit,
    directory: Path,
    script_path: Path,
    config_path: Path,
) -> None:
    group = "range_extension_stress"
    required = (
        "summary.json",
        "frozen_config.json",
        "valid_kernel_names.txt",
        "RANGE_EXTENSION_STRESS.md",
    )
    if not require_files(audit, group, directory, required):
        return
    summary = load_json(directory / "summary.json")
    frozen_config_path = directory / "frozen_config.json"
    config = load_json(frozen_config_path)
    markdown = (directory / "RANGE_EXTENSION_STRESS.md").read_text(
        encoding="utf-8"
    )
    current_config_hash = sha256_file(config_path)
    frozen_config_hash = sha256_file(frozen_config_path)
    audit.add(
        group,
        "pre-frozen config bytes and hashes",
        config_path.read_bytes() == frozen_config_path.read_bytes()
        and summary.get("config_sha256")
        == current_config_hash
        == frozen_config_hash,
        observed={
            "summary": summary.get("config_sha256"),
            "current": current_config_hash,
            "frozen": frozen_config_hash,
            "byte_identical": (
                config_path.read_bytes() == frozen_config_path.read_bytes()
            ),
        },
    )
    audit.equal(
        group,
        "stress script hash",
        summary.get("script_sha256"),
        sha256_file(script_path),
    )
    audit.equal(
        group,
        "stress core-source provenance",
        summary.get("core_source_sha256"),
        sha256_source_tree(SOURCE_ROOT),
    )

    expected_grid = {
        "goal_start": [44, 64, 84],
        "start_speed": [15, 18, 21],
        "horizon": [16, 20],
        "goal_width": [1, 5],
        "brake_force": [1, 2, 3, 4],
        "friction": [0, 1, 2],
    }
    grid_size = math.prod(
        len(values) for values in config["stress_grid"].values()
    )
    audit.equal(
        group,
        "predeclared range-extension grid",
        {
            "frozen_before_execution": config.get(
                "frozen_before_execution"
            ),
            "stress_grid": config.get("stress_grid"),
            "grid_size": grid_size,
            "presentations": config.get("presentations"),
            "factorized_workers": config.get("factorized_workers"),
        },
        {
            "frozen_before_execution": True,
            "stress_grid": expected_grid,
            "grid_size": 432,
            "presentations": EXPECTED_PRESENTATIONS,
            "factorized_workers": 8,
        },
    )
    flat = summary["results"]["Flat"]
    factorized = summary["results"]["LayerProbe-P8"]
    expected_gates = {
        "identical_valid_kernel_names": True,
        "identical_candidate_signatures": True,
        "identical_minimum_suite": True,
        "flat_graph_builds_equal_requested_kernels_times_presentations": True,
        "factorized_graph_builds_equal_requested_kernels": True,
    }
    audit.equal(
        group,
        "range-extension status, exactness gates, and no mismatch",
        {
            "status": summary.get("status"),
            "requested_kernels": summary.get("requested_kernels"),
            "presentations": summary.get("presentations"),
            "gates": summary.get("gates"),
            "first_mismatch": summary.get("first_mismatch"),
        },
        {
            "status": "PASS",
            "requested_kernels": 432,
            "presentations": EXPECTED_PRESENTATIONS,
            "gates": expected_gates,
            "first_mismatch": None,
        },
    )
    audit.equal(
        group,
        "stress candidate exactness and expected cardinality",
        {
            "flat_valid": flat.get("valid_kernels"),
            "factorized_valid": factorized.get("valid_kernels"),
            "flat_candidates": flat.get("candidates"),
            "factorized_candidates": factorized.get("candidates"),
            "flat_metric_candidates": flat.get("metrics", {}).get(
                "candidates"
            ),
            "factorized_metric_candidates": factorized.get(
                "metrics", {}
            ).get("candidates"),
            "signature_hash_equal": (
                flat.get("candidate_signature_sha256")
                == factorized.get("candidate_signature_sha256")
            ),
            "minimum_suite_equal": (
                flat.get("minimum_suite") == factorized.get("minimum_suite")
            ),
        },
        {
            "flat_valid": 272,
            "factorized_valid": 272,
            "flat_candidates": 4_896,
            "factorized_candidates": 4_896,
            "flat_metric_candidates": 4_896,
            "factorized_metric_candidates": 4_896,
            "signature_hash_equal": True,
            "minimum_suite_equal": True,
        },
    )
    flat_metrics = flat["metrics"]
    factorized_metrics = factorized["metrics"]
    work_errors: list[str] = []
    if flat_metrics["graph_builds"] != 432 * EXPECTED_PRESENTATIONS:
        work_errors.append("flat graph builds")
    if factorized_metrics["graph_builds"] != 432:
        work_errors.append("factorized graph builds")
    for label, metrics in (
        ("flat", flat_metrics),
        ("factorized", factorized_metrics),
    ):
        if metrics["policy_calls"] != metrics["transition_calls"]:
            work_errors.append(f"{label} policy/transition equality")
    expected_work_deltas = {
        "graph_build_reduction": (
            1
            - factorized_metrics["graph_builds"]
            / flat_metrics["graph_builds"]
        ),
        "graph_build_ratio_flat_over_factorized": (
            flat_metrics["graph_builds"]
            / factorized_metrics["graph_builds"]
        ),
        "policy_transition_call_reduction": (
            1
            - factorized_metrics["policy_calls"]
            / flat_metrics["policy_calls"]
        ),
    }
    if not values_match(summary["work_deltas"], expected_work_deltas):
        work_errors.append("work delta arithmetic")
    audit.add(
        group,
        "stress work-accounting invariants",
        not work_errors,
        observed={
            "errors": work_errors,
            "work_deltas": summary.get("work_deltas"),
        },
        expected={"errors": [], "work_deltas": expected_work_deltas},
    )

    names = [
        line.strip()
        for line in (directory / "valid_kernel_names.txt").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    name_errors: list[str] = []
    for name in names:
        match = STRESS_KERNEL_RE.fullmatch(name)
        if match is None or not 0 <= int(match.group(1)) < 432:
            name_errors.append(name)
    audit.add(
        group,
        "stress valid-kernel name list",
        len(names) == 272
        and len(set(names)) == 272
        and names == sorted(names)
        and not name_errors,
        observed={
            "rows": len(names),
            "unique": len(set(names)),
            "sorted": names == sorted(names),
            "bad_names": name_errors[:10],
        },
        expected={
            "rows": 272,
            "unique": 272,
            "sorted": True,
            "bad_names": [],
        },
    )

    boundary = (
        str(summary.get("reporting_boundary", "")).lower()
        + " "
        + str(config.get("reporting_boundary", "")).lower()
    )
    role = (
        str(summary.get("scientific_role", "")).lower()
        + " "
        + str(config.get("scientific_role", "")).lower()
    )
    markdown_lower = markdown.lower()
    elapsed_keys = [
        key
        for key in nested_keys(summary)
        if key.rsplit(".", 1)[-1].startswith("elapsed")
    ]
    forbidden_performance_keys = [
        key
        for key in nested_keys(summary)
        if any(
            token in key.rsplit(".", 1)[-1].lower()
            for token in ("speedup", "throughput", "efficiency", "p_value")
        )
    ]
    audit.add(
        group,
        "diagnostic timing is explicitly excluded from performance evidence",
        "diagnostic" in boundary
        and "not be used as performance evidence" in boundary
        and "not a timing benchmark" in role
        and "matched timing benchmark" in markdown_lower
        and (
            "not a matched timing benchmark" in markdown_lower
            or "nor a matched timing benchmark" in markdown_lower
        )
        and "diagnostic only" in markdown_lower
        and set(elapsed_keys)
        == {
            "results.Flat.elapsed_diagnostic_seconds",
            "results.LayerProbe-P8.elapsed_diagnostic_seconds",
        }
        and not forbidden_performance_keys,
        observed={
            "reporting_boundary": summary.get("reporting_boundary"),
            "scientific_role": summary.get("scientific_role"),
            "elapsed_keys": elapsed_keys,
            "forbidden_performance_keys": forbidden_performance_keys,
        },
    )


def run_group(
    audit: Audit,
    group: str,
    callable_obj: Any,
    *args: Any,
) -> None:
    try:
        callable_obj(audit, *args)
    except BaseException as exc:
        audit.add(
            group,
            "verifier completed without exception",
            False,
            observed=f"{type(exc).__name__}: {exc}",
            expected="no exception",
        )


def write_markdown(report: Mapping[str, Any], path: Path) -> None:
    lines = [
        "# Enhanced evidence read-only audit",
        "",
        f"Overall: **{report['overall']}**",
        "",
        "This verifier did not import or call the simulator. It recomputed the "
        "saved structural, arithmetic, hash, and claim-boundary checks.",
        "",
        "| Evidence bundle | Check | Status |",
        "|---|---|---:|",
    ]
    for check in report["checks"]:
        lines.append(
            f"| {check['group']} | {check['check']} | "
            f"**{check['status']}** |"
        )
    failed = [
        check for check in report["checks"] if check["status"] != "PASS"
    ]
    if failed:
        lines.extend(["", "## Failures", ""])
        for check in failed:
            detail = check.get("detail", check.get("observed", ""))
            lines.append(
                f"- `{check['group']}/{check['check']}`: {detail}"
            )
    lines.extend(
        [
            "",
            "Scope boundaries: the oracle is independent differential evidence on "
            "a finite domain, cache-key necessity is component-wise on that domain, "
            "agent sensitivity is not human-effect evidence, and stress-run elapsed "
            "times are diagnostic only.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-dir", type=Path, default=DEFAULT_ORACLE_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--agent-dir", type=Path, default=DEFAULT_AGENT_DIR)
    parser.add_argument("--stress-dir", type=Path, default=DEFAULT_STRESS_DIR)
    parser.add_argument(
        "--communication-dir",
        type=Path,
        default=DEFAULT_COMMUNICATION_DIR,
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=SOURCE_ROOT,
        help=(
            "source tree paired with the saved evidence; defaults to the "
            "current repository src directory"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    global SOURCE_ROOT
    args = parse_args()
    SOURCE_ROOT = args.source_root.resolve()
    if not (SOURCE_ROOT / "layerprobe").is_dir():
        raise FileNotFoundError(
            f"source root must contain a layerprobe package: {SOURCE_ROOT}"
        )
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(
            f"refusing to overwrite verifier output directory: {output}"
        )
    output.mkdir(parents=True)

    audit = Audit()
    run_group(
        audit,
        "independent_trace_oracle",
        verify_oracle,
        args.oracle_dir.resolve(),
        ORACLE_SCRIPT.resolve(),
        ORACLE_CONFIG.resolve(),
    )
    run_group(
        audit,
        "cache_key_ablation",
        verify_cache,
        args.cache_dir.resolve(),
        CACHE_SCRIPT.resolve(),
        CACHE_CONFIG.resolve(),
    )
    run_group(
        audit,
        "agent_sensitivity",
        verify_agent,
        args.agent_dir.resolve(),
        args.communication_dir.resolve(),
        AGENT_SCRIPT.resolve(),
    )
    run_group(
        audit,
        "range_extension_stress",
        verify_stress,
        args.stress_dir.resolve(),
        STRESS_SCRIPT.resolve(),
        STRESS_CONFIG.resolve(),
    )
    overall = "PASS" if audit.passed else "FAIL"
    report = {
        "schema_version": 1,
        "overall": overall,
        "verified_at": datetime.now().astimezone().isoformat(),
        "verifier_mode": (
            "read_only_inputs_no_simulator_import_or_execution"
        ),
        "inputs": {
            "oracle_dir": str(args.oracle_dir.resolve()),
            "cache_dir": str(args.cache_dir.resolve()),
            "agent_dir": str(args.agent_dir.resolve()),
            "stress_dir": str(args.stress_dir.resolve()),
            "communication_dir": str(args.communication_dir.resolve()),
            "source_root": str(SOURCE_ROOT),
        },
        "checks": audit.checks,
        "summary": {
            "passed": sum(
                check["status"] == "PASS" for check in audit.checks
            ),
            "failed": sum(
                check["status"] == "FAIL" for check in audit.checks
            ),
            "total": len(audit.checks),
        },
    }
    (output / "verification_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_markdown(report, output / "VERIFICATION_REPORT.md")
    print(
        json.dumps(
            {
                "overall": overall,
                **report["summary"],
                "output": str(output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if audit.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
