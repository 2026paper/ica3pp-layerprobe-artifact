"""Reproducible single-process cProfile decomposition for LayerProbe.

This diagnostic profiles the production ``run_factorized(..., workers=1)``
path.  Its purpose is coarse attribution, not a speedup claim: deterministic
function instrumentation changes absolute timings, and cumulative function
times overlap.  The profile therefore reports additive *self* time separately
from non-additive cumulative time.

The named project functions are assigned to five interpretable categories.
Everything else -- including dictionary operations, generated dataclass
hash/equality methods, interpreter/runtime work, and the evaluator's cache
loop -- is assigned to ``evaluator/cache-loop residual``.  That residual must
not be described as an exact lookup, key-serialization, or hashing breakdown.

Default workload
----------------
The command below profiles the complete deterministic braking family of
24,624 mechanisms and all 18 presentation variants:

    python experiments/cost_profile.py --output results/cost_profile_full

``--smoke`` selects a deterministic midpoint sample while exercising all 18
presentations.  Workload construction and artifact serialization are outside
the profiled region.
"""

from __future__ import annotations

import argparse
import cProfile
import csv
import hashlib
import io
import json
import os
import platform
import pstats
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from layerprobe import evaluator as evaluator_module  # noqa: E402
from layerprobe import mechanics as mechanics_module  # noqa: E402
from layerprobe.evaluator import RunResult, run_factorized  # noqa: E402
from layerprobe.model import (  # noqa: E402
    KernelSpec,
    PresentationSpec,
    WorkMetrics,
)
from layerprobe.workloads import make_kernels, make_presentations  # noqa: E402


MAX_KERNELS = 24_624
MAX_PRESENTATIONS = 18
SMOKE_KERNELS = 120

RAW_PROFILE_NAME = "layerprobe_profile.prof"
FUNCTION_CSV_NAME = "functions.csv"
SUMMARY_NAME = "summary.json"
METADATA_NAME = "metadata.json"

OBSERVATION = "observation"
POLICY_MEMORY = "policy/memory"
TRANSITION_TERMINAL = "transition/terminal"
MECHANISM_VALIDATION = "mechanism validation"
AGGREGATION_DIGEST = "aggregation/digest"
EVALUATOR_RESIDUAL = "evaluator/cache-loop residual"

CATEGORY_ORDER = (
    OBSERVATION,
    POLICY_MEMORY,
    TRANSITION_TERMINAL,
    MECHANISM_VALIDATION,
    AGGREGATION_DIGEST,
    EVALUATOR_RESIDUAL,
)

CATEGORY_NOTES = {
    OBSERVATION: (
        "Presentation validation, observation encoding, and read-only display "
        "memory updates."
    ),
    POLICY_MEMORY: (
        "Observation ingestion, policy choice, stopping-distance calculation, "
        "and agent-memory initialization/update."
    ),
    TRANSITION_TERMINAL: (
        "World-state initialization, state transition, and terminal checks."
    ),
    MECHANISM_VALIDATION: (
        "Kernel specification validation and exhaustive bounded graph "
        "verification."
    ),
    AGGREGATION_DIGEST: (
        "Candidate-signature construction, frontier reduction, exact cover, "
        "and result/metric aggregation. The semantic SHA-256 is computed "
        "after profiling and is therefore not charged to this category."
    ),
    EVALUATOR_RESIDUAL: (
        "Remainder after the named project functions above. It includes the "
        "evaluator/cache loop, Python container operations, generated "
        "dataclass hash/equality methods, interpreter/runtime work, and "
        "profiler overhead. It is not an exact measurement of cache lookup, "
        "key serialization, hashing, or allocation."
    ),
}


@dataclass(frozen=True, slots=True)
class ProfilePlan:
    """Frozen inputs for one non-parallel diagnostic profile."""

    output: Path
    kernel_count: int
    presentation_count: int
    smoke: bool = False


@dataclass(frozen=True, slots=True)
class FunctionRow:
    """One row extracted from the binary cProfile artifact."""

    rank_by_self_time: int
    category: str
    category_basis: str
    filename: str
    first_line: int
    function: str
    primitive_calls: int
    total_calls: int
    self_seconds: float
    cumulative_seconds: float


def now_local() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def stable_json(value: object, *, indent: int | None = None) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":") if indent is None else None,
        indent=indent,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_source_tree(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py"), key=lambda item: item.as_posix()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def digest_value(value: object) -> str:
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: object) -> None:
    atomic_write_text(path, stable_json(value, indent=2) + "\n")


def semantic_digest(result: RunResult) -> str:
    """Hash the same semantic payload used by the deadline experiment runner."""

    payload = {
        "candidate_signatures": sorted(result.candidate_signatures.items()),
        "minimum_suite": result.minimum_suite,
        "valid_kernels": result.valid_kernels,
    }
    encoded = json.dumps(
        payload,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _code_key(function: Callable[..., object]) -> tuple[str, int, str]:
    code = function.__code__
    return (
        os.path.normcase(os.path.abspath(code.co_filename)),
        int(code.co_firstlineno),
        code.co_name,
    )


def _register(
    mapping: dict[tuple[str, int, str], str],
    category: str,
    functions: Iterable[Callable[..., object]],
) -> None:
    for function in functions:
        mapping[_code_key(function)] = category


def _named_function_categories() -> dict[tuple[str, int, str], str]:
    """Build source-location rules without hard-coding line numbers."""

    mapping: dict[tuple[str, int, str], str] = {}
    _register(
        mapping,
        OBSERVATION,
        (
            mechanics_module._encode,
            mechanics_module._raw_observation,
            mechanics_module.observe,
            PresentationSpec.validate,
        ),
    )
    _register(
        mapping,
        POLICY_MEMORY,
        (
            mechanics_module.initial_agent_memory,
            mechanics_module.ingest,
            mechanics_module.stopping_distance,
            mechanics_module.choose_action,
            mechanics_module.advance_belief,
        ),
    )
    _register(
        mapping,
        TRANSITION_TERMINAL,
        (
            mechanics_module.initial_state,
            mechanics_module.terminal_status,
            mechanics_module.transition,
        ),
    )
    _register(
        mapping,
        MECHANISM_VALIDATION,
        (
            KernelSpec.validate,
            mechanics_module.verify_kernel,
        ),
    )
    _register(
        mapping,
        AGGREGATION_DIGEST,
        (
            evaluator_module.signature_for,
            evaluator_module.reduce_signature_frontier,
            evaluator_module.minimum_cover,
            evaluator_module._finish,
            WorkMetrics.add,
            WorkMetrics.as_dict,
        ),
    )
    return mapping


NAMED_FUNCTION_CATEGORIES = _named_function_categories()


def classify_function(
    function_key: tuple[str, int, str],
) -> tuple[str, str]:
    """Return the auditable coarse category and the basis for attribution."""

    filename, first_line, function = function_key
    normalized = (os.path.normcase(os.path.abspath(filename)), first_line, function)
    category = NAMED_FUNCTION_CATEGORIES.get(normalized)
    if category is not None:
        return category, "named project function"
    return EVALUATOR_RESIDUAL, "residual by subtraction"


def _display_filename(filename: str) -> str:
    if filename.startswith("<") and filename.endswith(">"):
        return filename
    path = Path(filename)
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except (OSError, ValueError):
        return f"<external>/{path.name}"


def function_rows(stats: pstats.Stats) -> list[FunctionRow]:
    """Extract stable per-function rows from a loaded cProfile artifact."""

    provisional: list[
        tuple[
            str,
            str,
            str,
            int,
            str,
            int,
            int,
            float,
            float,
        ]
    ] = []
    for (filename, first_line, function), values in stats.stats.items():
        primitive_calls, total_calls, self_seconds, cumulative_seconds, _ = values
        category, basis = classify_function(
            (filename, int(first_line), function)
        )
        provisional.append(
            (
                category,
                basis,
                _display_filename(filename),
                int(first_line),
                function,
                int(primitive_calls),
                int(total_calls),
                float(self_seconds),
                float(cumulative_seconds),
            )
        )
    provisional.sort(
        key=lambda row: (
            -row[7],
            -row[8],
            row[2],
            row[3],
            row[4],
        )
    )
    return [
        FunctionRow(index, *row)
        for index, row in enumerate(provisional, start=1)
    ]


def aggregate_categories(
    rows: Iterable[FunctionRow],
) -> dict[str, dict[str, int | float | str]]:
    """Aggregate additive self time; label cumulative sums as non-additive."""

    row_list = list(rows)
    total_self = sum(row.self_seconds for row in row_list)
    result: dict[str, dict[str, int | float | str]] = {}
    for category in CATEGORY_ORDER:
        selected = [row for row in row_list if row.category == category]
        self_seconds = sum(row.self_seconds for row in selected)
        result[category] = {
            "function_rows": len(selected),
            "primitive_calls_sum": sum(row.primitive_calls for row in selected),
            "total_calls_sum_nonexclusive": sum(
                row.total_calls for row in selected
            ),
            "self_seconds": self_seconds,
            "self_time_share": (
                self_seconds / total_self if total_self > 0 else 0.0
            ),
            "cumulative_seconds_sum_nonadditive": sum(
                row.cumulative_seconds for row in selected
            ),
            "interpretation": CATEGORY_NOTES[category],
        }
    return result


def cache_counters(metrics: Mapping[str, int]) -> dict[str, Any]:
    """Derive only counters justified by the production evaluator API."""

    semantic_requests = int(metrics["observation_calls"])
    computed_steps = int(metrics["policy_calls"])
    transition_calls = int(metrics["transition_calls"])
    reported_prefix_groups = int(metrics["prefix_groups"])
    cache_hits = semantic_requests - computed_steps
    if cache_hits < 0:
        raise ValueError("computed steps exceed semantic requests")
    if transition_calls != computed_steps:
        raise ValueError("policy and transition counters disagree")

    # In the current production factorized evaluator, prefix_groups increments
    # once exactly when one group-local cache entry is inserted. Guard the
    # derivation so a future semantic change is not silently misreported.
    total_entries: int | None
    if reported_prefix_groups == computed_steps:
        total_entries = reported_prefix_groups
    else:
        total_entries = None

    return {
        "semantic_requests": semantic_requests,
        "computed_steps": computed_steps,
        "cache_hits": cache_hits,
        "hit_rate": (
            cache_hits / semantic_requests if semantic_requests else 0.0
        ),
        "reported_prefix_groups": reported_prefix_groups,
        "total_cache_entries_across_ephemeral_caches": total_entries,
        "total_cache_entries_basis": (
            "prefix_groups equals computed_steps and is incremented once per "
            "cache insertion in _memoized_agent_traces"
            if total_entries is not None
            else "unavailable: production counters do not establish equality"
        ),
        "peak_cache_entries": None,
        "peak_cache_entries_basis": (
            "unavailable from the production RunResult/WorkMetrics API; the "
            "profiler does not alter evaluator semantics to instrument it"
        ),
    }


def _midpoint_indices(population: int, count: int) -> tuple[int, ...]:
    if not 1 <= count <= population:
        raise ValueError(f"count must be between 1 and {population}")
    if count == population:
        return tuple(range(population))
    return tuple(
        (position * population + population // 2) // count
        for position in range(count)
    )


def select_workload(
    kernel_count: int,
    presentation_count: int,
) -> tuple[
    tuple[KernelSpec, ...],
    tuple[PresentationSpec, ...],
    dict[str, object],
]:
    """Select a deterministic complete workload or midpoint subset."""

    kernel_indices = _midpoint_indices(MAX_KERNELS, kernel_count)
    presentation_indices = _midpoint_indices(
        MAX_PRESENTATIONS,
        presentation_count,
    )
    all_kernels = make_kernels(MAX_KERNELS)
    all_presentations = make_presentations(MAX_PRESENTATIONS)
    kernels = tuple(all_kernels[index] for index in kernel_indices)
    presentations = tuple(
        all_presentations[index] for index in presentation_indices
    )
    selection = {
        "kernel_population": MAX_KERNELS,
        "kernel_count": kernel_count,
        "kernel_selection": (
            "complete_grid"
            if kernel_count == MAX_KERNELS
            else "deterministic_midpoint"
        ),
        "kernel_indices_sha256": digest_value(kernel_indices),
        "presentation_population": MAX_PRESENTATIONS,
        "presentation_count": presentation_count,
        "presentation_selection": (
            "complete_grid"
            if presentation_count == MAX_PRESENTATIONS
            else "deterministic_midpoint"
        ),
        "presentation_indices": list(presentation_indices),
        "presentation_names": [item.name for item in presentations],
    }
    return kernels, presentations, selection


def code_fingerprints() -> dict[str, str]:
    script = Path(__file__).resolve()
    pyproject = ROOT / "pyproject.toml"
    components = {
        "script_sha256": sha256_file(script),
        "core_source_sha256": sha256_source_tree(SOURCE_ROOT),
        "pyproject_sha256": sha256_file(pyproject),
    }
    return {
        **components,
        "code_fingerprint_sha256": digest_value(components),
    }


def _write_function_csv(path: Path, rows: Iterable[FunctionRow]) -> None:
    buffer = io.StringIO(newline="")
    fieldnames = tuple(FunctionRow.__dataclass_fields__)
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(asdict(row))
    atomic_write_text(path, buffer.getvalue())


def _dump_profile_atomic(profiler: cProfile.Profile, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    profiler.dump_stats(str(temporary))
    os.replace(temporary, path)


def _top_functions(
    rows: Iterable[FunctionRow],
    attribute: str,
    count: int = 15,
) -> list[dict[str, object]]:
    selected = sorted(
        rows,
        key=lambda row: (
            -float(getattr(row, attribute)),
            row.filename,
            row.first_line,
            row.function,
        ),
    )[:count]
    return [
        {
            "category": row.category,
            "function": (
                f"{row.filename}:{row.first_line}({row.function})"
            ),
            "primitive_calls": row.primitive_calls,
            "total_calls": row.total_calls,
            "self_seconds": row.self_seconds,
            "cumulative_seconds": row.cumulative_seconds,
        }
        for row in selected
    ]


def profile_artifacts(
    plan: ProfilePlan,
    kernels: tuple[KernelSpec, ...],
    presentations: tuple[PresentationSpec, ...],
    selection: Mapping[str, object],
) -> dict[str, object]:
    """Run one profile and atomically write its auditable artifacts."""

    if not kernels or not presentations:
        raise ValueError("profile workload must not be empty")
    if plan.kernel_count != len(kernels):
        raise ValueError("plan kernel_count does not match workload")
    if plan.presentation_count != len(presentations):
        raise ValueError("plan presentation_count does not match workload")

    output = plan.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    artifact_names = (
        RAW_PROFILE_NAME,
        FUNCTION_CSV_NAME,
        SUMMARY_NAME,
        METADATA_NAME,
    )
    existing = [name for name in artifact_names if (output / name).exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite existing profile artifacts: "
            + ", ".join(existing)
        )

    fingerprints = code_fingerprints()
    metadata: dict[str, object] = {
        "schema_version": 1,
        "status": "running",
        "started_at": now_local(),
        "profile_plan": {
            "kernel_count": plan.kernel_count,
            "presentation_count": plan.presentation_count,
            "workers": 1,
            "smoke": plan.smoke,
        },
        "selection": dict(selection),
        "profile_scope": (
            "production run_factorized(kernels, presentations, workers=1); "
            "workload construction, semantic digest, and artifact writes are "
            "outside the profiled region"
        ),
        "environment": {
            "python_executable": Path(sys.executable).name,
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
        "fingerprints": fingerprints,
        "interpretation_limits": [
            (
                "cProfile instrumentation changes absolute timings; these "
                "seconds are not used for a wall-clock speedup claim"
            ),
            (
                "self time is additive, whereas cumulative function time "
                "overlaps across callers and callees"
            ),
            CATEGORY_NOTES[EVALUATOR_RESIDUAL],
        ],
    }
    metadata_path = output / METADATA_NAME
    atomic_write_json(metadata_path, metadata)

    raw_path = output / RAW_PROFILE_NAME
    function_path = output / FUNCTION_CSV_NAME
    summary_path = output / SUMMARY_NAME
    profiler = cProfile.Profile()
    try:
        profiler.enable()
        result = run_factorized(kernels, presentations, workers=1)
        profiler.disable()

        _dump_profile_atomic(profiler, raw_path)
        stats = pstats.Stats(str(raw_path))
        rows = function_rows(stats)
        _write_function_csv(function_path, rows)

        counters = cache_counters(result.metrics)
        category_summary = aggregate_categories(rows)
        summary: dict[str, object] = {
            "schema_version": 1,
            "profile_status": "complete",
            "profile_scope": metadata["profile_scope"],
            "profile_total_self_seconds": float(stats.total_tt),
            "profile_total_primitive_calls": int(stats.prim_calls),
            "profile_total_calls": int(stats.total_calls),
            "semantic_digest_sha256": semantic_digest(result),
            "result_counts": {
                "valid_kernels": len(result.valid_kernels),
                "candidates": len(result.candidate_signatures),
                "frontier": len(result.frontier),
                "minimum_suite_size": (
                    None
                    if result.minimum_suite is None
                    else len(result.minimum_suite)
                ),
            },
            "production_metrics": dict(result.metrics),
            "cache_counters": counters,
            "category_breakdown": category_summary,
            "top_functions_by_self_time": _top_functions(
                rows,
                "self_seconds",
            ),
            "top_functions_by_cumulative_time": _top_functions(
                rows,
                "cumulative_seconds",
            ),
            "interpretation_limits": metadata["interpretation_limits"],
            "artifact_sha256": {
                RAW_PROFILE_NAME: sha256_file(raw_path),
                FUNCTION_CSV_NAME: sha256_file(function_path),
            },
        }
        atomic_write_json(summary_path, summary)

        metadata["status"] = "complete"
        metadata["completed_at"] = now_local()
        metadata["artifacts"] = {
            RAW_PROFILE_NAME: sha256_file(raw_path),
            FUNCTION_CSV_NAME: sha256_file(function_path),
            SUMMARY_NAME: sha256_file(summary_path),
        }
        atomic_write_json(metadata_path, metadata)
        return summary
    except BaseException as error:
        profiler.disable()
        metadata["status"] = "failed"
        metadata["failed_at"] = now_local()
        metadata["failure"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
        atomic_write_json(metadata_path, metadata)
        raise


def run_plan(plan: ProfilePlan) -> dict[str, object]:
    kernels, presentations, selection = select_workload(
        plan.kernel_count,
        plan.presentation_count,
    )
    return profile_artifacts(
        plan,
        kernels,
        presentations,
        selection,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Coarsely attribute production LayerProbe single-worker work with "
            "cProfile. Absolute profile time is not a speedup measurement."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="new or artifact-empty output directory",
    )
    parser.add_argument(
        "--kernels",
        type=int,
        default=None,
        help=f"mechanisms to profile (default: {MAX_KERNELS})",
    )
    parser.add_argument(
        "--presentations",
        type=int,
        default=None,
        help=f"presentation variants to profile (default: {MAX_PRESENTATIONS})",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            f"default to a deterministic {SMOKE_KERNELS}-mechanism sample "
            "while retaining all 18 presentations"
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    kernel_count = (
        args.kernels
        if args.kernels is not None
        else (SMOKE_KERNELS if args.smoke else MAX_KERNELS)
    )
    presentation_count = (
        args.presentations
        if args.presentations is not None
        else MAX_PRESENTATIONS
    )
    if not 1 <= kernel_count <= MAX_KERNELS:
        raise ValueError(
            f"--kernels must be between 1 and {MAX_KERNELS}"
        )
    if not 1 <= presentation_count <= MAX_PRESENTATIONS:
        raise ValueError(
            f"--presentations must be between 1 and {MAX_PRESENTATIONS}"
        )
    plan = ProfilePlan(
        output=args.output,
        kernel_count=kernel_count,
        presentation_count=presentation_count,
        smoke=bool(args.smoke),
    )
    summary = run_plan(plan)
    counters = summary["cache_counters"]
    assert isinstance(counters, dict)
    print(
        "Profile complete: "
        f"{summary['profile_total_calls']} calls, "
        f"{counters['cache_hits']} cache hits, "
        f"artifacts={plan.output.resolve()}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
