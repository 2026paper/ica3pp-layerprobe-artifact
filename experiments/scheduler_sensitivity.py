"""Single-host scheduling sensitivity study for kernel-group evaluation.

The benchmark holds the semantic workload, worker count, cache scope, and
deterministic sorted reduction fixed.  It changes only how kernel-group tasks
are assigned to a local ``ProcessPoolExecutor``:

``current_chunksize``
    The production rule ``max(1, kernels // (workers * 4))``.
``fine_chunksize_1``
    A fine-grained dynamic queue with one kernel group per submitted chunk.
``static_contiguous``
    Exactly one precomputed contiguous kernel batch per independently spawned
    OS process.  This provides one fixed process for each logical slot.

Every repeat checks the digest of every candidate signature across all three
schedules.  A mismatching timing is rejected before it is appended.  Worker
    load is the sum of measured production kernel-group durations assigned to
    an observed process (dynamic schedules) or to a fixed spawned process
    (static schedule).
Consequently, load imbalance is interpretable without conflating it with pool
startup, IPC, or deterministic reduction overhead.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import multiprocessing
import os
import platform
import random
import statistics
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

# Direct script execution does not inherit pytest's ``pythonpath = ["src"]``.
# Add the repository source tree before importing the project package; spawned
# Windows workers inherit the same importable path.
ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from layerprobe.evaluator import _factorized_kernel_group
from layerprobe.model import KernelSpec, PresentationSpec
from layerprobe.workloads import make_kernels, make_presentations

SCHEDULES = (
    "current_chunksize",
    "fine_chunksize_1",
    "static_contiguous",
)
GroupTask = tuple[KernelSpec, tuple[PresentationSpec, ...]]


@dataclass(frozen=True, slots=True)
class GroupResult:
    """Timed wrapper around one production evaluator mechanism group."""

    kernel_name: str
    valid: bool
    candidate_signatures: dict[str, int]
    semantic_requests: int
    computed_steps: int
    cache_hits: int
    elapsed_s: float


@dataclass(frozen=True, slots=True)
class AggregateResult:
    """Deterministic reduction of production group results."""

    digest: str
    valid_kernels: int
    candidates: int
    semantic_requests: int
    computed_steps: int
    cache_hits: int
    group_elapsed_p50_ms: float
    group_elapsed_p95_ms: float
    group_elapsed_max_ms: float


@dataclass(frozen=True, slots=True)
class TaggedGroupResult:
    """One group result plus the process that actually executed it."""

    result: GroupResult
    worker_pid: int


@dataclass(frozen=True, slots=True)
class StaticBatchResult:
    """Results for one indivisible, pre-assigned contiguous worker batch."""

    logical_slot: int
    worker_pid: int
    results: tuple[GroupResult, ...]


@dataclass(frozen=True, slots=True)
class ScheduleMeasurement:
    aggregate: AggregateResult
    elapsed_s: float
    observed_workers: int
    worker_loads_s: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class RunRow:
    job_id: str
    repeat: int
    order_index: int
    schedule: str
    workers: int
    kernels: int
    presentations: int
    current_chunksize: int
    elapsed_s: float
    digest: str
    valid_kernels: int
    candidates: int
    semantic_requests: int
    computed_steps: int
    cache_hits: int
    observed_workers: int
    worker_loads_json: str
    group_elapsed_p50_ms: float
    group_elapsed_p95_ms: float
    group_elapsed_max_ms: float
    total_group_work_s: float
    worker_load_p50_s: float
    worker_load_p95_s: float
    worker_load_max_s: float
    worker_load_mean_s: float
    load_imbalance_max_over_mean: float
    critical_path_over_ideal: float
    straggler_excess_over_ideal_s: float
    approximate_unattributed_time_s: float
    completed_at: str


def _production_group(task: GroupTask) -> GroupResult:
    """Call the production semantic evaluator and add only outer group timing."""

    started = time.perf_counter()
    kernel_name, valid, signatures, metrics = _factorized_kernel_group(task)
    elapsed_s = time.perf_counter() - started
    semantic_requests = int(metrics.observation_calls)
    computed_steps = int(metrics.policy_calls)
    if int(metrics.transition_calls) != computed_steps:
        raise AssertionError("production policy/transition counters disagree")
    if computed_steps > semantic_requests:
        raise AssertionError("production computed steps exceed requests")
    return GroupResult(
        kernel_name=kernel_name,
        valid=valid,
        candidate_signatures=signatures,
        semantic_requests=semantic_requests,
        computed_steps=computed_steps,
        cache_hits=semantic_requests - computed_steps,
        elapsed_s=elapsed_s,
    )


def _tagged_factorized_group(task: GroupTask) -> TaggedGroupResult:
    return TaggedGroupResult(
        result=_production_group(task),
        worker_pid=os.getpid(),
    )


def _static_batch_worker(
    payload: tuple[int, tuple[GroupTask, ...]],
) -> StaticBatchResult:
    logical_slot, tasks = payload
    return StaticBatchResult(
        logical_slot=logical_slot,
        worker_pid=os.getpid(),
        results=tuple(_production_group(task) for task in tasks),
    )


def _static_slot_process(
    payload: tuple[int, tuple[GroupTask, ...]],
    sender: object,
) -> None:
    """Run exactly one logical slot in its own spawned OS process."""

    try:
        result: object = ("ok", _static_batch_worker(payload))
    except BaseException as error:  # pragma: no cover - exercised on worker failure
        result = ("error", type(error).__name__, str(error))
    try:
        sender.send(result)  # type: ignore[attr-defined]
    finally:
        sender.close()  # type: ignore[attr-defined]


def _percentile(values: Iterable[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile requires at least one value")
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _aggregate(group_results: Iterable[GroupResult]) -> AggregateResult:
    """Reduce complete production outputs in mechanism-name order."""

    ordered = sorted(group_results, key=lambda item: item.kernel_name)
    if not ordered:
        raise ValueError("at least one group result is required")
    digest = hashlib.sha256()
    valid_kernels = 0
    candidates = 0
    semantic_requests = 0
    computed_steps = 0
    cache_hits = 0
    group_times: list[float] = []
    seen_candidates: set[str] = set()
    for group in ordered:
        digest.update(group.kernel_name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(int(group.valid)).encode("ascii"))
        digest.update(b"\0")
        valid_kernels += int(group.valid)
        semantic_requests += group.semantic_requests
        computed_steps += group.computed_steps
        cache_hits += group.cache_hits
        group_times.append(group.elapsed_s)
        for candidate, mask in sorted(group.candidate_signatures.items()):
            if candidate in seen_candidates:
                raise AssertionError(f"duplicate production candidate: {candidate}")
            seen_candidates.add(candidate)
            digest.update(candidate.encode("utf-8"))
            digest.update(b"\0")
            digest.update(str(mask).encode("ascii"))
            digest.update(b"\n")
            candidates += 1
    return AggregateResult(
        digest=digest.hexdigest(),
        valid_kernels=valid_kernels,
        candidates=candidates,
        semantic_requests=semantic_requests,
        computed_steps=computed_steps,
        cache_hits=cache_hits,
        group_elapsed_p50_ms=1_000.0 * _percentile(group_times, 0.50),
        group_elapsed_p95_ms=1_000.0 * _percentile(group_times, 0.95),
        group_elapsed_max_ms=1_000.0 * max(group_times),
    )


def _contiguous_batches(
    tasks: tuple[GroupTask, ...],
    slots: int,
) -> tuple[tuple[GroupTask, ...], ...]:
    """Partition tasks in order into nearly equal, non-empty batches."""

    if not tasks:
        raise ValueError("at least one task is required")
    if slots < 1:
        raise ValueError("slots must be positive")
    actual_slots = min(slots, len(tasks))
    base, remainder = divmod(len(tasks), actual_slots)
    batches: list[tuple[GroupTask, ...]] = []
    offset = 0
    for slot in range(actual_slots):
        size = base + int(slot < remainder)
        batches.append(tasks[offset : offset + size])
        offset += size
    if offset != len(tasks):
        raise AssertionError("static partition did not cover every task")
    return tuple(batches)


def _pad_worker_loads(
    observed_loads: Iterable[float],
    effective_workers: int,
) -> tuple[float, ...]:
    loads = list(observed_loads)
    if len(loads) > effective_workers:
        raise AssertionError("observed more workers than requested")
    loads.extend([0.0] * (effective_workers - len(loads)))
    return tuple(sorted(loads))


def run_schedule(
    schedule: str,
    kernels: tuple[KernelSpec, ...],
    presentations: tuple[PresentationSpec, ...],
    workers: int,
) -> ScheduleMeasurement:
    """Run one schedule and return semantic and load measurements."""

    if schedule not in SCHEDULES:
        raise ValueError(f"unsupported schedule: {schedule}")
    if workers < 1:
        raise ValueError("workers must be positive")
    tasks: tuple[GroupTask, ...] = tuple(
        (kernel, presentations) for kernel in kernels
    )
    if not tasks:
        raise ValueError("at least one kernel is required")
    effective_workers = min(workers, len(tasks))

    started = time.perf_counter()
    if schedule == "static_contiguous":
        batches = _contiguous_batches(tasks, effective_workers)
        payloads = tuple(enumerate(batches))
        context = multiprocessing.get_context("spawn")
        receivers = []
        processes = []
        for payload in payloads:
            receiver, sender = context.Pipe(duplex=False)
            process = context.Process(
                target=_static_slot_process,
                args=(payload, sender),
            )
            process.start()
            sender.close()
            receivers.append(receiver)
            processes.append(process)
        messages: list[object] = []
        try:
            for receiver in receivers:
                messages.append(receiver.recv())
        finally:
            for receiver in receivers:
                receiver.close()
            for process in processes:
                process.join()
        failures = [
            message
            for message in messages
            if not isinstance(message, tuple)
            or not message
            or message[0] != "ok"
        ]
        bad_exit_codes = [
            process.exitcode for process in processes if process.exitcode != 0
        ]
        if failures or bad_exit_codes:
            raise RuntimeError(
                "static slot worker failed: "
                f"messages={failures!r}, exit_codes={bad_exit_codes!r}"
            )
        batch_results = tuple(
            message[1]  # type: ignore[index]
            for message in messages
        )
        if not all(isinstance(item, StaticBatchResult) for item in batch_results):
            raise RuntimeError("static slot worker returned an invalid payload")
        worker_pids = [batch.worker_pid for batch in batch_results]
        if len(set(worker_pids)) != effective_workers:
            raise AssertionError(
                "static schedule did not use one distinct process per logical slot"
            )
        group_results = tuple(
            group
            for batch in sorted(
                batch_results,
                key=lambda item: item.logical_slot,
            )
            for group in batch.results
        )
        # Each static load belongs to one independently spawned fixed process.
        worker_loads = tuple(
            sum(group.elapsed_s for group in batch.results)
            for batch in sorted(
                batch_results,
                key=lambda item: item.logical_slot,
            )
        )
        observed_workers = len(set(worker_pids))
    else:
        chunksize = (
            1
            if schedule == "fine_chunksize_1"
            else max(1, len(tasks) // (effective_workers * 4))
        )
        with ProcessPoolExecutor(
            max_workers=effective_workers,
            mp_context=multiprocessing.get_context("spawn"),
        ) as executor:
            tagged_results = tuple(
                executor.map(
                    _tagged_factorized_group,
                    tasks,
                    chunksize=chunksize,
                )
            )
        group_results = tuple(item.result for item in tagged_results)
        load_by_pid: dict[int, float] = defaultdict(float)
        for item in tagged_results:
            load_by_pid[item.worker_pid] += item.result.elapsed_s
        observed_workers = len(load_by_pid)
        worker_loads = _pad_worker_loads(
            load_by_pid.values(),
            effective_workers,
        )
    if len(group_results) != len(tasks):
        raise AssertionError("schedule lost or duplicated kernel groups")
    aggregate = _aggregate(group_results)
    elapsed_s = time.perf_counter() - started
    return ScheduleMeasurement(
        aggregate=aggregate,
        elapsed_s=elapsed_s,
        observed_workers=observed_workers,
        worker_loads_s=worker_loads,
    )


def _row_from_measurement(
    *,
    repeat: int,
    order_index: int,
    schedule: str,
    workers: int,
    kernels: int,
    presentations: int,
    measurement: ScheduleMeasurement,
) -> RunRow:
    loads = measurement.worker_loads_s
    total_group_work = sum(loads)
    mean_load = statistics.fmean(loads)
    max_load = max(loads)
    ideal_load = total_group_work / len(loads)
    imbalance = max_load / mean_load if mean_load else 1.0
    critical_path_ratio = max_load / ideal_load if ideal_load else 1.0
    return RunRow(
        job_id=f"scheduler-r{repeat}-{schedule}-w{workers}",
        repeat=repeat,
        order_index=order_index,
        schedule=schedule,
        workers=workers,
        kernels=kernels,
        presentations=presentations,
        current_chunksize=max(
            1,
            kernels // (min(workers, kernels) * 4),
        ),
        elapsed_s=measurement.elapsed_s,
        digest=measurement.aggregate.digest,
        valid_kernels=measurement.aggregate.valid_kernels,
        candidates=measurement.aggregate.candidates,
        semantic_requests=measurement.aggregate.semantic_requests,
        computed_steps=measurement.aggregate.computed_steps,
        cache_hits=measurement.aggregate.cache_hits,
        observed_workers=measurement.observed_workers,
        worker_loads_json=json.dumps(
            list(loads),
            separators=(",", ":"),
        ),
        group_elapsed_p50_ms=measurement.aggregate.group_elapsed_p50_ms,
        group_elapsed_p95_ms=measurement.aggregate.group_elapsed_p95_ms,
        group_elapsed_max_ms=measurement.aggregate.group_elapsed_max_ms,
        total_group_work_s=total_group_work,
        worker_load_p50_s=_percentile(loads, 0.50),
        worker_load_p95_s=_percentile(loads, 0.95),
        worker_load_max_s=max_load,
        worker_load_mean_s=mean_load,
        load_imbalance_max_over_mean=imbalance,
        critical_path_over_ideal=critical_path_ratio,
        straggler_excess_over_ideal_s=max_load - ideal_load,
        approximate_unattributed_time_s=max(
            0.0,
            measurement.elapsed_s - max_load,
        ),
        completed_at=datetime.now().astimezone().isoformat(),
    )


def _schedule_order(repeat: int) -> tuple[str, str, str]:
    """Use all six permutations before repeating the order cycle."""

    orders = (
        (
            "current_chunksize",
            "fine_chunksize_1",
            "static_contiguous",
        ),
        (
            "fine_chunksize_1",
            "static_contiguous",
            "current_chunksize",
        ),
        (
            "static_contiguous",
            "current_chunksize",
            "fine_chunksize_1",
        ),
        (
            "static_contiguous",
            "fine_chunksize_1",
            "current_chunksize",
        ),
        (
            "fine_chunksize_1",
            "current_chunksize",
            "static_contiguous",
        ),
        (
            "current_chunksize",
            "static_contiguous",
            "fine_chunksize_1",
        ),
    )
    return orders[repeat % len(orders)]


def _assert_digest_compatible(
    row: RunRow,
    existing_rows: Iterable[RunRow],
) -> None:
    peers = [
        peer
        for peer in existing_rows
        if peer.repeat == row.repeat and peer.schedule != row.schedule
    ]
    for peer in peers:
        if peer.digest != row.digest:
            raise AssertionError(
                "semantic digest mismatch before timing admission: "
                f"repeat={row.repeat}, {peer.schedule}={peer.digest}, "
                f"{row.schedule}={row.digest}"
            )


def _append_row(path: Path, row: RunRow) -> None:
    exists = path.exists()
    with path.open("a", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(row)))
        if not exists:
            writer.writeheader()
        writer.writerow(asdict(row))
        handle.flush()
        os.fsync(handle.fileno())


def _load_rows(path: Path) -> list[RunRow]:
    if not path.exists():
        return []
    rows: list[RunRow] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for payload in csv.DictReader(handle):
            rows.append(
                RunRow(
                    job_id=payload["job_id"],
                    repeat=int(payload["repeat"]),
                    order_index=int(payload["order_index"]),
                    schedule=payload["schedule"],
                    workers=int(payload["workers"]),
                    kernels=int(payload["kernels"]),
                    presentations=int(payload["presentations"]),
                    current_chunksize=int(payload["current_chunksize"]),
                    elapsed_s=float(payload["elapsed_s"]),
                    digest=payload["digest"],
                    valid_kernels=int(payload["valid_kernels"]),
                    candidates=int(payload["candidates"]),
                    semantic_requests=int(payload["semantic_requests"]),
                    computed_steps=int(payload["computed_steps"]),
                    cache_hits=int(payload["cache_hits"]),
                    observed_workers=int(payload["observed_workers"]),
                    worker_loads_json=payload["worker_loads_json"],
                    group_elapsed_p50_ms=float(
                        payload["group_elapsed_p50_ms"]
                    ),
                    group_elapsed_p95_ms=float(
                        payload["group_elapsed_p95_ms"]
                    ),
                    group_elapsed_max_ms=float(
                        payload["group_elapsed_max_ms"]
                    ),
                    total_group_work_s=float(payload["total_group_work_s"]),
                    worker_load_p50_s=float(payload["worker_load_p50_s"]),
                    worker_load_p95_s=float(payload["worker_load_p95_s"]),
                    worker_load_max_s=float(payload["worker_load_max_s"]),
                    worker_load_mean_s=float(payload["worker_load_mean_s"]),
                    load_imbalance_max_over_mean=float(
                        payload["load_imbalance_max_over_mean"]
                    ),
                    critical_path_over_ideal=float(
                        payload["critical_path_over_ideal"]
                    ),
                    straggler_excess_over_ideal_s=float(
                        payload["straggler_excess_over_ideal_s"]
                    ),
                    approximate_unattributed_time_s=float(
                        payload["approximate_unattributed_time_s"]
                    ),
                    completed_at=payload["completed_at"],
                )
            )
    identifiers = [row.job_id for row in rows]
    if len(identifiers) != len(set(identifiers)):
        raise RuntimeError("runs.csv contains duplicate job_id values")
    return rows


def _validate_loaded_rows(
    rows: Iterable[RunRow],
    *,
    workers: int,
    kernels: int,
    presentations: int,
    repeats: int,
) -> None:
    """Reject stale configurations and any cross-session partial repeat."""

    expected_chunksize = max(
        1,
        kernels // (min(workers, kernels) * 4),
    )
    by_repeat: dict[int, set[str]] = defaultdict(set)
    for row in rows:
        if not 0 <= row.repeat < repeats:
            raise RuntimeError(f"runs.csv repeat is outside configuration: {row.repeat}")
        if (
            row.workers != workers
            or row.kernels != kernels
            or row.presentations != presentations
            or row.current_chunksize != expected_chunksize
        ):
            raise RuntimeError(
                f"runs.csv row configuration mismatch: {row.job_id}"
            )
        if row.schedule not in SCHEDULES:
            raise RuntimeError(f"runs.csv has unknown schedule: {row.schedule}")
        expected_job_id = (
            f"scheduler-r{row.repeat}-{row.schedule}-w{workers}"
        )
        if row.job_id != expected_job_id:
            raise RuntimeError(f"runs.csv job id mismatch: {row.job_id}")
        expected_order = _schedule_order(row.repeat)
        if not 0 <= row.order_index < len(expected_order):
            raise RuntimeError(f"runs.csv order index is invalid: {row.job_id}")
        if expected_order[row.order_index] != row.schedule:
            raise RuntimeError(f"runs.csv schedule order mismatch: {row.job_id}")
        if row.schedule in by_repeat[row.repeat]:
            raise RuntimeError(
                f"runs.csv repeats a schedule in repeat {row.repeat}: "
                f"{row.schedule}"
            )
        by_repeat[row.repeat].add(row.schedule)
    for repeat, schedules in sorted(by_repeat.items()):
        if schedules != set(SCHEDULES):
            raise RuntimeError(
                "resume refused: incomplete three-schedule repeat "
                f"{repeat} has {sorted(schedules)}"
            )


def _bootstrap_median_ci(
    values: list[float],
    *,
    seed: int,
) -> tuple[float, float]:
    if not values:
        raise ValueError("bootstrap requires at least one value")
    if len(values) == 1:
        return values[0], values[0]
    rng = random.Random(seed)
    medians = [
        statistics.median(rng.choices(values, k=len(values)))
        for _ in range(4_000)
    ]
    return _percentile(medians, 0.025), _percentile(medians, 0.975)


def summarize(rows: list[RunRow]) -> dict[str, object]:
    by_repeat: dict[int, dict[str, RunRow]] = defaultdict(dict)
    for row in rows:
        if row.schedule not in SCHEDULES:
            raise ValueError(f"unexpected schedule in runs.csv: {row.schedule}")
        if row.schedule in by_repeat[row.repeat]:
            raise RuntimeError(
                f"duplicate schedule in repeat {row.repeat}: {row.schedule}"
            )
        by_repeat[row.repeat][row.schedule] = row

    repeat_checks: list[dict[str, object]] = []
    complete_repeats: list[int] = []
    for repeat, group in sorted(by_repeat.items()):
        complete = set(group) == set(SCHEDULES)
        digests = {row.digest for row in group.values()}
        digest_equal = len(digests) == 1
        pairwise = {
            f"{left}_vs_{right}": group[left].digest == group[right].digest
            for index, left in enumerate(SCHEDULES)
            for right in SCHEDULES[index + 1 :]
            if left in group and right in group
        }
        repeat_checks.append(
            {
                "repeat": repeat,
                "complete": complete,
                "digest_equal": digest_equal,
                "pairwise_digest_equal": pairwise,
            }
        )
        if not digest_equal:
            raise AssertionError(
                f"semantic digest mismatch in persisted repeat {repeat}"
            )
        if complete:
            complete_repeats.append(repeat)

    schedule_summary: list[dict[str, object]] = []
    for schedule_index, schedule in enumerate(SCHEDULES):
        schedule_rows = [
            by_repeat[repeat][schedule] for repeat in complete_repeats
        ]
        if not schedule_rows:
            continue
        elapsed = [row.elapsed_s for row in schedule_rows]
        low, high = _bootstrap_median_ci(
            elapsed,
            seed=20260724 + schedule_index,
        )
        baseline_ratios = [
            by_repeat[repeat]["current_chunksize"].elapsed_s
            / by_repeat[repeat][schedule].elapsed_s
            for repeat in complete_repeats
        ]
        ratio_low, ratio_high = _bootstrap_median_ci(
            baseline_ratios,
            seed=20260734 + schedule_index,
        )
        schedule_summary.append(
            {
                "schedule": schedule,
                "paired_repeats": len(schedule_rows),
                "elapsed_median_s": statistics.median(elapsed),
                "elapsed_median_ci95_low_s": low,
                "elapsed_median_ci95_high_s": high,
                "current_over_schedule_ratio_median": statistics.median(
                    baseline_ratios
                ),
                "current_over_schedule_ratio_ci95_low": ratio_low,
                "current_over_schedule_ratio_ci95_high": ratio_high,
                "group_elapsed_p50_ms_median": statistics.median(
                    row.group_elapsed_p50_ms for row in schedule_rows
                ),
                "group_elapsed_p95_ms_median": statistics.median(
                    row.group_elapsed_p95_ms for row in schedule_rows
                ),
                "group_elapsed_max_ms_median": statistics.median(
                    row.group_elapsed_max_ms for row in schedule_rows
                ),
                "worker_load_max_s_median": statistics.median(
                    row.worker_load_max_s for row in schedule_rows
                ),
                "load_imbalance_max_over_mean_median": statistics.median(
                    row.load_imbalance_max_over_mean
                    for row in schedule_rows
                ),
                "critical_path_over_ideal_median": statistics.median(
                    row.critical_path_over_ideal for row in schedule_rows
                ),
                "straggler_excess_over_ideal_s_median": statistics.median(
                    row.straggler_excess_over_ideal_s
                    for row in schedule_rows
                ),
                "approximate_unattributed_time_s_median": statistics.median(
                    row.approximate_unattributed_time_s
                    for row in schedule_rows
                ),
            }
        )

    all_complete = bool(repeat_checks) and all(
        item["complete"] and item["digest_equal"] for item in repeat_checks
    )
    return {
        "status": (
            "complete_semantics_checked"
            if all_complete
            else "partial_semantics_checked"
        ),
        "run_count": len(rows),
        "complete_paired_repeats": len(complete_repeats),
        "repeat_checks": repeat_checks,
        "schedules": schedule_summary,
        "metric_notes": {
            "worker_load": (
                "sum of measured kernel-group durations per observed process "
                "(dynamic) or pre-assigned logical slot (static)"
            ),
            "load_imbalance_max_over_mean": (
                "maximum worker load divided by mean worker load; 1 is ideal"
            ),
            "critical_path_over_ideal": (
                "maximum worker load divided by total group work/workers"
            ),
            "approximate_unattributed_time": (
                "nonnegative max(0, end-to-end elapsed minus maximum measured "
                "logical-slot/process load); a coarse residual that may include "
                "startup, IPC, queueing gaps, reduction, clock overlap, and "
                "measurement noise, not an exact scheduler-overhead estimate"
            ),
        },
    }


def _fingerprint() -> str:
    digest = hashlib.sha256()
    paths = list((ROOT / "src").rglob("*.py"))
    paths.append(Path(__file__).resolve())
    for path in sorted(paths):
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _configuration(args: argparse.Namespace) -> dict[str, int | bool]:
    return {
        "workers": args.workers,
        "kernels": args.kernels,
        "presentations": args.presentations,
        "repeats": args.repeats,
        "warmup_kernels": args.warmup_kernels,
        "smoke": args.smoke,
    }


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--kernels", type=int, default=24_624)
    parser.add_argument("--presentations", type=int, default=18)
    parser.add_argument("--warmup-kernels", type=int)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "allow fewer than 10 repeats for a non-reportable plumbing check"
        ),
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.warmup_kernels is None:
        args.warmup_kernels = min(256, args.kernels)
    logical_cpus = os.cpu_count() or 1
    if not 1 <= args.workers <= logical_cpus:
        raise ValueError(
            f"workers must be between 1 and host logical CPU count {logical_cpus}"
        )
    if args.repeats < 1:
        raise ValueError("repeats must be positive")
    if args.repeats < 10 and not args.smoke:
        raise ValueError(
            "formal scheduler sensitivity requires at least 10 repeats; "
            "pass --smoke only for a non-reportable plumbing check"
        )
    if not 1 <= args.kernels <= 24_624:
        raise ValueError("kernels must be between 1 and 24624")
    if not 1 <= args.presentations <= 18:
        raise ValueError("presentations must be between 1 and 18")
    if not 0 <= args.warmup_kernels <= args.kernels:
        raise ValueError("warmup-kernels must be between 0 and kernels")

    output = args.output.resolve()
    if output.exists() and not args.resume:
        raise FileExistsError(
            f"{output} exists; choose a new directory or pass --resume"
        )
    output.mkdir(parents=True, exist_ok=True)
    metadata_path = output / "metadata.json"
    runs_path = output / "runs.csv"
    summary_path = output / "summary.json"
    configuration = _configuration(args)
    fingerprint = _fingerprint()

    if metadata_path.exists():
        if not args.resume:
            raise FileExistsError(
                f"{metadata_path} exists; pass --resume to continue"
            )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("code_fingerprint_sha256") != fingerprint:
            raise RuntimeError("code fingerprint changed; resume refused")
        if metadata.get("configuration") != configuration:
            raise RuntimeError("configuration changed; resume refused")
    else:
        metadata = {
            "started_at": datetime.now().astimezone().isoformat(),
            "platform": platform.platform(),
            "python": sys.version,
            "python_executable": Path(sys.executable).name,
            "configuration": configuration,
            "workload_implementation": (
                "layerprobe.evaluator._factorized_kernel_group from the "
                "production evaluator; the benchmark wrapper adds only "
                "outer group timing and worker PID"
            ),
            "schedules": {
                "current_chunksize": (
                    "ProcessPoolExecutor.map with "
                    "max(1, kernels // (workers * 4))"
                ),
                "fine_chunksize_1": (
                    "ProcessPoolExecutor.map with chunksize=1"
                ),
                "static_contiguous": (
                    "one independently spawned OS process per logical slot; "
                    "each process executes one indivisible contiguous batch"
                ),
            },
            "code_fingerprint_sha256": fingerprint,
            "claim_scope": (
                "single-host scheduling sensitivity with unchanged semantic "
                "work, worker count, cache scope, and deterministic sorted "
                "candidate-signature reduction"
            ),
            "reportable": not args.smoke,
        }
        _write_json(metadata_path, metadata)

    rows = _load_rows(runs_path)
    _validate_loaded_rows(
        rows,
        workers=args.workers,
        kernels=args.kernels,
        presentations=args.presentations,
        repeats=args.repeats,
    )
    completed = {row.job_id for row in rows}
    # Validate any already-persisted cross-schedule pairs before resuming.
    summarize(rows)

    kernels = make_kernels(args.kernels)
    presentations = make_presentations(args.presentations)
    if args.warmup_kernels:
        warmup_kernels = make_kernels(args.warmup_kernels)
        for schedule in SCHEDULES:
            run_schedule(
                schedule,
                warmup_kernels,
                presentations,
                args.workers,
            )
            gc.collect()

    for repeat in range(args.repeats):
        for order_index, schedule in enumerate(_schedule_order(repeat)):
            job_id = f"scheduler-r{repeat}-{schedule}-w{args.workers}"
            if job_id in completed:
                continue
            gc.collect()
            measurement = run_schedule(
                schedule,
                kernels,
                presentations,
                args.workers,
            )
            row = _row_from_measurement(
                repeat=repeat,
                order_index=order_index,
                schedule=schedule,
                workers=args.workers,
                kernels=args.kernels,
                presentations=args.presentations,
                measurement=measurement,
            )
            # The hard gate runs before persistence: once another schedule from
            # this repeat exists, mismatching semantics are never admitted.
            _assert_digest_compatible(row, rows)
            _append_row(runs_path, row)
            rows.append(row)
            completed.add(job_id)
            print(
                f"{job_id}: {row.elapsed_s:.3f}s "
                f"digest={row.digest[:12]} "
                f"load_imbalance={row.load_imbalance_max_over_mean:.3f}",
                flush=True,
            )

    summary = summarize(rows)
    metadata["finished_at"] = datetime.now().astimezone().isoformat()
    metadata["status"] = summary["status"]
    _write_json(metadata_path, metadata)
    _write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
