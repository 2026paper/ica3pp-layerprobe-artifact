"""Deadline-aware experiment runner for the 8-core LayerProbe paper scope.

The runner deliberately evaluates only claims supported by the current
single-domain implementation:

1. semantic equivalence of flat, kernel-memoized, and factorized execution;
2. separate effects of mechanism reuse and semantic-step reuse;
3. deterministic scaling on the available physical cores; and
4. sensitivity to the number of presentation variants.

Every completed job is appended to ``runs.csv`` immediately.  An interrupted
run can be continued with ``--resume`` without overwriting completed rows.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import platform
import random
import statistics
import sys
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable

from layerprobe.evaluator import (
    RunResult,
    run_factorized,
    run_flat,
    run_kernel_memo,
    run_kernel_memo_parallel,
)
from layerprobe.workloads import make_kernels, make_presentations

try:
    import psutil
except ImportError:  # pragma: no cover - the paper environment includes psutil
    psutil = None


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path(__file__).with_name("deadline_profile_8c32g.json")
METHODS: dict[str, Callable[..., RunResult]] = {
    "flat": run_flat,
    "kernel_memo": run_kernel_memo,
    "kernel_memo_parallel": run_kernel_memo_parallel,
    "factorized": run_factorized,
}


@dataclass(frozen=True, slots=True)
class Job:
    job_id: str
    study: str
    case: str
    repeat: int
    order_index: int
    method: str
    workers: int
    kernel_count: int
    presentation_indices: tuple[int, ...]


@dataclass(slots=True)
class RunRow:
    job_id: str
    study: str
    case: str
    repeat: int
    order_index: int
    method: str
    workers: int
    kernel_count: int
    kernel_selection: str
    presentation_count: int
    presentation_set: str
    elapsed_s: float
    worker_slot_s: float
    peak_process_tree_rss_mb: float | None
    digest: str
    valid_kernels: int
    candidates: int
    frontier: int
    suite_size: int | None
    graph_builds: int
    graph_states: int
    graph_transitions: int
    observation_calls: int
    policy_calls: int
    transition_calls: int
    prefix_groups: int
    completed_at: str


class PeakMemorySampler:
    """Sample parent-plus-child RSS while a job is active."""

    def __init__(self, interval_s: float = 0.05) -> None:
        self.interval_s = interval_s
        self.peak_bytes: int | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample_once(self) -> None:
        if psutil is None:
            return
        try:
            root = psutil.Process()
            processes = [root, *root.children(recursive=True)]
            total = 0
            for process in processes:
                try:
                    total += process.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            if self.peak_bytes is None or total > self.peak_bytes:
                self.peak_bytes = total
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            self._sample_once()

    def start(self) -> None:
        if psutil is None:
            return
        self._sample_once()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> float | None:
        if psutil is None:
            return None
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1)
        self._sample_once()
        if self.peak_bytes is None:
            return None
        return self.peak_bytes / (1024 * 1024)


def semantic_digest(result: RunResult) -> str:
    payload = {
        "candidate_signatures": sorted(result.candidate_signatures.items()),
        "minimum_suite": result.minimum_suite,
        "valid_kernels": result.valid_kernels,
    }
    encoded = json.dumps(payload, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def presentation_subset(count: int, replicate: int) -> tuple[int, ...]:
    """Return a deterministic, spread-out subset of the 18 presentations."""

    if not 1 <= count <= 18:
        raise ValueError("presentation count must be between 1 and 18")
    if count == 18:
        return tuple(range(18))
    # Midpoint sampling gives unique, well-spread indices because count <= 18.
    base = [math.floor((position + 0.5) * 18 / count) for position in range(count)]
    offset = (replicate * 5) % 18
    return tuple(sorted((index + offset) % 18 for index in base))


def stratified_kernels(count: int):
    """Use the complete grid or a midpoint-stratified subset of that grid."""

    maximum = 24_624
    if not 1 <= count <= maximum:
        raise ValueError(f"kernel count must be between 1 and {maximum}")
    complete = make_kernels(maximum)
    if count == maximum:
        return complete, "complete_grid"
    indices = tuple(
        math.floor((position + 0.5) * maximum / count) for position in range(count)
    )
    return tuple(complete[index] for index in indices), "stratified_midpoint"


def rotate(items: tuple[object, ...], offset: int) -> tuple[object, ...]:
    offset %= len(items)
    return items[offset:] + items[:offset]


def physical_cpu_count() -> int:
    if psutil is not None:
        detected = psutil.cpu_count(logical=False)
        if detected:
            return int(detected)
    logical = os.cpu_count() or 1
    return max(1, logical // 2)


def available_memory_gib() -> float | None:
    if psutil is None:
        return None
    return psutil.virtual_memory().available / (1024**3)


def load_config(path: Path) -> dict[str, object]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("unsupported experiment profile schema")
    return config


def clamp_workers(values: Iterable[int], maximum: int) -> tuple[int, ...]:
    selected = tuple(sorted({int(value) for value in values if 1 <= int(value) <= maximum}))
    return selected or (1,)


def build_jobs(
    config: dict[str, object],
    mode: str,
    primary_workers: int,
    throughput_workers: int,
    maximum_workers: int,
) -> list[Job]:
    all_presentations = tuple(range(18))
    jobs: list[Job] = []

    if mode == "smoke":
        gate_kernel_counts = (120,)
        gate_presentation_counts = (18,)
        gate_workers = clamp_workers((1, min(2, maximum_workers)), maximum_workers)
        method_counts = (300,)
        sequential_pair_repeats = 1
        flat_pair_repeats = 1
        parallel_pair_repeats = 1
        scaling_kernels = 500
        scaling_workers = gate_workers
        scaling_repeats = 1
        presentation_kernels = 500
        presentation_counts = (2, 18)
        subset_replicates = 1
        presentation_repeats = 1
        capacity_counts = (500,)
        capacity_repeats = 1
    else:
        gate = config["correctness_gate"]
        methods = config["method_ladder"]
        scaling = config["parallel_scaling"]
        presentations = config["presentation_scaling"]
        capacity = config["capacity_scan"]
        gate_kernel_counts = tuple(int(value) for value in gate["kernel_counts"])
        gate_presentation_counts = tuple(
            int(value) for value in gate["presentation_counts"]
        )
        gate_workers = clamp_workers(gate["workers"], maximum_workers)
        method_counts = tuple(int(value) for value in methods["kernel_counts"])
        sequential_pair_repeats = int(methods["sequential_pair_repeats"])
        flat_pair_repeats = int(methods["flat_pair_repeats"])
        parallel_pair_repeats = int(methods["parallel_pair_repeats"])
        scaling_kernels = int(scaling["kernels"])
        scaling_workers = clamp_workers(scaling["workers"], maximum_workers)
        scaling_repeats = int(scaling["repeats"])
        presentation_kernels = int(presentations["kernels"])
        presentation_counts = tuple(int(value) for value in presentations["presentation_counts"])
        subset_replicates = int(presentations["subset_replicates"])
        presentation_repeats = int(presentations["repeats"])
        capacity_counts = tuple(int(value) for value in capacity["kernel_counts"])
        capacity_repeats = int(capacity["repeats"])

    # Gate: compare all implementations and every available worker count once.
    gate_order: list[tuple[str, int]] = [
        ("flat", 1),
        ("kernel_memo", 1),
        ("kernel_memo_parallel", primary_workers),
    ]
    gate_order.extend(("factorized", workers) for workers in gate_workers)
    for gate_kernels in gate_kernel_counts:
        for gate_presentations in gate_presentation_counts:
            indices = presentation_subset(gate_presentations, 0)
            for index, (method, workers) in enumerate(gate_order):
                jobs.append(
                    Job(
                        job_id=(
                            f"gate-k{gate_kernels}-p{gate_presentations}-"
                            f"{method}-w{workers}"
                        ),
                        study="correctness_gate",
                        case=f"{gate_kernels}k_{gate_presentations}p",
                        repeat=0,
                        order_index=index,
                        method=method,
                        workers=workers,
                        kernel_count=gate_kernels,
                        presentation_indices=indices,
                    )
                )

    # Main method ladder: spend repeats on the near-break-even strong pair,
    # while retaining fewer flat and schedule-matched parallel repetitions.
    for kernel_count in method_counts:
        maximum_repeats = max(
            sequential_pair_repeats,
            flat_pair_repeats,
            parallel_pair_repeats,
        )
        for repeat in range(maximum_repeats):
            ladder: list[tuple[str, int]] = []
            if repeat < flat_pair_repeats:
                ladder.append(("flat", 1))
            if repeat < sequential_pair_repeats:
                ladder.extend((("kernel_memo", 1), ("factorized", 1)))
            if repeat < parallel_pair_repeats:
                ladder.extend(
                    (
                        ("kernel_memo_parallel", primary_workers),
                        ("factorized", primary_workers),
                    )
                )
            for index, (method, workers) in enumerate(
                rotate(tuple(ladder), repeat)
            ):
                jobs.append(
                    Job(
                        job_id=(
                            f"methods-k{kernel_count}-p18-r{repeat}-"
                            f"{method}-w{workers}"
                        ),
                        study="method_ladder",
                        case=f"{kernel_count}k_18p",
                        repeat=repeat,
                        order_index=index,
                        method=method,
                        workers=workers,
                        kernel_count=kernel_count,
                        presentation_indices=all_presentations,
                    )
                )

    # Strong scaling: Latin-rotate worker order over repeats.
    for repeat in range(scaling_repeats):
        for index, workers in enumerate(rotate(scaling_workers, repeat)):
            jobs.append(
                Job(
                    job_id=f"scaling-k{scaling_kernels}-p18-r{repeat}-w{workers}",
                    study="parallel_scaling",
                    case=f"{scaling_kernels}k_18p",
                    repeat=repeat,
                    order_index=index,
                    method="factorized",
                    workers=workers,
                    kernel_count=scaling_kernels,
                    presentation_indices=all_presentations,
                )
            )

    # Presentation-family scaling: three spread-out subsets reduce composition bias.
    for count in presentation_counts:
        actual_subset_replicates = 1 if count == 18 else subset_replicates
        for subset_rep in range(actual_subset_replicates):
            indices = presentation_subset(count, subset_rep)
            subset_id = hashlib.sha1(repr(indices).encode()).hexdigest()[:8]
            for repeat in range(presentation_repeats):
                order = (
                    (("kernel_memo", 1), ("factorized", 1))
                    if repeat % 2 == 0
                    else (("factorized", 1), ("kernel_memo", 1))
                )
                for index, (method, workers) in enumerate(order):
                    jobs.append(
                        Job(
                            job_id=(
                                f"present-k{presentation_kernels}-p{count}-"
                                f"s{subset_id}-r{repeat}-{method}"
                            ),
                            study="presentation_scaling",
                            case=f"{presentation_kernels}k_{count}p_s{subset_id}",
                            repeat=repeat,
                            order_index=index,
                            method=method,
                            workers=workers,
                            kernel_count=presentation_kernels,
                            presentation_indices=indices,
                        )
                    )

    # Capacity is a throughput curve, not another strong-scaling claim.
    for kernel_count in capacity_counts:
        for repeat in range(capacity_repeats):
            jobs.append(
                Job(
                    job_id=(
                        f"capacity-k{kernel_count}-p18-r{repeat}-"
                        f"w{throughput_workers}"
                    ),
                    study="capacity_scan",
                    case=f"{kernel_count}k_18p",
                    repeat=repeat,
                    order_index=0,
                    method="factorized",
                    workers=throughput_workers,
                    kernel_count=kernel_count,
                    presentation_indices=all_presentations,
                )
            )
    return jobs


def run_job(job: Job) -> RunRow:
    kernels, kernel_selection = stratified_kernels(job.kernel_count)
    all_presentations = make_presentations(18)
    presentations = tuple(all_presentations[index] for index in job.presentation_indices)
    sampler = PeakMemorySampler()
    gc.collect()
    sampler.start()
    started = time.perf_counter()
    if job.method in {"factorized", "kernel_memo_parallel"}:
        result = METHODS[job.method](kernels, presentations, workers=job.workers)
    else:
        result = METHODS[job.method](kernels, presentations)
    elapsed = time.perf_counter() - started
    peak_mb = sampler.stop()
    metrics = result.metrics
    row = RunRow(
        job_id=job.job_id,
        study=job.study,
        case=job.case,
        repeat=job.repeat,
        order_index=job.order_index,
        method=job.method,
        workers=job.workers,
        kernel_count=job.kernel_count,
        kernel_selection=kernel_selection,
        presentation_count=len(presentations),
        presentation_set="|".join(item.name for item in presentations),
        elapsed_s=elapsed,
        worker_slot_s=elapsed * job.workers,
        peak_process_tree_rss_mb=peak_mb,
        digest=semantic_digest(result),
        valid_kernels=len(result.valid_kernels),
        candidates=len(result.candidate_signatures),
        frontier=len(result.frontier),
        suite_size=None if result.minimum_suite is None else len(result.minimum_suite),
        graph_builds=metrics["graph_builds"],
        graph_states=metrics["graph_states"],
        graph_transitions=metrics["graph_transitions"],
        observation_calls=metrics["observation_calls"],
        policy_calls=metrics["policy_calls"],
        transition_calls=metrics["transition_calls"],
        prefix_groups=metrics["prefix_groups"],
        completed_at=datetime.now().astimezone().isoformat(),
    )
    print(
        f"[{job.study}] {job.job_id}: {elapsed:.3f}s, "
        f"digest={row.digest[:10]}, candidates={row.candidates}",
        flush=True,
    )
    del result, kernels, presentations
    gc.collect()
    return row


def row_from_dict(payload: dict[str, str]) -> RunRow:
    integer_fields = {
        "repeat",
        "order_index",
        "workers",
        "kernel_count",
        "presentation_count",
        "valid_kernels",
        "candidates",
        "frontier",
        "graph_builds",
        "graph_states",
        "graph_transitions",
        "observation_calls",
        "policy_calls",
        "transition_calls",
        "prefix_groups",
    }
    float_fields = {"elapsed_s", "worker_slot_s", "peak_process_tree_rss_mb"}
    converted: dict[str, object] = dict(payload)
    for field in integer_fields:
        converted[field] = int(payload[field])
    for field in float_fields:
        converted[field] = None if payload[field] == "" else float(payload[field])
    converted["suite_size"] = None if payload["suite_size"] == "" else int(payload["suite_size"])
    return RunRow(**converted)


def append_row(path: Path, row: RunRow) -> None:
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(row)))
        if not exists:
            writer.writeheader()
        writer.writerow(asdict(row))
        handle.flush()
        os.fsync(handle.fileno())


def load_rows(path: Path) -> list[RunRow]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return [row_from_dict(dict(row)) for row in csv.DictReader(handle)]


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def bootstrap_median_ci(values: list[float], seed: int) -> tuple[float, float]:
    if len(values) < 2:
        return values[0], values[0]
    rng = random.Random(seed)
    bootstraps = [
        statistics.median(rng.choices(values, k=len(values))) for _ in range(4000)
    ]
    return percentile(bootstraps, 0.025), percentile(bootstraps, 0.975)


def grouped(rows: Iterable[RunRow], key_fields: tuple[str, ...]) -> dict[tuple[object, ...], list[RunRow]]:
    result: dict[tuple[object, ...], list[RunRow]] = {}
    for row in rows:
        key = tuple(getattr(row, field) for field in key_fields)
        result.setdefault(key, []).append(row)
    return result


def validate_semantics(rows: list[RunRow]) -> list[dict[str, object]]:
    checks: list[dict[str, object]] = []
    for (study, case, repeat), group in grouped(
        rows, ("study", "case", "repeat")
    ).items():
        if study == "capacity_scan":
            continue
        digests = {row.digest for row in group}
        status = "PASS" if len(digests) == 1 else "FAIL"
        checks.append(
            {
                "study": study,
                "case": case,
                "repeat": repeat,
                "runs": len(group),
                "digests": len(digests),
                "status": status,
            }
        )
    if any(check["status"] == "FAIL" for check in checks):
        raise AssertionError("semantic digest mismatch; inspect semantic_checks.json")
    return checks


def summarize(rows: list[RunRow], metadata: dict[str, object]) -> dict[str, object]:
    checks = validate_semantics(rows)
    method_rows = [row for row in rows if row.study == "method_ladder"]
    scaling_rows = [row for row in rows if row.study == "parallel_scaling"]
    presentation_rows = [row for row in rows if row.study == "presentation_scaling"]
    capacity_rows = [row for row in rows if row.study == "capacity_scan"]

    method_summary: list[dict[str, object]] = []
    for (case, method, workers), group in grouped(
        method_rows, ("case", "method", "workers")
    ).items():
        values = [row.elapsed_s for row in group]
        low, high = bootstrap_median_ci(values, seed=17 + len(method_summary))
        method_summary.append(
            {
                "case": case,
                "method": method,
                "workers": workers,
                "runs": len(values),
                "median_s": statistics.median(values),
                "ci95_low_s": low,
                "ci95_high_s": high,
                "median_peak_rss_mb": statistics.median(
                    row.peak_process_tree_rss_mb
                    for row in group
                    if row.peak_process_tree_rss_mb is not None
                )
                if any(row.peak_process_tree_rss_mb is not None for row in group)
                else None,
                "policy_calls": group[0].policy_calls,
                "transition_calls": group[0].transition_calls,
                "graph_builds": group[0].graph_builds,
            }
        )

    method_effects: list[dict[str, object]] = []
    for case in sorted({row.case for row in method_rows}):
        case_rows = [row for row in method_rows if row.case == case]
        by_repeat = grouped(case_rows, ("repeat",))
        memo_to_factorized: list[float] = []
        flat_to_memo: list[float] = []
        flat_to_parallel: list[float] = []
        parallel_speedups: list[float] = []
        for group in by_repeat.values():
            lookup = {(row.method, row.workers): row for row in group}
            memo = lookup.get(("kernel_memo", 1))
            flat = lookup.get(("flat", 1))
            factor_one = lookup.get(("factorized", 1))
            factor_parallel_rows = [
                row
                for row in group
                if row.method == "factorized" and row.workers > 1
            ]
            memo_parallel_rows = [
                row for row in group if row.method == "kernel_memo_parallel"
            ]
            if memo is not None and factor_one is not None:
                memo_to_factorized.append(memo.elapsed_s / factor_one.elapsed_s)
            if flat is not None and memo is not None:
                flat_to_memo.append(flat.elapsed_s / memo.elapsed_s)
            if factor_parallel_rows and memo_parallel_rows:
                parallel_speedups.append(
                    memo_parallel_rows[0].elapsed_s
                    / factor_parallel_rows[0].elapsed_s
                )
            if flat is not None and factor_parallel_rows:
                flat_to_parallel.append(
                    flat.elapsed_s / factor_parallel_rows[0].elapsed_s
                )
        memo_row = next(row for row in case_rows if row.method == "kernel_memo")
        factor_row = next(
            row for row in case_rows if row.method == "factorized" and row.workers == 1
        )
        method_effects.append(
            {
                "case": case,
                "sequential_pair_repeats": len(memo_to_factorized),
                "flat_pair_repeats": len(flat_to_memo),
                "parallel_pair_repeats": len(parallel_speedups),
                "flat_to_kernel_memo_paired_median": statistics.median(flat_to_memo),
                "kernel_memo_to_factorized1_paired_median": statistics.median(
                    memo_to_factorized
                ),
                "parallel_kernel_memo_to_factorized_paired_median": statistics.median(
                    parallel_speedups
                ),
                "flat_to_factorized_parallel_paired_median": statistics.median(
                    flat_to_parallel
                ),
                "semantic_step_call_reduction": (
                    1 - factor_row.policy_calls / memo_row.policy_calls
                ),
                "mechanism_graph_reduction": (
                    1
                    - memo_row.graph_builds
                    / next(row for row in case_rows if row.method == "flat").graph_builds
                ),
            }
        )

    scaling_summary: list[dict[str, object]] = []
    if scaling_rows:
        baseline = statistics.median(
            row.elapsed_s for row in scaling_rows if row.workers == 1
        )
        for (workers,), group in grouped(scaling_rows, ("workers",)).items():
            values = [row.elapsed_s for row in group]
            median_s = statistics.median(values)
            scaling_summary.append(
                {
                    "workers": workers,
                    "runs": len(values),
                    "median_s": median_s,
                    "speedup": baseline / median_s,
                    "efficiency": baseline / (workers * median_s),
                    "median_worker_slot_s": statistics.median(
                        row.worker_slot_s for row in group
                    ),
                }
            )

    presentation_summary: list[dict[str, object]] = []
    for (presentation_count,), count_group in grouped(
        presentation_rows, ("presentation_count",)
    ).items():
        paired: list[float] = []
        for group in grouped(count_group, ("case", "repeat")).values():
            lookup = {row.method: row for row in group}
            paired.append(
                lookup["kernel_memo"].elapsed_s / lookup["factorized"].elapsed_s
            )
        memo = next(row for row in count_group if row.method == "kernel_memo")
        factor = next(row for row in count_group if row.method == "factorized")
        presentation_summary.append(
            {
                "presentation_count": presentation_count,
                "paired_runs": len(paired),
                "paired_speedup_median": statistics.median(paired),
                "paired_speedup_min": min(paired),
                "paired_speedup_max": max(paired),
                "semantic_step_call_reduction": 1
                - factor.policy_calls / memo.policy_calls,
            }
        )

    capacity_summary: list[dict[str, object]] = []
    for (kernel_count,), group in grouped(capacity_rows, ("kernel_count",)).items():
        median_s = statistics.median(row.elapsed_s for row in group)
        median_candidates = statistics.median(row.candidates for row in group)
        capacity_summary.append(
            {
                "kernel_count": kernel_count,
                "workers": group[0].workers,
                "runs": len(group),
                "median_s": median_s,
                "median_candidates": median_candidates,
                "candidate_throughput_per_s": median_candidates / median_s,
            }
        )

    completed_studies = {row.study for row in rows}
    expected_studies = {
        "correctness_gate",
        "method_ladder",
        "parallel_scaling",
        "presentation_scaling",
        "capacity_scan",
    }
    return {
        "status": (
            "paper_candidate_results_semantics_checked"
            if expected_studies <= completed_studies
            else "partial_results_semantics_checked"
        ),
        "metadata": metadata,
        "run_count": len(rows),
        "semantic_checks": checks,
        "method_summary": sorted(
            method_summary, key=lambda item: (item["case"], item["method"], item["workers"])
        ),
        "method_effects": method_effects,
        "parallel_scaling": sorted(scaling_summary, key=lambda item: item["workers"]),
        "presentation_scaling": sorted(
            presentation_summary, key=lambda item: item["presentation_count"]
        ),
        "capacity_scan": sorted(
            capacity_summary, key=lambda item: item["kernel_count"]
        ),
    }


def write_summary_markdown(summary: dict[str, object], path: Path) -> None:
    metadata = summary["metadata"]
    lines = [
        "# LayerProbe 截止日前实验汇总",
        "",
        f"- 状态：`{summary['status']}`",
        f"- 配置：`{metadata['profile_name']}` / `{metadata['mode']}`",
        f"- Python：`{metadata['python_executable']}`",
        f"- 物理核 / 正式主 worker / SMT 吞吐 worker："
        f"{metadata['physical_cores']} / {metadata['primary_workers']} / "
        f"{metadata['throughput_workers']}",
        f"- 总任务数：{summary['run_count']}",
        "",
        "## 方法阶梯",
        "",
        "| 规模 | 方法 | worker | 重复 | 中位秒 | 95% bootstrap CI | 中位峰值 RSS MB |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for item in summary["method_summary"]:
        memory = item["median_peak_rss_mb"]
        lines.append(
            f"| {item['case']} | {item['method']} | {item['workers']} | "
            f"{item['runs']} | {item['median_s']:.3f} | "
            f"[{item['ci95_low_s']:.3f}, {item['ci95_high_s']:.3f}] | "
            f"{'NA' if memory is None else f'{memory:.1f}'} |"
        )
    lines.extend(
        [
            "",
            "## 复用效应",
            "",
            "| 规模 | flat→kernel_memo | kernel_memo→factorized(1) | 同调度并行 memo→factorized | flat→factorized(并行) | 语义步调用减少 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for item in summary["method_effects"]:
        lines.append(
            f"| {item['case']} | {item['flat_to_kernel_memo_paired_median']:.3f}× | "
            f"{item['kernel_memo_to_factorized1_paired_median']:.3f}× | "
            f"{item['parallel_kernel_memo_to_factorized_paired_median']:.3f}× | "
            f"{item['flat_to_factorized_parallel_paired_median']:.3f}× | "
            f"{100 * item['semantic_step_call_reduction']:.1f}% |"
        )
    lines.extend(
        [
            "",
            "## 单机强扩展",
            "",
            "| worker | 重复 | 中位秒 | 相对 1 worker 加速 | 并行效率 | worker-slot 秒 |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in summary["parallel_scaling"]:
        lines.append(
            f"| {item['workers']} | {item['runs']} | {item['median_s']:.3f} | "
            f"{item['speedup']:.3f}× | {100 * item['efficiency']:.1f}% | "
            f"{item['median_worker_slot_s']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## 呈现族规模",
            "",
            "| 呈现数 | 配对数 | kernel_memo→factorized 中位加速 | 范围 | 语义步调用减少 |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for item in summary["presentation_scaling"]:
        lines.append(
            f"| {item['presentation_count']} | {item['paired_runs']} | "
            f"{item['paired_speedup_median']:.3f}× | "
            f"[{item['paired_speedup_min']:.3f}, {item['paired_speedup_max']:.3f}] | "
            f"{100 * item['semantic_step_call_reduction']:.1f}% |"
        )
    lines.extend(
        [
            "",
            "## 使用边界",
            "",
            "这些数据只支持当前离散刹车任务、4 个声明代理和 18 种呈现。",
            "它们不支持真实学习效果、真实受众诊断、多领域泛化、GPU 或多节点扩展性主张。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def code_fingerprint(config_path: Path) -> tuple[str, list[str]]:
    files = sorted((ROOT / "src").rglob("*.py"))
    files.extend(
        [
            Path(__file__).resolve(),
            config_path.resolve(),
        ]
    )
    digest = hashlib.sha256()
    relative: list[str] = []
    for path in files:
        label = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
        relative.append(label)
        digest.update(label.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest(), relative


def machine_metadata(
    config: dict[str, object],
    config_path: Path,
    mode: str,
    primary_workers: int,
    throughput_workers: int,
    maximum_workers: int,
) -> dict[str, object]:
    fingerprint, fingerprint_files = code_fingerprint(config_path)
    total_memory = None
    if psutil is not None:
        total_memory = psutil.virtual_memory().total / (1024**3)
    return {
        "started_at": datetime.now().astimezone().isoformat(),
        "profile_name": config["profile_name"],
        "mode": mode,
        "platform": platform.platform(),
        "python": sys.version,
        "python_executable": sys.executable,
        "physical_cores": physical_cpu_count(),
        "logical_cores": os.cpu_count(),
        "maximum_workers": maximum_workers,
        "primary_workers": primary_workers,
        "throughput_workers": throughput_workers,
        "physical_worker_endpoint": physical_cpu_count(),
        "total_memory_gib": total_memory,
        "available_memory_gib_at_start": available_memory_gib(),
        "psutil_available": psutil is not None,
        "code_fingerprint_sha256": fingerprint,
        "fingerprint_files": fingerprint_files,
        "internal_experiment_freeze": config["internal_experiment_freeze"],
        "venue_deadline_date": config["venue_deadline_date"],
        "claim_scope": config["paper_scope"],
    }


def enforce_preflight(
    config: dict[str, object],
    *,
    maximum_workers: int,
    ignore_freeze: bool,
) -> None:
    if sys.version_info < (3, 12):
        raise RuntimeError("paper profile requires Python 3.12 or newer")
    physical = physical_cpu_count()
    minimum_cores = int(config["minimum_physical_cores"])
    if physical < minimum_cores:
        raise RuntimeError(
            f"detected {physical} physical cores; profile requires at least {minimum_cores}"
        )
    logical = os.cpu_count() or physical
    if maximum_workers > logical:
        raise RuntimeError("maximum workers must not exceed detected logical processors")
    available = available_memory_gib()
    minimum_memory = float(config["minimum_available_memory_gib"])
    if available is not None and available < minimum_memory:
        raise RuntimeError(
            f"only {available:.1f} GiB available; wait until at least "
            f"{minimum_memory:.1f} GiB is free"
        )
    freeze = datetime.fromisoformat(str(config["internal_experiment_freeze"]))
    if not ignore_freeze and datetime.now().astimezone() >= freeze:
        raise RuntimeError(
            f"internal experiment freeze {freeze.isoformat()} has passed; "
            "use remaining time for analysis and submission"
        )


def select_jobs(jobs: list[Job], only: tuple[str, ...]) -> list[Job]:
    if not only:
        return jobs
    selected = set(only)
    return [job for job in jobs if job.study in selected]


def write_progress(
    path: Path,
    *,
    planned: int,
    completed: int,
    current_job: str | None,
    status: str,
) -> None:
    payload = {
        "status": status,
        "planned_jobs": planned,
        "completed_jobs": completed,
        "current_job": current_job,
        "updated_at": datetime.now().astimezone().isoformat(),
    }
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "paper"), default="paper")
    parser.add_argument(
        "--only",
        action="append",
        choices=(
            "correctness_gate",
            "method_ladder",
            "parallel_scaling",
            "presentation_scaling",
            "capacity_scan",
        ),
        default=[],
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--ignore-freeze", action="store_true")
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = load_config(config_path)
    maximum_workers = min(int(config["maximum_workers"]), os.cpu_count() or 1)
    primary_workers = min(int(config["primary_workers"]), maximum_workers)
    throughput_workers = min(int(config["throughput_workers"]), maximum_workers)
    enforce_preflight(
        config,
        maximum_workers=maximum_workers,
        ignore_freeze=args.ignore_freeze,
    )
    jobs = select_jobs(
        build_jobs(
            config,
            args.mode,
            primary_workers,
            throughput_workers,
            maximum_workers,
        ),
        tuple(args.only),
    )
    if args.dry_run:
        payload = {
            "profile": config["profile_name"],
            "mode": args.mode,
            "maximum_workers": maximum_workers,
            "primary_workers": primary_workers,
            "throughput_workers": throughput_workers,
            "job_count": len(jobs),
            "studies": {
                study: sum(job.study == study for job in jobs)
                for study in sorted({job.study for job in jobs})
            },
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    output = args.output.resolve()
    if output.exists() and not args.resume:
        raise FileExistsError(
            f"{output} already exists; choose a new directory or pass --resume"
        )
    output.mkdir(parents=True, exist_ok=True)
    runs_path = output / "runs.csv"
    progress_path = output / "progress.json"
    rows = load_rows(runs_path)
    completed_ids = {row.job_id for row in rows}

    metadata_path = output / "metadata.json"
    if metadata_path.exists() and args.resume:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        current_fingerprint, _ = code_fingerprint(config_path)
        if current_fingerprint != metadata["code_fingerprint_sha256"]:
            raise RuntimeError("code/config fingerprint changed; resume is refused")
    else:
        metadata = machine_metadata(
            config,
            config_path,
            args.mode,
            primary_workers,
            throughput_workers,
            maximum_workers,
        )
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (output / "frozen_config.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # Warm-up is intentionally outside the paper ledger.
    if not rows:
        warmup = config["warmup"]
        warm_kernels = min(
            int(warmup["kernels"]), 60 if args.mode == "smoke" else int(warmup["kernels"])
        )
        warm_presentations = make_presentations(int(warmup["presentations"]))
        run_factorized(
            make_kernels(warm_kernels),
            warm_presentations,
            workers=primary_workers,
        )
        gc.collect()

    pending = [job for job in jobs if job.job_id not in completed_ids]
    write_progress(
        progress_path,
        planned=len(jobs),
        completed=len(rows),
        current_job=None,
        status="running",
    )
    try:
        for job in pending:
            write_progress(
                progress_path,
                planned=len(jobs),
                completed=len(rows),
                current_job=job.job_id,
                status="running",
            )
            row = run_job(job)
            append_row(runs_path, row)
            rows.append(row)
    except BaseException as error:
        write_progress(
            progress_path,
            planned=len(jobs),
            completed=len(rows),
            current_job=job.job_id if "job" in locals() else None,
            status=f"failed: {type(error).__name__}: {error}",
        )
        raise

    metadata["finished_at"] = datetime.now().astimezone().isoformat()
    summary = summarize(rows, metadata)
    (output / "semantic_checks.json").write_text(
        json.dumps(summary["semantic_checks"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_summary_markdown(summary, output / "SUMMARY.md")
    write_progress(
        progress_path,
        planned=len(jobs),
        completed=len(rows),
        current_job=None,
        status="completed",
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "output": str(output),
                "runs": len(rows),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
