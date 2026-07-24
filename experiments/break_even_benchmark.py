"""Controlled single-host break-even study for semantic-step reuse.

The benchmark adds a deterministic CPU microkernel to each *computed*
policy/transition step.  Cache hits skip that microkernel together with the
policy and transition they replace.  The microkernel does not change actions,
states, traces, or cache keys; it isolates how the cost of one semantic step
changes the wall-clock value of exact reuse.

This is a sensitivity experiment, not a claim that the synthetic microkernel is
a realistic agent.  Every paired run checks a digest of all candidate
signatures before its timing is admitted to the summary.
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
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from layerprobe.evaluator import signature_for
from layerprobe.mechanics import (
    AGENT_NAMES,
    advance_belief,
    choose_action,
    ingest,
    initial_agent_memory,
    initial_state,
    observe,
    terminal_status,
    transition,
    verify_kernel,
)
from layerprobe.model import (
    AgentMemory,
    DisplayMemory,
    KernelSpec,
    Observation,
    PresentationSpec,
    Trace,
    WorldState,
)
from layerprobe.workloads import make_kernels, make_presentations


MASK64 = (1 << 64) - 1


@dataclass(frozen=True, slots=True)
class GroupResult:
    kernel_name: str
    valid: bool
    signature_digest: str
    candidates: int
    semantic_requests: int
    computed_steps: int
    cache_hits: int
    peak_cache_entries: int
    multiplicity_histogram: tuple[tuple[int, int], ...]
    elapsed_s: float


@dataclass(frozen=True, slots=True)
class AggregateResult:
    digest: str
    valid_kernels: int
    candidates: int
    semantic_requests: int
    computed_steps: int
    cache_hits: int
    peak_cache_entries: int
    multiplicity_histogram: dict[int, int]
    group_elapsed_p50_ms: float
    group_elapsed_p95_ms: float
    group_elapsed_max_ms: float


@dataclass(frozen=True, slots=True)
class RunRow:
    job_id: str
    target_added_us: float
    calibrated_rounds: int
    measured_added_us: float
    repeat: int
    order_index: int
    method: str
    workers: int
    kernels: int
    presentations: int
    elapsed_s: float
    digest: str
    valid_kernels: int
    candidates: int
    semantic_requests: int
    computed_steps: int
    cache_hits: int
    hit_rate: float
    peak_cache_entries: int
    multiplicity_histogram_json: str
    group_elapsed_p50_ms: float
    group_elapsed_p95_ms: float
    group_elapsed_max_ms: float
    completed_at: str


def _burn(rounds: int, seed: int) -> int:
    """Execute a fixed amount of deterministic integer work."""

    value = (seed ^ 0x9E3779B97F4A7C15) & MASK64
    for index in range(rounds):
        value ^= value >> 12
        value ^= (value << 25) & MASK64
        value ^= value >> 27
        value = (value * 0x2545F4914F6CDD1D + index) & MASK64
    return value


def _semantic_seed(
    state: WorldState,
    memory: AgentMemory,
    observation: Observation,
) -> int:
    speed, distance, in_goal, status = observation
    return (
        state.position * 1_000_003
        + state.speed * 100_003
        + state.step * 10_007
        + int(state.used_brake) * 1_009
        + memory.believed_speed * 101
        + memory.believed_distance * 17
        + speed * 13
        + distance * 7
        + in_goal * 3
        + status
    ) & MASK64


def _semantic_step(
    agent: str,
    state: WorldState,
    memory: AgentMemory,
    observation: Observation,
    kernel: KernelSpec,
    rounds: int,
) -> tuple[str, WorldState, AgentMemory, str]:
    _burn(rounds, _semantic_seed(state, memory, observation))
    perceived = ingest(memory, observation)
    action = choose_action(agent, perceived, kernel)
    next_state = transition(state, action, kernel)
    next_memory = advance_belief(agent, perceived, action, kernel)
    status = terminal_status(next_state, kernel)
    return action, next_state, next_memory, status


def _signature_digest(signatures: dict[str, int]) -> str:
    payload = json.dumps(
        sorted(signatures.items()),
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _simulate_costed_flat(
    kernel: KernelSpec,
    presentation: PresentationSpec,
    agent: str,
    rounds: int,
) -> Trace:
    state = initial_state(kernel)
    display_memory = DisplayMemory()
    agent_memory = initial_agent_memory(kernel)
    trace: list[tuple[Observation, str, str]] = []
    while terminal_status(state, kernel) == "running":
        observation, display_memory = observe(
            state,
            kernel,
            presentation,
            display_memory,
        )
        action, state, agent_memory, status = _semantic_step(
            agent,
            state,
            agent_memory,
            observation,
            kernel,
            rounds,
        )
        trace.append((observation, action, status))
    return tuple(trace)


def _kernel_memo_group(
    args: tuple[KernelSpec, tuple[PresentationSpec, ...], int],
) -> GroupResult:
    started = time.perf_counter()
    kernel, presentations, rounds = args
    report = verify_kernel(kernel)
    if not report.valid:
        return GroupResult(
            kernel_name=kernel.name,
            valid=False,
            signature_digest="",
            candidates=0,
            semantic_requests=0,
            computed_steps=0,
            cache_hits=0,
            peak_cache_entries=0,
            multiplicity_histogram=(),
            elapsed_s=time.perf_counter() - started,
        )

    signatures: dict[str, int] = {}
    semantic_requests = 0
    for presentation in presentations:
        traces: dict[str, Trace] = {}
        for agent in AGENT_NAMES:
            trace = _simulate_costed_flat(kernel, presentation, agent, rounds)
            traces[agent] = trace
            semantic_requests += len(trace)
        signatures[f"{kernel.name}::{presentation.name}"] = signature_for(traces)
    return GroupResult(
        kernel_name=kernel.name,
        valid=True,
        signature_digest=_signature_digest(signatures),
        candidates=len(signatures),
        semantic_requests=semantic_requests,
        computed_steps=semantic_requests,
        cache_hits=0,
        peak_cache_entries=0,
        multiplicity_histogram=(),
        elapsed_s=time.perf_counter() - started,
    )


def _memoized_costed_agent(
    kernel: KernelSpec,
    presentations: tuple[PresentationSpec, ...],
    agent: str,
    rounds: int,
) -> tuple[dict[str, Trace], int, int, Counter[int]]:
    cache: dict[
        tuple[WorldState, AgentMemory, Observation],
        tuple[str, WorldState, AgentMemory, str],
    ] = {}
    multiplicities: Counter[tuple[WorldState, AgentMemory, Observation]] = Counter()
    traces: dict[str, Trace] = {}
    requests = 0
    hits = 0
    for presentation in presentations:
        state = initial_state(kernel)
        agent_memory = initial_agent_memory(kernel)
        display_memory = DisplayMemory()
        trace: list[tuple[Observation, str, str]] = []
        while terminal_status(state, kernel) == "running":
            observation, display_memory = observe(
                state,
                kernel,
                presentation,
                display_memory,
            )
            requests += 1
            key = (state, agent_memory, observation)
            multiplicities[key] += 1
            cached = cache.get(key)
            if cached is None:
                cached = _semantic_step(
                    agent,
                    state,
                    agent_memory,
                    observation,
                    kernel,
                    rounds,
                )
                cache[key] = cached
            else:
                hits += 1
            action, state, agent_memory, status = cached
            trace.append((observation, action, status))
        traces[presentation.name] = tuple(trace)

    histogram: Counter[int] = Counter(multiplicities.values())
    return traces, requests, hits, histogram


def _factorized_group(
    args: tuple[KernelSpec, tuple[PresentationSpec, ...], int],
) -> GroupResult:
    started = time.perf_counter()
    kernel, presentations, rounds = args
    report = verify_kernel(kernel)
    if not report.valid:
        return GroupResult(
            kernel_name=kernel.name,
            valid=False,
            signature_digest="",
            candidates=0,
            semantic_requests=0,
            computed_steps=0,
            cache_hits=0,
            peak_cache_entries=0,
            multiplicity_histogram=(),
            elapsed_s=time.perf_counter() - started,
        )

    traces_by_presentation: dict[str, dict[str, Trace]] = {
        presentation.name: {} for presentation in presentations
    }
    semantic_requests = 0
    cache_hits = 0
    peak_cache_entries = 0
    multiplicity_histogram: Counter[int] = Counter()
    for agent in AGENT_NAMES:
        traces, requests, hits, histogram = _memoized_costed_agent(
            kernel,
            presentations,
            agent,
            rounds,
        )
        semantic_requests += requests
        cache_hits += hits
        unique_keys = requests - hits
        peak_cache_entries = max(peak_cache_entries, unique_keys)
        multiplicity_histogram.update(histogram)
        for presentation_name, trace in traces.items():
            traces_by_presentation[presentation_name][agent] = trace

    signatures = {
        f"{kernel.name}::{presentation.name}": signature_for(
            traces_by_presentation[presentation.name]
        )
        for presentation in presentations
    }
    return GroupResult(
        kernel_name=kernel.name,
        valid=True,
        signature_digest=_signature_digest(signatures),
        candidates=len(signatures),
        semantic_requests=semantic_requests,
        computed_steps=semantic_requests - cache_hits,
        cache_hits=cache_hits,
        peak_cache_entries=peak_cache_entries,
        multiplicity_histogram=tuple(sorted(multiplicity_histogram.items())),
        elapsed_s=time.perf_counter() - started,
    )


def _percentile(values: list[float], probability: float) -> float:
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


def _aggregate(group_results: Iterable[GroupResult]) -> AggregateResult:
    ordered = sorted(group_results, key=lambda item: item.kernel_name)
    digest = hashlib.sha256()
    valid_kernels = 0
    candidates = 0
    semantic_requests = 0
    computed_steps = 0
    cache_hits = 0
    peak_cache_entries = 0
    multiplicity_histogram: Counter[int] = Counter()
    group_times: list[float] = []
    for group in ordered:
        digest.update(group.kernel_name.encode())
        digest.update(b"\0")
        digest.update(str(int(group.valid)).encode())
        digest.update(b"\0")
        digest.update(group.signature_digest.encode())
        digest.update(b"\0")
        valid_kernels += int(group.valid)
        candidates += group.candidates
        semantic_requests += group.semantic_requests
        computed_steps += group.computed_steps
        cache_hits += group.cache_hits
        peak_cache_entries = max(peak_cache_entries, group.peak_cache_entries)
        multiplicity_histogram.update(dict(group.multiplicity_histogram))
        group_times.append(group.elapsed_s)
    return AggregateResult(
        digest=digest.hexdigest(),
        valid_kernels=valid_kernels,
        candidates=candidates,
        semantic_requests=semantic_requests,
        computed_steps=computed_steps,
        cache_hits=cache_hits,
        peak_cache_entries=peak_cache_entries,
        multiplicity_histogram=dict(sorted(multiplicity_histogram.items())),
        group_elapsed_p50_ms=1_000 * _percentile(group_times, 0.50),
        group_elapsed_p95_ms=1_000 * _percentile(group_times, 0.95),
        group_elapsed_max_ms=1_000 * max(group_times),
    )


def run_method(
    method: str,
    kernels: tuple[KernelSpec, ...],
    presentations: tuple[PresentationSpec, ...],
    rounds: int,
    workers: int,
) -> AggregateResult:
    if method not in {"kernel_memo", "factorized"}:
        raise ValueError(f"unsupported method: {method}")
    function = _kernel_memo_group if method == "kernel_memo" else _factorized_group
    tasks = tuple((kernel, presentations, rounds) for kernel in kernels)
    if workers == 1 or len(tasks) <= 1:
        return _aggregate(function(task) for task in tasks)
    chunksize = max(1, len(tasks) // (workers * 4))
    with ProcessPoolExecutor(max_workers=min(workers, len(tasks))) as executor:
        return _aggregate(executor.map(function, tasks, chunksize=chunksize))


def _measure_burn(rounds: int, samples: int) -> float:
    started = time.perf_counter_ns()
    accumulator = 0
    for sample in range(samples):
        accumulator ^= _burn(rounds, sample)
    elapsed = time.perf_counter_ns() - started
    if accumulator == -1:  # pragma: no cover - keeps the result observably live
        raise AssertionError("unreachable")
    return elapsed / samples / 1_000


def calibrate_targets(targets_us: tuple[float, ...]) -> dict[float, tuple[int, float]]:
    maximum = max(targets_us)
    calibration_rounds = 512
    samples = 2_000
    base_us = _measure_burn(0, samples)
    calibrated_us = _measure_burn(calibration_rounds, samples)
    per_round_us = max((calibrated_us - base_us) / calibration_rounds, 1e-6)
    result: dict[float, tuple[int, float]] = {}
    for target in targets_us:
        rounds = 0 if target <= 0 else max(1, round(target / per_round_us))
        measured = _measure_burn(rounds, max(400, min(samples, round(30_000 / max(1, rounds)))))
        result[target] = (rounds, max(0.0, measured - base_us))
    if result[maximum][1] <= 0:
        raise RuntimeError("CPU microkernel calibration failed")
    return result


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
                    target_added_us=float(payload["target_added_us"]),
                    calibrated_rounds=int(payload["calibrated_rounds"]),
                    measured_added_us=float(payload["measured_added_us"]),
                    repeat=int(payload["repeat"]),
                    order_index=int(payload["order_index"]),
                    method=payload["method"],
                    workers=int(payload["workers"]),
                    kernels=int(payload["kernels"]),
                    presentations=int(payload["presentations"]),
                    elapsed_s=float(payload["elapsed_s"]),
                    digest=payload["digest"],
                    valid_kernels=int(payload["valid_kernels"]),
                    candidates=int(payload["candidates"]),
                    semantic_requests=int(payload["semantic_requests"]),
                    computed_steps=int(payload["computed_steps"]),
                    cache_hits=int(payload["cache_hits"]),
                    hit_rate=float(payload["hit_rate"]),
                    peak_cache_entries=int(payload["peak_cache_entries"]),
                    multiplicity_histogram_json=payload.get(
                        "multiplicity_histogram_json", "{}"
                    ),
                    group_elapsed_p50_ms=float(payload["group_elapsed_p50_ms"]),
                    group_elapsed_p95_ms=float(payload["group_elapsed_p95_ms"]),
                    group_elapsed_max_ms=float(payload["group_elapsed_max_ms"]),
                    completed_at=payload["completed_at"],
                )
            )
    return rows


def _bootstrap_median_ci(values: list[float], seed: int) -> tuple[float, float]:
    if len(values) == 1:
        return values[0], values[0]
    rng = random.Random(seed)
    medians = [
        statistics.median(rng.choices(values, k=len(values))) for _ in range(4_000)
    ]
    return _percentile(medians, 0.025), _percentile(medians, 0.975)


def summarize(rows: list[RunRow]) -> dict[str, object]:
    pairs: dict[tuple[float, int], dict[str, RunRow]] = {}
    for row in rows:
        pairs.setdefault((row.target_added_us, row.repeat), {})[row.method] = row

    semantic_checks: list[dict[str, object]] = []
    ratios: dict[float, list[float]] = {}
    for (target, repeat), pair in sorted(pairs.items()):
        complete = {"kernel_memo", "factorized"} <= pair.keys()
        digest_equal = complete and pair["kernel_memo"].digest == pair["factorized"].digest
        semantic_checks.append(
            {
                "target_added_us": target,
                "repeat": repeat,
                "complete": complete,
                "digest_equal": digest_equal,
            }
        )
        if complete:
            if not digest_equal:
                raise AssertionError(
                    f"semantic digest mismatch at target={target}, repeat={repeat}"
                )
            ratios.setdefault(target, []).append(
                pair["kernel_memo"].elapsed_s / pair["factorized"].elapsed_s
            )

    target_summary: list[dict[str, object]] = []
    for target, values in sorted(ratios.items()):
        low, high = _bootstrap_median_ci(values, seed=20260724 + round(target))
        factorized = next(
            row
            for row in rows
            if row.target_added_us == target and row.method == "factorized"
        )
        target_summary.append(
            {
                "target_added_us": target,
                "measured_added_us": factorized.measured_added_us,
                "paired_repeats": len(values),
                "paired_ratio_median": statistics.median(values),
                "paired_ratio_ci95_low": low,
                "paired_ratio_ci95_high": high,
                "paired_ratio_min": min(values),
                "paired_ratio_max": max(values),
                "semantic_requests": factorized.semantic_requests,
                "computed_steps": factorized.computed_steps,
                "cache_hits": factorized.cache_hits,
                "hit_rate": factorized.hit_rate,
                "peak_cache_entries": factorized.peak_cache_entries,
                "multiplicity_histogram": {
                    str(key): value
                    for key, value in sorted(
                        json.loads(
                            factorized.multiplicity_histogram_json
                        ).items(),
                        key=lambda item: int(item[0]),
                    )
                },
            }
        )
    return {
        "status": (
            "complete_semantics_checked"
            if semantic_checks
            and all(item["complete"] and item["digest_equal"] for item in semantic_checks)
            else "partial_semantics_checked"
        ),
        "run_count": len(rows),
        "semantic_checks": semantic_checks,
        "targets": target_summary,
    }


def _fingerprint() -> str:
    digest = hashlib.sha256()
    for path in sorted((ROOT / "src").rglob("*.py")) + [Path(__file__).resolve()]:
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--kernels", type=int, default=24_624)
    parser.add_argument("--presentations", type=int, default=18)
    parser.add_argument(
        "--target-us",
        type=float,
        nargs="+",
        default=(0.0, 10.0, 50.0, 100.0),
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.workers < 1 or args.workers > (os.cpu_count() or 1):
        raise ValueError("workers must be within the host's logical CPU count")
    if args.repeats < 1:
        raise ValueError("repeats must be positive")
    if not 1 <= args.kernels <= 24_624:
        raise ValueError("kernels must be between 1 and 24624")
    if not 1 <= args.presentations <= 18:
        raise ValueError("presentations must be between 1 and 18")
    targets = tuple(sorted({float(value) for value in args.target_us}))
    if targets[0] < 0:
        raise ValueError("target microseconds must be non-negative")

    output = args.output.resolve()
    if output.exists() and not args.resume:
        raise FileExistsError(
            f"{output} exists; choose a new directory or pass --resume"
        )
    output.mkdir(parents=True, exist_ok=True)
    runs_path = output / "runs.csv"
    rows = _load_rows(runs_path)
    completed = {row.job_id for row in rows}

    metadata_path = output / "metadata.json"
    if metadata_path.exists() and args.resume:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata["code_fingerprint_sha256"] != _fingerprint():
            raise RuntimeError("code fingerprint changed; resume refused")
        calibration = {
            float(key): (int(value["rounds"]), float(value["measured_added_us"]))
            for key, value in metadata["calibration"].items()
        }
    else:
        calibration = calibrate_targets(targets)
        metadata = {
            "started_at": datetime.now().astimezone().isoformat(),
            "platform": platform.platform(),
            "python": sys.version,
            "python_executable": sys.executable,
            "workers": args.workers,
            "kernels": args.kernels,
            "presentations": args.presentations,
            "repeats": args.repeats,
            "targets_us": targets,
            "calibration": {
                str(target): {
                    "rounds": rounds,
                    "measured_added_us": measured,
                }
                for target, (rounds, measured) in calibration.items()
            },
            "code_fingerprint_sha256": _fingerprint(),
            "claim_scope": (
                "controlled deterministic per-semantic-step CPU-cost sensitivity; "
                "not a real-agent workload"
            ),
        }
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    kernels = make_kernels(args.kernels)
    presentations = make_presentations(args.presentations)
    run_method(
        "factorized",
        make_kernels(min(300, args.kernels)),
        presentations,
        0,
        args.workers,
    )
    gc.collect()

    for target in targets:
        rounds, measured = calibration[target]
        for repeat in range(args.repeats):
            order = (
                ("kernel_memo", "factorized")
                if repeat % 2 == 0
                else ("factorized", "kernel_memo")
            )
            for order_index, method in enumerate(order):
                job_id = (
                    f"cost-{target:g}us-r{repeat}-{method}-w{args.workers}"
                )
                if job_id in completed:
                    continue
                gc.collect()
                started = time.perf_counter()
                result = run_method(
                    method,
                    kernels,
                    presentations,
                    rounds,
                    args.workers,
                )
                elapsed_s = time.perf_counter() - started
                hit_rate = (
                    result.cache_hits / result.semantic_requests
                    if result.semantic_requests
                    else 0.0
                )
                row = RunRow(
                    job_id=job_id,
                    target_added_us=target,
                    calibrated_rounds=rounds,
                    measured_added_us=measured,
                    repeat=repeat,
                    order_index=order_index,
                    method=method,
                    workers=args.workers,
                    kernels=args.kernels,
                    presentations=args.presentations,
                    elapsed_s=elapsed_s,
                    digest=result.digest,
                    valid_kernels=result.valid_kernels,
                    candidates=result.candidates,
                    semantic_requests=result.semantic_requests,
                    computed_steps=result.computed_steps,
                    cache_hits=result.cache_hits,
                    hit_rate=hit_rate,
                    peak_cache_entries=result.peak_cache_entries,
                    multiplicity_histogram_json=json.dumps(
                        result.multiplicity_histogram,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    group_elapsed_p50_ms=result.group_elapsed_p50_ms,
                    group_elapsed_p95_ms=result.group_elapsed_p95_ms,
                    group_elapsed_max_ms=result.group_elapsed_max_ms,
                    completed_at=datetime.now().astimezone().isoformat(),
                )
                _append_row(runs_path, row)
                rows.append(row)
                completed.add(job_id)
                print(
                    f"{job_id}: {elapsed_s:.3f}s digest={result.digest[:12]} "
                    f"requests={result.semantic_requests} "
                    f"computed={result.computed_steps}",
                    flush=True,
                )

    summary = summarize(rows)
    metadata["finished_at"] = datetime.now().astimezone().isoformat()
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
