"""Paired single-host benchmark with a finite-depth deliberative policy.

This experiment addresses a narrow performance question: does exact
cross-presentation reuse become useful when one policy invocation performs
meaningful decision work rather than a few arithmetic operations?

Depth zero is the original one-step policy from :mod:`layerprobe.mechanics`.
For depth ``d > 0``, the policy exhaustively enumerates all coast/brake
sequences up to ``d`` in the agent's own belief model.  Each leaf is scored by
the predicted stopping location after braking, with deviations from the
agent's original heuristic used only as a deterministic tie-breaker.  Thus the
extra work is a small, deterministic planning problem; it is neither
``sleep`` nor an artificial busy loop.

The two compared implementations deliberately differ only in semantic-step
reuse:

``kernel_memo_p8``
    Verifies a mechanism once, then evaluates every presentation separately.
``layerprobe_p8``
    Uses a cache scoped to one fixed mechanism-agent pair and keyed by the
    complete ``(world state, agent memory, observation)`` tuple.  Display
    memory and trace construction remain presentation-local.

Both methods use the same ordered mechanism-group tasks, process count, and
``ProcessPoolExecutor.map`` chunksize.  A pair is written to disk only after
both the complete candidate-signature mapping and every candidate's four-agent
observation/action/status trace digest have been compared exactly.
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
    ACTIONS,
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
    Action,
    AgentMemory,
    DisplayMemory,
    KernelSpec,
    Observation,
    PresentationSpec,
    Trace,
    WorldState,
)
from layerprobe.workloads import make_kernels, make_presentations


METHODS = ("kernel_memo_p8", "layerprobe_p8")
MAX_KERNELS = 24_624
MAX_PRESENTATIONS = 18


@dataclass(frozen=True, slots=True)
class PlannerResult:
    """One deterministic policy decision and its actual model-search work."""

    action: Action
    expanded_nodes: int


@dataclass(frozen=True, slots=True)
class GroupResult:
    """Result of one independently scheduled mechanism group."""

    kernel_name: str
    valid: bool
    candidate_signatures: tuple[tuple[str, int], ...]
    candidate_trace_digests: tuple[tuple[str, str], ...]
    semantic_requests: int
    policy_invocations: int
    expanded_nodes: int
    cache_hits: int
    peak_cache_entries: int
    elapsed_s: float


@dataclass(frozen=True, slots=True)
class AggregateResult:
    """Deterministically reduced result for one timed method execution."""

    digest: str
    trace_digest: str
    candidate_signatures: tuple[tuple[str, int], ...]
    candidate_trace_digests: tuple[tuple[str, str], ...]
    valid_kernels: int
    candidates: int
    semantic_requests: int
    policy_invocations: int
    expanded_nodes: int
    cache_hits: int
    peak_cache_entries: int
    group_elapsed_p50_ms: float
    group_elapsed_p95_ms: float
    group_elapsed_max_ms: float


@dataclass(frozen=True, slots=True)
class RunRow:
    """One admitted timing row in the append-only raw CSV."""

    job_id: str
    depth: int
    repeat: int
    order_index: int
    method: str
    workers: int
    population_kernels: int
    sample_kernels: int
    sampling: str
    presentations: int
    chunksize: int
    elapsed_s: float
    candidate_digest: str
    trace_digest: str
    valid_kernels: int
    candidates: int
    semantic_requests: int
    policy_invocations: int
    expanded_nodes: int
    cache_hits: int
    cache_hit_rate: float
    peak_cache_entries: int
    group_elapsed_p50_ms: float
    group_elapsed_p95_ms: float
    group_elapsed_max_ms: float
    completed_at: str


def _distance_outside_goal(remaining_distance: int, goal_width: int) -> tuple[int, bool]:
    """Return distance outside ``[-goal_width, 0]`` and overshoot direction."""

    if remaining_distance > 0:
        return remaining_distance, False
    if remaining_distance < -goal_width:
        return -goal_width - remaining_distance, True
    return 0, False


def _project_braking_stop(
    agent: str,
    memory: AgentMemory,
    spec: KernelSpec,
) -> tuple[AgentMemory, int]:
    """Project a full-braking stop in the agent's declared belief dynamics."""

    projected = memory
    expanded = 0
    for _ in range(spec.horizon):
        if projected.believed_speed <= 0:
            break
        projected = advance_belief(agent, projected, "brake", spec)
        expanded += 1
    return projected, expanded


def _leaf_score(
    agent: str,
    memory: AgentMemory,
    spec: KernelSpec,
    heuristic_deviations: int,
    brake_actions: int,
    action_ranks: tuple[int, ...],
) -> tuple[tuple[int, int, int, int, int, tuple[int, ...]], int]:
    """Score a search leaf and return additional projection nodes.

    The first term is a safety-oriented stopping error: an overshoot carries
    twice the distance penalty of an equally large undershoot.  When two
    sequences have equal stopping error, the planner prefers the center of the
    target interval, fewer deviations from the original agent heuristic, fewer
    brake commands, and finally lexicographic coast-before-brake order.
    """

    stopped, rollout_nodes = _project_braking_stop(agent, memory, spec)
    goal_width = spec.goal_end - spec.goal_start
    remaining = stopped.believed_distance
    outside, overshot = _distance_outside_goal(remaining, goal_width)
    safety_cost = outside * (2 if overshot else 1)
    center_error_twice = abs(2 * remaining + goal_width)
    score = (
        safety_cost,
        outside,
        center_error_twice,
        heuristic_deviations,
        brake_actions,
        action_ranks,
    )
    return score, rollout_nodes


def deliberative_action(
    agent: str,
    memory: AgentMemory,
    spec: KernelSpec,
    depth: int,
) -> PlannerResult:
    """Choose an action with exhaustive finite-depth belief-space lookahead.

    ``depth == 0`` is exactly the original policy and expands no planning
    nodes.  At positive depth, every reachable coast/brake branch is examined
    until that depth or until the believed vehicle stops.  The returned node
    count includes both tree transitions and the full-braking projections used
    to score leaves.
    """

    if depth < 0:
        raise ValueError("depth must be non-negative")
    if agent not in AGENT_NAMES:
        raise ValueError(f"unknown agent: {agent}")
    if depth == 0:
        return PlannerResult(choose_action(agent, memory, spec), 0)

    # Stack entries are (belief, remaining depth, action path, heuristic
    # deviations, brake count, 0/1 action ranks).  Explicit storage keeps the
    # planner deterministic and makes the finite search auditable.
    stack: list[
        tuple[
            AgentMemory,
            int,
            tuple[Action, ...],
            int,
            int,
            tuple[int, ...],
        ]
    ] = [(memory, depth, (), 0, 0, ())]
    best_score: tuple[int, int, int, int, int, tuple[int, ...]] | None = None
    best_action: Action | None = None
    expanded_nodes = 0

    while stack:
        (
            current,
            remaining_depth,
            path,
            deviations,
            brake_count,
            action_ranks,
        ) = stack.pop()
        heuristic_action = choose_action(agent, current, spec)

        # Push brake first so coast is visited first by the LIFO stack.  The
        # final lexicographic rank makes the result independent of visit order.
        for action in reversed(ACTIONS):
            next_memory = advance_belief(agent, current, action, spec)
            expanded_nodes += 1
            next_path = path + (action,)
            next_deviations = deviations + int(action != heuristic_action)
            next_brake_count = brake_count + int(action == "brake")
            next_ranks = action_ranks + (0 if action == "coast" else 1,)

            if remaining_depth == 1 or next_memory.believed_speed <= 0:
                score, rollout_nodes = _leaf_score(
                    agent,
                    next_memory,
                    spec,
                    next_deviations,
                    next_brake_count,
                    next_ranks,
                )
                expanded_nodes += rollout_nodes
                if best_score is None or score < best_score:
                    best_score = score
                    best_action = next_path[0]
            else:
                stack.append(
                    (
                        next_memory,
                        remaining_depth - 1,
                        next_path,
                        next_deviations,
                        next_brake_count,
                        next_ranks,
                    )
                )

    if best_action is None:  # pragma: no cover - positive depth always branches
        raise AssertionError("deliberative search produced no leaf")
    return PlannerResult(best_action, expanded_nodes)


def _semantic_step(
    agent: str,
    state: WorldState,
    memory: AgentMemory,
    observation: Observation,
    spec: KernelSpec,
    depth: int,
) -> tuple[Action, WorldState, AgentMemory, str, int]:
    """Execute the deterministic policy and mechanism transition once."""

    perceived = ingest(memory, observation)
    decision = deliberative_action(agent, perceived, spec, depth)
    next_state = transition(state, decision.action, spec)
    next_memory = advance_belief(agent, perceived, decision.action, spec)
    status = terminal_status(next_state, spec)
    return (
        decision.action,
        next_state,
        next_memory,
        status,
        decision.expanded_nodes,
    )


def _simulate_deliberative_flat(
    kernel: KernelSpec,
    presentation: PresentationSpec,
    agent: str,
    depth: int,
) -> tuple[Trace, int, int]:
    """Run one presentation with no semantic-step cache."""

    state = initial_state(kernel)
    display_memory = DisplayMemory()
    agent_memory = initial_agent_memory(kernel)
    trace: list[tuple[Observation, Action, str]] = []
    policy_invocations = 0
    expanded_nodes = 0

    while terminal_status(state, kernel) == "running":
        observation, display_memory = observe(
            state,
            kernel,
            presentation,
            display_memory,
        )
        action, state, agent_memory, status, nodes = _semantic_step(
            agent,
            state,
            agent_memory,
            observation,
            kernel,
            depth,
        )
        policy_invocations += 1
        expanded_nodes += nodes
        trace.append((observation, action, status))
    return tuple(trace), policy_invocations, expanded_nodes


def _complete_trace_digest(traces: dict[str, Trace]) -> str:
    """Hash every agent's complete observation/action/status trace stably."""

    payload = [
        {
            "agent": agent,
            "trace": traces[agent],
        }
        for agent in sorted(traces)
    ]
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _invalid_group(kernel_name: str, started: float) -> GroupResult:
    return GroupResult(
        kernel_name=kernel_name,
        valid=False,
        candidate_signatures=(),
        candidate_trace_digests=(),
        semantic_requests=0,
        policy_invocations=0,
        expanded_nodes=0,
        cache_hits=0,
        peak_cache_entries=0,
        elapsed_s=time.perf_counter() - started,
    )


def _kernel_memo_group(
    task: tuple[KernelSpec, tuple[PresentationSpec, ...], int],
) -> GroupResult:
    """Evaluate a mechanism once, without cross-presentation step reuse."""

    started = time.perf_counter()
    kernel, presentations, depth = task
    report = verify_kernel(kernel)
    if not report.valid:
        return _invalid_group(kernel.name, started)

    # Agent-major ordering matches LayerProbe below; the only semantic-step
    # difference is lookup/reuse, not a presentation-major loop permutation.
    traces_by_presentation: dict[str, dict[str, Trace]] = {
        presentation.name: {} for presentation in presentations
    }
    semantic_requests = 0
    policy_invocations = 0
    expanded_nodes = 0
    for agent in AGENT_NAMES:
        for presentation in presentations:
            trace, invocations, nodes = _simulate_deliberative_flat(
                kernel,
                presentation,
                agent,
                depth,
            )
            traces_by_presentation[presentation.name][agent] = trace
            semantic_requests += len(trace)
            policy_invocations += invocations
            expanded_nodes += nodes

    signatures = tuple(
        (
            f"{kernel.name}::{presentation.name}",
            signature_for(traces_by_presentation[presentation.name]),
        )
        for presentation in presentations
    )
    trace_digests = tuple(
        (
            f"{kernel.name}::{presentation.name}",
            _complete_trace_digest(
                traces_by_presentation[presentation.name]
            ),
        )
        for presentation in presentations
    )
    if policy_invocations != semantic_requests:
        raise AssertionError("uncached policy invocation accounting is inconsistent")
    return GroupResult(
        kernel_name=kernel.name,
        valid=True,
        candidate_signatures=tuple(sorted(signatures)),
        candidate_trace_digests=tuple(sorted(trace_digests)),
        semantic_requests=semantic_requests,
        policy_invocations=policy_invocations,
        expanded_nodes=expanded_nodes,
        cache_hits=0,
        peak_cache_entries=0,
        elapsed_s=time.perf_counter() - started,
    )


def _memoized_deliberative_agent(
    kernel: KernelSpec,
    presentations: tuple[PresentationSpec, ...],
    agent: str,
    depth: int,
) -> tuple[dict[str, Trace], int, int, int, int]:
    """Evaluate one mechanism-agent scope with the complete semantic key."""

    cache: dict[
        tuple[WorldState, AgentMemory, Observation],
        tuple[Action, WorldState, AgentMemory, str],
    ] = {}
    traces: dict[str, Trace] = {}
    semantic_requests = 0
    policy_invocations = 0
    expanded_nodes = 0
    cache_hits = 0

    for presentation in presentations:
        # These values are deliberately presentation-local and never cached.
        state = initial_state(kernel)
        agent_memory = initial_agent_memory(kernel)
        display_memory = DisplayMemory()
        trace: list[tuple[Observation, Action, str]] = []
        while terminal_status(state, kernel) == "running":
            observation, display_memory = observe(
                state,
                kernel,
                presentation,
                display_memory,
            )
            semantic_requests += 1
            key = (state, agent_memory, observation)
            cached = cache.get(key)
            if cached is None:
                (
                    action,
                    next_state,
                    next_memory,
                    status,
                    nodes,
                ) = _semantic_step(
                    agent,
                    state,
                    agent_memory,
                    observation,
                    kernel,
                    depth,
                )
                cached = (action, next_state, next_memory, status)
                cache[key] = cached
                policy_invocations += 1
                expanded_nodes += nodes
            else:
                cache_hits += 1
            action, state, agent_memory, status = cached
            trace.append((observation, action, status))
        traces[presentation.name] = tuple(trace)

    if semantic_requests != policy_invocations + cache_hits:
        raise AssertionError("cache request accounting is inconsistent")
    return (
        traces,
        semantic_requests,
        policy_invocations,
        expanded_nodes,
        cache_hits,
    )


def _layerprobe_group(
    task: tuple[KernelSpec, tuple[PresentationSpec, ...], int],
) -> GroupResult:
    """Evaluate one mechanism using fixed-scope complete-key exact reuse."""

    started = time.perf_counter()
    kernel, presentations, depth = task
    report = verify_kernel(kernel)
    if not report.valid:
        return _invalid_group(kernel.name, started)

    traces_by_presentation: dict[str, dict[str, Trace]] = {
        presentation.name: {} for presentation in presentations
    }
    semantic_requests = 0
    policy_invocations = 0
    expanded_nodes = 0
    cache_hits = 0
    peak_cache_entries = 0
    for agent in AGENT_NAMES:
        (
            traces,
            requests,
            invocations,
            nodes,
            hits,
        ) = _memoized_deliberative_agent(
            kernel,
            presentations,
            agent,
            depth,
        )
        semantic_requests += requests
        policy_invocations += invocations
        expanded_nodes += nodes
        cache_hits += hits
        peak_cache_entries = max(peak_cache_entries, invocations)
        for presentation_name, trace in traces.items():
            traces_by_presentation[presentation_name][agent] = trace

    signatures = tuple(
        (
            f"{kernel.name}::{presentation.name}",
            signature_for(traces_by_presentation[presentation.name]),
        )
        for presentation in presentations
    )
    trace_digests = tuple(
        (
            f"{kernel.name}::{presentation.name}",
            _complete_trace_digest(
                traces_by_presentation[presentation.name]
            ),
        )
        for presentation in presentations
    )
    return GroupResult(
        kernel_name=kernel.name,
        valid=True,
        candidate_signatures=tuple(sorted(signatures)),
        candidate_trace_digests=tuple(sorted(trace_digests)),
        semantic_requests=semantic_requests,
        policy_invocations=policy_invocations,
        expanded_nodes=expanded_nodes,
        cache_hits=cache_hits,
        peak_cache_entries=peak_cache_entries,
        elapsed_s=time.perf_counter() - started,
    )


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
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _candidate_digest(signatures: Iterable[tuple[str, int]]) -> str:
    digest = hashlib.sha256()
    for candidate, mask in signatures:
        digest.update(candidate.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(mask).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _candidate_trace_digest(
    trace_digests: Iterable[tuple[str, str]],
) -> str:
    digest = hashlib.sha256()
    for candidate, trace_digest in trace_digests:
        digest.update(candidate.encode("utf-8"))
        digest.update(b"\0")
        digest.update(trace_digest.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _aggregate(group_results: Iterable[GroupResult]) -> AggregateResult:
    """Reduce mechanism groups in kernel-name order, independent of completion."""

    ordered = sorted(group_results, key=lambda item: item.kernel_name)
    signatures: list[tuple[str, int]] = []
    trace_digests: list[tuple[str, str]] = []
    valid_kernels = 0
    semantic_requests = 0
    policy_invocations = 0
    expanded_nodes = 0
    cache_hits = 0
    peak_cache_entries = 0
    group_times: list[float] = []

    for group in ordered:
        valid_kernels += int(group.valid)
        signatures.extend(group.candidate_signatures)
        trace_digests.extend(group.candidate_trace_digests)
        semantic_requests += group.semantic_requests
        policy_invocations += group.policy_invocations
        expanded_nodes += group.expanded_nodes
        cache_hits += group.cache_hits
        peak_cache_entries = max(peak_cache_entries, group.peak_cache_entries)
        group_times.append(group.elapsed_s)

    signature_items = tuple(sorted(signatures))
    trace_digest_items = tuple(sorted(trace_digests))
    if tuple(candidate for candidate, _ in signature_items) != tuple(
        candidate for candidate, _ in trace_digest_items
    ):
        raise AssertionError("signature and trace-digest candidate ledgers differ")
    return AggregateResult(
        digest=_candidate_digest(signature_items),
        trace_digest=_candidate_trace_digest(trace_digest_items),
        candidate_signatures=signature_items,
        candidate_trace_digests=trace_digest_items,
        valid_kernels=valid_kernels,
        candidates=len(signature_items),
        semantic_requests=semantic_requests,
        policy_invocations=policy_invocations,
        expanded_nodes=expanded_nodes,
        cache_hits=cache_hits,
        peak_cache_entries=peak_cache_entries,
        group_elapsed_p50_ms=1_000.0 * _percentile(group_times, 0.50),
        group_elapsed_p95_ms=1_000.0 * _percentile(group_times, 0.95),
        group_elapsed_max_ms=1_000.0 * max(group_times),
    )


def run_method(
    method: str,
    kernels: tuple[KernelSpec, ...],
    presentations: tuple[PresentationSpec, ...],
    depth: int,
    workers: int,
) -> AggregateResult:
    """Run one schedule-matched method execution."""

    if method == "kernel_memo_p8":
        function = _kernel_memo_group
    elif method == "layerprobe_p8":
        function = _layerprobe_group
    else:
        raise ValueError(f"unsupported method: {method}")
    if workers < 1:
        raise ValueError("workers must be positive")
    tasks = tuple((kernel, presentations, depth) for kernel in kernels)
    if not tasks:
        raise ValueError("at least one kernel is required")

    if workers == 1 or len(tasks) == 1:
        return _aggregate(function(task) for task in tasks)

    chunksize = max(1, len(tasks) // (workers * 4))
    with ProcessPoolExecutor(max_workers=min(workers, len(tasks))) as executor:
        return _aggregate(executor.map(function, tasks, chunksize=chunksize))


def evenly_spaced_indices(population: int, sample_size: int) -> tuple[int, ...]:
    """Return deterministic endpoint-preserving indices over a finite grid.

    ``sample_size == 0`` selects the complete population.  Positive samples
    include both endpoints (except the one-element midpoint case), which makes
    the expensive-policy study span the full deterministic mechanism ordering
    without pseudorandom selection.
    """

    if population < 1:
        raise ValueError("population must be positive")
    if sample_size < 0 or sample_size > population:
        raise ValueError("sample_size must be zero or within the population")
    if sample_size == 0 or sample_size == population:
        return tuple(range(population))
    if sample_size == 1:
        return (population // 2,)
    indices = tuple(
        (position * (population - 1)) // (sample_size - 1)
        for position in range(sample_size)
    )
    if len(set(indices)) != sample_size:
        raise AssertionError("equidistant sampler produced duplicate indices")
    return indices


def select_kernels(
    population: int,
    sample_size: int,
) -> tuple[tuple[KernelSpec, ...], tuple[int, ...], str]:
    """Construct either the full grid or its deterministic equidistant sample."""

    population_kernels = make_kernels(population)
    indices = evenly_spaced_indices(population, sample_size)
    sampling = "full_grid" if len(indices) == population else "equidistant_grid_index"
    return tuple(population_kernels[index] for index in indices), indices, sampling


def _first_signature_difference(
    left: tuple[tuple[str, int], ...],
    right: tuple[tuple[str, int], ...],
) -> str:
    """Describe the first exact mapping mismatch for an actionable failure."""

    for index, (left_item, right_item) in enumerate(zip(left, right)):
        if left_item != right_item:
            return f"index {index}: {left_item!r} != {right_item!r}"
    if len(left) != len(right):
        return f"candidate counts differ: {len(left)} != {len(right)}"
    return "unknown mapping difference"


def _append_pair(path: Path, rows: tuple[RunRow, RunRow]) -> None:
    """Append a verified pair with one header and one filesystem flush."""

    exists = path.exists()
    with path.open("a", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0])))
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
        handle.flush()
        os.fsync(handle.fileno())


def _load_rows(path: Path) -> list[RunRow]:
    if not path.exists():
        return []
    rows: list[RunRow] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for payload in csv.DictReader(handle):
            row = RunRow(
                job_id=payload["job_id"],
                depth=int(payload["depth"]),
                repeat=int(payload["repeat"]),
                order_index=int(payload["order_index"]),
                method=payload["method"],
                workers=int(payload["workers"]),
                population_kernels=int(payload["population_kernels"]),
                sample_kernels=int(payload["sample_kernels"]),
                sampling=payload["sampling"],
                presentations=int(payload["presentations"]),
                chunksize=int(payload["chunksize"]),
                elapsed_s=float(payload["elapsed_s"]),
                candidate_digest=payload["candidate_digest"],
                trace_digest=payload["trace_digest"],
                valid_kernels=int(payload["valid_kernels"]),
                candidates=int(payload["candidates"]),
                semantic_requests=int(payload["semantic_requests"]),
                policy_invocations=int(payload["policy_invocations"]),
                expanded_nodes=int(payload["expanded_nodes"]),
                cache_hits=int(payload["cache_hits"]),
                cache_hit_rate=float(payload["cache_hit_rate"]),
                peak_cache_entries=int(payload["peak_cache_entries"]),
                group_elapsed_p50_ms=float(payload["group_elapsed_p50_ms"]),
                group_elapsed_p95_ms=float(payload["group_elapsed_p95_ms"]),
                group_elapsed_max_ms=float(payload["group_elapsed_max_ms"]),
                completed_at=payload["completed_at"],
            )
            if row.job_id in seen:
                raise RuntimeError(f"duplicate job id in raw CSV: {row.job_id}")
            seen.add(row.job_id)
            rows.append(row)
    return rows


def _bootstrap_median_ci(
    paired_ratios: list[float],
    seed: int,
    samples: int,
) -> tuple[float, float]:
    if not paired_ratios:
        raise ValueError("at least one paired ratio is required")
    if len(paired_ratios) == 1:
        return paired_ratios[0], paired_ratios[0]
    rng = random.Random(seed)
    medians = [
        statistics.median(rng.choices(paired_ratios, k=len(paired_ratios)))
        for _ in range(samples)
    ]
    return _percentile(medians, 0.025), _percentile(medians, 0.975)


def _median_field(rows: list[RunRow], field: str) -> float:
    return statistics.median(float(getattr(row, field)) for row in rows)


def summarize(
    rows: list[RunRow],
    depths: tuple[int, ...],
    repeats: int,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, object]:
    """Build a semantics-gated paired summary from admitted raw rows."""

    pairs: dict[tuple[int, int], dict[str, RunRow]] = {}
    for row in rows:
        pair = pairs.setdefault((row.depth, row.repeat), {})
        if row.method in pair:
            raise RuntimeError(
                f"duplicate method in depth/repeat pair: {row.depth}/{row.repeat}"
            )
        pair[row.method] = row

    semantic_checks: list[dict[str, object]] = []
    depth_summaries: list[dict[str, object]] = []
    all_complete_and_equal = True
    for depth in depths:
        ratios: list[float] = []
        method_rows: dict[str, list[RunRow]] = {method: [] for method in METHODS}
        for repeat in range(repeats):
            pair = pairs.get((depth, repeat), {})
            complete = set(METHODS) <= pair.keys()
            digest_equal = (
                complete
                and pair["kernel_memo_p8"].candidate_digest
                == pair["layerprobe_p8"].candidate_digest
            )
            trace_digest_equal = (
                complete
                and pair["kernel_memo_p8"].trace_digest
                == pair["layerprobe_p8"].trace_digest
            )
            semantic_checks.append(
                {
                    "depth": depth,
                    "repeat": repeat,
                    "complete": complete,
                    "candidate_digest_equal": digest_equal,
                    "complete_trace_digest_equal": trace_digest_equal,
                    "kernel_memo_trace_digest": (
                        pair["kernel_memo_p8"].trace_digest
                        if complete
                        else None
                    ),
                    "layerprobe_trace_digest": (
                        pair["layerprobe_p8"].trace_digest
                        if complete
                        else None
                    ),
                }
            )
            all_complete_and_equal &= bool(
                complete and digest_equal and trace_digest_equal
            )
            if not complete:
                continue
            if not digest_equal:
                raise AssertionError(
                    f"candidate digest mismatch at depth={depth}, repeat={repeat}"
                )
            if not trace_digest_equal:
                raise AssertionError(
                    f"complete trace digest mismatch at depth={depth}, "
                    f"repeat={repeat}"
                )
            for method in METHODS:
                method_rows[method].append(pair[method])
            ratios.append(
                pair["kernel_memo_p8"].elapsed_s
                / pair["layerprobe_p8"].elapsed_s
            )

        if not ratios:
            continue
        low, high = _bootstrap_median_ci(
            ratios,
            seed=bootstrap_seed + depth,
            samples=bootstrap_samples,
        )
        methods_payload: dict[str, object] = {}
        for method in METHODS:
            selected = method_rows[method]
            methods_payload[method] = {
                "elapsed_s_median": _median_field(selected, "elapsed_s"),
                "elapsed_s_min": min(row.elapsed_s for row in selected),
                "elapsed_s_max": max(row.elapsed_s for row in selected),
                "policy_invocations_median": _median_field(
                    selected, "policy_invocations"
                ),
                "expanded_nodes_median": _median_field(
                    selected, "expanded_nodes"
                ),
                "expanded_nodes_per_policy_invocation_median": (
                    statistics.median(
                        row.expanded_nodes / row.policy_invocations
                        for row in selected
                        if row.policy_invocations
                    )
                    if any(row.policy_invocations for row in selected)
                    else 0.0
                ),
                "group_elapsed_p50_ms_median": _median_field(
                    selected, "group_elapsed_p50_ms"
                ),
                "group_elapsed_p95_ms_median": _median_field(
                    selected, "group_elapsed_p95_ms"
                ),
                "group_elapsed_max_ms_median": _median_field(
                    selected, "group_elapsed_max_ms"
                ),
            }

        layerprobe_row = method_rows["layerprobe_p8"][0]
        kernel_row = method_rows["kernel_memo_p8"][0]
        depth_summaries.append(
            {
                "depth": depth,
                "paired_repeats": len(ratios),
                "paired_speedup_kernel_memo_over_layerprobe_median": (
                    statistics.median(ratios)
                ),
                "paired_speedup_ci95_low": low,
                "paired_speedup_ci95_high": high,
                "paired_speedup_min": min(ratios),
                "paired_speedup_max": max(ratios),
                "pairs_layerprobe_faster": sum(value > 1.0 for value in ratios),
                "semantic_requests": layerprobe_row.semantic_requests,
                "policy_invocations_saved": (
                    kernel_row.policy_invocations
                    - layerprobe_row.policy_invocations
                ),
                "policy_invocation_reduction": (
                    1.0
                    - layerprobe_row.policy_invocations
                    / kernel_row.policy_invocations
                    if kernel_row.policy_invocations
                    else 0.0
                ),
                "expanded_nodes_saved": (
                    kernel_row.expanded_nodes - layerprobe_row.expanded_nodes
                ),
                "expanded_node_reduction": (
                    1.0
                    - layerprobe_row.expanded_nodes / kernel_row.expanded_nodes
                    if kernel_row.expanded_nodes
                    else 0.0
                ),
                "cache_hits": layerprobe_row.cache_hits,
                "cache_hit_rate": layerprobe_row.cache_hit_rate,
                "peak_cache_entries_per_mechanism_agent": (
                    layerprobe_row.peak_cache_entries
                ),
                "methods": methods_payload,
            }
        )

    expected_pairs = len(depths) * repeats
    complete_pairs = sum(
        set(METHODS) <= pair.keys() for pair in pairs.values()
    )
    return {
        "status": (
            "complete_semantics_checked"
            if complete_pairs == expected_pairs and all_complete_and_equal
            else "partial_semantics_checked"
        ),
        "raw_run_count": len(rows),
        "expected_pairs": expected_pairs,
        "complete_pairs": complete_pairs,
        "ratio_definition": (
            "KernelMemo-P8 elapsed / LayerProbe-P8 elapsed; values above 1 "
            "favor LayerProbe"
        ),
        "depth_zero_definition": (
            "the original layerprobe.mechanics.choose_action policy; "
            "expanded_nodes is zero"
        ),
        "positive_depth_definition": (
            "exhaustive coast/brake belief-space lookahead with predicted "
            "full-braking stopping-location scoring"
        ),
        "semantic_checks": semantic_checks,
        "depths": depth_summaries,
    }


def _fingerprint() -> str:
    """Fingerprint every executed source file, excluding outputs and tests."""

    digest = hashlib.sha256()
    paths = sorted((ROOT / "src").rglob("*.py")) + [Path(__file__).resolve()]
    for path in paths:
        digest.update(str(path.relative_to(ROOT)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _indices_digest(indices: tuple[int, ...]) -> str:
    payload = json.dumps(indices, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _write_json_atomic(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _method_row(
    *,
    depth: int,
    repeat: int,
    order_index: int,
    method: str,
    workers: int,
    population_kernels: int,
    sample_kernels: int,
    sampling: str,
    presentations: int,
    chunksize: int,
    elapsed_s: float,
    result: AggregateResult,
) -> RunRow:
    return RunRow(
        job_id=f"depth-{depth}-repeat-{repeat}-{method}-w{workers}",
        depth=depth,
        repeat=repeat,
        order_index=order_index,
        method=method,
        workers=workers,
        population_kernels=population_kernels,
        sample_kernels=sample_kernels,
        sampling=sampling,
        presentations=presentations,
        chunksize=chunksize,
        elapsed_s=elapsed_s,
        candidate_digest=result.digest,
        trace_digest=result.trace_digest,
        valid_kernels=result.valid_kernels,
        candidates=result.candidates,
        semantic_requests=result.semantic_requests,
        policy_invocations=result.policy_invocations,
        expanded_nodes=result.expanded_nodes,
        cache_hits=result.cache_hits,
        cache_hit_rate=(
            result.cache_hits / result.semantic_requests
            if result.semantic_requests
            else 0.0
        ),
        peak_cache_entries=result.peak_cache_entries,
        group_elapsed_p50_ms=result.group_elapsed_p50_ms,
        group_elapsed_p95_ms=result.group_elapsed_p95_ms,
        group_elapsed_max_ms=result.group_elapsed_max_ms,
        completed_at=datetime.now().astimezone().isoformat(),
    )


def _parse_and_validate() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument(
        "--depths",
        type=int,
        nargs="+",
        default=(0, 2, 4, 6),
        help="depth zero is the original policy; positive values use lookahead",
    )
    parser.add_argument(
        "--population-kernels",
        type=int,
        default=MAX_KERNELS,
        help="size of the deterministic mechanism grid before sampling",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=512,
        help="equidistant sample size; zero selects the full population",
    )
    parser.add_argument("--presentations", type=int, default=MAX_PRESENTATIONS)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20_260_724)
    parser.add_argument(
        "--order-seed",
        type=int,
        default=20_260_724,
        help="deterministically permutes depth order within each repeat",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "relax n>=10 only for a tiny functional check; enforces at most "
            "20 selected kernels, 4 presentations, 2 workers, and 1 repeat"
        ),
    )
    args = parser.parse_args()

    logical_cpus = os.cpu_count() or 1
    if not 1 <= args.workers <= logical_cpus:
        raise ValueError("workers must be within the host logical CPU count")
    if not 1 <= args.population_kernels <= MAX_KERNELS:
        raise ValueError(
            f"population-kernels must be between 1 and {MAX_KERNELS}"
        )
    if not 0 <= args.sample_size <= args.population_kernels:
        raise ValueError("sample-size must be zero or within population-kernels")
    selected_count = args.population_kernels if args.sample_size == 0 else args.sample_size
    if not 1 <= args.presentations <= MAX_PRESENTATIONS:
        raise ValueError(
            f"presentations must be between 1 and {MAX_PRESENTATIONS}"
        )
    depths = tuple(sorted(set(args.depths)))
    if not depths or depths[0] < 0:
        raise ValueError("depths must contain non-negative integers")
    if max(depths) > 12:
        raise ValueError("depths above 12 are refused to prevent accidental runaway")
    if 0 not in depths and not args.smoke:
        raise ValueError("formal runs must include depth 0 as the original-policy baseline")
    if args.bootstrap_samples < 1_000:
        raise ValueError("bootstrap-samples must be at least 1000")

    if args.smoke:
        if (
            args.repeats != 1
            or selected_count > 20
            or args.presentations > 4
            or args.workers > 2
        ):
            raise ValueError(
                "smoke mode requires repeats=1, <=20 selected kernels, "
                "<=4 presentations, and <=2 workers"
            )
    elif args.repeats < 10:
        raise ValueError("formal paired timing requires repeats >= 10")

    args.depths = depths
    return args


def main() -> None:
    args = _parse_and_validate()
    output = args.output.resolve()
    if output.exists() and not args.resume:
        raise FileExistsError(
            f"{output} exists; choose a new directory or pass --resume"
        )
    if args.resume and not output.exists():
        raise FileNotFoundError(
            f"{output} does not exist; resume requires an existing run"
        )
    output.mkdir(parents=True, exist_ok=True)

    kernels, sample_indices, sampling = select_kernels(
        args.population_kernels,
        args.sample_size,
    )
    presentations = make_presentations(args.presentations)
    chunksize = max(1, len(kernels) // (args.workers * 4))
    code_fingerprint = _fingerprint()
    config = {
        "workers": args.workers,
        "repeats": args.repeats,
        "depths": list(args.depths),
        "population_kernels": args.population_kernels,
        "sample_size_argument": args.sample_size,
        "selected_kernels": len(kernels),
        "sampling": sampling,
        "presentations": args.presentations,
        "chunksize": chunksize,
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
        "order_seed": args.order_seed,
        "smoke": args.smoke,
    }

    metadata_path = output / "metadata.json"
    runs_path = output / "runs.csv"
    summary_path = output / "summary.json"
    if args.resume:
        if not metadata_path.exists():
            raise FileNotFoundError("resume refused: metadata.json is missing")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("code_fingerprint_sha256") != code_fingerprint:
            raise RuntimeError("code fingerprint changed; resume refused")
        if metadata.get("config") != config:
            raise RuntimeError("run configuration changed; resume refused")
        if metadata.get("sample_indices") != list(sample_indices):
            raise RuntimeError("sample indices changed; resume refused")
    else:
        metadata = {
            "schema_version": 1,
            "benchmark": "deliberative_policy_exact_reuse",
            "started_at": datetime.now().astimezone().isoformat(),
            "command": [
                Path(sys.executable).name,
                Path(__file__).resolve().relative_to(ROOT).as_posix(),
                *sys.argv[1:],
            ],
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logical_cpus": os.cpu_count(),
            "python": sys.version,
            "python_executable": Path(sys.executable).name,
            "code_fingerprint_sha256": code_fingerprint,
            "config": config,
            "sample_indices": list(sample_indices),
            "sample_indices_sha256": _indices_digest(sample_indices),
            "planner": {
                "depth_0": (
                    "original layerprobe.mechanics.choose_action; zero search nodes"
                ),
                "depth_positive": (
                    "exhaustively enumerate coast/brake sequences in the "
                    "agent belief model; stop a branch when believed speed is "
                    "zero or the requested depth is reached"
                ),
                "leaf_score": (
                    "predicted full-braking stopping error relative to the "
                    "goal interval, then interval-center error, deviations "
                    "from the original heuristic, brake count, and "
                    "coast-before-brake lexical order"
                ),
                "expanded_nodes": (
                    "actual predicted belief transitions in search-tree edges "
                    "plus leaf full-braking projections"
                ),
            },
            "reuse_contract": {
                "scope": "one fixed mechanism-agent pair",
                "key": "(WorldState, AgentMemory, Observation)",
                "shared": "deterministic policy and mechanism transition result",
                "presentation_local": ["DisplayMemory", "trace construction"],
            },
            "timing_protocol": {
                "pairing": "same sampled mechanisms, presentations, and workers",
                "schedule": (
                    "same ordered mechanism-group tasks and ProcessPoolExecutor "
                    "chunksize for both methods"
                ),
                "order": (
                    "depth order deterministically permuted per repeat; method "
                    "order alternates by repeat, balancing every configured "
                    "depth across even and odd repeats"
                ),
                "admission_gate": (
                    "exact equality of the complete candidate-signature mapping "
                    "and every candidate's stable four-agent full-trace digest "
                    "before both timing rows are appended"
                ),
                "ratio": "KernelMemo-P8 elapsed / LayerProbe-P8 elapsed",
            },
        }
        _write_json_atomic(metadata_path, metadata)

    rows = _load_rows(runs_path)
    pair_methods: dict[tuple[int, int], set[str]] = {}
    for row in rows:
        pair_methods.setdefault((row.depth, row.repeat), set()).add(row.method)
    for pair_key, methods in pair_methods.items():
        if methods != set(METHODS):
            raise RuntimeError(
                f"resume refused: incomplete raw pair {pair_key} has {sorted(methods)}"
            )

    for repeat in range(args.repeats):
        depth_order = list(args.depths)
        random.Random(args.order_seed + repeat).shuffle(depth_order)
        for depth in depth_order:
            pair_key = (depth, repeat)
            if pair_key in pair_methods:
                continue
            order = (
                METHODS
                if repeat % 2 == 0
                else tuple(reversed(METHODS))
            )
            measured: dict[str, tuple[AggregateResult, float, int]] = {}
            for order_index, method in enumerate(order):
                gc.collect()
                started = time.perf_counter()
                result = run_method(
                    method,
                    kernels,
                    presentations,
                    depth,
                    args.workers,
                )
                elapsed_s = time.perf_counter() - started
                measured[method] = (result, elapsed_s, order_index)
                print(
                    f"depth={depth} repeat={repeat} method={method} "
                    f"elapsed={elapsed_s:.3f}s digest={result.digest[:12]} "
                    f"invocations={result.policy_invocations} "
                    f"nodes={result.expanded_nodes}",
                    flush=True,
                )

            kernel_result = measured["kernel_memo_p8"][0]
            layerprobe_result = measured["layerprobe_p8"][0]
            if (
                kernel_result.candidate_signatures
                != layerprobe_result.candidate_signatures
            ):
                difference = _first_signature_difference(
                    kernel_result.candidate_signatures,
                    layerprobe_result.candidate_signatures,
                )
                raise AssertionError(
                    f"semantic mismatch at depth={depth}, repeat={repeat}: "
                    f"{difference}"
                )
            if (
                kernel_result.candidate_trace_digests
                != layerprobe_result.candidate_trace_digests
            ):
                for index, (left_item, right_item) in enumerate(
                    zip(
                        kernel_result.candidate_trace_digests,
                        layerprobe_result.candidate_trace_digests,
                    )
                ):
                    if left_item != right_item:
                        trace_difference = (
                            f"index {index}: {left_item!r} != {right_item!r}"
                        )
                        break
                else:
                    trace_difference = (
                        "candidate trace-digest counts differ: "
                        f"{len(kernel_result.candidate_trace_digests)} != "
                        f"{len(layerprobe_result.candidate_trace_digests)}"
                    )
                raise AssertionError(
                    f"complete trace mismatch at depth={depth}, "
                    f"repeat={repeat}: {trace_difference}"
                )
            if kernel_result.digest != layerprobe_result.digest:
                raise AssertionError(
                    "candidate mappings are equal but their digests differ"
                )
            if kernel_result.trace_digest != layerprobe_result.trace_digest:
                raise AssertionError(
                    "candidate trace ledgers are equal but their aggregate "
                    "digests differ"
                )

            admitted_rows: list[RunRow] = []
            for method in order:
                result, elapsed_s, order_index = measured[method]
                admitted_rows.append(
                    _method_row(
                        depth=depth,
                        repeat=repeat,
                        order_index=order_index,
                        method=method,
                        workers=args.workers,
                        population_kernels=args.population_kernels,
                        sample_kernels=len(kernels),
                        sampling=sampling,
                        presentations=args.presentations,
                        chunksize=chunksize,
                        elapsed_s=elapsed_s,
                        result=result,
                    )
                )
            _append_pair(runs_path, (admitted_rows[0], admitted_rows[1]))
            rows.extend(admitted_rows)
            pair_methods[pair_key] = set(METHODS)

            summary = summarize(
                rows,
                args.depths,
                args.repeats,
                args.bootstrap_samples,
                args.bootstrap_seed,
            )
            _write_json_atomic(summary_path, summary)

    summary = summarize(
        rows,
        args.depths,
        args.repeats,
        args.bootstrap_samples,
        args.bootstrap_seed,
    )
    metadata["finished_at"] = datetime.now().astimezone().isoformat()
    metadata["status"] = summary["status"]
    _write_json_atomic(metadata_path, metadata)
    _write_json_atomic(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
