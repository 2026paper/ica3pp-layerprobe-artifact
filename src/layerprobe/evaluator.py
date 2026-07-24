"""Flat and factorized evaluators for the LayerProbe vertical slice."""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from itertools import combinations
from typing import Iterable

from .mechanics import (
    AGENT_NAMES,
    advance_belief,
    choose_action,
    ingest,
    initial_agent_memory,
    initial_state,
    observe,
    simulate_flat,
    terminal_status,
    transition,
    verify_kernel,
)
from .model import (
    AgentMemory,
    DisplayMemory,
    KernelSpec,
    Observation,
    PresentationSpec,
    Trace,
    WorkMetrics,
    WorldState,
)

MODEL_PAIRS: tuple[tuple[str, str], ...] = tuple(combinations(AGENT_NAMES, 2))


@dataclass(frozen=True, slots=True)
class RunResult:
    candidate_signatures: dict[str, int]
    frontier: dict[str, int]
    minimum_suite: tuple[str, ...] | None
    valid_kernels: tuple[str, ...]
    metrics: dict[str, int]
    model_pairs: tuple[tuple[str, str], ...] = MODEL_PAIRS

    def comparable(self) -> tuple[dict[str, int], tuple[str, ...] | None, tuple[str, ...]]:
        return self.candidate_signatures, self.minimum_suite, self.valid_kernels


@dataclass(slots=True)
class _Member:
    presentation: PresentationSpec
    display_memory: DisplayMemory
    trace: list[tuple[Observation, str, str]]


@dataclass(slots=True)
class _ExecutionGroup:
    state: WorldState
    agent_memory: AgentMemory
    members: list[_Member]


def signature_for(traces: dict[str, Trace]) -> int:
    mask = 0
    for index, (left, right) in enumerate(MODEL_PAIRS):
        if traces[left] != traces[right]:
            mask |= 1 << index
    return mask


def reduce_signature_frontier(candidate_signatures: dict[str, int]) -> dict[str, int]:
    """Keep one deterministic representative for each maximal non-zero mask."""

    representative: dict[int, str] = {}
    for candidate, mask in sorted(candidate_signatures.items()):
        if mask == 0:
            continue
        representative.setdefault(mask, candidate)

    maximal_masks: list[int] = []
    for mask in sorted(representative, key=lambda item: (-item.bit_count(), -item)):
        if any(mask | kept == kept for kept in maximal_masks):
            continue
        maximal_masks = [kept for kept in maximal_masks if kept | mask != mask]
        maximal_masks.append(mask)

    return {
        representative[mask]: mask
        for mask in sorted(maximal_masks, key=lambda item: representative[item])
    }


def minimum_cover(frontier: dict[str, int], target_mask: int) -> tuple[str, ...] | None:
    """Exact cardinality minimum cover over a small bit-mask universe."""

    dp: dict[int, tuple[str, ...]] = {0: ()}
    for candidate, signature in sorted(frontier.items()):
        snapshot = list(dp.items())
        for covered, suite in snapshot:
            combined = covered | signature
            proposal = suite + (candidate,)
            incumbent = dp.get(combined)
            if incumbent is None or len(proposal) < len(incumbent) or (
                len(proposal) == len(incumbent) and proposal < incumbent
            ):
                dp[combined] = proposal
    return dp.get(target_mask)


def _finish(
    signatures: dict[str, int],
    valid_kernels: Iterable[str],
    metrics: WorkMetrics,
) -> RunResult:
    signatures = dict(sorted(signatures.items()))
    frontier = reduce_signature_frontier(signatures)
    target_mask = (1 << len(MODEL_PAIRS)) - 1
    suite = minimum_cover(frontier, target_mask)
    return RunResult(
        candidate_signatures=signatures,
        frontier=frontier,
        minimum_suite=suite,
        valid_kernels=tuple(sorted(valid_kernels)),
        metrics=metrics.as_dict(),
    )


def run_flat(
    kernels: Iterable[KernelSpec],
    presentations: Iterable[PresentationSpec],
) -> RunResult:
    """Product-based baseline that repeats mechanism verification per variant."""

    kernel_list = tuple(kernels)
    presentation_list = tuple(presentations)
    metrics = WorkMetrics()
    signatures: dict[str, int] = {}
    valid_kernels: set[str] = set()

    for kernel in kernel_list:
        for presentation in presentation_list:
            report = verify_kernel(kernel)
            metrics.graph_builds += 1
            metrics.graph_states += report.states
            metrics.graph_transitions += report.transitions
            if not report.valid:
                continue
            valid_kernels.add(kernel.name)
            traces: dict[str, Trace] = {}
            for agent in AGENT_NAMES:
                trace = simulate_flat(kernel, presentation, agent)
                traces[agent] = trace
                metrics.observation_calls += len(trace)
                metrics.policy_calls += len(trace)
                metrics.transition_calls += len(trace)
            candidate = f"{kernel.name}::{presentation.name}"
            signatures[candidate] = signature_for(traces)
            metrics.candidates += 1
    return _finish(signatures, valid_kernels, metrics)


def _flat_kernel_group(
    args: tuple[KernelSpec, tuple[PresentationSpec, ...]],
) -> tuple[str, bool, dict[str, int], WorkMetrics]:
    """Evaluate one mechanism group while preserving the flat baseline work."""

    kernel, presentations = args
    signatures: dict[str, int] = {}
    metrics = WorkMetrics()
    valid = False
    for presentation in presentations:
        report = verify_kernel(kernel)
        metrics.graph_builds += 1
        metrics.graph_states += report.states
        metrics.graph_transitions += report.transitions
        if not report.valid:
            continue
        valid = True
        traces: dict[str, Trace] = {}
        for agent in AGENT_NAMES:
            trace = simulate_flat(kernel, presentation, agent)
            traces[agent] = trace
            metrics.observation_calls += len(trace)
            metrics.policy_calls += len(trace)
            metrics.transition_calls += len(trace)
        candidate = f"{kernel.name}::{presentation.name}"
        signatures[candidate] = signature_for(traces)
        metrics.candidates += 1
    return kernel.name, valid, signatures, metrics


def run_flat_parallel(
    kernels: Iterable[KernelSpec],
    presentations: Iterable[PresentationSpec],
    workers: int = 1,
) -> RunResult:
    """Flat baseline with the same per-mechanism process schedule as LayerProbe."""

    kernel_list = tuple(kernels)
    presentation_list = tuple(presentations)
    if workers < 1:
        raise ValueError("workers must be at least one")
    tasks = tuple((kernel, presentation_list) for kernel in kernel_list)
    if workers == 1 or len(tasks) <= 1:
        group_results = [_flat_kernel_group(task) for task in tasks]
    else:
        chunksize = max(1, len(tasks) // (workers * 4))
        with ProcessPoolExecutor(max_workers=min(workers, len(tasks))) as executor:
            group_results = list(
                executor.map(_flat_kernel_group, tasks, chunksize=chunksize)
            )

    signatures: dict[str, int] = {}
    valid_kernels: list[str] = []
    metrics = WorkMetrics()
    for kernel_name, valid, group_signatures, group_metrics in sorted(group_results):
        metrics.add(group_metrics)
        if valid:
            valid_kernels.append(kernel_name)
            signatures.update(group_signatures)
    return _finish(signatures, valid_kernels, metrics)


def run_kernel_memo(
    kernels: Iterable[KernelSpec],
    presentations: Iterable[PresentationSpec],
) -> RunResult:
    """Verify each mechanism once, but execute every presentation independently.

    This baseline isolates the value of presentation-prefix sharing from the
    simpler optimization of memoizing the mechanism-level validity check.
    """

    kernel_list = tuple(kernels)
    presentation_list = tuple(presentations)
    metrics = WorkMetrics()
    signatures: dict[str, int] = {}
    valid_kernels: list[str] = []

    for kernel in kernel_list:
        report = verify_kernel(kernel)
        metrics.graph_builds += 1
        metrics.graph_states += report.states
        metrics.graph_transitions += report.transitions
        if not report.valid:
            continue
        valid_kernels.append(kernel.name)
        for presentation in presentation_list:
            traces: dict[str, Trace] = {}
            for agent in AGENT_NAMES:
                trace = simulate_flat(kernel, presentation, agent)
                traces[agent] = trace
                metrics.observation_calls += len(trace)
                metrics.policy_calls += len(trace)
                metrics.transition_calls += len(trace)
            candidate = f"{kernel.name}::{presentation.name}"
            signatures[candidate] = signature_for(traces)
            metrics.candidates += 1
    return _finish(signatures, valid_kernels, metrics)


def _kernel_memo_kernel_group(
    args: tuple[KernelSpec, tuple[PresentationSpec, ...]],
) -> tuple[str, bool, dict[str, int], WorkMetrics]:
    """Evaluate one mechanism group without cross-presentation step reuse."""

    kernel, presentations = args
    report = verify_kernel(kernel)
    metrics = WorkMetrics(
        graph_builds=1,
        graph_states=report.states,
        graph_transitions=report.transitions,
    )
    if not report.valid:
        return kernel.name, False, {}, metrics

    signatures: dict[str, int] = {}
    for presentation in presentations:
        traces: dict[str, Trace] = {}
        for agent in AGENT_NAMES:
            trace = simulate_flat(kernel, presentation, agent)
            traces[agent] = trace
            metrics.observation_calls += len(trace)
            metrics.policy_calls += len(trace)
            metrics.transition_calls += len(trace)
        candidate = f"{kernel.name}::{presentation.name}"
        signatures[candidate] = signature_for(traces)
        metrics.candidates += 1
    return kernel.name, True, signatures, metrics


def run_kernel_memo_parallel(
    kernels: Iterable[KernelSpec],
    presentations: Iterable[PresentationSpec],
    workers: int = 1,
) -> RunResult:
    """Kernel-memo baseline with the same process-pool schedule as LayerProbe.

    This comparator isolates semantic-step reuse from the benefit of assigning
    independent mechanism groups to multiple processes.
    """

    kernel_list = tuple(kernels)
    presentation_list = tuple(presentations)
    if workers < 1:
        raise ValueError("workers must be at least one")
    tasks = tuple((kernel, presentation_list) for kernel in kernel_list)
    if workers == 1 or len(tasks) <= 1:
        group_results = [_kernel_memo_kernel_group(task) for task in tasks]
    else:
        chunksize = max(1, len(tasks) // (workers * 4))
        with ProcessPoolExecutor(max_workers=min(workers, len(tasks))) as executor:
            group_results = list(
                executor.map(_kernel_memo_kernel_group, tasks, chunksize=chunksize)
            )

    signatures: dict[str, int] = {}
    valid_kernels: list[str] = []
    metrics = WorkMetrics()
    for kernel_name, valid, group_signatures, group_metrics in sorted(group_results):
        metrics.add(group_metrics)
        if valid:
            valid_kernels.append(kernel_name)
            signatures.update(group_signatures)
    return _finish(signatures, valid_kernels, metrics)


def _batched_agent_traces(
    kernel: KernelSpec,
    presentations: tuple[PresentationSpec, ...],
    agent: str,
) -> tuple[dict[str, Trace], WorkMetrics]:
    members = [
        _Member(
            presentation=presentation,
            display_memory=DisplayMemory(),
            trace=[],
        )
        for presentation in presentations
    ]
    active_groups = [
        _ExecutionGroup(
            state=initial_state(kernel),
            agent_memory=initial_agent_memory(kernel),
            members=members,
        )
    ]
    metrics = WorkMetrics()

    while active_groups:
        observation_groups: dict[
            tuple[WorldState, AgentMemory, Observation],
            list[tuple[_Member, DisplayMemory, Observation]],
        ] = defaultdict(list)

        for execution_group in active_groups:
            if terminal_status(execution_group.state, kernel) != "running":
                continue
            for member in execution_group.members:
                observation, next_display = observe(
                    execution_group.state,
                    kernel,
                    member.presentation,
                    member.display_memory,
                )
                metrics.observation_calls += 1
                observation_groups[
                    (execution_group.state, execution_group.agent_memory, observation)
                ].append((member, next_display, observation))

        metrics.prefix_groups += len(observation_groups)
        next_execution_groups: list[_ExecutionGroup] = []
        for (state, memory, observation), group in observation_groups.items():
            perceived = ingest(memory, observation)
            action = choose_action(agent, perceived, kernel)
            next_state = transition(state, action, kernel)
            next_memory = advance_belief(agent, perceived, action, kernel)
            status = terminal_status(next_state, kernel)
            metrics.policy_calls += 1
            metrics.transition_calls += 1
            next_members: list[_Member] = []
            for member, next_display, record_observation in group:
                member.display_memory = next_display
                member.trace.append((record_observation, action, status))
                next_members.append(member)
            if status == "running":
                next_execution_groups.append(
                    _ExecutionGroup(
                        state=next_state,
                        agent_memory=next_memory,
                        members=next_members,
                    )
                )
        active_groups = next_execution_groups

    return {member.presentation.name: tuple(member.trace) for member in members}, metrics


def _memoized_agent_traces(
    kernel: KernelSpec,
    presentations: tuple[PresentationSpec, ...],
    agent: str,
) -> tuple[dict[str, Trace], WorkMetrics]:
    """Execute presentations independently while reusing identical semantic steps.

    The cache key contains the complete immutable world state, agent memory, and
    observation. A hit therefore replaces only a deterministic policy/transition
    prefix computation; presentation state remains local to each trace.
    """

    cache: dict[
        tuple[WorldState, AgentMemory, Observation],
        tuple[str, WorldState, AgentMemory, str],
    ] = {}
    traces: dict[str, Trace] = {}
    metrics = WorkMetrics()
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
            metrics.observation_calls += 1
            key = (state, agent_memory, observation)
            cached = cache.get(key)
            if cached is None:
                perceived = ingest(agent_memory, observation)
                action = choose_action(agent, perceived, kernel)
                next_state = transition(state, action, kernel)
                next_memory = advance_belief(agent, perceived, action, kernel)
                status = terminal_status(next_state, kernel)
                cached = (action, next_state, next_memory, status)
                cache[key] = cached
                metrics.policy_calls += 1
                metrics.transition_calls += 1
                metrics.prefix_groups += 1
            action, state, agent_memory, status = cached
            trace.append((observation, action, status))
        traces[presentation.name] = tuple(trace)
    return traces, metrics


def _factorized_kernel_group(
    args: tuple[KernelSpec, tuple[PresentationSpec, ...]],
) -> tuple[str, bool, dict[str, int], WorkMetrics]:
    kernel, presentations = args
    report = verify_kernel(kernel)
    metrics = WorkMetrics(
        graph_builds=1,
        graph_states=report.states,
        graph_transitions=report.transitions,
    )
    if not report.valid:
        return kernel.name, False, {}, metrics

    traces_by_presentation: dict[str, dict[str, Trace]] = {
        presentation.name: {} for presentation in presentations
    }
    for agent in AGENT_NAMES:
        agent_traces, agent_metrics = _memoized_agent_traces(kernel, presentations, agent)
        metrics.add(agent_metrics)
        for presentation_name, trace in agent_traces.items():
            traces_by_presentation[presentation_name][agent] = trace

    signatures: dict[str, int] = {}
    for presentation in presentations:
        candidate = f"{kernel.name}::{presentation.name}"
        signatures[candidate] = signature_for(traces_by_presentation[presentation.name])
        metrics.candidates += 1
    return kernel.name, True, signatures, metrics


def run_factorized(
    kernels: Iterable[KernelSpec],
    presentations: Iterable[PresentationSpec],
    workers: int = 1,
) -> RunResult:
    """Factorized implementation with one task per mechanism group."""

    kernel_list = tuple(kernels)
    presentation_list = tuple(presentations)
    if workers < 1:
        raise ValueError("workers must be at least one")
    tasks = tuple((kernel, presentation_list) for kernel in kernel_list)
    if workers == 1 or len(tasks) <= 1:
        group_results = [_factorized_kernel_group(task) for task in tasks]
    else:
        chunksize = max(1, len(tasks) // (workers * 4))
        with ProcessPoolExecutor(max_workers=min(workers, len(tasks))) as executor:
            group_results = list(executor.map(_factorized_kernel_group, tasks, chunksize=chunksize))

    signatures: dict[str, int] = {}
    valid_kernels: list[str] = []
    metrics = WorkMetrics()
    for kernel_name, valid, group_signatures, group_metrics in sorted(group_results):
        metrics.add(group_metrics)
        if valid:
            valid_kernels.append(kernel_name)
            signatures.update(group_signatures)
    return _finish(signatures, valid_kernels, metrics)
