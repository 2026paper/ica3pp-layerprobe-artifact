"""Focused tests for the reviewer-motivated deliberative-policy benchmark."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

from deliberative_policy_benchmark import (
    _complete_trace_digest,
    deliberative_action,
    evenly_spaced_indices,
    run_method,
)
from layerprobe.evaluator import run_kernel_memo
from layerprobe.mechanics import (
    AGENT_NAMES,
    choose_action,
    initial_agent_memory,
)
from layerprobe.model import KernelSpec, PresentationSpec


def _valid_kernel() -> KernelSpec:
    return KernelSpec(
        name="deliberative_test",
        start_speed=3,
        friction=0,
        brake_force=1,
        goal_start=4,
        goal_end=5,
        horizon=8,
    )


def _presentations() -> tuple[PresentationSpec, ...]:
    # The first two presentations intentionally have different identities but
    # identical view semantics, guaranteeing at least one exact reuse path.
    return (
        PresentationSpec("exact_a", "exact", "exact", 0),
        PresentationSpec("exact_b", "exact", "exact", 0),
        PresentationSpec("hidden_delay", "hidden", "coarse", 1),
    )


def test_depth_zero_is_exactly_the_original_policy() -> None:
    kernel = _valid_kernel()
    memory = initial_agent_memory(kernel)
    for agent in AGENT_NAMES:
        decision = deliberative_action(agent, memory, kernel, depth=0)
        assert decision.action == choose_action(agent, memory, kernel)
        assert decision.expanded_nodes == 0


def test_positive_depth_search_is_deterministic_and_expands_nodes() -> None:
    kernel = _valid_kernel()
    memory = initial_agent_memory(kernel)
    first = deliberative_action("reference", memory, kernel, depth=4)
    second = deliberative_action("reference", memory, kernel, depth=4)
    assert first == second
    assert first.action in {"coast", "brake"}
    assert first.expanded_nodes > 0


def test_equidistant_sample_is_stable_unique_and_endpoint_preserving() -> None:
    assert evenly_spaced_indices(10, 0) == tuple(range(10))
    assert evenly_spaced_indices(10, 1) == (5,)
    sample = evenly_spaced_indices(10, 4)
    assert sample == (0, 3, 6, 9)
    assert len(sample) == len(set(sample))


def test_complete_key_reuse_preserves_signatures_and_complete_traces() -> None:
    kernels = (_valid_kernel(),)
    presentations = _presentations()
    baseline = run_method(
        "kernel_memo_p8",
        kernels,
        presentations,
        depth=2,
        workers=1,
    )
    reused = run_method(
        "layerprobe_p8",
        kernels,
        presentations,
        depth=2,
        workers=1,
    )

    assert baseline.candidate_signatures == reused.candidate_signatures
    assert baseline.digest == reused.digest
    assert baseline.candidate_trace_digests == reused.candidate_trace_digests
    assert baseline.trace_digest == reused.trace_digest
    assert baseline.semantic_requests == reused.semantic_requests
    assert baseline.policy_invocations == baseline.semantic_requests
    assert reused.semantic_requests == reused.policy_invocations + reused.cache_hits
    assert reused.cache_hits > 0
    assert reused.policy_invocations < baseline.policy_invocations
    assert reused.expanded_nodes < baseline.expanded_nodes


def test_complete_trace_digest_covers_every_agent_and_step() -> None:
    traces = {
        "agent_a": (((1, 2, 0, 0), "coast", "running"),),
        "agent_b": (((1, 2, 0, 0), "brake", "win"),),
    }
    same_different_order = {
        "agent_b": traces["agent_b"],
        "agent_a": traces["agent_a"],
    }
    changed = {
        **traces,
        "agent_b": (((1, 2, 0, 0), "brake", "stopped"),),
    }
    assert _complete_trace_digest(traces) == _complete_trace_digest(
        same_different_order
    )
    assert _complete_trace_digest(traces) != _complete_trace_digest(changed)


def test_depth_zero_candidate_mapping_matches_the_existing_evaluator() -> None:
    kernels = (_valid_kernel(),)
    presentations = _presentations()
    existing = run_kernel_memo(kernels, presentations)
    benchmark = run_method(
        "kernel_memo_p8",
        kernels,
        presentations,
        depth=0,
        workers=1,
    )
    assert dict(benchmark.candidate_signatures) == existing.candidate_signatures
