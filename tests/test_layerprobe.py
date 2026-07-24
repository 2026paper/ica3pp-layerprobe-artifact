from __future__ import annotations

from layerprobe.evaluator import (
    minimum_cover,
    reduce_signature_frontier,
    run_factorized,
    run_flat,
    run_flat_parallel,
    run_kernel_memo,
    run_kernel_memo_parallel,
)
from layerprobe.mechanics import ingest, initial_state, observe, transition
from layerprobe.model import (
    AgentMemory,
    DisplayMemory,
    PresentationSpec,
    WorldState,
)
from layerprobe.workloads import make_kernels, make_presentations


def test_factorized_matches_flat_on_small_finite_domain() -> None:
    kernels = make_kernels(12)
    presentations = make_presentations(8)
    flat = run_flat(kernels, presentations)
    memo = run_kernel_memo(kernels, presentations)
    factorized = run_factorized(kernels, presentations, workers=1)
    assert factorized.comparable() == flat.comparable()
    assert memo.comparable() == flat.comparable()
    assert factorized.minimum_suite is not None
    assert factorized.metrics["graph_builds"] == len(kernels)
    assert flat.metrics["graph_builds"] == len(kernels) * len(presentations)
    assert factorized.metrics["policy_calls"] < flat.metrics["policy_calls"]
    assert memo.metrics["graph_builds"] == factorized.metrics["graph_builds"]
    assert memo.metrics["policy_calls"] == flat.metrics["policy_calls"]


def test_parallel_schedule_preserves_semantics() -> None:
    kernels = make_kernels(16)
    presentations = make_presentations(6)
    flat = run_flat(kernels, presentations)
    flat_grouped = run_flat_parallel(kernels, presentations, workers=1)
    flat_parallel = run_flat_parallel(kernels, presentations, workers=8)
    sequential = run_factorized(kernels, presentations, workers=1)
    parallel = run_factorized(kernels, presentations, workers=2)
    memo_sequential = run_kernel_memo(kernels, presentations)
    memo_parallel = run_kernel_memo_parallel(kernels, presentations, workers=2)
    assert flat_grouped.comparable() == flat.comparable()
    assert flat_parallel.comparable() == flat.comparable()
    assert flat_grouped.metrics == flat.metrics
    assert flat_parallel.metrics == flat.metrics
    assert list(flat_parallel.candidate_signatures) == sorted(
        flat_parallel.candidate_signatures
    )
    assert flat_parallel.valid_kernels == tuple(sorted(flat_parallel.valid_kernels))
    assert parallel.comparable() == sequential.comparable()
    assert memo_parallel.comparable() == memo_sequential.comparable()
    assert memo_parallel.metrics == memo_sequential.metrics


def test_presentation_is_read_only_for_mechanics() -> None:
    kernel = make_kernels(1)[0]
    presentations = make_presentations(4)
    state = initial_state(kernel)
    expected_next = transition(state, "brake", kernel)
    for presentation in presentations:
        before = state
        observe(state, kernel, presentation, DisplayMemory())
        assert state == before
        assert transition(state, "brake", kernel) == expected_next


def test_signature_frontier_preserves_exact_minimum_cover() -> None:
    signatures = {
        "a": 0b000001,
        "b": 0b000011,
        "c": 0b001100,
        "d": 0b111100,
        "e": 0b000000,
    }
    frontier = reduce_signature_frontier(signatures)
    assert frontier == {"b": 0b000011, "d": 0b111100}
    assert minimum_cover(frontier, 0b111111) == ("b", "d")


def test_delayed_and_immediate_presentations_can_diverge() -> None:
    kernel = make_kernels(12)[-1]
    presentations = make_presentations(2)
    result = run_factorized((kernel,), presentations, workers=1)
    candidate_masks = list(result.candidate_signatures.values())
    assert len(candidate_masks) in {0, 2}


def test_visible_distance_is_nonnegative_inside_goal_region() -> None:
    kernel = make_kernels(1)[0]
    state = WorldState(
        position=kernel.goal_start + 1,
        speed=2,
        step=1,
        used_brake=False,
    )
    presentation = PresentationSpec(
        name="exact_inside_goal",
        speed_mode="exact",
        distance_mode="exact",
        delay=0,
    )
    observation, _ = observe(
        state,
        kernel,
        presentation,
        DisplayMemory(),
    )
    assert observation[1] == 0
    assert observation[2] == 1

    perceived = ingest(
        AgentMemory(
            believed_speed=3,
            believed_distance=2,
        ),
        observation,
    )
    assert perceived.believed_distance == 0
