from __future__ import annotations

import pytest

from experiments.scheduler_sensitivity import (
    RunRow,
    SCHEDULES,
    _contiguous_batches,
    _production_group,
    _schedule_order,
    _validate_loaded_rows,
    run_schedule,
)
from layerprobe.evaluator import _factorized_kernel_group, run_factorized
from layerprobe.workloads import make_kernels, make_presentations


def test_static_contiguous_batches_preserve_order_and_balance() -> None:
    kernels = make_kernels(11)
    presentations = make_presentations(2)
    tasks = tuple((kernel, presentations) for kernel in kernels)
    batches = _contiguous_batches(tasks, 3)
    assert [len(batch) for batch in batches] == [4, 4, 3]
    flattened = tuple(task for batch in batches for task in batch)
    assert flattened == tasks


def test_order_cycle_contains_all_six_schedule_permutations() -> None:
    orders = {_schedule_order(repeat) for repeat in range(6)}
    assert len(orders) == 6
    assert all(set(order) == set(SCHEDULES) for order in orders)


def test_all_schedules_preserve_candidate_signature_digest() -> None:
    kernels = make_kernels(9)
    presentations = make_presentations(4)
    results = {
        schedule: run_schedule(
            schedule,
            kernels,
            presentations,
            workers=2,
        )
        for schedule in SCHEDULES
    }
    assert len({result.aggregate.digest for result in results.values()}) == 1
    assert all(
        len(result.worker_loads_s) == 2 for result in results.values()
    )
    assert results["static_contiguous"].observed_workers == 2
    production = run_factorized(kernels, presentations, workers=1)
    assert results["current_chunksize"].aggregate.valid_kernels == len(
        production.valid_kernels
    )
    assert results["current_chunksize"].aggregate.candidates == len(
        production.candidate_signatures
    )
    assert results["current_chunksize"].aggregate.semantic_requests == (
        production.metrics["observation_calls"]
    )
    assert results["current_chunksize"].aggregate.computed_steps == (
        production.metrics["policy_calls"]
    )


def test_group_wrapper_preserves_complete_production_output() -> None:
    kernel = make_kernels(1)[0]
    presentations = make_presentations(3)
    production = _factorized_kernel_group((kernel, presentations))
    wrapped = _production_group((kernel, presentations))
    assert wrapped.kernel_name == production[0]
    assert wrapped.valid == production[1]
    assert dict(wrapped.candidate_signatures) == production[2]
    assert wrapped.semantic_requests == production[3].observation_calls
    assert wrapped.computed_steps == production[3].policy_calls


def _row(repeat: int, schedule: str, order_index: int) -> RunRow:
    return RunRow(
        job_id=f"scheduler-r{repeat}-{schedule}-w2",
        repeat=repeat,
        order_index=order_index,
        schedule=schedule,
        workers=2,
        kernels=9,
        presentations=4,
        current_chunksize=1,
        elapsed_s=1.0,
        digest="a" * 64,
        valid_kernels=9,
        candidates=36,
        semantic_requests=100,
        computed_steps=68,
        cache_hits=32,
        observed_workers=2,
        worker_loads_json="[0.5,0.5]",
        group_elapsed_p50_ms=1.0,
        group_elapsed_p95_ms=2.0,
        group_elapsed_max_ms=3.0,
        total_group_work_s=1.0,
        worker_load_p50_s=0.5,
        worker_load_p95_s=0.5,
        worker_load_max_s=0.5,
        worker_load_mean_s=0.5,
        load_imbalance_max_over_mean=1.0,
        critical_path_over_ideal=1.0,
        straggler_excess_over_ideal_s=0.0,
        approximate_unattributed_time_s=0.5,
        completed_at="2026-07-24T00:00:00+08:00",
    )


def test_resume_validation_rejects_partial_repeat_and_stale_config() -> None:
    first_order = _schedule_order(0)
    complete = [
        _row(0, schedule, index)
        for index, schedule in enumerate(first_order)
    ]
    _validate_loaded_rows(
        complete,
        workers=2,
        kernels=9,
        presentations=4,
        repeats=10,
    )
    with pytest.raises(RuntimeError, match="incomplete three-schedule repeat"):
        _validate_loaded_rows(
            complete[:2],
            workers=2,
            kernels=9,
            presentations=4,
            repeats=10,
        )
    with pytest.raises(RuntimeError, match="configuration mismatch"):
        _validate_loaded_rows(
            complete,
            workers=3,
            kernels=9,
            presentations=4,
            repeats=10,
        )
