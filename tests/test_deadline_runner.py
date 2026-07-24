from __future__ import annotations

from dataclasses import replace

import pytest

from experiments import deadline_runner


def _row(
    method: str,
    workers: int,
    elapsed_s: float,
    *,
    graph_builds: int,
    policy_calls: int,
) -> deadline_runner.RunRow:
    return deadline_runner.RunRow(
        job_id=f"{method}-w{workers}",
        study="method_ladder",
        case="16k_18p",
        repeat=0,
        order_index=0,
        method=method,
        workers=workers,
        kernel_count=16,
        kernel_selection="test",
        presentation_count=18,
        presentation_set="test",
        elapsed_s=elapsed_s,
        worker_slot_s=elapsed_s * workers,
        peak_process_tree_rss_mb=None,
        digest="same-semantics",
        valid_kernels=2,
        candidates=36,
        frontier=4,
        suite_size=2,
        graph_builds=graph_builds,
        graph_states=10,
        graph_transitions=20,
        observation_calls=100,
        policy_calls=policy_calls,
        transition_calls=policy_calls,
        prefix_groups=0,
        completed_at="2026-07-24T00:00:00+08:00",
    )


def test_smoke_jobs_include_schedule_matched_flat_p8() -> None:
    config = deadline_runner.load_config(deadline_runner.DEFAULT_CONFIG)
    jobs = deadline_runner.build_jobs(
        config,
        mode="smoke",
        primary_workers=8,
        throughput_workers=8,
        maximum_workers=8,
    )

    gate = [job for job in jobs if job.study == "correctness_gate"]
    assert ("flat_parallel", 8) in {
        (job.method, job.workers) for job in gate
    }

    ladder = [
        job
        for job in jobs
        if job.study == "method_ladder"
        and job.case == "300k_18p"
        and job.repeat == 0
    ]
    lookup = {(job.method, job.workers): job for job in ladder}
    flat_p8 = lookup[("flat_parallel", 8)]
    for key in (("kernel_memo_parallel", 8), ("factorized", 8)):
        comparator = lookup[key]
        assert comparator.repeat == flat_p8.repeat
        assert comparator.kernel_count == flat_p8.kernel_count
        assert comparator.presentation_indices == flat_p8.presentation_indices


def test_summary_reports_schedule_matched_flat_ratio(tmp_path) -> None:
    flat = _row("flat", 1, 8.0, graph_builds=288, policy_calls=100)
    rows = [
        flat,
        _row("kernel_memo", 1, 4.0, graph_builds=16, policy_calls=100),
        _row("factorized", 1, 2.0, graph_builds=16, policy_calls=40),
        _row("flat_parallel", 8, 5.0, graph_builds=288, policy_calls=100),
        _row(
            "kernel_memo_parallel",
            8,
            2.5,
            graph_builds=16,
            policy_calls=100,
        ),
        _row("factorized", 8, 1.25, graph_builds=16, policy_calls=40),
    ]
    rows = [replace(row, order_index=index) for index, row in enumerate(rows)]

    summary = deadline_runner.summarize(
        rows,
        metadata={},
        selected_studies={row.study for row in rows},
        planned_job_ids={row.job_id for row in rows},
    )
    effect = summary["method_effects"][0]
    assert effect["flat_parallel_pair_repeats"] == 1
    assert effect["parallel_flat_to_kernel_memo_paired_median"] == pytest.approx(2.0)
    assert effect["parallel_flat_to_factorized_paired_median"] == pytest.approx(4.0)

    metadata = {
        "profile_name": "test",
        "mode": "smoke",
        "python_executable": "python",
        "physical_cores": 8,
        "primary_workers": 8,
        "throughput_workers": 8,
    }
    summary["metadata"] = metadata
    markdown = tmp_path / "SUMMARY.md"
    deadline_runner.write_summary_markdown(summary, markdown)
    assert "同调度并行 flat→factorized" in markdown.read_text(encoding="utf-8")
