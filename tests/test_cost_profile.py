from __future__ import annotations

import csv
import json
import pstats

import pytest

from experiments import cost_profile
from layerprobe import evaluator, mechanics
from layerprobe.workloads import make_kernels, make_presentations


def test_named_function_categories_and_residual_are_explicit() -> None:
    observation_key = cost_profile._code_key(mechanics.observe)
    policy_key = cost_profile._code_key(mechanics.choose_action)
    transition_key = cost_profile._code_key(mechanics.transition)
    validation_key = cost_profile._code_key(mechanics.verify_kernel)
    aggregation_key = cost_profile._code_key(evaluator.signature_for)

    assert cost_profile.classify_function(observation_key) == (
        cost_profile.OBSERVATION,
        "named project function",
    )
    assert cost_profile.classify_function(policy_key)[0] == (
        cost_profile.POLICY_MEMORY
    )
    assert cost_profile.classify_function(transition_key)[0] == (
        cost_profile.TRANSITION_TERMINAL
    )
    assert cost_profile.classify_function(validation_key)[0] == (
        cost_profile.MECHANISM_VALIDATION
    )
    assert cost_profile.classify_function(aggregation_key)[0] == (
        cost_profile.AGGREGATION_DIGEST
    )
    assert cost_profile.classify_function(
        ("<built-in>", 0, "dict.get")
    ) == (
        cost_profile.EVALUATOR_RESIDUAL,
        "residual by subtraction",
    )
    assert "not an exact measurement" in cost_profile.CATEGORY_NOTES[
        cost_profile.EVALUATOR_RESIDUAL
    ]


def test_cache_counter_derivation_uses_only_production_metrics() -> None:
    counters = cost_profile.cache_counters(
        {
            "observation_calls": 100,
            "policy_calls": 68,
            "transition_calls": 68,
            "prefix_groups": 68,
        }
    )
    assert counters["semantic_requests"] == 100
    assert counters["computed_steps"] == 68
    assert counters["cache_hits"] == 32
    assert counters["hit_rate"] == pytest.approx(0.32)
    assert counters["total_cache_entries_across_ephemeral_caches"] == 68
    assert counters["peak_cache_entries"] is None

    changed_semantics = cost_profile.cache_counters(
        {
            "observation_calls": 100,
            "policy_calls": 68,
            "transition_calls": 68,
            "prefix_groups": 67,
        }
    )
    assert (
        changed_semantics["total_cache_entries_across_ephemeral_caches"]
        is None
    )

    with pytest.raises(ValueError, match="policy and transition"):
        cost_profile.cache_counters(
            {
                "observation_calls": 100,
                "policy_calls": 68,
                "transition_calls": 67,
                "prefix_groups": 68,
            }
        )


def test_profile_artifacts_are_loadable_and_semantically_auditable(
    tmp_path,
) -> None:
    # Use the same known-valid deterministic prefix as the core semantic test.
    kernels = make_kernels(12)
    presentations = make_presentations(6)
    plan = cost_profile.ProfilePlan(
        output=tmp_path / "profile",
        kernel_count=len(kernels),
        presentation_count=len(presentations),
        smoke=True,
    )
    summary = cost_profile.profile_artifacts(
        plan,
        kernels,
        presentations,
        {
            "kernel_selection": "test_prefix",
            "presentation_selection": "test_prefix",
        },
    )

    output = plan.output
    raw_path = output / cost_profile.RAW_PROFILE_NAME
    csv_path = output / cost_profile.FUNCTION_CSV_NAME
    summary_path = output / cost_profile.SUMMARY_NAME
    metadata_path = output / cost_profile.METADATA_NAME
    assert raw_path.is_file()
    assert csv_path.is_file()
    assert summary_path.is_file()
    assert metadata_path.is_file()

    raw_stats = pstats.Stats(str(raw_path))
    assert raw_stats.total_calls == summary["profile_total_calls"]
    assert len(summary["semantic_digest_sha256"]) == 64
    assert summary["cache_counters"]["cache_hits"] >= 0
    assert summary["cache_counters"]["peak_cache_entries"] is None
    assert set(summary["category_breakdown"]) == set(
        cost_profile.CATEGORY_ORDER
    )
    self_share = sum(
        item["self_time_share"]
        for item in summary["category_breakdown"].values()
    )
    assert self_share == pytest.approx(1.0)

    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert {row["category"] for row in rows} <= set(
        cost_profile.CATEGORY_ORDER
    )
    assert any(
        row["function"] == "observe"
        and row["category"] == cost_profile.OBSERVATION
        for row in rows
    )

    persisted_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert persisted_summary["semantic_digest_sha256"] == (
        summary["semantic_digest_sha256"]
    )
    assert metadata["status"] == "complete"
    assert len(
        metadata["fingerprints"]["code_fingerprint_sha256"]
    ) == 64

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        cost_profile.profile_artifacts(
            plan,
            kernels,
            presentations,
            {
                "kernel_selection": "test_prefix",
                "presentation_selection": "test_prefix",
            },
        )
