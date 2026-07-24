from __future__ import annotations

import ast
from pathlib import Path

from experiments import parameter_heterogeneity_analysis as analysis


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    PROJECT_ROOT / "experiments" / "parameter_heterogeneity_analysis.py"
)
CONFIG_PATH = (
    PROJECT_ROOT / "experiments" / "independent_trace_oracle_config.json"
)
SPEC_PATH = (
    PROJECT_ROOT / "experiments" / "parameter_heterogeneity_spec.json"
)


def test_analysis_has_no_simulator_or_timing_imports() -> None:
    tree = ast.parse(SCRIPT_PATH.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.add(node.module)
    assert not any(name == "layerprobe" or name.startswith("layerprobe.") for name in imported)
    assert "time" not in imported
    assert "datetime" not in imported


def test_independent_grid_and_frozen_bands_are_exact() -> None:
    config = analysis.read_json(CONFIG_PATH)
    spec = analysis.read_json(SPEC_PATH)
    kernels = analysis.build_kernel_grid(config)
    presentations = analysis.build_presentations(config)

    assert len(kernels) == 24_624
    assert kernels[0] == analysis.KernelParams(
        name="brake_0000",
        goal_start=4,
        brake_force=1,
        horizon=8,
        goal_width=1,
        start_speed=3,
        friction=0,
    )
    assert kernels[-1] == analysis.KernelParams(
        name="brake_24623",
        goal_start=40,
        brake_force=4,
        horizon=12,
        goal_width=3,
        start_speed=14,
        friction=2,
    )
    assert len(presentations) == 18
    assert presentations[0].name == "view_00_exact_exact_d0"
    assert presentations[-1].name == "view_17_hidden_hidden_d1"

    speed_bands = spec["primary_stratification"]["start_speed_bands"]
    assert [analysis.band_for(value, speed_bands) for value in range(3, 15)] == [
        "low",
        "low",
        "low",
        "low",
        "medium",
        "medium",
        "medium",
        "medium",
        "high",
        "high",
        "high",
        "high",
    ]


def test_group_metrics_and_promotion_rule_are_deterministic() -> None:
    requested = (
        analysis.KernelParams("brake_0000", 4, 1, 8, 1, 3, 0),
        analysis.KernelParams("brake_0001", 4, 1, 8, 1, 3, 1),
    )
    valid = (
        analysis.KernelMetrics(
            params=requested[0],
            robust_mask=0b000101,
            union_mask=0b111111,
            delay_sum=-9,
            reference_friction_blind_candidate_hits=0,
        ),
        analysis.KernelMetrics(
            params=requested[1],
            robust_mask=0,
            union_mask=0b000111,
            delay_sum=9,
            reference_friction_blind_candidate_hits=9,
        ),
    )
    metrics = analysis.aggregate_group(
        requested=requested,
        valid=valid,
        presentation_count=18,
        target_mask=0b111111,
    )
    assert metrics["validity_rate"] == 1.0
    assert metrics["robust_nonzero_rate"] == 0.5
    assert metrics["robust_full_rate"] == 0.0
    assert metrics["mean_robust_pairs"] == 1.0
    assert metrics["mean_union_pairs"] == 4.5
    assert metrics["mean_presentation_gap"] == 3.5
    assert metrics["mean_delay_delta"] == 0.0
    assert metrics["delay_positive_kernels"] == 1
    assert metrics["delay_negative_kernels"] == 1
    assert metrics["reference_friction_blind_candidate_rate"] == 0.25

    rows = [
        {
            "valid_kernels": 100,
            "robust_nonzero_rate": 0.80,
            "mean_robust_pairs": 1.0,
            "mean_delay_delta": -0.2,
        },
        {
            "valid_kernels": 120,
            "robust_nonzero_rate": 0.91,
            "mean_robust_pairs": 1.6,
            "mean_delay_delta": 0.1,
        },
    ]
    promotion = analysis.promotion_decision(
        rows,
        {
            "minimum_primary_range": 0.10,
            "minimum_valid_kernels_per_primary_cell": 100,
            "secondary_mean_robust_pairs_range": 0.5,
            "secondary_delay_sign_change_minimum_range": 0.25,
            "rule": "frozen test rule",
        },
    )
    assert promotion["primary_threshold_pass"]
    assert promotion["cell_size_threshold_pass"]
    assert promotion["secondary_mean_robust_pairs_threshold_pass"]
    assert promotion["secondary_delay_threshold_pass"]
    assert promotion["main_text_promotion_pass"]
