from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from layerprobe.workloads import make_kernels, make_presentations


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "cache_key_ablation.py"
)
SPEC = importlib.util.spec_from_file_location("cache_key_ablation", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
cache_key_ablation = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = cache_key_ablation
SPEC.loader.exec_module(cache_key_ablation)


def test_first_valid_kernel_exposes_all_three_incomplete_keys() -> None:
    kernel = make_kernels(1)[0]
    result = cache_key_ablation.analyze_kernel(
        selection_position=0,
        kernel_index=0,
        kernel=kernel,
        presentations=make_presentations(18),
        variants=cache_key_ablation.EXPECTED_VARIANTS,
        orders=cache_key_ablation.EXPECTED_ORDERS,
        minimum_step_guard=64,
        horizon_guard_multiplier=8,
    )

    assert result["valid"]
    assert result["collision"]["full"]["unsafe_key_classes"] == 0
    for order in cache_key_ablation.EXPECTED_ORDERS:
        control = result["replay"]["full"][order]
        assert control["trace_mismatches"] == 0
        assert control["candidate_signature_mismatches"] == 0
        assert control["unsafe_cache_hits"] == 0
        assert control["nontermination_guards"] == 0

    for variant in (
        "drop_state",
        "drop_memory",
        "drop_observation",
    ):
        assert result["collision"][variant]["unsafe_key_classes"] > 0
        assert variant in result["first_collision_witnesses"]
        for order in cache_key_ablation.EXPECTED_ORDERS:
            replay = result["replay"][variant][order]
            assert replay["trace_mismatches"] > 0
            assert replay["nontermination_guards"] == 0


def test_selection_is_deterministic_and_includes_first_kernel() -> None:
    selected = cache_key_ablation.selection_indices(24_624, 120)
    assert len(selected) == 120
    assert selected[0] == 0
    assert selected == tuple(sorted(set(selected)))
    assert cache_key_ablation.selection_indices(24_624, 24_624) == tuple(
        range(24_624)
    )
