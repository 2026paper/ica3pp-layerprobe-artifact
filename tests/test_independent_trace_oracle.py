from __future__ import annotations

import ast
import inspect
from pathlib import Path

from experiments import independent_trace_oracle as oracle
from layerprobe.evaluator import run_flat_parallel


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "independent_trace_oracle_config.json"
)
SCRIPT_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "independent_trace_oracle.py"
)


def test_oracle_has_no_static_import_from_modules_under_test() -> None:
    tree = ast.parse(SCRIPT_PATH.read_text(encoding="utf-8"))
    forbidden = {
        "layerprobe.mechanics",
        "layerprobe.evaluator",
        "layerprobe.workloads",
    }
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.add(node.module)
    assert not (imported & forbidden)

    for name, function in inspect.getmembers(oracle, inspect.isfunction):
        if not name.startswith("oracle_"):
            continue
        source = inspect.getsource(function)
        assert "load_system_under_test" not in source
        assert "layerprobe.mechanics" not in source
        assert "layerprobe.evaluator" not in source
        assert "layerprobe.workloads" not in source


def test_independent_frozen_spec_has_expected_cardinality() -> None:
    config = oracle.load_config(CONFIG_PATH)
    kernels = oracle.oracle_make_kernels(config)
    presentations = oracle.oracle_make_presentations(config)
    assert len(kernels) == 24_624
    assert kernels[0].name == "brake_0000"
    assert kernels[-1].name == "brake_24623"
    assert len(presentations) == 18
    assert presentations[0].name == "view_00_exact_exact_d0"
    assert presentations[-1].name == "view_17_hidden_hidden_d1"


def test_small_independent_comparison_matches_system_under_test() -> None:
    config = oracle.load_config(CONFIG_PATH)
    kernels = oracle.oracle_make_kernels(config)
    presentations = oracle.oracle_make_presentations(config)[:6]
    kernel = next(
        item for item in kernels[:128] if oracle.oracle_verify_kernel(item).valid
    )
    record = oracle.compare_kernel_task(
        (kernel, presentations, oracle.ORACLE_AGENT_NAMES)
    )
    assert record["oracle_valid"]
    assert record["sut_valid"]
    assert record["factorized_valid"]
    assert record["validity_mismatch_count"] == 0
    assert record["factorized_validity_mismatch_count"] == 0
    assert record["trace_cases"] == 6 * 4
    assert record["flat_trace_comparisons"] == 6 * 4
    assert record["factorized_trace_comparisons"] == 6 * 4
    assert record["flat_trace_mismatch_count"] == 0
    assert record["factorized_trace_mismatch_count"] == 0
    assert record["direct_candidate_mismatch_count"] == 0
    assert record["factorized_candidate_mismatch_count"] == 0
    assert record["oracle_trace_sha256"] == record["flat_trace_sha256"]
    assert (
        record["oracle_trace_sha256"]
        == record["factorized_trace_sha256"]
    )
    assert record["oracle_candidate_sha256"] == record["sut_candidate_sha256"]
    assert (
        record["oracle_candidate_sha256"]
        == record["factorized_candidate_sha256"]
    )

    expected_signatures = {}
    for presentation in presentations:
        traces = {
            agent: oracle.oracle_simulate_trace(kernel, presentation, agent)
            for agent in oracle.ORACLE_AGENT_NAMES
        }
        expected_signatures[f"{kernel.name}::{presentation.name}"] = (
            oracle.oracle_signature_for(traces)
        )
    flat_parallel = run_flat_parallel((kernel,), presentations, workers=8)
    assert flat_parallel.candidate_signatures == dict(
        sorted(expected_signatures.items())
    )
    assert flat_parallel.valid_kernels == (kernel.name,)


def test_all_seven_predefined_semantic_mutants_are_detected() -> None:
    config = oracle.load_config(CONFIG_PATH)
    limit = int(config["smoke"]["mutant_search_kernel_count"])
    kernels = oracle.oracle_make_kernels(config)[:limit]
    presentations = oracle.oracle_make_presentations(config)
    report = oracle.run_mutant_smoke(kernels, presentations)
    assert report["mutants_total"] == 7
    assert report["mutants_detected"] == 7
    assert report["all_detected"]
    assert report["undetected_mutants"] == []
    assert all(row["first_witness"] is not None for row in report["mutants"])
    assert all(
        len(row["first_witness"]["oracle_trace_sha256"]) == 64
        for row in report["mutants"]
    )
    assert all(
        len(row["first_witness"]["observed_trace_sha256"]) == 64
        for row in report["mutants"]
    )
