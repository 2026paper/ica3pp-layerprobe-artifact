"""Run a frozen out-of-primary-grid braking stress test.

This experiment extends numeric ranges within the same braking mechanism
family.  It is an exactness and work-accounting check, not a second domain and
not a formal timing benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any

from layerprobe.evaluator import RunResult, run_factorized, run_flat
from layerprobe.model import KernelSpec
from layerprobe.workloads import make_presentations


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*.py") if item.is_file()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        data = path.read_bytes()
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


def _signature_sha256(signatures: dict[str, int]) -> str:
    payload = json.dumps(
        sorted(signatures.items()),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "experiment_id",
        "stress_grid",
        "presentations",
        "factorized_workers",
        "acceptance_gates",
        "reporting_boundary",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"config missing required keys: {missing}")
    if not config.get("frozen_before_execution"):
        raise ValueError("config must declare frozen_before_execution=true")
    if int(config["presentations"]) != 18:
        raise ValueError("this stress test must use the exhaustive 18 presentations")
    return config


def _make_stress_kernels(config: dict[str, Any]) -> tuple[KernelSpec, ...]:
    grid = config["stress_grid"]
    keys = (
        "goal_start",
        "start_speed",
        "horizon",
        "goal_width",
        "brake_force",
        "friction",
    )
    missing = [key for key in keys if key not in grid]
    if missing:
        raise ValueError(f"stress_grid missing keys: {missing}")

    kernels: list[KernelSpec] = []
    for index, values in enumerate(product(*(grid[key] for key in keys))):
        goal_start, start_speed, horizon, goal_width, brake_force, friction = (
            int(value) for value in values
        )
        kernel = KernelSpec(
            name=f"stress_brake_{index:04d}",
            start_speed=start_speed,
            friction=friction,
            brake_force=brake_force,
            goal_start=goal_start,
            goal_end=goal_start + goal_width,
            horizon=horizon,
        )
        kernel.validate()
        kernels.append(kernel)
    if len({kernel.name for kernel in kernels}) != len(kernels):
        raise AssertionError("stress kernel names are not unique")
    return tuple(kernels)


def _timed(callable_obj: Any, *args: Any, **kwargs: Any) -> tuple[RunResult, float]:
    start = time.perf_counter()
    result = callable_obj(*args, **kwargs)
    return result, time.perf_counter() - start


def _first_mismatch(flat: RunResult, factorized: RunResult) -> dict[str, Any] | None:
    candidates = sorted(
        set(flat.candidate_signatures) | set(factorized.candidate_signatures)
    )
    for candidate in candidates:
        left = flat.candidate_signatures.get(candidate)
        right = factorized.candidate_signatures.get(candidate)
        if left != right:
            return {
                "candidate": candidate,
                "flat_signature": left,
                "factorized_signature": right,
            }
    return None


def _result_record(result: RunResult, elapsed: float) -> dict[str, Any]:
    return {
        "valid_kernels": len(result.valid_kernels),
        "candidates": len(result.candidate_signatures),
        "minimum_suite": list(result.minimum_suite) if result.minimum_suite else None,
        "candidate_signature_sha256": _signature_sha256(result.candidate_signatures),
        "metrics": result.metrics,
        "elapsed_diagnostic_seconds": elapsed,
    }


def _write_markdown(summary: dict[str, Any], path: Path) -> None:
    flat = summary["results"]["Flat"]
    factorized = summary["results"]["LayerProbe-P8"]
    policy_reduction = summary["work_deltas"]["policy_transition_call_reduction"]
    graph_reduction = summary["work_deltas"]["graph_build_reduction"]
    gates = summary["gates"]
    lines = [
        "# Range-extension braking stress test",
        "",
        "This is a frozen numeric range extension within the braking mechanism family.",
        "It is neither a second domain nor a matched timing benchmark.",
        "",
        f"- Requested kernels: {summary['requested_kernels']:,}",
        f"- Valid kernels: {flat['valid_kernels']:,}",
        f"- Presentations: {summary['presentations']}",
        f"- Candidates: {flat['candidates']:,}",
        f"- Exact candidate-signature agreement: {gates['identical_candidate_signatures']}",
        f"- Exact minimum-suite agreement: {gates['identical_minimum_suite']}",
        "",
        "## Work accounting",
        "",
        "| Quantity | Flat | LayerProbe-P8 | Reduction |",
        "|---|---:|---:|---:|",
        (
            f"| Graph builds | {flat['metrics']['graph_builds']:,} | "
            f"{factorized['metrics']['graph_builds']:,} | {graph_reduction:.3%} |"
        ),
        (
            f"| Policy calls | {flat['metrics']['policy_calls']:,} | "
            f"{factorized['metrics']['policy_calls']:,} | {policy_reduction:.3%} |"
        ),
        (
            f"| Transition calls | {flat['metrics']['transition_calls']:,} | "
            f"{factorized['metrics']['transition_calls']:,} | {policy_reduction:.3%} |"
        ),
        "",
        "Elapsed times in `summary.json` are diagnostic only and are not performance evidence.",
        "",
        "## Gates",
        "",
    ]
    lines.extend(
        f"- {name}: {'PASS' if passed else 'FAIL'}"
        for name, passed in sorted(gates.items())
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("range_extension_stress_config.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config_path = args.config.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    output.mkdir(parents=True)

    try:
        config = _load_config(config_path)
        kernels = _make_stress_kernels(config)
        presentations = make_presentations(int(config["presentations"]))
        flat, flat_elapsed = _timed(run_flat, kernels, presentations)
        factorized, factorized_elapsed = _timed(
            run_factorized,
            kernels,
            presentations,
            workers=int(config["factorized_workers"]),
        )

        gates = {
            "identical_valid_kernel_names": flat.valid_kernels
            == factorized.valid_kernels,
            "identical_candidate_signatures": flat.candidate_signatures
            == factorized.candidate_signatures,
            "identical_minimum_suite": flat.minimum_suite
            == factorized.minimum_suite,
            "flat_graph_builds_equal_requested_kernels_times_presentations": (
                flat.metrics["graph_builds"] == len(kernels) * len(presentations)
            ),
            "factorized_graph_builds_equal_requested_kernels": (
                factorized.metrics["graph_builds"] == len(kernels)
            ),
        }
        flat_calls = flat.metrics["policy_calls"]
        factorized_calls = factorized.metrics["policy_calls"]
        summary = {
            "status": "PASS" if all(gates.values()) else "FAIL",
            "generated_at": datetime.now().astimezone().isoformat(),
            "experiment_id": config["experiment_id"],
            "scientific_role": config["scientific_role"],
            "reporting_boundary": config["reporting_boundary"],
            "config_sha256": _sha256(config_path),
            "script_sha256": _sha256(Path(__file__).resolve()),
            "core_source_sha256": _source_tree_sha256(
                Path(__file__).resolve().parents[1] / "src"
            ),
            "requested_kernels": len(kernels),
            "presentations": len(presentations),
            "results": {
                "Flat": _result_record(flat, flat_elapsed),
                "LayerProbe-P8": _result_record(
                    factorized, factorized_elapsed
                ),
            },
            "work_deltas": {
                "graph_build_reduction": 1.0
                - factorized.metrics["graph_builds"] / flat.metrics["graph_builds"],
                "graph_build_ratio_flat_over_factorized": (
                    flat.metrics["graph_builds"]
                    / factorized.metrics["graph_builds"]
                ),
                "policy_transition_call_reduction": 1.0
                - factorized_calls / flat_calls,
            },
            "gates": gates,
            "first_mismatch": _first_mismatch(flat, factorized),
        }

        (output / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        (output / "valid_kernel_names.txt").write_text(
            "\n".join(flat.valid_kernels) + "\n",
            encoding="utf-8",
        )
        shutil.copyfile(config_path, output / "frozen_config.json")
        _write_markdown(summary, output / "RANGE_EXTENSION_STRESS.md")
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0 if all(gates.values()) else 2
    except BaseException:
        failure = {
            "status": "ERROR",
            "generated_at": datetime.now().astimezone().isoformat(),
            "config_sha256": _sha256(config_path) if config_path.exists() else None,
        }
        (output / "failure.json").write_text(
            json.dumps(failure, indent=2) + "\n",
            encoding="utf-8",
        )
        raise


if __name__ == "__main__":
    sys.exit(main())
