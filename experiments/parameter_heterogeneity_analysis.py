"""Saved-output parameter-stratified robustness census.

This module reads a frozen candidate-signature table and independently
reconstructs the declared Cartesian parameter grid from JSON.  It never
imports or calls the simulator.  The analysis is a complete finite-domain
description of four declared computational agents, not an inferential,
causal, human-effect, or cross-domain study.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from itertools import combinations, product
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPEC = Path(__file__).with_name("parameter_heterogeneity_spec.json")


@dataclass(frozen=True, slots=True)
class KernelParams:
    name: str
    goal_start: int
    brake_force: int
    horizon: int
    goal_width: int
    start_speed: int
    friction: int


@dataclass(frozen=True, slots=True)
class Presentation:
    name: str
    speed_mode: str
    distance_mode: str
    delay: int


@dataclass(frozen=True, slots=True)
class KernelMetrics:
    params: KernelParams
    robust_mask: int
    union_mask: int
    delay_sum: int
    reference_friction_blind_candidate_hits: int

    @property
    def robust_pairs(self) -> int:
        return self.robust_mask.bit_count()

    @property
    def union_pairs(self) -> int:
        return self.union_mask.bit_count()

    @property
    def presentation_gap(self) -> int:
        return self.union_pairs - self.robust_pairs


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write an empty table: {path.name}")
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def inclusive_range(specification: Mapping[str, Any]) -> tuple[int, ...]:
    start = int(specification["start"])
    stop = int(specification["stop_inclusive"])
    step = int(specification["step"])
    if step <= 0 or start > stop:
        raise ValueError(f"invalid inclusive range: {dict(specification)}")
    values = tuple(range(start, stop + 1, step))
    if not values or values[-1] > stop:
        raise ValueError(f"range expansion failed: {dict(specification)}")
    return values


def build_kernel_grid(config: Mapping[str, Any]) -> tuple[KernelParams, ...]:
    grid = config["kernel_grid"]
    kernels: list[KernelParams] = []
    for index, values in enumerate(
        product(
            inclusive_range(grid["goal_start"]),
            tuple(int(value) for value in grid["brake_force"]),
            tuple(int(value) for value in grid["horizon"]),
            tuple(int(value) for value in grid["goal_width"]),
            inclusive_range(grid["start_speed"]),
            tuple(int(value) for value in grid["friction"]),
        )
    ):
        (
            goal_start,
            brake_force,
            horizon,
            goal_width,
            start_speed,
            friction,
        ) = values
        kernels.append(
            KernelParams(
                name=f"brake_{index:04d}",
                goal_start=goal_start,
                brake_force=brake_force,
                horizon=horizon,
                goal_width=goal_width,
                start_speed=start_speed,
                friction=friction,
            )
        )
    return tuple(kernels)


def build_presentations(config: Mapping[str, Any]) -> tuple[Presentation, ...]:
    presentation_config = config["presentations"]
    presentations: list[Presentation] = []
    for index, (speed_mode, distance_mode, delay) in enumerate(
        product(
            tuple(str(value) for value in presentation_config["speed_modes"]),
            tuple(str(value) for value in presentation_config["distance_modes"]),
            tuple(int(value) for value in presentation_config["delays"]),
        )
    ):
        presentations.append(
            Presentation(
                name=(
                    f"view_{index:02d}_{speed_mode}_{distance_mode}_d{delay}"
                ),
                speed_mode=speed_mode,
                distance_mode=distance_mode,
                delay=delay,
            )
        )
    return tuple(presentations)


def band_for(
    value: int,
    bands: Sequence[Mapping[str, Any]],
) -> str:
    matches = [
        str(band["name"])
        for band in bands
        if int(band["minimum"]) <= value <= int(band["maximum"])
    ]
    if len(matches) != 1:
        raise ValueError(
            f"value {value} belongs to {len(matches)} frozen bands"
        )
    return matches[0]


def validate_input_hashes(
    spec: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    manifest: dict[str, dict[str, Any]] = {}
    for input_id, entry in spec["inputs"].items():
        path = (PROJECT_ROOT / str(entry["relative_path"])).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"missing frozen input: {path}")
        observed = sha256_file(path)
        expected = str(entry["sha256"]).lower()
        if observed != expected:
            raise ValueError(
                f"frozen input hash mismatch for {input_id}: "
                f"expected {expected}, observed {observed}"
            )
        manifest[str(input_id)] = {
            "relative_path": path.relative_to(PROJECT_ROOT).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": observed,
        }
    return manifest


def validate_presentation_metadata(
    summary: Mapping[str, Any],
    presentations: Sequence[Presentation],
) -> None:
    expected = {item.name: item for item in presentations}
    rows = summary["presentation_conditions"]
    observed = {
        str(row["presentation"]): (
            str(row["speed_mode"]),
            str(row["distance_mode"]),
            int(row["delay"]),
        )
        for row in rows
    }
    expected_payload = {
        name: (item.speed_mode, item.distance_mode, item.delay)
        for name, item in expected.items()
    }
    if observed != expected_payload:
        raise ValueError("presentation metadata differs from the frozen grid")


def load_candidate_masks(
    path: Path,
    *,
    kernel_names: set[str],
    presentation_names: set[str],
    expected_pairs: int,
) -> tuple[dict[str, dict[str, int]], int]:
    masks_by_kernel: dict[str, dict[str, int]] = {}
    row_count = 0
    with gzip.open(
        path,
        "rt",
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        reader = csv.DictReader(handle)
        expected_fields = [
            "kernel",
            "presentation",
            "signature_mask",
            "pairs_separated",
        ]
        if reader.fieldnames != expected_fields:
            raise ValueError(
                f"unexpected candidate table fields: {reader.fieldnames}"
            )
        maximum_mask = (1 << expected_pairs) - 1
        for row in reader:
            row_count += 1
            kernel = str(row["kernel"])
            presentation = str(row["presentation"])
            if kernel not in kernel_names:
                raise ValueError(f"unknown kernel in signature table: {kernel}")
            if presentation not in presentation_names:
                raise ValueError(
                    f"unknown presentation in signature table: {presentation}"
                )
            mask = int(row["signature_mask"])
            pairs_separated = int(row["pairs_separated"])
            if not 0 <= mask <= maximum_mask:
                raise ValueError(f"signature mask outside domain: {mask}")
            if pairs_separated != mask.bit_count():
                raise ValueError(
                    f"stored bit count mismatch for {kernel}/{presentation}"
                )
            kernel_masks = masks_by_kernel.setdefault(kernel, {})
            if presentation in kernel_masks:
                raise ValueError(
                    f"duplicate candidate key: {kernel}/{presentation}"
                )
            kernel_masks[presentation] = mask
    return masks_by_kernel, row_count


def compute_kernel_metrics(
    masks_by_kernel: Mapping[str, Mapping[str, int]],
    *,
    kernel_map: Mapping[str, KernelParams],
    presentations: Sequence[Presentation],
    reference_friction_blind_bit: int,
    target_mask: int,
) -> dict[str, KernelMetrics]:
    presentation_names = tuple(item.name for item in presentations)
    paired_names: list[tuple[str, str]] = []
    for speed_mode in ("exact", "coarse", "hidden"):
        for distance_mode in ("exact", "coarse", "hidden"):
            immediate = next(
                item.name
                for item in presentations
                if item.speed_mode == speed_mode
                and item.distance_mode == distance_mode
                and item.delay == 0
            )
            delayed = next(
                item.name
                for item in presentations
                if item.speed_mode == speed_mode
                and item.distance_mode == distance_mode
                and item.delay == 1
            )
            paired_names.append((immediate, delayed))
    if len(paired_names) != 9:
        raise ValueError("frozen delay-pair design must contain nine pairs")

    result: dict[str, KernelMetrics] = {}
    for kernel_name, raw_masks in masks_by_kernel.items():
        if set(raw_masks) != set(presentation_names):
            raise ValueError(
                f"{kernel_name} does not contain exactly the 18 presentations"
            )
        robust_mask = target_mask
        union_mask = 0
        reference_friction_blind_hits = 0
        for name in presentation_names:
            mask = int(raw_masks[name])
            robust_mask &= mask
            union_mask |= mask
            reference_friction_blind_hits += int(
                bool(mask & reference_friction_blind_bit)
            )
        delay_sum = sum(
            int(raw_masks[delayed]).bit_count()
            - int(raw_masks[immediate]).bit_count()
            for immediate, delayed in paired_names
        )
        result[kernel_name] = KernelMetrics(
            params=kernel_map[kernel_name],
            robust_mask=robust_mask,
            union_mask=union_mask,
            delay_sum=delay_sum,
            reference_friction_blind_candidate_hits=(
                reference_friction_blind_hits
            ),
        )
    return result


def mean(values: Iterable[int | float]) -> float:
    materialized = tuple(values)
    if not materialized:
        raise ValueError("mean is undefined for an empty group")
    return math.fsum(float(value) for value in materialized) / len(
        materialized
    )


def aggregate_group(
    *,
    requested: Sequence[KernelParams],
    valid: Sequence[KernelMetrics],
    presentation_count: int,
    target_mask: int,
) -> dict[str, Any]:
    requested_count = len(requested)
    valid_count = len(valid)
    if requested_count == 0 or valid_count == 0:
        raise ValueError("frozen strata must contain requested and valid kernels")
    robust_nonzero = sum(item.robust_mask != 0 for item in valid)
    robust_full = sum(item.robust_mask == target_mask for item in valid)
    delay_positive = sum(item.delay_sum > 0 for item in valid)
    delay_zero = sum(item.delay_sum == 0 for item in valid)
    delay_negative = sum(item.delay_sum < 0 for item in valid)
    return {
        "requested_kernels": requested_count,
        "valid_kernels": valid_count,
        "validity_rate": valid_count / requested_count,
        "robust_nonzero_kernels": robust_nonzero,
        "robust_nonzero_rate": robust_nonzero / valid_count,
        "robust_full_kernels": robust_full,
        "robust_full_rate": robust_full / valid_count,
        "mean_robust_pairs": mean(
            item.robust_pairs for item in valid
        ),
        "mean_union_pairs": mean(item.union_pairs for item in valid),
        "mean_presentation_gap": mean(
            item.presentation_gap for item in valid
        ),
        "mean_delay_delta": (
            sum(item.delay_sum for item in valid) / (valid_count * 9)
        ),
        "delay_positive_kernels": delay_positive,
        "delay_positive_rate": delay_positive / valid_count,
        "delay_zero_kernels": delay_zero,
        "delay_zero_rate": delay_zero / valid_count,
        "delay_negative_kernels": delay_negative,
        "delay_negative_rate": delay_negative / valid_count,
        "reference_friction_blind_candidate_rate": (
            sum(
                item.reference_friction_blind_candidate_hits
                for item in valid
            )
            / (valid_count * presentation_count)
        ),
        "reference_friction_blind_robust_rate": (
            sum(
                bool(
                    item.robust_mask
                    & (
                        1
                        << 2
                    )
                )
                for item in valid
            )
            / valid_count
        ),
    }


def build_primary_rows(
    *,
    spec: Mapping[str, Any],
    kernels: Sequence[KernelParams],
    metrics: Mapping[str, KernelMetrics],
    presentation_count: int,
    target_mask: int,
) -> list[dict[str, Any]]:
    primary = spec["primary_stratification"]
    speed_bands = primary["start_speed_bands"]
    rows: list[dict[str, Any]] = []
    for friction in primary["friction_levels"]:
        friction_value = int(friction)
        for order, band in enumerate(speed_bands):
            band_name = str(band["name"])
            minimum = int(band["minimum"])
            maximum = int(band["maximum"])
            requested = [
                item
                for item in kernels
                if item.friction == friction_value
                and minimum <= item.start_speed <= maximum
            ]
            valid = [
                item
                for item in metrics.values()
                if item.params.friction == friction_value
                and minimum <= item.params.start_speed <= maximum
            ]
            rows.append(
                {
                    "friction": friction_value,
                    "start_speed_band": band_name,
                    "start_speed_minimum": minimum,
                    "start_speed_maximum": maximum,
                    "speed_band_order": order,
                    **aggregate_group(
                        requested=requested,
                        valid=valid,
                        presentation_count=presentation_count,
                        target_mask=target_mask,
                    ),
                }
            )
    return rows


def build_marginal_rows(
    *,
    spec: Mapping[str, Any],
    kernels: Sequence[KernelParams],
    metrics: Mapping[str, KernelMetrics],
    presentation_count: int,
    target_mask: int,
) -> list[dict[str, Any]]:
    primary = spec["primary_stratification"]
    secondary = spec["secondary_marginals"]
    speed_bands = primary["start_speed_bands"]
    goal_bands = secondary["goal_start_bands"]

    factor_levels: list[
        tuple[
            str,
            str,
            str,
            Callable[[KernelParams], bool],
        ]
    ] = []
    for value in primary["friction_levels"]:
        integer = int(value)
        factor_levels.append(
            (
                "friction",
                str(integer),
                f"friction == {integer}",
                lambda item, expected=integer: item.friction == expected,
            )
        )
    for band in speed_bands:
        name = str(band["name"])
        minimum = int(band["minimum"])
        maximum = int(band["maximum"])
        factor_levels.append(
            (
                "start_speed_band",
                name,
                f"{minimum} <= start_speed <= {maximum}",
                lambda item, lower=minimum, upper=maximum: (
                    lower <= item.start_speed <= upper
                ),
            )
        )
    for factor in ("brake_force", "goal_width", "horizon"):
        for value in secondary[factor]:
            integer = int(value)
            factor_levels.append(
                (
                    factor,
                    str(integer),
                    f"{factor} == {integer}",
                    lambda item, field=factor, expected=integer: (
                        getattr(item, field) == expected
                    ),
                )
            )
    for band in goal_bands:
        name = str(band["name"])
        minimum = int(band["minimum"])
        maximum = int(band["maximum"])
        factor_levels.append(
            (
                "goal_start_band",
                name,
                f"{minimum} <= goal_start <= {maximum}",
                lambda item, lower=minimum, upper=maximum: (
                    lower <= item.goal_start <= upper
                ),
            )
        )

    factor_order: dict[str, int] = {}
    rows: list[dict[str, Any]] = []
    for factor, level, definition, selector in factor_levels:
        order = factor_order.get(factor, 0)
        factor_order[factor] = order + 1
        requested = [item for item in kernels if selector(item)]
        valid = [
            item
            for item in metrics.values()
            if selector(item.params)
        ]
        rows.append(
            {
                "factor": factor,
                "level": level,
                "level_order": order,
                "definition": definition,
                **aggregate_group(
                    requested=requested,
                    valid=valid,
                    presentation_count=presentation_count,
                    target_mask=target_mask,
                ),
            }
        )
    return rows


def promotion_decision(
    rows: Sequence[Mapping[str, Any]],
    promotion_spec: Mapping[str, Any],
) -> dict[str, Any]:
    robust_rates = [
        float(row["robust_nonzero_rate"]) for row in rows
    ]
    robust_means = [float(row["mean_robust_pairs"]) for row in rows]
    delay_values = [float(row["mean_delay_delta"]) for row in rows]
    primary_range = max(robust_rates) - min(robust_rates)
    robust_pairs_range = max(robust_means) - min(robust_means)
    delay_range = max(delay_values) - min(delay_values)
    primary_threshold = float(
        promotion_spec["minimum_primary_range"]
    )
    minimum_cell_size = int(
        promotion_spec["minimum_valid_kernels_per_primary_cell"]
    )
    all_cells_large_enough = all(
        int(row["valid_kernels"]) >= minimum_cell_size for row in rows
    )
    primary_pass = primary_range >= primary_threshold
    secondary_robust_pass = robust_pairs_range >= float(
        promotion_spec["secondary_mean_robust_pairs_range"]
    )
    delay_sign_change = (
        any(value > 0 for value in delay_values)
        and any(value < 0 for value in delay_values)
    )
    secondary_delay_pass = (
        delay_sign_change
        and delay_range
        >= float(
            promotion_spec[
                "secondary_delay_sign_change_minimum_range"
            ]
        )
    )
    main_text_pass = primary_pass and all_cells_large_enough
    return {
        "primary_metric": "robust_nonzero_rate",
        "primary_range": primary_range,
        "primary_threshold": primary_threshold,
        "primary_threshold_pass": primary_pass,
        "minimum_observed_valid_kernels_per_cell": min(
            int(row["valid_kernels"]) for row in rows
        ),
        "minimum_required_valid_kernels_per_cell": minimum_cell_size,
        "cell_size_threshold_pass": all_cells_large_enough,
        "secondary_mean_robust_pairs_range": robust_pairs_range,
        "secondary_mean_robust_pairs_threshold_pass": secondary_robust_pass,
        "secondary_delay_delta_range": delay_range,
        "secondary_delay_sign_change": delay_sign_change,
        "secondary_delay_threshold_pass": secondary_delay_pass,
        "main_text_promotion_pass": main_text_pass,
        "decision": (
            "main_text_one_sentence_plus_appendix_table"
            if main_text_pass
            else "appendix_only"
        ),
        "rule": str(promotion_spec["rule"]),
    }


def gate(
    gates: list[dict[str, Any]],
    name: str,
    passed: bool,
    observed: Any,
    expected: Any,
) -> None:
    gates.append(
        {
            "gate": name,
            "status": "PASS" if passed else "FAIL",
            "observed": observed,
            "expected": expected,
        }
    )


def summary_aggregate_delay(summary: Mapping[str, Any]) -> float:
    rows = [
        row
        for row in summary["factor_effects"]
        if str(row["factor"]) == "delay"
    ]
    by_level = {int(row["level"]): row for row in rows}
    if set(by_level) != {0, 1}:
        raise ValueError("communication summary lacks both delay levels")
    return float(by_level[1]["mean_pairs_separated"]) - float(
        by_level[0]["mean_pairs_separated"]
    )


def write_markdown(
    *,
    path: Path,
    spec: Mapping[str, Any],
    primary_rows: Sequence[Mapping[str, Any]],
    global_metrics: Mapping[str, Any],
    promotion: Mapping[str, Any],
    hashes: Mapping[str, Any],
    gates: Sequence[Mapping[str, Any]],
) -> None:
    lines = [
        "# Mechanism-parameter robustness heterogeneity",
        "",
        "Overall: **PASS**",
        "",
        "## Frozen question",
        "",
        str(spec["analysis_question"]),
        "",
        "This is a complete saved-output census for the declared finite braking "
        "domain. It does not call the simulator and is not evidence about human "
        "learning, diagnostic accuracy, causality, or cross-domain behavior.",
        "",
        "## Primary friction × start-speed strata",
        "",
        "| Friction | Speed band | Requested | Valid | Validity | "
        "Robust nonzero | Mean robust pairs | Mean gap | Delay delta |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in primary_rows:
        lines.append(
            f"| {row['friction']} | {row['start_speed_band']} | "
            f"{row['requested_kernels']} | {row['valid_kernels']} | "
            f"{100 * float(row['validity_rate']):.2f}% | "
            f"{100 * float(row['robust_nonzero_rate']):.2f}% | "
            f"{float(row['mean_robust_pairs']):.3f} | "
            f"{float(row['mean_presentation_gap']):.3f} | "
            f"{float(row['mean_delay_delta']):+.3f} |"
        )
    lines.extend(
        [
            "",
            "## Global reconciliation",
            "",
            f"- Requested/valid mechanisms: "
            f"{global_metrics['requested_kernels']:,}/"
            f"{global_metrics['valid_kernels']:,}.",
            f"- All-presentation robust-nonzero mechanisms: "
            f"{global_metrics['robust_nonzero_kernels']:,}.",
            f"- All-presentation robust-full mechanisms: "
            f"{global_metrics['robust_full_kernels']:,}.",
            f"- Aggregate matched delay delta: "
            f"{float(global_metrics['mean_delay_delta']):+.12f} pairs.",
            "",
            "## Frozen main-text promotion rule",
            "",
            f"- Robust-nonzero range: "
            f"{100 * float(promotion['primary_range']):.3f} percentage points "
            f"(threshold "
            f"{100 * float(promotion['primary_threshold']):.1f}; "
            f"{'PASS' if promotion['primary_threshold_pass'] else 'FAIL'}).",
            f"- Smallest primary cell: "
            f"{promotion['minimum_observed_valid_kernels_per_cell']} valid "
            f"mechanisms (minimum "
            f"{promotion['minimum_required_valid_kernels_per_cell']}; "
            f"{'PASS' if promotion['cell_size_threshold_pass'] else 'FAIL'}).",
            f"- Placement decision: `{promotion['decision']}`.",
            "",
            "Secondary thresholds are descriptive support only and cannot replace "
            "the primary threshold. Complete marginal results are in "
            "`marginal_parameter_effects.csv`.",
            "",
            "## Acceptance gates",
            "",
        ]
    )
    lines.extend(
        f"- {item['gate']}: **{item['status']}**" for item in gates
    )
    lines.extend(
        [
            "",
            "## Provenance",
            "",
            f"- Spec SHA-256: `{hashes['spec_sha256']}`",
            f"- Script SHA-256: `{hashes['script_sha256']}`",
            f"- Candidate table SHA-256: "
            f"`{hashes['inputs']['candidate_signatures']['sha256']}`",
            "",
            "No adaptive bins, additional interactions, best-cell search, "
            "per-cell cover search, p-values, or sampled confidence intervals "
            "were used.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    spec_path = args.spec.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(
            f"refusing to overwrite existing output directory: {output}"
        )

    spec = read_json(spec_path)
    if int(spec.get("schema_version", 0)) != 1:
        raise ValueError("unsupported frozen analysis specification")
    input_manifest = validate_input_hashes(spec)
    summary_path = (
        PROJECT_ROOT
        / spec["inputs"]["communication_summary"]["relative_path"]
    ).resolve()
    candidate_path = (
        PROJECT_ROOT
        / spec["inputs"]["candidate_signatures"]["relative_path"]
    ).resolve()
    grid_config_path = (
        PROJECT_ROOT
        / spec["inputs"]["frozen_grid_config"]["relative_path"]
    ).resolve()
    communication_summary = read_json(summary_path)
    grid_config = read_json(grid_config_path)
    expected = spec["expected_domain"]

    kernels = build_kernel_grid(grid_config)
    kernel_map = {item.name: item for item in kernels}
    presentations = build_presentations(grid_config)
    validate_presentation_metadata(
        communication_summary,
        presentations,
    )
    agents = tuple(str(value) for value in grid_config["agents"])
    expected_pairs = tuple(combinations(agents, 2))
    observed_pairs = tuple(
        tuple(str(value) for value in pair)
        for pair in communication_summary["model_pairs"]
    )
    if observed_pairs != expected_pairs:
        raise ValueError("agent-pair ordering differs from the frozen design")
    reference_pair = ("reference", "friction_blind")
    reference_pair_index = expected_pairs.index(reference_pair)
    target_mask = (1 << len(expected_pairs)) - 1

    masks_by_kernel, candidate_rows = load_candidate_masks(
        candidate_path,
        kernel_names=set(kernel_map),
        presentation_names={item.name for item in presentations},
        expected_pairs=len(expected_pairs),
    )
    metrics = compute_kernel_metrics(
        masks_by_kernel,
        kernel_map=kernel_map,
        presentations=presentations,
        reference_friction_blind_bit=1 << reference_pair_index,
        target_mask=target_mask,
    )
    primary_rows = build_primary_rows(
        spec=spec,
        kernels=kernels,
        metrics=metrics,
        presentation_count=len(presentations),
        target_mask=target_mask,
    )
    marginal_rows = build_marginal_rows(
        spec=spec,
        kernels=kernels,
        metrics=metrics,
        presentation_count=len(presentations),
        target_mask=target_mask,
    )
    global_metrics = aggregate_group(
        requested=kernels,
        valid=tuple(metrics.values()),
        presentation_count=len(presentations),
        target_mask=target_mask,
    )
    promotion = promotion_decision(
        primary_rows,
        spec["main_text_promotion"],
    )

    all_18 = next(
        row
        for row in communication_summary["robust_families"]
        if str(row["family"]) == "all_18"
    )
    summary_delay = summary_aggregate_delay(communication_summary)
    friction_zero = next(
        row
        for row in marginal_rows
        if row["factor"] == "friction" and row["level"] == "0"
    )
    gates: list[dict[str, Any]] = []
    gate(
        gates,
        "all frozen input hashes match",
        len(input_manifest) == len(spec["inputs"]),
        len(input_manifest),
        len(spec["inputs"]),
    )
    gate(
        gates,
        "independently reconstructed requested grid",
        len(kernels) == int(expected["requested_kernels"]),
        len(kernels),
        int(expected["requested_kernels"]),
    )
    gate(
        gates,
        "frozen presentation design",
        len(presentations) == int(expected["presentations"]),
        len(presentations),
        int(expected["presentations"]),
    )
    gate(
        gates,
        "candidate-table row count",
        candidate_rows == int(expected["candidates"]),
        candidate_rows,
        int(expected["candidates"]),
    )
    gate(
        gates,
        "valid-kernel count",
        len(metrics) == int(expected["valid_kernels"]),
        len(metrics),
        int(expected["valid_kernels"]),
    )
    gate(
        gates,
        "each valid kernel has every presentation",
        all(
            len(raw_masks) == len(presentations)
            for raw_masks in masks_by_kernel.values()
        ),
        sorted(
            {
                len(raw_masks)
                for raw_masks in masks_by_kernel.values()
            }
        ),
        [len(presentations)],
    )
    gate(
        gates,
        "primary cells partition requested and valid kernels",
        (
            sum(int(row["requested_kernels"]) for row in primary_rows)
            == len(kernels)
            and sum(int(row["valid_kernels"]) for row in primary_rows)
            == len(metrics)
        ),
        {
            "requested": sum(
                int(row["requested_kernels"]) for row in primary_rows
            ),
            "valid": sum(int(row["valid_kernels"]) for row in primary_rows),
        },
        {"requested": len(kernels), "valid": len(metrics)},
    )
    gate(
        gates,
        "all-18 robust-nonzero reconciliation",
        (
            int(global_metrics["robust_nonzero_kernels"])
            == int(expected["all_18_robust_nonzero_kernels"])
            == int(all_18["robust_nonzero_kernels"])
        ),
        int(global_metrics["robust_nonzero_kernels"]),
        int(expected["all_18_robust_nonzero_kernels"]),
    )
    gate(
        gates,
        "all-18 robust-full reconciliation",
        (
            int(global_metrics["robust_full_kernels"])
            == int(expected["all_18_robust_full_kernels"])
            == int(all_18["robust_full_kernels"])
        ),
        int(global_metrics["robust_full_kernels"]),
        int(expected["all_18_robust_full_kernels"]),
    )
    gate(
        gates,
        "aggregate delay reconciliation",
        (
            math.isclose(
                float(global_metrics["mean_delay_delta"]),
                float(expected["aggregate_delay_delta"]),
                rel_tol=0.0,
                abs_tol=1e-15,
            )
            and math.isclose(
                float(global_metrics["mean_delay_delta"]),
                summary_delay,
                rel_tol=0.0,
                abs_tol=1e-15,
            )
        ),
        float(global_metrics["mean_delay_delta"]),
        float(expected["aggregate_delay_delta"]),
    )
    gate(
        gates,
        "friction-zero reference/friction-blind negative control",
        math.isclose(
            float(
                friction_zero[
                    "reference_friction_blind_candidate_rate"
                ]
            ),
            0.0,
            rel_tol=0.0,
            abs_tol=0.0,
        ),
        float(
            friction_zero[
                "reference_friction_blind_candidate_rate"
            ]
        ),
        0.0,
    )
    failed = [item for item in gates if item["status"] != "PASS"]
    if failed:
        raise RuntimeError(
            "acceptance gates failed: "
            + ", ".join(str(item["gate"]) for item in failed)
        )

    spec_hash = sha256_file(spec_path)
    script_hash = sha256_file(Path(__file__).resolve())
    hashes = {
        "spec_sha256": spec_hash,
        "script_sha256": script_hash,
        "inputs": input_manifest,
    }
    summary = {
        "schema_version": 1,
        "analysis_id": spec["analysis_id"],
        "status": "PASS_saved_output_parameter_heterogeneity",
        "analysis_question": spec["analysis_question"],
        "claim_scope": spec["claim_scope"],
        "design": {
            "primary_stratification": spec["primary_stratification"],
            "secondary_marginals": spec["secondary_marginals"],
            "frozen_metrics": spec["frozen_metrics"],
            "anti_data_mining_boundaries": spec[
                "anti_data_mining_boundaries"
            ],
        },
        "global": global_metrics,
        "primary_strata": primary_rows,
        "promotion": promotion,
        "acceptance_gates": gates,
        "hashes": hashes,
    }
    manifest = {
        "schema_version": 1,
        "analysis_id": spec["analysis_id"],
        "saved_output_only": True,
        "simulator_imported_or_called": False,
        "spec": {
            "relative_path": spec_path.relative_to(PROJECT_ROOT).as_posix(),
            "bytes": spec_path.stat().st_size,
            "sha256": spec_hash,
        },
        "script": {
            "relative_path": (
                Path(__file__).resolve().relative_to(PROJECT_ROOT).as_posix()
            ),
            "bytes": Path(__file__).resolve().stat().st_size,
            "sha256": script_hash,
        },
        "inputs": input_manifest,
    }

    output.mkdir(parents=True)
    write_csv(output / "strata_friction_speed.csv", primary_rows)
    write_csv(output / "marginal_parameter_effects.csv", marginal_rows)
    write_json(output / "summary.json", summary)
    write_json(output / "input_manifest.json", manifest)
    write_markdown(
        path=output / "PARAMETER_HETEROGENEITY.md",
        spec=spec,
        primary_rows=primary_rows,
        global_metrics=global_metrics,
        promotion=promotion,
        hashes=hashes,
        gates=gates,
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "output": str(output),
                "acceptance_gates": {
                    "passed": len(gates),
                    "failed": 0,
                },
                "main_text_promotion_pass": promotion[
                    "main_text_promotion_pass"
                ],
                "primary_range": promotion["primary_range"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
