"""Decompose presentation effects by agent pair and leave-one-agent-out subsets.

This is a saved-output analysis: it reads the frozen candidate-signature table
and does not re-run the simulator.  The results describe only the four declared
computational agents and must not be interpreted as human-subject evidence.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterable

from layerprobe.workloads import make_kernels


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write an empty table: {path.name}")
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def minimum_cover(
    signatures: Iterable[tuple[str, int]],
    target_mask: int,
) -> tuple[str, ...] | None:
    """Exact deterministic minimum cover for the at-most-three-bit subsets."""

    representatives: dict[int, str] = {}
    for candidate, raw_mask in sorted(signatures):
        mask = raw_mask & target_mask
        if mask:
            representatives.setdefault(mask, candidate)

    dp: dict[int, tuple[str, ...]] = {0: ()}
    for mask, candidate in sorted(
        ((mask, candidate) for mask, candidate in representatives.items()),
        key=lambda item: item[1],
    ):
        for covered, suite in list(dp.items()):
            combined = covered | mask
            proposal = suite + (candidate,)
            incumbent = dp.get(combined)
            if incumbent is None or len(proposal) < len(incumbent) or (
                len(proposal) == len(incumbent) and proposal < incumbent
            ):
                dp[combined] = proposal
    return dp.get(target_mask)


def direction(value: float, tolerance: float = 1e-15) -> str:
    if value < -tolerance:
        return "negative"
    if value > tolerance:
        return "positive"
    return "zero"


def write_markdown(
    path: Path,
    *,
    pair_rows: list[dict[str, object]],
    leave_rows: list[dict[str, object]],
    construct_rows: list[dict[str, object]],
    candidate_count: int,
    kernel_count: int,
) -> None:
    lines = [
        "# Declared-agent sensitivity analysis",
        "",
        f"This saved-output analysis covers {kernel_count:,} valid mechanisms and "
        f"{candidate_count:,} mechanism-presentation candidates.",
        "It does not re-run the simulator and does not constitute human learning, "
        "diagnostic, or communication-effect evidence.",
        "",
        "## Delay effect by declared agent pair",
        "",
        "| Pair | Candidate separation | all-18 robust kernels | "
        "Immediate rate | Delayed rate | Delta |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in pair_rows:
        lines.append(
            f"| {row['pair']} | {100 * float(row['candidate_separation_rate']):.2f}% | "
            f"{row['robust_kernel_count']} | "
            f"{100 * float(row['immediate_rate']):.2f}% | "
            f"{100 * float(row['delayed_rate']):.2f}% | "
            f"{float(row['delayed_minus_immediate_rate']):+.4f} |"
        )
    lines.extend(
        [
            "",
            "## Leave-one-agent-out robustness",
            "",
            "| Omitted agent | Retained pairs | Delay delta | all-18 robust full | "
            "all-18 robust suite |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in leave_rows:
        lines.append(
            f"| {row['omitted_agent']} | {row['retained_pair_count']} | "
            f"{float(row['mean_pair_delta_delayed_minus_immediate']):+.4f} | "
            f"{row['robust_full_kernels']} | "
            f"{row['robust_minimum_suite_size']} |"
        )
    lines.extend(
        [
            "",
            "## Construct-validity negative control",
            "",
            "The reference and friction-blind agents are definitionally equivalent "
            "when friction is zero. Their separation bit must therefore remain zero.",
            "",
            "| Friction | Separated candidates | Total candidates | Rate |",
            "|---:|---:|---:|---:|",
        ]
    )
    for row in construct_rows:
        lines.append(
            f"| {row['friction']} | {row['separated_candidates']} | "
            f"{row['candidate_count']} | "
            f"{100 * float(row['separation_rate']):.3f}% |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "- Pair decomposition tests whether the aggregate delay result is dominated "
            "by one declared pair.",
            "- Leave-one-agent-out results test directional sensitivity to removing one "
            "hand-written agent; they do not establish population robustness.",
            "- A positive pair-level delta is not evidence that delay helps people. It "
            "means only that this deterministic pair diverged more often.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--communication-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    communication_dir = args.communication_dir.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"output directory already exists: {output}")

    summary_path = communication_dir / "summary.json"
    signature_path = communication_dir / "candidate_signatures.csv.gz"
    if not summary_path.is_file() or not signature_path.is_file():
        raise FileNotFoundError("communication directory lacks summary/signature files")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    model_pairs = tuple(tuple(pair) for pair in summary["model_pairs"])
    if len(model_pairs) != 6:
        raise ValueError(f"expected six declared pairs, found {len(model_pairs)}")

    agents: list[str] = []
    for pair in model_pairs:
        for agent in pair:
            if agent not in agents:
                agents.append(agent)
    if len(agents) != 4:
        raise ValueError(f"expected four declared agents, found {len(agents)}")

    presentation_rows = summary["presentation_conditions"]
    presentation_meta = {
        row["presentation"]: {
            "speed_mode": row["speed_mode"],
            "distance_mode": row["distance_mode"],
            "delay": int(row["delay"]),
        }
        for row in presentation_rows
    }
    if len(presentation_meta) != 18:
        raise ValueError(f"expected 18 presentation conditions, found {len(presentation_meta)}")

    by_kernel: dict[str, dict[str, int]] = defaultdict(dict)
    rows_seen = 0
    with gzip.open(signature_path, "rt", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            kernel = row["kernel"]
            presentation = row["presentation"]
            if presentation not in presentation_meta:
                raise ValueError(f"unknown presentation in signature table: {presentation}")
            if presentation in by_kernel[kernel]:
                raise ValueError(f"duplicate kernel-presentation row: {kernel}::{presentation}")
            mask = int(row["signature_mask"])
            if mask < 0 or mask >= 1 << len(model_pairs):
                raise ValueError(f"signature outside six-bit universe: {mask}")
            by_kernel[kernel][presentation] = mask
            rows_seen += 1

    expected_rows = int(summary["candidates"])
    expected_kernels = int(summary["valid_kernels"])
    if rows_seen != expected_rows:
        raise ValueError(f"candidate rows {rows_seen} != summary {expected_rows}")
    if len(by_kernel) != expected_kernels:
        raise ValueError(f"valid kernels {len(by_kernel)} != summary {expected_kernels}")
    for kernel, signatures in by_kernel.items():
        if set(signatures) != set(presentation_meta):
            raise ValueError(f"incomplete presentation family for {kernel}")

    requested_kernels = int(summary["requested_kernels"])
    kernel_specs = {item.name: item for item in make_kernels(requested_kernels)}
    if not set(by_kernel).issubset(kernel_specs):
        unknown = sorted(set(by_kernel) - set(kernel_specs))
        raise ValueError(f"signature table contains unknown mechanisms: {unknown[:3]}")

    cell_pairs: list[tuple[str, str]] = []
    for speed_mode in ("exact", "coarse", "hidden"):
        for distance_mode in ("exact", "coarse", "hidden"):
            immediate = next(
                name
                for name, meta in presentation_meta.items()
                if meta["speed_mode"] == speed_mode
                and meta["distance_mode"] == distance_mode
                and meta["delay"] == 0
            )
            delayed = next(
                name
                for name, meta in presentation_meta.items()
                if meta["speed_mode"] == speed_mode
                and meta["distance_mode"] == distance_mode
                and meta["delay"] == 1
            )
            cell_pairs.append((immediate, delayed))

    pair_rows: list[dict[str, object]] = []
    pair_deltas: dict[int, list[int]] = defaultdict(list)
    pair_immediate: dict[int, list[int]] = defaultdict(list)
    pair_delayed: dict[int, list[int]] = defaultdict(list)
    for signatures in by_kernel.values():
        for immediate, delayed in cell_pairs:
            left = signatures[immediate]
            right = signatures[delayed]
            for bit_index in range(len(model_pairs)):
                before = int(bool(left & (1 << bit_index)))
                after = int(bool(right & (1 << bit_index)))
                pair_immediate[bit_index].append(before)
                pair_delayed[bit_index].append(after)
                pair_deltas[bit_index].append(after - before)

    for bit_index, pair in enumerate(model_pairs):
        deltas = pair_deltas[bit_index]
        immediate_values = pair_immediate[bit_index]
        delayed_values = pair_delayed[bit_index]
        mean_delta = statistics.fmean(deltas)
        bit = 1 << bit_index
        candidate_values = [
            int(bool(mask & bit))
            for signatures in by_kernel.values()
            for mask in signatures.values()
        ]
        robust_values = []
        for signatures in by_kernel.values():
            robust_mask = (1 << len(model_pairs)) - 1
            for mask in signatures.values():
                robust_mask &= mask
            robust_values.append(int(bool(robust_mask & bit)))
        pair_rows.append(
            {
                "pair_index": bit_index,
                "left_agent": pair[0],
                "right_agent": pair[1],
                "pair": f"{pair[0]} vs {pair[1]}",
                "candidate_count": len(candidate_values),
                "candidate_separated_count": sum(candidate_values),
                "candidate_separation_rate": statistics.fmean(candidate_values),
                "robust_kernel_count": sum(robust_values),
                "robust_kernel_rate": statistics.fmean(robust_values),
                "mechanism_presentation_bases": len(deltas),
                "immediate_rate": statistics.fmean(immediate_values),
                "delayed_rate": statistics.fmean(delayed_values),
                "delayed_minus_immediate_rate": mean_delta,
                "improved_count": sum(delta > 0 for delta in deltas),
                "degraded_count": sum(delta < 0 for delta in deltas),
                "same_count": sum(delta == 0 for delta in deltas),
                "direction": direction(mean_delta),
            }
        )

    try:
        construct_pair_index = model_pairs.index(("reference", "friction_blind"))
    except ValueError as exc:
        raise ValueError("reference/friction_blind pair is absent") from exc
    construct_bit = 1 << construct_pair_index
    construct_rows: list[dict[str, object]] = []
    for friction in sorted({kernel_specs[name].friction for name in by_kernel}):
        matching_kernels = [
            name for name in by_kernel if kernel_specs[name].friction == friction
        ]
        masks = [
            by_kernel[kernel][presentation]
            for kernel in matching_kernels
            for presentation in presentation_meta
        ]
        separated = sum(bool(mask & construct_bit) for mask in masks)
        construct_rows.append(
            {
                "friction": friction,
                "valid_kernels": len(matching_kernels),
                "candidate_count": len(masks),
                "separated_candidates": separated,
                "separation_rate": separated / len(masks),
                "expected_equivalent": friction == 0,
                "negative_control_pass": friction != 0 or separated == 0,
            }
        )

    all_presentations = tuple(sorted(presentation_meta))
    leave_rows: list[dict[str, object]] = []
    for omitted_agent in agents:
        retained_indices = tuple(
            index
            for index, pair in enumerate(model_pairs)
            if omitted_agent not in pair
        )
        target_mask = sum(1 << index for index in retained_indices)
        if len(retained_indices) != 3:
            raise ValueError(f"leave-one-out target for {omitted_agent} is malformed")

        robust_signatures: dict[str, int] = {}
        union_signatures: dict[str, int] = {}
        ordinary_signatures: list[tuple[str, int]] = []
        delay_deltas: list[int] = []
        for kernel, signatures in by_kernel.items():
            robust_mask = target_mask
            union_mask = 0
            for presentation in all_presentations:
                subset_mask = signatures[presentation] & target_mask
                robust_mask &= subset_mask
                union_mask |= subset_mask
                ordinary_signatures.append(
                    (f"{kernel}::{presentation}", subset_mask)
                )
            robust_signatures[kernel] = robust_mask
            union_signatures[kernel] = union_mask
            for immediate, delayed in cell_pairs:
                left = signatures[immediate] & target_mask
                right = signatures[delayed] & target_mask
                delay_deltas.append(right.bit_count() - left.bit_count())

        robust_suite = minimum_cover(robust_signatures.items(), target_mask)
        union_suite = minimum_cover(union_signatures.items(), target_mask)
        ordinary_suite = minimum_cover(ordinary_signatures, target_mask)
        mean_delta = statistics.fmean(delay_deltas)
        leave_rows.append(
            {
                "omitted_agent": omitted_agent,
                "retained_pair_count": len(retained_indices),
                "retained_pairs": " | ".join(
                    f"{model_pairs[index][0]}__{model_pairs[index][1]}"
                    for index in retained_indices
                ),
                "target_mask": target_mask,
                "mean_pair_delta_delayed_minus_immediate": mean_delta,
                "delay_direction": direction(mean_delta),
                "robust_nonzero_kernels": sum(
                    mask != 0 for mask in robust_signatures.values()
                ),
                "robust_full_kernels": sum(
                    mask == target_mask for mask in robust_signatures.values()
                ),
                "robust_mean_pairs": statistics.fmean(
                    mask.bit_count() for mask in robust_signatures.values()
                ),
                "robust_minimum_suite_size": (
                    None if robust_suite is None else len(robust_suite)
                ),
                "robust_minimum_suite": " | ".join(robust_suite or ()),
                "union_minimum_suite_size": (
                    None if union_suite is None else len(union_suite)
                ),
                "union_minimum_suite": " | ".join(union_suite or ()),
                "ordinary_minimum_suite_size": (
                    None if ordinary_suite is None else len(ordinary_suite)
                ),
                "ordinary_minimum_suite": " | ".join(ordinary_suite or ()),
            }
        )

    output.mkdir(parents=True, exist_ok=False)
    save_csv(output / "pair_delay_sensitivity.csv", pair_rows)
    save_csv(output / "leave_one_agent_out.csv", leave_rows)
    save_csv(output / "construct_negative_control.csv", construct_rows)

    payload = {
        "status": "saved_output_agent_sensitivity_not_human_effect_evidence",
        "generated_at": datetime.now().astimezone().isoformat(),
        "communication_dir": str(communication_dir),
        "source_sha256": {
            "summary.json": sha256(summary_path),
            "candidate_signatures.csv.gz": sha256(signature_path),
            "script": sha256(Path(__file__).resolve()),
        },
        "valid_kernels": len(by_kernel),
        "presentations": len(presentation_meta),
        "candidates": rows_seen,
        "agents": agents,
        "model_pairs": model_pairs,
        "pair_delay_sensitivity": pair_rows,
        "leave_one_agent_out": leave_rows,
        "construct_negative_control": construct_rows,
        "gates": {
            "all_leave_one_out_delay_directions_negative": all(
                row["delay_direction"] == "negative" for row in leave_rows
            ),
            "pair_direction_counts": {
                label: sum(row["direction"] == label for row in pair_rows)
                for label in ("negative", "zero", "positive")
            },
            "all_leave_one_out_robust_suites_exist": all(
                row["robust_minimum_suite_size"] is not None for row in leave_rows
            ),
            "friction_zero_negative_control_pass": all(
                row["negative_control_pass"] for row in construct_rows
            ),
        },
        "interpretation": (
            "Finite-domain sensitivity for four declared computational agents; "
            "not evidence about human learning or communication effectiveness."
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_markdown(
        output / "AGENT_SENSITIVITY.md",
        pair_rows=pair_rows,
        leave_rows=leave_rows,
        construct_rows=construct_rows,
        candidate_count=rows_seen,
        kernel_count=len(by_kernel),
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "valid_kernels": len(by_kernel),
                "candidates": rows_seen,
                "gates": payload["gates"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
