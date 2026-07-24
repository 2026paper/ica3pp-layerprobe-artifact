"""Analyze how communication-layer choices affect behavioral separability."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import statistics
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from layerprobe.evaluator import MODEL_PAIRS, minimum_cover, reduce_signature_frontier, run_factorized
from layerprobe.model import PresentationSpec
from layerprobe.workloads import make_kernels, make_presentations


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_source_tree(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*.py") if item.is_file()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        data = path.read_bytes()
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


def save_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def coverage_fields(masks: list[int]) -> dict[str, object]:
    target = (1 << len(MODEL_PAIRS)) - 1
    count = len(masks)
    fields: dict[str, object] = {
        "candidate_count": count,
        "nonzero_rate": sum(mask != 0 for mask in masks) / count,
        "full_separation_rate": sum(mask == target for mask in masks) / count,
        "mean_pairs_separated": statistics.fmean(mask.bit_count() for mask in masks),
        "unique_signatures": len(set(masks)),
    }
    for index, pair in enumerate(MODEL_PAIRS):
        fields[f"pair_{pair[0]}__{pair[1]}_rate"] = (
            sum(bool(mask & (1 << index)) for mask in masks) / count
        )
    return fields


def exact_suite(signatures: dict[str, int]) -> tuple[str, ...] | None:
    target = (1 << len(MODEL_PAIRS)) - 1
    return minimum_cover(reduce_signature_frontier(signatures), target)


def family_members(
    presentations: tuple[PresentationSpec, ...],
) -> dict[str, tuple[PresentationSpec, ...]]:
    return {
        "all_18": presentations,
        "immediate_9": tuple(item for item in presentations if item.delay == 0),
        "delayed_9": tuple(item for item in presentations if item.delay == 1),
        "no_hidden_8": tuple(
            item
            for item in presentations
            if item.speed_mode != "hidden" and item.distance_mode != "hidden"
        ),
        "no_exact_8": tuple(
            item
            for item in presentations
            if item.speed_mode != "exact" and item.distance_mode != "exact"
        ),
        "at_least_one_hidden_10": tuple(
            item
            for item in presentations
            if item.speed_mode == "hidden" or item.distance_mode == "hidden"
        ),
        "not_both_exact_16": tuple(
            item
            for item in presentations
            if not (item.speed_mode == "exact" and item.distance_mode == "exact")
        ),
    }


def write_markdown(
    *,
    elapsed: float,
    kernel_count: int,
    candidate_count: int,
    presentation_rows: list[dict[str, object]],
    factor_rows: list[dict[str, object]],
    robust_rows: list[dict[str, object]],
    delay_rows: list[dict[str, object]],
    global_suite: tuple[str, ...] | None,
    path: Path,
) -> None:
    best = max(presentation_rows, key=lambda row: float(row["mean_pairs_separated"]))
    worst = min(presentation_rows, key=lambda row: float(row["mean_pairs_separated"]))
    delay_same = sum(int(row["same_count"]) for row in delay_rows)
    delay_total = sum(int(row["kernel_count"]) for row in delay_rows)
    lines = [
        "# 信息呈现层预分析（无人工实验）",
        "",
        f"本次分析覆盖 {kernel_count} 个机制参数组合、18 种信息呈现和 {candidate_count} 个有效候选，",
        f"计算耗时 {elapsed:.3f} 秒。全部分析基于确定性代理模型，不代表真实受众效果。",
        "",
        "## 可直接复用的初步发现",
        "",
        f"- 平均可区分模型对最多的条件是 `{best['presentation']}`：{float(best['mean_pairs_separated']):.3f}/6。",
        f"- 平均可区分模型对最少的条件是 `{worst['presentation']}`：{float(worst['mean_pairs_separated']):.3f}/6。",
        f"- 全部候选的精确最小覆盖套件大小为 {None if global_suite is None else len(global_suite)}；该数值仅针对当前有限模板。",
        f"- 延迟开关在 {delay_same}/{delay_total} 个机制—呈现基组上未改变签名；详细改善、退化和混合变化见 `delay_effects.csv`。",
        "",
        "## 呈现因素汇总",
        "",
        "| 因素 | 水平 | 候选数 | 平均可区分模型对 | 全区分率 | 非零区分率 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in factor_rows:
        lines.append(
            f"| {row['factor']} | {row['level']} | {row['candidate_count']} | "
            f"{float(row['mean_pairs_separated']):.3f} | "
            f"{100 * float(row['full_separation_rate']):.2f}% | "
            f"{100 * float(row['nonzero_rate']):.2f}% |"
        )
    lines.extend(
        [
            "",
            "## 跨呈现稳健性",
            "",
            "这里的“稳健签名”取同一机制在一个呈现家族中的签名交集：只有在该家族每一种呈现下都能区分的模型对才保留。",
            "",
            "| 呈现家族 | 条件数 | 至少稳健区分一对的机制 | 可稳健区分全部六对的机制 | 最小稳健套件 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in robust_rows:
        lines.append(
            f"| {row['family']} | {row['presentation_count']} | {row['robust_nonzero_kernels']} | "
            f"{row['robust_full_kernels']} | {row['robust_minimum_suite_size']} |"
        )
    lines.extend(
        [
            "",
            "## 论文解释边界",
            "",
            "- 这组结果适合支撑“传播层设计会改变计算模型下的行为可区分性”这一计算性命题。",
            "- 它不能支撑“某种界面让真实用户学得更好”“能诊断真实误解”或“具有更高传播效果”等人类受众结论。",
            "- 下一台电脑应加入更困难的行为模型、信息预算和稳健性约束，避免当前最小套件退化为单任务。",
            "- `candidate_signatures.csv.gz` 已保存全部候选签名，后续统计分析无需重新执行状态模拟。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kernels", type=int, default=20000)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)

    kernels = make_kernels(args.kernels)
    presentations = make_presentations(18)
    presentation_by_name = {item.name: item for item in presentations}
    started = time.perf_counter()
    result = run_factorized(kernels, presentations, workers=args.workers)
    elapsed = time.perf_counter() - started

    by_presentation: dict[str, dict[str, int]] = defaultdict(dict)
    by_kernel: dict[str, dict[str, int]] = defaultdict(dict)
    with gzip.open(output / "candidate_signatures.csv.gz", "wt", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("kernel", "presentation", "signature_mask", "pairs_separated"),
        )
        writer.writeheader()
        for candidate, mask in sorted(result.candidate_signatures.items()):
            kernel_name, presentation_name = candidate.split("::", maxsplit=1)
            by_presentation[presentation_name][kernel_name] = mask
            by_kernel[kernel_name][presentation_name] = mask
            writer.writerow(
                {
                    "kernel": kernel_name,
                    "presentation": presentation_name,
                    "signature_mask": mask,
                    "pairs_separated": mask.bit_count(),
                }
            )

    presentation_rows: list[dict[str, object]] = []
    for presentation in presentations:
        signatures = by_presentation[presentation.name]
        suite = exact_suite(
            {f"{kernel}::{presentation.name}": mask for kernel, mask in signatures.items()}
        )
        presentation_rows.append(
            {
                "presentation": presentation.name,
                "speed_mode": presentation.speed_mode,
                "distance_mode": presentation.distance_mode,
                "delay": presentation.delay,
                **coverage_fields(list(signatures.values())),
                "minimum_suite_size": None if suite is None else len(suite),
                "minimum_suite": " | ".join(suite or ()),
            }
        )
    save_rows(output / "presentation_conditions.csv", presentation_rows)

    factor_rows: list[dict[str, object]] = []
    for factor, levels in (
        ("speed_mode", ("exact", "coarse", "hidden")),
        ("distance_mode", ("exact", "coarse", "hidden")),
        ("delay", (0, 1)),
    ):
        for level in levels:
            matching = [
                presentation
                for presentation in presentations
                if getattr(presentation, factor) == level
            ]
            masks = [
                by_presentation[presentation.name][kernel]
                for presentation in matching
                for kernel in by_presentation[presentation.name]
            ]
            factor_rows.append(
                {"factor": factor, "level": level, **coverage_fields(masks)}
            )
    save_rows(output / "factor_effects.csv", factor_rows)

    target = (1 << len(MODEL_PAIRS)) - 1
    robust_rows: list[dict[str, object]] = []
    for family, members in family_members(presentations).items():
        member_names = tuple(item.name for item in members)
        robust_signatures: dict[str, int] = {}
        union_signatures: dict[str, int] = {}
        for kernel, signatures in by_kernel.items():
            robust_mask = target
            union_mask = 0
            for name in member_names:
                robust_mask &= signatures[name]
                union_mask |= signatures[name]
            robust_signatures[kernel] = robust_mask
            union_signatures[kernel] = union_mask
        robust_suite = exact_suite(robust_signatures)
        union_suite = exact_suite(union_signatures)
        robust_rows.append(
            {
                "family": family,
                "presentation_count": len(members),
                "kernel_count": len(robust_signatures),
                "robust_nonzero_kernels": sum(mask != 0 for mask in robust_signatures.values()),
                "robust_full_kernels": sum(mask == target for mask in robust_signatures.values()),
                "robust_mean_pairs": statistics.fmean(
                    mask.bit_count() for mask in robust_signatures.values()
                ),
                "robust_minimum_suite_size": None if robust_suite is None else len(robust_suite),
                "robust_minimum_suite": " | ".join(robust_suite or ()),
                "union_minimum_suite_size": None if union_suite is None else len(union_suite),
                "union_minimum_suite": " | ".join(union_suite or ()),
            }
        )
    save_rows(output / "robust_families.csv", robust_rows)

    delay_rows: list[dict[str, object]] = []
    for speed_mode in ("exact", "coarse", "hidden"):
        for distance_mode in ("exact", "coarse", "hidden"):
            immediate = next(
                item
                for item in presentations
                if item.speed_mode == speed_mode
                and item.distance_mode == distance_mode
                and item.delay == 0
            )
            delayed = next(
                item
                for item in presentations
                if item.speed_mode == speed_mode
                and item.distance_mode == distance_mode
                and item.delay == 1
            )
            counts = defaultdict(int)
            deltas: list[int] = []
            for kernel in by_kernel:
                left = by_kernel[kernel][immediate.name]
                right = by_kernel[kernel][delayed.name]
                deltas.append(right.bit_count() - left.bit_count())
                if left == right:
                    counts["same"] += 1
                elif right | left == right:
                    counts["improved"] += 1
                elif right | left == left:
                    counts["degraded"] += 1
                else:
                    counts["mixed"] += 1
            delay_rows.append(
                {
                    "speed_mode": speed_mode,
                    "distance_mode": distance_mode,
                    "kernel_count": len(by_kernel),
                    "same_count": counts["same"],
                    "improved_count": counts["improved"],
                    "degraded_count": counts["degraded"],
                    "mixed_count": counts["mixed"],
                    "mean_pair_delta_delayed_minus_immediate": statistics.fmean(deltas),
                }
            )
    save_rows(output / "delay_effects.csv", delay_rows)

    summary = {
        "status": "computational_preanalysis_not_human_effect_evidence",
        "generated_at": datetime.now().astimezone().isoformat(),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "core_source_sha256": sha256_source_tree(PROJECT_ROOT / "src"),
        "elapsed_s": elapsed,
        "requested_kernels": args.kernels,
        "valid_kernels": len(result.valid_kernels),
        "presentations": len(presentations),
        "candidates": len(result.candidate_signatures),
        "model_pairs": MODEL_PAIRS,
        "global_minimum_suite": result.minimum_suite,
        "presentation_conditions": presentation_rows,
        "factor_effects": factor_rows,
        "robust_families": robust_rows,
        "delay_effects": delay_rows,
        "metrics": result.metrics,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_markdown(
        elapsed=elapsed,
        kernel_count=args.kernels,
        candidate_count=len(result.candidate_signatures),
        presentation_rows=presentation_rows,
        factor_rows=factor_rows,
        robust_rows=robust_rows,
        delay_rows=delay_rows,
        global_suite=result.minimum_suite,
        path=output / "COMMUNICATION_ANALYSIS.md",
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "elapsed_s": elapsed,
                "candidates": len(result.candidate_signatures),
                "global_minimum_suite": result.minimum_suite,
            },
            ensure_ascii=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
