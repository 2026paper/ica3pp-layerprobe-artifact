"""Independent structural checks for saved preflight and communication outputs."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--communication", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)

    checks: list[dict[str, object]] = []

    preflight_summary = json.loads(
        (args.preflight / "summary.json").read_text(encoding="utf-8")
    )
    with (args.preflight / "runs.csv").open(newline="", encoding="utf-8-sig") as handle:
        runs = list(csv.DictReader(handle))
    checks.append(
        {
            "check": "preflight_row_count",
            "pass": len(runs) == int(preflight_summary["run_count"]),
            "observed": len(runs),
            "expected": int(preflight_summary["run_count"]),
        }
    )

    digest_groups: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    metric_errors: list[str] = []
    for row in runs:
        key = (row["study"], row["case"], row["repeat"])
        digest_groups[key].add(row["digest"])
        kernels = int(row["kernel_count"])
        presentations = int(row["presentation_count"])
        valid = int(row["valid_kernels"])
        candidates = int(row["candidates"])
        graph_builds = int(row["graph_builds"])
        if candidates != valid * presentations:
            metric_errors.append(f"{key}: candidates != valid * presentations")
        expected_graph_builds = (
            kernels * presentations if row["method"] == "flat" else kernels
        )
        if graph_builds != expected_graph_builds:
            metric_errors.append(f"{key}: unexpected graph_builds for {row['method']}")
    bad_digest_groups = [key for key, values in digest_groups.items() if len(values) != 1]
    checks.append(
        {
            "check": "semantic_digest_consistency",
            "pass": not bad_digest_groups,
            "groups_checked": len(digest_groups),
            "bad_groups": bad_digest_groups,
        }
    )
    checks.append(
        {
            "check": "preflight_metric_invariants",
            "pass": not metric_errors,
            "errors": metric_errors,
        }
    )

    communication_summary = json.loads(
        (args.communication / "summary.json").read_text(encoding="utf-8")
    )
    signature_count = 0
    kernels_seen: set[str] = set()
    presentations_seen: set[str] = set()
    masks_by_presentation: dict[str, list[int]] = defaultdict(list)
    signature_errors: list[str] = []
    with gzip.open(
        args.communication / "candidate_signatures.csv.gz",
        "rt",
        newline="",
        encoding="utf-8",
    ) as handle:
        for row_number, row in enumerate(csv.DictReader(handle), start=2):
            signature_count += 1
            kernels_seen.add(row["kernel"])
            presentations_seen.add(row["presentation"])
            mask = int(row["signature_mask"])
            pairs = int(row["pairs_separated"])
            if not 0 <= mask < 64:
                signature_errors.append(f"row {row_number}: mask outside six-pair universe")
            if pairs != mask.bit_count():
                signature_errors.append(f"row {row_number}: bit count mismatch")
            masks_by_presentation[row["presentation"]].append(mask)
    expected_shape = (
        int(communication_summary["candidates"]),
        int(communication_summary["valid_kernels"]),
        int(communication_summary["presentations"]),
    )
    observed_shape = (signature_count, len(kernels_seen), len(presentations_seen))
    checks.append(
        {
            "check": "compressed_signature_shape",
            "pass": observed_shape == expected_shape and not signature_errors,
            "observed": observed_shape,
            "expected": expected_shape,
            "errors": signature_errors[:20],
        }
    )

    condition_errors: list[str] = []
    condition_lookup = {
        row["presentation"]: row
        for row in communication_summary["presentation_conditions"]
    }
    for presentation, masks in masks_by_presentation.items():
        expected = condition_lookup[presentation]
        observed_mean = sum(mask.bit_count() for mask in masks) / len(masks)
        observed_full = sum(mask == 63 for mask in masks) / len(masks)
        observed_nonzero = sum(mask != 0 for mask in masks) / len(masks)
        comparisons = (
            ("mean", observed_mean, float(expected["mean_pairs_separated"])),
            ("full", observed_full, float(expected["full_separation_rate"])),
            ("nonzero", observed_nonzero, float(expected["nonzero_rate"])),
        )
        for label, observed, target in comparisons:
            if not math.isclose(observed, target, rel_tol=0, abs_tol=1e-12):
                condition_errors.append(
                    f"{presentation}/{label}: observed {observed}, expected {target}"
                )
    checks.append(
        {
            "check": "communication_aggregate_recalculation",
            "pass": not condition_errors,
            "conditions_checked": len(masks_by_presentation),
            "errors": condition_errors,
        }
    )

    overall_pass = all(bool(check["pass"]) for check in checks)
    report = {
        "verified_at": datetime.now().astimezone().isoformat(),
        "overall_pass": overall_pass,
        "checks": checks,
    }
    (output / "verification_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    markdown = [
        "# 本机输出独立结构校验",
        "",
        f"总状态：{'PASS' if overall_pass else 'FAIL'}",
        "",
    ]
    for check in checks:
        markdown.append(
            f"- {'PASS' if check['pass'] else 'FAIL'}：`{check['check']}`"
        )
    markdown.extend(
        [
            "",
            "该校验器不调用模拟器；它重新读取 CSV、压缩候选签名和 JSON 汇总，",
            "检查运行计数、语义哈希组、工作量不变量、六模型对掩码及传播条件聚合值。",
            "它属于结构和一致性检查，不替代科学模型正确性审查。",
            "",
        ]
    )
    (output / "VERIFICATION_REPORT.md").write_text(
        "\n".join(markdown), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=True, indent=2))
    if not overall_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
