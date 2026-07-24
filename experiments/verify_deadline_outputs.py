"""Structural and arithmetic audit for deadline-runner paper outputs.

This verifier never calls the LayerProbe simulator.  It checks saved rows,
digests, work-count invariants, the compressed signature table, and selected
communication aggregates.  Passing it is not an independent semantic proof.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def add_check(
    checks: list[dict[str, object]],
    label: str,
    passed: bool,
    detail: object,
) -> None:
    checks.append(
        {
            "label": label,
            "status": "PASS" if passed else "FAIL",
            "detail": detail,
        }
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_frozen_snapshot(
    run_dir: Path,
    snapshot_root: Path,
    checks: list[dict[str, object]],
) -> None:
    manifest_path = snapshot_root / "SNAPSHOT_MANIFEST.json"
    if not manifest_path.is_file():
        add_check(
            checks,
            "frozen source manifest",
            False,
            {"missing": str(manifest_path)},
        )
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    file_failures: list[dict[str, object]] = []
    for item in manifest.get("files", []):
        relative = str(item["path"])
        path = snapshot_root / Path(relative)
        if not path.is_file():
            file_failures.append({"path": relative, "failure": "missing"})
            continue
        observed_bytes = path.stat().st_size
        observed_hash = sha256_file(path)
        if (
            observed_bytes != int(item["bytes"])
            or observed_hash != str(item["sha256"])
        ):
            file_failures.append(
                {
                    "path": relative,
                    "expected_bytes": item["bytes"],
                    "observed_bytes": observed_bytes,
                    "expected_sha256": item["sha256"],
                    "observed_sha256": observed_hash,
                }
            )
    add_check(
        checks,
        "frozen source manifest",
        not file_failures,
        {
            "manifest": str(manifest_path),
            "files": len(manifest.get("files", [])),
            "failures": file_failures,
        },
    )

    source_files = sorted((snapshot_root / "src").rglob("*.py"))
    fingerprint_files = source_files + [
        snapshot_root / "experiments" / "deadline_runner.py",
        snapshot_root / "experiments" / "deadline_profile_8c32g.json",
    ]
    digest = hashlib.sha256()
    labels: list[str] = []
    for path in fingerprint_files:
        label = str(path.relative_to(snapshot_root))
        labels.append(label)
        digest.update(label.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    observed_fingerprint = digest.hexdigest()
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    expected_fingerprint = str(
        manifest.get("deadline_code_fingerprint_sha256", "")
    )
    add_check(
        checks,
        "runner frozen code fingerprint",
        bool(expected_fingerprint)
        and observed_fingerprint == expected_fingerprint
        and metadata.get("code_fingerprint_sha256") == expected_fingerprint
        and metadata.get("fingerprint_files") == labels,
        {
            "manifest": expected_fingerprint,
            "recomputed": observed_fingerprint,
            "metadata": metadata.get("code_fingerprint_sha256"),
            "metadata_files": metadata.get("fingerprint_files"),
            "snapshot_files": labels,
        },
    )

    frozen_config = json.loads(
        (run_dir / "frozen_config.json").read_text(encoding="utf-8")
    )
    snapshot_config = json.loads(
        (
            snapshot_root
            / "experiments"
            / "deadline_profile_8c32g.json"
        ).read_text(encoding="utf-8")
    )
    add_check(
        checks,
        "runner frozen configuration matches snapshot",
        frozen_config == snapshot_config,
        {
            "frozen_config": str(run_dir / "frozen_config.json"),
            "snapshot_config": str(
                snapshot_root
                / "experiments"
                / "deadline_profile_8c32g.json"
            ),
        },
    )


def verify_runner(
    run_dir: Path,
    checks: list[dict[str, object]],
    frozen_snapshot_root: Path | None,
) -> None:
    required = (
        "runs.csv",
        "metadata.json",
        "frozen_config.json",
        "progress.json",
        "semantic_checks.json",
        "summary.json",
        "SUMMARY.md",
    )
    missing = [name for name in required if not (run_dir / name).is_file()]
    add_check(checks, "runner required files", not missing, {"missing": missing})
    if missing:
        return
    if frozen_snapshot_root is not None:
        verify_frozen_snapshot(run_dir, frozen_snapshot_root, checks)

    rows = read_csv(run_dir / "runs.csv")
    progress = json.loads((run_dir / "progress.json").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    semantic = json.loads(
        (run_dir / "semantic_checks.json").read_text(encoding="utf-8")
    )

    job_ids = [row["job_id"] for row in rows]
    add_check(
        checks,
        "unique runner job IDs",
        len(job_ids) == len(set(job_ids)),
        {"rows": len(job_ids), "unique": len(set(job_ids))},
    )
    add_check(
        checks,
        "runner completed plan",
        progress.get("status") == "completed"
        and int(progress.get("completed_jobs", -1)) == len(rows)
        and int(progress.get("planned_jobs", -2)) == len(rows),
        progress,
    )
    add_check(
        checks,
        "summary run count",
        int(summary.get("run_count", -1)) == len(rows),
        {"summary": summary.get("run_count"), "rows": len(rows)},
    )
    add_check(
        checks,
        "summary full-result status",
        summary.get("status") == "paper_candidate_results_semantics_checked",
        summary.get("status"),
    )

    invariant_failures: list[dict[str, object]] = []
    for row in rows:
        method = row["method"]
        kernels = int(row["kernel_count"])
        presentations = int(row["presentation_count"])
        valid = int(row["valid_kernels"])
        candidates = int(row["candidates"])
        graph_builds = int(row["graph_builds"])
        policy = int(row["policy_calls"])
        transitions = int(row["transition_calls"])
        expected_graphs = (
            kernels * presentations if method == "flat" else kernels
        )
        failures: list[str] = []
        if candidates != valid * presentations:
            failures.append("candidates != valid_kernels * presentations")
        if graph_builds != expected_graphs:
            failures.append("unexpected graph_builds")
        if policy != transitions:
            failures.append("policy_calls != transition_calls")
        if not row["digest"] or len(row["digest"]) != 64:
            failures.append("invalid digest")
        if not row["peak_process_tree_rss_mb"]:
            failures.append("missing peak RSS")
        if failures:
            invariant_failures.append({"job_id": row["job_id"], "failures": failures})
    add_check(
        checks,
        "runner row invariants",
        not invariant_failures,
        {"failure_count": len(invariant_failures), "examples": invariant_failures[:5]},
    )

    comparison_digests: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for row in rows:
        if row["study"] == "capacity_scan":
            continue
        key = (row["study"], row["case"], row["repeat"])
        comparison_digests[key].add(row["digest"])
    mismatches = {
        "|".join(key): sorted(values)
        for key, values in comparison_digests.items()
        if len(values) != 1
    }
    add_check(
        checks,
        "independently regrouped semantic digests",
        not mismatches,
        {"groups": len(comparison_digests), "mismatches": mismatches},
    )
    semantic_failures = [item for item in semantic if item.get("status") != "PASS"]
    add_check(
        checks,
        "saved semantic checks",
        not semantic_failures,
        {"checks": len(semantic), "failures": semantic_failures},
    )


def verify_communication(
    communication_dir: Path,
    checks: list[dict[str, object]],
) -> None:
    required = (
        "candidate_signatures.csv.gz",
        "presentation_conditions.csv",
        "factor_effects.csv",
        "delay_effects.csv",
        "robust_families.csv",
        "summary.json",
        "COMMUNICATION_ANALYSIS.md",
    )
    missing = [name for name in required if not (communication_dir / name).is_file()]
    add_check(checks, "communication required files", not missing, {"missing": missing})
    if missing:
        return

    summary = json.loads(
        (communication_dir / "summary.json").read_text(encoding="utf-8")
    )
    presentations = int(summary["presentations"])
    valid = int(summary["valid_kernels"])
    expected_candidates = valid * presentations
    add_check(
        checks,
        "communication candidate arithmetic",
        int(summary["candidates"]) == expected_candidates,
        {
            "summary_candidates": summary["candidates"],
            "valid_x_presentations": expected_candidates,
        },
    )
    add_check(
        checks,
        "communication frozen full domain",
        int(summary["requested_kernels"]) == 24_624 and presentations == 18,
        {
            "requested_kernels": summary["requested_kernels"],
            "presentations": presentations,
        },
    )
    add_check(
        checks,
        "six declared model pairs",
        len(summary["model_pairs"]) == 6,
        {"model_pairs": len(summary["model_pairs"])},
    )

    signature_path = communication_dir / "candidate_signatures.csv.gz"
    signature_rows = 0
    invalid_masks = 0
    duplicate_candidates = 0
    seen: set[tuple[str, str]] = set()
    with gzip.open(signature_path, "rt", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            signature_rows += 1
            key = (row["kernel"], row["presentation"])
            if key in seen:
                duplicate_candidates += 1
            seen.add(key)
            mask = int(row["signature_mask"])
            separated = int(row["pairs_separated"])
            if not 0 <= mask <= 63 or separated != mask.bit_count():
                invalid_masks += 1
    add_check(
        checks,
        "compressed signature rows",
        signature_rows == expected_candidates and duplicate_candidates == 0,
        {
            "rows": signature_rows,
            "expected": expected_candidates,
            "duplicates": duplicate_candidates,
        },
    )
    add_check(
        checks,
        "signature mask invariants",
        invalid_masks == 0,
        {"invalid_rows": invalid_masks},
    )

    presentation_rows = read_csv(communication_dir / "presentation_conditions.csv")
    delay_rows = read_csv(communication_dir / "delay_effects.csv")
    robust_rows = read_csv(communication_dir / "robust_families.csv")
    add_check(
        checks,
        "communication table dimensions",
        len(presentation_rows) == 18 and len(delay_rows) == 9,
        {
            "presentation_rows": len(presentation_rows),
            "delay_rows": len(delay_rows),
            "robust_rows": len(robust_rows),
        },
    )
    all_18 = [row for row in robust_rows if row["family"] == "all_18"]
    add_check(
        checks,
        "all-18 robust family present",
        len(all_18) == 1 and int(all_18[0]["presentation_count"]) == 18,
        all_18,
    )


def write_markdown(report: dict[str, object], path: Path) -> None:
    lines = [
        "# Deadline output structural audit",
        "",
        f"Overall: **{report['overall']}**",
        "",
        "| Check | Status |",
        "|---|---|",
    ]
    for check in report["checks"]:
        lines.append(f"| {check['label']} | {check['status']} |")
    lines.extend(
        [
            "",
            "This audit does not call the simulator and is not an independent semantic proof.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--communication-dir", type=Path)
    parser.add_argument("--frozen-snapshot-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    checks: list[dict[str, object]] = []
    verify_runner(
        args.run_dir.resolve(),
        checks,
        None
        if args.frozen_snapshot_root is None
        else args.frozen_snapshot_root.resolve(),
    )
    if args.communication_dir is not None:
        verify_communication(args.communication_dir.resolve(), checks)
    overall = "PASS" if all(check["status"] == "PASS" for check in checks) else "FAIL"
    report = {
        "overall": overall,
        "generated_at": datetime.now().astimezone().isoformat(),
        "run_dir": str(args.run_dir.resolve()),
        "communication_dir": (
            None
            if args.communication_dir is None
            else str(args.communication_dir.resolve())
        ),
        "frozen_snapshot_root": (
            None
            if args.frozen_snapshot_root is None
            else str(args.frozen_snapshot_root.resolve())
        ),
        "checks": checks,
    }
    (output / "verification_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_markdown(report, output / "VERIFICATION_REPORT.md")
    print(json.dumps({"overall": overall, "checks": len(checks)}, indent=2))
    if overall != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
