"""Build the review-response performance evidence figure.

The figure has one narrative: decomposing repeated work is useful only when
the saved semantic work is expensive enough, and the remaining single-host
turnaround still depends on parallel scheduling.

Panels
------
(a) Flat-P8, KernelMemo-P8, and LayerProbe-P8 elapsed distributions.  Every
    dot is one technical repeat, faint lines join the same repeat, and the
    diamond/short bar marks the median.
(b) The paired ratio ``KernelMemo-P8 elapsed / LayerProbe-P8 elapsed`` for the
    original policy (depth zero) and finite-depth deliberative policies.
    Values above one favor LayerProbe.  Every raw paired repeat is visible;
    whiskers are 95% *technical bootstrap* intervals for the median, not
    population-level inferential confidence intervals.
(c) Single-host strong-scaling medians from one to sixteen workers.  The
    guide is linear scaling only through the physical-core endpoint; the
    shaded region is SMT.
(d) Scheduler elapsed time against load imbalance (maximum worker load divided
    by mean worker load; one is ideal). Every technical repeat and the
    bivariate median are visible, so balance cannot be mistaken for turnaround.

This script intentionally contains no fallback data and no pre-filled result
values.  It fails closed if an input is incomplete, has fewer/more than the
required paired repeats, has a semantic mismatch, disagrees with its summary,
or cannot resolve Times New Roman exactly.

Example
-------
python experiments/build_review_response_figure.py \
  --method-runs results/review_method_ladder_n10/runs.csv \
  --method-summary results/review_method_ladder_n10/summary.json \
  --deliberative-runs results/deliberative_n10/runs.csv \
  --deliberative-summary results/deliberative_n10/summary.json \
  --scaling-runs results/deadline_paper_corrected/runs.csv \
  --scaling-summary results/deadline_paper_corrected/summary.json \
  --scheduler-runs results/scheduler_n10/runs.csv \
  --scheduler-summary results/scheduler_n10/summary.json \
  --output-dir results/review_response_figure
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, findfont
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.ticker import MaxNLocator
from PIL import Image, ImageOps


FIGURE_WIDTH_IN = 4.72
FIGURE_HEIGHT_IN = 4.20
MIN_FONT_PT = 6.0
DEFAULT_DPI = 300
SCRIPT_VERSION = "2026-07-25-review-response-v4-scheduler-tradeoff"

# Okabe-Ito colors.  Categories also differ by marker shape so that the figure
# remains interpretable in the generated grayscale preview.
BLUE = "#0072B2"
ORANGE = "#E69F00"
PURPLE = "#CC79A7"
SKY = "#56B4E9"
BLACK = "#1A1A1A"
MID_GRAY = "#6B6B6B"
LIGHT_GRAY = "#D7D7D7"
GRID_GRAY = "#E2E2E2"
SMT_GRAY = "#F0F0F0"

METHODS = ("flat_parallel", "kernel_memo_parallel", "factorized")
METHOD_LABELS = {
    "flat_parallel": "Flat-P8",
    "kernel_memo_parallel": "KernelMemo-P8",
    "factorized": "LayerProbe-P8",
}
METHOD_COLORS = {
    "flat_parallel": BLUE,
    "kernel_memo_parallel": ORANGE,
    "factorized": PURPLE,
}
METHOD_MARKERS = {
    "flat_parallel": "o",
    "kernel_memo_parallel": "s",
    "factorized": "^",
}

DELIBERATIVE_METHODS = ("kernel_memo_p8", "layerprobe_p8")

SCHEDULES = (
    "current_chunksize",
    "fine_chunksize_1",
    "static_contiguous",
)
SCHEDULE_LABELS = {
    "current_chunksize": "Current\nchunks",
    "fine_chunksize_1": "Fine\nchunks=1",
    "static_contiguous": "Static\nblocks",
}
SCHEDULE_COLORS = {
    "current_chunksize": BLUE,
    "fine_chunksize_1": ORANGE,
    "static_contiguous": PURPLE,
}
SCHEDULE_MARKERS = {
    "current_chunksize": "o",
    "fine_chunksize_1": "s",
    "static_contiguous": "^",
}


@dataclass(frozen=True)
class MethodPanelData:
    case: str
    repeats: tuple[int, ...]
    elapsed_by_method: Mapping[str, tuple[float, ...]]
    medians: Mapping[str, float]


@dataclass(frozen=True)
class DeliberativeDepthData:
    depth: int
    repeats: tuple[int, ...]
    ratios: tuple[float, ...]
    median: float
    ci95_low: float
    ci95_high: float


@dataclass(frozen=True)
class ScalingPoint:
    workers: int
    repeats: int
    median_s: float
    speedup: float


@dataclass(frozen=True)
class ScalingPanelData:
    physical_cores: int
    logical_cores: int
    points: tuple[ScalingPoint, ...]


@dataclass(frozen=True)
class SchedulerPanelData:
    repeats: tuple[int, ...]
    imbalance_by_schedule: Mapping[str, tuple[float, ...]]
    elapsed_by_schedule: Mapping[str, tuple[float, ...]]
    imbalance_medians: Mapping[str, float]
    elapsed_medians: Mapping[str, float]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_csv(path: Path, label: str) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{label} is empty: {path}")
    return rows


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{label} must contain a JSON object: {path}")
    return payload


def _require_columns(
    rows: Sequence[Mapping[str, str]],
    required: Iterable[str],
    label: str,
) -> None:
    available = set(rows[0])
    missing = sorted(set(required) - available)
    if missing:
        raise ValueError(f"{label} lacks required columns: {missing}")


def _as_int(value: Any, label: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not an integer: {value!r}") from exc
    return result


def _as_float(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not numeric: {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} is not finite: {value!r}")
    return result


def _assert_close(
    observed: float,
    recorded: float,
    label: str,
    *,
    relative_tolerance: float = 1e-9,
) -> None:
    scale = max(1.0, abs(observed), abs(recorded))
    if abs(observed - recorded) > relative_tolerance * scale:
        raise ValueError(
            f"{label} disagrees with raw data: "
            f"observed={observed!r}, recorded={recorded!r}"
        )


def _summary_semantic_check(
    summary: Mapping[str, Any],
    *,
    study: str,
    case: str | None = None,
) -> None:
    checks = summary.get("semantic_checks")
    if not isinstance(checks, list):
        raise ValueError("method summary lacks semantic_checks")
    selected = [
        item
        for item in checks
        if isinstance(item, dict)
        and item.get("study") == study
        and (case is None or item.get("case") == case)
    ]
    if not selected:
        raise ValueError(
            f"method summary has no semantic checks for {study}/{case}"
        )
    failed = [item for item in selected if item.get("status") != "PASS"]
    if failed:
        raise ValueError(
            f"method summary contains failed semantic checks for {study}/{case}"
        )


def _prepare_method_panel(
    rows: Sequence[Mapping[str, str]],
    summary: Mapping[str, Any],
    required_repeats: int,
    requested_case: str | None,
) -> MethodPanelData:
    _require_columns(
        rows,
        {
            "study",
            "case",
            "repeat",
            "method",
            "workers",
            "elapsed_s",
            "digest",
        },
        "method runs",
    )
    candidates = [
        row
        for row in rows
        if row["study"] == "method_ladder"
        and _as_int(row["workers"], "method workers") == 8
        and row["method"] in METHODS
    ]
    if requested_case is None:
        cases = sorted({row["case"] for row in candidates})
        if len(cases) != 1:
            raise ValueError(
                "method runs must contain exactly one P8 method-ladder case, "
                f"or --method-case must be supplied; found {cases}"
            )
        selected_case = cases[0]
    else:
        selected_case = requested_case
    candidates = [row for row in candidates if row["case"] == selected_case]
    if not candidates:
        raise ValueError(f"no P8 method-ladder rows for case {selected_case!r}")

    paired: dict[int, dict[str, Mapping[str, str]]] = {}
    for row in candidates:
        repeat = _as_int(row["repeat"], "method repeat")
        method = row["method"]
        group = paired.setdefault(repeat, {})
        if method in group:
            raise ValueError(
                f"duplicate method row for case={selected_case}, "
                f"repeat={repeat}, method={method}"
            )
        group[method] = row

    incomplete = {
        repeat: sorted(set(METHODS) - set(group))
        for repeat, group in paired.items()
        if set(group) != set(METHODS)
    }
    if incomplete:
        raise ValueError(f"incomplete P8 method pairs: {incomplete}")
    repeats = tuple(sorted(paired))
    if len(repeats) != required_repeats:
        raise ValueError(
            f"P8 method ladder requires exactly {required_repeats} paired "
            f"technical repeats; found {len(repeats)}"
        )

    elapsed_by_method: dict[str, tuple[float, ...]] = {}
    medians: dict[str, float] = {}
    for repeat, group in paired.items():
        if len({group[method]["digest"] for method in METHODS}) != 1:
            raise ValueError(
                f"semantic digest mismatch in method-ladder repeat {repeat}"
            )
    for method in METHODS:
        values = tuple(
            _as_float(paired[repeat][method]["elapsed_s"], "method elapsed_s")
            for repeat in repeats
        )
        if any(value <= 0.0 for value in values):
            raise ValueError(f"non-positive elapsed time for {method}")
        elapsed_by_method[method] = values
        medians[method] = statistics.median(values)

    _summary_semantic_check(
        summary,
        study="method_ladder",
        case=selected_case,
    )
    method_summary = summary.get("method_summary")
    if not isinstance(method_summary, list):
        raise ValueError("method summary lacks method_summary")
    for method in METHODS:
        matches = [
            item
            for item in method_summary
            if isinstance(item, dict)
            and item.get("case") == selected_case
            and item.get("method") == method
            and _as_int(item.get("workers"), "summary method workers") == 8
        ]
        if len(matches) != 1:
            raise ValueError(
                f"expected one summary row for {selected_case}/{method}/P8; "
                f"found {len(matches)}"
            )
        item = matches[0]
        if _as_int(item.get("runs"), "summary method runs") != required_repeats:
            raise ValueError(f"summary repeat count is not {required_repeats}")
        _assert_close(
            medians[method],
            _as_float(item.get("median_s"), "summary method median_s"),
            f"{method} median",
        )

    return MethodPanelData(
        case=selected_case,
        repeats=repeats,
        elapsed_by_method=elapsed_by_method,
        medians=medians,
    )


def _prepare_deliberative_panel(
    rows: Sequence[Mapping[str, str]],
    summary: Mapping[str, Any],
    required_repeats: int,
) -> tuple[DeliberativeDepthData, ...]:
    _require_columns(
        rows,
        {
            "depth",
            "repeat",
            "method",
            "workers",
            "elapsed_s",
            "candidate_digest",
            "trace_digest",
        },
        "deliberative runs",
    )
    if summary.get("status") != "complete_semantics_checked":
        raise ValueError(
            "deliberative summary is not complete_semantics_checked"
        )
    ratio_note = str(summary.get("ratio_definition", ""))
    if "KernelMemo-P8 elapsed / LayerProbe-P8 elapsed" not in ratio_note:
        raise ValueError("deliberative summary has an unexpected ratio definition")
    if "above 1" not in ratio_note or "favor LayerProbe" not in ratio_note:
        raise ValueError("deliberative ratio direction is not explicit")
    semantic_checks = summary.get("semantic_checks")
    if not isinstance(semantic_checks, list):
        raise ValueError("deliberative summary lacks semantic_checks")
    summary_checks_by_pair: dict[tuple[int, int], Mapping[str, Any]] = {}
    for check in semantic_checks:
        if not isinstance(check, dict):
            raise ValueError("invalid deliberative semantic-check record")
        depth = _as_int(check.get("depth"), "semantic-check depth")
        repeat = _as_int(check.get("repeat"), "semantic-check repeat")
        pair_key = (depth, repeat)
        if pair_key in summary_checks_by_pair:
            raise ValueError(
                "duplicate deliberative semantic check at "
                f"depth={depth}, repeat={repeat}"
            )
        if check.get("complete") is not True:
            raise ValueError(
                "incomplete deliberative semantic check at "
                f"depth={depth}, repeat={repeat}"
            )
        if check.get("candidate_digest_equal") is not True:
            raise ValueError(
                "candidate digest inequality recorded at "
                f"depth={depth}, repeat={repeat}"
            )
        if check.get("complete_trace_digest_equal") is not True:
            raise ValueError(
                "complete trace digest inequality recorded at "
                f"depth={depth}, repeat={repeat}"
            )
        if (
            check.get("kernel_memo_trace_digest")
            != check.get("layerprobe_trace_digest")
        ):
            raise ValueError(
                "summary trace digests disagree at "
                f"depth={depth}, repeat={repeat}"
            )
        summary_checks_by_pair[pair_key] = check

    depth_summary = summary.get("depths")
    if not isinstance(depth_summary, list) or not depth_summary:
        raise ValueError("deliberative summary lacks depth records")
    summary_by_depth: dict[int, Mapping[str, Any]] = {}
    for item in depth_summary:
        if not isinstance(item, dict):
            raise ValueError("invalid deliberative depth summary record")
        depth = _as_int(item.get("depth"), "summary depth")
        if depth in summary_by_depth:
            raise ValueError(f"duplicate depth summary: {depth}")
        summary_by_depth[depth] = item
    if 0 not in summary_by_depth:
        raise ValueError("deliberative experiment lacks depth-zero original policy")
    if not any(depth > 0 for depth in summary_by_depth):
        raise ValueError("deliberative experiment lacks a positive depth")

    raw_by_depth_repeat: dict[
        int, dict[int, dict[str, Mapping[str, str]]]
    ] = {}
    for row in rows:
        method = row["method"]
        if method not in DELIBERATIVE_METHODS:
            raise ValueError(f"unexpected deliberative method: {method!r}")
        workers = _as_int(row["workers"], "deliberative workers")
        if workers != 8:
            raise ValueError(
                f"deliberative figure requires P8 rows; observed {workers}"
            )
        depth = _as_int(row["depth"], "deliberative depth")
        repeat = _as_int(row["repeat"], "deliberative repeat")
        pair = raw_by_depth_repeat.setdefault(depth, {}).setdefault(repeat, {})
        if method in pair:
            raise ValueError(
                f"duplicate deliberative row: depth={depth}, "
                f"repeat={repeat}, method={method}"
            )
        pair[method] = row

    if set(raw_by_depth_repeat) != set(summary_by_depth):
        raise ValueError(
            "raw and summarized deliberative depth sets disagree: "
            f"raw={sorted(raw_by_depth_repeat)}, "
            f"summary={sorted(summary_by_depth)}"
        )

    prepared: list[DeliberativeDepthData] = []
    for depth in sorted(raw_by_depth_repeat):
        by_repeat = raw_by_depth_repeat[depth]
        repeats = tuple(sorted(by_repeat))
        if len(repeats) != required_repeats:
            raise ValueError(
                f"depth {depth} requires exactly {required_repeats} paired "
                f"technical repeats; found {len(repeats)}"
            )
        ratios: list[float] = []
        for repeat in repeats:
            pair = by_repeat[repeat]
            if set(pair) != set(DELIBERATIVE_METHODS):
                raise ValueError(
                    f"incomplete deliberative pair at depth={depth}, "
                    f"repeat={repeat}: {sorted(pair)}"
                )
            if (
                pair["kernel_memo_p8"]["candidate_digest"]
                != pair["layerprobe_p8"]["candidate_digest"]
            ):
                raise ValueError(
                    f"candidate digest mismatch at depth={depth}, repeat={repeat}"
                )
            if (
                pair["kernel_memo_p8"]["trace_digest"]
                != pair["layerprobe_p8"]["trace_digest"]
            ):
                raise ValueError(
                    f"complete trace digest mismatch at depth={depth}, "
                    f"repeat={repeat}"
                )
            check = summary_checks_by_pair.get((depth, repeat))
            if check is None:
                raise ValueError(
                    "missing summarized semantic check at "
                    f"depth={depth}, repeat={repeat}"
                )
            if (
                check.get("kernel_memo_trace_digest")
                != pair["kernel_memo_p8"]["trace_digest"]
            ):
                raise ValueError(
                    "raw and summarized trace digests disagree at "
                    f"depth={depth}, repeat={repeat}"
                )
            numerator = _as_float(
                pair["kernel_memo_p8"]["elapsed_s"],
                "KernelMemo-P8 elapsed_s",
            )
            denominator = _as_float(
                pair["layerprobe_p8"]["elapsed_s"],
                "LayerProbe-P8 elapsed_s",
            )
            if numerator <= 0.0 or denominator <= 0.0:
                raise ValueError("deliberative elapsed times must be positive")
            ratios.append(numerator / denominator)

        median = statistics.median(ratios)
        item = summary_by_depth[depth]
        if _as_int(item.get("paired_repeats"), "paired_repeats") != required_repeats:
            raise ValueError(
                f"summary paired repeats at depth {depth} is not "
                f"{required_repeats}"
            )
        recorded_median = _as_float(
            item.get("paired_speedup_kernel_memo_over_layerprobe_median"),
            "deliberative summary median",
        )
        ci_low = _as_float(
            item.get("paired_speedup_ci95_low"),
            "deliberative summary CI low",
        )
        ci_high = _as_float(
            item.get("paired_speedup_ci95_high"),
            "deliberative summary CI high",
        )
        _assert_close(
            median,
            recorded_median,
            f"deliberative depth {depth} ratio median",
        )
        if not ci_low <= median <= ci_high:
            raise ValueError(
                f"deliberative depth {depth} bootstrap interval does not "
                "bracket its median"
            )
        prepared.append(
            DeliberativeDepthData(
                depth=depth,
                repeats=repeats,
                ratios=tuple(ratios),
                median=median,
                ci95_low=ci_low,
                ci95_high=ci_high,
            )
        )

    expected_pairs = len(prepared) * required_repeats
    if _as_int(summary.get("complete_pairs"), "complete_pairs") != expected_pairs:
        raise ValueError("deliberative summary complete_pairs is inconsistent")
    if len(summary_checks_by_pair) != expected_pairs:
        raise ValueError(
            "deliberative semantic-check count is inconsistent with complete pairs"
        )
    return tuple(prepared)


def _prepare_scaling_panel(
    rows: Sequence[Mapping[str, str]],
    summary: Mapping[str, Any],
    required_repeats: int,
) -> ScalingPanelData:
    _require_columns(
        rows,
        {
            "study",
            "repeat",
            "method",
            "workers",
            "elapsed_s",
            "worker_slot_s",
            "digest",
        },
        "scaling runs",
    )
    scaling_rows = [row for row in rows if row["study"] == "parallel_scaling"]
    if len(scaling_rows) != len(rows):
        raise ValueError(
            "scaling runs input contains rows outside parallel_scaling"
        )
    if not scaling_rows:
        raise ValueError("scaling runs input contains no parallel_scaling rows")
    if any(row["method"] != "factorized" for row in scaling_rows):
        raise ValueError("scaling runs must use the factorized method")
    if len({row["digest"] for row in scaling_rows}) != 1:
        raise ValueError("scaling runs do not preserve one semantic digest")

    raw_by_worker: dict[int, dict[int, Mapping[str, str]]] = {}
    for row in scaling_rows:
        workers = _as_int(row["workers"], "raw scaling workers")
        repeat = _as_int(row["repeat"], "raw scaling repeat")
        by_repeat = raw_by_worker.setdefault(workers, {})
        if repeat in by_repeat:
            raise ValueError(
                f"duplicate raw scaling row at workers={workers}, repeat={repeat}"
            )
        by_repeat[repeat] = row

    metadata = summary.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("scaling summary lacks metadata")
    physical = _as_int(metadata.get("physical_cores"), "physical_cores")
    logical = _as_int(metadata.get("logical_cores"), "logical_cores")
    if physical != 8:
        raise ValueError(
            f"this review-response figure expects 8 physical cores; found {physical}"
        )
    if logical < 16:
        raise ValueError(
            f"this review-response figure expects at least 16 logical CPUs; "
            f"found {logical}"
        )
    _summary_semantic_check(summary, study="parallel_scaling")

    records = summary.get("parallel_scaling")
    if not isinstance(records, list) or not records:
        raise ValueError("scaling summary lacks parallel_scaling")
    points: list[ScalingPoint] = []
    seen: set[int] = set()
    for item in records:
        if not isinstance(item, dict):
            raise ValueError("invalid parallel_scaling record")
        workers = _as_int(item.get("workers"), "scaling workers")
        if workers in seen:
            raise ValueError(f"duplicate scaling worker count: {workers}")
        seen.add(workers)
        repeats = _as_int(item.get("runs"), "scaling runs")
        median_s = _as_float(item.get("median_s"), "scaling median_s")
        speedup = _as_float(item.get("speedup"), "scaling speedup")
        if repeats < 3 or median_s <= 0.0 or speedup <= 0.0:
            raise ValueError(f"invalid scaling record for {workers} workers")
        points.append(
            ScalingPoint(
                workers=workers,
                repeats=repeats,
                median_s=median_s,
                speedup=speedup,
            )
        )
    wrong_repeat_counts = {
        point.workers: point.repeats
        for point in points
        if point.repeats != required_repeats
    }
    if wrong_repeat_counts:
        raise ValueError(
            "strong-scaling points do not all contain exactly "
            f"{required_repeats} technical repeats: {wrong_repeat_counts}"
        )
    points.sort(key=lambda item: item.workers)
    worker_counts = [item.workers for item in points]
    expected_worker_counts = [1, 2, 4, 6, 8, 12, 16]
    if worker_counts != expected_worker_counts:
        raise ValueError(
            "strong-scaling summary must contain the fixed worker set "
            f"{expected_worker_counts}; "
            f"found {worker_counts}"
        )
    if sorted(raw_by_worker) != expected_worker_counts:
        raise ValueError(
            "raw strong-scaling rows contain the wrong worker set: "
            f"{sorted(raw_by_worker)}"
        )
    expected_repeat_ids = set(raw_by_worker[expected_worker_counts[0]])
    if len(expected_repeat_ids) != required_repeats:
        raise ValueError(
            "raw one-worker scaling baseline does not contain exactly "
            f"{required_repeats} repeat IDs"
        )
    for workers in expected_worker_counts[1:]:
        if set(raw_by_worker[workers]) != expected_repeat_ids:
            raise ValueError(
                "raw scaling repeat IDs are not paired across worker counts"
            )
    if physical not in worker_counts:
        raise ValueError("scaling summary lacks the 8-physical-core endpoint")
    baseline = points[0].median_s
    for point in points:
        raw_group = raw_by_worker[point.workers]
        if len(raw_group) != required_repeats:
            raise ValueError(
                f"raw scaling worker={point.workers} requires exactly "
                f"{required_repeats} technical repeats; found {len(raw_group)}"
            )
        raw_elapsed = [
            _as_float(row["elapsed_s"], "raw scaling elapsed_s")
            for row in raw_group.values()
        ]
        raw_worker_slot = [
            _as_float(row["worker_slot_s"], "raw scaling worker_slot_s")
            for row in raw_group.values()
        ]
        if any(value <= 0.0 for value in raw_elapsed + raw_worker_slot):
            raise ValueError("raw scaling times must be positive")
        raw_median = statistics.median(raw_elapsed)
        _assert_close(
            raw_median,
            point.median_s,
            f"raw/summary scaling median at {point.workers} workers",
        )
        _assert_close(
            baseline / point.median_s,
            point.speedup,
            f"scaling speedup at {point.workers} workers",
        )
        summary_item = next(
            item
            for item in records
            if _as_int(item.get("workers"), "scaling workers")
            == point.workers
        )
        _assert_close(
            baseline / (point.workers * raw_median),
            _as_float(summary_item.get("efficiency"), "scaling efficiency"),
            f"scaling efficiency at {point.workers} workers",
        )
        _assert_close(
            statistics.median(raw_worker_slot),
            _as_float(
                summary_item.get("median_worker_slot_s"),
                "scaling median_worker_slot_s",
            ),
            f"scaling worker-slot median at {point.workers} workers",
        )
    _assert_close(points[0].speedup, 1.0, "one-worker scaling baseline")
    return ScalingPanelData(
        physical_cores=physical,
        logical_cores=logical,
        points=tuple(points),
    )


def _prepare_scheduler_panel(
    rows: Sequence[Mapping[str, str]],
    summary: Mapping[str, Any],
    required_repeats: int,
) -> SchedulerPanelData:
    _require_columns(
        rows,
        {
            "repeat",
            "schedule",
            "workers",
            "kernels",
            "presentations",
            "elapsed_s",
            "digest",
            "load_imbalance_max_over_mean",
        },
        "scheduler runs",
    )
    if summary.get("status") != "complete_semantics_checked":
        raise ValueError("scheduler summary is not complete_semantics_checked")
    if (
        _as_int(
            summary.get("complete_paired_repeats"),
            "scheduler complete_paired_repeats",
        )
        != required_repeats
    ):
        raise ValueError(
            f"scheduler summary does not contain exactly "
            f"{required_repeats} complete paired repeats"
        )
    metric_notes = summary.get("metric_notes")
    if not isinstance(metric_notes, dict):
        raise ValueError("scheduler summary lacks metric_notes")
    imbalance_note = str(metric_notes.get("load_imbalance_max_over_mean", ""))
    if "maximum worker load divided by mean worker load" not in imbalance_note:
        raise ValueError("scheduler load-imbalance definition is unexpected")

    by_repeat: dict[int, dict[str, Mapping[str, str]]] = {}
    configurations: set[tuple[int, int, int]] = set()
    for row in rows:
        schedule = row["schedule"]
        if schedule not in SCHEDULES:
            raise ValueError(f"unexpected scheduler strategy: {schedule!r}")
        repeat = _as_int(row["repeat"], "scheduler repeat")
        workers = _as_int(row["workers"], "scheduler workers")
        if workers != 8:
            raise ValueError(
                f"scheduler figure requires eight workers; observed {workers}"
            )
        configuration = (
            workers,
            _as_int(row["kernels"], "scheduler kernels"),
            _as_int(row["presentations"], "scheduler presentations"),
        )
        configurations.add(configuration)
        group = by_repeat.setdefault(repeat, {})
        if schedule in group:
            raise ValueError(
                f"duplicate scheduler row for repeat={repeat}, "
                f"schedule={schedule}"
            )
        group[schedule] = row
    if len(configurations) != 1:
        raise ValueError(
            f"scheduler rows do not share one workload: {sorted(configurations)}"
        )
    repeats = tuple(sorted(by_repeat))
    if len(repeats) != required_repeats:
        raise ValueError(
            f"scheduler experiment requires exactly {required_repeats} paired "
            f"technical repeats; found {len(repeats)}"
        )
    for repeat in repeats:
        group = by_repeat[repeat]
        if set(group) != set(SCHEDULES):
            raise ValueError(
                f"incomplete scheduler repeat {repeat}: {sorted(group)}"
            )
        if len({group[schedule]["digest"] for schedule in SCHEDULES}) != 1:
            raise ValueError(f"scheduler semantic mismatch in repeat {repeat}")

    imbalance: dict[str, tuple[float, ...]] = {}
    elapsed: dict[str, tuple[float, ...]] = {}
    imbalance_medians: dict[str, float] = {}
    elapsed_medians: dict[str, float] = {}
    for schedule in SCHEDULES:
        imbalance_values = tuple(
            _as_float(
                by_repeat[repeat][schedule]["load_imbalance_max_over_mean"],
                "load_imbalance_max_over_mean",
            )
            for repeat in repeats
        )
        elapsed_values = tuple(
            _as_float(
                by_repeat[repeat][schedule]["elapsed_s"],
                "scheduler elapsed_s",
            )
            for repeat in repeats
        )
        if any(value < 1.0 - 1e-9 for value in imbalance_values):
            raise ValueError(
                f"load imbalance below its mathematical lower bound for {schedule}"
            )
        if any(value <= 0.0 for value in elapsed_values):
            raise ValueError(f"non-positive elapsed time for {schedule}")
        imbalance[schedule] = imbalance_values
        elapsed[schedule] = elapsed_values
        imbalance_medians[schedule] = statistics.median(imbalance_values)
        elapsed_medians[schedule] = statistics.median(elapsed_values)

    schedule_summary = summary.get("schedules")
    if not isinstance(schedule_summary, list):
        raise ValueError("scheduler summary lacks schedules")
    for schedule in SCHEDULES:
        matches = [
            item
            for item in schedule_summary
            if isinstance(item, dict) and item.get("schedule") == schedule
        ]
        if len(matches) != 1:
            raise ValueError(
                f"expected one scheduler summary for {schedule}; "
                f"found {len(matches)}"
            )
        item = matches[0]
        if _as_int(item.get("paired_repeats"), "scheduler paired_repeats") != (
            required_repeats
        ):
            raise ValueError(
                f"scheduler summary repeats for {schedule} is not "
                f"{required_repeats}"
            )
        _assert_close(
            imbalance_medians[schedule],
            _as_float(
                item.get("load_imbalance_max_over_mean_median"),
                "scheduler imbalance median",
            ),
            f"scheduler {schedule} imbalance median",
        )
        _assert_close(
            elapsed_medians[schedule],
            _as_float(
                item.get("elapsed_median_s"),
                "scheduler elapsed median",
            ),
            f"scheduler {schedule} elapsed median",
        )
    return SchedulerPanelData(
        repeats=repeats,
        imbalance_by_schedule=imbalance,
        elapsed_by_schedule=elapsed,
        imbalance_medians=imbalance_medians,
        elapsed_medians=elapsed_medians,
    )


def _resolve_times_new_roman() -> tuple[str, str]:
    """Resolve Times New Roman exactly; never accept a silent fallback."""

    requested = "Times New Roman"
    try:
        path = findfont(
            FontProperties(family=requested),
            fallback_to_default=False,
        )
    except ValueError as exc:
        raise RuntimeError(
            "Times New Roman is required but Matplotlib could not resolve it"
        ) from exc
    resolved = FontProperties(fname=path).get_name()
    if resolved.casefold() != requested.casefold():
        raise RuntimeError(
            f"Times New Roman is required; resolved {resolved!r} at {path}"
        )
    return resolved, str(Path(path).resolve())


def _configure_style() -> tuple[str, str]:
    resolved, path = _resolve_times_new_roman()
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "font.size": 6.4,
            "axes.titlesize": 7.0,
            "axes.titleweight": "bold",
            "axes.labelsize": 6.6,
            "axes.linewidth": 0.55,
            "axes.unicode_minus": True,
            "xtick.labelsize": 6.1,
            "ytick.labelsize": 6.1,
            "xtick.major.size": 2.4,
            "ytick.major.size": 2.4,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "legend.fontsize": 6.0,
            "legend.frameon": False,
            "lines.linewidth": 0.85,
            "lines.markersize": 3.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.transparent": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    return resolved, path


def _offsets(count: int, span: float = 0.14) -> tuple[float, ...]:
    if count < 1:
        raise ValueError("at least one point is required")
    if count == 1:
        return (0.0,)
    return tuple(
        -span / 2.0 + span * index / (count - 1)
        for index in range(count)
    )


def _style_axis(axis: mpl.axes.Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(
        axis="y",
        color=GRID_GRAY,
        linewidth=0.42,
        linestyle="-",
        zorder=0,
    )
    axis.tick_params(direction="out", pad=1.6)
    axis.yaxis.set_major_locator(MaxNLocator(nbins=5))
    axis.set_axisbelow(True)


def _median_glyph(
    axis: mpl.axes.Axes,
    x: float,
    y: float,
    color: str,
) -> None:
    axis.hlines(
        y,
        x - 0.16,
        x + 0.16,
        color=BLACK,
        linewidth=1.05,
        zorder=4,
    )
    axis.scatter(
        [x],
        [y],
        marker="D",
        s=18,
        facecolor=color,
        edgecolor=BLACK,
        linewidth=0.5,
        zorder=5,
    )


def _plot_method_panel(
    axis: mpl.axes.Axes,
    data: MethodPanelData,
) -> None:
    positions = tuple(range(len(METHODS)))
    offsets = _offsets(len(data.repeats))
    for repeat_index, _repeat in enumerate(data.repeats):
        xs = [position + offsets[repeat_index] for position in positions]
        ys = [
            data.elapsed_by_method[method][repeat_index]
            for method in METHODS
        ]
        axis.plot(
            xs,
            ys,
            color=LIGHT_GRAY,
            linewidth=0.55,
            alpha=0.78,
            zorder=1,
        )
    for position, method in zip(positions, METHODS):
        axis.scatter(
            [position + offset for offset in offsets],
            data.elapsed_by_method[method],
            s=12,
            marker=METHOD_MARKERS[method],
            facecolor=METHOD_COLORS[method],
            edgecolor=BLACK,
            linewidth=0.35,
            alpha=0.88,
            zorder=3,
        )
        _median_glyph(
            axis,
            position,
            data.medians[method],
            METHOD_COLORS[method],
        )
    upper = max(
        value
        for method in METHODS
        for value in data.elapsed_by_method[method]
    )
    axis.set_ylim(0.0, upper * 1.12)
    axis.set_xlim(-0.35, len(METHODS) - 0.65)
    axis.set_xticks(positions, [METHOD_LABELS[method] for method in METHODS])
    axis.set_ylabel("Elapsed time (s)")
    axis.set_title("(a) P8 decomposition ladder", loc="left", pad=3.0)
    _style_axis(axis)


def _plot_deliberative_panel(
    axis: mpl.axes.Axes,
    depths: Sequence[DeliberativeDepthData],
) -> None:
    positions = tuple(range(len(depths)))
    palette = (BLUE, ORANGE, PURPLE, SKY)
    markers = ("o", "s", "^", "v", "P", "X")
    all_values: list[float] = [1.0]
    for position, item in zip(positions, depths):
        offsets = _offsets(len(item.ratios))
        color = palette[position % len(palette)]
        marker = markers[position % len(markers)]
        axis.scatter(
            [position + offset for offset in offsets],
            item.ratios,
            s=12,
            marker=marker,
            facecolor=color,
            edgecolor=BLACK,
            linewidth=0.35,
            alpha=0.88,
            zorder=3,
        )
        axis.errorbar(
            [position],
            [item.median],
            yerr=[
                [item.median - item.ci95_low],
                [item.ci95_high - item.median],
            ],
            fmt="none",
            ecolor=BLACK,
            elinewidth=0.8,
            capsize=2.2,
            capthick=0.7,
            zorder=4,
        )
        _median_glyph(axis, position, item.median, color)
        all_values.extend(item.ratios)
        all_values.extend((item.ci95_low, item.ci95_high))
    axis.axhline(
        1.0,
        color=MID_GRAY,
        linestyle=(0, (3, 2)),
        linewidth=0.75,
        zorder=1,
    )
    labels = [
        "0\noriginal" if item.depth == 0 else str(item.depth)
        for item in depths
    ]
    axis.set_xticks(positions, labels)
    axis.set_xlabel("Lookahead depth", labelpad=1.5)
    axis.set_ylabel("T_KM / T_LP")
    axis.set_xlim(-0.35, len(depths) - 0.65)
    lower = min(all_values)
    upper = max(all_values)
    margin = max(0.025, 0.14 * (upper - lower))
    axis.set_ylim(max(0.0, lower - margin), upper + margin)
    axis.text(
        0.98,
        0.96,
        ">1 favors LayerProbe",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=MIN_FONT_PT,
        color=MID_GRAY,
    )
    axis.set_title("(b) Policy-cost sensitivity", loc="left", pad=3.0)
    _style_axis(axis)


def _plot_scaling_panel(
    axis: mpl.axes.Axes,
    data: ScalingPanelData,
) -> None:
    workers = [point.workers for point in data.points]
    speedups = [point.speedup for point in data.points]
    maximum_worker = max(workers)
    axis.axvspan(
        data.physical_cores,
        maximum_worker + 0.5,
        color=SMT_GRAY,
        linewidth=0.0,
        zorder=0,
    )
    axis.plot(
        workers,
        speedups,
        color=BLUE,
        marker="o",
        markerfacecolor=BLUE,
        markeredgecolor=BLACK,
        markeredgewidth=0.4,
        linewidth=1.05,
        zorder=3,
    )
    axis.plot(
        [1, data.physical_cores],
        [1, data.physical_cores],
        color=BLACK,
        linestyle=(0, (3, 2)),
        linewidth=0.75,
        zorder=2,
    )
    axis.axvline(
        data.physical_cores,
        color=MID_GRAY,
        linestyle=(0, (1.5, 2)),
        linewidth=0.65,
        zorder=1,
    )
    upper = max(max(speedups), float(data.physical_cores)) * 1.13
    axis.set_ylim(0.0, upper)
    axis.set_xlim(min(workers) - 0.5, maximum_worker + 0.5)
    axis.set_xticks(workers)
    axis.set_xlabel("Workers", labelpad=1.5)
    axis.set_ylabel("Speedup over 1 worker")
    axis.text(
        (data.physical_cores + maximum_worker) / 2.0,
        upper * 0.91,
        "SMT",
        ha="center",
        va="center",
        fontsize=MIN_FONT_PT,
        color=MID_GRAY,
    )
    physical_point = next(
        point
        for point in data.points
        if point.workers == data.physical_cores
    )
    axis.annotate(
        "8 physical cores",
        xy=(physical_point.workers, physical_point.speedup),
        xytext=(-3, 7),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=MIN_FONT_PT,
        color=MID_GRAY,
        arrowprops={
            "arrowstyle": "-",
            "color": MID_GRAY,
            "linewidth": 0.5,
            "shrinkA": 1,
            "shrinkB": 2,
        },
    )
    legend_handles = [
        Line2D(
            [0],
            [0],
            color=BLUE,
            marker="o",
            markeredgecolor=BLACK,
            markeredgewidth=0.35,
            linewidth=1.0,
            label="Measured median",
        ),
        Line2D(
            [0],
            [0],
            color=BLACK,
            linestyle=(0, (3, 2)),
            linewidth=0.75,
            label="Linear to 8 cores",
        ),
    ]
    axis.legend(
        handles=legend_handles,
        loc="upper left",
        borderaxespad=0.15,
        handlelength=1.8,
        labelspacing=0.25,
    )
    axis.set_title("(c) Single-host strong scaling", loc="left", pad=3.0)
    _style_axis(axis)


def _plot_scheduler_panel(
    axis: mpl.axes.Axes,
    data: SchedulerPanelData,
) -> None:
    label_offsets = {
        "current_chunksize": (4, 6, "left", "bottom"),
        "fine_chunksize_1": (4, -5, "left", "top"),
        "static_contiguous": (-4, 7, "right", "bottom"),
    }
    short_labels = {
        "current_chunksize": "Current",
        "fine_chunksize_1": "Fine",
        "static_contiguous": "Static",
    }
    for schedule in SCHEDULES:
        imbalance_values = data.imbalance_by_schedule[schedule]
        elapsed_values = data.elapsed_by_schedule[schedule]
        axis.scatter(
            imbalance_values,
            elapsed_values,
            s=13,
            marker=SCHEDULE_MARKERS[schedule],
            facecolor=SCHEDULE_COLORS[schedule],
            edgecolor=BLACK,
            linewidth=0.35,
            alpha=0.72,
            zorder=3,
        )
        median_x = data.imbalance_medians[schedule]
        median_y = data.elapsed_medians[schedule]
        axis.scatter(
            [median_x],
            [median_y],
            s=48,
            marker=SCHEDULE_MARKERS[schedule],
            facecolor=SCHEDULE_COLORS[schedule],
            edgecolor=BLACK,
            linewidth=0.8,
            zorder=5,
        )
        dx, dy, horizontal, vertical = label_offsets[schedule]
        axis.annotate(
            f"{short_labels[schedule]}  {median_y:.2f} s",
            xy=(median_x, median_y),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=horizontal,
            va=vertical,
            fontsize=MIN_FONT_PT,
            color=BLACK,
            zorder=6,
        )
    axis.axvline(
        1.0,
        color=MID_GRAY,
        linestyle=(0, (3, 2)),
        linewidth=0.75,
        zorder=1,
    )
    max_imbalance = max(
        value
        for schedule in SCHEDULES
        for value in data.imbalance_by_schedule[schedule]
    )
    elapsed_values = [
        value
        for schedule in SCHEDULES
        for value in data.elapsed_by_schedule[schedule]
    ]
    axis.set_xlim(0.99, max_imbalance + 0.035)
    axis.set_ylim(min(elapsed_values) - 0.55, max(elapsed_values) + 0.55)
    axis.set_xticks((1.0, 1.1, 1.2))
    axis.set_xlabel("Load imbalance (max / mean)")
    axis.set_ylabel("Elapsed time (s)")
    axis.text(
        0.025,
        0.045,
        "1 = ideal",
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=MIN_FONT_PT,
        color=MID_GRAY,
    )
    axis.set_title("(d) Scheduler trade-off", loc="left", pad=3.0)
    _style_axis(axis)


def _make_figure(
    method: MethodPanelData,
    deliberative: Sequence[DeliberativeDepthData],
    scaling: ScalingPanelData,
    scheduler: SchedulerPanelData,
) -> tuple[mpl.figure.Figure, tuple[mpl.axes.Axes, ...]]:
    figure, axes_grid = plt.subplots(
        2,
        2,
        figsize=(FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN),
        constrained_layout=True,
    )
    figure.set_constrained_layout_pads(
        w_pad=0.025,
        h_pad=0.220,
        wspace=0.045,
        hspace=0.065,
    )
    axes = tuple(axes_grid.flat)
    _plot_method_panel(axes[0], method)
    _plot_deliberative_panel(axes[1], deliberative)
    _plot_scaling_panel(axes[2], scaling)
    _plot_scheduler_panel(axes[3], scheduler)
    return figure, axes


def _text_layout_audit(
    figure: mpl.figure.Figure,
    axes: Sequence[mpl.axes.Axes],
    expected_font: str,
) -> dict[str, Any]:
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    figure_box = figure.bbox
    visible_text = [
        artist
        for artist in figure.findobj(match=Text)
        if artist.get_visible() and artist.get_text().strip()
    ]
    font_sizes = [float(artist.get_fontsize()) for artist in visible_text]
    font_names = sorted(
        {artist.get_fontproperties().get_name() for artist in visible_text}
    )
    out_of_bounds: list[dict[str, Any]] = []
    for artist in visible_text:
        box = artist.get_window_extent(renderer=renderer)
        if (
            box.x0 < figure_box.x0 - 1.0
            or box.y0 < figure_box.y0 - 1.0
            or box.x1 > figure_box.x1 + 1.0
            or box.y1 > figure_box.y1 + 1.0
        ):
            out_of_bounds.append(
                {
                    "text": artist.get_text(),
                    "bbox_pixels": [
                        float(box.x0),
                        float(box.y0),
                        float(box.x1),
                        float(box.y1),
                    ],
                }
            )

    overlapping_x_ticks: list[dict[str, Any]] = []
    for axis_index, axis in enumerate(axes):
        tick_boxes = [
            label.get_window_extent(renderer=renderer)
            for label in axis.get_xticklabels()
            if label.get_visible() and label.get_text().strip()
        ]
        for left_index, left in enumerate(tick_boxes):
            for right_index in range(left_index + 1, len(tick_boxes)):
                if left.overlaps(tick_boxes[right_index]):
                    overlapping_x_ticks.append(
                        {
                            "axis": axis_index,
                            "tick_indices": [left_index, right_index],
                        }
                    )

    minimum = min(font_sizes) if font_sizes else 0.0
    font_pass = (
        len(font_names) == 1
        and font_names[0].casefold() == expected_font.casefold()
    )
    return {
        "visible_text_count": len(visible_text),
        "minimum_font_size_pt": minimum,
        "resolved_text_fonts": font_names,
        "font_exact_match": font_pass,
        "text_out_of_bounds": out_of_bounds,
        "overlapping_x_tick_pairs": overlapping_x_ticks,
        "panel_label_strategy": (
            "panel labels are prefixes of left-aligned axes titles, with the "
            "same title x-position, pad, font size, and weight"
        ),
        "pass": (
            minimum >= MIN_FONT_PT
            and font_pass
            and not out_of_bounds
            and not overlapping_x_ticks
        ),
    }


def _caption(
    method: MethodPanelData,
    deliberative: Sequence[DeliberativeDepthData],
    scaling: ScalingPanelData,
    scheduler: SchedulerPanelData,
) -> str:
    scaling_repeat_counts = sorted({point.repeats for point in scaling.points})
    scaling_n = (
        f"n={scaling_repeat_counts[0]}"
        if len(scaling_repeat_counts) == 1
        else "n=" + "/".join(str(value) for value in scaling_repeat_counts)
    )
    return (
        "Work decomposition and when exact reuse affects elapsed time. "
        f"(a) P8 elapsed time for Flat, KernelMemo, and LayerProbe; each point "
        f"is one of n={len(method.repeats)} technical repeats, gray segments "
        "join the same repeat, and diamonds/short bars mark medians. Flat "
        "repeats all presentation-loop work, KernelMemo hoists "
        "mechanism-invariant construction, and LayerProbe additionally reuses "
        "complete-key semantic steps. "
        f"(b) Paired KernelMemo-P8/LayerProbe-P8 elapsed-time ratios for the "
        f"original policy (depth 0) and finite-depth deliberation "
        f"(n={len(deliberative[0].repeats)} technical repeats per depth); "
        "values above 1 favor LayerProbe. Whiskers are 95% technical bootstrap "
        "intervals for the median and are not population-level inferential "
        "confidence intervals. "
        f"(c) Single-host strong scaling ({scaling_n} "
        "technical repeats per worker count); the dashed guide shows linear "
        f"scaling through {scaling.physical_cores} physical cores and shading "
        "marks worker counts that use SMT. "
        f"(d) Elapsed time against load imbalance (maximum/mean worker load; "
        f"1 is ideal) for three semantics-equivalent schedules; each small "
        f"point is one of n={len(scheduler.repeats)} paired technical repeats "
        "and the enlarged marker is the bivariate median. All paired "
        "comparisons require identical semantic digests and deterministic "
        "reduction."
    )


def _statistics_payload(
    method: MethodPanelData,
    deliberative: Sequence[DeliberativeDepthData],
    scaling: ScalingPanelData,
    scheduler: SchedulerPanelData,
) -> dict[str, Any]:
    return {
        "panel_a": {
            "case": method.case,
            "repeat_ids": list(method.repeats),
            "ratio_direction": None,
            "technical_repeat_definition": (
                "one complete schedule-matched execution per method"
            ),
            "elapsed_s": {
                method_name: list(method.elapsed_by_method[method_name])
                for method_name in METHODS
            },
            "medians_s": {
                method_name: method.medians[method_name]
                for method_name in METHODS
            },
        },
        "panel_b": {
            "ratio_definition": (
                "KernelMemo-P8 elapsed / LayerProbe-P8 elapsed; "
                "values above 1 favor LayerProbe"
            ),
            "interval_definition": (
                "95% technical bootstrap interval for the median; "
                "not a population-level inferential confidence interval"
            ),
            "depths": [
                {
                    "depth": item.depth,
                    "repeat_ids": list(item.repeats),
                    "raw_paired_ratios": list(item.ratios),
                    "median": item.median,
                    "technical_bootstrap_ci95": [
                        item.ci95_low,
                        item.ci95_high,
                    ],
                }
                for item in deliberative
            ],
        },
        "panel_c": {
            "physical_cores": scaling.physical_cores,
            "logical_cores": scaling.logical_cores,
            "points": [
                {
                    "workers": point.workers,
                    "technical_repeats": point.repeats,
                    "median_elapsed_s": point.median_s,
                    "speedup_over_one_worker": point.speedup,
                }
                for point in scaling.points
            ],
        },
        "panel_d": {
            "metric_definition": {
                "x": (
                    "maximum worker load divided by mean worker load; "
                    "1 is ideal"
                ),
                "y": "wall-clock elapsed time in seconds",
            },
            "repeat_ids": list(scheduler.repeats),
            "raw_imbalance": {
                schedule: list(scheduler.imbalance_by_schedule[schedule])
                for schedule in SCHEDULES
            },
            "raw_elapsed_s": {
                schedule: list(scheduler.elapsed_by_schedule[schedule])
                for schedule in SCHEDULES
            },
            "imbalance_medians": {
                schedule: scheduler.imbalance_medians[schedule]
                for schedule in SCHEDULES
            },
            "elapsed_medians_s": {
                schedule: scheduler.elapsed_medians[schedule]
                for schedule in SCHEDULES
            },
        },
    }


def _save_outputs(
    figure: mpl.figure.Figure,
    *,
    pdf_path: Path,
    png_path: Path,
    grayscale_path: Path,
    dpi: int,
) -> dict[str, Any]:
    figure.savefig(
        pdf_path,
        format="pdf",
        dpi=dpi,
        facecolor="white",
        metadata={
            "Title": "Work decomposition and when reuse affects time",
            "Creator": "build_review_response_figure.py",
        },
    )
    figure.savefig(
        png_path,
        format="png",
        dpi=dpi,
        facecolor="white",
        metadata={
            "Title": "Work decomposition and when reuse affects time",
            "Software": "build_review_response_figure.py",
        },
    )
    with Image.open(png_path) as color_image:
        color_rgb = color_image.convert("RGB")
        grayscale_rgb = ImageOps.grayscale(color_rgb).convert("RGB")
        grayscale_rgb.save(grayscale_path, dpi=(dpi, dpi))
        png_size = color_rgb.size
    with Image.open(grayscale_path) as gray_image:
        grayscale_size = gray_image.size

    expected_size = (
        round(FIGURE_WIDTH_IN * dpi),
        round(FIGURE_HEIGHT_IN * dpi),
    )
    size_tolerance = 2
    dimensions_pass = all(
        abs(observed - expected) <= size_tolerance
        for observed, expected in zip(png_size, expected_size)
    )
    outputs = (pdf_path, png_path, grayscale_path)
    nonempty_pass = all(path.is_file() and path.stat().st_size > 0 for path in outputs)
    return {
        "expected_pixels": list(expected_size),
        "png_pixels": list(png_size),
        "grayscale_pixels": list(grayscale_size),
        "dimensions_pass": dimensions_pass and png_size == grayscale_size,
        "nonempty_files_pass": nonempty_pass,
        "pdf_vector_configuration": {
            "matplotlib_pdf_fonttype": mpl.rcParams["pdf.fonttype"],
            "rasterized_artists_added_by_script": False,
        },
        "pass": (
            dimensions_pass
            and png_size == grayscale_size
            and nonempty_pass
        ),
    }


def _output_paths(
    output_dir: Path,
    stem: str,
) -> dict[str, Path]:
    if not stem or Path(stem).name != stem:
        raise ValueError("--stem must be a plain file stem without directories")
    return {
        "pdf": output_dir / f"{stem}.pdf",
        "png": output_dir / f"{stem}.png",
        "grayscale": output_dir / f"{stem}_grayscale.png",
        "qa": output_dir / f"{stem}_qa.json",
        "caption": output_dir / f"{stem}_caption.txt",
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a semantics-gated, publication-size 2x2 performance "
            "evidence figure from completed single-host experiments."
        )
    )
    parser.add_argument("--method-runs", type=Path, required=True)
    parser.add_argument("--method-summary", type=Path, required=True)
    parser.add_argument("--method-case")
    parser.add_argument("--deliberative-runs", type=Path, required=True)
    parser.add_argument("--deliberative-summary", type=Path, required=True)
    parser.add_argument("--scaling-runs", type=Path, required=True)
    parser.add_argument("--scaling-summary", type=Path, required=True)
    parser.add_argument("--scheduler-runs", type=Path, required=True)
    parser.add_argument("--scheduler-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stem", default="fig_review_response")
    parser.add_argument("--required-new-repeats", type=int, default=10)
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace files with the selected new stem",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    generator_path = Path(__file__).resolve()
    generator_sha256_before = _sha256(generator_path)
    if args.required_new_repeats != 10:
        raise ValueError(
            "this reportable review-response figure requires exactly 10 "
            "technical repeats for each new paired experiment"
        )
    if args.dpi < 300:
        raise ValueError("publication PNG output requires at least 300 dpi")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = _output_paths(output_dir, args.stem)
    existing = [path for path in paths.values() if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "output files already exist; choose a new --stem or pass "
            f"--overwrite: {[str(path) for path in existing]}"
        )

    input_paths = {
        "method_runs": args.method_runs.resolve(),
        "method_summary": args.method_summary.resolve(),
        "deliberative_runs": args.deliberative_runs.resolve(),
        "deliberative_summary": args.deliberative_summary.resolve(),
        "scaling_runs": args.scaling_runs.resolve(),
        "scaling_summary": args.scaling_summary.resolve(),
        "scheduler_runs": args.scheduler_runs.resolve(),
        "scheduler_summary": args.scheduler_summary.resolve(),
    }
    input_sha256_before: dict[str, str] = {}

    try:
        input_sha256_before = {
            name: _sha256(path)
            for name, path in input_paths.items()
        }
        method_rows = _read_csv(input_paths["method_runs"], "method runs")
        method_summary = _read_json(
            input_paths["method_summary"],
            "method summary",
        )
        deliberative_rows = _read_csv(
            input_paths["deliberative_runs"],
            "deliberative runs",
        )
        deliberative_summary = _read_json(
            input_paths["deliberative_summary"],
            "deliberative summary",
        )
        scaling_rows = _read_csv(
            input_paths["scaling_runs"],
            "scaling runs",
        )
        scaling_summary = _read_json(
            input_paths["scaling_summary"],
            "scaling summary",
        )
        scheduler_rows = _read_csv(
            input_paths["scheduler_runs"],
            "scheduler runs",
        )
        scheduler_summary = _read_json(
            input_paths["scheduler_summary"],
            "scheduler summary",
        )

        method = _prepare_method_panel(
            method_rows,
            method_summary,
            args.required_new_repeats,
            args.method_case,
        )
        deliberative = _prepare_deliberative_panel(
            deliberative_rows,
            deliberative_summary,
            args.required_new_repeats,
        )
        scaling = _prepare_scaling_panel(
            scaling_rows,
            scaling_summary,
            args.required_new_repeats,
        )
        scheduler = _prepare_scheduler_panel(
            scheduler_rows,
            scheduler_summary,
            args.required_new_repeats,
        )
        resolved_font, font_path = _configure_style()
        figure, axes = _make_figure(
            method,
            deliberative,
            scaling,
            scheduler,
        )
        layout_qa = _text_layout_audit(figure, axes, resolved_font)
        if not layout_qa["pass"]:
            raise RuntimeError(
                "pre-export layout audit failed: "
                + json.dumps(layout_qa, ensure_ascii=False)
            )

        export_qa = _save_outputs(
            figure,
            pdf_path=paths["pdf"],
            png_path=paths["png"],
            grayscale_path=paths["grayscale"],
            dpi=args.dpi,
        )
        plt.close(figure)
        if not export_qa["pass"]:
            raise RuntimeError(
                "post-export file audit failed: "
                + json.dumps(export_qa, ensure_ascii=False)
            )

        caption = _caption(method, deliberative, scaling, scheduler)
        paths["caption"].write_text(caption + "\n", encoding="utf-8")
        changed_inputs = [
            name
            for name, path in input_paths.items()
            if _sha256(path) != input_sha256_before[name]
        ]
        if changed_inputs:
            raise RuntimeError(
                "figure inputs changed while outputs were being produced: "
                + ", ".join(changed_inputs)
            )
        if _sha256(generator_path) != generator_sha256_before:
            raise RuntimeError(
                "figure generator changed while outputs were being produced"
            )
        qa_payload = {
            "status": "PASS_MACHINE_CHECKS_MANUAL_VISUAL_REVIEW_PENDING",
            "script_version": SCRIPT_VERSION,
            "generated_at": datetime.now().astimezone().isoformat(),
            "claim_scope": (
                "single-host technical-repeat evidence; bootstrap intervals "
                "describe repeat-level timing stability and are not "
                "population-level inferential intervals"
            ),
            "figure": {
                "width_in": FIGURE_WIDTH_IN,
                "height_in": FIGURE_HEIGHT_IN,
                "dpi": args.dpi,
                "minimum_allowed_font_pt": MIN_FONT_PT,
                "palette": "Okabe-Ito categorical colors plus marker redundancy",
                "dual_y_axes": False,
                "mean_only_bars": False,
                "raw_new_experiment_points_visible": True,
                "grayscale_preview_generated": True,
            },
            "font": {
                "required": "Times New Roman",
                "resolved": resolved_font,
                "path": Path(font_path).name,
                "sha256": _sha256(Path(font_path)),
                "fallback_allowed": False,
            },
            "generator": {
                "path": Path(__file__).name,
                "sha256": generator_sha256_before,
                "unchanged_during_generation": True,
            },
            "inputs": {
                name: {
                    "path": path.name,
                    "sha256": input_sha256_before[name],
                }
                for name, path in input_paths.items()
            },
            "outputs": {
                name: {
                    "path": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
                for name, path in paths.items()
                if name in {"pdf", "png", "grayscale", "caption"}
            },
            "checks": {
                "all_new_experiments_exactly_n10": True,
                "method_digest_equality_per_repeat": True,
                "deliberative_candidate_digest_equality_per_pair": True,
                "deliberative_complete_trace_digest_equality_per_pair": True,
                "scheduler_digest_equality_per_repeat": True,
                "scaling_digest_equality_across_all_runs": True,
                "scaling_summary_recomputed_from_raw_runs": True,
                "all_inputs_unchanged_during_generation": True,
                "summary_raw_median_agreement": True,
                "ratio_direction_explicit": True,
                "technical_bootstrap_interpretation_explicit": True,
                "layout": layout_qa,
                "export": export_qa,
            },
            "statistics": _statistics_payload(
                method,
                deliberative,
                scaling,
                scheduler,
            ),
            "caption": caption,
            "manual_visual_review": {
                "status": "PENDING",
                "required": True,
                "clipping_free": None,
                "overlap_free": None,
                "alignment_pass": None,
                "grayscale_separation_pass": None,
                "instructions": (
                    "Inspect the 300-dpi color PNG and grayscale preview at "
                    "final 4.72-inch width for text clipping, annotation/data "
                    "overlap, panel-title alignment, and grayscale category "
                    "separation before inserting the PDF in the manuscript."
                ),
            },
        }
        _write_json(paths["qa"], qa_payload)
    except Exception as exc:
        failure_payload = {
            "status": "FAIL",
            "script_version": SCRIPT_VERSION,
            "generated_at": datetime.now().astimezone().isoformat(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "inputs": {
                name: str(path)
                for name, path in input_paths.items()
            },
            "generator": {
                "path": Path(__file__).name,
                "sha256": _sha256(generator_path),
                "unchanged_during_generation": (
                    _sha256(generator_path) == generator_sha256_before
                ),
            },
            "manual_visual_review": {
                "status": "NOT_REACHED",
                "required": True,
            },
        }
        _write_json(paths["qa"], failure_payload)
        raise


if __name__ == "__main__":
    main()
