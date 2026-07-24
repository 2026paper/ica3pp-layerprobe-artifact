#!/usr/bin/env python
"""Build the compact four-panel LayerProbe paper-results figure.

The figure is derived only from the frozen, distance-sentinel-corrected formal
results.  The script validates provenance and result semantics before plotting,
then emits a vector PDF, a 400-dpi colour PNG, a grayscale preview, and a
reproducibility/figure-QA report.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.text import Text
from PIL import Image, ImageOps
from pypdf import PdfReader


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
DEFAULT_RUN_DIR = (
    PROJECT_ROOT / "results" / "deadline_paper_distancefix_20260723_xeon"
)
DEFAULT_COMMUNICATION_DIR = (
    PROJECT_ROOT
    / "results"
    / "communication_full_24624_distancefix_provenance_v2_20260723_xeon"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "results" / "deadline_figures_distancefix_final_20260724_xeon"
)
REFERENCE_BUILDER = SCRIPT_PATH.with_name("build_deadline_figures.py")

RUNS_SHA256 = "f518d8ce452e23b91c54715ab70822b56548713d9131050156e49e56cd2ea4c4"
RUN_SUMMARY_SHA256 = (
    "5a0a9260a53c49def44f5d79ca28f73319e3069816079fad49539a15115be09a"
)
DELAY_SHA256 = "a12372cfe60014381db3902c85cddf88d97b5d82fa5376cbd0aea5a895415031"
COMM_SUMMARY_SHA256 = (
    "6dbb98660d761c1b43d5369c8a4349e542166a667cc91b02a8b8db82add2a6a1"
)

PRESENTATION_COUNTS = [2, 6, 10, 14, 18]
EXPECTED_RUNTIME_COUNTS = {2: 9, 6: 9, 10: 9, 14: 9, 18: 3}
EXPECTED_SUBSET_COUNTS = {2: 3, 6: 3, 10: 3, 14: 3, 18: 1}
FROZEN_REDUCTION_MEDIANS = [0.000, 18.886, 21.798, 25.492, 32.349]
EXPECTED_WORKERS = [1, 2, 4, 6, 8, 12, 16]
MODE_ORDER = ["exact", "coarse", "hidden"]

FIGURE_WIDTH_IN = 4.72
FIGURE_HEIGHT_IN = 4.62
PNG_DPI = 400
MIN_FONT_PT = 6.0

# Okabe-Ito-derived, colour-blind-safe accents.  Shape/fill/line redundancies
# keep the plots distinguishable in grayscale.
BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
DARK_GRAY = "#555555"
MID_GRAY = "#7A7A7A"
LIGHT_GRAY = "#C7C7C7"
GRID_GRAY = "#DEDEDE"
BALANCED_DIVERGING = LinearSegmentedColormap.from_list(
    "layerprobe_balanced_diverging",
    ["#3F6F8C", "#8FB2C5", "#F5F1E8", "#D9A074", "#A85D43"],
)

# This record is updated after inspecting the final colour and grayscale
# renderings at their publication size.
VISUAL_REVIEW_STATUS = "PASS"
VISUAL_REVIEW_NOTES = (
    "Round 1 inspected the 1888 x 1848 colour rendering: all four panels, "
    "panel labels, annotations, error bars, and colourbar are legible with no "
    "visible clipping, collision, or panel misalignment.",
    "Round 1 inspected the true-grayscale rendering: raw versus median "
    "markers and physical-core versus SMT segments remain distinguishable by "
        "fill, shape, and line style; the restrained blue-to-orange heatmap "
        "preserves sign and ordering while retaining a clear, non-neon hue.",
    "The data-rich panels remain readable at the fixed 4.72 x 4.62 inch "
    "publication size; no decorative legend or dual-axis encoding is needed.",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument(
        "--communication-dir", type=Path, default=DEFAULT_COMMUNICATION_DIR
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Destination for the four reproducible outputs. This may point "
            "directly at manuscript/draft/figures."
        ),
    )
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def configure_matplotlib() -> Path:
    font_path = Path(
        font_manager.findfont("Times New Roman", fallback_to_default=False)
    ).resolve()
    require(font_path.exists(), "Times New Roman is not installed.")
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 6.2,
            "axes.labelsize": 6.4,
            "axes.titlesize": 6.9,
            "axes.titleweight": "normal",
            "xtick.labelsize": 6.0,
            "ytick.labelsize": 6.0,
            "legend.fontsize": 6.0,
            "axes.linewidth": 0.65,
            "lines.linewidth": 1.0,
            "lines.markersize": 3.8,
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "axes.axisbelow": True,
            "axes.unicode_minus": True,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.dpi": PNG_DPI,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )
    return font_path


def validate_provenance(
    runs_path: Path,
    run_summary_path: Path,
    delay_path: Path,
    communication_summary_path: Path,
) -> dict[str, str]:
    expected = {
        "runs.csv": (runs_path, RUNS_SHA256),
        "run summary.json": (run_summary_path, RUN_SUMMARY_SHA256),
        "delay_effects.csv": (delay_path, DELAY_SHA256),
        "communication summary.json": (
            communication_summary_path,
            COMM_SUMMARY_SHA256,
        ),
    }
    actual: dict[str, str] = {}
    for label, (path, frozen_hash) in expected.items():
        require(path.is_file(), f"Missing required input: {path}")
        digest = sha256(path)
        require(
            digest == frozen_hash,
            f"{label} is not the frozen distance-fix input: {digest}",
        )
        actual[label] = digest
    return actual


def validate_inputs(
    summary: dict[str, Any],
    runs: list[dict[str, str]],
    communication_summary: dict[str, Any],
    delay_rows: list[dict[str, str]],
) -> None:
    require(
        summary.get("status") == "paper_candidate_results_semantics_checked",
        "Formal run is not marked paper_candidate_results_semantics_checked.",
    )
    require(summary.get("run_count") == 257, "Expected the complete 257-job run.")
    require(len(runs) == 257, f"Expected 257 run rows, found {len(runs)}.")
    semantic_checks = summary.get("semantic_checks", [])
    require(semantic_checks, "The formal run contains no semantic checks.")
    require(
        all(item.get("status") == "PASS" for item in semantic_checks),
        "At least one formal semantic-equivalence check did not pass.",
    )
    metadata = summary.get("metadata", {})
    require(
        int(metadata.get("physical_cores", -1)) == 8,
        "Expected the verified 8-physical-core workstation.",
    )
    require(
        int(metadata.get("logical_cores", -1)) == 16,
        "Expected the verified 16-logical-processor workstation.",
    )
    require(
        communication_summary.get("status")
        == "computational_preanalysis_not_human_effect_evidence",
        "Unexpected communication-analysis status.",
    )
    require(
        communication_summary.get("candidates") == 189792,
        "Expected 189,792 communication-analysis candidates.",
    )
    require(len(delay_rows) == 9, "Expected the complete 3 x 3 delay table.")
    require(
        all(int(row["kernel_count"]) == 10544 for row in delay_rows),
        "Each delay-effect cell must use all 10,544 valid kernels.",
    )


def derive_presentation_data(
    runs: Iterable[dict[str, str]],
) -> tuple[dict[int, list[float]], dict[int, list[float]]]:
    matched: dict[tuple[str, int], dict[str, dict[str, str]]] = defaultdict(dict)
    by_case: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in runs:
        if row["study"] != "presentation_scaling":
            continue
        key = (row["case"], int(row["repeat"]))
        require(
            row["method"] not in matched[key],
            f"Duplicate presentation row for {key} and {row['method']}.",
        )
        matched[key][row["method"]] = row
        by_case[row["case"]].append(row)

    runtime_ratios: dict[int, list[float]] = defaultdict(list)
    for (case, repeat), methods in matched.items():
        require(
            set(methods) == {"kernel_memo", "factorized"},
            f"Incomplete method pair for {case}, repeat {repeat}.",
        )
        memo = methods["kernel_memo"]
        layerprobe = methods["factorized"]
        require(
            memo["digest"] == layerprobe["digest"],
            f"Digest mismatch for {case}, repeat {repeat}.",
        )
        count = int(memo["presentation_count"])
        require(
            int(layerprobe["presentation_count"]) == count,
            f"Presentation-count mismatch for {case}, repeat {repeat}.",
        )
        runtime_ratios[count].append(
            float(memo["elapsed_s"]) / float(layerprobe["elapsed_s"])
        )

    reductions: dict[int, list[float]] = defaultdict(list)
    for case, case_rows in by_case.items():
        method_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in case_rows:
            method_rows[row["method"]].append(row)
        require(
            set(method_rows) == {"kernel_memo", "factorized"},
            f"Unexpected methods in presentation case {case}.",
        )
        counts = {int(row["presentation_count"]) for row in case_rows}
        require(len(counts) == 1, f"Inconsistent presentation count in {case}.")
        count = counts.pop()

        replicate_reductions: list[float] = []
        repeats = sorted({int(row["repeat"]) for row in case_rows})
        for repeat in repeats:
            memo = next(
                row
                for row in method_rows["kernel_memo"]
                if int(row["repeat"]) == repeat
            )
            layerprobe = next(
                row
                for row in method_rows["factorized"]
                if int(row["repeat"]) == repeat
            )
            require(
                memo["policy_calls"] == memo["transition_calls"]
                and layerprobe["policy_calls"] == layerprobe["transition_calls"],
                f"Policy/transition accounting mismatch in {case}.",
            )
            replicate_reductions.append(
                100.0
                * (
                    1.0
                    - float(layerprobe["policy_calls"])
                    / float(memo["policy_calls"])
                )
            )
        require(
            max(replicate_reductions) - min(replicate_reductions) < 1e-12,
            f"Deterministic exact-work count changed across repeats in {case}.",
        )
        reductions[count].append(replicate_reductions[0])

    require(
        sorted(runtime_ratios) == PRESENTATION_COUNTS,
        "Unexpected presentation counts in paired-runtime data.",
    )
    require(
        sorted(reductions) == PRESENTATION_COUNTS,
        "Unexpected presentation counts in exact-work data.",
    )
    require(
        all(
            len(runtime_ratios[count]) == EXPECTED_RUNTIME_COUNTS[count]
            for count in PRESENTATION_COUNTS
        ),
        "Unexpected number of paired technical runtime replicates.",
    )
    require(
        all(
            len(reductions[count]) == EXPECTED_SUBSET_COUNTS[count]
            for count in PRESENTATION_COUNTS
        ),
        "Unexpected number of frozen presentation subsets.",
    )
    return runtime_ratios, reductions


def derive_scaling_data(
    summary: dict[str, Any],
    runs: Iterable[dict[str, str]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    raw: dict[int, dict[int, float]] = defaultdict(dict)
    for row in runs:
        if row["study"] != "parallel_scaling":
            continue
        worker = int(row["workers"])
        repeat = int(row["repeat"])
        require(
            repeat not in raw[worker],
            f"Duplicate scaling row for worker {worker}, repeat {repeat}.",
        )
        raw[worker][repeat] = float(row["elapsed_s"])

    require(
        sorted(raw) == EXPECTED_WORKERS,
        f"Unexpected scaling worker set: {sorted(raw)}.",
    )
    require(
        all(sorted(raw[worker]) == [0, 1, 2, 3, 4] for worker in EXPECTED_WORKERS),
        "Each scaling worker count must have the same five matched repeats.",
    )

    medians = np.array(
        [
            statistics.median(raw[worker].values())
            for worker in EXPECTED_WORKERS
        ],
        dtype=float,
    )
    speedups = medians[0] / medians
    lows: list[float] = []
    highs: list[float] = []
    for worker in EXPECTED_WORKERS:
        paired = [
            raw[1][repeat] / raw[worker][repeat]
            for repeat in sorted(raw[worker])
        ]
        lows.append(min(paired))
        highs.append(max(paired))
    low_array = np.array(lows, dtype=float)
    high_array = np.array(highs, dtype=float)

    recorded = {
        int(item["workers"]): float(item["speedup"])
        for item in summary["parallel_scaling"]
    }
    require(
        sorted(recorded) == EXPECTED_WORKERS,
        "Unexpected scaling summary worker set.",
    )
    require(
        all(
            abs(recorded[worker] - speedups[index]) < 1e-12
            for index, worker in enumerate(EXPECTED_WORKERS)
        ),
        "Recorded speedups are not ratios of median elapsed times.",
    )
    return medians, speedups, low_array, high_array


def derive_delay_matrix(delay_rows: Iterable[dict[str, str]]) -> np.ndarray:
    lookup = {
        (row["speed_mode"], row["distance_mode"]): float(
            row["mean_pair_delta_delayed_minus_immediate"]
        )
        for row in delay_rows
    }
    expected = {
        (speed_mode, distance_mode)
        for speed_mode in MODE_ORDER
        for distance_mode in MODE_ORDER
    }
    require(set(lookup) == expected, "Delay rows do not form the required 3 x 3.")
    matrix = np.array(
        [
            [lookup[(speed_mode, distance_mode)] for distance_mode in MODE_ORDER]
            for speed_mode in MODE_ORDER
        ],
        dtype=float,
    )
    require(np.isfinite(matrix).all(), "Delay matrix contains a non-finite value.")
    require(np.max(np.abs(matrix)) <= 0.4, "Frozen delay scale exceeds +/-0.4.")
    return matrix


def style_cartesian_axis(axis: plt.Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="y", color=GRID_GRAY, linewidth=0.45)


def plot_runtime_panel(
    axis: plt.Axes, runtime_ratios: dict[int, list[float]]
) -> dict[str, np.ndarray]:
    x = np.array(PRESENTATION_COUNTS, dtype=float)
    medians = np.array(
        [statistics.median(runtime_ratios[count]) for count in PRESENTATION_COUNTS]
    )
    lows = np.array([min(runtime_ratios[count]) for count in PRESENTATION_COUNTS])
    highs = np.array([max(runtime_ratios[count]) for count in PRESENTATION_COUNTS])

    for count in PRESENTATION_COUNTS:
        values = sorted(runtime_ratios[count])
        offsets = np.linspace(-0.28, 0.28, len(values))
        axis.scatter(
            count + offsets,
            values,
            s=9,
            marker="o",
            facecolors="white",
            edgecolors=MID_GRAY,
            linewidths=0.5,
            zorder=2,
        )
    axis.errorbar(
        x,
        medians,
        yerr=np.vstack((medians - lows, highs - medians)),
        fmt="D-",
        color=BLUE,
        markerfacecolor=BLUE,
        markeredgecolor=BLUE,
        markersize=3.6,
        capsize=2.2,
        capthick=0.75,
        elinewidth=0.75,
        zorder=3,
    )
    axis.axhline(1.0, color=DARK_GRAY, linestyle=(0, (3, 2)), linewidth=0.7)
    axis.text(
        2.25,
        1.0045,
        "LayerProbe faster",
        color=DARK_GRAY,
        ha="left",
        va="bottom",
        fontsize=MIN_FONT_PT,
    )
    axis.text(
        2.25,
        0.9955,
        "memo faster",
        color=DARK_GRAY,
        ha="left",
        va="top",
        fontsize=MIN_FONT_PT,
    )
    axis.annotate(
        f"{medians[-1]:.3f}x",
        (18, medians[-1]),
        xytext=(-2, 5),
        textcoords="offset points",
        ha="right",
        va="bottom",
        color=BLUE,
        fontsize=MIN_FONT_PT,
    )
    axis.set_title("Matched single-worker time", loc="left", pad=3)
    axis.set_xlabel("Presentations")
    axis.set_ylabel("Matched time ratio\nTmemo / TLayerProbe")
    axis.set_xticks(PRESENTATION_COUNTS)
    axis.set_xlim(1.0, 19.0)
    axis.set_ylim(0.875, 1.040)
    axis.set_yticks([0.88, 0.92, 0.96, 1.00, 1.04])
    style_cartesian_axis(axis)
    return {"median": medians, "low": lows, "high": highs}


def plot_reduction_panel(
    axis: plt.Axes, reductions: dict[int, list[float]]
) -> dict[str, np.ndarray]:
    x = np.array(PRESENTATION_COUNTS, dtype=float)
    medians = np.array(
        [statistics.median(reductions[count]) for count in PRESENTATION_COUNTS]
    )
    lows = np.array([min(reductions[count]) for count in PRESENTATION_COUNTS])
    highs = np.array([max(reductions[count]) for count in PRESENTATION_COUNTS])
    require(
        [round(value, 3) for value in medians] == FROZEN_REDUCTION_MEDIANS,
        (
            "Exact-work medians changed: "
            f"{[round(value, 6) for value in medians]}"
        ),
    )

    for count in PRESENTATION_COUNTS:
        values = sorted(reductions[count])
        offsets = [0.0] if len(values) == 1 else [-0.22, 0.0, 0.22]
        axis.scatter(
            count + np.array(offsets),
            values,
            s=12,
            marker="o",
            facecolors="white",
            edgecolors=MID_GRAY,
            linewidths=0.6,
            zorder=3,
        )
    axis.errorbar(
        x,
        medians,
        yerr=np.vstack((medians - lows, highs - medians)),
        fmt="D-",
        color=GREEN,
        markerfacecolor=GREEN,
        markeredgecolor=GREEN,
        markersize=3.6,
        capsize=2.2,
        capthick=0.75,
        elinewidth=0.75,
        zorder=2,
    )
    axis.text(
        2.0,
        34.2,
        "open = subsets; diamond/bar = median/range",
        color=DARK_GRAY,
        ha="left",
        va="top",
        fontsize=MIN_FONT_PT,
    )
    axis.annotate(
        f"{medians[-1]:.3f}%",
        (18, medians[-1]),
        xytext=(-2, 5),
        textcoords="offset points",
        ha="right",
        va="bottom",
        color=GREEN,
        fontsize=MIN_FONT_PT,
    )
    axis.set_title("Deterministic exact-work reduction", loc="left", pad=3)
    axis.set_xlabel("Presentations")
    axis.set_ylabel("Policy/transition-call\nreduction (%)")
    axis.set_xticks(PRESENTATION_COUNTS)
    axis.set_xlim(1.0, 19.0)
    axis.set_ylim(-1.5, 35.5)
    axis.set_yticks([0, 10, 20, 30])
    style_cartesian_axis(axis)
    return {"median": medians, "low": lows, "high": highs}


def plot_scaling_panel(
    axis: plt.Axes,
    speedups: np.ndarray,
    lows: np.ndarray,
    highs: np.ndarray,
) -> None:
    workers = np.array(EXPECTED_WORKERS)
    physical = workers <= 8
    smt = workers > 8
    lower_error = np.maximum(speedups - lows, 0.0)
    upper_error = np.maximum(highs - speedups, 0.0)

    axis.axvspan(8.5, 16.5, facecolor=ORANGE, alpha=0.07, linewidth=0)
    axis.plot(
        [1, 16],
        [1, 16],
        color=LIGHT_GRAY,
        linestyle=(0, (3, 2)),
        linewidth=0.8,
        zorder=1,
    )
    axis.errorbar(
        workers[physical],
        speedups[physical],
        yerr=np.vstack((lower_error[physical], upper_error[physical])),
        fmt="o-",
        color=BLUE,
        markerfacecolor=BLUE,
        markeredgecolor=BLUE,
        markersize=3.7,
        capsize=2.0,
        capthick=0.7,
        elinewidth=0.7,
        zorder=3,
    )
    axis.plot(
        [8, 12, 16],
        [speedups[4], speedups[5], speedups[6]],
        color=ORANGE,
        linestyle=(0, (3, 2)),
        linewidth=1.0,
        zorder=2,
    )
    axis.errorbar(
        workers[smt],
        speedups[smt],
        yerr=np.vstack((lower_error[smt], upper_error[smt])),
        fmt="s",
        color=ORANGE,
        markerfacecolor="white",
        markeredgecolor=ORANGE,
        markersize=3.8,
        capsize=2.0,
        capthick=0.7,
        elinewidth=0.7,
        zorder=4,
    )
    axis.axvline(8.5, color=ORANGE, linestyle=":", linewidth=0.65)
    axis.text(
        6.0,
        7.15,
        "ideal",
        color=MID_GRAY,
        ha="center",
        va="bottom",
        fontsize=MIN_FONT_PT,
    )
    axis.text(
        8.15,
        6.22,
        "8 physical cores",
        color=BLUE,
        ha="right",
        va="bottom",
        fontsize=MIN_FONT_PT,
    )
    axis.text(
        11.8,
        8.90,
        "SMT",
        color=ORANGE,
        ha="center",
        va="top",
        fontsize=MIN_FONT_PT,
    )
    axis.annotate(
        f"{speedups[-1]:.2f}x",
        (16, speedups[-1]),
        xytext=(-2, 5),
        textcoords="offset points",
        ha="right",
        va="bottom",
        color=ORANGE,
        fontsize=MIN_FONT_PT,
    )
    axis.set_title("Strong scaling on this workstation", loc="left", pad=3)
    axis.set_xlabel("Worker processes")
    axis.set_ylabel("Speedup vs 1 worker")
    axis.set_xticks(EXPECTED_WORKERS)
    axis.set_xlim(0.5, 16.5)
    axis.set_ylim(0.0, 9.2)
    axis.set_yticks([0, 2, 4, 6, 8])
    style_cartesian_axis(axis)


def plot_delay_panel(axis: plt.Axes, matrix: np.ndarray) -> None:
    edges = np.arange(4)
    mesh = axis.pcolormesh(
        edges,
        edges,
        matrix,
        cmap=BALANCED_DIVERGING,
        norm=TwoSlopeNorm(vmin=-0.4, vcenter=0.0, vmax=0.4),
        shading="flat",
        edgecolors="white",
        linewidth=0.55,
    )
    for row_index in range(3):
        for column_index in range(3):
            value = matrix[row_index, column_index]
            color = "white" if value < -0.245 else "black"
            axis.text(
                column_index + 0.5,
                row_index + 0.5,
                f"{value:+.3f}" if value else "0.000",
                ha="center",
                va="center",
                color=color,
                fontsize=6.2,
            )
    pretty_modes = [mode.capitalize() for mode in MODE_ORDER]
    axis.set_xticks(np.arange(3) + 0.5, pretty_modes)
    axis.set_yticks(np.arange(3) + 0.5, pretty_modes)
    axis.set_xlim(0, 3)
    axis.set_ylim(3, 0)
    axis.set_aspect("equal")
    axis.set_anchor("N")
    axis.set_xlabel("Distance mode")
    axis.set_ylabel("Speed mode")
    axis.set_title("Delay effect (delayed - immediate)", loc="left", pad=3)
    axis.tick_params(length=0)
    colorbar = axis.figure.colorbar(
        mesh,
        ax=axis,
        fraction=0.060,
        pad=0.035,
        ticks=[-0.4, -0.2, 0.0, 0.2, 0.4],
    )
    colorbar.set_label(
        "Mean delta pairs\nper valid kernel",
        rotation=270,
        labelpad=10,
        fontsize=6.2,
    )
    colorbar.ax.tick_params(labelsize=MIN_FONT_PT, width=0.6, length=2.2)
    colorbar.outline.set_linewidth(0.6)
    # Matplotlib rasterizes dense colourbar gradients by default.  Restore the
    # QuadMesh to vector paths so the complete PDF remains resolution-free.
    colorbar.solids.set_rasterized(False)


def add_aligned_panel_labels(
    fig: plt.Figure, axes: list[plt.Axes]
) -> None:
    fig.canvas.draw()
    positions = [axis.get_position() for axis in axes]
    row_tops = [
        max(positions[0].y1, positions[1].y1),
        max(positions[2].y1, positions[3].y1),
    ]
    for index, (axis, label) in enumerate(
        zip(axes, ("(a)", "(b)", "(c)", "(d)"))
    ):
        box = axis.get_position()
        row_top = row_tops[0 if index < 2 else 1]
        fig.text(
            box.x0 - 0.037,
            row_top + 0.008,
            label,
            ha="right",
            va="bottom",
            fontsize=7.4,
            fontweight="bold",
        )


def build_figure(
    runtime_ratios: dict[int, list[float]],
    reductions: dict[int, list[float]],
    speedups: np.ndarray,
    scaling_lows: np.ndarray,
    scaling_highs: np.ndarray,
    delay_matrix: np.ndarray,
) -> tuple[plt.Figure, dict[str, dict[str, np.ndarray]]]:
    fig = plt.figure(figsize=(FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN))
    grid = fig.add_gridspec(
        2,
        2,
        left=0.123,
        right=0.910,
        bottom=0.090,
        top=0.945,
        wspace=0.37,
        hspace=0.48,
    )
    axes = [
        fig.add_subplot(grid[0, 0]),
        fig.add_subplot(grid[0, 1]),
        fig.add_subplot(grid[1, 0]),
        fig.add_subplot(grid[1, 1]),
    ]
    runtime_stats = plot_runtime_panel(axes[0], runtime_ratios)
    reduction_stats = plot_reduction_panel(axes[1], reductions)
    plot_scaling_panel(axes[2], speedups, scaling_lows, scaling_highs)
    plot_delay_panel(axes[3], delay_matrix)
    add_aligned_panel_labels(fig, axes)
    fig.canvas.draw()
    return fig, {"runtime": runtime_stats, "reduction": reduction_stats}


def inspect_figure_text(fig: plt.Figure) -> dict[str, Any]:
    renderer = fig.canvas.get_renderer()
    texts = [
        item
        for item in fig.findobj(match=Text)
        if item.get_visible() and item.get_text().strip()
    ]
    require(texts, "Figure contains no visible text.")
    font_sizes = [float(item.get_fontsize()) for item in texts]
    require(
        min(font_sizes) >= MIN_FONT_PT,
        f"Figure contains text below {MIN_FONT_PT} pt.",
    )
    require(
        all("Times New Roman" in item.get_fontfamily() for item in texts),
        "A visible text object is not configured for Times New Roman.",
    )

    width_px, height_px = fig.canvas.get_width_height()
    out_of_bounds: list[str] = []
    for item in texts:
        bounds = item.get_window_extent(renderer=renderer)
        if (
            bounds.x0 < -1
            or bounds.y0 < -1
            or bounds.x1 > width_px + 1
            or bounds.y1 > height_px + 1
        ):
            out_of_bounds.append(item.get_text().replace("\n", " / "))
    require(
        not out_of_bounds,
        "Text extends beyond the figure canvas: " + ", ".join(out_of_bounds),
    )
    return {
        "min_font_pt": min(font_sizes),
        "max_font_pt": max(font_sizes),
        "visible_text_objects": len(texts),
    }


def _pdf_resource_inventory(
    resources: Any,
    font_subtypes: set[str],
    base_fonts: set[str],
    embedded_fonts: set[str],
    unembedded_fonts: set[str],
    image_count: list[int],
) -> None:
    if resources is None:
        return
    resources = resources.get_object()
    fonts = resources.get("/Font")
    if fonts is not None:
        for font_reference in fonts.get_object().values():
            font = font_reference.get_object()
            subtype = str(font.get("/Subtype", ""))
            font_subtypes.add(subtype)
            base_font = font.get("/BaseFont")
            base_font_name = str(base_font) if base_font is not None else "(unnamed)"
            if base_font is not None:
                base_fonts.add(base_font_name)
            descendants = font.get("/DescendantFonts")
            if descendants is not None:
                descendant_embedded = False
                for descendant_reference in descendants:
                    descendant = descendant_reference.get_object()
                    font_subtypes.add(str(descendant.get("/Subtype", "")))
                    descendant_base = descendant.get("/BaseFont")
                    if descendant_base is not None:
                        base_fonts.add(str(descendant_base))
                    descriptor_reference = descendant.get("/FontDescriptor")
                    if descriptor_reference is not None:
                        descriptor = descriptor_reference.get_object()
                        descendant_embedded = descendant_embedded or any(
                            descriptor.get(key) is not None
                            for key in ("/FontFile", "/FontFile2", "/FontFile3")
                        )
                if descendant_embedded:
                    embedded_fonts.add(base_font_name)
                else:
                    unembedded_fonts.add(base_font_name)
            else:
                descriptor_reference = font.get("/FontDescriptor")
                descriptor = (
                    descriptor_reference.get_object()
                    if descriptor_reference is not None
                    else {}
                )
                is_embedded = any(
                    descriptor.get(key) is not None
                    for key in ("/FontFile", "/FontFile2", "/FontFile3")
                )
                if is_embedded:
                    embedded_fonts.add(base_font_name)
                else:
                    unembedded_fonts.add(base_font_name)
    xobjects = resources.get("/XObject")
    if xobjects is not None:
        for reference in xobjects.get_object().values():
            xobject = reference.get_object()
            subtype = str(xobject.get("/Subtype", ""))
            if subtype == "/Image":
                image_count[0] += 1
            elif subtype == "/Form":
                _pdf_resource_inventory(
                    xobject.get("/Resources"),
                    font_subtypes,
                    base_fonts,
                    embedded_fonts,
                    unembedded_fonts,
                    image_count,
                )


def inspect_pdf(path: Path) -> dict[str, Any]:
    reader = PdfReader(str(path))
    require(len(reader.pages) == 1, "Combined figure PDF must have exactly one page.")
    page = reader.pages[0]
    width_in = float(page.mediabox.width) / 72.0
    height_in = float(page.mediabox.height) / 72.0
    require(abs(width_in - FIGURE_WIDTH_IN) < 0.01, "Unexpected PDF width.")
    require(abs(height_in - FIGURE_HEIGHT_IN) < 0.01, "Unexpected PDF height.")

    font_subtypes: set[str] = set()
    base_fonts: set[str] = set()
    embedded_fonts: set[str] = set()
    unembedded_fonts: set[str] = set()
    image_count = [0]
    _pdf_resource_inventory(
        page.get("/Resources"),
        font_subtypes,
        base_fonts,
        embedded_fonts,
        unembedded_fonts,
        image_count,
    )
    require("/Type3" not in font_subtypes, "PDF contains a Type 3 font.")
    require(image_count[0] == 0, "PDF contains raster image XObjects.")
    require(
        not unembedded_fonts,
        f"PDF contains unembedded fonts: {sorted(unembedded_fonts)}",
    )
    require(
        any("Times" in name for name in base_fonts),
        f"Times New Roman was not embedded: {sorted(base_fonts)}",
    )
    return {
        "width_in": width_in,
        "height_in": height_in,
        "font_subtypes": sorted(font_subtypes),
        "base_fonts": sorted(base_fonts),
        "embedded_fonts": sorted(embedded_fonts),
        "unembedded_fonts": sorted(unembedded_fonts),
        "image_xobjects": image_count[0],
    }


def inspect_png(path: Path) -> dict[str, Any]:
    with Image.open(path) as image:
        dpi = image.info.get("dpi", (0.0, 0.0))
        info = {
            "mode": image.mode,
            "width_px": image.width,
            "height_px": image.height,
            "dpi_x": float(dpi[0]),
            "dpi_y": float(dpi[1]),
        }
    expected_width = round(FIGURE_WIDTH_IN * PNG_DPI)
    expected_height = round(FIGURE_HEIGHT_IN * PNG_DPI)
    require(info["width_px"] == expected_width, "Unexpected PNG width.")
    require(info["height_px"] == expected_height, "Unexpected PNG height.")
    require(399.0 <= info["dpi_x"] <= 401.0, "PNG x-DPI is not 400.")
    require(399.0 <= info["dpi_y"] <= 401.0, "PNG y-DPI is not 400.")
    return info


def save_outputs(fig: plt.Figure, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "pdf": output_dir / "fig1_combined_results.pdf",
        "png": output_dir / "fig1_combined_results.png",
        "grayscale": output_dir / "fig1_combined_results_grayscale.png",
        "qa": output_dir / "FIGURE_QA.md",
    }
    title = "LayerProbe exact-work reduction and workstation scaling"
    fig.savefig(
        paths["pdf"],
        metadata={"Title": title, "Creator": SCRIPT_PATH.name},
    )
    fig.savefig(
        paths["png"],
        dpi=PNG_DPI,
        metadata={"Title": title, "Software": SCRIPT_PATH.name},
    )
    with Image.open(paths["png"]) as colour:
        grayscale = ImageOps.grayscale(colour)
        grayscale.save(paths["grayscale"], dpi=(PNG_DPI, PNG_DPI), optimize=True)
    return paths


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def write_qa_report(
    path: Path,
    *,
    args: argparse.Namespace,
    summary: dict[str, Any],
    provenance: dict[str, str],
    font_path: Path,
    text_qa: dict[str, Any],
    pdf_qa: dict[str, Any],
    png_qa: dict[str, Any],
    grayscale_qa: dict[str, Any],
    runtime_ratios: dict[int, list[float]],
    reductions: dict[int, list[float]],
    runtime_stats: dict[str, np.ndarray],
    reduction_stats: dict[str, np.ndarray],
    scaling_medians: np.ndarray,
    speedups: np.ndarray,
    scaling_lows: np.ndarray,
    scaling_highs: np.ndarray,
    delay_matrix: np.ndarray,
) -> None:
    source_rows = [
        [label, f"`{digest}`"] for label, digest in provenance.items()
    ]
    source_rows.extend(
        [
            ["generator", f"`{sha256(SCRIPT_PATH)}`"],
            ["reference builder", f"`{sha256(REFERENCE_BUILDER)}`"],
        ]
    )

    runtime_rows: list[list[str]] = []
    for index, count in enumerate(PRESENTATION_COUNTS):
        runtime_rows.append(
            [
                str(count),
                str(len(runtime_ratios[count])),
                f"{runtime_stats['median'][index]:.6f}",
                (
                    f"{runtime_stats['low'][index]:.6f} - "
                    f"{runtime_stats['high'][index]:.6f}"
                ),
            ]
        )

    reduction_rows: list[list[str]] = []
    for index, count in enumerate(PRESENTATION_COUNTS):
        reduction_rows.append(
            [
                str(count),
                str(len(reductions[count])),
                f"{reduction_stats['median'][index]:.3f}",
                (
                    f"{reduction_stats['low'][index]:.3f} - "
                    f"{reduction_stats['high'][index]:.3f}"
                ),
            ]
        )

    scaling_rows: list[list[str]] = []
    for index, worker in enumerate(EXPECTED_WORKERS):
        scaling_rows.append(
            [
                str(worker),
                f"{scaling_medians[index]:.6f}",
                f"{speedups[index]:.6f}",
                f"{scaling_lows[index]:.6f} - {scaling_highs[index]:.6f}",
            ]
        )

    matrix_rows = [
        [
            MODE_ORDER[row_index].capitalize(),
            *[f"{value:+.6f}" for value in delay_matrix[row_index]],
        ]
        for row_index in range(3)
    ]

    technical_rows = [
        ["PDF physical size", f"{pdf_qa['width_in']:.3f} x {pdf_qa['height_in']:.3f} in", "PASS"],
        [
            "Colour PNG",
            (
                f"{png_qa['width_px']} x {png_qa['height_px']} px; "
                f"{png_qa['dpi_x']:.2f} dpi"
            ),
            "PASS",
        ],
        [
            "Grayscale PNG",
            (
                f"{grayscale_qa['width_px']} x {grayscale_qa['height_px']} px; "
                f"{grayscale_qa['dpi_x']:.2f} dpi; mode {grayscale_qa['mode']}"
            ),
            "PASS",
        ],
        [
            "Visible text",
            (
                f"Times New Roman; minimum {text_qa['min_font_pt']:.1f} pt; "
                f"{text_qa['visible_text_objects']} objects"
            ),
            "PASS",
        ],
        [
            "PDF fonts",
            (
                f"subtypes {', '.join(pdf_qa['font_subtypes'])}; "
                f"{len(pdf_qa['embedded_fonts'])} embedded; no Type 3"
            ),
            "PASS",
        ],
        [
            "Vector integrity",
            f"{pdf_qa['image_xobjects']} raster image XObjects",
            "PASS",
        ],
        [
            "Formal semantics",
            (
                f"{len(summary['semantic_checks'])} of "
                f"{len(summary['semantic_checks'])} checks PASS"
            ),
            "PASS",
        ],
    ]

    report = f"""# Combined Results Figure QA

Generated: `{datetime.now().astimezone().isoformat(timespec="seconds")}`

Generator: `{SCRIPT_PATH}`

Output directory: `{Path(args.output_dir).resolve()}`

## Scope and claim

The 2 x 2 figure supports the paper's bounded claim that LayerProbe reduces
exact policy/transition work as the presentation family grows and that the
remaining workload scales on the verified 8-physical-core/16-logical-processor
workstation. Panel (a) reports the measured runtime crossover without
interpolation or a universal speed claim. Panel (d) remains computational
pre-analysis, not human-effect evidence.

## Frozen provenance

{markdown_table(["Input/source", "SHA-256"], source_rows)}

Formal status: `{summary["status"]}`; rows: `{summary["run_count"]}`; workstation:
8 physical cores, 16 logical processors. Installed font:
`{font_path}`.

## Statistical encodings

- Panel (a): every matched technical replicate is an open point. The diamond is
  the median of matched `Tmemo / TLayerProbe` ratios; bars are observed
  minimum-maximum ranges. These are descriptive ranges, not confidence
  intervals.
- Panel (b): every point is a deterministic frozen presentation subset. The
  diamond is the subset median and bars are the subset minimum-maximum. No
  stochastic uncertainty is implied. Policy calls equal transition calls for
  every plotted case.
- Panel (c): each center is `median(T1) / median(Tw)` from five repeats. Bars are
  the minimum-maximum of repeat-matched `T1,r / Tw,r` ratios. The shaded region
  begins after the 8-physical-core boundary and denotes SMT-only worker counts.
- Panel (d): each cell is the mean change in separated model pairs per valid
  kernel, `delayed - immediate`, over 10,544 valid kernels. The colour scale is
  fixed and symmetric at `[-0.4, +0.4]`.

### Panel (a): matched elapsed-time ratio

{markdown_table(["Presentations", "Technical pairs", "Median", "Observed range"], runtime_rows)}

### Panel (b): deterministic exact-work reduction (%)

{markdown_table(["Presentations", "Frozen subsets", "Median", "Subset range"], reduction_rows)}

Required frozen medians are exactly `0.000, 18.886, 21.798, 25.492, 32.349%`
at three decimals.

### Panel (c): strong scaling

{markdown_table(["Workers", "Median time (s)", "Ratio of medians", "Matched range"], scaling_rows)}

### Panel (d): delayed-minus-immediate mean separated-pair delta

Rows are speed mode; columns are distance mode.

{markdown_table(["Speed mode", "Exact", "Coarse", "Hidden"], matrix_rows)}

## Programmatic publication QA

{markdown_table(["Check", "Result", "Status"], technical_rows)}

Embedded PDF base fonts: `{", ".join(pdf_qa["base_fonts"])}`.

The figure uses no dual y-axis, no categorical connecting line, no pie chart,
and no rainbow palette. Raw points remain visible for the small-n panels.
Colour encodings have marker-fill and line-style redundancy; every heatmap cell
also carries a signed numeric value.

## Visual QA

Status: **{VISUAL_REVIEW_STATUS}**

{chr(10).join(f"- {note}" for note in VISUAL_REVIEW_NOTES)}

## Reproduction

From the project root:

```powershell
python experiments/build_combined_results_figure.py
```

To generate directly into a manuscript tree:

```powershell
python experiments/build_combined_results_figure.py --output-dir manuscript/draft/figures
```

The script only writes `fig1_combined_results.pdf`,
`fig1_combined_results.png`, `fig1_combined_results_grayscale.png`, and
`FIGURE_QA.md` in the selected output directory; it does not modify the
existing single-panel figures or the manuscript source.
"""
    path.write_text(report, encoding="utf-8", newline="\n")


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    communication_dir = args.communication_dir.resolve()
    args.output_dir = args.output_dir.resolve()

    runs_path = run_dir / "runs.csv"
    run_summary_path = run_dir / "summary.json"
    delay_path = communication_dir / "delay_effects.csv"
    communication_summary_path = communication_dir / "summary.json"
    provenance = validate_provenance(
        runs_path,
        run_summary_path,
        delay_path,
        communication_summary_path,
    )

    summary = read_json(run_summary_path)
    runs = read_csv(runs_path)
    communication_summary = read_json(communication_summary_path)
    delay_rows = read_csv(delay_path)
    validate_inputs(summary, runs, communication_summary, delay_rows)

    runtime_ratios, reductions = derive_presentation_data(runs)
    scaling_medians, speedups, scaling_lows, scaling_highs = (
        derive_scaling_data(summary, runs)
    )
    delay_matrix = derive_delay_matrix(delay_rows)

    font_path = configure_matplotlib()
    fig, plotted_stats = build_figure(
        runtime_ratios,
        reductions,
        speedups,
        scaling_lows,
        scaling_highs,
        delay_matrix,
    )
    text_qa = inspect_figure_text(fig)
    output_paths = save_outputs(fig, args.output_dir)
    plt.close(fig)

    pdf_qa = inspect_pdf(output_paths["pdf"])
    png_qa = inspect_png(output_paths["png"])
    grayscale_qa = inspect_png(output_paths["grayscale"])
    require(
        grayscale_qa["mode"] == "L",
        "Grayscale preview is not an actual grayscale image.",
    )
    write_qa_report(
        output_paths["qa"],
        args=args,
        summary=summary,
        provenance=provenance,
        font_path=font_path,
        text_qa=text_qa,
        pdf_qa=pdf_qa,
        png_qa=png_qa,
        grayscale_qa=grayscale_qa,
        runtime_ratios=runtime_ratios,
        reductions=reductions,
        runtime_stats=plotted_stats["runtime"],
        reduction_stats=plotted_stats["reduction"],
        scaling_medians=scaling_medians,
        speedups=speedups,
        scaling_lows=scaling_lows,
        scaling_highs=scaling_highs,
        delay_matrix=delay_matrix,
    )

    for label in ("pdf", "png", "grayscale", "qa"):
        print(f"{label}: {output_paths[label]}")
    print(
        "QA: frozen provenance PASS; result semantics PASS; "
        "400-dpi dimensions PASS; vector PDF PASS; no Type 3 fonts PASS."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
