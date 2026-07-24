#!/usr/bin/env python
"""Build the LayerProbe semantic-safety and robustness explanation figures."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.text import Text
from PIL import Image, ImageOps
from pypdf import PdfReader


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
RESULTS_ROOT = PROJECT_ROOT / "results"
DEFAULT_OUTPUT_DIR = (
    RESULTS_ROOT / "deadline_figures_ccfa_candidate_20260724_xeon"
)

ABLATION_PATH = (
    RESULTS_ROOT
    / "cache_key_ablation_full_24624_distancefix_20260723_xeon"
    / "ablation_summary.csv"
)
ORACLE_PATH = (
    RESULTS_ROOT
    / "independent_trace_oracle_full_24624_distancefix_20260723_xeon"
    / "summary.json"
)
SENSITIVITY_ROOT = (
    RESULTS_ROOT
    / "agent_sensitivity_full_24624_distancefix_provenance_v2_20260723_xeon"
)
PAIR_PATH = SENSITIVITY_ROOT / "pair_delay_sensitivity.csv"
LEAVE_ONE_OUT_PATH = SENSITIVITY_ROOT / "leave_one_agent_out.csv"
COMMUNICATION_ROOT = (
    RESULTS_ROOT
    / "communication_full_24624_distancefix_provenance_v2_20260723_xeon"
)
ROBUST_FAMILIES_PATH = COMMUNICATION_ROOT / "robust_families.csv"
CANDIDATE_PATH = COMMUNICATION_ROOT / "candidate_signatures.csv.gz"
MUTATION_PATH = (
    RESULTS_ROOT
    / "randomized_mutation_audit_seed20260724_128"
    / "randomized_mutation_results.json"
)

EXPECTED_HASHES = {
    ABLATION_PATH: "b9150aa23377b72b8428100194788dbe83015188cf40a01a73f42adea7ee2c7f",
    ORACLE_PATH: "a5d6531a346092b8185aab916515de9a69729981770823ab5b2b5b70614e64e7",
    PAIR_PATH: "a6b164ad8850a5bc2f088308224c8491422105279693d6e850d33a51b41b7d1c",
    LEAVE_ONE_OUT_PATH: "7ae964293fc2a82ac2e3f86c9bb68332b8f5018c1a81a253c4af0d8c08e11e43",
    ROBUST_FAMILIES_PATH: "b5e0b7ae64abc0307010598bfc3dec23a5053a0c2daafbf89ce274cd50c96e31",
    CANDIDATE_PATH: "3505e0be250e919e524281caf75a826b8580bdb46ba0206012aaafed4e7bbc72",
    MUTATION_PATH: "91c5c2874d24939e9b6300db5d1d4f33f6a6df07925e2b78abf15841b82e30b2",
}

WIDTH_IN = 4.72
FIG3_HEIGHT_IN = 2.72
FIG4_HEIGHT_IN = 3.26
PNG_DPI = 400
MIN_FONT_PT = 6.5

INK = "#1F2933"
BLUE = "#0072B2"
SKY = "#56B4E9"
ORANGE = "#E69F00"
VERMILION = "#D55E00"
GREEN = "#009E73"
DARK_GREEN = "#006B50"
MID_GRAY = "#70777C"
LIGHT_GRAY = "#D7DCE0"
PALE_GRAY = "#F4F6F7"
PALE_BLUE = "#E6F2F8"
PALE_ORANGE = "#FFF3DC"
PALE_GREEN = "#E4F4EE"
GRID = "#D9DEE2"
WHITE = "#FFFFFF"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def configure_style() -> Path:
    font_path = Path(
        font_manager.findfont("Times New Roman", fallback_to_default=False)
    ).resolve()
    require(font_path.exists(), "Times New Roman is not installed.")
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 6.6,
            "axes.labelsize": 6.7,
            "axes.titlesize": 7.2,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.5,
            "axes.linewidth": 0.65,
            "lines.linewidth": 0.9,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.4,
            "ytick.major.size": 2.4,
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": PNG_DPI,
            "savefig.facecolor": WHITE,
            "figure.facecolor": WHITE,
        }
    )
    return font_path


def validate_provenance() -> dict[str, str]:
    actual: dict[str, str] = {}
    for path, expected in EXPECTED_HASHES.items():
        require(path.is_file(), f"Missing frozen input: {path}")
        digest = sha256(path)
        require(digest == expected, f"Unexpected hash for {path}: {digest}")
        actual[str(path.relative_to(PROJECT_ROOT))] = digest
    return actual


def load_ablation() -> tuple[list[dict[str, str]], dict[str, Any]]:
    rows = read_csv(ABLATION_PATH)
    require(len(rows) == 8, "Expected four key variants in two replay orders.")
    oracle = read_json(ORACLE_PATH)
    counts = oracle["comparison"]["counts"]
    require(counts["validity_mismatch_count"] == 0, "Oracle validity mismatch.")
    require(counts["flat_trace_mismatch_count"] == 0, "Oracle/Flat trace mismatch.")
    require(
        counts["factorized_trace_mismatch_count"] == 0,
        "Oracle/LayerProbe trace mismatch.",
    )
    require(
        counts["direct_candidate_mismatch_count"] == 0,
        "Oracle/Flat signature mismatch.",
    )
    require(
        counts["factorized_candidate_mismatch_count"] == 0,
        "Oracle/LayerProbe signature mismatch.",
    )
    mutant = oracle["mutant_smoke"]
    require(
        mutant["mutants_total"] == 7
        and mutant["mutants_detected"] == 7
        and mutant["all_detected"],
        "Expected all seven seeded mutants to be detected.",
    )
    return rows, oracle


def load_mutation_sensitivity() -> dict[str, int]:
    payload = read_json(MUTATION_PATH)
    records = payload["records"]
    require(len(records) == 60, "Expected the fixed-seed catalog of 60 mutants.")
    trace_changing = sum(bool(row["trace_detected"]) for row in records)
    signature_changing = sum(bool(row["signature_detected"]) for row in records)
    inactive_or_equivalent = len(records) - trace_changing
    trace_only = sum(
        bool(row["trace_detected"]) and not bool(row["signature_detected"])
        for row in records
    )
    require(
        (
            trace_changing,
            signature_changing,
            inactive_or_equivalent,
            trace_only,
        )
        == (56, 49, 4, 7),
        "Unexpected fixed-seed mutation census.",
    )
    return {
        "total": len(records),
        "trace_changing": trace_changing,
        "signature_changing": signature_changing,
        "inactive_or_equivalent": inactive_or_equivalent,
        "trace_only": trace_only,
    }


def load_robustness() -> tuple[
    list[dict[str, str]],
    list[dict[str, str]],
    dict[str, int],
    dict[str, int],
]:
    pair_rows = read_csv(PAIR_PATH)
    leave_rows = read_csv(LEAVE_ONE_OUT_PATH)
    families = read_csv(ROBUST_FAMILIES_PATH)
    require(len(pair_rows) == 6, "Expected six declared-agent pairs.")
    require(len(leave_rows) == 4, "Expected four leave-one-agent-out rows.")
    require(
        all(row["delay_direction"] == "negative" for row in leave_rows),
        "Every leave-one-agent-out aggregate must remain negative.",
    )
    all_18 = next(row for row in families if row["family"] == "all_18")
    require(
        int(all_18["robust_minimum_suite_size"]) == 2,
        "Expected the verified all-18 minimum robust suite size of two.",
    )
    require(
        int(all_18["union_minimum_suite_size"]) == 1,
        "Expected the any-presentation union minimum suite size of one.",
    )
    require(
        int(all_18["robust_nonzero_kernels"]) == 9494
        and int(all_18["robust_full_kernels"]) == 0,
        "Unexpected all-18 robust-kernel census.",
    )
    require(
        all(int(row["robust_minimum_suite_size"]) == 1 for row in leave_rows),
        "Every leave-one-agent-out robust cover must have size one.",
    )
    selected = {"brake_10377", "brake_10387"}
    masks: dict[str, list[int]] = defaultdict(list)
    with gzip.open(CANDIDATE_PATH, "rt", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["kernel"] in selected:
                masks[row["kernel"]].append(int(row["signature_mask"]))
    require(set(masks) == selected, "Missing a selected robust-cover mechanism.")
    robust_masks: dict[str, int] = {}
    for kernel, values in masks.items():
        require(len(values) == 18, f"{kernel} does not have all 18 presentations.")
        intersection = values[0]
        for value in values[1:]:
            intersection &= value
        robust_masks[kernel] = intersection
    require(
        robust_masks["brake_10377"] == 59
        and robust_masks["brake_10387"] == 31,
        f"Unexpected robust masks: {robust_masks}",
    )
    require(
        robust_masks["brake_10377"] | robust_masks["brake_10387"] == 63,
        "The selected pair does not cover all six agent-pair bits.",
    )
    cover_stats = {
        "kernel_count": int(all_18["kernel_count"]),
        "robust_nonzero_kernels": int(all_18["robust_nonzero_kernels"]),
        "robust_full_kernels": int(all_18["robust_full_kernels"]),
        "union_minimum_suite_size": int(all_18["union_minimum_suite_size"]),
        "robust_minimum_suite_size": int(all_18["robust_minimum_suite_size"]),
    }
    return pair_rows, leave_rows, robust_masks, cover_stats


def style_axis(axis: plt.Axes, *, grid_axis: str = "x") -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis=grid_axis, color=GRID, linewidth=0.55)
    axis.set_axisbelow(True)


def add_panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        0.0,
        1.02,
        label,
        transform=axis.transAxes,
        fontsize=7.2,
        fontweight="bold",
        ha="left",
        va="bottom",
        clip_on=False,
    )


def draw_semantic_audit_matrix(
    axis: plt.Axes,
    ablation_rows: list[dict[str, str]],
    oracle: dict[str, Any],
    mutation: dict[str, int],
) -> None:
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    counts = oracle["comparison"]["counts"]
    by_variant: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in ablation_rows:
        by_variant[row["variant"]][row["order"]] = row
    variants = [
        ("full", "Full key  (s, m, o)", True),
        ("drop_state", "Drop world state  s", False),
        ("drop_memory", "Drop agent memory  m", False),
        ("drop_observation", "Drop observation  o", False),
    ]
    require(
        all(set(by_variant[name]) == {"canonical", "reverse"} for name, _, _ in variants),
        "Missing a canonical or reverse key-ablation row.",
    )

    axis.text(
        0.005,
        0.965,
        "Full semantic key preserves every audited output",
        ha="left",
        va="top",
        fontsize=7.6,
        fontweight="bold",
        color=INK,
    )
    axis.plot([0.005, 0.995], [0.925, 0.925], color=BLUE, linewidth=1.0)

    audit_y0, audit_height = 0.755, 0.145
    axis.add_patch(
        Rectangle(
            (0.005, audit_y0),
            0.585,
            audit_height,
            facecolor=PALE_GREEN,
            edgecolor=GREEN,
            linewidth=0.7,
        )
    )
    axis.text(
        0.018,
        audit_y0 + 0.098,
        "Independent oracle",
        ha="left",
        va="center",
        fontsize=6.7,
        fontweight="bold",
        color=DARK_GREEN,
    )
    axis.text(
        0.018,
        audit_y0 + 0.044,
        "full key (s, m, o)",
        ha="left",
        va="center",
        fontsize=6.5,
        color=INK,
    )
    oracle_items = [
        ("Validity", f"0 / {counts['requested_kernels']:,}"),
        ("Traces", f"0 / {counts['trace_cases']:,}"),
        ("Signatures", f"0 / {counts['candidate_comparisons']:,}"),
    ]
    for x, (label, value) in zip(
        [0.275, 0.410, 0.525],
        oracle_items,
    ):
        axis.text(
            x,
            audit_y0 + 0.098,
            label,
            ha="center",
            va="center",
            fontsize=6.5,
            color=MID_GRAY,
        )
        axis.text(
            x,
            audit_y0 + 0.044,
            value,
            ha="center",
            va="center",
            fontsize=6.7,
            fontweight="bold",
            color=DARK_GREEN,
        )

    mutation_x0 = 0.605
    axis.add_patch(
        Rectangle(
            (mutation_x0, audit_y0),
            0.39,
            audit_height,
            facecolor=PALE_ORANGE,
            edgecolor=ORANGE,
            linewidth=0.7,
        )
    )
    axis.text(
        mutation_x0 + 0.012,
        audit_y0 + 0.112,
        "Fault sensitivity",
        ha="left",
        va="center",
        fontsize=6.7,
        fontweight="bold",
        color=INK,
    )
    sensitivity_items = [
        ("Targeted", "7 / 7"),
        ("Trace", f"{mutation['trace_changing']} / {mutation['total']}"),
        ("Six-bit", f"{mutation['signature_changing']} / {mutation['trace_changing']}"),
    ]
    for x, (label, value) in zip(
        [0.665, 0.800, 0.930],
        sensitivity_items,
    ):
        axis.text(
            x,
            audit_y0 + 0.068,
            label,
            ha="center",
            va="center",
            fontsize=6.5,
            color=MID_GRAY,
        )
        axis.text(
            x,
            audit_y0 + 0.025,
            value,
            ha="center",
            va="center",
            fontsize=6.7,
            fontweight="bold",
            color=VERMILION,
        )
    axis.text(
        0.995,
        0.728,
        "Fixed seed: 4 inactive/equivalent; six-bit summary misses 7 trace changes",
        ha="right",
        va="center",
        fontsize=6.5,
        color=MID_GRAY,
    )

    column_edges = [0.005, 0.255, 0.395, 0.555, 0.745, 0.905, 0.995]
    column_centers = [
        (column_edges[index] + column_edges[index + 1]) / 2
        for index in range(len(column_edges) - 1)
    ]
    headers = [
        "Key variant",
        "Unsafe\nclasses",
        "Affected\nmechanisms",
        "Trace\nfailures C / R",
        "Signature\nfailures C / R",
        "Verdict",
    ]
    header_y0, header_height = 0.600, 0.090
    axis.add_patch(
        Rectangle(
            (column_edges[0], header_y0),
            column_edges[-1] - column_edges[0],
            header_height,
            facecolor=PALE_BLUE,
            edgecolor=BLUE,
            linewidth=0.65,
        )
    )
    for index, (center, header) in enumerate(zip(column_centers, headers)):
        axis.text(
            column_edges[0] + 0.012 if index == 0 else center,
            header_y0 + header_height / 2,
            header,
            ha="left" if index == 0 else "center",
            va="center",
            fontsize=6.5,
            fontweight="bold",
            color=INK,
            linespacing=0.92,
        )

    row_centers = [0.525, 0.395, 0.265, 0.135]
    row_height = 0.110
    for (variant, label, is_safe), y in zip(variants, row_centers):
        canonical = by_variant[variant]["canonical"]
        reverse = by_variant[variant]["reverse"]
        require(
            canonical["unsafe_key_classes"] == reverse["unsafe_key_classes"]
            and canonical["affected_valid_kernels"]
            == reverse["affected_valid_kernels"],
            f"Order-dependent collision census for {variant}.",
        )
        axis.add_patch(
            Rectangle(
                (column_edges[0], y - row_height / 2),
                column_edges[-1] - column_edges[0],
                row_height,
                facecolor=PALE_GREEN if is_safe else WHITE,
                edgecolor=LIGHT_GRAY,
                linewidth=0.45,
            )
        )
        axis.add_patch(
            Rectangle(
                (column_edges[0], y - row_height / 2),
                0.006,
                row_height,
                facecolor=GREEN if is_safe else ORANGE,
                edgecolor="none",
            )
        )
        values = [
            label,
            f"{int(canonical['unsafe_key_classes']):,}",
            f"{int(canonical['affected_valid_kernels']):,} / 10,544",
            (
                f"{int(canonical['trace_mismatches']):,} / "
                f"{int(reverse['trace_mismatches']):,}"
            ),
            (
                f"{int(canonical['candidate_signature_mismatches']):,} / "
                f"{int(reverse['candidate_signature_mismatches']):,}"
            ),
            "PASS" if is_safe else "FAIL",
        ]
        for index, (center, value) in enumerate(zip(column_centers, values)):
            axis.text(
                column_edges[0] + 0.014 if index == 0 else center,
                y,
                value,
                ha="left" if index == 0 else "center",
                va="center",
                fontsize=6.5,
                fontweight="bold" if index in (0, 5) else "normal",
                color=(
                    DARK_GREEN
                    if index == 5 and is_safe
                    else VERMILION
                    if index == 5
                    else INK
                ),
            )
    for edge in column_edges[1:-1]:
        axis.plot([edge, edge], [0.080, header_y0 + header_height], color=LIGHT_GRAY, linewidth=0.45)
    axis.text(
        0.995,
        0.035,
        "C / R: canonical / reverse replay order",
        ha="right",
        va="center",
        fontsize=6.5,
        color=MID_GRAY,
    )


def build_figure3(
    ablation_rows: list[dict[str, str]],
    oracle: dict[str, Any],
    mutation: dict[str, int],
) -> plt.Figure:
    fig = plt.figure(figsize=(WIDTH_IN, FIG3_HEIGHT_IN))
    audit_axis = fig.add_axes([0.045, 0.045, 0.91, 0.91])
    draw_semantic_audit_matrix(audit_axis, ablation_rows, oracle, mutation)
    return fig


PAIR_CODES = ["R-I", "R-S", "R-F", "I-S", "I-F", "S-F"]


def add_panel_heading(axis: plt.Axes, label: str, title: str) -> None:
    add_panel_label(axis, label)
    axis.text(
        0.15,
        1.02,
        title,
        transform=axis.transAxes,
        fontsize=7.0,
        fontweight="bold",
        ha="left",
        va="bottom",
        color=INK,
        clip_on=False,
    )


def draw_pair_delay(axis: plt.Axes, rows: list[dict[str, str]]) -> None:
    values = [100.0 * float(row["delayed_minus_immediate_rate"]) for row in rows]
    y = np.arange(len(rows))[::-1]
    axis.axvline(0, color=INK, linewidth=0.7, zorder=1)
    axis.hlines(y, 0, values, color=LIGHT_GRAY, linewidth=1.15, zorder=1)
    for code, yi, value in zip(PAIR_CODES, y, values):
        is_decrease = value < 0
        axis.scatter(
            [value],
            [yi],
            s=31,
            facecolor=BLUE if is_decrease else ORANGE,
            edgecolor=INK,
            linewidth=0.55,
            marker="o" if is_decrease else "D",
            zorder=3,
        )
        axis.text(
            -9.72,
            yi,
            code,
            ha="left",
            va="center",
            fontsize=6.5,
            fontweight="bold",
            color=INK,
        )
        axis.text(
            value + 0.28,
            yi,
            f"{value:+.2f}",
            ha="left",
            va="center",
            fontsize=6.5,
            color=BLUE if is_decrease else VERMILION,
        )
    axis.set_yticks([])
    axis.set_ylim(-0.62, len(rows) - 0.01)
    axis.set_xlim(-9.9, 1.75)
    axis.set_xticks([-8, -4, 0])
    axis.set_xlabel("Delayed - immediate (pp)", labelpad=2)
    style_axis(axis)
    axis.spines["left"].set_visible(False)
    axis.tick_params(axis="y", length=0)
    add_panel_heading(axis, "(a)", "All-view effect census")


def draw_cover_quantifiers(
    axis: plt.Axes,
    leave_rows: list[dict[str, str]],
    cover_stats: dict[str, int],
) -> None:
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    union_size = cover_stats["union_minimum_suite_size"]
    robust_size = cover_stats["robust_minimum_suite_size"]
    axis.scatter(
        [0.20],
        [0.74],
        s=205,
        facecolor=BLUE,
        edgecolor=INK,
        linewidth=0.65,
        zorder=3,
    )
    axis.text(
        0.20,
        0.74,
        str(union_size),
        ha="center",
        va="center",
        fontsize=7.5,
        fontweight="bold",
        color=WHITE,
        zorder=4,
    )
    axis.scatter(
        [0.80],
        [0.74],
        s=205,
        facecolor=ORANGE,
        edgecolor=INK,
        linewidth=0.65,
        marker="s",
        zorder=3,
    )
    axis.text(
        0.80,
        0.74,
        str(robust_size),
        ha="center",
        va="center",
        fontsize=7.5,
        fontweight="bold",
        color=INK,
        zorder=4,
    )
    axis.add_patch(
        FancyArrowPatch(
            (0.34, 0.74),
            (0.66, 0.74),
            arrowstyle="-|>",
            mutation_scale=8,
            linewidth=0.8,
            color=MID_GRAY,
        )
    )
    axis.text(
        0.20,
        0.53,
        "Any-view union",
        ha="center",
        va="center",
        fontsize=6.5,
        color=BLUE,
        fontweight="bold",
    )
    axis.text(
        0.80,
        0.53,
        "All-view intersection",
        ha="center",
        va="center",
        fontsize=6.5,
        color=VERMILION,
        fontweight="bold",
    )
    require(len(leave_rows) == 4, "Expected four leave-one-agent-out rows.")
    axis.plot([0.02, 0.98], [0.43, 0.43], color=LIGHT_GRAY, linewidth=0.6)
    axis.text(
        0.02,
        0.31,
        "Leave one agent out: 1 mechanism (4/4)",
        ha="left",
        va="center",
        fontsize=6.5,
        color=INK,
    )
    axis.text(
        0.02,
        0.16,
        (
            f"{cover_stats['robust_nonzero_kernels']:,} / "
            f"{cover_stats['kernel_count']:,} retain a robust bit; "
            f"{cover_stats['robust_full_kernels']} retain all six"
        ),
        ha="left",
        va="center",
        fontsize=6.5,
        color=MID_GRAY,
    )
    add_panel_heading(axis, "(b)", "The quantifier changes the answer")


def draw_cover_matrix(axis: plt.Axes, robust_masks: dict[str, int]) -> None:
    row_names = ["brake_10377", "brake_10387", "combined"]
    masks = [
        robust_masks["brake_10377"],
        robust_masks["brake_10387"],
        robust_masks["brake_10377"] | robust_masks["brake_10387"],
    ]
    row_y = [1.35, 0.75, 0.05]
    column_x = np.linspace(0.0, 5.0, 6)
    axis.set_xlim(-1.05, 5.55)
    axis.set_ylim(-0.30, 1.78)
    axis.axis("off")

    for x, code in zip(column_x, PAIR_CODES):
        axis.text(
            x,
            1.58,
            code,
            ha="center",
            va="center",
            fontsize=6.5,
            fontweight="bold",
            color=INK,
        )
    for row_index, (name, mask, y) in enumerate(zip(row_names, masks, row_y)):
        axis.text(
            -0.18,
            y,
            name,
            ha="right",
            va="center",
            fontsize=6.5,
            fontweight="bold" if row_index == 2 else "normal",
            color=BLUE if row_index == 2 else INK,
        )
        for column_index, x in enumerate(column_x):
            covered = bool(mask & (1 << column_index))
            if row_index == 2:
                axis.scatter(
                    [x],
                    [y],
                    s=37,
                    marker="s",
                    facecolor=BLUE,
                    edgecolor=INK,
                    linewidth=0.45,
                    zorder=3,
                )
            elif covered:
                axis.scatter(
                    [x],
                    [y],
                    s=39,
                    marker="o",
                    facecolor=GREEN,
                    edgecolor=INK,
                    linewidth=0.45,
                    zorder=3,
                )
            else:
                axis.scatter(
                    [x],
                    [y],
                    s=35,
                    marker="x",
                    color=VERMILION,
                    linewidth=1.1,
                    zorder=3,
                )
    axis.plot([-0.98, 5.47], [0.39, 0.39], color=BLUE, linewidth=0.75)
    add_panel_label(axis, "(c)")
    axis.text(
        0.5,
        1.02,
        "Two mechanisms cover all six pairs in every view",
        transform=axis.transAxes,
        fontsize=7.0,
        fontweight="bold",
        ha="center",
        va="bottom",
        color=INK,
        clip_on=False,
    )


def build_figure4(
    pair_rows: list[dict[str, str]],
    leave_rows: list[dict[str, str]],
    robust_masks: dict[str, int],
    cover_stats: dict[str, int],
) -> plt.Figure:
    fig = plt.figure(figsize=(WIDTH_IN, FIG4_HEIGHT_IN))
    grid = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.28, 0.92],
        width_ratios=[1.0, 1.0],
        hspace=0.43,
        wspace=0.25,
        left=0.06,
        right=0.94,
        bottom=0.070,
        top=0.915,
    )
    delay_axis = fig.add_subplot(grid[0, 0])
    quantifier_axis = fig.add_subplot(grid[0, 1])
    cover_axis = fig.add_subplot(grid[1, :])
    draw_pair_delay(delay_axis, pair_rows)
    draw_cover_quantifiers(quantifier_axis, leave_rows, cover_stats)
    draw_cover_matrix(cover_axis, robust_masks)
    cover_position = cover_axis.get_position()
    fig._horizontal_balance_band = (  # type: ignore[attr-defined]
        max(0.0, cover_position.y0 - 0.025),
        min(1.0, cover_position.y1 + 0.055),
    )
    return fig


def audit_figure(fig: plt.Figure, expected_height: float) -> list[str]:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    canvas = fig.bbox
    text_items = [
        artist
        for artist in fig.findobj(match=lambda item: isinstance(item, Text))
        if artist.get_visible() and artist.get_text()
    ]
    sizes = [float(item.get_fontsize()) for item in text_items]
    require(
        sizes and min(sizes) >= MIN_FONT_PT,
        f"Text below the {MIN_FONT_PT:.1f} pt minimum.",
    )
    outside: list[str] = []
    for item in text_items:
        bounds = item.get_window_extent(renderer=renderer)
        if (
            bounds.x0 < canvas.x0 - 0.5
            or bounds.y0 < canvas.y0 - 0.5
            or bounds.x1 > canvas.x1 + 0.5
            or bounds.y1 > canvas.y1 + 0.5
        ):
            outside.append(item.get_text())
    require(not outside, f"Text outside the canvas: {outside}")
    width, height = fig.get_size_inches()
    require(abs(width - WIDTH_IN) < 1e-6, f"Unexpected width: {width}")
    require(abs(height - expected_height) < 1e-6, f"Unexpected height: {height}")
    return [
        f"minimum text size: {min(sizes):.1f} pt",
        "text outside canvas: none",
    ]


def inspect_pdf(path: Path) -> list[str]:
    reader = PdfReader(str(path))
    require(len(reader.pages) == 1, f"{path.name} is not a one-page figure.")
    type3: list[str] = []
    unembedded: list[str] = []
    image_xobjects = 0
    page = reader.pages[0]
    resources = page["/Resources"].get_object()
    font_dict = resources.get("/Font", {})
    if hasattr(font_dict, "get_object"):
        font_dict = font_dict.get_object()
    for _, ref in font_dict.items():
        font = ref.get_object()
        subtype = str(font.get("/Subtype"))
        base = str(font.get("/BaseFont"))
        if subtype == "/Type3":
            type3.append(base)
        descriptor = None
        if "/FontDescriptor" in font:
            descriptor = font["/FontDescriptor"].get_object()
        elif "/DescendantFonts" in font:
            descendants = font["/DescendantFonts"].get_object()
            if descendants:
                descendant = descendants[0].get_object()
                if "/FontDescriptor" in descendant:
                    descriptor = descendant["/FontDescriptor"].get_object()
        if descriptor is not None and not any(
            key in descriptor for key in ("/FontFile", "/FontFile2", "/FontFile3")
        ):
            unembedded.append(base)
    xobjects = resources.get("/XObject", {})
    if hasattr(xobjects, "get_object"):
        xobjects = xobjects.get_object()
    for _, ref in xobjects.items():
        if str(ref.get_object().get("/Subtype")) == "/Image":
            image_xobjects += 1
    require(not type3, f"Type 3 fonts in {path.name}: {type3}")
    require(not unembedded, f"Unembedded fonts in {path.name}: {unembedded}")
    require(image_xobjects == 0, f"Raster image XObjects in {path.name}.")
    return [
        "embedded fonts: PASS",
        "Type 3 fonts: none",
        "raster image XObjects: none",
    ]


def inspect_horizontal_balance(
    image: Image.Image,
    band: tuple[float, float],
) -> list[str]:
    width, height = image.size
    lower, upper = band
    y0 = max(0, int(np.floor((1.0 - upper) * height)))
    y1 = min(height, int(np.ceil((1.0 - lower) * height)))
    grayscale = np.asarray(image.convert("L"))
    visible = grayscale[y0:y1, :] < 248
    active_columns = np.flatnonzero(visible.any(axis=0))
    require(active_columns.size > 0, "No visible ink in the balance-audit band.")
    left_margin = int(active_columns[0])
    right_margin = int(width - 1 - active_columns[-1])
    imbalance = 100.0 * abs(left_margin - right_margin) / width
    require(
        imbalance <= 3.0,
        (
            "Cover-ribbon horizontal imbalance exceeds 3%: "
            f"left={left_margin}px, right={right_margin}px, "
            f"difference={imbalance:.2f}%."
        ),
    )
    return [
        (
            "cover-ribbon visible margins: "
            f"{left_margin}px left / {right_margin}px right "
            f"(difference {imbalance:.2f}% of canvas)"
        )
    ]


def export_figure(
    fig: plt.Figure,
    output_dir: Path,
    basename: str,
    expected_height: float,
) -> dict[str, Path]:
    audit_notes = audit_figure(fig, expected_height)
    balance_band = getattr(fig, "_horizontal_balance_band", None)
    pdf_path = output_dir / f"{basename}.pdf"
    png_path = output_dir / f"{basename}.png"
    grayscale_path = output_dir / f"{basename}_grayscale.png"
    fig.savefig(
        pdf_path,
        metadata={"Title": basename, "Creator": SCRIPT_PATH.name},
    )
    fig.savefig(png_path, dpi=PNG_DPI, metadata={"Title": basename})
    plt.close(fig)
    with Image.open(png_path) as image:
        expected = (round(WIDTH_IN * PNG_DPI), round(expected_height * PNG_DPI))
        require(
            image.size == expected,
            f"Unexpected PNG dimensions for {basename}: {image.size} != {expected}",
        )
        gray = ImageOps.grayscale(image).convert("RGB")
        gray.save(grayscale_path, dpi=(PNG_DPI, PNG_DPI))
        if balance_band is not None:
            audit_notes.extend(inspect_horizontal_balance(image, balance_band))
    pdf_notes = inspect_pdf(pdf_path)
    require(all(path.is_file() and path.stat().st_size > 0 for path in (pdf_path, png_path, grayscale_path)), "Missing output.")
    return {
        "pdf": pdf_path,
        "png": png_path,
        "grayscale": grayscale_path,
        "notes": audit_notes + pdf_notes,
    }


def write_qa(
    output_dir: Path,
    provenance: dict[str, str],
    outputs: dict[str, dict[str, Any]],
) -> Path:
    lines = [
        "# Explanatory figures QA",
        "",
        "Both figures are exact-data visualizations from frozen, distance-fix inputs.",
        "They contain no inferential error bars or population-level claims.",
        "",
        "## Provenance",
        "",
    ]
    for path, digest in provenance.items():
        lines.append(f"- `{path}`: `{digest}`")
    for label, record in outputs.items():
        lines.extend(
            [
                "",
                f"## {label}",
                "",
                f"- PDF: `{record['pdf'].name}`",
                f"- PNG: `{record['png'].name}`",
                f"- grayscale: `{record['grayscale'].name}`",
            ]
        )
        lines.extend(f"- {note}" for note in record["notes"])
    lines.extend(
        [
            "",
            "## Visual-review checklist",
            "",
            "- Review at the final 4.72-inch width.",
            "- Confirm all panel labels align and direct labels do not touch marks.",
            "- Confirm color and grayscale versions retain the same ordering.",
            "- Confirm the Figure 3 consequence matrix matches both replay orders.",
            "- Confirm the Figure 4 cover glyphs match robust masks 59 and 31.",
            "",
        ]
    )
    qa_path = output_dir / "EXPLANATORY_FIGURES_QA.md"
    qa_path.write_text("\n".join(lines), encoding="utf-8")
    return qa_path


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_style()
    provenance = validate_provenance()
    ablation_rows, oracle = load_ablation()
    mutation = load_mutation_sensitivity()
    pair_rows, leave_rows, robust_masks, cover_stats = load_robustness()
    outputs = {
        "Figure 3": export_figure(
            build_figure3(ablation_rows, oracle, mutation),
            output_dir,
            "fig3_semantic_safety_audit",
            FIG3_HEIGHT_IN,
        ),
        "Figure 4": export_figure(
            build_figure4(pair_rows, leave_rows, robust_masks, cover_stats),
            output_dir,
            "fig4_robustness_decomposition",
            FIG4_HEIGHT_IN,
        ),
    }
    qa_path = write_qa(output_dir, provenance, outputs)
    for label, record in outputs.items():
        print(f"{label}: {record['pdf']}")
        print(f"{label} PNG: {record['png']}")
        print(f"{label} grayscale: {record['grayscale']}")
    print(f"QA: {qa_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
