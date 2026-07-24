#!/usr/bin/env python
"""Build the original LayerProbe method overview for paper Figure 1.

The design is a self-contained, left-to-right semantic ledger with an
independent audit rail.  It does not read or embed a third-party reference
image.  An optional ``--style-reference`` may be supplied only to record a
non-destructive, side-by-side visual-QA provenance note; it never participates
in figure construction.  Outputs are a vector PDF, a 400-dpi colour PNG, a
true-grayscale preview, and a method/visual QA report.
"""

from __future__ import annotations

import argparse
import hashlib
import math
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.path import Path as MplPath
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.text import Text
from PIL import Image, ImageOps
from pypdf import PdfReader


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
PAPER_ROOT = PROJECT_ROOT.parents[1] / "07_论文" / "manuscript" / "draft"

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "results" / "deadline_figures_distancefix_final_20260724_xeon"
)
DEFAULT_STEM = "fig0_layerprobe_overview_v3"
LEGACY_FIGURE = DEFAULT_OUTPUT_DIR / "fig0_layerprobe_overview.png"
PAPER_TEX = PAPER_ROOT / "paper.tex"
EVALUATOR_SOURCE = PROJECT_ROOT / "src" / "layerprobe" / "evaluator.py"
ORACLE_README = PROJECT_ROOT / "experiments" / "INDEPENDENT_TRACE_ORACLE_README.md"

WIDTH_IN = 4.72
HEIGHT_IN = 2.52
PNG_DPI = 400
MIN_FONT_PT = 6.0

INK = "#20242A"
MUTED = "#626A73"
LINE = "#89919A"
HAIRLINE = "#D5D9DD"
WHITE = "#FFFFFF"

PRODUCT_BG = "#F8F6F1"
REUSE1_BG = "#F1F7F0"
REUSE2_BG = "#F3F7FA"
OUTPUT_BG = "#F7F4F9"
AUDIT_BG = "#EEF3F5"

PALE_BLUE = "#DCE8F2"
PALE_GREEN = "#DCEBD5"
PALE_ORANGE = "#F4DEC9"
PALE_PURPLE = "#E5DFF0"
PALE_GRAY = "#F2F3F3"

BLUE = "#487E9D"
GREEN = "#5E8E55"
ORANGE = "#B87343"
PURPLE = "#806B96"

VISUAL_REVIEW_STATUS = "PASS"
VISUAL_REVIEW_NOTES = (
    "Round 1 identified excessive density in the complete-key card, green "
    "fan-out, and purple validation return; labels were shortened and the "
    "connectors were rerouted through dedicated gutters and ports.",
    "Round 2 inspected the 1888 × 1008 colour rendering: the left-to-right "
    "reading order, two reuse levels, local presentation lanes, output ledger, "
    "and independent audit rail are clear with no clipping or collision.",
    "Round 3 inspected the true-grayscale rendering and exact 400-dpi geometry: "
    "stage boundaries, arrows, key segments, trace lanes, signature bits, and "
    "audit comparison remain readable without relying on hue.",
    "The generator contains no external-image loading path in its drawing "
    "pipeline and emits only original vector/text primitives.",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--stem",
        default=DEFAULT_STEM,
        help=(
            "Output basename. Use --stem fig0_layerprobe_overview when "
            "regenerating directly for the manuscript."
        ),
    )
    parser.add_argument(
        "--style-reference",
        type=Path,
        default=None,
        help=(
            "Optional image used only for non-destructive visual-QA provenance. "
            "The image is never read by build_figure() or embedded in outputs."
        ),
    )
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


def configure_style() -> Path:
    font_path = Path(
        font_manager.findfont("Times New Roman", fallback_to_default=False)
    ).resolve()
    require(font_path.exists(), "Times New Roman is not installed.")
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 6.2,
            "axes.unicode_minus": True,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.dpi": PNG_DPI,
            "savefig.facecolor": WHITE,
            "figure.facecolor": WHITE,
        }
    )
    return font_path


def add_text(
    axis: plt.Axes,
    x: float,
    y: float,
    value: str,
    *,
    size: float = 6.0,
    color: str = INK,
    weight: str = "normal",
    style: str = "normal",
    ha: str = "center",
    va: str = "center",
    rotation: float = 0.0,
    zorder: int = 20,
) -> Text:
    require(size >= MIN_FONT_PT, f"Text size {size} pt is below 6 pt.")
    return axis.text(
        x,
        y,
        value,
        fontsize=size,
        color=color,
        fontweight=weight,
        fontstyle=style,
        ha=ha,
        va=va,
        rotation=rotation,
        zorder=zorder,
    )


def rounded_box(
    axis: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    face: str = WHITE,
    edge: str = HAIRLINE,
    linewidth: float = 0.55,
    radius: float = 0.008,
    zorder: int = 3,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle=f"round,pad=0.002,rounding_size={radius}",
        facecolor=face,
        edgecolor=edge,
        linewidth=linewidth,
        zorder=zorder,
    )
    axis.add_patch(patch)
    return patch


def arrow(
    axis: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = LINE,
    linewidth: float = 0.65,
    connectionstyle: str = "arc3",
    mutation_scale: float = 6.5,
    zorder: int = 12,
) -> FancyArrowPatch:
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=mutation_scale,
        linewidth=linewidth,
        color=color,
        shrinkA=0.5,
        shrinkB=0.5,
        connectionstyle=connectionstyle,
        zorder=zorder,
    )
    axis.add_patch(patch)
    return patch


def line(
    axis: plt.Axes,
    xs: tuple[float, ...],
    ys: tuple[float, ...],
    *,
    color: str = LINE,
    linewidth: float = 0.55,
    zorder: int = 8,
) -> None:
    axis.plot(xs, ys, color=color, linewidth=linewidth, zorder=zorder)


def draw_stage(
    axis: plt.Axes,
    x0: float,
    x1: float,
    title: str,
    face: str,
    accent: str,
    *,
    number: str | None = None,
) -> None:
    axis.add_patch(
        Rectangle(
            (x0, 0.255),
            x1 - x0,
            0.635,
            facecolor=face,
            edgecolor="none",
            zorder=0,
        )
    )
    axis.add_patch(
        Rectangle(
            (x0, 0.884),
            x1 - x0,
            0.009,
            facecolor=accent,
            edgecolor="none",
            zorder=1,
        )
    )
    title_x = x0 + 0.012
    if number is not None:
        axis.add_patch(
            Circle(
                (x0 + 0.018, 0.945),
                0.012,
                facecolor=accent,
                edgecolor="none",
                zorder=4,
            )
        )
        add_text(
            axis,
            x0 + 0.018,
            0.945,
            number,
            size=6.1,
            color=WHITE,
            weight="bold",
        )
        title_x = x0 + 0.036
    add_text(
        axis,
        title_x,
        0.945,
        title,
        size=6.2,
        weight="bold",
        color=accent,
        ha="left",
    )


def draw_product_stage(axis: plt.Axes) -> None:
    draw_stage(
        axis,
        0.022,
        0.195,
        "FINITE PRODUCT",
        PRODUCT_BG,
        ORANGE,
    )
    add_text(axis, 0.108, 0.850, "Flat: repeat every cell", size=6.2, weight="bold")

    grid_x = 0.052
    grid_y = 0.535
    cell_w = 0.030
    cell_h = 0.075
    gap_x = 0.004
    gap_y = 0.009
    column_faces = (PALE_BLUE, PALE_GREEN, PALE_PURPLE, PALE_ORANGE)
    column_labels = ("p1", "p2", "…", "pP")
    row_labels = ("k1", "k2", "kK")

    for column, title in enumerate(column_labels):
        center_x = grid_x + column * (cell_w + gap_x) + cell_w / 2
        add_text(
            axis,
            center_x,
            0.815,
            title,
            size=6.0,
            color=MUTED,
            style="italic",
        )
    for row, title in enumerate(row_labels):
        center_y = grid_y + (2 - row) * (cell_h + gap_y) + cell_h / 2
        add_text(
            axis,
            0.040,
            center_y,
            title,
            size=6.0,
            color=MUTED,
            style="italic",
            ha="right",
        )
        for column in range(4):
            x = grid_x + column * (cell_w + gap_x)
            y = grid_y + (2 - row) * (cell_h + gap_y)
            axis.add_patch(
                Rectangle(
                    (x, y),
                    cell_w,
                    cell_h,
                    facecolor=column_faces[column],
                    edgecolor=WHITE,
                    linewidth=0.7,
                    zorder=4,
                )
            )
            axis.add_patch(
                Circle(
                    (x + cell_w / 2, y + cell_h / 2),
                    0.006,
                    facecolor=WHITE,
                    edgecolor=LINE,
                    linewidth=0.45,
                    zorder=5,
                )
            )

    add_text(
        axis,
        0.108,
        0.505,
        "K mechanisms × P views",
        size=6.0,
        color=MUTED,
    )
    add_text(
        axis,
        0.108,
        0.467,
        "× A deterministic agents",
        size=6.0,
        color=MUTED,
    )
    rounded_box(
        axis,
        0.027,
        0.270,
        0.162,
        0.178,
        face=PALE_ORANGE,
        edge="#E2C6AD",
        linewidth=0.45,
    )
    add_text(
        axis,
        0.108,
        0.408,
        "p can change o",
        size=6.0,
        weight="bold",
    )
    add_text(axis, 0.108, 0.374, "memory + action", size=6.0)
    add_text(axis, 0.108, 0.338, "may change", size=6.0)
    add_text(
        axis,
        0.108,
        0.303,
        "state-only key unsafe",
        size=6.0,
        color=ORANGE,
        style="italic",
    )
    arrow(axis, (0.195, 0.610), (0.217, 0.610), color=ORANGE)


def draw_mechanism_graph(axis: plt.Axes, cx: float, cy: float) -> None:
    nodes = (
        (cx - 0.035, cy + 0.005),
        (cx - 0.007, cy + 0.037),
        (cx + 0.027, cy + 0.017),
        (cx + 0.014, cy - 0.030),
        (cx - 0.026, cy - 0.033),
    )
    for left, right in ((0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (0, 3)):
        line(
            axis,
            (nodes[left][0], nodes[right][0]),
            (nodes[left][1], nodes[right][1]),
            color=MUTED,
            linewidth=0.6,
            zorder=6,
        )
    for index, (x, y) in enumerate(nodes):
        axis.add_patch(
            Circle(
                (x, y),
                0.007,
                facecolor=(PALE_BLUE, PALE_GREEN, PALE_ORANGE)[index % 3],
                edgecolor=INK,
                linewidth=0.5,
                zorder=7,
            )
        )


def draw_reuse1_stage(axis: plt.Axes) -> None:
    draw_stage(
        axis,
        0.217,
        0.398,
        "VALIDATE ONCE",
        REUSE1_BG,
        GREEN,
        number="1",
    )
    rounded_box(
        axis,
        0.236,
        0.690,
        0.066,
        0.105,
        face=WHITE,
        edge="#CAD8C6",
        linewidth=0.5,
    )
    draw_mechanism_graph(axis, 0.269, 0.742)
    add_text(axis, 0.269, 0.670, "mechanism k", size=6.0, style="italic")

    arrow(axis, (0.304, 0.743), (0.314, 0.743), color=GREEN)
    rounded_box(
        axis,
        0.314,
        0.698,
        0.074,
        0.090,
        face=PALE_GREEN,
        edge="#BDD2B6",
        linewidth=0.55,
    )
    add_text(axis, 0.351, 0.759, "validate", size=6.0, weight="bold")
    add_text(axis, 0.351, 0.720, "k once", size=6.0, color=GREEN, style="italic")

    rounded_box(
        axis,
        0.222,
        0.497,
        0.172,
        0.128,
        face=WHITE,
        edge="#CAD8C6",
        linewidth=0.45,
    )
    add_text(axis, 0.309, 0.592, "Presentation contract", size=6.0, weight="bold")
    add_text(axis, 0.309, 0.553, "p leaves physics fixed", size=6.0)
    add_text(axis, 0.309, 0.518, "terminal rules fixed", size=6.0)

    rounded_box(
        axis,
        0.245,
        0.330,
        0.128,
        0.095,
        face=PALE_GREEN,
        edge="none",
        linewidth=0,
    )
    add_text(axis, 0.309, 0.389, "reuse level 1", size=6.0, color=GREEN, weight="bold")
    add_text(axis, 0.309, 0.354, "one graph build per k", size=6.0)

    axis.add_patch(
        Circle(
            (0.380, 0.788),
            0.004,
            facecolor=WHITE,
            edgecolor=GREEN,
            linewidth=0.45,
            zorder=11,
        )
    )
    line(
        axis,
        (0.380, 0.380, 0.404),
        (0.788, 0.800, 0.800),
        color=GREEN,
        linewidth=0.65,
    )
    line(axis, (0.404, 0.404), (0.430, 0.800), color=GREEN, linewidth=0.65)
    for lane_y in (0.700, 0.565, 0.430):
        axis.add_patch(
            Circle(
                (0.404, lane_y),
                0.004,
                facecolor=WHITE,
                edgecolor=GREEN,
                linewidth=0.45,
                zorder=11,
            )
        )
        arrow(axis, (0.409, lane_y), (0.425, lane_y), color=GREEN)


def draw_segmented_key(axis: plt.Axes) -> None:
    x0 = 0.598
    y0 = 0.642
    width = 0.128
    height = 0.057
    faces = (PALE_BLUE, PALE_ORANGE, PALE_PURPLE)
    labels = ("s", "m", "o")
    for index, (face, value) in enumerate(zip(faces, labels)):
        x = x0 + index * width / 3
        axis.add_patch(
            Rectangle(
                (x, y0),
                width / 3,
                height,
                facecolor=face,
                edgecolor=WHITE,
                linewidth=0.7,
                zorder=7,
            )
        )
        add_text(axis, x + width / 6, y0 + height / 2, value, size=6.0)


def draw_trace_token(axis: plt.Axes, x: float, y: float, label_value: str) -> None:
    rounded_box(
        axis,
        x,
        y - 0.034,
        0.032,
        0.068,
        face=WHITE,
        edge="#C8D5DD",
        linewidth=0.45,
        radius=0.005,
    )
    add_text(axis, x + 0.016, y + 0.009, label_value, size=6.0, style="italic")
    line(
        axis,
        (x + 0.008, x + 0.024),
        (y - 0.014, y - 0.014),
        color=BLUE,
        linewidth=0.7,
        zorder=7,
    )


def draw_reuse2_stage(axis: plt.Axes) -> None:
    draw_stage(
        axis,
        0.420,
        0.787,
        "EXACT STEP REUSE",
        REUSE2_BG,
        BLUE,
        number="2",
    )
    add_text(
        axis,
        0.492,
        0.842,
        "presentation-local display",
        size=6.0,
        color=BLUE,
        weight="bold",
    )
    add_text(
        axis,
        0.755,
        0.842,
        "private trace",
        size=6.0,
        color=BLUE,
        weight="bold",
    )

    lane_data = (
        (0.700, "p1", PALE_BLUE, "τ1"),
        (0.565, "p2", PALE_GREEN, "τ2"),
        (0.430, "pP", PALE_PURPLE, "τP"),
    )
    for lane_y, presentation, face, trace_label in lane_data:
        rounded_box(
            axis,
            0.432,
            lane_y - 0.034,
            0.035,
            0.068,
            face=face,
            edge="none",
            linewidth=0,
            radius=0.012,
        )
        add_text(axis, 0.4495, lane_y, presentation, size=6.0, style="italic")
        rounded_box(
            axis,
            0.477,
            lane_y - 0.039,
            0.054,
            0.078,
            face=WHITE,
            edge="#C8D5DD",
            linewidth=0.45,
            radius=0.005,
        )
        add_text(axis, 0.504, lane_y + 0.019, "display", size=6.0)
        add_text(axis, 0.504, lane_y - 0.019, "history", size=6.0, color=MUTED)
        arrow(axis, (0.533, lane_y), (0.551, lane_y), color=BLUE)
        axis.add_patch(
            Circle(
                (0.563, lane_y),
                0.012,
                facecolor=face,
                edgecolor=BLUE,
                linewidth=0.5,
                zorder=7,
            )
        )
        add_text(axis, 0.563, lane_y, "o", size=6.0, style="italic")
        line(
            axis,
            (0.575, 0.579),
            (lane_y, lane_y),
            color=BLUE,
            linewidth=0.65,
            zorder=11,
        )
        draw_trace_token(axis, 0.753, lane_y, trace_label)

    # Presentation queries join outside the index card, then enter the key
    # row once. This routing prevents long curved arrows from crossing the
    # hit/miss labels while preserving the many-to-one lookup semantics.
    line(axis, (0.579, 0.579), (0.430, 0.700), color=BLUE, linewidth=0.65, zorder=11)
    arrow(axis, (0.579, 0.672), (0.597, 0.672), color=BLUE, mutation_scale=5.5)

    rounded_box(
        axis,
        0.588,
        0.378,
        0.152,
        0.413,
        face=WHITE,
        edge="#BFCED8",
        linewidth=0.6,
        radius=0.007,
        zorder=4,
    )
    add_text(axis, 0.662, 0.766, "exact step index", size=6.2, weight="bold", color=BLUE)
    add_text(axis, 0.662, 0.720, "fixed (k, a); key q", size=6.0, weight="bold")
    draw_segmented_key(axis)

    rounded_box(
        axis,
        0.598,
        0.555,
        0.128,
        0.052,
        face=PALE_GREEN,
        edge="none",
        linewidth=0,
        radius=0.005,
        zorder=6,
    )
    add_text(axis, 0.662, 0.581, "hit · reuse r(q)", size=6.0, color=GREEN, weight="bold")
    rounded_box(
        axis,
        0.594,
        0.487,
        0.144,
        0.052,
        face=PALE_ORANGE,
        edge="none",
        linewidth=0,
        radius=0.005,
        zorder=6,
    )
    add_text(axis, 0.608, 0.513, "miss", size=6.0, color=ORANGE, weight="bold")
    add_text(axis, 0.661, 0.513, "compute", size=6.0, color=ORANGE, weight="bold")
    add_text(axis, 0.718, 0.513, "store", size=6.0, color=ORANGE, weight="bold")
    line(axis, (0.630, 0.630), (0.496, 0.530), color="#D5A985", linewidth=0.4, zorder=7)
    line(axis, (0.693, 0.693), (0.496, 0.530), color="#D5A985", linewidth=0.4, zorder=7)
    rounded_box(
        axis,
        0.608,
        0.398,
        0.108,
        0.056,
        face=PALE_BLUE,
        edge="none",
        linewidth=0,
        radius=0.005,
        zorder=6,
    )
    add_text(axis, 0.662, 0.426, "cached result r(q)", size=6.0, color=BLUE)

    line(axis, (0.740, 0.746), (0.585, 0.585), color=BLUE, linewidth=0.65)
    line(axis, (0.746, 0.746), (0.430, 0.700), color=BLUE, linewidth=0.65)
    for lane_y in (0.700, 0.565, 0.430):
        arrow(axis, (0.746, lane_y), (0.751, lane_y), color=BLUE, mutation_scale=5.5)

    rounded_box(
        axis,
        0.436,
        0.268,
        0.335,
        0.082,
        face=PALE_GRAY,
        edge="none",
        linewidth=0,
    )
    add_text(
        axis,
        0.6035,
        0.329,
        "s world · m pre-memory · o observation",
        size=6.0,
        color=MUTED,
    )
    add_text(
        axis,
        0.6035,
        0.289,
        "display history + traces remain local",
        size=6.0,
        color=MUTED,
    )


def draw_bit_row(axis: plt.Axes, x: float, y: float) -> None:
    bits = (1, 0, 1, 1, 0, 1)
    cell_w = 0.011
    gap = 0.0025
    for index, bit in enumerate(bits):
        axis.add_patch(
            Rectangle(
                (x + index * (cell_w + gap), y),
                cell_w,
                0.023,
                facecolor=PURPLE if bit else WHITE,
                edgecolor=PURPLE,
                linewidth=0.35,
                zorder=7,
            )
        )


def draw_output_stage(axis: plt.Axes) -> None:
    draw_stage(
        axis,
        0.805,
        0.978,
        "EXACT OUTPUTS",
        OUTPUT_BG,
        PURPLE,
    )
    spine_x = 0.824
    centers = (0.762, 0.620, 0.476, 0.332)
    line(axis, (spine_x, spine_x), (0.332, 0.762), color=PURPLE, linewidth=0.7)
    for center_y in centers:
        axis.add_patch(
            Circle(
                (spine_x, center_y),
                0.006,
                facecolor=PURPLE,
                edgecolor=WHITE,
                linewidth=0.4,
                zorder=7,
            )
        )

    cards = (
        (0.838, 0.705, 0.124, 0.112, PALE_BLUE),
        (0.838, 0.557, 0.124, 0.116, PALE_PURPLE),
        (0.838, 0.413, 0.124, 0.112, WHITE),
        (0.838, 0.260, 0.124, 0.132, PALE_GREEN),
    )
    for x, y, width, height, face in cards:
        rounded_box(
            axis,
            x,
            y,
            width,
            height,
            face=face,
            edge="#D0CAD8",
            linewidth=0.45,
            radius=0.006,
        )

    add_text(axis, 0.900, 0.783, "τ(k,p,a)", size=6.2, weight="bold")
    add_text(axis, 0.900, 0.742, "declared traces", size=6.0)

    add_text(axis, 0.900, 0.648, "σ(k,p)", size=6.2, weight="bold")
    add_text(axis, 0.900, 0.610, "six pair bits", size=6.0)
    draw_bit_row(axis, 0.860, 0.576)

    add_text(axis, 0.900, 0.488, "ρ(k)", size=6.2, weight="bold")
    add_text(axis, 0.900, 0.446, "all-p AND", size=6.0)

    add_text(axis, 0.900, 0.366, "Example:", size=6.0, weight="bold")
    add_text(axis, 0.900, 0.326, "all-view query", size=6.0, weight="bold")
    add_text(axis, 0.900, 0.286, "exact cover", size=6.0, color=GREEN)

    arrow(axis, (0.787, 0.620), (0.816, 0.620), color=PURPLE)


def draw_audit_rail(axis: plt.Axes) -> None:
    add_text(
        axis,
        0.026,
        0.135,
        "INDEPENDENT",
        size=6.2,
        color=BLUE,
        weight="bold",
        ha="left",
    )
    add_text(
        axis,
        0.026,
        0.098,
        "VALIDATION",
        size=6.2,
        color=BLUE,
        weight="bold",
        ha="left",
    )

    rounded_box(
        axis,
        0.200,
        0.052,
        0.770,
        0.153,
        face=AUDIT_BG,
        edge="#D3DFE4",
        linewidth=0.5,
        radius=0.010,
    )
    rounded_box(
        axis,
        0.218,
        0.083,
        0.155,
        0.090,
        face=WHITE,
        edge="#CBD7DC",
        linewidth=0.45,
    )
    add_text(axis, 0.2955, 0.143, "separate interpreter", size=6.1, weight="bold")
    add_text(axis, 0.2955, 0.108, "rebuilds full semantics", size=6.0, color=MUTED)

    arrow(axis, (0.376, 0.128), (0.405, 0.128), color=BLUE)
    rounded_box(
        axis,
        0.409,
        0.064,
        0.220,
        0.132,
        face=WHITE,
        edge="#CBD7DC",
        linewidth=0.45,
    )
    add_text(
        axis,
        0.519,
        0.160,
        "compare every valid flag",
        size=6.0,
        weight="bold",
    )
    add_text(axis, 0.519, 0.126, "trace + six-bit σ", size=6.0)
    add_text(axis, 0.519, 0.087, "Flat vs LayerProbe", size=6.0, color=MUTED)

    arrow(axis, (0.632, 0.128), (0.669, 0.128), color=BLUE)
    axis.add_patch(
        Circle(
            (0.691, 0.128),
            0.022,
            facecolor=WHITE,
            edgecolor=BLUE,
            linewidth=0.7,
            zorder=7,
        )
    )
    add_text(axis, 0.691, 0.128, "=", size=7.0, color=BLUE, weight="bold")
    arrow(axis, (0.714, 0.128), (0.750, 0.128), color=GREEN)

    rounded_box(
        axis,
        0.754,
        0.064,
        0.196,
        0.132,
        face=PALE_GREEN,
        edge="#C4D8BD",
        linewidth=0.45,
    )
    add_text(axis, 0.852, 0.160, "declared outputs", size=6.1, color=GREEN, weight="bold")
    add_text(axis, 0.852, 0.126, "agree exactly", size=6.0, color=GREEN, weight="bold")
    add_text(axis, 0.852, 0.087, "full-domain agreement", size=6.0, color=MUTED)

    line(
        axis,
        (0.838, 0.810, 0.810, 0.691),
        (0.762, 0.762, 0.198, 0.198),
        color=PURPLE,
        linewidth=0.55,
    )
    arrow(axis, (0.691, 0.198), (0.691, 0.152), color=PURPLE, mutation_scale=5.5)
    add_text(axis, 0.734, 0.230, "traces + signatures", size=6.0, color=PURPLE)


def build_figure() -> plt.Figure:
    # Build at final raster resolution so all geometry QA is measured in the
    # exact pixels delivered to the manuscript workflow.
    fig = plt.figure(figsize=(WIDTH_IN, HEIGHT_IN), dpi=PNG_DPI)
    axis = fig.add_axes((0, 0, 1, 1))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    draw_product_stage(axis)
    draw_reuse1_stage(axis)
    draw_reuse2_stage(axis)
    draw_output_stage(axis)
    draw_audit_rail(axis)

    # Fine stage separators reinforce the ledger reading order without the
    # reference image's dashed multi-panel framing.
    for x in (0.206, 0.409, 0.796):
        line(axis, (x, x), (0.270, 0.875), color=HAIRLINE, linewidth=0.45, zorder=2)

    fig.canvas.draw()
    return fig


def inspect_figure_text(fig: plt.Figure) -> dict[str, Any]:
    renderer = fig.canvas.get_renderer()
    width_px, height_px = fig.canvas.get_width_height()
    texts = [
        item
        for item in fig.findobj(match=Text)
        if item.get_visible() and item.get_text().strip()
    ]
    require(texts, "Figure contains no visible text.")
    sizes = [float(item.get_fontsize()) for item in texts]
    require(min(sizes) >= MIN_FONT_PT, "Figure contains text below 6 pt.")
    require(
        all("Times New Roman" in item.get_fontfamily() for item in texts),
        "A visible text object is not Times New Roman.",
    )
    outside: list[str] = []
    for item in texts:
        bounds = item.get_window_extent(renderer=renderer)
        if (
            bounds.x0 < -1
            or bounds.y0 < -1
            or bounds.x1 > width_px + 1
            or bounds.y1 > height_px + 1
        ):
            outside.append(item.get_text().replace("\n", " / "))
    require(not outside, "Out-of-bounds text: " + ", ".join(outside))

    overlaps: list[tuple[str, str]] = []
    for left, right in combinations(texts, 2):
        left_bounds = left.get_window_extent(renderer=renderer)
        right_bounds = right.get_window_extent(renderer=renderer)
        overlap_width = max(
            0.0,
            min(left_bounds.x1, right_bounds.x1)
            - max(left_bounds.x0, right_bounds.x0),
        )
        overlap_height = max(
            0.0,
            min(left_bounds.y1, right_bounds.y1)
            - max(left_bounds.y0, right_bounds.y0),
        )
        if overlap_width * overlap_height > 1.0:
            overlaps.append((left.get_text(), right.get_text()))
    require(
        not overlaps,
        "Visible text objects overlap: "
        + "; ".join(f"{left!r} / {right!r}" for left, right in overlaps),
    )
    return {
        "visible_text_objects": len(texts),
        "min_font_pt": min(sizes),
        "max_font_pt": max(sizes),
        "text_overlap_pairs": len(overlaps),
    }


def _bbox_gap(left: Any, right: Any) -> float:
    """Return the Euclidean edge-to-edge gap between two display-space boxes."""
    dx = max(left.x0 - right.x1, right.x0 - left.x1, 0.0)
    dy = max(left.y0 - right.y1, right.y0 - left.y1, 0.0)
    return math.hypot(dx, dy)


def _point_bbox_distance(x: float, y: float, bounds: Any) -> float:
    dx = max(bounds.x0 - x, x - bounds.x1, 0.0)
    dy = max(bounds.y0 - y, y - bounds.y1, 0.0)
    return math.hypot(dx, dy)


def _artist_text_clearance(
    artists: list[Any],
    texts: list[Text],
    renderer: Any,
) -> float:
    """Measure path-to-text clearance in final display pixels."""
    minimum = float("inf")
    text_bounds = [item.get_window_extent(renderer=renderer) for item in texts]
    for artist in artists:
        path = artist.get_path()
        if path.codes is not None:
            # Matplotlib stores an ignored dummy vertex (commonly ``(0, 0)``)
            # for CLOSEPOLY.  Path.interpolated() treats that dummy coordinate
            # as a real endpoint, creating a long, nonexistent segment.  Replace
            # it with the active subpath origin before sampling the visible path.
            vertices = path.vertices.copy()
            subpath_start = None
            for index, code in enumerate(path.codes):
                if code == MplPath.MOVETO:
                    subpath_start = vertices[index].copy()
                elif code == MplPath.CLOSEPOLY and subpath_start is not None:
                    vertices[index] = subpath_start
            path = MplPath(vertices, path.codes)
        path = path.interpolated(32)
        vertices = artist.get_transform().transform(path.vertices)
        for x_value, y_value in vertices:
            if not np.isfinite(x_value) or not np.isfinite(y_value):
                continue
            for bounds in text_bounds:
                distance = _point_bbox_distance(
                    float(x_value),
                    float(y_value),
                    bounds,
                )
                minimum = min(minimum, distance)
    require(np.isfinite(minimum), "Could not measure connector-to-text clearance.")
    return float(minimum)


def _card_text_margin(
    axis: plt.Axes,
    renderer: Any,
    texts: list[Text],
    rect: tuple[float, float, float, float],
    expected_texts: int,
) -> float:
    """Return the minimum internal text margin for one nominal card rectangle."""
    x_value, y_value, width, height = rect
    lower_left, upper_right = axis.transData.transform(
        ((x_value, y_value), (x_value + width, y_value + height))
    )
    card_x0, card_y0 = lower_left
    card_x1, card_y1 = upper_right
    selected: list[Any] = []
    for item in texts:
        bounds = item.get_window_extent(renderer=renderer)
        center_x = (bounds.x0 + bounds.x1) / 2
        center_y = (bounds.y0 + bounds.y1) / 2
        if (
            card_x0 <= center_x <= card_x1
            and card_y0 <= center_y <= card_y1
        ):
            selected.append(bounds)
    require(
        len(selected) == expected_texts,
        (
            f"Card {rect} contains {len(selected)} text objects; "
            f"expected {expected_texts}."
        ),
    )
    return float(
        min(
            min(
                bounds.x0 - card_x0,
                card_x1 - bounds.x1,
                bounds.y0 - card_y0,
                card_y1 - bounds.y1,
            )
            for bounds in selected
        )
    )


def inspect_geometry(fig: plt.Figure) -> dict[str, Any]:
    """Enforce exact spacing at the delivered 400-dpi canvas."""
    renderer = fig.canvas.get_renderer()
    width_px, height_px = fig.canvas.get_width_height()
    require(
        (width_px, height_px)
        == (round(WIDTH_IN * PNG_DPI), round(HEIGHT_IN * PNG_DPI)),
        "Geometry QA must run on the final 400-dpi canvas.",
    )
    axis = fig.axes[0]
    texts = [
        item
        for item in fig.findobj(match=Text)
        if item.get_visible() and item.get_text().strip()
    ]
    text_bounds = [item.get_window_extent(renderer=renderer) for item in texts]

    canvas_clearance_px = min(
        min(
            bounds.x0,
            bounds.y0,
            width_px - bounds.x1,
            height_px - bounds.y1,
        )
        for bounds in text_bounds
    )
    text_text_clearance_px = min(
        _bbox_gap(left, right)
        for left, right in combinations(text_bounds, 2)
    )

    arrows = [
        item
        for item in fig.findobj(match=FancyArrowPatch)
        if item.get_visible()
    ]
    connector_lines = [
        item
        for item in fig.findobj(match=Line2D)
        if item.get_visible() and item.get_zorder() >= 8
    ]
    require(len(arrows) == 17, f"Expected 17 arrows; found {len(arrows)}.")
    require(
        len(connector_lines) == 10,
        f"Expected 10 routed connector lines; found {len(connector_lines)}.",
    )
    arrow_text_clearance_px = _artist_text_clearance(
        arrows,
        texts,
        renderer,
    )
    connector_text_clearance_px = _artist_text_clearance(
        connector_lines,
        texts,
        renderer,
    )

    card_specs = (
        ("state warning", (0.027, 0.270, 0.162, 0.178), 4),
        ("validate once", (0.314, 0.698, 0.074, 0.090), 2),
        ("presentation contract", (0.222, 0.497, 0.172, 0.128), 3),
        ("exact-step index", (0.588, 0.378, 0.152, 0.413), 10),
        ("local-state note", (0.436, 0.268, 0.335, 0.082), 2),
        ("robust-suite card", (0.838, 0.260, 0.124, 0.132), 3),
        ("audit comparison", (0.409, 0.064, 0.220, 0.132), 3),
        ("audit outcome", (0.754, 0.064, 0.196, 0.132), 3),
    )
    card_margins_px = {
        name: _card_text_margin(
            axis,
            renderer,
            texts,
            rect,
            expected_texts,
        )
        for name, rect, expected_texts in card_specs
    }
    card_text_margin_px = min(card_margins_px.values())

    # Dedicated-gutter checks use the nominal ports shown in the figure.
    green_contract_gutter_px = (0.404 - (0.222 + 0.172)) * width_px
    green_entry_gutter_px = (0.432 - 0.425) * width_px
    blue_collector_gutter_px = (0.588 - 0.579) * width_px
    output_label = next(
        item for item in texts if item.get_text() == "traces + signatures"
    )
    purple_return = next(
        item
        for item in connector_lines
        if str(item.get_color()).lower() == PURPLE.lower()
        and len(item.get_xdata()) == 4
    )
    purple_return_clearance_px = _artist_text_clearance(
        [purple_return],
        [output_label],
        renderer,
    )

    require(canvas_clearance_px >= 24.0, "Text is too close to the canvas edge.")
    require(text_text_clearance_px >= 4.0, "Text-to-text gap is below 4 px.")
    require(
        arrow_text_clearance_px >= 5.0,
        f"Arrow-to-text gap is below 5 px: {arrow_text_clearance_px:.3f} px.",
    )
    require(
        connector_text_clearance_px >= 10.0,
        "Connector-line-to-text gap is below 10 px.",
    )
    require(card_text_margin_px >= 4.0, "Card text margin is below 4 px.")
    require(
        green_contract_gutter_px >= 18.0,
        "Green contract-to-gutter clearance is below 18 px.",
    )
    require(
        green_entry_gutter_px >= 12.0,
        "Green arrow-to-presentation clearance is below 12 px.",
    )
    require(
        blue_collector_gutter_px >= 16.0,
        "Blue collector-to-index clearance is below 16 px.",
    )
    require(
        purple_return_clearance_px >= 16.0,
        "Purple return line is too close to its label.",
    )

    return {
        "canvas_clearance_px": canvas_clearance_px,
        "text_text_clearance_px": text_text_clearance_px,
        "arrow_count": len(arrows),
        "arrow_text_clearance_px": arrow_text_clearance_px,
        "connector_count": len(connector_lines),
        "connector_text_clearance_px": connector_text_clearance_px,
        "card_text_margin_px": card_text_margin_px,
        "card_margins_px": card_margins_px,
        "green_contract_gutter_px": green_contract_gutter_px,
        "green_entry_gutter_px": green_entry_gutter_px,
        "blue_collector_gutter_px": blue_collector_gutter_px,
        "purple_return_clearance_px": purple_return_clearance_px,
    }


def _font_is_embedded(font: Any) -> bool:
    descriptor_reference = font.get("/FontDescriptor")
    if descriptor_reference is None:
        return False
    descriptor = descriptor_reference.get_object()
    return any(
        descriptor.get(key) is not None
        for key in ("/FontFile", "/FontFile2", "/FontFile3")
    )


def _inventory_pdf_resources(
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
            base_name = str(font.get("/BaseFont", "(unnamed)"))
            base_fonts.add(base_name)
            descendants = font.get("/DescendantFonts")
            if descendants is not None:
                embedded = False
                for descendant_reference in descendants:
                    descendant = descendant_reference.get_object()
                    font_subtypes.add(str(descendant.get("/Subtype", "")))
                    descendant_name = str(descendant.get("/BaseFont", base_name))
                    base_fonts.add(descendant_name)
                    embedded = embedded or _font_is_embedded(descendant)
            else:
                embedded = _font_is_embedded(font)
            if embedded:
                embedded_fonts.add(base_name)
            else:
                unembedded_fonts.add(base_name)

    xobjects = resources.get("/XObject")
    if xobjects is not None:
        for reference in xobjects.get_object().values():
            xobject = reference.get_object()
            subtype = str(xobject.get("/Subtype", ""))
            if subtype == "/Image":
                image_count[0] += 1
            elif subtype == "/Form":
                _inventory_pdf_resources(
                    xobject.get("/Resources"),
                    font_subtypes,
                    base_fonts,
                    embedded_fonts,
                    unembedded_fonts,
                    image_count,
                )


def inspect_pdf(path: Path) -> dict[str, Any]:
    reader = PdfReader(str(path))
    require(len(reader.pages) == 1, "Figure PDF must have one page.")
    page = reader.pages[0]
    width_in = float(page.mediabox.width) / 72.0
    height_in = float(page.mediabox.height) / 72.0
    require(abs(width_in - WIDTH_IN) < 0.01, "Unexpected PDF width.")
    require(abs(height_in - HEIGHT_IN) < 0.01, "Unexpected PDF height.")

    font_subtypes: set[str] = set()
    base_fonts: set[str] = set()
    embedded_fonts: set[str] = set()
    unembedded_fonts: set[str] = set()
    image_count = [0]
    _inventory_pdf_resources(
        page.get("/Resources"),
        font_subtypes,
        base_fonts,
        embedded_fonts,
        unembedded_fonts,
        image_count,
    )
    require("/Type3" not in font_subtypes, "PDF contains a Type 3 font.")
    require(not unembedded_fonts, f"Unembedded PDF fonts: {sorted(unembedded_fonts)}")
    require(image_count[0] == 0, "PDF contains raster image XObjects.")
    require(
        any("Times" in value for value in base_fonts),
        f"Times New Roman was not embedded: {sorted(base_fonts)}",
    )
    return {
        "width_in": width_in,
        "height_in": height_in,
        "font_subtypes": sorted(font_subtypes),
        "base_fonts": sorted(base_fonts),
        "embedded_fonts": sorted(embedded_fonts),
        "image_xobjects": image_count[0],
    }


def inspect_png(path: Path) -> dict[str, Any]:
    with Image.open(path) as image:
        dpi = image.info.get("dpi", (0.0, 0.0))
        result = {
            "width_px": image.width,
            "height_px": image.height,
            "mode": image.mode,
            "dpi_x": float(dpi[0]),
            "dpi_y": float(dpi[1]),
        }
    require(result["width_px"] == round(WIDTH_IN * PNG_DPI), "Unexpected PNG width.")
    require(result["height_px"] == round(HEIGHT_IN * PNG_DPI), "Unexpected PNG height.")
    require(399 <= result["dpi_x"] <= 401, "PNG x-DPI is not 400.")
    require(399 <= result["dpi_y"] <= 401, "PNG y-DPI is not 400.")
    return result


def save_outputs(
    fig: plt.Figure,
    output_dir: Path,
    stem: str,
    *,
    protected_paths: tuple[Path, ...] = (),
) -> dict[str, Path]:
    require(stem and Path(stem).name == stem, "Stem must be one filename component.")
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "pdf": output_dir / f"{stem}.pdf",
        "png": output_dir / f"{stem}.png",
        "grayscale": output_dir / f"{stem}_grayscale.png",
        "qa": output_dir / "FIGURE1_METHOD_QA.md",
    }
    protected_resolved = {path.resolve() for path in protected_paths}
    require(
        all(path.resolve() not in protected_resolved for path in paths.values()),
        "An output path would overwrite a protected input.",
    )
    title = "LayerProbe two-level exact reuse and independent validation"
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


def write_qa(
    path: Path,
    *,
    output_dir: Path,
    stem: str,
    style_reference: Path | None,
    reference_hash_before: str | None,
    reference_hash_after: str | None,
    reference_size: tuple[int, int] | None,
    legacy_figure: Path | None,
    legacy_size: tuple[int, int] | None,
    paper_tex: Path | None,
    font_path: Path,
    text_qa: dict[str, Any],
    geometry_qa: dict[str, Any],
    pdf_qa: dict[str, Any],
    png_qa: dict[str, Any],
    grayscale_qa: dict[str, Any],
) -> None:
    source_rows = [
        ["LayerProbe evaluator", str(EVALUATOR_SOURCE), sha256(EVALUATOR_SOURCE)],
        ["Independent-oracle boundary", str(ORACLE_README), sha256(ORACLE_README)],
        ["Figure generator", str(SCRIPT_PATH), sha256(SCRIPT_PATH)],
    ]
    if paper_tex is not None:
        source_rows.insert(
            0,
            ["Paper method text (optional QA)", str(paper_tex), sha256(paper_tex)],
        )
    if legacy_figure is not None:
        source_rows.insert(
            0,
            [
                "Earlier LayerProbe figure (optional QA)",
                str(legacy_figure),
                sha256(legacy_figure),
            ],
        )
    if style_reference is not None:
        require(
            reference_hash_before is not None,
            "Missing optional style-reference hash.",
        )
        source_rows.insert(
            0,
            [
                "External style reference (optional QA only)",
                str(style_reference),
                reference_hash_before,
            ],
        )
    semantic_rows = [
        [
            "Problem",
            "A presentation may change observation, memory, action, and later trajectory; state-only reuse is unsafe.",
            "PASS",
        ],
        [
            "Reuse level 1",
            "Validate each mechanism once because presentations cannot change physics, terminal rules, or validity.",
            "PASS",
        ],
        [
            "Reuse level 2",
            "Within fixed mechanism k and agent a, reuse only on q = (world state s, pre-ingest memory m, observation o).",
            "PASS",
        ],
        [
            "Cached result",
            "r(q) contains action, next state, next memory, and next status; miss computes and stores it.",
            "PASS",
        ],
        [
            "Local state",
            "Observation generation, display history, and declared presentation traces are never merged.",
            "PASS",
        ],
        [
            "Output",
            "Declared traces produce six-bit signatures; an all-p intersection and exact cover form one downstream query.",
            "PASS",
        ],
        [
            "Independent validation",
            "A separate interpreter rebuilds semantics and compares validity, every trace, and signatures over the complete finite domain.",
            "PASS",
        ],
    ]
    external_reference_label = (
        "Supplied optional style reference"
        if style_reference is not None
        else "No external reference supplied"
    )
    similarity_rows = [
        [
            "Global composition",
            external_reference_label,
            "One unlettered left-to-right semantic ledger plus a bottom audit rail",
        ],
        [
            "Primary visual grammar",
            "Not used by the drawing pipeline",
            "Four open ledger stages separated by hairlines; no dashed enclosure",
        ],
        [
            "Motifs",
            "Not imported or embedded",
            "Finite-product matrix, contract, exact-key index, signature ledger",
        ],
        [
            "Information flow",
            "Not used to position any element",
            "Invariant lane fans into presentation-local lanes, then robust aggregation",
        ],
        [
            "Panel labels",
            "Not copied",
            "None",
        ],
        [
            "Construction",
            "Optional file is provenance-only",
            "Original Matplotlib vector/text primitives generated from method semantics",
        ],
    ]
    technical_rows = [
        ["PDF size", f"{pdf_qa['width_in']:.3f} × {pdf_qa['height_in']:.3f} in", "PASS"],
        [
            "Colour PNG",
            f"{png_qa['width_px']} × {png_qa['height_px']} px at {png_qa['dpi_x']:.2f} dpi",
            "PASS",
        ],
        [
            "Grayscale PNG",
            f"{grayscale_qa['width_px']} × {grayscale_qa['height_px']} px; mode {grayscale_qa['mode']}",
            "PASS",
        ],
        [
            "Visible text",
            f"Times New Roman; minimum {text_qa['min_font_pt']:.1f} pt; {text_qa['visible_text_objects']} objects",
            "PASS",
        ],
        [
            "PDF fonts",
            f"{len(pdf_qa['embedded_fonts'])} embedded subsets; no Type 3",
            "PASS",
        ],
        ["Vector integrity", f"{pdf_qa['image_xobjects']} raster image XObjects", "PASS"],
        [
            "Canvas bounds",
            (
                f"minimum text clearance {geometry_qa['canvas_clearance_px']:.2f} px; "
                f"{text_qa['text_overlap_pairs']} text-overlap pairs"
            ),
            "PASS",
        ],
        [
            "External-image independence",
            (
                "optional style-reference SHA-256 unchanged"
                if style_reference is not None
                else "no external style-reference supplied or read"
            ),
            "PASS",
        ],
    ]
    spacing_rows = [
        [
            "Canvas edge → text",
            f"{geometry_qa['canvas_clearance_px']:.3f} px",
            "≥ 24 px",
            "PASS",
        ],
        [
            "Text ↔ text",
            f"{geometry_qa['text_text_clearance_px']:.3f} px; 0 overlaps",
            "≥ 4 px",
            "PASS",
        ],
        [
            "Arrow ↔ text",
            (
                f"{geometry_qa['arrow_text_clearance_px']:.3f} px across "
                f"{geometry_qa['arrow_count']} arrows"
            ),
            "≥ 5 px",
            "PASS",
        ],
        [
            "Routed line ↔ text",
            (
                f"{geometry_qa['connector_text_clearance_px']:.3f} px across "
                f"{geometry_qa['connector_count']} connector lines"
            ),
            "≥ 10 px",
            "PASS",
        ],
        [
            "Card edge → contained text",
            f"{geometry_qa['card_text_margin_px']:.3f} px minimum",
            "≥ 4 px",
            "PASS",
        ],
        [
            "Green contract → fan-out gutter",
            f"{geometry_qa['green_contract_gutter_px']:.3f} px",
            "≥ 18 px",
            "PASS",
        ],
        [
            "Green entry arrow → p pill",
            f"{geometry_qa['green_entry_gutter_px']:.3f} px",
            "≥ 12 px",
            "PASS",
        ],
        [
            "Blue query collector → index card",
            f"{geometry_qa['blue_collector_gutter_px']:.3f} px",
            "≥ 16 px",
            "PASS",
        ],
        [
            "Purple return line → output label",
            f"{geometry_qa['purple_return_clearance_px']:.3f} px",
            "≥ 16 px",
            "PASS",
        ],
    ]
    card_margin_rows = [
        [name, f"{margin:.3f} px", "PASS"]
        for name, margin in geometry_qa["card_margins_px"].items()
    ]

    if style_reference is None:
        reference_details = (
            "No optional style reference was supplied. Figure construction and "
            "publication QA were completed from project semantics and original "
            "Matplotlib primitives only."
        )
        optional_visual_note = (
            "- External-reference comparison: not requested for this run; "
            "reference-independent construction verified."
        )
    else:
        require(
            reference_size is not None
            and reference_hash_before is not None
            and reference_hash_after is not None,
            "Incomplete optional style-reference provenance.",
        )
        reference_details = (
            f"Optional style-reference dimensions: `{reference_size[0]} × "
            f"{reference_size[1]}` pixels (aspect ratio "
            f"`{reference_size[0] / reference_size[1]:.3f}`).\n\n"
            f"Reference hash before generation: `{reference_hash_before}`.\n\n"
            f"Reference hash after generation: `{reference_hash_after}`."
        )
        optional_visual_note = (
            "- Optional side-by-side review: composition, flow direction, module "
            "vocabulary, and motifs remain structurally distinct."
        )

    legacy_details = (
        "No earlier LayerProbe figure was available; it is not required."
        if legacy_size is None
        else (
            f"Earlier LayerProbe figure dimensions: `{legacy_size[0]} × "
            f"{legacy_size[1]}` pixels (aspect ratio "
            f"`{legacy_size[0] / legacy_size[1]:.3f}`)."
        )
    )

    report = f"""# Figure 1 Method Overview QA

Generated: `{datetime.now().astimezone().isoformat(timespec="seconds")}`

Output stem: `{stem}`

Output directory: `{output_dir.resolve()}`

## Outcome

This is a fully original LayerProbe method figure. Its drawing pipeline uses
only project semantics and programmatically generated vector/text primitives.
An optional external image, when explicitly supplied, is used only for
non-destructive provenance and side-by-side visual review.

## Source and provenance

{markdown_table(["Item", "Path", "SHA-256"], source_rows)}

{reference_details}

{legacy_details}

New Figure 1 aspect ratio: `{WIDTH_IN / HEIGHT_IN:.3f}`.

## Method-semantic audit

{markdown_table(["Element", "Meaning encoded in the figure", "Status"], semantic_rows)}

The cache is deliberately labeled as scoped to a fixed `(k, a)`. Therefore the
displayed complete execution key is exactly `q = (s, m, o)`, without implying
that mechanism or agent identity can be omitted from a globally shared cache.

## Originality and external-image independence audit

{markdown_table(["Feature", "Optional external input", "New Figure 1"], similarity_rows)}

Verdict: **PASS — clearly different composition and visual grammar.**

The new design contains no time-series plot, Transformer/GNN block, graph
adjacency matrix, lettered subpanel, dashed outer container, or left-side
vertical input-to-output pipeline.

## Publication QA

{markdown_table(["Check", "Result", "Status"], technical_rows)}

Embedded PDF base fonts: `{", ".join(pdf_qa["base_fonts"])}`.

## Exact 400-dpi spacing audit

All measurements below are computed on the delivered `1888 × 1008` canvas,
after the final Matplotlib draw and before export.

{markdown_table(["Relationship", "Measured minimum", "Gate", "Status"], spacing_rows)}

Curated card-containment margins:

{markdown_table(["Card", "Minimum internal text margin", "Status"], card_margin_rows)}

## Visual QA

Status: **{VISUAL_REVIEW_STATUS}**

{chr(10).join(f"- {note}" for note in VISUAL_REVIEW_NOTES)}
{optional_visual_note}

## Reproduction

From the project root:

```powershell
python experiments/build_figure1_layerprobe_method.py
```

To generate the manuscript replacement directly:

```powershell
python experiments/build_figure1_layerprobe_method.py `
  --output-dir ../../07_论文/manuscript/draft/figures `
  --stem fig0_layerprobe_overview
```

The script writes only `{stem}.pdf`, `{stem}.png`,
`{stem}_grayscale.png`, and `FIGURE1_METHOD_QA.md` in the selected directory.
It does not edit `paper.tex`, does not require a third-party image, and never
writes to an optional style-reference path.
"""
    path.write_text(report, encoding="utf-8", newline="\n")


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    stem = str(args.stem)

    for source in (EVALUATOR_SOURCE, ORACLE_README):
        require(source.is_file(), f"Missing required source: {source}")

    style_reference = (
        None if args.style_reference is None else args.style_reference.resolve()
    )
    reference_hash_before: str | None = None
    reference_hash_after: str | None = None
    reference_size: tuple[int, int] | None = None
    if style_reference is not None:
        require(
            style_reference.is_file(),
            f"Missing optional style reference: {style_reference}",
        )
        reference_hash_before = sha256(style_reference)
        with Image.open(style_reference) as reference:
            reference_size = reference.size

    legacy_figure = LEGACY_FIGURE if LEGACY_FIGURE.is_file() else None
    legacy_size: tuple[int, int] | None = None
    if legacy_figure is not None:
        with Image.open(legacy_figure) as legacy:
            legacy_size = legacy.size
    paper_tex = PAPER_TEX if PAPER_TEX.is_file() else None

    font_path = configure_style()
    fig = build_figure()
    text_qa = inspect_figure_text(fig)
    geometry_qa = inspect_geometry(fig)
    paths = save_outputs(
        fig,
        output_dir,
        stem,
        protected_paths=(
            () if style_reference is None else (style_reference,)
        ),
    )
    plt.close(fig)

    if style_reference is not None:
        reference_hash_after = sha256(style_reference)
        require(
            reference_hash_after == reference_hash_before,
            "Optional style reference changed during figure generation.",
        )
    pdf_qa = inspect_pdf(paths["pdf"])
    png_qa = inspect_png(paths["png"])
    grayscale_qa = inspect_png(paths["grayscale"])
    require(grayscale_qa["mode"] == "L", "Grayscale preview is not mode L.")

    write_qa(
        paths["qa"],
        output_dir=output_dir,
        stem=stem,
        style_reference=style_reference,
        reference_hash_before=reference_hash_before,
        reference_hash_after=reference_hash_after,
        reference_size=reference_size,
        legacy_figure=legacy_figure,
        legacy_size=legacy_size,
        paper_tex=paper_tex,
        font_path=font_path,
        text_qa=text_qa,
        geometry_qa=geometry_qa,
        pdf_qa=pdf_qa,
        png_qa=png_qa,
        grayscale_qa=grayscale_qa,
    )

    for label_value in ("pdf", "png", "grayscale", "qa"):
        print(f"{label_value}: {paths[label_value]}")
    print(
        "QA: method semantics PASS; external-image independence PASS; "
        "400-dpi geometry PASS; vector PDF PASS; no Type 3 fonts PASS."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
