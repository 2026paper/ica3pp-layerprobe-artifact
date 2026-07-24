#!/usr/bin/env python
"""Build a publication-grade all-view evidence figure from frozen results.

The figure has one argument:

    Replacing an any-view union by an all-view intersection changes both
    per-mechanism coverage and the minimum covering suite.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from functools import reduce
from operator import and_, or_
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
DEFAULT_OUTPUT_DIR = RESULTS_ROOT / "figure4_allview_evidence_ccfa_20260725_v1"

COMMUNICATION_ROOT = (
    RESULTS_ROOT
    / "communication_full_24624_distancefix_provenance_v2_20260723_xeon"
)
SENSITIVITY_ROOT = (
    RESULTS_ROOT
    / "agent_sensitivity_full_24624_distancefix_provenance_v2_20260723_xeon"
)
CANDIDATE_PATH = COMMUNICATION_ROOT / "candidate_signatures.csv.gz"
ROBUST_FAMILIES_PATH = COMMUNICATION_ROOT / "robust_families.csv"
PAIR_PATH = SENSITIVITY_ROOT / "pair_delay_sensitivity.csv"
LEAVE_ONE_OUT_PATH = SENSITIVITY_ROOT / "leave_one_agent_out.csv"

EXPECTED_HASHES = {
    CANDIDATE_PATH: "3505e0be250e919e524281caf75a826b8580bdb46ba0206012aaafed4e7bbc72",
    ROBUST_FAMILIES_PATH: "b5e0b7ae64abc0307010598bfc3dec23a5053a0c2daafbf89ce274cd50c96e31",
    PAIR_PATH: "a6b164ad8850a5bc2f088308224c8491422105279693d6e850d33a51b41b7d1c",
    LEAVE_ONE_OUT_PATH: "7ae964293fc2a82ac2e3f86c9bb68332b8f5018c1a81a253c4af0d8c08e11e43",
}

WIDTH_IN = 4.72
HEIGHT_IN = 3.82
PNG_DPI = 400
MIN_FONT_PT = 6.5
TARGET_MASK = 63
PAIR_CODES = ("R-I", "R-S", "R-F", "I-S", "I-F", "S-F")

# Okabe-Ito-derived palette with light fills for print.
INK = "#202A32"
BLUE = "#0072B2"
PALE_BLUE = "#CFE6F3"
ORANGE = "#D98200"
PALE_ORANGE = "#F7D9A6"
GREEN = "#007A5A"
PALE_GREEN = "#D7EEE6"
VERMILION = "#B84A2B"
GRAY = "#687178"
LIGHT_GRAY = "#D8DDE1"
PALE_GRAY = "#F3F5F6"
WHITE = "#FFFFFF"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--visual-review-status",
        choices=("PENDING", "PASS"),
        default="PENDING",
        help="Set PASS only after inspecting both color and grayscale PNGs.",
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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def validate_provenance() -> dict[str, str]:
    actual: dict[str, str] = {}
    for path, expected in EXPECTED_HASHES.items():
        require(path.is_file(), f"Missing frozen input: {path}")
        observed = sha256(path)
        require(observed == expected, f"Unexpected hash for {path}: {observed}")
        actual[str(path.relative_to(PROJECT_ROOT))] = observed
    return actual


def configure_style() -> str:
    font_path = Path(
        font_manager.findfont("Times New Roman", fallback_to_default=False)
    ).resolve()
    require(font_path.is_file(), "Times New Roman is not installed.")
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
            "patch.linewidth": 0.6,
            "hatch.linewidth": 0.45,
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
    return str(font_path)


def load_frozen_data() -> dict[str, Any]:
    pair_rows = read_csv(PAIR_PATH)
    leave_rows = read_csv(LEAVE_ONE_OUT_PATH)
    family_rows = read_csv(ROBUST_FAMILIES_PATH)
    require(len(pair_rows) == 6, "Expected six declared-agent pairs.")
    require(len(leave_rows) == 4, "Expected four leave-one-agent-out rows.")

    all_18 = next(row for row in family_rows if row["family"] == "all_18")
    require(int(all_18["kernel_count"]) == 10_544, "Unexpected mechanism count.")
    require(
        int(all_18["union_minimum_suite_size"]) == 1,
        "Expected an any-view minimum cover of one.",
    )
    require(
        int(all_18["robust_minimum_suite_size"]) == 2,
        "Expected an all-view minimum cover of two.",
    )
    require(
        int(all_18["robust_full_kernels"]) == 0,
        "Expected no single all-view full-mask mechanism.",
    )
    require(
        all(int(row["robust_minimum_suite_size"]) == 1 for row in leave_rows),
        "Every leave-one-agent-out minimum cover must be one.",
    )

    masks_by_kernel: dict[str, list[int]] = defaultdict(list)
    row_count = 0
    with gzip.open(CANDIDATE_PATH, "rt", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            masks_by_kernel[row["kernel"]].append(int(row["signature_mask"]))
            row_count += 1
    require(row_count == 189_792, "Unexpected candidate-signature row count.")
    require(len(masks_by_kernel) == 10_544, "Unexpected number of mechanisms.")
    require(
        all(len(values) == 18 for values in masks_by_kernel.values()),
        "Every mechanism must have exactly 18 presentations.",
    )

    union_masks = {
        kernel: reduce(or_, values) for kernel, values in masks_by_kernel.items()
    }
    robust_masks = {
        kernel: reduce(and_, values) for kernel, values in masks_by_kernel.items()
    }
    union_counts = Counter(mask.bit_count() for mask in union_masks.values())
    robust_counts = Counter(mask.bit_count() for mask in robust_masks.values())
    union_distribution = [union_counts.get(bits, 0) for bits in range(7)]
    robust_distribution = [robust_counts.get(bits, 0) for bits in range(7)]

    require(union_distribution == [0, 0, 0, 0, 39, 5409, 5096], "Union distribution changed.")
    require(
        robust_distribution == [1050, 156, 393, 5738, 2082, 1125, 0],
        "All-view distribution changed.",
    )
    selected_masks = {
        "brake_10377": robust_masks["brake_10377"],
        "brake_10387": robust_masks["brake_10387"],
    }
    require(selected_masks == {"brake_10377": 59, "brake_10387": 31}, "Cover masks changed.")
    require(reduce(or_, selected_masks.values()) == TARGET_MASK, "Cover is incomplete.")

    delay_values = [
        100.0 * float(row["delayed_minus_immediate_rate"]) for row in pair_rows
    ]
    require(sum(value < 0 for value in delay_values) == 5, "Expected five decreases.")
    require(sum(value > 0 for value in delay_values) == 1, "Expected one increase.")

    return {
        "pair_rows": pair_rows,
        "leave_rows": leave_rows,
        "all_18": all_18,
        "union_distribution": union_distribution,
        "robust_distribution": robust_distribution,
        "selected_masks": selected_masks,
    }


def style_axis(axis: plt.Axes, *, grid_axis: str = "y") -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis=grid_axis, color=LIGHT_GRAY, linewidth=0.5)
    axis.set_axisbelow(True)


def panel_title(axis: plt.Axes, label: str, title: str, *, right_note: str = "") -> None:
    axis.text(
        0.0,
        1.045,
        label,
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.2,
        fontweight="bold",
        color=INK,
        clip_on=False,
    )
    axis.text(
        0.055,
        1.045,
        title,
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.2,
        fontweight="bold",
        color=INK,
        clip_on=False,
    )
    if right_note:
        axis.text(
            1.0,
            1.045,
            right_note,
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            fontsize=6.5,
            color=GRAY,
            clip_on=False,
        )


def draw_delay_census(axis: plt.Axes, pair_rows: list[dict[str, str]]) -> None:
    values = np.asarray(
        [100.0 * float(row["delayed_minus_immediate_rate"]) for row in pair_rows]
    )
    y = np.arange(len(values))[::-1]
    axis.axvline(0, color=INK, linewidth=0.75, zorder=1)
    axis.hlines(y, 0, values, color=LIGHT_GRAY, linewidth=1.4, zorder=1)
    for yi, value in zip(y, values):
        is_negative = value < 0
        axis.scatter(
            value,
            yi,
            s=30,
            marker="o" if is_negative else "D",
            facecolor=BLUE if is_negative else ORANGE,
            edgecolor=INK,
            linewidth=0.55,
            zorder=3,
        )
        axis.text(
            value + 0.24,
            yi,
            f"{value:+.2f}",
            ha="left",
            va="center",
            fontsize=6.5,
            color=BLUE if is_negative else ORANGE,
            fontweight="bold",
        )
    axis.set_yticks(y, PAIR_CODES)
    axis.set_xlim(-8.9, 1.75)
    axis.set_xticks([-8, -6, -4, -2, 0])
    axis.set_ylim(-0.55, 5.55)
    axis.set_xlabel("Change in separation rate (percentage points)", labelpad=1.5)
    axis.tick_params(axis="y", length=0, pad=2)
    style_axis(axis, grid_axis="x")
    axis.spines["left"].set_visible(False)
    panel_title(
        axis,
        "(a)",
        "Delay has pair-specific effects",
        right_note="complete census: 5 decrease, 1 increase",
    )


def draw_bit_count_distribution(
    axis: plt.Axes,
    union_distribution: list[int],
    robust_distribution: list[int],
) -> None:
    total = float(sum(union_distribution))
    x = np.arange(7)
    union_pct = 100.0 * np.asarray(union_distribution) / total
    robust_pct = 100.0 * np.asarray(robust_distribution) / total
    width = 0.34
    axis.bar(
        x - width / 2,
        union_pct,
        width,
        label="Any-view union",
        facecolor=PALE_BLUE,
        edgecolor=BLUE,
        hatch="////",
        linewidth=0.7,
        zorder=3,
    )
    axis.bar(
        x + width / 2,
        robust_pct,
        width,
        label="All-view intersection",
        facecolor=PALE_ORANGE,
        edgecolor=ORANGE,
        hatch="xx",
        linewidth=0.7,
        zorder=3,
    )
    axis.set_xlim(-0.55, 6.55)
    axis.set_ylim(0, 60)
    axis.set_xticks(x)
    axis.set_yticks([0, 20, 40, 60])
    axis.set_xlabel("Separated agent pairs per mechanism (of 6)", labelpad=1.5)
    axis.set_ylabel("Mechanisms (%)", labelpad=2)
    style_axis(axis, grid_axis="y")
    axis.legend(
        loc="upper left",
        ncols=2,
        frameon=False,
        handlelength=1.6,
        columnspacing=1.0,
        borderaxespad=0.25,
    )
    axis.annotate(
        "5,096",
        xy=(6 - width / 2, union_pct[6]),
        xytext=(5.50, 53.6),
        ha="center",
        va="bottom",
        fontsize=6.5,
        color=BLUE,
        fontweight="bold",
        arrowprops=dict(arrowstyle="-", color=BLUE, linewidth=0.65),
    )
    axis.scatter(
        [6 + width / 2],
        [0.7],
        marker="x",
        s=22,
        linewidth=0.9,
        color=ORANGE,
        zorder=4,
    )
    axis.text(
        6 + width / 2,
        3.8,
        "0",
        ha="center",
        va="bottom",
        fontsize=6.5,
        color=ORANGE,
        fontweight="bold",
    )
    panel_title(
        axis,
        "(b)",
        "The stronger quantifier removes every 6/6 mechanism",
        right_note="10,544 mechanisms",
    )


def draw_mask_matrix(axis: plt.Axes, selected_masks: dict[str, int]) -> None:
    row_names = ("brake_10377", "brake_10387", "suite OR")
    masks = (
        selected_masks["brake_10377"],
        selected_masks["brake_10387"],
        reduce(or_, selected_masks.values()),
    )
    axis.set_xlim(-1.82, 6.05)
    axis.set_ylim(-0.55, 3.15)
    axis.axis("off")

    cell_w = 0.78
    cell_h = 0.70
    x_positions = np.arange(6) + 0.18
    y_positions = (2.05, 1.15, 0.15)
    for x, code in zip(x_positions, PAIR_CODES):
        axis.text(
            x + cell_w / 2,
            2.88,
            code,
            ha="center",
            va="center",
            fontsize=6.5,
            fontweight="bold",
            color=INK,
        )
    axis.text(
        -0.05,
        2.88,
        "all-view bit",
        ha="right",
        va="center",
        fontsize=6.5,
        color=GRAY,
    )

    for row_index, (name, mask, y) in enumerate(zip(row_names, masks, y_positions)):
        is_suite = row_index == 2
        axis.text(
            -0.05,
            y + cell_h / 2,
            name,
            ha="right",
            va="center",
            fontsize=6.5,
            fontweight="bold" if is_suite else "normal",
            color=GREEN if is_suite else INK,
        )
        for bit, x in enumerate(x_positions):
            covered = bool(mask & (1 << bit))
            if is_suite:
                face = PALE_GREEN
                edge = GREEN
                hatch = "///"
            elif covered:
                face = PALE_BLUE
                edge = BLUE
                hatch = ""
            else:
                face = WHITE
                edge = VERMILION
                hatch = ""
            axis.add_patch(
                Rectangle(
                    (x, y),
                    cell_w,
                    cell_h,
                    facecolor=face,
                    edgecolor=edge,
                    linewidth=0.75,
                    hatch=hatch,
                )
            )
            axis.text(
                x + cell_w / 2,
                y + cell_h / 2,
                "1" if covered else "0",
                ha="center",
                va="center",
                fontsize=7.0,
                fontweight="bold",
                color=GREEN if is_suite else (BLUE if covered else VERMILION),
            )
    axis.plot(
        [-1.65, 5.92],
        [0.99, 0.99],
        color=GREEN,
        linewidth=0.65,
        linestyle=(0, (3, 2)),
    )
    panel_title(
        axis,
        "(c)",
        "Two complementary all-view masks cover all six pairs",
    )


def draw_cover_summary(axis: plt.Axes, leave_rows: list[dict[str, str]]) -> None:
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    axis.add_patch(
        Rectangle(
            (0.02, 0.04),
            0.96,
            0.92,
            facecolor=PALE_GRAY,
            edgecolor=LIGHT_GRAY,
            linewidth=0.65,
        )
    )
    axis.text(
        0.08,
        0.88,
        "MINIMUM COVER",
        ha="left",
        va="center",
        fontsize=6.5,
        fontweight="bold",
        color=GRAY,
    )
    rows = (
        ("Any view  (union)", "1", BLUE),
        ("All 18 views  (intersection)", "2", ORANGE),
    )
    for (label, value, color), y in zip(rows, (0.70, 0.47)):
        axis.text(
            0.08,
            y,
            label,
            ha="left",
            va="center",
            fontsize=6.5,
            color=INK,
        )
        axis.text(
            0.90,
            y,
            value,
            ha="right",
            va="center",
            fontsize=9.0,
            fontweight="bold",
            color=color,
        )
    axis.add_patch(
        FancyArrowPatch(
            (0.82, 0.66),
            (0.82, 0.52),
            arrowstyle="-|>",
            mutation_scale=7,
            linewidth=0.7,
            color=GRAY,
        )
    )
    axis.plot([0.08, 0.92], [0.34, 0.34], color=LIGHT_GRAY, linewidth=0.65)
    axis.text(
        0.08,
        0.23,
        "Leave one agent out",
        ha="left",
        va="center",
        fontsize=6.5,
        color=INK,
    )
    axis.text(
        0.90,
        0.23,
        "1  (4/4)",
        ha="right",
        va="center",
        fontsize=7.0,
        fontweight="bold",
        color=GREEN,
    )
    require(len(leave_rows) == 4, "Expected four leave-one-agent-out cases.")
    axis.text(
        0.08,
        0.10,
        "All four subsets return to\none mechanism.",
        ha="left",
        va="center",
        fontsize=6.5,
        color=GRAY,
        linespacing=0.95,
    )


def build_figure(data: dict[str, Any]) -> plt.Figure:
    fig = plt.figure(figsize=(WIDTH_IN, HEIGHT_IN))
    grid = fig.add_gridspec(
        3,
        1,
        height_ratios=(1.06, 1.10, 1.08),
        hspace=0.63,
        left=0.105,
        right=0.975,
        bottom=0.050,
        top=0.965,
    )
    delay_axis = fig.add_subplot(grid[0, 0])
    distribution_axis = fig.add_subplot(grid[1, 0])
    bottom = grid[2, 0].subgridspec(
        1,
        2,
        width_ratios=(1.64, 1.00),
        wspace=0.12,
    )
    matrix_axis = fig.add_subplot(bottom[0, 0])
    summary_axis = fig.add_subplot(bottom[0, 1])
    draw_delay_census(delay_axis, data["pair_rows"])
    draw_bit_count_distribution(
        distribution_axis,
        data["union_distribution"],
        data["robust_distribution"],
    )
    draw_mask_matrix(matrix_axis, data["selected_masks"])
    draw_cover_summary(summary_axis, data["leave_rows"])
    return fig


def audit_layout(fig: plt.Figure) -> list[str]:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    canvas = fig.bbox
    text_items = [
        artist
        for artist in fig.findobj(match=lambda item: isinstance(item, Text))
        if artist.get_visible() and artist.get_text()
    ]
    font_sizes = [float(item.get_fontsize()) for item in text_items]
    require(font_sizes, "No visible text found.")
    require(
        min(font_sizes) >= MIN_FONT_PT,
        f"Text below {MIN_FONT_PT:.1f} pt: {min(font_sizes):.2f} pt.",
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
    require(not outside, f"Text outside canvas: {outside}")
    width, height = fig.get_size_inches()
    require(abs(width - WIDTH_IN) < 1e-6, f"Unexpected width: {width}")
    require(abs(height - HEIGHT_IN) < 1e-6, f"Unexpected height: {height}")
    return [
        f"final size: {width:.2f} x {height:.2f} in",
        f"minimum text size: {min(font_sizes):.1f} pt",
        "text outside canvas: none",
    ]


def inspect_pdf(path: Path) -> list[str]:
    reader = PdfReader(str(path))
    require(len(reader.pages) == 1, "Expected a single-page figure PDF.")
    page = reader.pages[0]
    resources = page["/Resources"].get_object()
    font_dict = resources.get("/Font", {})
    if hasattr(font_dict, "get_object"):
        font_dict = font_dict.get_object()
    type3: list[str] = []
    unembedded: list[str] = []
    for ref in font_dict.values():
        font = ref.get_object()
        base = str(font.get("/BaseFont"))
        if str(font.get("/Subtype")) == "/Type3":
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
    raster_images = [
        name
        for name, ref in xobjects.items()
        if str(ref.get_object().get("/Subtype")) == "/Image"
    ]
    require(not type3, f"Type 3 fonts found: {type3}")
    require(not unembedded, f"Unembedded fonts found: {unembedded}")
    require(not raster_images, f"Raster image XObjects found: {raster_images}")
    return [
        "embedded fonts: PASS",
        "Type 3 fonts: none",
        "raster image XObjects: none",
    ]


def export_outputs(
    fig: plt.Figure,
    output_dir: Path,
) -> tuple[dict[str, str], list[str]]:
    layout_notes = audit_layout(fig)
    basename = "fig4_allview_evidence"
    pdf_path = output_dir / f"{basename}.pdf"
    png_path = output_dir / f"{basename}.png"
    grayscale_path = output_dir / f"{basename}_grayscale.png"
    fig.savefig(
        pdf_path,
        metadata={
            "Title": "All-view evidence",
            "Creator": SCRIPT_PATH.name,
        },
    )
    fig.savefig(
        png_path,
        dpi=PNG_DPI,
        metadata={"Title": "All-view evidence"},
    )
    plt.close(fig)

    with Image.open(png_path) as image:
        expected_size = (round(WIDTH_IN * PNG_DPI), round(HEIGHT_IN * PNG_DPI))
        require(image.size == expected_size, f"Unexpected PNG size: {image.size}")
        gray = ImageOps.grayscale(image).convert("RGB")
        gray.save(grayscale_path, dpi=(PNG_DPI, PNG_DPI))
    with Image.open(grayscale_path) as gray:
        array = np.asarray(gray)
        require(
            np.array_equal(array[:, :, 0], array[:, :, 1])
            and np.array_equal(array[:, :, 1], array[:, :, 2]),
            "Grayscale preview is not true grayscale.",
        )

    pdf_notes = inspect_pdf(pdf_path)
    paths = {
        "pdf": str(pdf_path),
        "png": str(png_path),
        "grayscale": str(grayscale_path),
    }
    require(
        all(Path(path).is_file() and Path(path).stat().st_size > 0 for path in paths.values()),
        "One or more output files are missing.",
    )
    return paths, layout_notes + pdf_notes + ["true grayscale preview: PASS"]


def write_caption(output_dir: Path) -> Path:
    caption = (
        "Figure 4. Changing the quantifier changes the evidence and the selected suite. "
        "(a) Across the complete mechanism-presentation census, delay decreases five "
        "agent-pair separation rates and increases one; R, I, S, and F denote reference, "
        "instant-stop, speed-only, and friction-blind agents. (b) The any-view union "
        "contains 5,096 mechanisms with all six separation bits, whereas the all-view "
        "intersection contains none. (c) The all-view masks of brake_10377 and "
        "brake_10387 are complementary and jointly cover all six pairs, raising the "
        "minimum cover from one under any-view union to two under all-view intersection. "
        "Removing any one agent reduces the all-view cover to one in all four cases."
    )
    path = output_dir / "fig4_allview_evidence_caption.txt"
    path.write_text(caption + "\n", encoding="utf-8")
    return path


def write_qa(
    output_dir: Path,
    provenance: dict[str, str],
    data: dict[str, Any],
    outputs: dict[str, str],
    notes: list[str],
    font_path: str,
    visual_review_status: str,
) -> Path:
    payload = {
        "status": "PASS" if visual_review_status == "PASS" else "MACHINE_PASS",
        "machine_qa": "PASS",
        "manual_visual_review": visual_review_status,
        "argument": (
            "Changing from any-view union to all-view intersection changes "
            "per-mechanism coverage and the minimum covering suite."
        ),
        "source_provenance": provenance,
        "data_assertions": {
            "mechanisms": 10_544,
            "presentations_per_mechanism": 18,
            "pair_delay_directions": "5 negative, 1 positive",
            "union_bit_count_distribution_0_to_6": data["union_distribution"],
            "all_view_bit_count_distribution_0_to_6": data["robust_distribution"],
            "selected_all_view_masks": data["selected_masks"],
            "selected_mask_or": reduce(or_, data["selected_masks"].values()),
            "minimum_cover_any_view": 1,
            "minimum_cover_all_18_views": 2,
            "leave_one_agent_out_cover_one": "4/4",
        },
        "render": {
            "width_inches": WIDTH_IN,
            "height_inches": HEIGHT_IN,
            "png_dpi": PNG_DPI,
            "minimum_font_pt": MIN_FONT_PT,
            "font_path": font_path,
        },
        "outputs": outputs,
        "checks": notes,
        "visual_review_checklist": {
            "text_clipping": visual_review_status,
            "text_mark_overlap": visual_review_status,
            "panel_alignment": visual_review_status,
            "subplot_spacing": visual_review_status,
            "color_and_grayscale_redundancy": visual_review_status,
            "data_completeness": visual_review_status,
            "cross_panel_semantic_consistency": visual_review_status,
            "overall_information_density": visual_review_status,
        },
    }
    path = output_dir / "fig4_allview_evidence_qa.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    font_path = configure_style()
    provenance = validate_provenance()
    data = load_frozen_data()
    outputs, notes = export_outputs(build_figure(data), output_dir)
    caption_path = write_caption(output_dir)
    outputs["caption"] = str(caption_path)
    qa_path = write_qa(
        output_dir,
        provenance,
        data,
        outputs,
        notes,
        font_path,
        args.visual_review_status,
    )
    print(f"PDF: {outputs['pdf']}")
    print(f"PNG: {outputs['png']}")
    print(f"grayscale: {outputs['grayscale']}")
    print(f"caption: {outputs['caption']}")
    print(f"QA: {qa_path}")
    print(f"visual review: {args.visual_review_status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
