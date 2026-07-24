#!/usr/bin/env python
"""Build a compact, evidence-first semantic-safety figure for the paper.

The figure deliberately replaces the previous table-like design.  It makes one
claim: the complete semantic key preserves every audited output, whereas
dropping any key field creates collisions that propagate to full traces and
their six-bit projection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.text import Text
from PIL import Image, ImageOps
from pypdf import PdfReader


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
RESULTS_ROOT = PROJECT_ROOT / "results"

ABLATION_PATH = (
    RESULTS_ROOT
    / "cache_key_ablation_full_24624_distancefix_20260723_xeon"
    / "summary.json"
)
ORACLE_PATH = (
    RESULTS_ROOT
    / "independent_trace_oracle_full_24624_distancefix_20260723_xeon"
    / "summary.json"
)
MUTATION_PATH = (
    RESULTS_ROOT
    / "randomized_mutation_audit_seed20260724_128"
    / "randomized_mutation_results.json"
)

EXPECTED_HASHES = {
    ABLATION_PATH: "babe4e941e22495d42f84b89d842d1b102edabf926989c40c51b1425d99aab76",
    ORACLE_PATH: "a5d6531a346092b8185aab916515de9a69729981770823ab5b2b5b70614e64e7",
    MUTATION_PATH: "91c5c2874d24939e9b6300db5d1d4f33f6a6df07925e2b78abf15841b82e30b2",
}

DEFAULT_OUTPUT_DIR = RESULTS_ROOT / "figure2_semantic_evidence_ccfa_20260725_v3"
WIDTH_IN = 4.72
HEIGHT_IN = 3.48
PNG_DPI = 400
MIN_FONT_PT = 6.5

# Okabe--Ito derived, with light neutral fills for grayscale separation.
INK = "#17202A"
BLUE = "#0072B2"
SKY = "#56B4E9"
GREEN = "#009E73"
ORANGE = "#E69F00"
VERMILION = "#D55E00"
MID_GRAY = "#69737D"
LIGHT_GRAY = "#D7DCE0"
PALE_GRAY = "#F4F6F7"
PALE_BLUE = "#E7F2F8"
PALE_GREEN = "#E5F4EE"
WHITE = "#FFFFFF"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--visual-review-pass",
        action="store_true",
        help="Record PASS only after the generated color and grayscale PNGs were inspected.",
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


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def configure_style() -> None:
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


def validate_provenance() -> dict[str, str]:
    observed: dict[str, str] = {}
    for path, expected in EXPECTED_HASHES.items():
        require(path.is_file(), f"Missing frozen input: {path}")
        digest = sha256(path)
        require(digest == expected, f"Unexpected hash for {path}: {digest}")
        observed[str(path.relative_to(PROJECT_ROOT))] = digest
    return observed


def load_evidence() -> dict[str, Any]:
    ablation = read_json(ABLATION_PATH)
    oracle = read_json(ORACLE_PATH)
    mutation = read_json(MUTATION_PATH)

    counts = oracle["comparison"]["counts"]
    require(oracle["status"] == "PASS_independent_trace_oracle_full_domain", "Oracle failed.")
    require(counts["requested_kernels"] == 24624, "Unexpected requested mechanism count.")
    require(counts["oracle_valid_kernels"] == 10544, "Unexpected valid mechanism count.")
    require(counts["trace_cases"] == 759168, "Unexpected trace census.")
    require(counts["candidate_comparisons"] == 189792, "Unexpected signature census.")
    mismatch_fields = (
        "validity_mismatch_count",
        "factorized_validity_mismatch_count",
        "flat_trace_mismatch_count",
        "factorized_trace_mismatch_count",
        "direct_candidate_mismatch_count",
        "factorized_candidate_mismatch_count",
    )
    require(all(counts[field] == 0 for field in mismatch_fields), "Oracle mismatch.")

    targeted = oracle["mutant_smoke"]
    require(
        targeted["mutants_total"] == 7
        and targeted["mutants_detected"] == 7
        and targeted["all_detected"],
        "Targeted mutation audit did not detect all seven faults.",
    )

    mutation_summary = mutation["summary"]
    require(mutation_summary["mutants"] == 60, "Expected 60 fixed-seed mutants.")
    require(mutation_summary["trace_detected"] == 56, "Expected 56 trace-changing mutants.")
    require(mutation_summary["signature_detected"] == 49, "Expected 49 signature detections.")
    require(
        mutation_summary["independent_oracle_detected"] == 56,
        "Unexpected independent-oracle detection count.",
    )

    census = ablation["collision_census"]
    replay = ablation["fault_replay"]
    require(ablation["valid_kernel_count"] == 10544, "Ablation mechanism count mismatch.")
    require(ablation["oracle_contexts"] == 3382177, "Ablation context count mismatch.")
    require(ablation["gates"]["full_key_control_pass"], "Full-key control failed.")
    require(census["full"]["unsafe_key_classes"] == 0, "Full key has unsafe classes.")
    require(
        replay["full"]["canonical"]["trace_mismatches"] == 0
        and replay["full"]["reverse"]["trace_mismatches"] == 0,
        "Full-key trace replay failed.",
    )

    variants = ("drop_state", "drop_memory", "drop_observation")
    for variant in variants:
        require(census[variant]["unsafe_key_classes"] > 0, f"No collision for {variant}.")
        require(census[variant]["affected_valid_kernels"] > 0, f"No affected mechanism for {variant}.")
        for order in ("canonical", "reverse"):
            require(replay[variant][order]["trace_mismatches"] > 0, f"No trace failure for {variant}/{order}.")
            require(
                replay[variant][order]["candidate_signature_mismatches"] > 0,
                f"No signature failure for {variant}/{order}.",
            )

    return {
        "oracle_counts": counts,
        "targeted": targeted,
        "mutation_summary": mutation_summary,
        "collision_census": census,
        "fault_replay": replay,
        "valid_kernels": ablation["valid_kernel_count"],
    }


def panel_label(axis: plt.Axes, label: str, title: str) -> None:
    axis.text(
        0.0,
        1.02,
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
        0.115,
        1.02,
        title,
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.2,
        fontweight="bold",
        color=INK,
        clip_on=False,
    )


def draw_oracle_panel(axis: plt.Axes, evidence: dict[str, Any]) -> None:
    counts = evidence["oracle_counts"]
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    panel_label(axis, "(a)", "Separate interpreter: zero mismatches")

    rows = [
        (
            f"Validity decisions  ({counts['oracle_valid_kernels']:,} valid)",
            counts["requested_kernels"],
        ),
        ("Complete traces", counts["trace_cases"]),
        ("Six-bit signatures", counts["candidate_comparisons"]),
    ]
    ys = [0.74, 0.46, 0.18]
    for (name, count), y in zip(rows, ys):
        axis.add_patch(
            FancyBboxPatch(
                (0.01, y - 0.105),
                0.98,
                0.205,
                boxstyle="round,pad=0.006,rounding_size=0.018",
                facecolor=PALE_BLUE,
                edgecolor=BLUE,
                linewidth=0.65,
            )
        )
        axis.text(0.04, y + 0.035, f"{count:,}", ha="left", va="center", fontsize=8.0, fontweight="bold", color=BLUE)
        axis.text(0.04, y - 0.045, name, ha="left", va="center", fontsize=6.5, color=INK)
        axis.plot([0.61, 0.75], [y, y], color=BLUE, linewidth=1.2, solid_capstyle="round")
        axis.scatter([0.77], [y], s=32, marker="o", facecolor=GREEN, edgecolor=INK, linewidth=0.55, zorder=3)
        axis.text(0.82, y, "0 mismatch", ha="left", va="center", fontsize=6.5, fontweight="bold", color=GREEN)


def draw_mutation_panel(axis: plt.Axes, evidence: dict[str, Any]) -> None:
    summary = evidence["mutation_summary"]
    axis.set_xlim(0, 60)
    axis.set_ylim(-0.58, 2.90)
    axis.axis("off")
    panel_label(axis, "(b)", "Fault sensitivity: trace vs. projection")

    labels = ("Catalog", "Behavior-changing", "Six-bit detected")
    y_positions = (2.28, 1.36, 0.44)
    segments = (
        ((60, BLUE, "", None),),
        ((56, GREEN, "", None), (4, LIGHT_GRAY, "////", MID_GRAY)),
        ((49, ORANGE, "", None), (7, SKY, "\\\\\\\\", BLUE), (4, LIGHT_GRAY, "////", MID_GRAY)),
    )
    for label, y, row in zip(labels, y_positions, segments):
        left = 0
        for width, color, hatch, edge in row:
            axis.barh(
                [y],
                [width],
                left=[left],
                height=0.36,
                color=color,
                edgecolor=edge or WHITE,
                linewidth=0.45,
                hatch=hatch,
            )
            left += width
        axis.text(
            0,
            y + 0.25,
            label,
            ha="left",
            va="bottom",
            fontsize=6.5,
            color=INK,
        )
        axis.text(row[0][0] - 1.0, y, str(row[0][0]), ha="right", va="center", fontsize=7.0, fontweight="bold", color=WHITE)
    axis.text(60.0, 1.61, "+ 4 inactive/equiv.", ha="right", va="bottom", fontsize=6.5, color=MID_GRAY)
    axis.text(58.0, 1.36, "4", ha="center", va="center", fontsize=6.5, fontweight="bold", color=INK)
    axis.text(60.0, 0.82, "+ 7 trace-only", ha="right", va="bottom", fontsize=6.5, color=BLUE)
    axis.text(60.0, 0.64, "+ 4 inactive/equiv.", ha="right", va="bottom", fontsize=6.5, color=MID_GRAY)
    axis.text(52.5, 0.44, "7", ha="center", va="center", fontsize=6.5, fontweight="bold", color=INK)
    axis.text(58.0, 0.44, "4", ha="center", va="center", fontsize=6.5, fontweight="bold", color=INK)

    axis.add_patch(
        FancyBboxPatch(
            (0.0, -0.49),
            60.0,
            0.29,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor=PALE_GREEN,
            edgecolor=GREEN,
            linewidth=0.65,
        )
    )
    axis.text(30.0, -0.345, "Targeted semantic faults: 7 / 7 detected", ha="center", va="center", fontsize=6.5, fontweight="bold", color=GREEN)


def style_metric_axis(axis: plt.Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="x", color=LIGHT_GRAY, linewidth=0.5)
    axis.set_axisbelow(True)
    axis.tick_params(axis="y", length=0)


def draw_key_consequences(fig: plt.Figure, evidence: dict[str, Any]) -> None:
    census = evidence["collision_census"]
    replay = evidence["fault_replay"]
    valid = evidence["valid_kernels"]
    variants = ("drop_state", "drop_memory", "drop_observation")
    labels = ("drop state", "drop memory", "drop observation")
    colors = (VERMILION, BLUE, ORANGE)
    markers = ("o", "s", "D")
    y = np.arange(3)[::-1]

    title_axis = fig.add_axes([0.06, 0.435, 0.90, 0.055])
    title_axis.axis("off")
    title_axis.text(0.0, 0.5, "(c)", ha="left", va="center", fontsize=7.2, fontweight="bold", color=INK)
    title_axis.text(0.045, 0.5, "Removing any key field propagates to downstream outputs", ha="left", va="center", fontsize=7.2, fontweight="bold", color=INK)

    ax_classes = fig.add_axes([0.19, 0.105, 0.20, 0.285])
    ax_affected = fig.add_axes([0.455, 0.105, 0.18, 0.285])
    ax_outputs = fig.add_axes([0.715, 0.105, 0.255, 0.285])

    unsafe = [census[v]["unsafe_key_classes"] for v in variants]
    affected = [100.0 * census[v]["affected_valid_kernels"] / valid for v in variants]

    for yi, value, color, marker in zip(y, unsafe, colors, markers):
        ax_classes.hlines(yi, 1e4, value, color=color, linewidth=2.3)
        ax_classes.scatter([value], [yi], s=28, marker=marker, facecolor=color, edgecolor=INK, linewidth=0.45, zorder=3)
        ax_classes.text(value * 1.13, yi, f"{value / 1000:.0f}k", ha="left", va="center", fontsize=6.5, color=INK)
    ax_classes.set_xscale("log")
    ax_classes.set_xlim(1e4, 7e5)
    ax_classes.set_xticks([1e4, 1e5])
    ax_classes.set_xticklabels(["10k", "100k"])
    ax_classes.set_yticks(y, labels)
    ax_classes.set_title("Unsafe key classes (log)", pad=4, fontsize=6.8, fontweight="bold")
    style_metric_axis(ax_classes)

    label_offsets = (-0.22, 0.22, 0.22)
    for yi, value, color, marker, y_offset in zip(y, affected, colors, markers, label_offsets):
        ax_affected.hlines(yi, 0, value, color=color, linewidth=2.3)
        ax_affected.scatter([value], [yi], s=28, marker=marker, facecolor=color, edgecolor=INK, linewidth=0.45, zorder=3)
        ax_affected.text(value - 2.0, yi + y_offset, f"{value:.1f}%", ha="right", va="center", fontsize=6.5, color=INK)
    ax_affected.set_xlim(0, 105)
    ax_affected.set_xticks([0, 50, 100])
    ax_affected.set_yticks(y)
    ax_affected.set_yticklabels([])
    ax_affected.set_title("Affected mechanisms", pad=4, fontsize=6.8, fontweight="bold")
    style_metric_axis(ax_affected)

    trace_c, trace_r, sig_c, sig_r = [], [], [], []
    for variant in variants:
        canonical = replay[variant]["canonical"]
        reverse = replay[variant]["reverse"]
        trace_c.append(100.0 * canonical["trace_mismatches"] / canonical["traces"])
        trace_r.append(100.0 * reverse["trace_mismatches"] / reverse["traces"])
        sig_c.append(100.0 * canonical["candidate_signature_mismatches"] / canonical["candidates"])
        sig_r.append(100.0 * reverse["candidate_signature_mismatches"] / reverse["candidates"])

    for yi, tc, tr, sc, sr, color in zip(y, trace_c, trace_r, sig_c, sig_r, colors):
        ax_outputs.plot([tc, tr], [yi + 0.13, yi + 0.13], color=color, linewidth=0.85)
        ax_outputs.scatter([tc], [yi + 0.13], s=25, marker="o", facecolor=color, edgecolor=INK, linewidth=0.45, zorder=3)
        ax_outputs.scatter([tr], [yi + 0.13], s=25, marker="o", facecolor=WHITE, edgecolor=color, linewidth=0.9, zorder=3)
        ax_outputs.plot([sc, sr], [yi - 0.13, yi - 0.13], color=color, linewidth=0.85)
        ax_outputs.scatter([sc], [yi - 0.13], s=27, marker="D", facecolor=color, edgecolor=INK, linewidth=0.45, zorder=3)
        ax_outputs.scatter([sr], [yi - 0.13], s=27, marker="D", facecolor=WHITE, edgecolor=color, linewidth=0.9, zorder=3)

    ax_outputs.set_xscale("log")
    ax_outputs.set_xlim(0.2, 70)
    ax_outputs.set_xticks([0.3, 1, 3, 10, 30])
    ax_outputs.set_xticklabels(["0.3", "1", "3", "10", "30"])
    ax_outputs.set_yticks(y)
    ax_outputs.set_yticklabels([])
    ax_outputs.set_title("Mismatched outputs (%, log)", pad=4, fontsize=6.8, fontweight="bold")
    style_metric_axis(ax_outputs)

    fig.text(
        0.705,
        0.025,
        "circles: trace   diamonds: signature   filled/open: canonical/reverse",
        ha="center",
        va="bottom",
        fontsize=6.5,
        color=MID_GRAY,
    )


def build_figure(evidence: dict[str, Any]) -> plt.Figure:
    fig = plt.figure(figsize=(WIDTH_IN, HEIGHT_IN))
    oracle_axis = fig.add_axes([0.06, 0.565, 0.405, 0.34])
    mutation_axis = fig.add_axes([0.57, 0.565, 0.40, 0.34])
    draw_oracle_panel(oracle_axis, evidence)
    draw_mutation_panel(mutation_axis, evidence)
    draw_key_consequences(fig, evidence)
    return fig


def audit_layout(fig: plt.Figure) -> list[str]:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    canvas = fig.bbox
    texts = [
        item
        for item in fig.findobj(match=lambda artist: isinstance(artist, Text))
        if item.get_visible() and item.get_text()
    ]
    sizes = [float(item.get_fontsize()) for item in texts]
    require(sizes and min(sizes) >= MIN_FONT_PT, f"Text below {MIN_FONT_PT:.1f} pt.")
    outside: list[str] = []
    for item in texts:
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
    require(abs(width - WIDTH_IN) < 1e-6 and abs(height - HEIGHT_IN) < 1e-6, "Wrong figure size.")
    return [
        f"final size: {width:.2f} x {height:.2f} in",
        f"minimum text size: {min(sizes):.1f} pt",
        "text outside canvas: none",
    ]


def inspect_pdf(path: Path) -> list[str]:
    reader = PdfReader(str(path))
    require(len(reader.pages) == 1, "Figure PDF must contain one page.")
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
    image_count = sum(
        str(ref.get_object().get("/Subtype")) == "/Image" for ref in xobjects.values()
    )
    require(not type3, f"Type 3 fonts: {type3}")
    require(not unembedded, f"Unembedded fonts: {unembedded}")
    require(image_count == 0, f"Raster image XObjects: {image_count}")
    return [
        "embedded fonts: PASS",
        "Type 3 fonts: none",
        "raster image XObjects: none",
    ]


def export(
    fig: plt.Figure,
    output_dir: Path,
    provenance: dict[str, str],
    visual_review_pass: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    notes = audit_layout(fig)
    base = output_dir / "fig2_semantic_evidence"
    pdf_path = base.with_suffix(".pdf")
    png_path = base.with_suffix(".png")
    gray_path = output_dir / "fig2_semantic_evidence_grayscale.png"
    caption_path = output_dir / "fig2_semantic_evidence_caption.txt"
    qa_path = output_dir / "fig2_semantic_evidence_qa.json"

    fig.savefig(pdf_path, metadata={"Title": "Semantic evidence", "Creator": SCRIPT_PATH.name})
    fig.savefig(png_path, dpi=PNG_DPI, metadata={"Title": "Semantic evidence"})
    plt.close(fig)

    expected_pixels = (round(WIDTH_IN * PNG_DPI), round(HEIGHT_IN * PNG_DPI))
    with Image.open(png_path) as image:
        require(image.size == expected_pixels, f"Unexpected PNG size: {image.size}")
        grayscale = ImageOps.grayscale(image)
        grayscale.save(gray_path, dpi=(PNG_DPI, PNG_DPI))
    with Image.open(gray_path) as grayscale:
        require(grayscale.mode == "L", f"Grayscale image mode is {grayscale.mode}, not L.")
        require(grayscale.size == expected_pixels, "Grayscale PNG dimensions differ.")

    notes.extend(inspect_pdf(pdf_path))
    caption = (
        "Semantic-safety evidence for the complete key. "
        "(a) Across the frozen domain, Flat and complete-key LayerProbe match an "
        "independent interpreter on validity, complete traces, and six-bit signatures. "
        "(b) Complete traces detect all 56 behavior-changing mutants in the fixed-seed "
        "catalog, whereas the six-bit projection intentionally hides seven; four mutants "
        "are inactive or behaviorally equivalent on the sampled domain. All seven targeted "
        "semantic faults are detected. (c) Omitting state, memory, or observation creates "
        "unsafe equivalence classes, affects valid mechanisms, and changes traces and "
        "signatures under both canonical and reverse replay orders."
    )
    caption_path.write_text(caption + "\n", encoding="utf-8")

    qa = {
        "status": "PASS" if visual_review_pass else "PENDING_VISUAL_REVIEW",
        "machine_qa": "PASS",
        "visual_review": "PASS" if visual_review_pass else "PENDING",
        "figure_width_in": WIDTH_IN,
        "figure_height_in": HEIGHT_IN,
        "png_dpi": PNG_DPI,
        "png_pixels": list(expected_pixels),
        "grayscale_mode": "L",
        "notes": notes,
        "provenance": provenance,
        "outputs": {
            "pdf": pdf_path.name,
            "png": png_path.name,
            "grayscale": gray_path.name,
            "caption": caption_path.name,
        },
        "visual_review_checklist": [
            "No missing glyphs or clipped text.",
            "No label overlaps or marks hidden by annotations.",
            "Panel labels and titles align.",
            "The three key variants remain distinct in grayscale by marker shape.",
            "Filled/open replay-order markers remain separable at final width.",
            "All plotted values match the frozen JSON inputs.",
        ],
    }
    qa_path.write_text(json.dumps(qa, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"PDF: {pdf_path}")
    print(f"PNG: {png_path}")
    print(f"Grayscale: {gray_path}")
    print(f"Caption: {caption_path}")
    print(f"QA: {qa_path}")


def main() -> int:
    args = parse_args()
    configure_style()
    provenance = validate_provenance()
    evidence = load_evidence()
    export(
        build_figure(evidence),
        args.output_dir.resolve(),
        provenance,
        args.visual_review_pass,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
