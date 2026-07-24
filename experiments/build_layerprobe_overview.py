#!/usr/bin/env python
"""Draw the LayerProbe overview in a compact academic-model-diagram style."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Rectangle


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "results" / "deadline_figures_distancefix_final_20260724_xeon"
)

WIDTH_IN = 4.72
HEIGHT_IN = 2.76
MIN_FONT_PT = 6.0

INK = "#111111"
ARROW = "#8A8A8A"
GRID = "#B9B9B9"
LIGHT_GRID = "#D8D8D8"
WHITE = "#FFFFFF"
LIGHT_GRAY = "#F3F3F3"

PALE_YELLOW = "#F8E8A4"
PALE_BLUE = "#D9E4F2"
PALE_GREEN = "#CDE7BE"
PALE_PEACH = "#F2C7A5"
PALE_PURPLE = "#DED8EE"
PALE_CYAN = "#DDEEF2"

RED = "#D94B43"
BLUE = "#4D8EAE"
PURPLE = "#94629C"
GREEN = "#6A9F65"
ORANGE = "#C97842"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 6.2,
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 400,
        }
    )


def label(
    ax: plt.Axes,
    x: float,
    y: float,
    value: str,
    *,
    size: float = 6.0,
    color: str = INK,
    weight: str = "normal",
    style: str = "normal",
    rotation: float = 0,
    ha: str = "center",
    va: str = "center",
    zorder: int = 20,
) -> None:
    if size < MIN_FONT_PT:
        raise ValueError(f"Font size {size} is below {MIN_FONT_PT} pt.")
    ax.text(
        x,
        y,
        value,
        fontsize=size,
        color=color,
        fontweight=weight,
        fontstyle=style,
        rotation=rotation,
        ha=ha,
        va=va,
        zorder=zorder,
    )


def rect(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    face: str = WHITE,
    edge: str = INK,
    linewidth: float = 0.65,
    radius: float = 0.004,
    linestyle: str | tuple = "-",
    zorder: int = 2,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.002,rounding_size={radius}",
        facecolor=face,
        edgecolor=edge,
        linewidth=linewidth,
        linestyle=linestyle,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def dashed_panel(ax: plt.Axes, x: float, y: float, w: float, h: float) -> None:
    rect(
        ax,
        x,
        y,
        w,
        h,
        face=WHITE,
        edge=INK,
        linewidth=0.65,
        radius=0.004,
        linestyle=(0, (4, 3)),
        zorder=1,
    )


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = ARROW,
    linewidth: float = 0.8,
    connectionstyle: str = "arc3",
    zorder: int = 12,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=7,
            color=color,
            linewidth=linewidth,
            connectionstyle=connectionstyle,
            shrinkA=1.0,
            shrinkB=1.0,
            zorder=zorder,
        )
    )


def grid_matrix(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    rows: int = 3,
    cols: int = 4,
    colors: tuple[str, ...] = (PALE_BLUE, PALE_PURPLE, PALE_GREEN, PALE_PEACH),
    edge: str = GRID,
) -> None:
    rect(ax, x, y, w, h, face=WHITE, edge=INK, linewidth=0.55, radius=0.002)
    pad_x = 0.008
    pad_y = 0.010
    cw = (w - 2 * pad_x) / cols
    ch = (h - 2 * pad_y) / rows
    for row in range(rows):
        for col in range(cols):
            ax.add_patch(
                Rectangle(
                    (x + pad_x + col * cw, y + pad_y + (rows - 1 - row) * ch),
                    cw - 0.002,
                    ch - 0.002,
                    facecolor=colors[(row + col) % len(colors)],
                    edgecolor=edge,
                    linewidth=0.25,
                    zorder=6,
                )
            )


def state_graph(ax: plt.Axes, x: float, y: float, w: float, h: float) -> None:
    rect(ax, x, y, w, h, face=WHITE, edge=INK, linewidth=0.5, radius=0.002)
    nodes = (
        (x + 0.18 * w, y + 0.56 * h),
        (x + 0.48 * w, y + 0.78 * h),
        (x + 0.78 * w, y + 0.57 * h),
        (x + 0.51 * w, y + 0.27 * h),
    )
    for a, b in ((0, 1), (1, 2), (0, 3), (3, 2)):
        ax.plot(
            (nodes[a][0], nodes[b][0]),
            (nodes[a][1], nodes[b][1]),
            color=INK,
            linewidth=0.55,
            zorder=7,
        )
    node_colors = (PALE_CYAN, PALE_PURPLE, PALE_PEACH, PALE_GREEN)
    for index, (px, py) in enumerate(nodes):
        ax.add_patch(
            Circle(
                (px, py),
                radius=0.005,
                facecolor=node_colors[index],
                edgecolor=INK,
                linewidth=0.55,
                zorder=8,
            )
        )


def token_row(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    hidden: bool = False,
) -> None:
    rect(ax, x, y, w, h, face=WHITE, edge=INK, linewidth=0.45, radius=0.002)
    colors = (PALE_BLUE, PALE_GREEN, PALE_PURPLE, PALE_PEACH)
    cell_w = (w - 0.018) / 6
    for index in range(6):
        face = WHITE if hidden and index in (1, 4) else colors[index % 4]
        ax.add_patch(
            Rectangle(
                (x + 0.007 + index * cell_w, y + 0.008),
                cell_w - 0.003,
                h - 0.016,
                facecolor=face,
                edgecolor=GRID,
                linewidth=0.3,
                zorder=7,
            )
        )


def mini_plot(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    output: bool,
) -> None:
    ax.add_patch(
        Rectangle(
            (x, y),
            w,
            h,
            facecolor=WHITE,
            edgecolor=INK,
            linewidth=0.55,
            zorder=3,
        )
    )
    xs = np.linspace(0, 1, 11)
    if output:
        curves = (
            np.array([0.82, 0.59, 0.66, 0.53, 0.70, 0.62, 0.86, 0.56, 0.74, 0.64, 0.90]),
            np.array([0.50, 0.58, 0.47, 0.62, 0.39, 0.64, 0.45, 0.58, 0.38, 0.53, 0.42]),
            np.array([0.18, 0.24, 0.32, 0.20, 0.38, 0.28, 0.42, 0.24, 0.36, 0.18, 0.31]),
        )
    else:
        curves = (
            np.array([0.76, 0.70, 0.67, 0.60, 0.56, 0.49, 0.41, 0.35, 0.26, 0.18, 0.08]),
            np.array([0.48, 0.55, 0.44, 0.51, 0.39, 0.46, 0.33, 0.37, 0.22, 0.26, 0.15]),
            np.array([0.18, 0.22, 0.31, 0.24, 0.34, 0.28, 0.38, 0.30, 0.42, 0.32, 0.47]),
        )
    for curve, color in zip(curves, (RED, BLUE, PURPLE)):
        px = x + 0.035 * w + xs * 0.93 * w
        py = y + curve * h
        ax.plot(px, py, color=color, linewidth=0.55, zorder=7)
        ax.scatter(
            px,
            py,
            s=2.5,
            facecolor=color,
            edgecolor=INK,
            linewidth=0.15,
            zorder=8,
        )
    if not output:
        split_x = x + 0.78 * w
        ax.plot(
            (split_x, split_x),
            (y + 0.05 * h, y + 0.95 * h),
            color=ARROW,
            linestyle=(0, (3, 2)),
            linewidth=0.5,
            zorder=8,
        )


def cache_cylinder(ax: plt.Axes, x: float, y: float, w: float, h: float) -> None:
    ax.add_patch(
        Rectangle(
            (x, y + 0.009),
            w,
            h - 0.018,
            facecolor=PALE_GREEN,
            edgecolor=INK,
            linewidth=0.55,
            zorder=5,
        )
    )
    for cy in (y + 0.009, y + h - 0.009):
        ax.add_patch(
            Ellipse(
                (x + w / 2, cy),
                width=w,
                height=0.019,
                facecolor=PALE_GREEN,
                edgecolor=INK,
                linewidth=0.55,
                zorder=6,
            )
        )
    ax.plot((x + 0.007, x + w - 0.007), (y + 0.44 * h, y + 0.44 * h), color=GRID, linewidth=0.35)


def vertical_stage(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    value: str,
    *,
    face: str,
) -> None:
    rect(ax, x, y, w, h, face=face, edge=INK, linewidth=0.6, radius=0.003)
    label(ax, x + w / 2, y + h / 2, value, size=6.2, rotation=90)


def left_pipeline(ax: plt.Axes) -> None:
    x = 0.012
    w = 0.205
    center = x + w / 2

    label(ax, center, 0.973, "Output", size=7.2)
    mini_plot(ax, x + 0.010, 0.823, w - 0.020, 0.135, output=True)
    arrow(ax, (center, 0.792), (center, 0.817))

    rect(ax, x + 0.010, 0.738, w - 0.020, 0.050, face=LIGHT_GRAY, edge=INK)
    label(ax, center, 0.763, "Candidate signatures")
    arrow(ax, (center, 0.703), (center, 0.733))

    dashed_panel(ax, x + 0.010, 0.505, w - 0.020, 0.195)
    rect(ax, x + 0.030, 0.610, w - 0.060, 0.058, face=PALE_YELLOW, edge=INK)
    label(ax, center, 0.639, "Transition + belief")
    arrow(ax, (center, 0.584), (center, 0.605))
    rect(ax, x + 0.030, 0.526, w - 0.060, 0.058, face=PALE_BLUE, edge=INK)
    label(ax, center, 0.555, "Agent policy")
    label(ax, x + w - 0.002, 0.674, r"$\times|\mathcal{P}|$", size=6.0, ha="right")
    arrow(ax, (center, 0.476), (center, 0.500))

    rect(ax, x + 0.010, 0.421, w - 0.020, 0.052, face=PALE_GREEN, edge=INK)
    label(ax, center, 0.447, "Presentation encoding")
    arrow(ax, (center, 0.392), (center, 0.416))

    rect(ax, x + 0.010, 0.337, w - 0.020, 0.052, face=PALE_PEACH, edge=INK)
    label(ax, center, 0.363, "Validate mechanism")
    arrow(ax, (center, 0.304), (center, 0.332))

    mini_plot(ax, x + 0.010, 0.092, w - 0.020, 0.208, output=False)
    label(ax, center, 0.060, "Input", size=7.2)

    # A side bracket indicates the repeated direct-product path.
    ax.plot(
        (x + w - 0.002, x + w + 0.010, x + w + 0.010, x + w - 0.002),
        (0.298, 0.298, 0.699, 0.699),
        color=ARROW,
        linewidth=0.7,
        zorder=9,
    )
    label(
        ax,
        x + w + 0.020,
        0.500,
        "direct-product path",
        size=6.0,
        rotation=90,
        color=ARROW,
    )


def main_evaluation_block(ax: plt.Axes) -> None:
    x, y, w, h = 0.250, 0.550, 0.738, 0.397
    dashed_panel(ax, x, y, w, h)
    label(ax, x + w / 2, 0.980, "(c) LayerProbe Shared Evaluation Block", size=7.2)
    label(ax, x + w / 2, y + h - 0.026, "Mechanism validation and complete-key semantic reuse", size=6.6)

    vertical_stage(ax, x + 0.012, y + 0.035, 0.033, h - 0.092, "validate once", face=PALE_YELLOW)

    # Valid mechanisms, represented by three matrices.
    rect(ax, x + 0.057, y + 0.035, 0.108, h - 0.092, face=WHITE, edge=INK)
    for index, cy in enumerate((y + 0.255, y + 0.176, y + 0.081)):
        label(ax, x + 0.070, cy + 0.020, rf"$V_{{{1 if index == 0 else (2 if index == 1 else 'K')}}}$", size=6.0)
        grid_matrix(
            ax,
            x + 0.095,
            cy,
            0.058,
            0.058,
            rows=3,
            cols=4,
            colors=(PALE_BLUE, WHITE, PALE_PURPLE, PALE_GREEN),
            edge=LIGHT_GRID,
        )
    label(ax, x + 0.111, y + 0.139, r"$\vdots$", size=7.0)
    arrow(ax, (x + 0.168, y + h / 2), (x + 0.195, y + h / 2))

    # Complete-key reuse group.
    gx, gy, gw, gh = x + 0.198, y + 0.035, 0.265, h - 0.092
    dashed_panel(ax, gx, gy, gw, gh)
    label(ax, gx + gw / 2, gy + gh - 0.024, "Complete-key reuse", size=6.4)

    rect(ax, gx + 0.014, gy + 0.134, 0.102, 0.105, face=WHITE, edge=INK)
    label(ax, gx + 0.065, gy + 0.250, r"$q=(s,m,o)$", size=6.1)
    cell_faces = (PALE_BLUE, PALE_PEACH, PALE_PURPLE)
    cell_labels = (r"$s$", r"$m$", r"$o$")
    for index, (face, value) in enumerate(zip(cell_faces, cell_labels)):
        cell_x = gx + 0.023 + index * 0.030
        rect(ax, cell_x, gy + 0.155, 0.025, 0.058, face=face, edge=GRID, linewidth=0.4, radius=0.001)
        label(ax, cell_x + 0.0125, gy + 0.184, value, size=6.1)

    arrow(ax, (gx + 0.119, gy + 0.187), (gx + 0.143, gy + 0.187))
    cache_cylinder(ax, gx + 0.148, gy + 0.137, 0.044, 0.098)
    label(ax, gx + 0.170, gy + 0.250, "$C$", size=6.1)

    rect(ax, gx + 0.205, gy + 0.164, 0.045, 0.054, face=PALE_GREEN, edge=INK)
    label(ax, gx + 0.228, gy + 0.191, "hit")
    arrow(ax, (gx + 0.194, gy + 0.187), (gx + 0.202, gy + 0.187))

    rect(ax, gx + 0.052, gy + 0.036, 0.157, 0.060, face=PALE_PEACH, edge=INK)
    label(ax, gx + 0.131, gy + 0.066, "miss: policy + transition")
    arrow(
        ax,
        (gx + 0.170, gy + 0.132),
        (gx + 0.168, gy + 0.101),
        connectionstyle="arc3",
    )
    arrow(
        ax,
        (gx + 0.205, gy + 0.096),
        (gx + 0.222, gy + 0.159),
        connectionstyle="arc3,rad=-0.2",
    )

    arrow(ax, (gx + gw + 0.004, y + h / 2), (x + 0.488, y + h / 2))

    # Traces.
    tx = x + 0.492
    rect(ax, tx, y + 0.035, 0.110, h - 0.092, face=PALE_CYAN, edge=INK)
    label(ax, tx + 0.055, y + h - 0.070, "Traces")
    for index, cy in enumerate((y + 0.244, y + 0.165, y + 0.081)):
        state_graph(ax, tx + 0.025, cy, 0.061, 0.055)
        label(ax, tx + 0.010, cy + 0.027, rf"$\tau_{{{1 if index == 0 else (2 if index == 1 else 'P')}}}$", size=6.0)
    label(ax, tx + 0.055, y + 0.138, r"$\vdots$", size=7.0)
    arrow(ax, (tx + 0.114, y + h / 2), (x + 0.625, y + h / 2))

    vertical_stage(ax, x + 0.631, y + 0.035, 0.031, h - 0.092, r"$\bigwedge_p$", face=PALE_BLUE)
    arrow(ax, (x + 0.666, y + h / 2), (x + 0.672, y + h / 2))

    # Robust masks.
    rect(ax, x + 0.674, y + 0.035, 0.054, h - 0.092, face=WHITE, edge=INK)
    label(ax, x + 0.701, y + h - 0.070, r"$\rho(k)$")
    for cy in (y + 0.232, y + 0.139, y + 0.055):
        grid_matrix(
            ax,
            x + 0.681,
            cy,
            0.040,
            0.055,
            rows=3,
            cols=3,
            colors=(PALE_GREEN, WHITE, PALE_PURPLE),
            edge=LIGHT_GRID,
        )


def presentation_family_panel(ax: plt.Axes) -> None:
    x, y, w, h = 0.250, 0.095, 0.330, 0.405
    dashed_panel(ax, x, y, w, h)
    labels = ((r"$p_1$", False), (r"$p_2$", False), (r"$p_{18}$", True))
    ys = (y + 0.283, y + 0.181, y + 0.079)
    for (name, hidden), row_y in zip(labels, ys):
        label(ax, x + 0.030, row_y + 0.035, name, size=6.2)
        token_row(ax, x + 0.060, row_y, w - 0.080, 0.070, hidden=hidden)
    label(ax, x + w / 2, 0.058, "(a) Presentation Family", size=7.0)


def complete_key_panel(ax: plt.Axes) -> None:
    x, y, w, h = 0.600, 0.095, 0.388, 0.405
    dashed_panel(ax, x, y, w, h)

    # Three semantic inputs.
    input_xs = (x + 0.022, x + 0.102, x + 0.182)
    input_faces = (PALE_BLUE, PALE_PEACH, PALE_PURPLE)
    input_labels = ((r"$s$", "world"), (r"$m$", "memory"), (r"$o$", "view"))
    for input_x, face, (symbol, caption) in zip(input_xs, input_faces, input_labels):
        rect(ax, input_x, y + 0.252, 0.065, 0.090, face=face, edge=INK)
        label(ax, input_x + 0.0325, y + 0.302, symbol, size=7.0)
        label(ax, input_x + 0.0325, y + 0.270, caption, size=6.0)

    for input_x in input_xs:
        arrow(
            ax,
            (input_x + 0.0325, y + 0.247),
            (x + 0.192, y + 0.209),
            connectionstyle="arc3",
        )

    rect(ax, x + 0.151, y + 0.145, 0.083, 0.065, face=WHITE, edge=INK)
    label(ax, x + 0.1925, y + 0.1775, r"$q=(s,m,o)$", size=6.1)
    arrow(ax, (x + 0.238, y + 0.177), (x + 0.265, y + 0.177))
    cache_cylinder(ax, x + 0.272, y + 0.130, 0.048, 0.099)
    label(ax, x + 0.296, y + 0.249, "$C[q]$")

    rect(ax, x + 0.328, y + 0.135, 0.046, 0.087, face=PALE_GREEN, edge=INK)
    label(ax, x + 0.351, y + 0.179, "next\nstep", size=6.0)
    arrow(ax, (x + 0.322, y + 0.177), (x + 0.326, y + 0.177))

    rect(ax, x + 0.066, y + 0.045, 0.125, 0.055, face=PALE_PEACH, edge=INK)
    label(ax, x + 0.1285, y + 0.0725, "miss: compute")
    arrow(
        ax,
        (x + 0.296, y + 0.126),
        (x + 0.191, y + 0.101),
        connectionstyle="arc3,rad=0.18",
    )
    rect(ax, x + 0.211, y + 0.045, 0.125, 0.055, face=PALE_GREEN, edge=INK)
    label(ax, x + 0.2735, y + 0.0725, "hit: reuse")
    arrow(
        ax,
        (x + 0.319, y + 0.104),
        (x + 0.349, y + 0.131),
        connectionstyle="arc3,rad=-0.16",
    )

    label(ax, x + w / 2, 0.058, "(b) Complete-Key Step", size=7.0)

    # Reference-style routing arrow from the detailed block into the main block.
    ax.plot(
        (x + 0.300, x + 0.300, x + 0.080),
        (y + h + 0.006, 0.530, 0.530),
        color=ARROW,
        linewidth=0.65,
        zorder=8,
    )
    arrow(ax, (x + 0.080, 0.530), (x + 0.080, 0.551), color=ARROW)


def audit_canvas(fig: plt.Figure) -> None:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    canvas = fig.bbox
    outside: list[str] = []
    for artist in fig.findobj(match=lambda item: hasattr(item, "get_text")):
        if not artist.get_visible() or not artist.get_text():
            continue
        bounds = artist.get_window_extent(renderer=renderer)
        if (
            bounds.x0 < canvas.x0 - 0.5
            or bounds.y0 < canvas.y0 - 0.5
            or bounds.x1 > canvas.x1 + 0.5
            or bounds.y1 > canvas.y1 + 0.5
        ):
            outside.append(artist.get_text())
    if outside:
        raise RuntimeError(f"Text outside canvas: {outside}")


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_style()

    fig, ax = plt.subplots(figsize=(WIDTH_IN, HEIGHT_IN))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    left_pipeline(ax)
    main_evaluation_block(ax)
    presentation_family_panel(ax)
    complete_key_panel(ax)
    audit_canvas(fig)

    pdf_path = output_dir / "fig0_layerprobe_overview.pdf"
    png_path = output_dir / "fig0_layerprobe_overview.png"
    fig.savefig(
        pdf_path,
        metadata={
            "Title": "LayerProbe factorized evaluation overview",
            "Creator": SCRIPT_PATH.name,
        },
    )
    fig.savefig(
        png_path,
        dpi=400,
        metadata={"Title": "LayerProbe factorized evaluation overview"},
    )
    plt.close(fig)

    for path in (pdf_path, png_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Empty output: {path}")
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
