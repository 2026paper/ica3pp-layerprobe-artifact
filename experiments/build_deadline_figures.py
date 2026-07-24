#!/usr/bin/env python
"""Build the minimal, paper-ready figure set from the frozen deadline results.

The script is intentionally read-only with respect to experiment outputs.  It
refuses to render figures unless the formal run is complete and every recorded
semantic equivalence check passed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.transforms import ScaledTranslation


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
TRANSFER_ROOT = SCRIPT_PATH.parents[3]
DEFAULT_RUN_DIR = PROJECT_ROOT / "results" / "deadline_paper_v2_20260723_xeon"
DEFAULT_COMMUNICATION_DIR = (
    PROJECT_ROOT / "results" / "communication_full_24624_20260723_xeon"
)
DEFAULT_OUTPUT_DIR = TRANSFER_ROOT / "07_论文" / "figures"

EXPECTED_WORKERS = [1, 2, 4, 6, 8, 12, 16]
PRESENTATION_COUNTS = [2, 6, 10, 14, 18]
MODE_ORDER = ["exact", "coarse", "hidden"]
LNCS_FIGURE_WIDTH_IN = 4.72

BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
GRAY = "#666666"
LIGHT_GRAY = "#B9B9B9"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument(
        "--communication-dir", type=Path, default=DEFAULT_COMMUNICATION_DIR
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


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


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.5,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 6.7,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.2,
            "lines.markersize": 4.5,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 400,
        }
    )


def save_figure(fig: plt.Figure, output_dir: Path, stem: str, title: str) -> list[Path]:
    outputs = [output_dir / f"{stem}.pdf", output_dir / f"{stem}.png"]
    fig.savefig(
        outputs[0],
        metadata={"Title": title, "Creator": SCRIPT_PATH.name},
    )
    fig.savefig(
        outputs[1],
        dpi=400,
        metadata={"Title": title},
    )
    plt.close(fig)
    return outputs


def add_panel_labels(fig: plt.Figure, axes: Iterable[plt.Axes]) -> None:
    """Place panel labels with one points-based offset for exact alignment."""

    offset = ScaledTranslation(-11 / 72, 4 / 72, fig.dpi_scale_trans)
    for axis, label in zip(axes, ("(a)", "(b)")):
        axis.text(
            0,
            1,
            label,
            transform=axis.transAxes + offset,
            fontweight="bold",
            va="bottom",
            ha="right",
        )


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
    checks = summary.get("semantic_checks", [])
    require(checks, "No semantic checks were recorded.")
    require(
        all(check.get("status") == "PASS" for check in checks),
        "At least one semantic equivalence check did not pass.",
    )
    require(len(runs) == 257, f"Expected 257 run rows, found {len(runs)}.")
    require(
        int(summary["metadata"]["physical_cores"]) == 8,
        "Figure design assumes the verified 8-physical-core workstation.",
    )
    require(
        int(summary["metadata"]["logical_cores"]) == 16,
        "Figure design assumes the verified 16-logical-processor workstation.",
    )
    require(
        len(delay_rows) == 9,
        f"Expected the complete 3x3 delay table, found {len(delay_rows)} rows.",
    )
    communication_candidates = communication_summary.get("candidate_count")
    if communication_candidates is None:
        communication_candidates = communication_summary.get("candidates")
    require(
        communication_candidates in (None, 189792),
        "Unexpected communication-analysis candidate count.",
    )


def build_scaling_figure(
    summary: dict[str, Any],
    runs: list[dict[str, str]],
    output_dir: Path,
) -> list[Path]:
    scaling_summary = {
        int(row["workers"]): row for row in summary["parallel_scaling"]
    }
    require(
        sorted(scaling_summary) == EXPECTED_WORKERS,
        "Unexpected worker set in parallel-scaling summary.",
    )

    raw: dict[int, dict[int, float]] = defaultdict(dict)
    for row in runs:
        if row["study"] == "parallel_scaling":
            raw[int(row["workers"])][int(row["repeat"])] = float(row["elapsed_s"])
    require(
        all(len(raw[worker]) == 5 for worker in EXPECTED_WORKERS),
        "Each scaling worker count must have five repetitions.",
    )

    speedups = np.array(
        [float(scaling_summary[worker]["speedup"]) for worker in EXPECTED_WORKERS]
    )
    paired_ranges: list[tuple[float, float]] = []
    for worker in EXPECTED_WORKERS:
        if worker == 1:
            paired_ranges.append((1.0, 1.0))
            continue
        paired = [
            raw[1][repeat] / raw[worker][repeat] for repeat in sorted(raw[worker])
        ]
        paired_ranges.append((min(paired), max(paired)))
    lower = speedups - np.array([low for low, _ in paired_ranges])
    upper = np.array([high for _, high in paired_ranges]) - speedups
    lower = np.maximum(lower, 0)
    upper = np.maximum(upper, 0)

    workers = np.array(EXPECTED_WORKERS)
    physical_mask = workers <= 8
    smt_mask = workers > 8

    fig, ax = plt.subplots(figsize=(LNCS_FIGURE_WIDTH_IN, 3.15))
    ax.axvspan(8.5, 16.5, color=ORANGE, alpha=0.07, linewidth=0)
    ax.plot([1, 16], [1, 16], linestyle="--", color=LIGHT_GRAY, label="Ideal")
    ax.errorbar(
        workers[physical_mask],
        speedups[physical_mask],
        yerr=np.vstack((lower[physical_mask], upper[physical_mask])),
        fmt="o-",
        color=BLUE,
        capsize=2,
        elinewidth=0.8,
        markeredgecolor=BLUE,
        markerfacecolor=BLUE,
        label="Physical-core range (1-8)",
        zorder=3,
    )
    ax.plot(
        [8, 12, 16],
        [speedups[4], speedups[5], speedups[6]],
        linestyle="--",
        color=ORANGE,
        linewidth=1.1,
        zorder=2,
    )
    ax.errorbar(
        workers[smt_mask],
        speedups[smt_mask],
        yerr=np.vstack((lower[smt_mask], upper[smt_mask])),
        fmt="s",
        color=ORANGE,
        capsize=2,
        elinewidth=0.8,
        markeredgecolor=ORANGE,
        markerfacecolor="white",
        label="SMT throughput (12, 16)",
        zorder=4,
    )
    ax.axvline(8.5, color=ORANGE, linewidth=0.7, linestyle=":")
    ax.annotate(
        f"{speedups[4]:.2f}×",
        (8, speedups[4]),
        xytext=(-4, 8),
        textcoords="offset points",
        ha="right",
        color=BLUE,
    )
    ax.annotate(
        f"{speedups[6]:.2f}×",
        (16, speedups[6]),
        xytext=(-3, 8),
        textcoords="offset points",
        ha="right",
        color=ORANGE,
    )
    ax.text(
        12.5,
        15.25,
        "SMT-only\nregion",
        ha="center",
        va="top",
        color=GRAY,
        fontsize=6.6,
    )
    ax.set_xlabel("Worker processes")
    ax.set_ylabel("Speedup over 1 worker")
    ax.set_xticks(EXPECTED_WORKERS)
    ax.set_xlim(0.5, 16.5)
    ax.set_ylim(0, 16.7)
    ax.set_yticks([0, 4, 8, 12, 16])
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.5)
    ax.legend(loc="upper left", frameon=False, handlelength=2.1)
    fig.tight_layout(pad=0.4)
    return save_figure(
        fig,
        output_dir,
        "fig1_strong_scaling",
        "LayerProbe strong scaling on an 8-core, 16-thread workstation",
    )


def paired_presentation_data(
    runs: Iterable[dict[str, str]],
) -> tuple[
    dict[int, list[float]],
    dict[int, list[float]],
    dict[int, list[tuple[str, float]]],
]:
    pairs: dict[tuple[str, int], dict[str, dict[str, str]]] = defaultdict(dict)
    cases: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in runs:
        if row["study"] != "presentation_scaling":
            continue
        pairs[(row["case"], int(row["repeat"]))][row["method"]] = row
        cases[row["case"]].append(row)

    ratios: dict[int, list[float]] = defaultdict(list)
    subset_reductions: dict[int, list[float]] = defaultdict(list)
    subset_values: dict[int, list[tuple[str, float]]] = defaultdict(list)
    for (case, _repeat), methods in pairs.items():
        require(
            set(methods) == {"kernel_memo", "factorized"},
            f"Incomplete presentation pair: {case}.",
        )
        memo = methods["kernel_memo"]
        factorized = methods["factorized"]
        require(memo["digest"] == factorized["digest"], f"Digest mismatch: {case}.")
        count = int(memo["presentation_count"])
        ratios[count].append(
            float(memo["elapsed_s"]) / float(factorized["elapsed_s"])
        )

    for case, case_rows in cases.items():
        memo = next(row for row in case_rows if row["method"] == "kernel_memo")
        factorized = next(row for row in case_rows if row["method"] == "factorized")
        require(
            memo["policy_calls"] == memo["transition_calls"]
            and factorized["policy_calls"] == factorized["transition_calls"],
            f"Policy/transition accounting mismatch: {case}.",
        )
        count = int(memo["presentation_count"])
        reduction = 1.0 - (
            float(factorized["policy_calls"]) / float(memo["policy_calls"])
        )
        if not any(name == case for name, _ in subset_values[count]):
            subset_values[count].append((case, reduction))
            subset_reductions[count].append(reduction)

    require(
        sorted(ratios) == PRESENTATION_COUNTS
        and sorted(subset_reductions) == PRESENTATION_COUNTS,
        "Unexpected presentation counts.",
    )
    expected_pairs = {2: 9, 6: 9, 10: 9, 14: 9, 18: 3}
    expected_subsets = {2: 3, 6: 3, 10: 3, 14: 3, 18: 1}
    require(
        all(len(ratios[count]) == expected_pairs[count] for count in PRESENTATION_COUNTS),
        "Unexpected paired-run count.",
    )
    require(
        all(
            len(subset_reductions[count]) == expected_subsets[count]
            for count in PRESENTATION_COUNTS
        ),
        "Unexpected spread-subset count.",
    )
    return ratios, subset_reductions, subset_values


def build_presentation_figure(
    runs: list[dict[str, str]],
    output_dir: Path,
) -> list[Path]:
    ratios, reductions, _subset_values = paired_presentation_data(runs)
    x = np.array(PRESENTATION_COUNTS, dtype=float)
    ratio_medians = np.array([statistics.median(ratios[count]) for count in x])
    ratio_lows = np.array([min(ratios[count]) for count in x])
    ratio_highs = np.array([max(ratios[count]) for count in x])
    reduction_medians = np.array(
        [statistics.median(reductions[count]) * 100 for count in x]
    )
    reduction_lows = np.array([min(reductions[count]) * 100 for count in x])
    reduction_highs = np.array([max(reductions[count]) * 100 for count in x])

    fig, (ax_ratio, ax_calls) = plt.subplots(
        2,
        1,
        figsize=(LNCS_FIGURE_WIDTH_IN, 5.15),
        sharex=True,
    )

    jitter9 = np.linspace(-0.32, 0.32, 9)
    jitter3 = np.array([-0.24, 0.0, 0.24])
    for count in PRESENTATION_COUNTS:
        raw = sorted(ratios[count])
        jitter = jitter9 if len(raw) == 9 else jitter3
        ax_ratio.scatter(
            count + jitter,
            raw,
            s=11,
            facecolors="white",
            edgecolors=GRAY,
            linewidths=0.55,
            alpha=0.85,
            zorder=2,
        )
    ax_ratio.errorbar(
        x,
        ratio_medians,
        yerr=np.vstack((ratio_medians - ratio_lows, ratio_highs - ratio_medians)),
        fmt="D-",
        color=BLUE,
        markerfacecolor=BLUE,
        capsize=3,
        elinewidth=0.9,
        label="Median and observed range",
        zorder=3,
    )
    ax_ratio.axhline(1.0, color=GRAY, linestyle="--", linewidth=0.8)
    ax_ratio.annotate(
        "First measured median > 1",
        (18, ratio_medians[-1]),
        xytext=(-85, 21),
        textcoords="offset points",
        arrowprops={"arrowstyle": "-", "color": GRAY, "linewidth": 0.7},
        color=GRAY,
        fontsize=6.7,
    )
    ax_ratio.text(
        2.2,
        1.073,
        "Factorized faster",
        color=GRAY,
        ha="left",
        va="top",
        fontsize=6.5,
    )
    ax_ratio.text(
        2.2,
        0.872,
        "Memo baseline faster",
        color=GRAY,
        ha="left",
        va="bottom",
        fontsize=6.5,
    )
    ax_ratio.set_ylabel(r"Paired time ratio $T_{\mathrm{memo}}/T_{\mathrm{fact}}$")
    ax_ratio.set_xticks(PRESENTATION_COUNTS)
    ax_ratio.set_xlim(0.8, 19.2)
    ax_ratio.set_ylim(0.865, 1.08)
    ax_ratio.grid(axis="y", color="#D9D9D9", linewidth=0.5)
    ax_ratio.legend(loc="lower right", frameon=False)

    subset_markers = ["o", "s", "^"]
    for count in PRESENTATION_COUNTS:
        values = sorted(reductions[count])
        offsets = [0] if len(values) == 1 else [-0.22, 0, 0.22]
        for index, (offset, value) in enumerate(zip(offsets, values)):
            ax_calls.scatter(
                count + offset,
                value * 100,
                marker=subset_markers[index],
                s=20,
                facecolors="white",
                edgecolors=GRAY,
                linewidths=0.7,
                zorder=3,
            )
    ax_calls.errorbar(
        x,
        reduction_medians,
        yerr=np.vstack(
            (reduction_medians - reduction_lows, reduction_highs - reduction_medians)
        ),
        fmt="D-",
        color=GREEN,
        markerfacecolor=GREEN,
        capsize=3,
        elinewidth=0.9,
        label="Subset median and range",
        zorder=2,
    )
    ax_calls.annotate(
        f"{reduction_medians[-1]:.1f}%",
        (18, reduction_medians[-1]),
        xytext=(-3, 8),
        textcoords="offset points",
        ha="right",
        color=GREEN,
    )
    ax_calls.set_xlabel("Number of presentations")
    ax_calls.set_ylabel("Policy/transition-call reduction (%)")
    ax_calls.set_xticks(PRESENTATION_COUNTS)
    ax_calls.set_xlim(0.8, 19.2)
    ax_calls.set_ylim(-1.5, 35)
    ax_calls.grid(axis="y", color="#D9D9D9", linewidth=0.5)
    ax_calls.legend(loc="upper left", frameon=False)
    add_panel_labels(fig, (ax_ratio, ax_calls))
    fig.tight_layout(h_pad=1.15, pad=0.65)
    return save_figure(
        fig,
        output_dir,
        "fig2_presentation_reuse",
        "Presentation-family size, paired runtime, and semantic-step reuse",
    )


def build_delay_heatmap(
    delay_rows: list[dict[str, str]],
    output_dir: Path,
) -> list[Path]:
    lookup = {
        (row["speed_mode"], row["distance_mode"]): float(
            row["mean_pair_delta_delayed_minus_immediate"]
        )
        for row in delay_rows
    }
    require(
        set(lookup)
        == {(speed, distance) for speed in MODE_ORDER for distance in MODE_ORDER},
        "Delay-effect cells do not form the required 3x3 design.",
    )
    matrix = np.array(
        [[lookup[(speed, distance)] for distance in MODE_ORDER] for speed in MODE_ORDER]
    )
    bound = max(abs(float(matrix.min())), abs(float(matrix.max())))

    fig, ax = plt.subplots(figsize=(LNCS_FIGURE_WIDTH_IN, 3.25))
    image = ax.imshow(
        matrix,
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-bound, vcenter=0.0, vmax=bound),
        aspect="equal",
    )
    for row_index in range(3):
        for column_index in range(3):
            value = matrix[row_index, column_index]
            text_color = "white" if abs(value) > 0.29 else "black"
            ax.text(
                column_index,
                row_index,
                f"{value:+.3f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=8,
            )
    pretty_modes = [mode.capitalize() for mode in MODE_ORDER]
    ax.set_xticks(range(3), pretty_modes)
    ax.set_yticks(range(3), pretty_modes)
    ax.set_xlabel("Distance-information mode")
    ax.set_ylabel("Speed-information mode")
    ax.tick_params(which="minor", bottom=False, left=False)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.050, pad=0.045)
    colorbar.set_label(
        "Delayed - immediate\n(mean separated pairs)", rotation=270, labelpad=18
    )
    colorbar.ax.tick_params(labelsize=6.6)
    fig.tight_layout(pad=0.45)
    return save_figure(
        fig,
        output_dir,
        "fig3_delay_delta_heatmap",
        "Effect of one-step delay on model-pair distinguishability",
    )


def write_readme(
    output_dir: Path,
    run_dir: Path,
    communication_dir: Path,
    summary: dict[str, Any],
    runs_path: Path,
    summary_path: Path,
    delay_path: Path,
) -> Path:
    scaling = {int(row["workers"]): row for row in summary["parallel_scaling"]}
    presentation = {
        int(row["presentation_count"]): row for row in summary["presentation_scaling"]
    }
    content = f"""# Paper-ready deadline figures

Generated with:

```powershell
python.exe experiments\\build_deadline_figures.py
```

Run the command from `02_代码/ICA3PP`. The script aborts unless the formal
257-job run is complete and every recorded semantic-equivalence check is
`PASS`.

## Files and caption-ready descriptions

- `fig1_strong_scaling.pdf` / `.png` — **Strong scaling on the deadline
  workstation.** Median speedup for the complete 24,624-mechanism ×
  18-presentation workload over five rotating-order repetitions. Error bars
  show the observed range of repeat-matched speedups. Workers 1–8 are within
  the eight physical cores; 12 and 16 are explicitly reported as SMT
  throughput points, not additional physical-core scaling. The measured
  speedups are {float(scaling[8]["speedup"]):.2f}× at 8 workers and
  {float(scaling[16]["speedup"]):.2f}× at 16 workers.
- `fig2_presentation_reuse.pdf` / `.png` — **Reuse versus presentation-family
  size.** (a) Paired wall-clock ratio of the schedule-matched kernel-memo
  baseline to LayerProbe factorization; small points are individual paired
  runs, diamonds are medians, and bars are observed minima–maxima. The first
  measured median above the 1.0 break-even line occurs at 18 presentations
  ({float(presentation[18]["paired_speedup_median"]):.3f}×); this is an
  observed crossover, not a fitted threshold. (b) Reduction in policy and
  transition calls; open markers are the three deterministic spread subsets
  at 2/6/10/14 presentations (the 18-presentation point is the full family),
  and bars give the across-subset range. The full family reduces these calls
  by {100 * float(presentation[18]["semantic_step_call_reduction"]):.1f}%.
- `fig3_delay_delta_heatmap.pdf` / `.png` — **Computational effect of
  one-step delay.** Each cell is the mean change in the number of separated
  model pairs, delayed minus immediate, over all 10,544 valid mechanisms for
  the corresponding speed- and distance-information modes. Negative values
  mean fewer model pairs were distinguished after delay.

## Frozen sources

- `{runs_path.relative_to(TRANSFER_ROOT).as_posix()}`  
  SHA-256: `{sha256(runs_path)}`
- `{summary_path.relative_to(TRANSFER_ROOT).as_posix()}`  
  SHA-256: `{sha256(summary_path)}`
- `{delay_path.relative_to(TRANSFER_ROOT).as_posix()}`  
  SHA-256: `{sha256(delay_path)}`

Formal result directory: `{run_dir.relative_to(TRANSFER_ROOT).as_posix()}`  
Communication-analysis directory:
`{communication_dir.relative_to(TRANSFER_ROOT).as_posix()}`

## Interpretation boundary

These figures support only the deterministic finite braking case study on one
8-core/16-thread workstation. The 12- and 16-worker points quantify SMT
throughput, not physical-core scalability. Min–max bars are descriptive
observed ranges, not confidence intervals. The presentation-count crossover
is sampled at only five family sizes and should not be interpolated into an
exact threshold. The call-reduction and delay heatmap are properties of the
declared computational agents; they do not establish human learning,
diagnostic accuracy, or communication effectiveness.
"""
    path = output_dir / "README.md"
    path.write_text(content, encoding="utf-8")
    return path


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    communication_dir = args.communication_dir.resolve()
    output_dir = args.output_dir.resolve()

    summary_path = run_dir / "summary.json"
    runs_path = run_dir / "runs.csv"
    communication_summary_path = communication_dir / "summary.json"
    delay_path = communication_dir / "delay_effects.csv"
    for path in (
        summary_path,
        runs_path,
        communication_summary_path,
        delay_path,
    ):
        require(path.is_file(), f"Missing required input: {path}")

    summary = read_json(summary_path)
    runs = read_csv(runs_path)
    communication_summary = read_json(communication_summary_path)
    delay_rows = read_csv(delay_path)
    validate_inputs(summary, runs, communication_summary, delay_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    configure_matplotlib()
    outputs: list[Path] = []
    outputs.extend(build_scaling_figure(summary, runs, output_dir))
    outputs.extend(build_presentation_figure(runs, output_dir))
    outputs.extend(build_delay_heatmap(delay_rows, output_dir))
    outputs.append(
        write_readme(
            output_dir,
            run_dir,
            communication_dir,
            summary,
            runs_path,
            summary_path,
            delay_path,
        )
    )

    for output in outputs:
        require(output.is_file() and output.stat().st_size > 0, f"Empty output: {output}")
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
