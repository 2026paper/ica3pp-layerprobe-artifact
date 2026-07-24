"""Run a bounded, reproducible single-workstation LayerProbe preflight.

The script intentionally uses only the Python standard library. It writes raw
run-level rows and a machine-readable summary so another computer can continue
without treating exploratory numbers as final paper results.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import gc
import hashlib
import json
import os
import platform
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

from layerprobe.evaluator import RunResult, run_factorized, run_flat, run_kernel_memo
from layerprobe.workloads import make_kernels, make_presentations


class _MemoryStatusEx(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def memory_mb() -> tuple[float | None, float | None]:
    """Return available and total physical memory on Windows, if available."""

    if os.name != "nt":
        return None, None
    status = _MemoryStatusEx()
    status.dwLength = ctypes.sizeof(_MemoryStatusEx)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return None, None
    divisor = 1024 * 1024
    return status.ullAvailPhys / divisor, status.ullTotalPhys / divisor


def result_digest(result: RunResult) -> str:
    payload = {
        "candidate_signatures": sorted(result.candidate_signatures.items()),
        "minimum_suite": result.minimum_suite,
        "valid_kernels": result.valid_kernels,
    }
    encoded = json.dumps(payload, ensure_ascii=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(slots=True)
class RunRow:
    study: str
    case: str
    repeat: int
    order_index: int
    method: str
    workers: int
    kernel_count: int
    presentation_count: int
    elapsed_s: float
    free_before_mb: float | None
    free_after_mb: float | None
    digest: str
    valid_kernels: int
    candidates: int
    frontier: int
    suite_size: int | None
    graph_builds: int
    graph_states: int
    graph_transitions: int
    observation_calls: int
    policy_calls: int
    transition_calls: int
    prefix_groups: int


METHODS: dict[str, Callable[..., RunResult]] = {
    "flat": run_flat,
    "kernel_memo": run_kernel_memo,
    "factorized": run_factorized,
}


def run_once(
    *,
    study: str,
    case: str,
    repeat: int,
    order_index: int,
    method: str,
    workers: int,
    kernel_count: int,
    presentation_count: int,
) -> RunRow:
    kernels = make_kernels(kernel_count)
    presentations = make_presentations(presentation_count)
    free_before, _ = memory_mb()
    started = time.perf_counter()
    if method == "factorized":
        result = run_factorized(kernels, presentations, workers=workers)
    else:
        result = METHODS[method](kernels, presentations)
    elapsed = time.perf_counter() - started
    free_after, _ = memory_mb()
    metrics = result.metrics
    row = RunRow(
        study=study,
        case=case,
        repeat=repeat,
        order_index=order_index,
        method=method,
        workers=workers,
        kernel_count=kernel_count,
        presentation_count=presentation_count,
        elapsed_s=elapsed,
        free_before_mb=free_before,
        free_after_mb=free_after,
        digest=result_digest(result),
        valid_kernels=len(result.valid_kernels),
        candidates=len(result.candidate_signatures),
        frontier=len(result.frontier),
        suite_size=None if result.minimum_suite is None else len(result.minimum_suite),
        graph_builds=metrics["graph_builds"],
        graph_states=metrics["graph_states"],
        graph_transitions=metrics["graph_transitions"],
        observation_calls=metrics["observation_calls"],
        policy_calls=metrics["policy_calls"],
        transition_calls=metrics["transition_calls"],
        prefix_groups=metrics["prefix_groups"],
    )
    print(
        f"[{study}/{case}] r{repeat} {method} w={workers}: "
        f"{elapsed:.3f}s, candidates={row.candidates}, digest={row.digest[:10]}",
        flush=True,
    )
    del result, kernels, presentations
    gc.collect()
    return row


def assert_equivalent(rows: list[RunRow], label: str) -> None:
    digests = {row.digest for row in rows}
    if len(digests) != 1:
        detail = [(row.method, row.workers, row.digest) for row in rows]
        raise AssertionError(f"semantic mismatch in {label}: {detail}")


def median_for(rows: list[RunRow], method: str, workers: int) -> float:
    values = [
        row.elapsed_s for row in rows if row.method == method and row.workers == workers
    ]
    if not values:
        raise ValueError(f"no values for {method}/{workers}")
    return statistics.median(values)


def save_csv(rows: list[RunRow], path: Path) -> None:
    fieldnames = list(asdict(rows[0]).keys())
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def build_summary(rows: list[RunRow], checks: list[dict[str, object]]) -> dict[str, object]:
    method_rows = [row for row in rows if row.study == "method_comparison"]
    scaling_rows = [row for row in rows if row.study == "parallel_scaling"]
    large_rows = [row for row in rows if row.study == "large_instance"]

    medians = {
        "flat_1": median_for(method_rows, "flat", 1),
        "kernel_memo_1": median_for(method_rows, "kernel_memo", 1),
        "factorized_1": median_for(method_rows, "factorized", 1),
        "factorized_4": median_for(method_rows, "factorized", 4),
    }
    factorization_speedup = medians["flat_1"] / medians["factorized_1"]
    prefix_speedup_over_memo = medians["kernel_memo_1"] / medians["factorized_1"]

    scaling_medians = {
        str(workers): median_for(scaling_rows, "factorized", workers)
        for workers in (1, 2, 4)
    }
    scaling = {
        workers: {
            "median_s": scaling_medians[str(workers)],
            "speedup": scaling_medians["1"] / scaling_medians[str(workers)],
            "efficiency": scaling_medians["1"]
            / (workers * scaling_medians[str(workers)]),
        }
        for workers in (1, 2, 4)
    }
    large = {
        str(workers): {
            "elapsed_s": median_for(large_rows, "factorized", workers),
            "candidates": next(row.candidates for row in large_rows if row.workers == workers),
        }
        for workers in (1, 4)
    }
    return {
        "status": "preliminary_only_not_for_paper_claims",
        "semantic_checks": checks,
        "method_comparison_medians_s": medians,
        "flat_to_factorized_1_speedup": factorization_speedup,
        "kernel_memo_to_factorized_1_speedup": prefix_speedup_over_memo,
        "parallel_scaling": scaling,
        "large_instance": large,
        "run_count": len(rows),
    }


def write_markdown(summary: dict[str, object], path: Path) -> None:
    methods = summary["method_comparison_medians_s"]
    scaling = summary["parallel_scaling"]
    large = summary["large_instance"]
    checks = summary["semantic_checks"]
    lines = [
        "# 本机预实验结果（不可直接作为论文最终结果）",
        "",
        "本文件由 `experiments/local_preflight.py` 根据原始运行记录自动生成。",
        "所有数值均属于实现预检；换机后需要冻结代码、基准与实验协议，再进行正式重复实验。",
        "",
        "## 正确性与可复现性检查",
        "",
    ]
    for check in checks:
        lines.append(
            f"- {check['label']}：{check['status']}（比较 {check['runs']} 次运行的完整语义哈希）"
        )
    lines.extend(
        [
            "",
            "## 三种执行方式（5000 个机制 × 18 个呈现）",
            "",
            "| 方法 | 中位耗时（秒） |",
            "|---|---:|",
            f"| 完全平铺 flat | {methods['flat_1']:.3f} |",
            f"| 仅缓存机制验证 kernel_memo | {methods['kernel_memo_1']:.3f} |",
            f"| 分层复用 factorized，1 进程 | {methods['factorized_1']:.3f} |",
            f"| 分层复用 factorized，4 进程 | {methods['factorized_4']:.3f} |",
            "",
            f"平铺到单进程分层复用的探索性加速为 {summary['flat_to_factorized_1_speedup']:.3f}×；",
            f"在已缓存机制验证之后，轨迹前缀复用仍有 {summary['kernel_memo_to_factorized_1_speedup']:.3f}× 的探索性加速。",
            "",
            "## 进程扩展性（10000 个机制 × 18 个呈现，3 次重复）",
            "",
            "| 进程 | 中位耗时（秒） | 相对 1 进程加速 | 并行效率 |",
            "|---:|---:|---:|---:|",
        ]
    )
    for workers in (1, 2, 4):
        item = scaling[str(workers)] if str(workers) in scaling else scaling[workers]
        lines.append(
            f"| {workers} | {item['median_s']:.3f} | {item['speedup']:.3f}× | "
            f"{100 * item['efficiency']:.1f}% |"
        )
    lines.extend(
        [
            "",
            "## 本机较大实例（20000 个机制 × 18 个呈现）",
            "",
            "| 进程 | 耗时（秒） | 完成候选数 |",
            "|---:|---:|---:|",
            f"| 1 | {large['1']['elapsed_s']:.3f} | {large['1']['candidates']} |",
            f"| 4 | {large['4']['elapsed_s']:.3f} | {large['4']['candidates']} |",
            "",
            "## 使用边界",
            "",
            "- 这些结果证明当前代码路径能运行、三种实现语义一致，并给出换机前的性能量级。",
            "- 当前基准的最小套件有时只有一个任务，不能据此支撑最终传播学或诊断有效性结论。",
            "- 当前电脑内存紧张且同时有其他任务，未强行启动 6/8 进程；这不是正式扩展性上限。",
            "- 正式论文需在下一台电脑上完成冻结基准、多次重复、统计区间及更高难度的传播层约束。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-workers", type=int, default=4, choices=(1, 2, 4))
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)

    available, total = memory_mb()
    metadata = {
        "started_at": datetime.now().astimezone().isoformat(),
        "platform": platform.platform(),
        "python": sys.version,
        "logical_cpu_count": os.cpu_count(),
        "max_workers": args.max_workers,
        "available_memory_mb_at_start": available,
        "total_memory_mb": total,
        "purpose": "implementation preflight; not frozen paper results",
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    rows: list[RunRow] = []
    checks: list[dict[str, object]] = []

    correctness = []
    for index, (method, workers) in enumerate(
        (("flat", 1), ("kernel_memo", 1), ("factorized", 1))
    ):
        row = run_once(
            study="correctness",
            case="1200k_18p",
            repeat=0,
            order_index=index,
            method=method,
            workers=workers,
            kernel_count=1200,
            presentation_count=18,
        )
        rows.append(row)
        correctness.append(row)
    assert_equivalent(correctness, "correctness")
    checks.append({"label": "flat/memo/factorized 语义一致", "status": "PASS", "runs": 3})

    method_groups: dict[int, list[RunRow]] = defaultdict(list)
    orders = [
        (("flat", 1), ("kernel_memo", 1), ("factorized", 1), ("factorized", 4)),
        (("factorized", 4), ("factorized", 1), ("kernel_memo", 1), ("flat", 1)),
    ]
    for repeat, order in enumerate(orders):
        for index, (method, workers) in enumerate(order):
            if workers > args.max_workers:
                continue
            row = run_once(
                study="method_comparison",
                case="5000k_18p",
                repeat=repeat,
                order_index=index,
                method=method,
                workers=workers,
                kernel_count=5000,
                presentation_count=18,
            )
            rows.append(row)
            method_groups[repeat].append(row)
    for repeat, group in method_groups.items():
        assert_equivalent(group, f"method_comparison repeat {repeat}")
    checks.append(
        {"label": "方法对照每轮语义一致", "status": "PASS", "runs": sum(map(len, method_groups.values()))}
    )

    scaling_groups: dict[int, list[RunRow]] = defaultdict(list)
    scaling_orders = ((1, 2, 4), (4, 2, 1), (2, 1, 4))
    for repeat, order in enumerate(scaling_orders):
        for index, workers in enumerate(order):
            if workers > args.max_workers:
                continue
            row = run_once(
                study="parallel_scaling",
                case="10000k_18p",
                repeat=repeat,
                order_index=index,
                method="factorized",
                workers=workers,
                kernel_count=10000,
                presentation_count=18,
            )
            rows.append(row)
            scaling_groups[repeat].append(row)
    for repeat, group in scaling_groups.items():
        assert_equivalent(group, f"parallel_scaling repeat {repeat}")
    checks.append(
        {"label": "不同进程数语义一致", "status": "PASS", "runs": sum(map(len, scaling_groups.values()))}
    )

    granularity_groups: dict[str, list[RunRow]] = defaultdict(list)
    for kernel_count in (100, 500, 1000, 2500, 5000):
        case = f"{kernel_count}k_18p"
        for index, workers in enumerate((1, args.max_workers)):
            row = run_once(
                study="kernel_granularity",
                case=case,
                repeat=0,
                order_index=index,
                method="factorized",
                workers=workers,
                kernel_count=kernel_count,
                presentation_count=18,
            )
            rows.append(row)
            granularity_groups[case].append(row)
    for case, group in granularity_groups.items():
        assert_equivalent(group, f"kernel_granularity {case}")
    checks.append(
        {"label": "规模扫描中串并行语义一致", "status": "PASS", "runs": sum(map(len, granularity_groups.values()))}
    )

    presentation_groups: dict[str, list[RunRow]] = defaultdict(list)
    for presentation_count in (2, 4, 8, 12, 18):
        case = f"5000k_{presentation_count}p"
        for index, method in enumerate(("kernel_memo", "factorized")):
            row = run_once(
                study="presentation_scaling",
                case=case,
                repeat=0,
                order_index=index,
                method=method,
                workers=1,
                kernel_count=5000,
                presentation_count=presentation_count,
            )
            rows.append(row)
            presentation_groups[case].append(row)
    for case, group in presentation_groups.items():
        assert_equivalent(group, f"presentation_scaling {case}")
    checks.append(
        {"label": "呈现层规模扫描方法语义一致", "status": "PASS", "runs": sum(map(len, presentation_groups.values()))}
    )

    large_group = []
    for index, workers in enumerate((1, args.max_workers)):
        row = run_once(
            study="large_instance",
            case="20000k_18p",
            repeat=0,
            order_index=index,
            method="factorized",
            workers=workers,
            kernel_count=20000,
            presentation_count=18,
        )
        rows.append(row)
        large_group.append(row)
    assert_equivalent(large_group, "large_instance")
    checks.append({"label": "本机较大实例串并行语义一致", "status": "PASS", "runs": 2})

    save_csv(rows, output / "runs.csv")
    summary = build_summary(rows, checks)
    summary["metadata"] = metadata
    summary["finished_at"] = datetime.now().astimezone().isoformat()
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_markdown(summary, output / "RESULTS_SUMMARY.md")
    print(f"Wrote {len(rows)} runs to {output}", flush=True)


if __name__ == "__main__":
    main()
