"""Balanced repeated benchmark of semantic-step reuse against kernel-only memoization."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import statistics
import time
from datetime import datetime
from pathlib import Path

from layerprobe.evaluator import RunResult, run_factorized, run_kernel_memo
from layerprobe.workloads import make_kernels, make_presentations


def digest(result: RunResult) -> str:
    payload = (
        sorted(result.candidate_signatures.items()),
        result.minimum_suite,
        result.valid_kernels,
    )
    return hashlib.sha256(repr(payload).encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)

    # The two strata test whether reuse starts paying for itself only after the
    # number of valid mechanism/presentation executions grows.
    cases = ((5000, 18, 8), (10000, 18, 5))
    rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []

    warm_kernels = make_kernels(500)
    presentations = make_presentations(18)
    run_kernel_memo(warm_kernels, presentations)
    run_factorized(warm_kernels, presentations, workers=1)

    for kernel_count, presentation_count, repeats in cases:
        kernels = make_kernels(kernel_count)
        presentations = make_presentations(presentation_count)
        pair_times: dict[int, dict[str, float]] = {}
        expected_digest: str | None = None
        factorized_metrics: dict[str, int] | None = None
        memo_metrics: dict[str, int] | None = None
        for repeat in range(repeats):
            order = ("kernel_memo", "factorized") if repeat % 2 == 0 else (
                "factorized",
                "kernel_memo",
            )
            pair_times[repeat] = {}
            for order_index, method in enumerate(order):
                started = time.perf_counter()
                if method == "kernel_memo":
                    result = run_kernel_memo(kernels, presentations)
                    memo_metrics = result.metrics
                else:
                    result = run_factorized(kernels, presentations, workers=1)
                    factorized_metrics = result.metrics
                elapsed = time.perf_counter() - started
                result_digest = digest(result)
                if expected_digest is None:
                    expected_digest = result_digest
                elif result_digest != expected_digest:
                    raise AssertionError(
                        f"semantic mismatch for {kernel_count}/{presentation_count}"
                    )
                rows.append(
                    {
                        "kernel_count": kernel_count,
                        "presentation_count": presentation_count,
                        "repeat": repeat,
                        "order_index": order_index,
                        "method": method,
                        "elapsed_s": elapsed,
                        "digest": result_digest,
                        "candidates": len(result.candidate_signatures),
                        "policy_calls": result.metrics["policy_calls"],
                        "transition_calls": result.metrics["transition_calls"],
                    }
                )
                pair_times[repeat][method] = elapsed
                print(
                    f"{kernel_count}k/{presentation_count}p r{repeat} {method}: {elapsed:.3f}s",
                    flush=True,
                )
                del result
                gc.collect()
        speedups = [
            timings["kernel_memo"] / timings["factorized"]
            for timings in pair_times.values()
        ]
        memo_times = [timings["kernel_memo"] for timings in pair_times.values()]
        factorized_times = [timings["factorized"] for timings in pair_times.values()]
        assert memo_metrics is not None and factorized_metrics is not None
        summaries.append(
            {
                "kernel_count": kernel_count,
                "presentation_count": presentation_count,
                "repeats": repeats,
                "memo_median_s": statistics.median(memo_times),
                "factorized_median_s": statistics.median(factorized_times),
                "paired_speedup_median": statistics.median(speedups),
                "paired_speedup_min": min(speedups),
                "paired_speedup_max": max(speedups),
                "factorized_wins": sum(speedup > 1 for speedup in speedups),
                "memo_policy_calls": memo_metrics["policy_calls"],
                "factorized_policy_calls": factorized_metrics["policy_calls"],
                "policy_call_reduction": 1
                - factorized_metrics["policy_calls"] / memo_metrics["policy_calls"],
            }
        )

    with (output / "runs.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "status": "focused_preliminary_benchmark",
        "generated_at": datetime.now().astimezone().isoformat(),
        "summaries": summaries,
    }
    (output / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# 语义步复用聚焦复测",
        "",
        "本实验采用交替运行顺序，专门比较 `kernel_memo` 与单进程 `factorized`。",
        "它仍受本机其他会话和温度影响，只用于判断是否值得在下一台电脑继续优化。",
        "",
        "| 机制×呈现 | 重复 | memo 中位秒 | factorized 中位秒 | 配对加速中位数 | factorized 胜出轮数 | 策略调用减少 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summaries:
        lines.append(
            f"| {item['kernel_count']}×{item['presentation_count']} | {item['repeats']} | "
            f"{item['memo_median_s']:.3f} | {item['factorized_median_s']:.3f} | "
            f"{item['paired_speedup_median']:.3f}× | {item['factorized_wins']}/{item['repeats']} | "
            f"{100 * item['policy_call_reduction']:.1f}% |"
        )
    lines.extend(
        [
            "",
            "判定规则：若配对中位加速不超过 1，或优势在多数重复中不出现，",
            "则不能把单进程前缀复用写成性能贡献；只能报告工作量减少，并继续优化实现。",
            "",
        ]
    )
    (output / "FOCUSED_PREFIX_REPORT.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=True, indent=2), flush=True)


if __name__ == "__main__":
    main()
