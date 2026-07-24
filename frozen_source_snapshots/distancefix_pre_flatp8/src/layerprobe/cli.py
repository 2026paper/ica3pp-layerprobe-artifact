"""Command-line entry points for correctness smoke tests and pilot timing."""

from __future__ import annotations

import argparse
import json
import platform
import time
from dataclasses import asdict

from .evaluator import run_factorized, run_flat
from .workloads import make_kernels, make_presentations


def _summary(result, elapsed: float) -> dict[str, object]:
    return {
        "elapsed_seconds": round(elapsed, 6),
        "valid_kernels": len(result.valid_kernels),
        "candidates": len(result.candidate_signatures),
        "frontier": len(result.frontier),
        "minimum_suite": result.minimum_suite,
        "metrics": result.metrics,
    }


def smoke(workers: int) -> dict[str, object]:
    kernels = make_kernels(12)
    presentations = make_presentations(8)
    started = time.perf_counter()
    flat = run_flat(kernels, presentations)
    flat_elapsed = time.perf_counter() - started
    started = time.perf_counter()
    factorized = run_factorized(kernels, presentations, workers=workers)
    factorized_elapsed = time.perf_counter() - started
    equivalent = flat.comparable() == factorized.comparable()
    if not equivalent:
        raise RuntimeError("flat and factorized outputs differ")
    return {
        "command": "smoke",
        "python": platform.python_version(),
        "workers": workers,
        "equivalent": equivalent,
        "flat": _summary(flat, flat_elapsed),
        "factorized": _summary(factorized, factorized_elapsed),
    }


def benchmark(
    workers_list: tuple[int, ...],
    kernel_count: int,
    presentation_count: int,
    repeats: int,
) -> dict[str, object]:
    kernels = make_kernels(kernel_count)
    presentations = make_presentations(presentation_count)
    rows: list[dict[str, object]] = []
    reference = None
    for workers in workers_list:
        for repeat in range(repeats):
            started = time.perf_counter()
            result = run_factorized(kernels, presentations, workers=workers)
            elapsed = time.perf_counter() - started
            if reference is None:
                reference = result.comparable()
            elif result.comparable() != reference:
                raise RuntimeError("parallel run changed the semantic output")
            rows.append(
                {
                    "workers": workers,
                    "repeat": repeat,
                    **_summary(result, elapsed),
                }
            )
    return {
        "command": "benchmark",
        "python": platform.python_version(),
        "kernel_count": kernel_count,
        "presentation_count": presentation_count,
        "repeats": repeats,
        "rows": rows,
    }


def parse_workers(value: str) -> tuple[int, ...]:
    workers = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not workers or any(item < 1 for item in workers):
        raise argparse.ArgumentTypeError("workers-list must contain positive integers")
    return workers


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="layerprobe")
    subparsers = parser.add_subparsers(dest="command", required=True)

    smoke_parser = subparsers.add_parser("smoke")
    smoke_parser.add_argument("--workers", type=int, default=2)

    benchmark_parser = subparsers.add_parser("benchmark")
    benchmark_parser.add_argument("--workers-list", type=parse_workers, default=(1, 2, 4))
    benchmark_parser.add_argument("--kernel-count", type=int, default=24)
    benchmark_parser.add_argument("--presentation-count", type=int, default=8)
    benchmark_parser.add_argument("--repeats", type=int, default=2)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "smoke":
        payload = smoke(args.workers)
    else:
        payload = benchmark(
            workers_list=args.workers_list,
            kernel_count=args.kernel_count,
            presentation_count=args.presentation_count,
            repeats=args.repeats,
        )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

