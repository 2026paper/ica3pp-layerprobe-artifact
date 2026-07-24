"""Deterministic development workloads for the first vertical slice."""

from __future__ import annotations

from itertools import product

from .model import KernelSpec, PresentationSpec


def make_kernels(count: int = 24) -> tuple[KernelSpec, ...]:
    if count < 1:
        raise ValueError("count must be positive")
    kernels: list[KernelSpec] = []
    index = 0
    # Vary friction fastest and speed next so every small deterministic prefix
    # spans several speeds and all friction regimes. This avoids a misleading
    # development sample in which the friction-blind agent is observationally
    # identical to the reference one.
    for goal_start, brake, horizon, goal_width, speed, friction in product(
        range(4, 41, 2),
        (1, 2, 3, 4),
        (8, 10, 12),
        (1, 2, 3),
        range(3, 15),
        (0, 1, 2),
    ):
        kernels.append(
            KernelSpec(
                name=f"brake_{index:04d}",
                start_speed=speed,
                friction=friction,
                brake_force=brake,
                goal_start=goal_start,
                goal_end=goal_start + goal_width,
                horizon=horizon,
            )
        )
        index += 1
        if len(kernels) >= count:
            return tuple(kernels)
    raise ValueError(f"requested {count} kernels but deterministic grid produced only {len(kernels)}")


def make_presentations(count: int = 8) -> tuple[PresentationSpec, ...]:
    if count < 1:
        raise ValueError("count must be positive")
    variants: list[PresentationSpec] = []
    index = 0
    for speed_mode, distance_mode, delay in product(
        ("exact", "coarse", "hidden"),
        ("exact", "coarse", "hidden"),
        (0, 1),
    ):
        variants.append(
            PresentationSpec(
                name=f"view_{index:02d}_{speed_mode}_{distance_mode}_d{delay}",
                speed_mode=speed_mode,
                distance_mode=distance_mode,
                delay=delay,
            )
        )
        index += 1
        if len(variants) >= count:
            return tuple(variants)
    return tuple(variants)
