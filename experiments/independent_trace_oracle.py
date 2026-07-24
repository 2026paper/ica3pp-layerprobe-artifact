"""Independent trace-level oracle for the frozen LayerProbe braking domain.

The reference semantics in this file deliberately import only immutable model
data types.  They do not call or read functions/constants from the mechanics,
evaluator, or workload modules under test.  A small adapter loads those modules
separately for differential comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import platform
import socket
import sys
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime
from itertools import combinations, product
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

# Separation rule: these are data containers/type aliases only.  The oracle
# does not import any executable semantics from the implementation under test.
from layerprobe.model import (  # noqa: E402
    Action,
    AgentMemory,
    DisplayMemory,
    KernelSpec,
    Observation,
    PresentationSpec,
    Trace,
    WorldState,
)


ORACLE_AGENT_NAMES: tuple[str, ...] = (
    "reference",
    "instant_stop",
    "speed_only",
    "friction_blind",
)
ORACLE_MODEL_PAIRS: tuple[tuple[str, str], ...] = tuple(
    combinations(ORACLE_AGENT_NAMES, 2)
)
ORACLE_ACTIONS: tuple[Action, ...] = ("coast", "brake")
MUTANT_IDS: tuple[str, ...] = (
    "cache_key_observation_only",
    "cache_scope_cross_agent",
    "delay_returns_current",
    "coarse_rounds_to_nearest",
    "presentation_intervenes_on_dynamics",
    "goal_end_is_exclusive",
    "signed_distance_missing_collision",
)


@dataclass(frozen=True, slots=True)
class OracleVerification:
    faithful: bool
    solvable: bool
    principle_required: bool
    states: int
    transitions: int

    @property
    def valid(self) -> bool:
        return self.faithful and self.solvable and self.principle_required


# ---------------------------------------------------------------------------
# Independent frozen-spec generation
# ---------------------------------------------------------------------------


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if int(config.get("schema_version", -1)) != 1:
        raise ValueError("unsupported independent-oracle config schema")
    if tuple(config["agents"]) != ORACLE_AGENT_NAMES:
        raise ValueError("config agents differ from the frozen oracle order")
    if tuple(config["semantic_mutants"]) != MUTANT_IDS:
        raise ValueError("config semantic mutant set/order is not frozen")
    default_workers = int(config["default_workers"])
    maximum_workers = int(config["maximum_workers"])
    if not 1 <= default_workers <= maximum_workers <= 16:
        raise ValueError("worker limits must satisfy 1 <= default <= maximum <= 16")
    return config


def _inclusive_range(spec: dict[str, Any]) -> tuple[int, ...]:
    start = int(spec["start"])
    stop = int(spec["stop_inclusive"])
    step = int(spec["step"])
    if step <= 0 or stop < start:
        raise ValueError("invalid inclusive integer range")
    return tuple(range(start, stop + 1, step))


def oracle_make_kernels(config: dict[str, Any]) -> tuple[KernelSpec, ...]:
    grid = config["kernel_grid"]
    kernels: list[KernelSpec] = []
    for index, values in enumerate(
        product(
            _inclusive_range(grid["goal_start"]),
            tuple(int(item) for item in grid["brake_force"]),
            tuple(int(item) for item in grid["horizon"]),
            tuple(int(item) for item in grid["goal_width"]),
            _inclusive_range(grid["start_speed"]),
            tuple(int(item) for item in grid["friction"]),
        )
    ):
        goal_start, brake_force, horizon, goal_width, start_speed, friction = values
        kernels.append(
            KernelSpec(
                name=f"brake_{index:04d}",
                start_speed=start_speed,
                friction=friction,
                brake_force=brake_force,
                goal_start=goal_start,
                goal_end=goal_start + goal_width,
                horizon=horizon,
            )
        )
    expected = int(config["full_domain_kernel_count"])
    if len(kernels) != expected:
        raise ValueError(
            f"frozen grid generated {len(kernels)} kernels; expected {expected}"
        )
    return tuple(kernels)


def oracle_make_presentations(config: dict[str, Any]) -> tuple[PresentationSpec, ...]:
    frozen = config["presentations"]
    variants: list[PresentationSpec] = []
    for index, (speed_mode, distance_mode, delay) in enumerate(
        product(
            tuple(str(item) for item in frozen["speed_modes"]),
            tuple(str(item) for item in frozen["distance_modes"]),
            tuple(int(item) for item in frozen["delays"]),
        )
    ):
        variants.append(
            PresentationSpec(
                name=f"view_{index:02d}_{speed_mode}_{distance_mode}_d{delay}",
                speed_mode=speed_mode,
                distance_mode=distance_mode,
                delay=delay,
            )
        )
    if len(variants) != 18:
        raise ValueError(f"frozen presentation grid generated {len(variants)}, not 18")
    return tuple(variants)


# ---------------------------------------------------------------------------
# Independent reference semantics
# ---------------------------------------------------------------------------


def oracle_kernel_well_formed(spec: KernelSpec) -> bool:
    return (
        bool(spec.name)
        and spec.start_speed > 0
        and spec.friction >= 0
        and spec.brake_force > 0
        and spec.goal_start >= spec.start_position
        and spec.goal_end >= spec.goal_start
        and spec.horizon > 0
    )


def oracle_presentation_well_formed(spec: PresentationSpec) -> bool:
    return (
        bool(spec.name)
        and spec.speed_mode in {"exact", "coarse", "hidden"}
        and spec.distance_mode in {"exact", "coarse", "hidden"}
        and spec.delay in {0, 1}
    )


def oracle_initial_state(spec: KernelSpec) -> WorldState:
    return WorldState(
        position=spec.start_position,
        speed=spec.start_speed,
        step=0,
        used_brake=False,
    )


def oracle_terminal_status(state: WorldState, spec: KernelSpec) -> str:
    if state.speed == 0:
        if spec.goal_start <= state.position <= spec.goal_end:
            return "win"
        return "stopped"
    if state.position > spec.goal_end:
        return "overshoot"
    if state.step >= spec.horizon:
        return "timeout"
    return "running"


def oracle_transition(
    state: WorldState,
    action: Action,
    spec: KernelSpec,
) -> WorldState:
    if oracle_terminal_status(state, spec) != "running":
        return state
    deceleration = spec.friction
    if action == "brake":
        deceleration += spec.brake_force
    new_speed = max(0, state.speed - deceleration)
    return WorldState(
        position=state.position + new_speed,
        speed=new_speed,
        step=state.step + 1,
        used_brake=state.used_brake or action == "brake",
    )


def oracle_verify_kernel(spec: KernelSpec) -> OracleVerification:
    if not oracle_kernel_well_formed(spec):
        return OracleVerification(False, False, False, 0, 0)

    start = oracle_initial_state(spec)
    queue: deque[WorldState] = deque((start,))
    visited: set[WorldState] = {start}
    transitions = 0
    solvable = False
    winning_without_brake = False

    while queue:
        state = queue.popleft()
        status = oracle_terminal_status(state, spec)
        if status == "win":
            solvable = True
            if not state.used_brake:
                winning_without_brake = True
            continue
        if status != "running":
            continue
        for action in ORACLE_ACTIONS:
            next_state = oracle_transition(state, action, spec)
            transitions += 1
            if next_state not in visited:
                visited.add(next_state)
                queue.append(next_state)

    return OracleVerification(
        faithful=True,
        solvable=solvable,
        principle_required=solvable and not winning_without_brake,
        states=len(visited),
        transitions=transitions,
    )


def oracle_encode(value: int, mode: str, width: int) -> int:
    if mode == "hidden":
        return -1
    if mode == "coarse":
        return (value // width) * width
    return value


def oracle_raw_observation(
    state: WorldState,
    spec: KernelSpec,
    presentation: PresentationSpec,
) -> Observation:
    status_code = {
        "running": 0,
        "win": 1,
        "stopped": 2,
        "overshoot": 3,
        "timeout": 4,
    }[oracle_terminal_status(state, spec)]
    return (
        oracle_encode(state.speed, presentation.speed_mode, 2),
        oracle_encode(
            max(0, spec.goal_start - state.position),
            presentation.distance_mode,
            3,
        ),
        int(spec.goal_start <= state.position <= spec.goal_end),
        status_code,
    )


def oracle_observe(
    state: WorldState,
    spec: KernelSpec,
    presentation: PresentationSpec,
    memory: DisplayMemory,
) -> tuple[Observation, DisplayMemory]:
    if not oracle_presentation_well_formed(presentation):
        raise ValueError(f"invalid presentation: {presentation}")
    current = oracle_raw_observation(state, spec, presentation)
    if presentation.delay == 0:
        return current, memory
    output = memory.previous
    if output is None:
        output = (-1, -1, current[2], current[3])
    return output, DisplayMemory(previous=current)


def oracle_initial_agent_memory(spec: KernelSpec) -> AgentMemory:
    return AgentMemory(
        believed_speed=spec.start_speed,
        believed_distance=spec.goal_start - spec.start_position,
    )


def oracle_ingest(
    memory: AgentMemory,
    observation: Observation,
) -> AgentMemory:
    speed, distance, _, _ = observation
    return AgentMemory(
        believed_speed=memory.believed_speed if speed < 0 else speed,
        believed_distance=memory.believed_distance if distance < 0 else distance,
        previous_action=memory.previous_action,
    )


def oracle_stopping_distance(speed: int, deceleration: int) -> int:
    if deceleration <= 0:
        return 10**9
    distance = 0
    current = speed
    while current > 0:
        current = max(0, current - deceleration)
        distance += current
    return distance


def oracle_choose_action(
    agent: str,
    memory: AgentMemory,
    spec: KernelSpec,
) -> Action:
    speed = max(0, memory.believed_speed)
    distance = memory.believed_distance
    if agent == "reference":
        deceleration = spec.friction + spec.brake_force
        return (
            "brake"
            if oracle_stopping_distance(speed, deceleration) >= distance
            else "coast"
        )
    if agent == "instant_stop":
        return "brake" if distance <= max(1, speed) else "coast"
    if agent == "speed_only":
        return "brake" if speed >= 3 else "coast"
    if agent == "friction_blind":
        return (
            "brake"
            if oracle_stopping_distance(speed, spec.brake_force) >= distance
            else "coast"
        )
    raise ValueError(f"unknown oracle agent: {agent}")


def oracle_advance_belief(
    agent: str,
    memory: AgentMemory,
    action: Action,
    spec: KernelSpec,
) -> AgentMemory:
    speed = max(0, memory.believed_speed)
    if agent == "instant_stop":
        new_speed = 0
    elif agent == "friction_blind":
        deceleration = spec.brake_force if action == "brake" else 0
        new_speed = max(0, speed - deceleration)
    else:
        deceleration = spec.friction
        if action == "brake":
            deceleration += spec.brake_force
        new_speed = max(0, speed - deceleration)
    return AgentMemory(
        believed_speed=new_speed,
        believed_distance=memory.believed_distance - new_speed,
        previous_action=action,
    )


def oracle_simulate_trace(
    spec: KernelSpec,
    presentation: PresentationSpec,
    agent: str,
) -> Trace:
    state = oracle_initial_state(spec)
    display_memory = DisplayMemory()
    agent_memory = oracle_initial_agent_memory(spec)
    trace: list[tuple[Observation, Action, str]] = []

    while oracle_terminal_status(state, spec) == "running":
        observation, display_memory = oracle_observe(
            state,
            spec,
            presentation,
            display_memory,
        )
        perceived = oracle_ingest(agent_memory, observation)
        action = oracle_choose_action(agent, perceived, spec)
        next_state = oracle_transition(state, action, spec)
        agent_memory = oracle_advance_belief(agent, perceived, action, spec)
        status = oracle_terminal_status(next_state, spec)
        trace.append((observation, action, status))
        state = next_state
    return tuple(trace)


def oracle_signature_for(traces: dict[str, Trace]) -> int:
    mask = 0
    for index, (left, right) in enumerate(ORACLE_MODEL_PAIRS):
        if traces[left] != traces[right]:
            mask |= 1 << index
    return mask


# ---------------------------------------------------------------------------
# Hashing and witness helpers
# ---------------------------------------------------------------------------


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def update_aggregate_hash(
    digest: "hashlib._Hash",
    label: str,
    item_digest: str,
) -> None:
    digest.update(label.encode("utf-8"))
    digest.update(b"\0")
    digest.update(item_digest.encode("ascii"))
    digest.update(b"\n")


def first_trace_difference(expected: Trace, actual: Trace) -> dict[str, Any]:
    limit = max(len(expected), len(actual))
    for index in range(limit):
        left = expected[index] if index < len(expected) else None
        right = actual[index] if index < len(actual) else None
        if left != right:
            return {
                "step_index": index,
                "oracle_step": left,
                "observed_step": right,
                "oracle_length": len(expected),
                "observed_length": len(actual),
            }
    raise ValueError("traces are equal; no first difference exists")


def trace_witness(
    *,
    kind: str,
    kernel: KernelSpec,
    presentation: PresentationSpec,
    agent: str,
    expected: Trace,
    actual: Trace,
) -> dict[str, Any]:
    return {
        "kind": kind,
        "kernel": asdict(kernel),
        "presentation": asdict(presentation),
        "agent": agent,
        "difference": first_trace_difference(expected, actual),
        "oracle_trace_sha256": sha256_value(expected),
        "observed_trace_sha256": sha256_value(actual),
    }


# ---------------------------------------------------------------------------
# System-under-test adapter.  No oracle function above calls this section.
# ---------------------------------------------------------------------------


def load_system_under_test() -> tuple[Any, Any]:
    mechanics_module = importlib.import_module("layerprobe." + "mechanics")
    evaluator_module = importlib.import_module("layerprobe." + "evaluator")
    return mechanics_module, evaluator_module


def compare_kernel_task(
    payload: tuple[
        KernelSpec,
        tuple[PresentationSpec, ...],
        tuple[str, ...],
    ],
) -> dict[str, Any]:
    kernel, presentations, agents = payload
    sut_mechanics, sut_evaluator = load_system_under_test()

    oracle_report = oracle_verify_kernel(kernel)
    sut_report = sut_mechanics.verify_kernel(kernel)
    factorized = sut_evaluator.run_factorized((kernel,), presentations, workers=1)
    factorized_valid = kernel.name in set(factorized.valid_kernels)

    record: dict[str, Any] = {
        "kernel": kernel.name,
        "oracle_valid": oracle_report.valid,
        "sut_valid": bool(sut_report.valid),
        "factorized_valid": factorized_valid,
        "validity_mismatch_count": int(oracle_report.valid != bool(sut_report.valid)),
        "factorized_validity_mismatch_count": int(
            oracle_report.valid != factorized_valid
        ),
        "trace_cases": 0,
        "flat_trace_comparisons": 0,
        "factorized_trace_comparisons": 0,
        "flat_trace_mismatch_count": 0,
        "factorized_trace_mismatch_count": 0,
        "candidate_comparisons": 0,
        "direct_candidate_mismatch_count": 0,
        "factorized_candidate_mismatch_count": 0,
        "oracle_trace_steps": 0,
        "flat_trace_steps": 0,
        "factorized_trace_steps": 0,
        "first_witness": None,
    }

    oracle_trace_digest = hashlib.sha256()
    flat_trace_digest = hashlib.sha256()
    factorized_trace_digest = hashlib.sha256()
    oracle_candidate_digest = hashlib.sha256()
    sut_candidate_digest = hashlib.sha256()
    factorized_candidate_digest = hashlib.sha256()

    if record["validity_mismatch_count"]:
        record["first_witness"] = {
            "kind": "validity",
            "kernel": asdict(kernel),
            "oracle_valid": oracle_report.valid,
            "sut_valid": bool(sut_report.valid),
        }
    elif record["factorized_validity_mismatch_count"]:
        record["first_witness"] = {
            "kind": "factorized_validity",
            "kernel": asdict(kernel),
            "oracle_valid": oracle_report.valid,
            "factorized_valid": factorized_valid,
        }

    if oracle_report.valid and bool(sut_report.valid):
        factorized_traces_by_agent: dict[str, dict[str, Trace]] = {}
        for agent in agents:
            # This private SUT helper is intentionally accessed only inside the
            # adapter.  It exposes the actual complete-key memoized traces
            # without changing production source code.
            traces, _ = sut_evaluator._memoized_agent_traces(  # noqa: SLF001
                kernel,
                presentations,
                agent,
            )
            factorized_traces_by_agent[agent] = traces
        for presentation in presentations:
            oracle_traces: dict[str, Trace] = {}
            sut_traces: dict[str, Trace] = {}
            for agent in agents:
                expected = oracle_simulate_trace(kernel, presentation, agent)
                observed_flat = sut_mechanics.simulate_flat(
                    kernel,
                    presentation,
                    agent,
                )
                observed_factorized = factorized_traces_by_agent[agent][
                    presentation.name
                ]
                oracle_traces[agent] = expected
                sut_traces[agent] = observed_flat
                record["trace_cases"] += 1
                record["flat_trace_comparisons"] += 1
                record["factorized_trace_comparisons"] += 1
                record["oracle_trace_steps"] += len(expected)
                record["flat_trace_steps"] += len(observed_flat)
                record["factorized_trace_steps"] += len(observed_factorized)
                label = f"{presentation.name}\0{agent}"
                update_aggregate_hash(
                    oracle_trace_digest,
                    label,
                    sha256_value(expected),
                )
                update_aggregate_hash(
                    flat_trace_digest,
                    label,
                    sha256_value(observed_flat),
                )
                update_aggregate_hash(
                    factorized_trace_digest,
                    label,
                    sha256_value(observed_factorized),
                )
                if expected != observed_flat:
                    record["flat_trace_mismatch_count"] += 1
                    if record["first_witness"] is None:
                        record["first_witness"] = trace_witness(
                            kind="flat_trace",
                            kernel=kernel,
                            presentation=presentation,
                            agent=agent,
                            expected=expected,
                            actual=observed_flat,
                        )
                if expected != observed_factorized:
                    record["factorized_trace_mismatch_count"] += 1
                    if record["first_witness"] is None:
                        record["first_witness"] = trace_witness(
                            kind="factorized_trace",
                            kernel=kernel,
                            presentation=presentation,
                            agent=agent,
                            expected=expected,
                            actual=observed_factorized,
                        )

            oracle_mask = oracle_signature_for(oracle_traces)
            sut_mask = oracle_signature_for(sut_traces)
            candidate = f"{kernel.name}::{presentation.name}"
            factorized_mask = factorized.candidate_signatures.get(candidate)
            record["candidate_comparisons"] += 1
            update_aggregate_hash(
                oracle_candidate_digest,
                candidate,
                sha256_value(oracle_mask),
            )
            update_aggregate_hash(
                sut_candidate_digest,
                candidate,
                sha256_value(sut_mask),
            )
            update_aggregate_hash(
                factorized_candidate_digest,
                candidate,
                sha256_value(factorized_mask),
            )
            if oracle_mask != sut_mask:
                record["direct_candidate_mismatch_count"] += 1
                if record["first_witness"] is None:
                    record["first_witness"] = {
                        "kind": "direct_candidate_signature",
                        "kernel": asdict(kernel),
                        "presentation": asdict(presentation),
                        "oracle_mask": oracle_mask,
                        "sut_mask": sut_mask,
                    }
            if oracle_mask != factorized_mask:
                record["factorized_candidate_mismatch_count"] += 1
                if record["first_witness"] is None:
                    record["first_witness"] = {
                        "kind": "factorized_candidate_signature",
                        "kernel": asdict(kernel),
                        "presentation": asdict(presentation),
                        "oracle_mask": oracle_mask,
                        "factorized_mask": factorized_mask,
                    }

    record["oracle_trace_sha256"] = oracle_trace_digest.hexdigest()
    record["flat_trace_sha256"] = flat_trace_digest.hexdigest()
    record["factorized_trace_sha256"] = factorized_trace_digest.hexdigest()
    record["oracle_candidate_sha256"] = oracle_candidate_digest.hexdigest()
    record["sut_candidate_sha256"] = sut_candidate_digest.hexdigest()
    record["factorized_candidate_sha256"] = factorized_candidate_digest.hexdigest()
    return record


# ---------------------------------------------------------------------------
# Fixed semantic mutants for checker-adequacy smoke
# ---------------------------------------------------------------------------


def _mutant_coarse_encode(value: int, mode: str, width: int) -> int:
    if mode == "hidden":
        return -1
    if mode == "coarse":
        return ((value + width // 2) // width) * width
    return value


def _mutant_terminal_status(
    state: WorldState,
    spec: KernelSpec,
    mutant_id: str,
) -> str:
    if mutant_id != "goal_end_is_exclusive":
        return oracle_terminal_status(state, spec)
    if state.speed == 0:
        if spec.goal_start <= state.position < spec.goal_end:
            return "win"
        return "stopped"
    if state.position >= spec.goal_end:
        return "overshoot"
    if state.step >= spec.horizon:
        return "timeout"
    return "running"


def _mutant_raw_observation(
    state: WorldState,
    spec: KernelSpec,
    presentation: PresentationSpec,
    mutant_id: str,
) -> Observation:
    terminal = _mutant_terminal_status(state, spec, mutant_id)
    status_code = {
        "running": 0,
        "win": 1,
        "stopped": 2,
        "overshoot": 3,
        "timeout": 4,
    }[terminal]
    encoder = (
        _mutant_coarse_encode
        if mutant_id == "coarse_rounds_to_nearest"
        else oracle_encode
    )
    distance = (
        spec.goal_start - state.position
        if mutant_id == "signed_distance_missing_collision"
        else max(0, spec.goal_start - state.position)
    )
    return (
        encoder(state.speed, presentation.speed_mode, 2),
        encoder(
            distance,
            presentation.distance_mode,
            3,
        ),
        int(spec.goal_start <= state.position <= spec.goal_end),
        status_code,
    )


def _mutant_observe(
    state: WorldState,
    spec: KernelSpec,
    presentation: PresentationSpec,
    memory: DisplayMemory,
    mutant_id: str,
) -> tuple[Observation, DisplayMemory]:
    current = _mutant_raw_observation(state, spec, presentation, mutant_id)
    if presentation.delay == 0:
        return current, memory
    if mutant_id == "delay_returns_current":
        return current, DisplayMemory(previous=current)
    output = memory.previous
    if output is None:
        output = (-1, -1, current[2], current[3])
    return output, DisplayMemory(previous=current)


def _mutant_transition(
    state: WorldState,
    action: Action,
    spec: KernelSpec,
    presentation: PresentationSpec,
    mutant_id: str,
) -> WorldState:
    if _mutant_terminal_status(state, spec, mutant_id) != "running":
        return state
    deceleration = spec.friction
    if action == "brake":
        deceleration += spec.brake_force
        if (
            mutant_id == "presentation_intervenes_on_dynamics"
            and presentation.distance_mode == "hidden"
        ):
            deceleration += 1
    new_speed = max(0, state.speed - deceleration)
    return WorldState(
        position=state.position + new_speed,
        speed=new_speed,
        step=state.step + 1,
        used_brake=state.used_brake or action == "brake",
    )


def _simulate_simple_mutant(
    spec: KernelSpec,
    presentation: PresentationSpec,
    agent: str,
    mutant_id: str,
) -> Trace:
    state = oracle_initial_state(spec)
    display_memory = DisplayMemory()
    agent_memory = oracle_initial_agent_memory(spec)
    trace: list[tuple[Observation, Action, str]] = []
    while _mutant_terminal_status(state, spec, mutant_id) == "running":
        observation, display_memory = _mutant_observe(
            state,
            spec,
            presentation,
            display_memory,
            mutant_id,
        )
        perceived = oracle_ingest(agent_memory, observation)
        action = oracle_choose_action(agent, perceived, spec)
        next_state = _mutant_transition(
            state,
            action,
            spec,
            presentation,
            mutant_id,
        )
        agent_memory = oracle_advance_belief(agent, perceived, action, spec)
        status = _mutant_terminal_status(next_state, spec, mutant_id)
        trace.append((observation, action, status))
        state = next_state
    return tuple(trace)


def _observation_only_cache_traces(
    kernel: KernelSpec,
    presentations: tuple[PresentationSpec, ...],
    agent: str,
) -> dict[str, Trace]:
    cache: dict[
        Observation,
        tuple[Action, WorldState, AgentMemory, str],
    ] = {}
    traces: dict[str, Trace] = {}
    for presentation in presentations:
        state = oracle_initial_state(kernel)
        agent_memory = oracle_initial_agent_memory(kernel)
        display_memory = DisplayMemory()
        trace: list[tuple[Observation, Action, str]] = []
        executed_steps = 0
        while oracle_terminal_status(state, kernel) == "running":
            executed_steps += 1
            if executed_steps > kernel.horizon + 2:
                trace.append(((-9, -9, -9, -9), "coast", "mutant_nontermination"))
                break
            observation, display_memory = oracle_observe(
                state,
                kernel,
                presentation,
                display_memory,
            )
            cached = cache.get(observation)
            if cached is None:
                perceived = oracle_ingest(agent_memory, observation)
                action = oracle_choose_action(agent, perceived, kernel)
                next_state = oracle_transition(state, action, kernel)
                next_memory = oracle_advance_belief(
                    agent,
                    perceived,
                    action,
                    kernel,
                )
                status = oracle_terminal_status(next_state, kernel)
                cached = (action, next_state, next_memory, status)
                cache[observation] = cached
            action, state, agent_memory, status = cached
            trace.append((observation, action, status))
        traces[presentation.name] = tuple(trace)
    return traces


def _cross_agent_cache_traces(
    kernel: KernelSpec,
    presentation: PresentationSpec,
    agents: tuple[str, ...],
) -> dict[str, Trace]:
    cache: dict[
        tuple[WorldState, AgentMemory, Observation],
        tuple[Action, WorldState, AgentMemory, str],
    ] = {}
    traces: dict[str, Trace] = {}
    for agent in agents:
        state = oracle_initial_state(kernel)
        agent_memory = oracle_initial_agent_memory(kernel)
        display_memory = DisplayMemory()
        trace: list[tuple[Observation, Action, str]] = []
        executed_steps = 0
        while oracle_terminal_status(state, kernel) == "running":
            executed_steps += 1
            if executed_steps > kernel.horizon + 2:
                trace.append(((-9, -9, -9, -9), "coast", "mutant_nontermination"))
                break
            observation, display_memory = oracle_observe(
                state,
                kernel,
                presentation,
                display_memory,
            )
            key = (state, agent_memory, observation)
            cached = cache.get(key)
            if cached is None:
                perceived = oracle_ingest(agent_memory, observation)
                action = oracle_choose_action(agent, perceived, kernel)
                next_state = oracle_transition(state, action, kernel)
                next_memory = oracle_advance_belief(
                    agent,
                    perceived,
                    action,
                    kernel,
                )
                status = oracle_terminal_status(next_state, kernel)
                cached = (action, next_state, next_memory, status)
                cache[key] = cached
            action, state, agent_memory, status = cached
            trace.append((observation, action, status))
        traces[agent] = tuple(trace)
    return traces


def mutant_traces_for_kernel(
    mutant_id: str,
    kernel: KernelSpec,
    presentations: tuple[PresentationSpec, ...],
    agents: tuple[str, ...] = ORACLE_AGENT_NAMES,
) -> dict[tuple[str, str], Trace]:
    if mutant_id not in MUTANT_IDS:
        raise ValueError(f"unknown semantic mutant: {mutant_id}")
    traces: dict[tuple[str, str], Trace] = {}
    if mutant_id == "cache_key_observation_only":
        for agent in agents:
            agent_traces = _observation_only_cache_traces(
                kernel,
                presentations,
                agent,
            )
            for presentation in presentations:
                traces[(presentation.name, agent)] = agent_traces[presentation.name]
        return traces
    if mutant_id == "cache_scope_cross_agent":
        for presentation in presentations:
            presentation_traces = _cross_agent_cache_traces(
                kernel,
                presentation,
                agents,
            )
            for agent in agents:
                traces[(presentation.name, agent)] = presentation_traces[agent]
        return traces
    for presentation in presentations:
        for agent in agents:
            traces[(presentation.name, agent)] = _simulate_simple_mutant(
                kernel,
                presentation,
                agent,
                mutant_id,
            )
    return traces


def run_mutant_smoke(
    kernels: Iterable[KernelSpec],
    presentations: tuple[PresentationSpec, ...],
    agents: tuple[str, ...] = ORACLE_AGENT_NAMES,
) -> dict[str, Any]:
    remaining = set(MUTANT_IDS)
    results: dict[str, dict[str, Any]] = {
        mutant_id: {
            "mutant": mutant_id,
            "detected": False,
            "valid_kernels_examined": 0,
            "trace_comparisons": 0,
            "first_witness": None,
        }
        for mutant_id in MUTANT_IDS
    }

    requested = 0
    valid = 0
    for kernel in kernels:
        requested += 1
        if not oracle_verify_kernel(kernel).valid:
            continue
        valid += 1
        expected = {
            (presentation.name, agent): oracle_simulate_trace(
                kernel,
                presentation,
                agent,
            )
            for presentation in presentations
            for agent in agents
        }
        for mutant_id in tuple(item for item in MUTANT_IDS if item in remaining):
            row = results[mutant_id]
            row["valid_kernels_examined"] += 1
            observed = mutant_traces_for_kernel(
                mutant_id,
                kernel,
                presentations,
                agents,
            )
            for presentation in presentations:
                for agent in agents:
                    key = (presentation.name, agent)
                    row["trace_comparisons"] += 1
                    if expected[key] != observed[key]:
                        row["detected"] = True
                        row["first_witness"] = trace_witness(
                            kind=f"semantic_mutant:{mutant_id}",
                            kernel=kernel,
                            presentation=presentation,
                            agent=agent,
                            expected=expected[key],
                            actual=observed[key],
                        )
                        remaining.remove(mutant_id)
                        break
                if mutant_id not in remaining:
                    break
        if not remaining:
            break

    ordered = [results[mutant_id] for mutant_id in MUTANT_IDS]
    return {
        "requested_kernels_examined": requested,
        "valid_kernels_examined": valid,
        "mutants_total": len(MUTANT_IDS),
        "mutants_detected": sum(bool(row["detected"]) for row in ordered),
        "all_detected": not remaining,
        "undetected_mutants": sorted(remaining),
        "mutants": ordered,
    }


# ---------------------------------------------------------------------------
# Experiment runner and artifact output
# ---------------------------------------------------------------------------


def machine_metadata() -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "logical_cpus": os.cpu_count(),
    }
    try:
        psutil = importlib.import_module("psutil")
    except ModuleNotFoundError:
        metadata["psutil_available"] = False
    else:
        memory = psutil.virtual_memory()
        metadata.update(
            {
                "psutil_available": True,
                "physical_cpus": psutil.cpu_count(logical=False),
                "memory_total_gib": memory.total / (1024**3),
                "memory_available_gib": memory.available / (1024**3),
            }
        )
    return metadata


def system_under_test_hashes() -> dict[str, str]:
    mechanics_module, evaluator_module = load_system_under_test()
    paths = {
        "model": Path(sys.modules["layerprobe.model"].__file__).resolve(),
        "mechanics": Path(mechanics_module.__file__).resolve(),
        "evaluator": Path(evaluator_module.__file__).resolve(),
    }
    return {name: sha256_file(path) for name, path in paths.items()}


def summarize_kernel_records(
    records: Iterable[dict[str, Any]],
    *,
    requested_kernels: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    counts = {
        "requested_kernels": requested_kernels,
        "processed_kernels": 0,
        "oracle_valid_kernels": 0,
        "sut_valid_kernels": 0,
        "factorized_valid_kernels": 0,
        "validity_mismatch_count": 0,
        "factorized_validity_mismatch_count": 0,
        "trace_cases": 0,
        "flat_trace_comparisons": 0,
        "factorized_trace_comparisons": 0,
        "flat_trace_mismatch_count": 0,
        "factorized_trace_mismatch_count": 0,
        "candidate_comparisons": 0,
        "direct_candidate_mismatch_count": 0,
        "factorized_candidate_mismatch_count": 0,
        "oracle_trace_steps": 0,
        "flat_trace_steps": 0,
        "factorized_trace_steps": 0,
    }
    oracle_valid_digest = hashlib.sha256()
    sut_valid_digest = hashlib.sha256()
    factorized_valid_digest = hashlib.sha256()
    oracle_trace_digest = hashlib.sha256()
    flat_trace_digest = hashlib.sha256()
    factorized_trace_digest = hashlib.sha256()
    oracle_candidate_digest = hashlib.sha256()
    sut_candidate_digest = hashlib.sha256()
    factorized_candidate_digest = hashlib.sha256()
    first_witnesses: list[dict[str, Any]] = []

    materialized: list[dict[str, Any]] = []
    for record in records:
        materialized.append(record)
        counts["processed_kernels"] += 1
        for name in (
            "validity_mismatch_count",
            "factorized_validity_mismatch_count",
            "trace_cases",
            "flat_trace_comparisons",
            "factorized_trace_comparisons",
            "flat_trace_mismatch_count",
            "factorized_trace_mismatch_count",
            "candidate_comparisons",
            "direct_candidate_mismatch_count",
            "factorized_candidate_mismatch_count",
            "oracle_trace_steps",
            "flat_trace_steps",
            "factorized_trace_steps",
        ):
            counts[name] += int(record[name])
        if record["oracle_valid"]:
            counts["oracle_valid_kernels"] += 1
            oracle_valid_digest.update((record["kernel"] + "\n").encode("utf-8"))
        if record["sut_valid"]:
            counts["sut_valid_kernels"] += 1
            sut_valid_digest.update((record["kernel"] + "\n").encode("utf-8"))
        if record["factorized_valid"]:
            counts["factorized_valid_kernels"] += 1
            factorized_valid_digest.update(
                (record["kernel"] + "\n").encode("utf-8")
            )
        update_aggregate_hash(
            oracle_trace_digest,
            record["kernel"],
            record["oracle_trace_sha256"],
        )
        update_aggregate_hash(
            flat_trace_digest,
            record["kernel"],
            record["flat_trace_sha256"],
        )
        update_aggregate_hash(
            factorized_trace_digest,
            record["kernel"],
            record["factorized_trace_sha256"],
        )
        update_aggregate_hash(
            oracle_candidate_digest,
            record["kernel"],
            record["oracle_candidate_sha256"],
        )
        update_aggregate_hash(
            sut_candidate_digest,
            record["kernel"],
            record["sut_candidate_sha256"],
        )
        update_aggregate_hash(
            factorized_candidate_digest,
            record["kernel"],
            record["factorized_candidate_sha256"],
        )
        if record["first_witness"] is not None:
            first_witnesses.append(record["first_witness"])

    hashes = {
        "oracle_valid_kernel_sha256": oracle_valid_digest.hexdigest(),
        "sut_valid_kernel_sha256": sut_valid_digest.hexdigest(),
        "factorized_valid_kernel_sha256": factorized_valid_digest.hexdigest(),
        "oracle_trace_sha256": oracle_trace_digest.hexdigest(),
        "flat_trace_sha256": flat_trace_digest.hexdigest(),
        "factorized_trace_sha256": factorized_trace_digest.hexdigest(),
        "oracle_candidate_sha256": oracle_candidate_digest.hexdigest(),
        "sut_candidate_sha256": sut_candidate_digest.hexdigest(),
        "factorized_candidate_sha256": factorized_candidate_digest.hexdigest(),
    }
    return {
        "counts": counts,
        "hashes": hashes,
        "first_witness": first_witnesses[0] if first_witnesses else None,
    }, materialized


def run_comparison(
    kernels: tuple[KernelSpec, ...],
    presentations: tuple[PresentationSpec, ...],
    agents: tuple[str, ...],
    workers: int,
) -> list[dict[str, Any]]:
    tasks = tuple((kernel, presentations, agents) for kernel in kernels)
    if workers == 1 or len(tasks) <= 1:
        return [compare_kernel_task(task) for task in tasks]
    chunksize = max(1, len(tasks) // (workers * 4))
    with ProcessPoolExecutor(max_workers=workers) as executor:
        return list(
            executor.map(
                compare_kernel_task,
                tasks,
                chunksize=chunksize,
            )
        )


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    config_path = args.config.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"output directory already exists: {output}")
    output.mkdir(parents=True)

    config = load_config(config_path)
    maximum_workers = int(config["maximum_workers"])
    workers = int(
        args.workers
        if args.workers is not None
        else config["default_workers"]
    )
    if not 1 <= workers <= maximum_workers:
        raise ValueError(
            f"workers must be between 1 and configured maximum {maximum_workers}"
        )
    logical_cpus = os.cpu_count()
    if logical_cpus is not None and workers > logical_cpus:
        raise ValueError(
            f"workers={workers} exceeds detected logical CPUs={logical_cpus}"
        )

    all_kernels = oracle_make_kernels(config)
    presentations = oracle_make_presentations(config)
    if args.full_domain:
        mode = "full_domain"
        kernel_count = int(config["full_domain_kernel_count"])
    elif args.kernels is not None:
        mode = "custom"
        kernel_count = int(args.kernels)
    else:
        mode = "smoke"
        kernel_count = int(config["smoke"]["kernel_count"])
    if not 1 <= kernel_count <= len(all_kernels):
        raise ValueError(
            f"kernel count must be in [1, {len(all_kernels)}], got {kernel_count}"
        )
    kernels = all_kernels[:kernel_count]
    mutant_limit = int(config["smoke"]["mutant_search_kernel_count"])
    mutant_kernels = all_kernels[:mutant_limit]

    started_wall = datetime.now().astimezone()
    started = time.perf_counter()
    metadata = {
        "status": "running",
        "mode": mode,
        "started_at": started_wall.isoformat(),
        "machine": machine_metadata(),
        "workers": workers,
        "kernel_count": kernel_count,
        "presentation_count": len(presentations),
        "agent_count": len(ORACLE_AGENT_NAMES),
        "trace_comparisons_planned_upper_bound": (
            kernel_count * len(presentations) * len(ORACLE_AGENT_NAMES)
        ),
        "config_path": str(config_path),
        "hashes": {
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "config_sha256": sha256_file(config_path),
            "system_under_test": system_under_test_hashes(),
        },
        "separation_contract": {
            "oracle_imports_only_model_data_types": True,
            "oracle_reimplements_spec_generation": True,
            "oracle_reimplements_validation": True,
            "oracle_reimplements_transition_observation_and_agents": True,
            "system_under_test_loaded_by_separate_adapter": True,
        },
    }
    write_json(output / "metadata.json", metadata)

    records = run_comparison(
        kernels,
        presentations,
        ORACLE_AGENT_NAMES,
        workers,
    )
    comparison, materialized = summarize_kernel_records(
        records,
        requested_kernels=kernel_count,
    )
    with (output / "kernel_checks.jsonl").open("w", encoding="utf-8") as handle:
        for record in materialized:
            handle.write(
                json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            )

    mutant_smoke = run_mutant_smoke(
        mutant_kernels,
        presentations,
        ORACLE_AGENT_NAMES,
    )
    write_json(output / "mutant_smoke.json", mutant_smoke)

    counts = comparison["counts"]
    primary_mismatches = sum(
        int(counts[name])
        for name in (
            "validity_mismatch_count",
            "factorized_validity_mismatch_count",
            "flat_trace_mismatch_count",
            "factorized_trace_mismatch_count",
            "direct_candidate_mismatch_count",
            "factorized_candidate_mismatch_count",
        )
    )
    complete = counts["processed_kernels"] == counts["requested_kernels"]
    passed = complete and primary_mismatches == 0 and mutant_smoke["all_detected"]
    completed_wall = datetime.now().astimezone()
    summary = {
        "status": (
            "PASS_independent_trace_oracle_smoke"
            if passed and mode != "full_domain"
            else (
                "PASS_independent_trace_oracle_full_domain"
                if passed
                else "FAIL_independent_trace_oracle"
            )
        ),
        "mode": mode,
        "started_at": started_wall.isoformat(),
        "completed_at": completed_wall.isoformat(),
        "elapsed_s": time.perf_counter() - started,
        "workers": workers,
        "comparison": comparison,
        "mutant_smoke": mutant_smoke,
        "claim_boundary": (
            "Independent differential evidence on the selected frozen finite "
            "domain; not a formal proof and not evidence about human users."
        ),
    }
    write_json(output / "summary.json", summary)
    metadata["status"] = summary["status"]
    metadata["completed_at"] = completed_wall.isoformat()
    metadata["elapsed_s"] = summary["elapsed_s"]
    write_json(output / "metadata.json", metadata)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Independent trace-level oracle for LayerProbe",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name(
            "independent_trace_oracle_config.json"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--smoke",
        action="store_true",
        help="run configured small smoke (also the default)",
    )
    mode.add_argument(
        "--full-domain",
        action="store_true",
        help="explicitly run all 24,624 frozen kernels",
    )
    mode.add_argument(
        "--kernels",
        type=int,
        help="developer-only prefix size between smoke and full-domain",
    )
    parser.add_argument(
        "--workers",
        type=int,
        help="kernel-level process count; default 8, maximum 16",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = run_experiment(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["status"].startswith("PASS_") else 2


if __name__ == "__main__":
    raise SystemExit(main())
