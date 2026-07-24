"""Fixed-seed randomized mutation sensitivity audit.

The seven hand-written mutants in ``independent_trace_oracle.py`` are useful
checker smoke tests, but they are not a sensitivity sample.  This independent
supplement generates a frozen, auditable catalog from three mutation families:

1. local changes to an agent's decision threshold or action mapping;
2. one-unit offsets at goal, observation, overshoot, or horizon boundaries;
3. field-level projections of the complete semantic cache key.

Each mutant is evaluated on a deterministic, stratified sample of valid
braking mechanisms.  Mutated traces are compared with the independent frozen
oracle.  The audit reports exact trace detection and, separately, downstream
six-bit candidate-signature detection.  It is intentionally a sampled
sensitivity analysis, not an exhaustive proof and not a second domain.

The script writes one atomic JSON record per mutant, so interrupted runs resume
without repeating completed mutants.  A manifest freezes the catalog, sample,
configuration, and relevant code fingerprint.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import socket
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime
from itertools import combinations, product
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src"
for import_root in (ROOT, SOURCE_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from experiments import independent_trace_oracle as oracle  # noqa: E402
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


DEFAULT_CONFIG = Path(__file__).with_name(
    "independent_trace_oracle_config.json"
)
DEFAULT_SEED = 20_260_724
DEFAULT_PER_FAMILY = 20
DEFAULT_SAMPLE_KERNELS = 128
DEFAULT_WORKERS = 8
MAX_WORKERS = 16

FAMILIES = (
    "agent_policy",
    "boundary_offset",
    "cache_key_projection",
)
KEY_ORDERS = ("canonical", "reverse")
BOUNDARY_TARGETS = (
    "terminal_goal_start",
    "terminal_goal_end",
    "terminal_overshoot_end",
    "terminal_horizon",
    "observation_goal_start",
    "observation_goal_end",
    "observation_distance_goal_start",
)
ALL_KEY_FIELDS = (
    "state.position",
    "state.speed",
    "state.step",
    "state.used_brake",
    "memory.believed_speed",
    "memory.believed_distance",
    "memory.previous_action",
    "observation.speed",
    "observation.distance",
    "observation.within_goal",
    "observation.status",
)


def now_local() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def stable_json(value: Any, *, indent: int | None = None) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":") if indent is None else None,
        indent=indent,
    )


def digest_value(value: Any) -> str:
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, stable_json(value, indent=2) + "\n")


def _family_rng(seed: int, family: str) -> random.Random:
    material = f"{seed}\0{family}".encode("utf-8")
    derived = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
    return random.Random(derived)


def _sample_without_replacement(
    candidates: Iterable[dict[str, Any]],
    count: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    pool = list(candidates)
    if count > len(pool):
        raise ValueError(
            f"requested {count} mutants from a universe of {len(pool)}"
        )
    rng.shuffle(pool)
    return pool[:count]


def _policy_candidates() -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for agent in oracle.ORACLE_AGENT_NAMES:
        for threshold_shift in (-3, -2, -1, 1, 2, 3):
            candidates.append(
                {
                    "operator": "threshold_shift",
                    "parameters": {
                        "agent": agent,
                        "threshold_shift": threshold_shift,
                    },
                }
            )
        for margin in (-2, -1, 0, 1, 2):
            candidates.append(
                {
                    "operator": "flip_action_at_margin",
                    "parameters": {
                        "agent": agent,
                        "margin": margin,
                    },
                }
            )
    return candidates


def _boundary_candidates() -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    singles = [
        {
            "operator": "boundary_offsets",
            "parameters": {"offsets": {target: delta}},
        }
        for target in BOUNDARY_TARGETS
        for delta in (-1, 1)
    ]
    pairs = [
        {
            "operator": "boundary_offsets",
            "parameters": {
                "offsets": {
                    left: left_delta,
                    right: right_delta,
                }
            },
        }
        for left, right in combinations(BOUNDARY_TARGETS, 2)
        for left_delta, right_delta in product((-1, 1), repeat=2)
    ]
    return singles, pairs


def _key_projection_candidates(
    operator: str,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    if operator == "delete_fields":
        for deleted_count in range(1, 5):
            for deleted in combinations(ALL_KEY_FIELDS, deleted_count):
                kept = tuple(
                    field
                    for field in ALL_KEY_FIELDS
                    if field not in set(deleted)
                )
                candidates.append(
                    {
                        "operator": operator,
                        "parameters": {
                            "deleted_fields": list(deleted),
                            "kept_fields": list(kept),
                        },
                    }
                )
    elif operator == "project_keep_fields":
        for kept_count in range(2, 8):
            for kept in combinations(ALL_KEY_FIELDS, kept_count):
                deleted = tuple(
                    field
                    for field in ALL_KEY_FIELDS
                    if field not in set(kept)
                )
                candidates.append(
                    {
                        "operator": operator,
                        "parameters": {
                            "deleted_fields": list(deleted),
                            "kept_fields": list(kept),
                        },
                    }
                )
    else:
        raise ValueError(f"unknown key projection operator: {operator}")
    return candidates


def _select_key_mutations(
    *,
    count: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Select both deletion and projection forms without duplicate keys."""

    delete_pool = _key_projection_candidates("delete_fields")
    keep_pool = _key_projection_candidates("project_keep_fields")
    rng.shuffle(delete_pool)
    rng.shuffle(keep_pool)
    selected: list[dict[str, Any]] = []
    seen_kept: set[tuple[str, ...]] = set()
    cursors = {"delete_fields": 0, "project_keep_fields": 0}
    pools = {
        "delete_fields": delete_pool,
        "project_keep_fields": keep_pool,
    }

    for position in range(count):
        preferred = (
            "delete_fields"
            if position % 2 == 0
            else "project_keep_fields"
        )
        alternatives = (preferred,) + tuple(
            item for item in pools if item != preferred
        )
        chosen: dict[str, Any] | None = None
        for operator in alternatives:
            pool = pools[operator]
            cursor = cursors[operator]
            while cursor < len(pool):
                candidate = pool[cursor]
                cursor += 1
                kept = tuple(candidate["parameters"]["kept_fields"])
                if kept in seen_kept:
                    continue
                chosen = candidate
                seen_kept.add(kept)
                break
            cursors[operator] = cursor
            if chosen is not None:
                break
        if chosen is None:
            raise ValueError("not enough distinct cache-key projections")
        selected.append(chosen)
    return selected


def _decorate_catalog(
    *,
    family: str,
    seed: int,
    selected: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    catalog: list[dict[str, Any]] = []
    for index, candidate in enumerate(selected):
        payload = {
            "family": family,
            "operator": candidate["operator"],
            "parameters": candidate["parameters"],
            "seed": seed,
        }
        short_digest = digest_value(payload)[:10]
        catalog.append(
            {
                "mutation_id": f"{family}_{index:03d}_{short_digest}",
                "family_index": index,
                **payload,
            }
        )
    return catalog


def generate_mutation_catalog(
    seed: int = DEFAULT_SEED,
    per_family: int = DEFAULT_PER_FAMILY,
) -> tuple[dict[str, Any], ...]:
    """Return a deterministic, JSON-serializable mutation catalog."""

    if per_family < 1:
        raise ValueError("per_family must be positive")

    policy = _sample_without_replacement(
        _policy_candidates(),
        per_family,
        _family_rng(seed, "agent_policy"),
    )

    boundary_rng = _family_rng(seed, "boundary_offset")
    singles, pairs = _boundary_candidates()
    single_count = min(len(singles), (per_family + 1) // 2)
    pair_count = per_family - single_count
    boundary = _sample_without_replacement(
        singles,
        single_count,
        boundary_rng,
    )
    boundary.extend(
        _sample_without_replacement(
            pairs,
            pair_count,
            boundary_rng,
        )
    )
    boundary_rng.shuffle(boundary)

    key_mutations = _select_key_mutations(
        count=per_family,
        rng=_family_rng(seed, "cache_key_projection"),
    )

    catalog = []
    catalog.extend(
        _decorate_catalog(
            family="agent_policy",
            seed=seed,
            selected=policy,
        )
    )
    catalog.extend(
        _decorate_catalog(
            family="boundary_offset",
            seed=seed,
            selected=boundary,
        )
    )
    catalog.extend(
        _decorate_catalog(
            family="cache_key_projection",
            seed=seed,
            selected=key_mutations,
        )
    )
    return tuple(catalog)


def _evenly_spaced_indices(population: int, count: int) -> tuple[int, ...]:
    if count < 0 or count > population:
        raise ValueError("sample count must be within the population")
    if count == 0:
        return ()
    if count == 1:
        return ((population - 1) // 2,)
    return tuple(
        round(index * (population - 1) / (count - 1))
        for index in range(count)
    )


def select_stratified_valid_kernels(
    kernels: Iterable[KernelSpec],
    sample_size: int,
) -> tuple[tuple[KernelSpec, ...], int]:
    """Spread a deterministic sample over friction/brake/horizon strata."""

    valid = [
        kernel
        for kernel in kernels
        if oracle.oracle_verify_kernel(kernel).valid
    ]
    if not 1 <= sample_size <= len(valid):
        raise ValueError(
            f"sample_size must be in [1, {len(valid)}], got {sample_size}"
        )

    strata: dict[tuple[int, int, int], list[KernelSpec]] = defaultdict(list)
    for kernel in valid:
        strata[
            (kernel.friction, kernel.brake_force, kernel.horizon)
        ].append(kernel)
    for members in strata.values():
        members.sort(
            key=lambda item: (
                item.goal_start,
                item.goal_end - item.goal_start,
                item.start_speed,
                item.name,
            )
        )

    keys = sorted(strata)
    base, remainder = divmod(sample_size, len(keys))
    quotas = {
        key: base + int(position < remainder)
        for position, key in enumerate(keys)
    }
    # If a tiny stratum cannot fill its first quota, reassign deterministically.
    deficit = 0
    for key in keys:
        available = len(strata[key])
        if quotas[key] > available:
            deficit += quotas[key] - available
            quotas[key] = available
    while deficit:
        progressed = False
        for key in keys:
            if quotas[key] < len(strata[key]):
                quotas[key] += 1
                deficit -= 1
                progressed = True
                if deficit == 0:
                    break
        if not progressed:
            raise RuntimeError("could not allocate the stratified sample")

    selected: list[KernelSpec] = []
    for key in keys:
        members = strata[key]
        selected.extend(
            members[index]
            for index in _evenly_spaced_indices(
                len(members),
                quotas[key],
            )
        )
    selected.sort(key=lambda item: item.name)
    if len(selected) != sample_size:
        raise RuntimeError("stratified selector returned the wrong sample size")
    return tuple(selected), len(valid)


def _policy_margin(
    agent: str,
    memory: AgentMemory,
    spec: KernelSpec,
) -> int:
    speed = max(0, memory.believed_speed)
    distance = memory.believed_distance
    if agent == "reference":
        stopping = oracle.oracle_stopping_distance(
            speed,
            spec.friction + spec.brake_force,
        )
        return stopping - distance
    if agent == "instant_stop":
        return max(1, speed) - distance
    if agent == "speed_only":
        return speed - 3
    if agent == "friction_blind":
        stopping = oracle.oracle_stopping_distance(
            speed,
            spec.brake_force,
        )
        return stopping - distance
    raise ValueError(f"unknown agent: {agent}")


def _mutated_action(
    mutation: dict[str, Any],
    agent: str,
    memory: AgentMemory,
    spec: KernelSpec,
) -> Action:
    parameters = mutation["parameters"]
    if agent != parameters["agent"]:
        return oracle.oracle_choose_action(agent, memory, spec)
    margin = _policy_margin(agent, memory, spec)
    base_brake = margin >= 0
    if mutation["operator"] == "threshold_shift":
        brake = margin >= int(parameters["threshold_shift"])
    elif mutation["operator"] == "flip_action_at_margin":
        brake = (
            not base_brake
            if margin == int(parameters["margin"])
            else base_brake
        )
    else:
        raise ValueError(
            f"unknown policy mutation operator: {mutation['operator']}"
        )
    return "brake" if brake else "coast"


def _offset(
    offsets: dict[str, int],
    target: str,
) -> int:
    return int(offsets.get(target, 0))


def _boundary_terminal_status(
    state: WorldState,
    spec: KernelSpec,
    offsets: dict[str, int],
) -> str:
    goal_start = spec.goal_start + _offset(
        offsets,
        "terminal_goal_start",
    )
    goal_end = spec.goal_end + _offset(
        offsets,
        "terminal_goal_end",
    )
    if state.speed == 0:
        if goal_start <= state.position <= goal_end:
            return "win"
        return "stopped"
    overshoot_end = spec.goal_end + _offset(
        offsets,
        "terminal_overshoot_end",
    )
    if state.position > overshoot_end:
        return "overshoot"
    horizon = spec.horizon + _offset(offsets, "terminal_horizon")
    if state.step >= horizon:
        return "timeout"
    return "running"


def _boundary_observation(
    state: WorldState,
    spec: KernelSpec,
    presentation: PresentationSpec,
    offsets: dict[str, int],
) -> Observation:
    status_code = {
        "running": 0,
        "win": 1,
        "stopped": 2,
        "overshoot": 3,
        "timeout": 4,
    }[_boundary_terminal_status(state, spec, offsets)]
    goal_start = spec.goal_start + _offset(
        offsets,
        "observation_goal_start",
    )
    goal_end = spec.goal_end + _offset(
        offsets,
        "observation_goal_end",
    )
    distance_origin = spec.goal_start + _offset(
        offsets,
        "observation_distance_goal_start",
    )
    return (
        oracle.oracle_encode(state.speed, presentation.speed_mode, 2),
        oracle.oracle_encode(
            max(0, distance_origin - state.position),
            presentation.distance_mode,
            3,
        ),
        int(goal_start <= state.position <= goal_end),
        status_code,
    )


def _boundary_observe(
    state: WorldState,
    spec: KernelSpec,
    presentation: PresentationSpec,
    memory: DisplayMemory,
    offsets: dict[str, int],
) -> tuple[Observation, DisplayMemory]:
    current = _boundary_observation(
        state,
        spec,
        presentation,
        offsets,
    )
    if presentation.delay == 0:
        return current, memory
    output = memory.previous
    if output is None:
        output = (-1, -1, current[2], current[3])
    return output, DisplayMemory(previous=current)


def _boundary_transition(
    state: WorldState,
    action: Action,
    spec: KernelSpec,
    offsets: dict[str, int],
) -> WorldState:
    if _boundary_terminal_status(state, spec, offsets) != "running":
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


def _nontermination_step() -> tuple[Observation, Action, str]:
    return ((-9, -9, -9, -9), "coast", "mutant_nontermination")


def _simulate_semantic_mutant(
    kernel: KernelSpec,
    presentation: PresentationSpec,
    agent: str,
    mutation: dict[str, Any],
) -> tuple[Trace, bool]:
    family = mutation["family"]
    offsets = (
        {
            str(key): int(value)
            for key, value in mutation["parameters"]["offsets"].items()
        }
        if family == "boundary_offset"
        else {}
    )
    state = oracle.oracle_initial_state(kernel)
    display_memory = DisplayMemory()
    agent_memory = oracle.oracle_initial_agent_memory(kernel)
    trace: list[tuple[Observation, Action, str]] = []
    guard = max(kernel.horizon * 2 + 4, kernel.horizon + 8)

    def terminal(current: WorldState) -> str:
        if family == "boundary_offset":
            return _boundary_terminal_status(current, kernel, offsets)
        return oracle.oracle_terminal_status(current, kernel)

    while terminal(state) == "running":
        if len(trace) >= guard:
            trace.append(_nontermination_step())
            return tuple(trace), True
        if family == "boundary_offset":
            observation, display_memory = _boundary_observe(
                state,
                kernel,
                presentation,
                display_memory,
                offsets,
            )
        else:
            observation, display_memory = oracle.oracle_observe(
                state,
                kernel,
                presentation,
                display_memory,
            )
        perceived = oracle.oracle_ingest(agent_memory, observation)
        if family == "agent_policy":
            action = _mutated_action(
                mutation,
                agent,
                perceived,
                kernel,
            )
        else:
            action = oracle.oracle_choose_action(
                agent,
                perceived,
                kernel,
            )
        if family == "boundary_offset":
            next_state = _boundary_transition(
                state,
                action,
                kernel,
                offsets,
            )
        else:
            next_state = oracle.oracle_transition(
                state,
                action,
                kernel,
            )
        next_memory = oracle.oracle_advance_belief(
            agent,
            perceived,
            action,
            kernel,
        )
        status = terminal(next_state)
        trace.append((observation, action, status))
        state = next_state
        agent_memory = next_memory
    return tuple(trace), False


def project_semantic_key(
    kept_fields: Iterable[str],
    state: WorldState,
    memory: AgentMemory,
    observation: Observation,
) -> tuple[Any, ...]:
    """Project a complete key onto named atomic fields in canonical order."""

    keep = frozenset(kept_fields)
    unknown = keep - frozenset(ALL_KEY_FIELDS)
    if unknown:
        raise ValueError(f"unknown semantic key fields: {sorted(unknown)}")
    values = {
        "state.position": state.position,
        "state.speed": state.speed,
        "state.step": state.step,
        "state.used_brake": state.used_brake,
        "memory.believed_speed": memory.believed_speed,
        "memory.believed_distance": memory.believed_distance,
        "memory.previous_action": memory.previous_action,
        "observation.speed": observation[0],
        "observation.distance": observation[1],
        "observation.within_goal": observation[2],
        "observation.status": observation[3],
    }
    return tuple(values[field] for field in ALL_KEY_FIELDS if field in keep)


def _simulate_key_projection(
    kernel: KernelSpec,
    presentations: tuple[PresentationSpec, ...],
    mutation: dict[str, Any],
    order: str,
) -> tuple[
    dict[str, dict[str, Trace]],
    int,
    int,
    int,
]:
    if order == "canonical":
        ordered_presentations = presentations
    elif order == "reverse":
        ordered_presentations = tuple(reversed(presentations))
    else:
        raise ValueError(f"unsupported key replay order: {order}")

    kept_fields = tuple(mutation["parameters"]["kept_fields"])
    traces: dict[str, dict[str, Trace]] = {
        presentation.name: {} for presentation in presentations
    }
    cache_hits = 0
    cache_misses = 0
    nonterminations = 0

    for agent in oracle.ORACLE_AGENT_NAMES:
        cache: dict[
            tuple[Any, ...],
            tuple[Action, WorldState, AgentMemory, str],
        ] = {}
        for presentation in ordered_presentations:
            state = oracle.oracle_initial_state(kernel)
            display_memory = DisplayMemory()
            agent_memory = oracle.oracle_initial_agent_memory(kernel)
            trace: list[tuple[Observation, Action, str]] = []
            guard = max(kernel.horizon * 2 + 4, kernel.horizon + 8)
            while oracle.oracle_terminal_status(state, kernel) == "running":
                if len(trace) >= guard:
                    trace.append(_nontermination_step())
                    nonterminations += 1
                    break
                observation, display_memory = oracle.oracle_observe(
                    state,
                    kernel,
                    presentation,
                    display_memory,
                )
                key = project_semantic_key(
                    kept_fields,
                    state,
                    agent_memory,
                    observation,
                )
                cached = cache.get(key)
                if cached is None:
                    cache_misses += 1
                    perceived = oracle.oracle_ingest(
                        agent_memory,
                        observation,
                    )
                    action = oracle.oracle_choose_action(
                        agent,
                        perceived,
                        kernel,
                    )
                    next_state = oracle.oracle_transition(
                        state,
                        action,
                        kernel,
                    )
                    next_memory = oracle.oracle_advance_belief(
                        agent,
                        perceived,
                        action,
                        kernel,
                    )
                    status = oracle.oracle_terminal_status(
                        next_state,
                        kernel,
                    )
                    cached = (
                        action,
                        next_state,
                        next_memory,
                        status,
                    )
                    cache[key] = cached
                else:
                    cache_hits += 1
                action, state, agent_memory, status = cached
                trace.append((observation, action, status))
            traces[presentation.name][agent] = tuple(trace)
    return traces, cache_hits, cache_misses, nonterminations


def _compare_candidate(
    *,
    mutation: dict[str, Any],
    sample_rank: int,
    kernel: KernelSpec,
    presentation: PresentationSpec,
    replay_order: str,
    expected: dict[str, Trace],
    actual: dict[str, Trace],
) -> tuple[int, int, dict[str, Any] | None]:
    trace_mismatches = 0
    first_witness: dict[str, Any] | None = None
    oracle_mask = oracle.oracle_signature_for(expected)
    mutant_mask = oracle.oracle_signature_for(actual)
    for agent in oracle.ORACLE_AGENT_NAMES:
        if expected[agent] == actual[agent]:
            continue
        trace_mismatches += 1
        if first_witness is None:
            first_witness = oracle.trace_witness(
                kind="randomized_mutation_trace",
                kernel=kernel,
                presentation=presentation,
                agent=agent,
                expected=expected[agent],
                actual=actual[agent],
            )
            first_witness.update(
                {
                    "mutation_id": mutation["mutation_id"],
                    "family": mutation["family"],
                    "operator": mutation["operator"],
                    "parameters": mutation["parameters"],
                    "sample_rank": sample_rank,
                    "replay_order": replay_order,
                    "oracle_signature_mask": oracle_mask,
                    "mutant_signature_mask": mutant_mask,
                }
            )
    return (
        trace_mismatches,
        int(oracle_mask != mutant_mask),
        first_witness,
    )


def evaluate_mutation(
    payload: tuple[
        dict[str, Any],
        tuple[KernelSpec, ...],
        tuple[PresentationSpec, ...],
        str,
    ],
) -> dict[str, Any]:
    mutation, kernels, presentations, experiment_fingerprint = payload
    trace_cases = 0
    trace_mismatches = 0
    signature_cases = 0
    signature_mismatches = 0
    affected_kernels: set[str] = set()
    cache_hits = 0
    cache_misses = 0
    nontermination_guards = 0
    first_witness: dict[str, Any] | None = None

    for sample_rank, kernel in enumerate(kernels):
        expected = {
            presentation.name: {
                agent: oracle.oracle_simulate_trace(
                    kernel,
                    presentation,
                    agent,
                )
                for agent in oracle.ORACLE_AGENT_NAMES
            }
            for presentation in presentations
        }

        if mutation["family"] == "cache_key_projection":
            actual_sets = []
            for replay_order in KEY_ORDERS:
                actual, hits, misses, guarded = _simulate_key_projection(
                    kernel,
                    presentations,
                    mutation,
                    replay_order,
                )
                cache_hits += hits
                cache_misses += misses
                nontermination_guards += guarded
                actual_sets.append((replay_order, actual))
        else:
            actual = {}
            guarded = 0
            for presentation in presentations:
                actual[presentation.name] = {}
                for agent in oracle.ORACLE_AGENT_NAMES:
                    trace, did_guard = _simulate_semantic_mutant(
                        kernel,
                        presentation,
                        agent,
                        mutation,
                    )
                    actual[presentation.name][agent] = trace
                    guarded += int(did_guard)
            nontermination_guards += guarded
            actual_sets = [("not_applicable", actual)]

        kernel_affected = False
        for replay_order, actual in actual_sets:
            for presentation in presentations:
                mismatches, signature_difference, witness = (
                    _compare_candidate(
                        mutation=mutation,
                        sample_rank=sample_rank,
                        kernel=kernel,
                        presentation=presentation,
                        replay_order=replay_order,
                        expected=expected[presentation.name],
                        actual=actual[presentation.name],
                    )
                )
                trace_cases += len(oracle.ORACLE_AGENT_NAMES)
                trace_mismatches += mismatches
                signature_cases += 1
                signature_mismatches += signature_difference
                if mismatches or signature_difference:
                    kernel_affected = True
                if first_witness is None and witness is not None:
                    first_witness = witness
        if kernel_affected:
            affected_kernels.add(kernel.name)

    trace_detected = trace_mismatches > 0
    signature_detected = signature_mismatches > 0
    independent_oracle_detected = (
        trace_detected or signature_detected
    )
    return {
        "schema_version": 1,
        "experiment_fingerprint": experiment_fingerprint,
        "mutation": mutation,
        "sample_scope": {
            "kernels": len(kernels),
            "presentations": len(presentations),
            "agents": len(oracle.ORACLE_AGENT_NAMES),
            "key_replay_orders": (
                list(KEY_ORDERS)
                if mutation["family"] == "cache_key_projection"
                else []
            ),
        },
        "independent_oracle_detected": independent_oracle_detected,
        "oracle_comparison_detected": independent_oracle_detected,
        "trace_detected": trace_detected,
        "signature_detected": signature_detected,
        "trace_cases": trace_cases,
        "trace_mismatches": trace_mismatches,
        "signature_cases": signature_cases,
        "signature_mismatches": signature_mismatches,
        "affected_kernels": len(affected_kernels),
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
        "nontermination_guards": nontermination_guards,
        "first_witness": first_witness,
        "completed_at": now_local(),
    }


def relevant_code_fingerprint(script_path: Path) -> tuple[str, dict[str, str]]:
    paths = (
        script_path,
        Path(oracle.__file__).resolve(),
        SOURCE_ROOT / "layerprobe" / "model.py",
    )
    file_hashes = {
        path.relative_to(ROOT).as_posix(): sha256_file(path)
        for path in paths
    }
    return digest_value(file_hashes), file_hashes


def _write_csv(path: Path, records: Iterable[dict[str, Any]]) -> None:
    fields = (
        "mutation_id",
        "family",
        "family_index",
        "operator",
        "parameters_json",
        "independent_oracle_detected",
        "oracle_comparison_detected",
        "trace_detected",
        "signature_detected",
        "trace_cases",
        "trace_mismatches",
        "signature_cases",
        "signature_mismatches",
        "affected_kernels",
        "cache_hits",
        "cache_misses",
        "nontermination_guards",
        "first_witness_json",
    )
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            mutation = record["mutation"]
            writer.writerow(
                {
                    "mutation_id": mutation["mutation_id"],
                    "family": mutation["family"],
                    "family_index": mutation["family_index"],
                    "operator": mutation["operator"],
                    "parameters_json": stable_json(
                        mutation["parameters"]
                    ),
                    "independent_oracle_detected": int(
                        record["independent_oracle_detected"]
                    ),
                    "oracle_comparison_detected": int(
                        record["oracle_comparison_detected"]
                    ),
                    "trace_detected": int(record["trace_detected"]),
                    "signature_detected": int(
                        record["signature_detected"]
                    ),
                    "trace_cases": record["trace_cases"],
                    "trace_mismatches": record["trace_mismatches"],
                    "signature_cases": record["signature_cases"],
                    "signature_mismatches": record[
                        "signature_mismatches"
                    ],
                    "affected_kernels": record["affected_kernels"],
                    "cache_hits": record["cache_hits"],
                    "cache_misses": record["cache_misses"],
                    "nontermination_guards": record[
                        "nontermination_guards"
                    ],
                    "first_witness_json": stable_json(
                        record["first_witness"]
                    ),
                }
            )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _summarize(
    records: tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    by_family: dict[str, dict[str, Any]] = {}
    for family in FAMILIES:
        members = [
            record
            for record in records
            if record["mutation"]["family"] == family
        ]
        detected = sum(
            bool(record["independent_oracle_detected"])
            for record in members
        )
        trace_detected = sum(
            bool(record["trace_detected"]) for record in members
        )
        signature_detected = sum(
            bool(record["signature_detected"]) for record in members
        )
        count = len(members)
        by_family[family] = {
            "mutants": count,
            "independent_oracle_detected": detected,
            "independent_oracle_detection_rate": (
                detected / count if count else None
            ),
            "oracle_comparison_detected": detected,
            "oracle_comparison_detection_rate": (
                detected / count if count else None
            ),
            "trace_detected": trace_detected,
            "trace_detection_rate": (
                trace_detected / count if count else None
            ),
            "signature_detected": signature_detected,
            "signature_detection_rate": (
                signature_detected / count if count else None
            ),
            "trace_cases": sum(
                int(record["trace_cases"]) for record in members
            ),
            "trace_mismatches": sum(
                int(record["trace_mismatches"]) for record in members
            ),
            "signature_cases": sum(
                int(record["signature_cases"]) for record in members
            ),
            "signature_mismatches": sum(
                int(record["signature_mismatches"])
                for record in members
            ),
            "affected_kernel_observations": sum(
                int(record["affected_kernels"]) for record in members
            ),
            "nontermination_guards": sum(
                int(record["nontermination_guards"])
                for record in members
            ),
        }

    return {
        "mutants": len(records),
        "family_counts": dict(
            Counter(
                record["mutation"]["family"] for record in records
            )
        ),
        "independent_oracle_detected": sum(
            bool(record["independent_oracle_detected"])
            for record in records
        ),
        "oracle_comparison_detected": sum(
            bool(record["independent_oracle_detected"])
            for record in records
        ),
        "trace_detected": sum(
            bool(record["trace_detected"]) for record in records
        ),
        "signature_detected": sum(
            bool(record["signature_detected"]) for record in records
        ),
        "by_family": by_family,
    }


def _record_path(output: Path, mutation_id: str) -> Path:
    return output / "records" / f"{mutation_id}.json"


def _load_completed_records(
    *,
    output: Path,
    catalog: tuple[dict[str, Any], ...],
    experiment_fingerprint: str,
    resume: bool,
) -> dict[str, dict[str, Any]]:
    completed: dict[str, dict[str, Any]] = {}
    for mutation in catalog:
        path = _record_path(output, mutation["mutation_id"])
        if not path.exists():
            continue
        if not resume:
            raise FileExistsError(
                f"record exists while --no-resume is active: {path}"
            )
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("experiment_fingerprint") != experiment_fingerprint:
            raise ValueError(f"stale record fingerprint: {path}")
        if record.get("mutation") != mutation:
            raise ValueError(f"stale mutation record: {path}")
        completed[mutation["mutation_id"]] = record
    return completed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a fixed-seed sampled mutation-sensitivity audit against "
            "the independent braking-domain trace oracle."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT
            / "results"
            / f"randomized_mutation_audit_seed{DEFAULT_SEED}"
        ),
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--per-family",
        type=int,
        default=DEFAULT_PER_FAMILY,
    )
    parser.add_argument(
        "--sample-kernels",
        type=int,
        default=DEFAULT_SAMPLE_KERNELS,
    )
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.per_family < DEFAULT_PER_FAMILY:
        raise ValueError(
            "the reported audit requires at least 20 mutants per family"
        )
    if not 1 <= args.workers <= MAX_WORKERS:
        raise ValueError(
            f"workers must be in [1, {MAX_WORKERS}], got {args.workers}"
        )
    config_path = args.config.resolve()
    output = args.output.resolve()
    config = oracle.load_config(config_path)
    catalog = generate_mutation_catalog(
        seed=args.seed,
        per_family=args.per_family,
    )
    presentations = oracle.oracle_make_presentations(config)
    sampled_kernels, valid_population = (
        select_stratified_valid_kernels(
            oracle.oracle_make_kernels(config),
            args.sample_kernels,
        )
    )

    code_fingerprint, file_hashes = relevant_code_fingerprint(
        Path(__file__).resolve()
    )
    sample_payload = [
        asdict(kernel) for kernel in sampled_kernels
    ]
    catalog_digest = digest_value(catalog)
    sample_digest = digest_value(sample_payload)
    experiment_fingerprint = digest_value(
        {
            "schema_version": 1,
            "code_fingerprint": code_fingerprint,
            "config_sha256": sha256_file(config_path),
            "seed": args.seed,
            "per_family": args.per_family,
            "catalog_digest": catalog_digest,
            "sample_kernels": args.sample_kernels,
            "sample_digest": sample_digest,
            "presentations": [
                asdict(item) for item in presentations
            ],
            "agents": list(oracle.ORACLE_AGENT_NAMES),
            "key_orders": list(KEY_ORDERS),
        }
    )
    manifest = {
        "schema_version": 1,
        "experiment": "randomized_mutation_sensitivity",
        "claim_scope": (
            "fixed-seed sampled sensitivity audit in the frozen braking "
            "domain; not exhaustive verification and not a second domain"
        ),
        "detection_semantics": {
            "independent_oracle_detected": (
                "at least one exact trace or downstream signature differs "
                "from the independent frozen oracle"
            ),
            "oracle_comparison_detected": (
                "at least one exact trace or downstream signature differs "
                "from the independent frozen oracle"
            ),
            "trace_detected": (
                "at least one complete observation/action/status trace "
                "differs exactly from the oracle"
            ),
            "signature_detected": (
                "at least one six-bit pairwise behavioral signature differs; "
                "this is a lower-resolution downstream check"
            ),
        },
        "experiment_fingerprint": experiment_fingerprint,
        "created_at": now_local(),
        "seed": args.seed,
        "per_family": args.per_family,
        "mutation_count": len(catalog),
        "families": list(FAMILIES),
        "catalog_digest": catalog_digest,
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "valid_population": valid_population,
        "sample_kernels": len(sampled_kernels),
        "sample_strategy": (
            "valid mechanisms stratified by friction, brake force, and "
            "horizon; evenly spaced within each stratum"
        ),
        "sample_digest": sample_digest,
        "presentations": len(presentations),
        "agents": list(oracle.ORACLE_AGENT_NAMES),
        "key_replay_orders": list(KEY_ORDERS),
        "workers": args.workers,
        "code_fingerprint": code_fingerprint,
        "file_sha256": file_hashes,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "hostname": socket.gethostname(),
            "logical_cpus": os.cpu_count(),
        },
        "command": [str(item) for item in sys.argv],
    }

    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("experiment_fingerprint") != experiment_fingerprint:
            raise ValueError(
                "output manifest belongs to a different audit configuration"
            )
        if not args.resume:
            raise FileExistsError(
                f"manifest exists while --no-resume is active: {manifest_path}"
            )
    else:
        atomic_write_json(manifest_path, manifest)

    catalog_payload = {
        "schema_version": 1,
        "seed": args.seed,
        "catalog_digest": catalog_digest,
        "mutations": list(catalog),
    }
    sample_file_payload = {
        "schema_version": 1,
        "sample_digest": sample_digest,
        "strategy": manifest["sample_strategy"],
        "valid_population": valid_population,
        "kernels": sample_payload,
    }
    for sidecar_path, sidecar_payload in (
        (output / "mutation_catalog.json", catalog_payload),
        (output / "sampled_kernels.json", sample_file_payload),
    ):
        if sidecar_path.exists():
            existing_sidecar = json.loads(
                sidecar_path.read_text(encoding="utf-8")
            )
            if existing_sidecar != sidecar_payload:
                raise ValueError(f"stale frozen sidecar: {sidecar_path}")
        else:
            atomic_write_json(sidecar_path, sidecar_payload)

    completed = _load_completed_records(
        output=output,
        catalog=catalog,
        experiment_fingerprint=experiment_fingerprint,
        resume=args.resume,
    )
    pending = [
        mutation
        for mutation in catalog
        if mutation["mutation_id"] not in completed
    ]
    payloads = [
        (
            mutation,
            sampled_kernels,
            presentations,
            experiment_fingerprint,
        )
        for mutation in pending
    ]

    if args.workers == 1:
        for payload in payloads:
            record = evaluate_mutation(payload)
            mutation_id = record["mutation"]["mutation_id"]
            atomic_write_json(
                _record_path(output, mutation_id),
                record,
            )
            completed[mutation_id] = record
    elif payloads:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(evaluate_mutation, payload): payload[0][
                    "mutation_id"
                ]
                for payload in payloads
            }
            for future in as_completed(futures):
                mutation_id = futures[future]
                record = future.result()
                if record["mutation"]["mutation_id"] != mutation_id:
                    raise RuntimeError("worker returned the wrong mutation")
                atomic_write_json(
                    _record_path(output, mutation_id),
                    record,
                )
                completed[mutation_id] = record

    records = tuple(
        completed[mutation["mutation_id"]]
        for mutation in catalog
    )
    summary = _summarize(records)
    aggregate = {
        "schema_version": 1,
        "experiment_fingerprint": experiment_fingerprint,
        "claim_scope": manifest["claim_scope"],
        "detection_semantics": manifest["detection_semantics"],
        "summary": summary,
        "records": records,
        "completed_at": now_local(),
    }
    atomic_write_json(
        output / "randomized_mutation_results.json",
        aggregate,
    )
    _write_csv(
        output / "randomized_mutation_results.csv",
        records,
    )
    print(stable_json(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
