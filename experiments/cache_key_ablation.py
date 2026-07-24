"""Component-wise ablation of LayerProbe's semantic-step cache key.

This experiment is deliberately separate from the production evaluator.  It
does not modify or monkeypatch ``src``.  Instead, it enumerates deterministic
oracle contexts using the public mechanics primitives and asks whether each
projected cache key remains a function:

    full                  = (world state, agent memory, observation)
    drop_state            = (agent memory, observation)
    drop_memory           = (world state, observation)
    drop_observation      = (world state, agent memory)

For every valid mechanism and declared agent, the collision census groups
oracle contexts by a projected key.  A key class is unsafe when it contains
more than one true deterministic output
``(action, next state, next memory, terminal status)``.  This census is
independent of presentation replay order.  A second phase performs actual
fault-injected cache replay in canonical and reverse presentation order and
compares traces and candidate signatures with the oracle.

One process-pool task analyzes one kernel.  Consecutive task chunks are written
atomically, so an interrupted run can resume without altering completed chunks.
The experiment supports at most 16 workers; the default profile uses the eight
physical cores of the paper workstation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from layerprobe.evaluator import signature_for
from layerprobe.mechanics import (
    AGENT_NAMES,
    advance_belief,
    choose_action,
    ingest,
    initial_agent_memory,
    initial_state,
    observe,
    terminal_status,
    transition,
    verify_kernel,
)
from layerprobe.model import (
    AgentMemory,
    DisplayMemory,
    KernelSpec,
    Observation,
    PresentationSpec,
    Trace,
    WorldState,
)
from layerprobe.workloads import make_kernels, make_presentations

try:
    import psutil
except ImportError:  # pragma: no cover - optional provenance enhancement
    psutil = None


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path(__file__).with_name("cache_key_ablation_profile_8c32g.json")
EXPECTED_VARIANTS = (
    "full",
    "drop_state",
    "drop_memory",
    "drop_observation",
)
EXPECTED_ORDERS = ("canonical", "reverse")
OUTPUT_FIELDS = ("action", "next_world_state", "next_agent_memory", "status")

_WORKER_PRESENTATIONS: tuple[PresentationSpec, ...] = ()
_WORKER_VARIANTS: tuple[str, ...] = ()
_WORKER_ORDERS: tuple[str, ...] = ()
_WORKER_MINIMUM_GUARD = 64
_WORKER_GUARD_MULTIPLIER = 8


@dataclass(frozen=True, slots=True)
class KernelTask:
    selection_position: int
    kernel_index: int
    kernel: KernelSpec


@dataclass(frozen=True, slots=True)
class SemanticContext:
    presentation_index: int
    step: int
    state: WorldState
    agent_memory: AgentMemory
    observation: Observation
    perceived_memory: AgentMemory
    action: str
    next_state: WorldState
    next_memory: AgentMemory
    status: str

    @property
    def output(self) -> tuple[str, WorldState, AgentMemory, str]:
        return self.action, self.next_state, self.next_memory, self.status

    @property
    def rank(self) -> tuple[int, int]:
        return self.presentation_index, self.step


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


def sha256_source_tree(root: Path) -> str:
    digest = hashlib.sha256()
    paths = sorted(path for path in root.rglob("*.py") if path.is_file())
    for path in paths:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        data = path.read_bytes()
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
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


def state_payload(state: WorldState) -> dict[str, Any]:
    return asdict(state)


def memory_payload(memory: AgentMemory) -> dict[str, Any]:
    return asdict(memory)


def observation_payload(observation: Observation) -> list[int]:
    return list(observation)


def trace_step_payload(step: tuple[Observation, str, str] | None) -> Any:
    if step is None:
        return None
    observation, action, status = step
    return {
        "observation": observation_payload(observation),
        "action": action,
        "status": status,
    }


def display_memory_payload(memory: DisplayMemory) -> dict[str, Any]:
    return {
        "previous": (
            None if memory.previous is None else observation_payload(memory.previous)
        )
    }


def output_payload(
    output: tuple[str, WorldState, AgentMemory, str],
) -> dict[str, Any]:
    action, next_state, next_memory, status = output
    return {
        "action": action,
        "next_world_state": state_payload(next_state),
        "next_agent_memory": memory_payload(next_memory),
        "status": status,
    }


def context_payload(
    context: SemanticContext,
    *,
    kernel_index: int,
    kernel: KernelSpec,
    agent: str,
    presentations: tuple[PresentationSpec, ...],
) -> dict[str, Any]:
    presentation = presentations[context.presentation_index]
    return {
        "kernel_index": kernel_index,
        "kernel": asdict(kernel),
        "agent": agent,
        "presentation_index": context.presentation_index,
        "presentation": asdict(presentation),
        "step": context.step,
        "world_state": state_payload(context.state),
        "agent_memory": memory_payload(context.agent_memory),
        "observation": observation_payload(context.observation),
        "perceived_memory": memory_payload(context.perceived_memory),
        "output": output_payload(context.output),
    }


def projected_key(
    variant: str,
    state: WorldState,
    memory: AgentMemory,
    observation: Observation,
) -> tuple[Any, ...]:
    if variant == "full":
        return state, memory, observation
    if variant == "drop_state":
        return memory, observation
    if variant == "drop_memory":
        return state, observation
    if variant == "drop_observation":
        return state, memory
    raise ValueError(f"unsupported cache-key variant: {variant}")


def projected_key_payload(
    variant: str,
    state: WorldState,
    memory: AgentMemory,
    observation: Observation,
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if variant != "drop_state":
        payload["world_state"] = state_payload(state)
    if variant != "drop_memory":
        payload["agent_memory"] = memory_payload(memory)
    if variant != "drop_observation":
        payload["observation"] = observation_payload(observation)
    return payload


def differing_output_fields(
    left: tuple[str, WorldState, AgentMemory, str],
    right: tuple[str, WorldState, AgentMemory, str],
) -> list[str]:
    left_payload = output_payload(left)
    right_payload = output_payload(right)
    return [
        field for field in OUTPUT_FIELDS if left_payload[field] != right_payload[field]
    ]


def first_trace_difference(left: Trace, right: Trace) -> int | None:
    common = min(len(left), len(right))
    for index in range(common):
        if left[index] != right[index]:
            return index
    if len(left) != len(right):
        return common
    return None


def empty_collision_metrics() -> dict[str, Any]:
    return {
        "semantic_contexts": 0,
        "unique_projected_keys": 0,
        "unsafe_key_classes": 0,
        "contexts_in_unsafe_classes": 0,
        "distinct_output_variants_in_unsafe_classes": 0,
        "conflicting_context_pairs": 0,
        "affected_kernel_agent_scopes": 0,
        "affected_kernel": False,
    }


def empty_replay_metrics() -> dict[str, Any]:
    return {
        "traces": 0,
        "trace_mismatches": 0,
        "candidates": 0,
        "candidate_signature_mismatches": 0,
        "signature_bit_flips": 0,
        "cache_hits": 0,
        "cache_misses": 0,
        "unsafe_cache_hits": 0,
        "nontermination_guards": 0,
        "trace_digest": "",
        "candidate_signature_digest": "",
        "oracle_trace_digest": "",
        "oracle_candidate_signature_digest": "",
    }


def collision_witness(
    *,
    variant: str,
    kernel_index: int,
    kernel: KernelSpec,
    agent: str,
    key: tuple[Any, ...],
    first: SemanticContext,
    second: SemanticContext,
    presentations: tuple[PresentationSpec, ...],
) -> dict[str, Any]:
    key_payload = projected_key_payload(
        variant,
        first.state,
        first.agent_memory,
        first.observation,
    )
    agent_index = AGENT_NAMES.index(agent)
    return {
        "rank": [
            kernel_index,
            agent_index,
            first.presentation_index,
            first.step,
            second.presentation_index,
            second.step,
        ],
        "variant": variant,
        "omitted_component": {
            "drop_state": "world_state",
            "drop_memory": "agent_memory",
            "drop_observation": "observation",
        }.get(variant),
        "scope": {"kernel": kernel.name, "agent": agent},
        "projected_key": key_payload,
        "projected_key_sha256": digest_value(key_payload),
        "first_context": context_payload(
            first,
            kernel_index=kernel_index,
            kernel=kernel,
            agent=agent,
            presentations=presentations,
        ),
        "second_context": context_payload(
            second,
            kernel_index=kernel_index,
            kernel=kernel,
            agent=agent,
            presentations=presentations,
        ),
        "differing_output_fields": differing_output_fields(
            first.output, second.output
        ),
        "key_repr_for_debugging": repr(key),
    }


def enumerate_oracle_and_collisions(
    *,
    kernel_index: int,
    kernel: KernelSpec,
    presentations: tuple[PresentationSpec, ...],
    variants: tuple[str, ...],
) -> tuple[
    dict[str, dict[int, Trace]],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    int,
]:
    """Enumerate true contexts and census projected-key collision classes."""

    oracle: dict[str, dict[int, Trace]] = {
        agent: {} for agent in AGENT_NAMES
    }
    collision = {variant: empty_collision_metrics() for variant in variants}
    first_witnesses: dict[str, dict[str, Any]] = {}
    total_contexts = 0

    for agent in AGENT_NAMES:
        groups: dict[
            str,
            dict[
                tuple[Any, ...],
                dict[
                    tuple[str, WorldState, AgentMemory, str],
                    list[Any],
                ],
            ],
        ] = {variant: {} for variant in variants}

        for presentation_index, presentation in enumerate(presentations):
            state = initial_state(kernel)
            agent_memory = initial_agent_memory(kernel)
            display_memory = DisplayMemory()
            trace: list[tuple[Observation, str, str]] = []
            step = 0

            while terminal_status(state, kernel) == "running":
                if step > kernel.horizon:
                    raise RuntimeError(
                        f"oracle exceeded horizon for {kernel.name}/{agent}"
                    )
                observation, display_memory = observe(
                    state,
                    kernel,
                    presentation,
                    display_memory,
                )
                perceived = ingest(agent_memory, observation)
                action = choose_action(agent, perceived, kernel)
                next_state = transition(state, action, kernel)
                next_memory = advance_belief(
                    agent, perceived, action, kernel
                )
                status = terminal_status(next_state, kernel)
                context = SemanticContext(
                    presentation_index=presentation_index,
                    step=step,
                    state=state,
                    agent_memory=agent_memory,
                    observation=observation,
                    perceived_memory=perceived,
                    action=action,
                    next_state=next_state,
                    next_memory=next_memory,
                    status=status,
                )
                trace.append((observation, action, status))
                total_contexts += 1

                for variant in variants:
                    key = projected_key(
                        variant, state, agent_memory, observation
                    )
                    output_groups = groups[variant].setdefault(key, {})
                    existing = output_groups.get(context.output)
                    if existing is None:
                        output_groups[context.output] = [1, context]
                    else:
                        existing[0] += 1

                state = next_state
                agent_memory = next_memory
                step += 1

            oracle[agent][presentation_index] = tuple(trace)

        for variant in variants:
            metrics = collision[variant]
            metrics["semantic_contexts"] += sum(
                sum(int(item[0]) for item in output_groups.values())
                for output_groups in groups[variant].values()
            )
            metrics["unique_projected_keys"] += len(groups[variant])
            scope_unsafe = False

            for key, output_groups in groups[variant].items():
                if len(output_groups) <= 1:
                    continue

                scope_unsafe = True
                metrics["unsafe_key_classes"] += 1
                counts = [int(item[0]) for item in output_groups.values()]
                total = sum(counts)
                metrics["contexts_in_unsafe_classes"] += total
                metrics[
                    "distinct_output_variants_in_unsafe_classes"
                ] += len(output_groups)
                all_pairs = total * (total - 1) // 2
                same_output_pairs = sum(
                    count * (count - 1) // 2 for count in counts
                )
                metrics["conflicting_context_pairs"] += (
                    all_pairs - same_output_pairs
                )

                representatives = sorted(
                    (item[1] for item in output_groups.values()),
                    key=lambda context: context.rank,
                )
                first = representatives[0]
                second = representatives[1]
                witness = collision_witness(
                    variant=variant,
                    kernel_index=kernel_index,
                    kernel=kernel,
                    agent=agent,
                    key=key,
                    first=first,
                    second=second,
                    presentations=presentations,
                )
                incumbent = first_witnesses.get(variant)
                if incumbent is None or witness["rank"] < incumbent["rank"]:
                    first_witnesses[variant] = witness

            if scope_unsafe:
                metrics["affected_kernel_agent_scopes"] += 1
                metrics["affected_kernel"] = True

    return oracle, collision, first_witnesses, total_contexts


def compute_true_step(
    *,
    kernel: KernelSpec,
    agent: str,
    state: WorldState,
    agent_memory: AgentMemory,
    observation: Observation,
) -> tuple[str, WorldState, AgentMemory, str]:
    perceived = ingest(agent_memory, observation)
    action = choose_action(agent, perceived, kernel)
    next_state = transition(state, action, kernel)
    next_memory = advance_belief(agent, perceived, action, kernel)
    status = terminal_status(next_state, kernel)
    return action, next_state, next_memory, status


def replay_variant(
    *,
    variant: str,
    order: str,
    kernel_index: int,
    kernel: KernelSpec,
    presentations: tuple[PresentationSpec, ...],
    oracle: dict[str, dict[int, Trace]],
    step_guard: int,
) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any] | None]:
    """Run one fault-injected variant and compare against oracle traces."""

    if order == "canonical":
        order_indices = tuple(range(len(presentations)))
    elif order == "reverse":
        order_indices = tuple(reversed(range(len(presentations))))
    else:
        raise ValueError(f"unsupported presentation order: {order}")

    metrics = empty_replay_metrics()
    traces: dict[int, dict[str, Trace]] = {
        index: {} for index in range(len(presentations))
    }
    first_trace_witness: dict[str, Any] | None = None
    first_nontermination: dict[str, Any] | None = None

    for agent_index, agent in enumerate(AGENT_NAMES):
        cache: dict[
            tuple[Any, ...],
            tuple[str, WorldState, AgentMemory, str],
        ] = {}

        for order_position, presentation_index in enumerate(order_indices):
            presentation = presentations[presentation_index]
            state = initial_state(kernel)
            agent_memory = initial_agent_memory(kernel)
            display_memory = DisplayMemory()
            trace: list[tuple[Observation, str, str]] = []
            step = 0

            while terminal_status(state, kernel) == "running":
                if step >= step_guard:
                    metrics["nontermination_guards"] += 1
                    witness = {
                        "rank": [
                            kernel_index,
                            agent_index,
                            order_position,
                            presentation_index,
                            step,
                        ],
                        "variant": variant,
                        "order": order,
                        "kernel_index": kernel_index,
                        "kernel": asdict(kernel),
                        "agent": agent,
                        "presentation_index": presentation_index,
                        "presentation": asdict(presentation),
                        "guard": step_guard,
                        "step": step,
                        "world_state": state_payload(state),
                        "agent_memory": memory_payload(agent_memory),
                        "display_memory": display_memory_payload(display_memory),
                    }
                    if (
                        first_nontermination is None
                        or witness["rank"] < first_nontermination["rank"]
                    ):
                        first_nontermination = witness
                    break

                observation, display_memory = observe(
                    state,
                    kernel,
                    presentation,
                    display_memory,
                )
                key = projected_key(
                    variant, state, agent_memory, observation
                )
                true_output = compute_true_step(
                    kernel=kernel,
                    agent=agent,
                    state=state,
                    agent_memory=agent_memory,
                    observation=observation,
                )
                cached = cache.get(key)
                if cached is None:
                    metrics["cache_misses"] += 1
                    cached = true_output
                    cache[key] = cached
                else:
                    metrics["cache_hits"] += 1
                    if cached != true_output:
                        metrics["unsafe_cache_hits"] += 1

                action, state, agent_memory, status = cached
                trace.append((observation, action, status))
                step += 1

            weak_trace = tuple(trace)
            traces[presentation_index][agent] = weak_trace
            metrics["traces"] += 1
            reference_trace = oracle[agent][presentation_index]
            difference = first_trace_difference(weak_trace, reference_trace)
            if difference is not None:
                metrics["trace_mismatches"] += 1
                witness = {
                    "rank": [
                        kernel_index,
                        agent_index,
                        order_position,
                        presentation_index,
                        difference,
                    ],
                    "variant": variant,
                    "order": order,
                    "kernel_index": kernel_index,
                    "kernel": asdict(kernel),
                    "agent": agent,
                    "presentation_index": presentation_index,
                    "presentation": asdict(presentation),
                    "first_difference_step": difference,
                    "weak_trace_length": len(weak_trace),
                    "oracle_trace_length": len(reference_trace),
                    "weak_step": trace_step_payload(
                        weak_trace[difference]
                        if difference < len(weak_trace)
                        else None
                    ),
                    "oracle_step": trace_step_payload(
                        reference_trace[difference]
                        if difference < len(reference_trace)
                        else None
                    ),
                }
                if (
                    first_trace_witness is None
                    or witness["rank"] < first_trace_witness["rank"]
                ):
                    first_trace_witness = witness

    weak_trace_payload: list[Any] = []
    oracle_trace_payload: list[Any] = []
    weak_signatures: list[list[Any]] = []
    oracle_signatures: list[list[Any]] = []

    for presentation_index, presentation in enumerate(presentations):
        weak_by_agent = traces[presentation_index]
        oracle_by_agent = {
            agent: oracle[agent][presentation_index]
            for agent in AGENT_NAMES
        }
        weak_signature = signature_for(weak_by_agent)
        oracle_signature = signature_for(oracle_by_agent)
        candidate = f"{kernel.name}::{presentation.name}"
        weak_signatures.append([candidate, weak_signature])
        oracle_signatures.append([candidate, oracle_signature])
        metrics["candidates"] += 1
        if weak_signature != oracle_signature:
            metrics["candidate_signature_mismatches"] += 1
            metrics["signature_bit_flips"] += (
                weak_signature ^ oracle_signature
            ).bit_count()

        for agent in AGENT_NAMES:
            weak_trace_payload.append(
                [agent, presentation_index, traces[presentation_index][agent]]
            )
            oracle_trace_payload.append(
                [agent, presentation_index, oracle[agent][presentation_index]]
            )

    metrics["trace_digest"] = digest_value(weak_trace_payload)
    metrics["candidate_signature_digest"] = digest_value(weak_signatures)
    metrics["oracle_trace_digest"] = digest_value(oracle_trace_payload)
    metrics["oracle_candidate_signature_digest"] = digest_value(
        oracle_signatures
    )
    return metrics, first_trace_witness, first_nontermination


def analyze_kernel(
    *,
    selection_position: int,
    kernel_index: int,
    kernel: KernelSpec,
    presentations: tuple[PresentationSpec, ...],
    variants: tuple[str, ...],
    orders: tuple[str, ...],
    minimum_step_guard: int,
    horizon_guard_multiplier: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    verification = verify_kernel(kernel)
    base = {
        "job_id": f"kernel-{kernel_index:05d}",
        "selection_position": selection_position,
        "kernel_index": kernel_index,
        "kernel_name": kernel.name,
        "valid": verification.valid,
        "verification": asdict(verification),
    }
    if not verification.valid:
        return {
            **base,
            "oracle_contexts": 0,
            "collision": {},
            "replay": {},
            "elapsed_s": time.perf_counter() - started,
            "first_collision_witnesses": {},
            "first_trace_mismatch_witnesses": {},
            "first_nontermination_witnesses": {},
        }

    oracle, collision, collision_witnesses, oracle_contexts = (
        enumerate_oracle_and_collisions(
            kernel_index=kernel_index,
            kernel=kernel,
            presentations=presentations,
            variants=variants,
        )
    )
    replay: dict[str, dict[str, dict[str, Any]]] = {}
    trace_witnesses: dict[str, dict[str, dict[str, Any]]] = {}
    nontermination_witnesses: dict[str, dict[str, dict[str, Any]]] = {}
    step_guard = max(
        minimum_step_guard,
        kernel.horizon * horizon_guard_multiplier,
    )

    for variant in variants:
        replay[variant] = {}
        for order in orders:
            (
                replay_metrics,
                trace_witness,
                nontermination_witness,
            ) = replay_variant(
                variant=variant,
                order=order,
                kernel_index=kernel_index,
                kernel=kernel,
                presentations=presentations,
                oracle=oracle,
                step_guard=step_guard,
            )
            replay[variant][order] = replay_metrics
            if trace_witness is not None:
                trace_witnesses.setdefault(variant, {})[
                    order
                ] = trace_witness
            if nontermination_witness is not None:
                nontermination_witnesses.setdefault(variant, {})[
                    order
                ] = nontermination_witness

    return {
        **base,
        "oracle_contexts": oracle_contexts,
        "collision": collision,
        "replay": replay,
        "elapsed_s": time.perf_counter() - started,
        "first_collision_witnesses": collision_witnesses,
        "first_trace_mismatch_witnesses": trace_witnesses,
        "first_nontermination_witnesses": nontermination_witnesses,
    }


def initialize_worker(
    presentations: tuple[PresentationSpec, ...],
    variants: tuple[str, ...],
    orders: tuple[str, ...],
    minimum_step_guard: int,
    horizon_guard_multiplier: int,
) -> None:
    global _WORKER_PRESENTATIONS
    global _WORKER_VARIANTS
    global _WORKER_ORDERS
    global _WORKER_MINIMUM_GUARD
    global _WORKER_GUARD_MULTIPLIER
    _WORKER_PRESENTATIONS = presentations
    _WORKER_VARIANTS = variants
    _WORKER_ORDERS = orders
    _WORKER_MINIMUM_GUARD = minimum_step_guard
    _WORKER_GUARD_MULTIPLIER = horizon_guard_multiplier


def analyze_kernel_task(task: KernelTask) -> dict[str, Any]:
    return analyze_kernel(
        selection_position=task.selection_position,
        kernel_index=task.kernel_index,
        kernel=task.kernel,
        presentations=_WORKER_PRESENTATIONS,
        variants=_WORKER_VARIANTS,
        orders=_WORKER_ORDERS,
        minimum_step_guard=_WORKER_MINIMUM_GUARD,
        horizon_guard_multiplier=_WORKER_GUARD_MULTIPLIER,
    )


def choose_earlier(
    incumbent: dict[str, Any] | None,
    candidate: dict[str, Any],
) -> dict[str, Any]:
    if incumbent is None or candidate["rank"] < incumbent["rank"]:
        return candidate
    return incumbent


def compact_chunk_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    collision_witnesses: dict[str, dict[str, Any]] = {}
    trace_witnesses: dict[str, dict[str, dict[str, Any]]] = {}
    nontermination_witnesses: dict[str, dict[str, dict[str, Any]]] = {}
    compact_jobs: list[dict[str, Any]] = []

    for result in results:
        for variant, witness in result.pop(
            "first_collision_witnesses"
        ).items():
            collision_witnesses[variant] = choose_earlier(
                collision_witnesses.get(variant), witness
            )
        for variant, order_map in result.pop(
            "first_trace_mismatch_witnesses"
        ).items():
            for order, witness in order_map.items():
                incumbent = trace_witnesses.setdefault(variant, {}).get(order)
                trace_witnesses[variant][order] = choose_earlier(
                    incumbent, witness
                )
        for variant, order_map in result.pop(
            "first_nontermination_witnesses"
        ).items():
            for order, witness in order_map.items():
                incumbent = nontermination_witnesses.setdefault(
                    variant, {}
                ).get(order)
                nontermination_witnesses[variant][order] = choose_earlier(
                    incumbent, witness
                )
        compact_jobs.append(result)

    return {
        "jobs": compact_jobs,
        "first_collision_witnesses": collision_witnesses,
        "first_trace_mismatch_witnesses": trace_witnesses,
        "first_nontermination_witnesses": nontermination_witnesses,
    }


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("unsupported cache-key ablation config schema")
    variants = tuple(config.get("variants", ()))
    orders = tuple(config.get("orders", ()))
    if variants != EXPECTED_VARIANTS:
        raise ValueError(
            f"variants must be exactly {EXPECTED_VARIANTS}, got {variants}"
        )
    if orders != EXPECTED_ORDERS:
        raise ValueError(
            f"orders must be exactly {EXPECTED_ORDERS}, got {orders}"
        )
    maximum_workers = int(config["maximum_workers"])
    if not 1 <= int(config["default_workers"]) <= maximum_workers <= 16:
        raise ValueError("worker limits must satisfy 1 <= default <= max <= 16")
    grid_size = int(config["frozen_grid_size"])
    if grid_size != 24_624:
        raise ValueError("frozen_grid_size must remain 24624")
    for field in ("paper_kernel_count", "smoke_kernel_count"):
        value = int(config[field])
        if not 1 <= value <= grid_size:
            raise ValueError(f"{field} must be between 1 and {grid_size}")
    if int(config["chunk_size"]) < 1:
        raise ValueError("chunk_size must be positive")
    if int(config["minimum_step_guard"]) < 1:
        raise ValueError("minimum_step_guard must be positive")
    if int(config["horizon_guard_multiplier"]) < 1:
        raise ValueError("horizon_guard_multiplier must be positive")
    return config


def selection_indices(grid_size: int, count: int) -> tuple[int, ...]:
    """Return the full grid or a deterministic first-plus-midpoint subset."""

    if not 1 <= count <= grid_size:
        raise ValueError("selected kernel count is outside the frozen grid")
    if count == grid_size:
        return tuple(range(grid_size))
    if count == 1:
        return (0,)
    selected = {0}
    remaining = count - 1
    for position in range(remaining):
        index = 1 + math.floor((position + 0.5) * (grid_size - 1) / remaining)
        selected.add(min(index, grid_size - 1))
    if len(selected) != count:
        raise RuntimeError("kernel selection unexpectedly contained duplicates")
    return tuple(sorted(selected))


def chunked(items: tuple[KernelTask, ...], size: int) -> list[tuple[KernelTask, ...]]:
    return [
        items[start : start + size]
        for start in range(0, len(items), size)
    ]


def chunk_name(chunk_index: int, tasks: tuple[KernelTask, ...]) -> str:
    return (
        f"chunk_{chunk_index:05d}_"
        f"{tasks[0].selection_position:05d}_"
        f"{tasks[-1].selection_position:05d}.json"
    )


def physical_cpu_count() -> int:
    if psutil is not None:
        detected = psutil.cpu_count(logical=False)
        if detected:
            return int(detected)
    logical = os.cpu_count() or 1
    return max(1, logical // 2)


def machine_payload() -> dict[str, Any]:
    memory_gib = None
    if psutil is not None:
        memory_gib = psutil.virtual_memory().total / (1024**3)
    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "physical_cpus": physical_cpu_count(),
        "logical_cpus": os.cpu_count(),
        "memory_gib": memory_gib,
    }


def build_fingerprints(
    *,
    config_path: Path,
    mode: str,
    workers: int,
    chunk_size: int,
    selected_indices: tuple[int, ...],
    config: dict[str, Any],
) -> dict[str, str]:
    script_hash = sha256_file(Path(__file__).resolve())
    config_hash = sha256_file(config_path)
    core_hash = sha256_source_tree(ROOT / "src")
    experiment_hash = digest_value(
        {
            "script_sha256": script_hash,
            "config_sha256": config_hash,
            "core_source_sha256": core_hash,
        }
    )
    run_hash = digest_value(
        {
            "experiment_fingerprint": experiment_hash,
            "mode": mode,
            "workers": workers,
            "chunk_size": chunk_size,
            "selected_indices": selected_indices,
            "variants": config["variants"],
            "orders": config["orders"],
            "minimum_step_guard": config["minimum_step_guard"],
            "horizon_guard_multiplier": config[
                "horizon_guard_multiplier"
            ],
        }
    )
    return {
        "script_sha256": script_hash,
        "config_sha256": config_hash,
        "core_source_sha256": core_hash,
        "experiment_fingerprint": experiment_hash,
        "run_fingerprint": run_hash,
    }


def prepare_output_directory(
    output: Path,
    *,
    resume: bool,
) -> None:
    if output.exists():
        entries = list(output.iterdir())
        if entries and not resume:
            raise FileExistsError(
                f"output directory is non-empty; pass --resume or choose a new path: {output}"
            )
    output.mkdir(parents=True, exist_ok=True)
    (output / "chunks").mkdir(parents=True, exist_ok=True)


def validate_or_write_plan(
    *,
    output: Path,
    resume: bool,
    plan: dict[str, Any],
) -> None:
    path = output / "plan.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if not resume:
            raise FileExistsError(f"plan already exists: {path}")
        if existing != plan:
            raise ValueError(
                "resume plan differs from saved plan; use the original code/config/options"
            )
    else:
        if resume and any((output / "chunks").glob("chunk_*.json")):
            raise ValueError("cannot resume chunks without a matching plan.json")
        atomic_write_json(path, plan)


def validate_chunk(
    *,
    payload: dict[str, Any],
    expected_tasks: tuple[KernelTask, ...],
    run_fingerprint: str,
) -> None:
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported saved chunk schema")
    if payload.get("run_fingerprint") != run_fingerprint:
        raise ValueError("saved chunk fingerprint mismatch")
    expected_job_ids = [
        f"kernel-{task.kernel_index:05d}" for task in expected_tasks
    ]
    actual_job_ids = [job["job_id"] for job in payload.get("jobs", [])]
    if actual_job_ids != expected_job_ids:
        raise ValueError("saved chunk job list does not match the run plan")


def aggregate_digest(
    jobs: Iterable[dict[str, Any]],
    *,
    variant: str,
    order: str,
    field: str,
) -> str:
    payload = [
        [job["job_id"], job["replay"][variant][order][field]]
        for job in jobs
        if job["valid"]
    ]
    return digest_value(payload)


def aggregate_chunks(
    *,
    chunk_payloads: list[dict[str, Any]],
    variants: tuple[str, ...],
    orders: tuple[str, ...],
    plan: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    jobs = [
        job
        for chunk_payload in chunk_payloads
        for job in chunk_payload["jobs"]
    ]
    jobs.sort(key=lambda item: item["selection_position"])
    if len(jobs) != plan["selected_kernel_count"]:
        raise ValueError("cannot summarize an incomplete run")
    if len({job["job_id"] for job in jobs}) != len(jobs):
        raise ValueError("duplicate kernel jobs detected")

    collision_witnesses: dict[str, dict[str, Any]] = {}
    trace_witnesses: dict[str, dict[str, dict[str, Any]]] = {}
    nontermination_witnesses: dict[str, dict[str, dict[str, Any]]] = {}
    for chunk_payload in chunk_payloads:
        for variant, witness in chunk_payload[
            "first_collision_witnesses"
        ].items():
            collision_witnesses[variant] = choose_earlier(
                collision_witnesses.get(variant), witness
            )
        for variant, order_map in chunk_payload[
            "first_trace_mismatch_witnesses"
        ].items():
            for order, witness in order_map.items():
                incumbent = trace_witnesses.setdefault(variant, {}).get(order)
                trace_witnesses[variant][order] = choose_earlier(
                    incumbent, witness
                )
        for variant, order_map in chunk_payload[
            "first_nontermination_witnesses"
        ].items():
            for order, witness in order_map.items():
                incumbent = nontermination_witnesses.setdefault(
                    variant, {}
                ).get(order)
                nontermination_witnesses[variant][order] = choose_earlier(
                    incumbent, witness
                )

    valid_jobs = [job for job in jobs if job["valid"]]
    if plan["mode"] == "smoke":
        result_status = "smoke_runner_validation_not_formal_evidence"
    elif plan["paper_evidence_eligible"]:
        result_status = "complete"
    else:
        result_status = "development_override_not_formal_evidence"

    summary: dict[str, Any] = {
        "schema_version": 1,
        "status": result_status,
        "completed_at": now_local(),
        "profile_id": plan["profile_id"],
        "mode": plan["mode"],
        "run_fingerprint": plan["fingerprints"]["run_fingerprint"],
        "selected_kernel_count": len(jobs),
        "valid_kernel_count": len(valid_jobs),
        "invalid_kernel_count": len(jobs) - len(valid_jobs),
        "presentations": len(make_presentations(18)),
        "agents": len(AGENT_NAMES),
        "oracle_contexts": sum(job["oracle_contexts"] for job in valid_jobs),
        "collision_census": {},
        "fault_replay": {},
        "gates": {},
    }

    for variant in variants:
        metrics = empty_collision_metrics()
        metrics["affected_kernel"] = False
        affected_kernels = 0
        for job in valid_jobs:
            source = job["collision"][variant]
            for field in (
                "semantic_contexts",
                "unique_projected_keys",
                "unsafe_key_classes",
                "contexts_in_unsafe_classes",
                "distinct_output_variants_in_unsafe_classes",
                "conflicting_context_pairs",
                "affected_kernel_agent_scopes",
            ):
                metrics[field] += int(source[field])
            if source["affected_kernel"]:
                affected_kernels += 1
        metrics["affected_valid_kernels"] = affected_kernels
        metrics.pop("affected_kernel")
        summary["collision_census"][variant] = metrics

        summary["fault_replay"][variant] = {}
        for order in orders:
            replay_metrics = {
                field: 0
                for field in (
                    "traces",
                    "trace_mismatches",
                    "candidates",
                    "candidate_signature_mismatches",
                    "signature_bit_flips",
                    "cache_hits",
                    "cache_misses",
                    "unsafe_cache_hits",
                    "nontermination_guards",
                )
            }
            for job in valid_jobs:
                source = job["replay"][variant][order]
                for field in replay_metrics:
                    replay_metrics[field] += int(source[field])
            for field in (
                "trace_digest",
                "candidate_signature_digest",
                "oracle_trace_digest",
                "oracle_candidate_signature_digest",
            ):
                replay_metrics[field] = aggregate_digest(
                    valid_jobs,
                    variant=variant,
                    order=order,
                    field=field,
                )
            summary["fault_replay"][variant][order] = replay_metrics

    full_control = (
        summary["collision_census"]["full"]["unsafe_key_classes"] == 0
    )
    for order in orders:
        full_replay = summary["fault_replay"]["full"][order]
        full_control = full_control and all(
            full_replay[field] == 0
            for field in (
                "trace_mismatches",
                "candidate_signature_mismatches",
                "unsafe_cache_hits",
                "nontermination_guards",
            )
        )
    full_control = full_control and (
        summary["fault_replay"]["full"]["canonical"]["trace_digest"]
        == summary["fault_replay"]["full"]["reverse"]["trace_digest"]
    )
    full_control = full_control and (
        summary["fault_replay"]["full"]["canonical"][
            "candidate_signature_digest"
        ]
        == summary["fault_replay"]["full"]["reverse"][
            "candidate_signature_digest"
        ]
    )

    component_necessity = {
        variant: (
            summary["collision_census"][variant]["unsafe_key_classes"] > 0
        )
        for variant in variants
        if variant != "full"
    }
    end_to_end_failures = {
        variant: {
            order: (
                summary["fault_replay"][variant][order][
                    "trace_mismatches"
                ]
                > 0
            )
            for order in orders
        }
        for variant in variants
        if variant != "full"
    }
    summary["gates"] = {
        "full_key_control_pass": full_control,
        "component_necessity_on_selected_domain": component_necessity,
        "end_to_end_trace_failure_observed": end_to_end_failures,
        "all_component_necessity_witnesses_present": all(
            component_necessity.values()
        ),
    }

    witnesses = {
        "schema_version": 1,
        "run_fingerprint": plan["fingerprints"]["run_fingerprint"],
        "collision_witnesses": collision_witnesses,
        "trace_mismatch_witnesses": trace_witnesses,
        "nontermination_witnesses": nontermination_witnesses,
    }
    return summary, witnesses


def summary_csv_text(
    summary: dict[str, Any],
    variants: tuple[str, ...],
    orders: tuple[str, ...],
) -> str:
    fieldnames = [
        "variant",
        "order",
        "unsafe_key_classes",
        "contexts_in_unsafe_classes",
        "affected_kernel_agent_scopes",
        "affected_valid_kernels",
        "traces",
        "trace_mismatches",
        "candidates",
        "candidate_signature_mismatches",
        "signature_bit_flips",
        "unsafe_cache_hits",
        "nontermination_guards",
    ]
    from io import StringIO

    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for variant in variants:
        census = summary["collision_census"][variant]
        for order in orders:
            replay = summary["fault_replay"][variant][order]
            writer.writerow(
                {
                    "variant": variant,
                    "order": order,
                    "unsafe_key_classes": census["unsafe_key_classes"],
                    "contexts_in_unsafe_classes": census[
                        "contexts_in_unsafe_classes"
                    ],
                    "affected_kernel_agent_scopes": census[
                        "affected_kernel_agent_scopes"
                    ],
                    "affected_valid_kernels": census[
                        "affected_valid_kernels"
                    ],
                    "traces": replay["traces"],
                    "trace_mismatches": replay["trace_mismatches"],
                    "candidates": replay["candidates"],
                    "candidate_signature_mismatches": replay[
                        "candidate_signature_mismatches"
                    ],
                    "signature_bit_flips": replay[
                        "signature_bit_flips"
                    ],
                    "unsafe_cache_hits": replay["unsafe_cache_hits"],
                    "nontermination_guards": replay[
                        "nontermination_guards"
                    ],
                }
            )
    return stream.getvalue()


def summary_markdown(
    summary: dict[str, Any],
    variants: tuple[str, ...],
    orders: tuple[str, ...],
) -> str:
    lines = [
        "# Cache-key ablation summary",
        "",
        f"- Status: `{summary['status']}`",
        f"- Selected kernels: {summary['selected_kernel_count']:,}",
        f"- Valid kernels: {summary['valid_kernel_count']:,}",
        f"- Oracle semantic contexts: {summary['oracle_contexts']:,}",
        f"- Full-key control: {'PASS' if summary['gates']['full_key_control_pass'] else 'FAIL'}",
        "",
        "| Variant | Unsafe key classes | Affected valid kernels | "
        + "Canonical trace mismatches | Reverse trace mismatches | "
        + "Canonical signature mismatches | Reverse signature mismatches |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in variants:
        census = summary["collision_census"][variant]
        canonical = summary["fault_replay"][variant]["canonical"]
        reverse = summary["fault_replay"][variant]["reverse"]
        lines.append(
            f"| {variant} | {census['unsafe_key_classes']:,} | "
            f"{census['affected_valid_kernels']:,} | "
            f"{canonical['trace_mismatches']:,} | "
            f"{reverse['trace_mismatches']:,} | "
            f"{canonical['candidate_signature_mismatches']:,} | "
            f"{reverse['candidate_signature_mismatches']:,} |"
        )
    lines.extend(
        [
            "",
            "The collision census is computed from oracle contexts and therefore "
            "does not depend on which presentation is replayed first. Fault replay "
            "is reported separately for canonical and reverse orders.",
            "",
            "This experiment establishes component-wise necessity only on the "
            "selected finite domain and declared agents. It is not a universal "
            "minimal-key theorem or an independent semantic proof.",
            "",
        ]
    )
    return "\n".join(lines)


def chunks_digest(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        name = path.name.encode("utf-8")
        digest.update(len(name).to_bytes(4, "big"))
        digest.update(name)
        data = path.read_bytes()
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--mode",
        choices=("smoke", "paper"),
        default="smoke",
        help="smoke is runner validation only; paper scans all configured kernels",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="defaults to 8; accepts at most 16",
    )
    parser.add_argument(
        "--kernels",
        type=int,
        default=None,
        help="override the selected kernel count for development only",
    )
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)
    grid_size = int(config["frozen_grid_size"])
    selected_count = (
        int(args.kernels)
        if args.kernels is not None
        else int(config[f"{args.mode}_kernel_count"])
    )
    selected_indices = selection_indices(grid_size, selected_count)
    workers = (
        int(args.workers)
        if args.workers is not None
        else int(config["default_workers"])
    )
    maximum_workers = int(config["maximum_workers"])
    if not 1 <= workers <= maximum_workers:
        raise ValueError(
            f"workers must be between 1 and {maximum_workers}; got {workers}"
        )
    chunk_size = (
        int(args.chunk_size)
        if args.chunk_size is not None
        else int(config["chunk_size"])
    )
    if chunk_size < 1:
        raise ValueError("chunk-size must be positive")

    variants = tuple(str(value) for value in config["variants"])
    orders = tuple(str(value) for value in config["orders"])
    fingerprints = build_fingerprints(
        config_path=config_path,
        mode=args.mode,
        workers=workers,
        chunk_size=chunk_size,
        selected_indices=selected_indices,
        config=config,
    )
    plan = {
        "schema_version": 1,
        "profile_id": config["profile_id"],
        "mode": args.mode,
        "paper_evidence_eligible": (
            args.mode == "paper" and selected_count == grid_size
        ),
        "selection": (
            "complete_grid"
            if selected_count == grid_size
            else "first_plus_midpoint"
        ),
        "frozen_grid_size": grid_size,
        "selected_kernel_count": selected_count,
        "selected_indices_sha256": digest_value(selected_indices),
        "workers": workers,
        "chunk_size": chunk_size,
        "variants": list(variants),
        "orders": list(orders),
        "minimum_step_guard": int(config["minimum_step_guard"]),
        "horizon_guard_multiplier": int(
            config["horizon_guard_multiplier"]
        ),
        "fingerprints": fingerprints,
        "machine": machine_payload(),
    }

    if args.dry_run:
        print(stable_json(plan, indent=2))
        return 0

    output = args.output.resolve()
    prepare_output_directory(output, resume=args.resume)
    validate_or_write_plan(output=output, resume=args.resume, plan=plan)

    all_kernels = make_kernels(grid_size)
    tasks = tuple(
        KernelTask(
            selection_position=selection_position,
            kernel_index=kernel_index,
            kernel=all_kernels[kernel_index],
        )
        for selection_position, kernel_index in enumerate(selected_indices)
    )
    task_chunks = chunked(tasks, chunk_size)
    expected_names = {
        chunk_name(index, task_chunk)
        for index, task_chunk in enumerate(task_chunks)
    }
    existing_names = {
        path.name for path in (output / "chunks").glob("chunk_*.json")
    }
    unexpected = sorted(existing_names - expected_names)
    if unexpected:
        raise ValueError(f"unexpected chunk files for this plan: {unexpected}")

    run_manifest = {
        **plan,
        "status": "running",
        "started_or_resumed_at": now_local(),
        "completed_chunks": 0,
        "total_chunks": len(task_chunks),
    }
    atomic_write_json(output / "run_manifest.json", run_manifest)

    presentations = make_presentations(18)
    worker_initializer_args = (
        presentations,
        variants,
        orders,
        int(config["minimum_step_guard"]),
        int(config["horizon_guard_multiplier"]),
    )

    executor: ProcessPoolExecutor | None = None
    if workers > 1:
        executor = ProcessPoolExecutor(
            max_workers=workers,
            initializer=initialize_worker,
            initargs=worker_initializer_args,
        )
    else:
        initialize_worker(*worker_initializer_args)

    started = time.perf_counter()
    completed_chunks = 0
    completed_jobs = 0
    chunk_payloads: list[dict[str, Any]] = []
    try:
        for chunk_index, task_chunk in enumerate(task_chunks):
            path = output / "chunks" / chunk_name(chunk_index, task_chunk)
            if path.exists():
                if not args.resume:
                    raise FileExistsError(f"chunk already exists: {path}")
                payload = json.loads(path.read_text(encoding="utf-8"))
                validate_chunk(
                    payload=payload,
                    expected_tasks=task_chunk,
                    run_fingerprint=fingerprints["run_fingerprint"],
                )
            else:
                if executor is None:
                    results = [
                        analyze_kernel_task(task) for task in task_chunk
                    ]
                else:
                    results = list(
                        executor.map(
                            analyze_kernel_task,
                            task_chunk,
                            chunksize=1,
                        )
                    )
                compacted = compact_chunk_results(results)
                payload = {
                    "schema_version": 1,
                    "run_fingerprint": fingerprints["run_fingerprint"],
                    "chunk_index": chunk_index,
                    "selection_start": task_chunk[0].selection_position,
                    "selection_end": task_chunk[-1].selection_position,
                    "written_at": now_local(),
                    **compacted,
                }
                validate_chunk(
                    payload=payload,
                    expected_tasks=task_chunk,
                    run_fingerprint=fingerprints["run_fingerprint"],
                )
                atomic_write_json(path, payload)

            chunk_payloads.append(payload)
            completed_chunks += 1
            completed_jobs += len(task_chunk)
            progress = {
                "status": "running",
                "run_fingerprint": fingerprints["run_fingerprint"],
                "completed_chunks": completed_chunks,
                "total_chunks": len(task_chunks),
                "completed_jobs": completed_jobs,
                "planned_jobs": len(tasks),
                "elapsed_s": time.perf_counter() - started,
                "updated_at": now_local(),
            }
            atomic_write_json(output / "progress.json", progress)
            print(
                f"[{completed_jobs}/{len(tasks)}] "
                f"completed chunk {completed_chunks}/{len(task_chunks)}",
                flush=True,
            )
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=False)

    summary, witnesses = aggregate_chunks(
        chunk_payloads=chunk_payloads,
        variants=variants,
        orders=orders,
        plan=plan,
    )
    if not summary["gates"]["full_key_control_pass"]:
        raise RuntimeError("full-key control failed; refusing to finalize")
    if not summary["gates"]["all_component_necessity_witnesses_present"]:
        raise RuntimeError(
            "one or more component ablations lacked a collision witness"
        )

    atomic_write_json(output / "summary.json", summary)
    atomic_write_json(output / "counterexamples.json", witnesses)
    atomic_write_text(
        output / "ablation_summary.csv",
        summary_csv_text(summary, variants, orders),
    )
    atomic_write_text(
        output / "SUMMARY.md",
        summary_markdown(summary, variants, orders),
    )
    final_progress = {
        "status": "completed",
        "run_fingerprint": fingerprints["run_fingerprint"],
        "completed_chunks": completed_chunks,
        "total_chunks": len(task_chunks),
        "completed_jobs": completed_jobs,
        "planned_jobs": len(tasks),
        "elapsed_s": time.perf_counter() - started,
        "updated_at": now_local(),
    }
    atomic_write_json(output / "progress.json", final_progress)

    artifact_paths = [
        output / "plan.json",
        output / "progress.json",
        output / "summary.json",
        output / "counterexamples.json",
        output / "ablation_summary.csv",
        output / "SUMMARY.md",
    ]
    completed_manifest = {
        **plan,
        "status": summary["status"],
        "completed_at": now_local(),
        "completed_chunks": completed_chunks,
        "total_chunks": len(task_chunks),
        "artifacts": {
            path.name: sha256_file(path) for path in artifact_paths
        },
        "chunks_sha256": chunks_digest(
            (output / "chunks").glob("chunk_*.json")
        ),
        "gates": summary["gates"],
    }
    atomic_write_json(output / "run_manifest.json", completed_manifest)
    print(
        f"completed {completed_jobs} kernel jobs; "
        f"full-key control PASS; output={output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
