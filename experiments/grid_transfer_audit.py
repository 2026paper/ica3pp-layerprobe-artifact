"""End-to-end audit for the second finite-state grid transfer case.

The audit compares four evidence paths over the same frozen mechanism family:

* a flat implementation replay;
* complete-key LayerProbe replay in canonical and reverse view order;
* an independently implemented naive interpreter; and
* three deliberately weakened cache keys in both replay orders.

One process-pool task owns one complete mechanism.  Results are checkpointed in
atomic chunks and can be resumed only when the domain and source fingerprints
match.  The final report contains compact digests rather than raw trace dumps;
expanded first witnesses are emitted separately for each weak key and order.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import platform
import socket
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.grid_transfer_domain import (  # noqa: E402
    AGENT_NAMES,
    CACHE_VARIANTS,
    ORDERS,
    GridCounters,
    GridMechanism,
    GridPresentation,
    GridSimulation,
    GridTraceStep,
    behavioral_projection,
    make_mechanisms,
    make_presentations,
    signature_for,
    simulate_flat,
    simulate_layerprobe_views,
    verify_mechanism,
)
from experiments.grid_transfer_oracle import (  # noqa: E402
    ORACLE_AGENTS,
    oracle_simulate,
    oracle_verify,
)


SCHEMA_VERSION = 1
EXPERIMENT_ID = "grid-transfer-domain-v1"
WEAK_VARIANTS = tuple(
    variant for variant in CACHE_VARIANTS if variant != "full"
)
EXPECTED_FULL_MECHANISMS = 1296
EXPECTED_PRESENTATIONS = 18
MAXIMUM_WORKERS = 16


def now_local() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _json_ready(value: Any) -> Any:
    if is_dataclass(value):
        return _json_ready(asdict(value))
    if isinstance(value, dict):
        return {
            str(key): _json_ready(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    return value


def stable_json(value: Any, *, indent: int | None = None) -> str:
    return json.dumps(
        _json_ready(value),
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


def atomic_write_csv(
    path: Path,
    rows: Iterable[dict[str, Any]],
    fields: tuple[str, ...],
) -> None:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field, "") for field in fields})
    atomic_write_text(path, buffer.getvalue())


def simulation_payload(simulation: GridSimulation) -> dict[str, Any]:
    return {
        "trace": [asdict(step) for step in simulation.trace],
        "termination": simulation.termination,
    }


def simulation_digest(simulation: GridSimulation) -> str:
    return digest_value(simulation_payload(simulation))


def _family_trace_digest(
    simulations_by_agent: dict[str, dict[str, GridSimulation]],
) -> str:
    ledger: list[dict[str, str]] = []
    for agent in sorted(simulations_by_agent):
        for presentation_name in sorted(simulations_by_agent[agent]):
            ledger.append(
                {
                    "agent": agent,
                    "presentation": presentation_name,
                    "digest": simulation_digest(
                        simulations_by_agent[agent][presentation_name]
                    ),
                }
            )
    return digest_value(ledger)


def _signatures(
    simulations_by_agent: dict[str, dict[str, GridSimulation]],
    presentations: tuple[GridPresentation, ...],
) -> dict[str, int]:
    result: dict[str, int] = {}
    for presentation in presentations:
        by_agent = {
            agent: simulations_by_agent[agent][presentation.name]
            for agent in AGENT_NAMES
        }
        result[presentation.name] = signature_for(by_agent)
    return result


def _trace_difference_count(
    expected: dict[str, dict[str, GridSimulation]],
    actual: dict[str, dict[str, GridSimulation]],
    presentations: tuple[GridPresentation, ...],
) -> int:
    return sum(
        expected[agent][presentation.name]
        != actual[agent][presentation.name]
        for agent in AGENT_NAMES
        for presentation in presentations
    )


def _signature_difference_count(
    expected: dict[str, int],
    actual: dict[str, int],
) -> int:
    return sum(
        expected[name] != actual[name]
        for name in sorted(expected)
    )


def _first_step_mismatch(
    expected: GridSimulation,
    actual: GridSimulation,
) -> tuple[int, str] | None:
    common = min(len(expected.trace), len(actual.trace))
    for index in range(common):
        if expected.trace[index] != actual.trace[index]:
            return index, "step"
    if len(expected.trace) != len(actual.trace):
        return common, "trace_length"
    if expected.termination != actual.termination:
        return common, "termination"
    return None


def _first_family_mismatch(
    expected: dict[str, dict[str, GridSimulation]],
    actual: dict[str, dict[str, GridSimulation]],
    presentations: tuple[GridPresentation, ...],
) -> dict[str, Any] | None:
    for agent in AGENT_NAMES:
        for presentation in presentations:
            expected_simulation = expected[agent][presentation.name]
            actual_simulation = actual[agent][presentation.name]
            mismatch = _first_step_mismatch(
                expected_simulation,
                actual_simulation,
            )
            if mismatch is None:
                continue
            index, mismatch_kind = mismatch
            expected_step = (
                expected_simulation.trace[index]
                if index < len(expected_simulation.trace)
                else None
            )
            actual_step = (
                actual_simulation.trace[index]
                if index < len(actual_simulation.trace)
                else None
            )
            return {
                "agent": agent,
                "presentation": presentation.name,
                "trace_index": index,
                "mismatch_kind": mismatch_kind,
                "expected_step_digest": (
                    None if expected_step is None else digest_value(expected_step)
                ),
                "actual_step_digest": (
                    None if actual_step is None else digest_value(actual_step)
                ),
                "expected_termination": expected_simulation.termination,
                "actual_termination": actual_simulation.termination,
                "behavioral_mismatch": (
                    behavioral_projection(expected_simulation)
                    != behavioral_projection(actual_simulation)
                ),
            }
    return None


def _run_flat_family(
    spec: GridMechanism,
    presentations: tuple[GridPresentation, ...],
) -> tuple[dict[str, dict[str, GridSimulation]], GridCounters]:
    simulations: dict[str, dict[str, GridSimulation]] = {}
    counters = GridCounters()
    for agent in AGENT_NAMES:
        simulations[agent] = {}
        for presentation in presentations:
            simulation, local = simulate_flat(spec, presentation, agent)
            simulations[agent][presentation.name] = simulation
            counters.add(local)
    return simulations, counters


def _run_oracle_family(
    spec: GridMechanism,
    presentations: tuple[GridPresentation, ...],
) -> dict[str, dict[str, GridSimulation]]:
    return {
        agent: {
            presentation.name: oracle_simulate(spec, presentation, agent)
            for presentation in presentations
        }
        for agent in ORACLE_AGENTS
    }


def _run_cached_family(
    spec: GridMechanism,
    presentations: tuple[GridPresentation, ...],
    *,
    variant: str,
    order: str,
) -> tuple[dict[str, dict[str, GridSimulation]], GridCounters]:
    simulations: dict[str, dict[str, GridSimulation]] = {}
    counters = GridCounters()
    for agent in AGENT_NAMES:
        batch = simulate_layerprobe_views(
            spec,
            presentations,
            agent,
            variant=variant,  # type: ignore[arg-type]
            order=order,
        )
        simulations[agent] = batch.simulations
        counters.add(batch.counters)
    return simulations, counters


def audit_mechanism(
    mechanism_index: int,
    spec: GridMechanism,
    presentations: tuple[GridPresentation, ...],
) -> dict[str, Any]:
    """Audit one mechanism completely; safe to call in a process worker."""

    core_verification = verify_mechanism(spec)
    oracle_verification = oracle_verify(spec)
    core_verification_payload = asdict(core_verification)
    oracle_verification_payload = asdict(oracle_verification)
    verification_equal = (
        core_verification_payload == oracle_verification_payload
    )
    base: dict[str, Any] = {
        "mechanism_index": mechanism_index,
        "mechanism_name": spec.name,
        "mechanism_digest": digest_value(spec),
        "core_verification": core_verification_payload,
        "oracle_verification": oracle_verification_payload,
        "verification_equal": verification_equal,
        "eligible": core_verification.valid and oracle_verification.valid,
        "candidate_count": 0,
        "flat": None,
        "oracle": None,
        "full": {},
        "weak": {},
    }
    if not base["eligible"]:
        return base

    flat, flat_counters = _run_flat_family(spec, presentations)
    oracle = _run_oracle_family(spec, presentations)
    flat_signatures = _signatures(flat, presentations)
    oracle_signatures = _signatures(oracle, presentations)
    oracle_trace_differences = _trace_difference_count(
        flat,
        oracle,
        presentations,
    )
    oracle_signature_differences = _signature_difference_count(
        flat_signatures,
        oracle_signatures,
    )
    base["candidate_count"] = len(presentations)
    base["flat"] = {
        "trace_digest": _family_trace_digest(flat),
        "signature_digest": digest_value(flat_signatures),
        "signature_masks": [
            flat_signatures[presentation.name]
            for presentation in presentations
        ],
        "counters": flat_counters.as_dict(),
    }
    base["oracle"] = {
        "trace_digest": _family_trace_digest(oracle),
        "signature_digest": digest_value(oracle_signatures),
        "trace_difference_count": oracle_trace_differences,
        "signature_difference_count": oracle_signature_differences,
    }

    for order in ORDERS:
        full, full_counters = _run_cached_family(
            spec,
            presentations,
            variant="full",
            order=order,
        )
        full_signatures = _signatures(full, presentations)
        base["full"][order] = {
            "trace_digest": _family_trace_digest(full),
            "signature_digest": digest_value(full_signatures),
            "flat_trace_difference_count": _trace_difference_count(
                flat,
                full,
                presentations,
            ),
            "oracle_trace_difference_count": _trace_difference_count(
                oracle,
                full,
                presentations,
            ),
            "signature_difference_count": _signature_difference_count(
                flat_signatures,
                full_signatures,
            ),
            "counters": full_counters.as_dict(),
        }

    for variant in WEAK_VARIANTS:
        base["weak"][variant] = {}
        for order in ORDERS:
            mutant, mutant_counters = _run_cached_family(
                spec,
                presentations,
                variant=variant,
                order=order,
            )
            mutant_signatures = _signatures(mutant, presentations)
            mismatch_count = _trace_difference_count(
                flat,
                mutant,
                presentations,
            )
            locator = _first_family_mismatch(
                flat,
                mutant,
                presentations,
            )
            base["weak"][variant][order] = {
                "semantic_trace_difference_count": mismatch_count,
                "signature_difference_count": _signature_difference_count(
                    flat_signatures,
                    mutant_signatures,
                ),
                "witness": locator,
                "counters": mutant_counters.as_dict(),
            }
    return base


def _worker(
    task: tuple[
        int,
        GridMechanism,
        tuple[GridPresentation, ...],
    ],
) -> dict[str, Any]:
    return audit_mechanism(*task)


def _sum_counters(
    records: list[dict[str, Any]],
    selector: tuple[str, ...],
) -> dict[str, int]:
    fields = tuple(GridCounters().as_dict())
    totals = {field: 0 for field in fields}
    peak = 0
    for record in records:
        current: Any = record
        try:
            for key in selector:
                current = current[key]
        except (KeyError, TypeError):
            continue
        if current is None:
            continue
        for field in fields:
            if field == "peak_cache_entries":
                peak = max(peak, int(current.get(field, 0)))
            else:
                totals[field] += int(current.get(field, 0))
    totals["peak_cache_entries"] = peak
    return totals


def _global_digest(
    records: list[dict[str, Any]],
    selector: tuple[str, ...],
) -> str:
    ledger: list[dict[str, Any]] = []
    for record in records:
        current: Any = record
        try:
            for key in selector:
                current = current[key]
        except (KeyError, TypeError):
            current = None
        ledger.append(
            {
                "mechanism_index": record["mechanism_index"],
                "value": current,
            }
        )
    return digest_value(ledger)


def _step_payload(step: GridTraceStep | None) -> Any:
    return None if step is None else asdict(step)


def _expanded_witness(
    record: dict[str, Any],
    spec: GridMechanism,
    presentations: tuple[GridPresentation, ...],
    variant: str,
    order: str,
) -> dict[str, Any]:
    locator = record["weak"][variant][order]["witness"]
    if locator is None:
        raise ValueError("cannot expand a missing weak-key witness")
    presentation_by_name = {
        presentation.name: presentation for presentation in presentations
    }
    presentation = presentation_by_name[locator["presentation"]]
    expected, _ = simulate_flat(spec, presentation, locator["agent"])
    actual_batch = simulate_layerprobe_views(
        spec,
        presentations,
        locator["agent"],
        variant=variant,  # type: ignore[arg-type]
        order=order,
    )
    actual = actual_batch.simulations[presentation.name]
    mismatch = _first_step_mismatch(expected, actual)
    if mismatch is None:
        raise AssertionError("stored witness no longer reproduces")
    trace_index, mismatch_kind = mismatch
    expected_step = (
        expected.trace[trace_index]
        if trace_index < len(expected.trace)
        else None
    )
    actual_step = (
        actual.trace[trace_index]
        if trace_index < len(actual.trace)
        else None
    )
    return {
        "variant": variant,
        "order": order,
        "mechanism_index": record["mechanism_index"],
        "mechanism": asdict(spec),
        "agent": locator["agent"],
        "presentation": asdict(presentation),
        "trace_index": trace_index,
        "mismatch_kind": mismatch_kind,
        "expected_step": _step_payload(expected_step),
        "actual_step": _step_payload(actual_step),
        "expected_termination": expected.termination,
        "actual_termination": actual.termination,
        "expected_trace_digest": simulation_digest(expected),
        "actual_trace_digest": simulation_digest(actual),
        "behavioral_mismatch": (
            behavioral_projection(expected)
            != behavioral_projection(actual)
        ),
        "mutant_counters": actual_batch.counters.as_dict(),
    }


def _build_summary(
    records: list[dict[str, Any]],
    mechanisms: tuple[GridMechanism, ...],
    presentations: tuple[GridPresentation, ...],
) -> tuple[dict[str, Any], dict[str, Any]]:
    eligible = [record for record in records if record["eligible"]]
    verification_differences = sum(
        not record["verification_equal"] for record in records
    )
    oracle_trace_differences = sum(
        int(record["oracle"]["trace_difference_count"])
        for record in eligible
    )
    oracle_signature_differences = sum(
        int(record["oracle"]["signature_difference_count"])
        for record in eligible
    )

    full_summary: dict[str, Any] = {}
    for order in ORDERS:
        counters = _sum_counters(
            eligible,
            ("full", order, "counters"),
        )
        full_summary[order] = {
            "flat_trace_difference_count": sum(
                int(
                    record["full"][order][
                        "flat_trace_difference_count"
                    ]
                )
                for record in eligible
            ),
            "oracle_trace_difference_count": sum(
                int(
                    record["full"][order][
                        "oracle_trace_difference_count"
                    ]
                )
                for record in eligible
            ),
            "signature_difference_count": sum(
                int(
                    record["full"][order][
                        "signature_difference_count"
                    ]
                )
                for record in eligible
            ),
            "trace_digest": _global_digest(
                eligible,
                ("full", order, "trace_digest"),
            ),
            "signature_digest": _global_digest(
                eligible,
                ("full", order, "signature_digest"),
            ),
            "counters": counters,
            "cache_hit_rate": (
                counters["cache_hits"] / counters["cache_lookups"]
                if counters["cache_lookups"]
                else 0.0
            ),
        }

    weak_summary: dict[str, Any] = {}
    expanded_witnesses: dict[str, Any] = {}
    mechanisms_by_index = {
        index: spec for index, spec in enumerate(mechanisms)
    }
    for variant in WEAK_VARIANTS:
        weak_summary[variant] = {}
        expanded_witnesses[variant] = {}
        for order in ORDERS:
            witnesses = [
                record
                for record in eligible
                if record["weak"][variant][order]["witness"] is not None
            ]
            semantic_differences = sum(
                int(
                    record["weak"][variant][order][
                        "semantic_trace_difference_count"
                    ]
                )
                for record in eligible
            )
            signature_differences = sum(
                int(
                    record["weak"][variant][order][
                        "signature_difference_count"
                    ]
                )
                for record in eligible
            )
            behavior_witnesses = sum(
                bool(
                    record["weak"][variant][order]["witness"][
                        "behavioral_mismatch"
                    ]
                )
                for record in witnesses
            )
            weak_summary[variant][order] = {
                "mechanisms_with_semantic_witness": len(witnesses),
                "mechanisms_with_behavioral_first_witness": behavior_witnesses,
                "semantic_trace_difference_count": semantic_differences,
                "signature_difference_count": signature_differences,
                "witness_found": bool(witnesses),
                "counters": _sum_counters(
                    eligible,
                    ("weak", variant, order, "counters"),
                ),
            }
            if witnesses:
                first_record = min(
                    witnesses,
                    key=lambda item: int(item["mechanism_index"]),
                )
                first_spec = mechanisms_by_index[
                    int(first_record["mechanism_index"])
                ]
                expanded_witnesses[variant][order] = _expanded_witness(
                    first_record,
                    first_spec,
                    presentations,
                    variant,
                    order,
                )
            else:
                expanded_witnesses[variant][order] = None

    flat_counters = _sum_counters(eligible, ("flat", "counters"))
    canonical_counters = full_summary["canonical"]["counters"]
    exact_gate = {
        "eligible_mechanism_count": len(eligible),
        "mechanism_verification_difference_count": verification_differences,
        "flat_vs_oracle_trace_difference_count": oracle_trace_differences,
        "flat_vs_oracle_signature_difference_count": (
            oracle_signature_differences
        ),
        "full_key": {
            order: {
                "flat_trace_difference_count": full_summary[order][
                    "flat_trace_difference_count"
                ],
                "oracle_trace_difference_count": full_summary[order][
                    "oracle_trace_difference_count"
                ],
                "signature_difference_count": full_summary[order][
                    "signature_difference_count"
                ],
            }
            for order in ORDERS
        },
        "canonical_reverse_trace_digest_equal": (
            full_summary["canonical"]["trace_digest"]
            == full_summary["reverse"]["trace_digest"]
        ),
        "canonical_reverse_signature_digest_equal": (
            full_summary["canonical"]["signature_digest"]
            == full_summary["reverse"]["signature_digest"]
        ),
        "complete_key_cache_hits_positive": all(
            full_summary[order]["counters"]["cache_hits"] > 0
            for order in ORDERS
        ),
        "weak_key_witness_found": {
            variant: {
                order: weak_summary[variant][order]["witness_found"]
                for order in ORDERS
            }
            for variant in WEAK_VARIANTS
        },
    }
    exact_gate["pass"] = bool(
        len(eligible) > 0
        and verification_differences == 0
        and oracle_trace_differences == 0
        and oracle_signature_differences == 0
        and all(
            full_summary[order]["flat_trace_difference_count"] == 0
            and full_summary[order]["oracle_trace_difference_count"] == 0
            and full_summary[order]["signature_difference_count"] == 0
            for order in ORDERS
        )
        and exact_gate["canonical_reverse_trace_digest_equal"]
        and exact_gate["canonical_reverse_signature_digest_equal"]
        and exact_gate["complete_key_cache_hits_positive"]
        and all(
            weak_summary[variant][order]["witness_found"]
            for variant in WEAK_VARIANTS
            for order in ORDERS
        )
    )

    summary = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "completed_at": now_local(),
        "domain": {
            "declared_mechanism_count": len(mechanisms),
            "processed_mechanism_count": len(records),
            "well_formed_mechanism_count": sum(
                bool(record["core_verification"]["well_formed"])
                for record in records
            ),
            "reachable_mechanism_count": len(eligible),
            "presentation_count": len(presentations),
            "agent_count": len(AGENT_NAMES),
            "candidate_count": sum(
                int(record["candidate_count"]) for record in records
            ),
            "semantic_simulation_count": (
                len(eligible) * len(presentations) * len(AGENT_NAMES)
            ),
        },
        "digests": {
            "mechanism_family": digest_value(mechanisms),
            "presentation_family": digest_value(presentations),
            "flat_traces": _global_digest(
                eligible,
                ("flat", "trace_digest"),
            ),
            "oracle_traces": _global_digest(
                eligible,
                ("oracle", "trace_digest"),
            ),
            "flat_signatures": _global_digest(
                eligible,
                ("flat", "signature_digest"),
            ),
            "oracle_signatures": _global_digest(
                eligible,
                ("oracle", "signature_digest"),
            ),
        },
        "work": {
            "flat": flat_counters,
            "full_key": full_summary,
            "canonical_policy_call_reduction_fraction": (
                1.0
                - canonical_counters["policy_calls"]
                / flat_counters["policy_calls"]
                if flat_counters["policy_calls"]
                else 0.0
            ),
            "canonical_transition_call_reduction_fraction": (
                1.0
                - canonical_counters["transition_calls"]
                / flat_counters["transition_calls"]
                if flat_counters["transition_calls"]
                else 0.0
            ),
        },
        "weak_keys": weak_summary,
        "semantic_gate": exact_gate,
        "claim_boundary": (
            "This is a second finite-state structural transfer case on one "
            "workstation, not evidence of real-world or cross-platform "
            "generality."
        ),
    }
    return summary, expanded_witnesses


def _chunk_filename(start: int, stop: int) -> str:
    return f"chunk_{start:04d}_{stop - 1:04d}.json"


def _source_fingerprints() -> dict[str, str]:
    files = (
        Path(__file__).resolve(),
        Path(__file__).with_name("grid_transfer_domain.py").resolve(),
        Path(__file__).with_name("grid_transfer_oracle.py").resolve(),
        Path(__file__).with_name("GRID_TRANSFER_AUDIT_README.md").resolve(),
        PROJECT_ROOT / "tests" / "test_grid_transfer_audit.py",
    )
    return {
        path.relative_to(PROJECT_ROOT).as_posix(): sha256_file(path)
        for path in files
        if path.exists()
    }


def _load_checkpoint(
    path: Path,
    *,
    expected_indices: list[int],
    domain_digest: str,
    source_fingerprints: dict[str, str],
) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"checkpoint schema mismatch: {path}")
    if payload.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError(f"checkpoint experiment mismatch: {path}")
    if payload.get("domain_digest") != domain_digest:
        raise ValueError(f"checkpoint domain mismatch: {path}")
    if payload.get("source_fingerprints") != source_fingerprints:
        raise ValueError(f"checkpoint source mismatch: {path}")
    indices = [
        int(record["mechanism_index"])
        for record in payload.get("records", [])
    ]
    if indices != expected_indices:
        raise ValueError(f"checkpoint task range mismatch: {path}")
    return list(payload["records"])


def run_audit(
    *,
    output: Path,
    workers: int,
    limit: int | None,
    chunk_size: int,
    resume: bool,
) -> dict[str, Any]:
    if not 1 <= workers <= MAXIMUM_WORKERS:
        raise ValueError(
            f"workers must be between 1 and {MAXIMUM_WORKERS}"
        )
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if tuple(AGENT_NAMES) != tuple(ORACLE_AGENTS):
        raise ValueError("implementation and oracle agent orders differ")

    mechanisms = make_mechanisms(limit)
    presentations = make_presentations()
    if limit is None and len(mechanisms) != EXPECTED_FULL_MECHANISMS:
        raise ValueError("unexpected full mechanism count")
    if len(presentations) != EXPECTED_PRESENTATIONS:
        raise ValueError("unexpected presentation count")

    output = output.resolve()
    chunks_directory = output / "chunks"
    chunks_directory.mkdir(parents=True, exist_ok=True)
    source_fingerprints = _source_fingerprints()
    domain_digest = digest_value(
        {
            "mechanisms": mechanisms,
            "presentations": presentations,
            "agents": AGENT_NAMES,
            "variants": CACHE_VARIANTS,
            "orders": ORDERS,
        }
    )
    domain_manifest_path = output / "domain_manifest.json"
    domain_manifest = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "domain_digest": domain_digest,
        "presentation_contract": {
            "observe_signature": (
                "observe(world_state, mechanism, presentation, "
                "display_memory) -> (observation, next_display_memory)"
            ),
            "invariants": [
                "presentation does not modify mechanism parameters",
                "presentation does not modify the initial world state",
                "presentation does not modify the action set or transition",
                "presentation does not modify goal or termination rules",
            ],
            "ownership": {
                "mechanism_owned": [
                    "GridMechanism",
                    "GridWorldState",
                ],
                "presentation_local": [
                    "GridPresentation",
                    "GridDisplayMemory",
                ],
                "agent_trajectory_owned": ["GridAgentMemory"],
                "cache_scope": "one GridMechanism--agent pair",
                "complete_key": [
                    "GridWorldState",
                    "pre-ingest GridAgentMemory",
                    "GridObservation",
                ],
            },
        },
        "mechanisms": [asdict(spec) for spec in mechanisms],
        "presentations": [
            asdict(presentation) for presentation in presentations
        ],
        "agents": list(AGENT_NAMES),
        "cache_variants": list(CACHE_VARIANTS),
        "replay_orders": list(ORDERS),
    }
    if domain_manifest_path.exists():
        existing_domain = json.loads(
            domain_manifest_path.read_text(encoding="utf-8")
        )
        if existing_domain.get("domain_digest") != domain_digest:
            raise ValueError("resume refused: domain manifest changed")
    else:
        atomic_write_json(domain_manifest_path, domain_manifest)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        previous_manifest = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )
        if not resume:
            raise FileExistsError(
                f"{manifest_path} exists; pass --resume to reuse checkpoints"
            )
        if previous_manifest.get("domain_digest") != domain_digest:
            raise ValueError("resume refused: domain digest changed")
        if previous_manifest.get("source_fingerprints") != source_fingerprints:
            raise ValueError("resume refused: audited source changed")
    else:
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "experiment_id": EXPERIMENT_ID,
            "started_at": now_local(),
            "status": "running",
            "command": sys.argv,
            "output": str(output),
            "workers": workers,
            "maximum_workers": MAXIMUM_WORKERS,
            "chunk_size": chunk_size,
            "limit": limit,
            "host": {
                "hostname": socket.gethostname(),
                "platform": platform.platform(),
                "python": sys.version,
                "logical_cpu_count": os.cpu_count(),
            },
            "domain_digest": domain_digest,
            "source_fingerprints": source_fingerprints,
            "mechanism_count": len(mechanisms),
            "presentation_count": len(presentations),
            "agent_count": len(AGENT_NAMES),
            "claim_boundary": (
                "Single-workstation finite-state transfer case; no "
                "real-world or multi-platform generality claim."
            ),
        }
        atomic_write_json(manifest_path, manifest)

    records: list[dict[str, Any]] = []
    started = time.perf_counter()
    for start in range(0, len(mechanisms), chunk_size):
        stop = min(len(mechanisms), start + chunk_size)
        chunk_path = chunks_directory / _chunk_filename(start, stop)
        expected_indices = list(range(start, stop))
        if chunk_path.exists():
            if not resume:
                raise FileExistsError(
                    f"{chunk_path} exists; pass --resume to reuse it"
                )
            chunk_records = _load_checkpoint(
                chunk_path,
                expected_indices=expected_indices,
                domain_digest=domain_digest,
                source_fingerprints=source_fingerprints,
            )
        else:
            tasks = tuple(
                (index, mechanisms[index], presentations)
                for index in expected_indices
            )
            if workers == 1 or len(tasks) == 1:
                chunk_records = [_worker(task) for task in tasks]
            else:
                with ProcessPoolExecutor(
                    max_workers=min(workers, len(tasks))
                ) as executor:
                    chunk_records = list(executor.map(_worker, tasks))
            chunk_payload = {
                "schema_version": SCHEMA_VERSION,
                "experiment_id": EXPERIMENT_ID,
                "created_at": now_local(),
                "domain_digest": domain_digest,
                "source_fingerprints": source_fingerprints,
                "records": chunk_records,
            }
            atomic_write_json(chunk_path, chunk_payload)
        records.extend(chunk_records)

    records.sort(key=lambda record: int(record["mechanism_index"]))
    expected_all_indices = list(range(len(mechanisms)))
    actual_all_indices = [
        int(record["mechanism_index"]) for record in records
    ]
    if actual_all_indices != expected_all_indices:
        raise AssertionError("merged checkpoint ledger is incomplete or duplicated")

    summary, expanded_witnesses = _build_summary(
        records,
        mechanisms,
        presentations,
    )
    summary["elapsed_s"] = time.perf_counter() - started
    atomic_write_json(output / "summary.json", summary)
    atomic_write_json(
        output / "semantic_gate.json",
        summary["semantic_gate"],
    )
    atomic_write_json(
        output / "weak_key_witnesses.json",
        expanded_witnesses,
    )

    mechanism_rows: list[dict[str, Any]] = []
    digest_rows: list[dict[str, Any]] = []
    for record in records:
        eligible = bool(record["eligible"])
        row = {
            "mechanism_index": record["mechanism_index"],
            "mechanism_name": record["mechanism_name"],
            "well_formed": int(
                bool(record["core_verification"]["well_formed"])
            ),
            "reachable": int(
                bool(record["core_verification"]["reachable"])
            ),
            "shortest_win": (
                ""
                if record["core_verification"]["shortest_win"] is None
                else record["core_verification"]["shortest_win"]
            ),
            "graph_states": record["core_verification"]["states"],
            "graph_transitions": record["core_verification"][
                "transitions"
            ],
            "verification_equal": int(record["verification_equal"]),
            "eligible": int(eligible),
            "candidate_count": record["candidate_count"],
        }
        for order in ORDERS:
            row[f"full_{order}_trace_differences"] = (
                record["full"][order]["flat_trace_difference_count"]
                if eligible
                else ""
            )
            row[f"full_{order}_cache_hits"] = (
                record["full"][order]["counters"]["cache_hits"]
                if eligible
                else ""
            )
            row[f"full_{order}_cache_misses"] = (
                record["full"][order]["counters"]["cache_misses"]
                if eligible
                else ""
            )
        for variant in WEAK_VARIANTS:
            for order in ORDERS:
                row[f"{variant}_{order}_witness"] = (
                    int(
                        record["weak"][variant][order]["witness"]
                        is not None
                    )
                    if eligible
                    else ""
                )
        mechanism_rows.append(row)
        digest_rows.append(
            {
                "mechanism_index": record["mechanism_index"],
                "mechanism_name": record["mechanism_name"],
                "mechanism_digest": record["mechanism_digest"],
                "flat_trace_digest": (
                    record["flat"]["trace_digest"] if eligible else ""
                ),
                "oracle_trace_digest": (
                    record["oracle"]["trace_digest"] if eligible else ""
                ),
                "full_canonical_trace_digest": (
                    record["full"]["canonical"]["trace_digest"]
                    if eligible
                    else ""
                ),
                "full_reverse_trace_digest": (
                    record["full"]["reverse"]["trace_digest"]
                    if eligible
                    else ""
                ),
                "flat_signature_digest": (
                    record["flat"]["signature_digest"] if eligible else ""
                ),
                "oracle_signature_digest": (
                    record["oracle"]["signature_digest"]
                    if eligible
                    else ""
                ),
            }
        )
    mechanism_fields = tuple(mechanism_rows[0]) if mechanism_rows else ()
    digest_fields = tuple(digest_rows[0]) if digest_rows else ()
    atomic_write_csv(
        output / "mechanisms.csv",
        mechanism_rows,
        mechanism_fields,
    )
    atomic_write_csv(
        output / "digest_ledger.csv",
        digest_rows,
        digest_fields,
    )

    final_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    final_manifest.update(
        {
            "completed_at": now_local(),
            "status": (
                "complete_pass"
                if summary["semantic_gate"]["pass"]
                else "complete_fail"
            ),
            "elapsed_s": summary["elapsed_s"],
            "summary_sha256": sha256_file(output / "summary.json"),
            "semantic_gate_sha256": sha256_file(
                output / "semantic_gate.json"
            ),
            "weak_key_witnesses_sha256": sha256_file(
                output / "weak_key_witnesses.json"
            ),
            "mechanisms_csv_sha256": sha256_file(
                output / "mechanisms.csv"
            ),
            "digest_ledger_csv_sha256": sha256_file(
                output / "digest_ledger.csv"
            ),
            "domain_manifest_sha256": sha256_file(
                output / "domain_manifest.json"
            ),
        }
    )
    atomic_write_json(manifest_path, final_manifest)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT
        / "results"
        / "grid_transfer_audit_20260724",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="deterministic prefix for smoke/testing; omit for all 1,296",
    )
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run_audit(
        output=args.output,
        workers=args.workers,
        limit=args.limit,
        chunk_size=args.chunk_size,
        resume=args.resume,
    )
    print(
        stable_json(
            {
                "output": str(args.output.resolve()),
                "elapsed_s": summary["elapsed_s"],
                "reachable_mechanisms": summary["domain"][
                    "reachable_mechanism_count"
                ],
                "candidate_count": summary["domain"]["candidate_count"],
                "policy_call_reduction_fraction": summary["work"][
                    "canonical_policy_call_reduction_fraction"
                ],
                "semantic_gate_pass": summary["semantic_gate"]["pass"],
            },
            indent=2,
        )
    )
    return 0 if summary["semantic_gate"]["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
