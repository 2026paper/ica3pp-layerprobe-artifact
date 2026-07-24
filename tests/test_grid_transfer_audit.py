from __future__ import annotations

from dataclasses import asdict

from experiments.grid_transfer_audit import audit_mechanism
from experiments.grid_transfer_domain import (
    AGENT_NAMES,
    ORDERS,
    GridDisplayMemory,
    initial_state,
    make_mechanisms,
    make_presentations,
    observe,
    simulate_flat,
    simulate_layerprobe_views,
    transition,
    verify_mechanism,
    witness_mechanism,
)
from experiments.grid_transfer_oracle import (
    ORACLE_AGENTS,
    oracle_simulate,
    oracle_verify,
)


def test_frozen_grid_family_and_presentations_have_declared_sizes() -> None:
    mechanisms = make_mechanisms()
    presentations = make_presentations()
    assert len(mechanisms) == 1296
    assert len({spec.name for spec in mechanisms}) == 1296
    assert len(presentations) == 18
    assert len({presentation.name for presentation in presentations}) == 18
    assert {presentation.x_mode for presentation in presentations} == {
        "exact",
        "coarse",
        "hidden",
    }
    assert {presentation.y_mode for presentation in presentations} == {
        "exact",
        "coarse",
        "hidden",
    }
    assert {presentation.delay for presentation in presentations} == {0, 1}
    assert tuple(AGENT_NAMES) == tuple(ORACLE_AGENTS)


def test_grid_presentation_is_observation_only() -> None:
    spec = witness_mechanism()
    state = transition(
        transition(
            # Enter a non-initial pose without using presentation semantics.
            initial_state(spec),
            "forward",
            spec,
        ),
        "turn_left",
        spec,
    )
    expected_forward = transition(state, "forward", spec)
    for presentation in make_presentations():
        before = state
        observe(state, spec, presentation, GridDisplayMemory())
        assert state == before
        assert transition(state, "forward", spec) == expected_forward


def test_independent_oracle_and_complete_key_match_in_both_orders() -> None:
    spec = witness_mechanism()
    presentations = make_presentations()
    assert asdict(verify_mechanism(spec)) == asdict(oracle_verify(spec))
    assert verify_mechanism(spec).valid
    for agent in AGENT_NAMES:
        expected = {
            presentation.name: simulate_flat(
                spec,
                presentation,
                agent,
            )[0]
            for presentation in presentations
        }
        independent = {
            presentation.name: oracle_simulate(
                spec,
                presentation,
                agent,
            )
            for presentation in presentations
        }
        assert independent == expected
        for order in ORDERS:
            batch = simulate_layerprobe_views(
                spec,
                presentations,
                agent,
                variant="full",
                order=order,
            )
            assert batch.simulations == expected
            assert batch.counters.cache_hits > 0
            assert (
                batch.counters.policy_calls
                < batch.counters.observation_calls
            )


def test_each_weakened_key_has_replay_witness_in_both_orders() -> None:
    record = audit_mechanism(
        0,
        witness_mechanism(),
        make_presentations(),
    )
    assert record["eligible"]
    assert record["verification_equal"]
    assert record["oracle"]["trace_difference_count"] == 0
    for order in ORDERS:
        assert record["full"][order]["flat_trace_difference_count"] == 0
        assert record["full"][order]["oracle_trace_difference_count"] == 0
        assert record["full"][order]["signature_difference_count"] == 0
    for variant in (
        "drop_state",
        "drop_memory",
        "drop_observation",
    ):
        for order in ORDERS:
            evidence = record["weak"][variant][order]
            assert evidence["semantic_trace_difference_count"] > 0
            assert evidence["witness"] is not None
