from __future__ import annotations

from collections import Counter

from experiments.randomized_mutation_audit import (
    ALL_KEY_FIELDS,
    BOUNDARY_TARGETS,
    FAMILIES,
    generate_mutation_catalog,
)


def test_fixed_seed_catalog_is_balanced_unique_and_well_formed() -> None:
    first = generate_mutation_catalog(seed=20260724, per_family=20)
    second = generate_mutation_catalog(seed=20260724, per_family=20)
    assert first == second
    assert len(first) == 60
    assert len({item["mutation_id"] for item in first}) == 60
    assert Counter(item["family"] for item in first) == {
        family: 20 for family in FAMILIES
    }

    boundary = [
        item for item in first if item["family"] == "boundary_offset"
    ]
    assert all(
        offsets
        and set(offsets) <= set(BOUNDARY_TARGETS)
        and set(offsets.values()) <= {-1, 1}
        for offsets in (
            item["parameters"]["offsets"] for item in boundary
        )
    )

    key_mutants = [
        item
        for item in first
        if item["family"] == "cache_key_projection"
    ]
    kept = [
        tuple(item["parameters"]["kept_fields"])
        for item in key_mutants
    ]
    assert len(set(kept)) == 20
    assert all(
        0 < len(fields) < len(ALL_KEY_FIELDS) for fields in kept
    )
