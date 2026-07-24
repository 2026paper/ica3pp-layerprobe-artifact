from __future__ import annotations

import hashlib
import json
from pathlib import Path

from experiments.distance_sentinel_impact_audit import (
    DEFAULT_SPEC,
    summarize_candidate_changes,
    update_aggregate_hash,
    validate_spec,
)


def test_candidate_change_summary_tracks_direction_and_hidden_control() -> None:
    presentations = [
        {
            "name": "visible",
            "speed_mode": "exact",
            "distance_mode": "exact",
            "delay": 0,
        },
        {
            "name": "hidden",
            "speed_mode": "exact",
            "distance_mode": "hidden",
            "delay": 0,
        },
    ]
    pairs = [["a", "b"], ["a", "c"]]
    old = {
        ("k0", "visible"): 0b11,
        ("k0", "hidden"): 0b01,
        ("k1", "visible"): 0b00,
        ("k1", "hidden"): 0b10,
    }
    new = {
        ("k0", "visible"): 0b10,
        ("k0", "hidden"): 0b01,
        ("k1", "visible"): 0b01,
        ("k1", "hidden"): 0b10,
    }

    result = summarize_candidate_changes(old, new, presentations, pairs)

    assert result["changed_candidates"] == 2
    assert result["changed_kernels"] == 2
    assert result["hidden_distance_changes"] == 0
    assert result["transitions"] == {(0b11, 0b10): 1, (0b00, 0b01): 1}
    assert result["directional_pair_flips"] == {
        "a__b": {"zero_to_one": 1, "one_to_zero": 1},
        "a__c": {"zero_to_one": 0, "one_to_zero": 0},
    }


def test_oracle_hash_framing_is_explicit_and_label_sensitive() -> None:
    item_digest = hashlib.sha256(b"value").hexdigest()
    observed = hashlib.sha256()
    update_aggregate_hash(observed, "kernel::presentation", item_digest)

    expected = hashlib.sha256()
    expected.update(b"kernel::presentation\0")
    expected.update(item_digest.encode("ascii"))
    expected.update(b"\n")
    assert observed.hexdigest() == expected.hexdigest()

    other = hashlib.sha256()
    update_aggregate_hash(other, "other-label", item_digest)
    assert other.hexdigest() != observed.hexdigest()


def test_frozen_spec_distributions_are_self_closing() -> None:
    spec = json.loads(Path(DEFAULT_SPEC).read_text(encoding="utf-8"))
    validate_spec(spec)

    expected = spec["expected"]
    assert (
        sum(spec["expected_candidate_changes_by_presentation"].values())
        == expected["changed_candidates"]
    )
    assert (
        sum(item["count"] for item in spec["expected_mask_transitions"])
        == expected["changed_candidates"]
    )
