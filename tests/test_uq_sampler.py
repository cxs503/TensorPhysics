"""Tests for the deterministic, hash-pinned sampler."""
from __future__ import annotations

import numpy as np
import pytest

from tensorphysics.uq.sample_spec import Marginal, SampleSpec
from tensorphysics.uq.sampler import RULES, compute_sample_hash, draw_unit, sample


@pytest.fixture()
def spec() -> SampleSpec:
    return SampleSpec(
        (
            Marginal.uniform("a", -1.0, 1.0),
            Marginal.normal("b", 0.0, 1.0),
            Marginal.uniform("c", 0.0, 1.0),
        )
    )


def test_rules_are_supported(spec: SampleSpec) -> None:
    for rule in RULES:
        ss = sample(spec, 256, rule=rule, seed=5)
        assert ss.points.shape == (256, 3)
        assert ss.hash and len(ss.hash) == 64
        assert ss.rule == rule


def test_draw_unit_validation() -> None:
    with pytest.raises(ValueError):
        draw_unit(0, 2)
    with pytest.raises(ValueError):
        draw_unit(4, 0)
    with pytest.raises(ValueError):
        draw_unit(4, 2, rule="halton")


def test_same_seed_same_points_and_hash(spec: SampleSpec) -> None:
    a = sample(spec, 512, seed=12345)
    b = sample(spec, 512, seed=12345)
    assert np.array_equal(a.unit, b.unit)
    assert np.array_equal(a.points, b.points)
    assert a.hash == b.hash
    c = sample(spec, 512, seed=12346)
    assert c.hash != a.hash


def test_hash_covers_rule_and_points(spec: SampleSpec) -> None:
    a = sample(spec, 128, rule="sobol", seed=1)
    b = sample(spec, 128, rule="lhs", seed=1)
    assert a.hash != b.hash
    mutated = a.points.copy()
    mutated[0, 0] += 1.0e-12
    h = compute_sample_hash(
        spec_hash=spec.fingerprint(), rule=a.rule, seed=a.seed, points=mutated
    )
    assert h != a.hash


def test_subset_has_its_own_hash(spec: SampleSpec) -> None:
    ss = sample(spec, 64, seed=9)
    part = ss.subset(np.arange(0, 32))
    assert part.n == 32
    assert part.hash != ss.hash
    assert np.array_equal(part.points, ss.points[:32])


def test_sobol_design_is_balanced(spec: SampleSpec) -> None:
    ss = sample(spec, 1024, seed=17)
    assert ss.balance_exponent == 10
    # a scrambled Sobol' net of 1024 points is far more balanced than MC
    assert np.all(np.abs(ss.unit.mean(axis=0) - 0.5) < 0.01)
    ss_odd = sample(spec, 1000, seed=17)
    assert ss_odd.balance_exponent is None


def test_metadata_is_json_serialisable(spec: SampleSpec) -> None:
    import json

    d = sample(spec, 32, seed=2).as_dict()
    text = json.dumps(d)
    assert d["spec_hash"] == spec.fingerprint()
    assert d["n"] == 32 and d["dim"] == 3
    assert "point_digest" in d and len(d["point_digest"]) == 64
    assert json.loads(text)["hash"] == d["hash"]