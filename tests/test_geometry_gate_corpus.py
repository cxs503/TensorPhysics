"""Acceptance tests for the five-hull geometry gate (IN50-N08, criterion C1).

The gate is the numeric front door of the geometry fidelity contract: CAD
integrals on one side, the nodal grid a solver would read on the other, five
deviations (L, B, Cb, nabla, S) compared with a 0.5 % tolerance.
"""
from __future__ import annotations

import numpy as np
import pytest

from tensorphysics.geometry_fidelity.gate import (
    C1_TOLERANCES,
    GeometryGate,
    bulk_carrier_hull,
    corrupted_discretisation,
    discretise,
    gate_corpus,
    series60_hull,
)


@pytest.fixture(scope="module")
def corpus():
    return gate_corpus()


@pytest.fixture(scope="module")
def gate():
    return GeometryGate()


@pytest.fixture(scope="module")
def reports(corpus, gate):
    return {name: gate.evaluate(hull) for name, hull in corpus.items()}


# -- corpus ----------------------------------------------------------------


def test_corpus_has_five_distinct_hulls(corpus):
    assert len(corpus) == 5
    assert set(corpus) == {"kcs", "kvlcc2", "suboff", "series60", "bulk_carrier"}
    ids = [h.geometry_id for h in corpus.values()]
    assert len(set(ids)) == 5


def test_new_hulls_close_their_contract_cb():
    for hull, target in ((series60_hull(), 0.600), (bulk_carrier_hull(), 0.830)):
        cb = hull.cad_integrals(240, 160)["cb"]
        assert abs(cb - target) / target < 1e-3, f"{hull.geometry_id}: Cb {cb:.4f}"


# -- C1: the five deviations ----------------------------------------------


def test_c1_length_and_breadth_are_exact(reports):
    for name, report in reports.items():
        assert report.deviations["l_rel"] < 1e-6, name
        assert report.deviations["b_rel"] < 1e-6, name


def test_c1_displacement_and_cb_within_half_percent(reports):
    for name, report in reports.items():
        assert report.deviations["nabla_rel"] < C1_TOLERANCES["nabla_rel"], name
        assert report.deviations["cb_rel"] < C1_TOLERANCES["cb_rel"], name


def test_c1_wetted_area_within_half_percent(reports):
    for name, report in reports.items():
        assert report.deviations["s_rel"] < C1_TOLERANCES["s_rel"], name


def test_c1_every_hull_passes_the_gate(reports):
    for name, report in reports.items():
        assert report.passed, f"{name}: {report.verdict} {dict(report.deviations)}"


def test_corpus_wide_worst_deviation(gate, corpus):
    payload = gate.evaluate_corpus(corpus, resolutions=((120, 48), (96, 40)))
    assert payload["verdict"] == "GATE_PASS"
    assert payload["n_hulls"] == 5
    assert max(payload["worst_deviation"].values()) < 5e-3


# -- provenance ------------------------------------------------------------


def test_provenance_digests_are_sha256_and_distinct(reports):
    digests = set()
    for name, report in reports.items():
        for key in ("cad_sha256", "grid_sha256"):
            value = report.provenance[key]
            assert len(value) == 64 and all(c in "0123456789abcdef" for c in value)
        digests.add(report.provenance["cad_sha256"])
    assert len(digests) == 5


def test_discretisation_digest_is_reproducible(gate):
    hull = series60_hull()
    first = gate.evaluate(hull)
    second = gate.evaluate(hull)
    assert first.provenance["grid_sha256"] == second.provenance["grid_sha256"]
    assert first.as_json() == second.as_json()


# -- negative controls -----------------------------------------------------


def test_gate_rejects_every_corrupted_grid(gate):
    payload = gate.negative_controls()
    assert payload["n_controls"] == payload["n_rejected"] >= 3
    for row in payload["controls"]:
        assert row["rejected"], row
        assert row["verdict"] == "GATE_REJECT"


def test_truncated_grid_fails_on_length(gate):
    hull = gate_corpus()["kcs"]
    report = gate.evaluate(hull, corrupted_discretisation(hull, "truncate_1pct"))
    assert not report.passed
    assert report.deviations["l_rel"] > 5e-3


# -- resolution behaviour --------------------------------------------------


def test_deviations_shrink_with_refinement(gate):
    hull = bulk_carrier_hull()
    coarse = gate.evaluate(hull, discretise(hull, 60, 24))
    fine = gate.evaluate(hull, discretise(hull, 180, 72))
    assert fine.deviations["nabla_rel"] < coarse.deviations["nabla_rel"]
    assert fine.deviations["s_rel"] < coarse.deviations["s_rel"]
    assert fine.deviations["nabla_rel"] < 5e-3


def test_discrete_volume_uses_actual_station_spacing(gate):
    """Cosine stations + a uniform-h rule would inflate the displacement ~20 %.

    The gate integrates with the true ``diff(xs)``; this test pins the pitfall:
    a uniform-``h`` Simpson on the same nodal areas must be visibly different.
    """
    hull = gate_corpus()["kcs"]
    disc = discretise(hull, 120, 48)
    areas = disc.section_areas()
    h_uniform = disc.length / disc.n_x
    w = np.ones(disc.n_x + 1)
    w[1:-1:2], w[2:-1:2] = 4.0, 2.0
    bad = h_uniform / 3.0 * float(np.dot(w, areas))
    good = disc.volume()
    assert abs(bad - good) / good > 0.1
    assert abs(good - hull.cad_volume_slice(240, 160)) / good < 5e-3