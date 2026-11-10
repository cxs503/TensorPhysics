"""C1 - modal frequencies of the torsional chain against closed forms.

Every reference here is algebra (see ``tensorphysics.torsional_chain.theory``),
not another simulation, so a green test is a physical-consistency statement and
not a self-agreement statement.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tensorphysics.torsional_chain.acceptance import c1_natural_frequencies
from tensorphysics.torsional_chain.drivetrain import (DrivetrainChain, ShaftSegment,
                                                      polar_section_modulus,
                                                      standard_chain,
                                                      uniform_segment_stiffness)
from tensorphysics.torsional_chain.theory import (reflected_inertia,
                                                  three_inertia_chain_frequencies,
                                                  two_inertia_fixed_frequency,
                                                  two_inertia_free_frequency)

G = 79.3e9


@pytest.fixture(scope="module")
def c1():
    return c1_natural_frequencies()


def test_two_inertia_grounded_matches_closed_form():
    j1, k = 1.2e3, uniform_segment_stiffness(6.0, 0.42, 0.18, G)
    chain = DrivetrainChain()
    a = chain.add_node("propeller", j1)
    b = chain.add_node("motor", 8.0e2, fixed=True)
    chain.add_shaft(ShaftSegment("shaft1", k, 0.0, a, b, 0.42, 0.18))
    sim = chain.modal_frequencies()[0]
    ref = two_inertia_fixed_frequency(k, j1) / (2.0 * math.pi)
    assert abs(sim - ref) / ref < 1.0e-9


def test_two_inertia_free_free_matches_closed_form():
    j1, j2, k = 1.2e3, 8.0e2, uniform_segment_stiffness(6.0, 0.42, 0.18, G)
    chain = DrivetrainChain()
    a = chain.add_node("propeller", j1)
    b = chain.add_node("motor", j2)
    chain.add_shaft(ShaftSegment("shaft1", k, 0.0, a, b, 0.42, 0.18))
    sim = chain.modal_frequencies()[0]
    ref = two_inertia_free_frequency(k, j1, j2) / (2.0 * math.pi)
    assert abs(sim - ref) / ref < 1.0e-9
    # a free-free chain has exactly one non-zero mode plus the rigid-body zero mode
    assert chain.modal_frequencies().size == 1


def test_three_inertia_chain_matches_characteristic_polynomial():
    k1 = uniform_segment_stiffness(5.0, 0.40, 0.16, G)
    k2 = uniform_segment_stiffness(2.0, 0.25, 0.0, G)
    jj = (9.0e2, 5.0e2, 1.5e3)
    chain = DrivetrainChain()
    a = chain.add_node("propeller", jj[0])
    b = chain.add_node("gear", jj[1])
    c = chain.add_node("motor", jj[2])
    chain.add_shaft(ShaftSegment("shaft1", k1, 0.0, a, b, 0.40, 0.16))
    chain.add_shaft(ShaftSegment("shaft2", k2, 0.0, b, c, 0.25, 0.0))
    sim = np.sort(chain.modal_frequencies())
    ref = np.sort(np.array(three_inertia_chain_frequencies(k1, k2, *jj)) / (2.0 * math.pi))
    assert sim.size == 2
    assert np.allclose(sim, ref, rtol=1e-9)


def test_reflected_inertia_law():
    """With a rigid mesh the propeller shaft sees ``J_motor * i**2``."""
    ratio = 4.0
    chain = standard_chain(gear_stiffness=1.0e9, mesh_k0=1.0e12, mesh_modulation=0.0)
    k = chain.shafts[0].stiffness
    effective = chain.nodes[1].inertia + reflected_inertia(chain.nodes[3].inertia, ratio)
    ref = math.sqrt(k * (1.0 / chain.nodes[0].inertia + 1.0 / effective)) / (2.0 * math.pi)
    sim = chain.modal_frequencies()[0]
    assert abs(sim - ref) / ref < 0.03


def test_polar_section_modulus_formula():
    d, di = 0.42, 0.18
    expected = math.pi * (d ** 4 - di ** 4) / (16.0 * d)
    assert polar_section_modulus(d, di) == pytest.approx(expected, rel=1e-14)
    with pytest.raises(ValueError):
        polar_section_modulus(0.1, 0.1)


def test_uniform_segment_stiffness_formula():
    jp = math.pi * (0.42 ** 4 - 0.18 ** 4) / 32.0
    expected = G * jp / 6.0
    assert uniform_segment_stiffness(6.0, 0.42, 0.18, G) == pytest.approx(expected, rel=1e-14)
    with pytest.raises(ValueError):
        uniform_segment_stiffness(-1.0, 0.42, 0.18, G)


def test_chained_builders_reject_bad_indices():
    chain = DrivetrainChain()
    a = chain.add_node("a", 1.0)
    with pytest.raises(IndexError):
        chain.add_shaft(ShaftSegment("s", 1.0, 0.0, a, 5, 0.1, 0.0))
    with pytest.raises(ValueError):
        chain.add_shaft(ShaftSegment("s", 1.0, 0.0, a, a, 0.1, 0.0))


def test_energy_conservation_of_undamped_chain():
    """The restoring sign is right iff the free-vibration energy does not drift."""
    k = uniform_segment_stiffness(6.0, 0.42, 0.18, G)
    chain = DrivetrainChain()
    a = chain.add_node("propeller", 1.2e3)
    b = chain.add_node("motor", 8.0e2)
    chain.add_shaft(ShaftSegment("shaft1", k, 0.0, a, b, 0.42, 0.18))
    omega_n = two_inertia_free_frequency(k, 1.2e3, 8.0e2)
    dt = (2.0 * math.pi / omega_n) / 400.0
    hist = chain.integrate(20.0 * 2.0 * math.pi / omega_n, dt,
                          lambda t, th, om: np.zeros(2),
                          theta0=[1.0e-3, 0.0])
    theta, omega = hist["theta"], hist["omega"]
    kinetic = 0.5 * (1.2e3 * omega[:, 0] ** 2 + 8.0e2 * omega[:, 1] ** 2)
    strain = 0.5 * k * (theta[:, 1] - theta[:, 0]) ** 2
    total = kinetic + strain
    assert abs(total[-1] - total[0]) / total[0] < 1.0e-6


def test_stress_recovery_matches_torque_over_section_modulus():
    k = uniform_segment_stiffness(6.0, 0.42, 0.18, G)
    chain = DrivetrainChain()
    a = chain.add_node("propeller", 1.2e3)
    b = chain.add_node("motor", 8.0e2, fixed=True)
    seg = chain.add_shaft(ShaftSegment("shaft1", k, 0.0, a, b, 0.42, 0.18))
    theta = np.array([2.0e-3, 0.0])
    omega = np.zeros(2)
    torque = seg.torque(theta, omega)
    assert seg.stress(theta, omega) == pytest.approx(abs(torque) / seg.section_modulus, rel=1e-14)


def test_c1_acceptance_all_three_configurations_within_3pct(c1):
    assert c1["criterion"] == "C1"
    assert c1["pass"] is True
    assert c1["worst_relative_error"] <= 0.03
    names = {case["configuration"] for case in c1["cases"]}
    assert {"two_inertia_grounded", "two_inertia_free_free"} <= names
    assert any("three_inertia_free_free" in n for n in names)