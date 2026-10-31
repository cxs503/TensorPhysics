"""Tests for the modal layer of the ice-induced vibration framework."""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorphysics.ice_vibration.modal import (
    FREE_FREE_BETA_L,
    HullParameters,
    ModalData,
    free_free_beam_frequency,
    free_free_beam_mode_shape,
    free_free_beam_modal_data,
    modal_data_from_fem_extraction,
    rayleigh_coefficients,
)


def make_hull(length=130.0, breadth=28.0, draught=8.4, cb=0.62):
    return HullParameters(
        length_m=length,
        breadth_m=breadth,
        draught_m=draught,
        block_coefficient=cb,
        displacement_kg=cb * length * breadth * draught * 1025.0,
    )


class TestHullParameters:
    def test_block_coefficient_round_trip(self):
        hull = make_hull()
        assert hull.cb_from_displacement == pytest.approx(0.62, rel=1e-12)
        report = hull.verify()
        assert report["consistent"] is True
        assert report["relative_error"] < 1e-12

    def test_inconsistent_geometry_is_flagged(self):
        hull = HullParameters(
            length_m=130.0, breadth_m=28.0, draught_m=8.4, block_coefficient=0.62, displacement_kg=1.0e7
        )
        report = hull.verify()
        assert report["consistent"] is False
        assert report["relative_error"] > 0.02

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"length_m": 0.0},
            {"breadth_m": -1.0},
            {"draught_m": float("nan")},
            {"block_coefficient": 1.5},
        ],
    )
    def test_rejects_invalid_principal_particulars(self, kwargs):
        payload = dict(length_m=130.0, breadth_m=28.0, draught_m=8.4, block_coefficient=0.62,
                       displacement_kg=1.9e7)
        payload.update(kwargs)
        with pytest.raises(ValueError):
            HullParameters(**payload)


class TestFreeFreeBeamReference:
    def test_frequency_matches_closed_form(self):
        length, ei, m_prime = 130.0, 4.849e12, 1.49e5
        freq = free_free_beam_frequency(FREE_FREE_BETA_L[0], length, ei, m_prime)
        expected = FREE_FREE_BETA_L[0] ** 2 / (2.0 * math.pi * length**2) * math.sqrt(ei / m_prime)
        assert freq == pytest.approx(expected, rel=1e-12)
        assert 1.1 < freq < 1.3  # plausible first vertical bending frequency

    def test_mode_shape_is_unit_maximum_and_symmetric(self):
        x = np.linspace(0.0, 130.0, 201)
        phi, phi_pp = free_free_beam_mode_shape(FREE_FREE_BETA_L[0], x, 130.0)
        assert np.max(np.abs(phi)) == pytest.approx(1.0, rel=1e-12)
        # free-free first mode is symmetric about midship
        assert phi[0] == pytest.approx(phi[-1], rel=1e-3)
        corr = float(np.corrcoef(phi, phi[::-1])[0, 1])
        assert corr > 0.999
        assert phi_pp.shape == phi.shape

    def test_modal_data_frequency_ratio_matches_free_free_roots(self):
        modal = free_free_beam_modal_data(make_hull(), 4.849e12, n_modes=2)
        ratio = modal.frequencies_hz[1] / modal.frequencies_hz[0]
        expected = (FREE_FREE_BETA_L[1] / FREE_FREE_BETA_L[0]) ** 2
        assert ratio == pytest.approx(expected, rel=1e-9)

    def test_second_mode_has_zero_midship_curvature(self):
        modal = free_free_beam_modal_data(make_hull(), 4.849e12, n_modes=2)
        curv = np.abs(np.asarray(modal.mode_shape_midship_curvature))
        assert curv[1] < 1e-12 * curv[0]

    def test_modal_masses_positive_and_of_order_half_hull_mass(self):
        hull = make_hull()
        modal = free_free_beam_modal_data(hull, 4.849e12, n_modes=2)
        for mass in modal.modal_masses_kg:
            assert 0.2 * hull.displacement_kg < mass < 0.8 * hull.displacement_kg
        assert len(modal.generalized_stiffness()) == 2
        assert np.all(np.asarray(modal.generalized_stiffness()) > 0.0)


class TestModalDataValidation:
    def test_rejects_non_increasing_frequencies(self):
        with pytest.raises(ValueError):
            ModalData(frequencies_hz=(2.0, 1.0), damping_ratios=(0.02, 0.02), modal_masses_kg=(1.0, 1.0))

    def test_rejects_invalid_damping(self):
        with pytest.raises(ValueError):
            ModalData(frequencies_hz=(1.0,), damping_ratios=(1.2,), modal_masses_kg=(1.0,))

    def test_rejects_mismatched_lengths(self):
        with pytest.raises(ValueError):
            ModalData(frequencies_hz=(1.0, 2.0), damping_ratios=(0.02,), modal_masses_kg=(1.0, 2.0))

    def test_truncation_keeps_leading_modes(self):
        modal = free_free_beam_modal_data(make_hull(), 4.849e12, n_modes=2)
        single = modal.truncated(1)
        assert single.n_modes == 1
        assert single.frequencies_hz == (modal.frequencies_hz[0],)
        assert single.mode_shape_quarter_curvature == (modal.mode_shape_quarter_curvature[0],)


class TestRayleighDamping:
    def test_rayleigh_reproduces_requested_modal_damping(self):
        f1, z1, f2, z2 = 1.2, 0.02, 3.3, 0.03
        alpha, beta = rayleigh_coefficients(f1, z1, f2, z2)
        for freq, zeta in ((f1, z1), (f2, z2)):
            w = 2.0 * math.pi * freq
            assert alpha / (2.0 * w) + beta * w / 2.0 == pytest.approx(zeta, rel=1e-10)


class TestFemExtractionInterface:
    def test_requires_modal_mass(self):
        payload = {"modes": [{"frequency_hz": 1.2, "mode_shape_bow": 1.0}]}
        with pytest.raises(ValueError, match="modal_mass_kg"):
            modal_data_from_fem_extraction(payload)

    def test_parses_payload_and_flags_defaulted_damping(self):
        payload = {
            "solver": "TensorFEM",
            "modes": [
                {"frequency_hz": 1.2, "modal_mass_kg": 4.8e6, "mode_shape_bow": 1.0,
                 "midship_curvature_per_m": 1.05e-3},
                {"frequency_hz": 3.3, "modal_mass_kg": 4.9e6, "damping_ratio": 0.03,
                 "mode_shape_bow": -1.0, "quarter_curvature_per_m": 2.0e-3},
            ],
            "meta": {"case": "icebreaker-vertical-bending"},
        }
        modal = modal_data_from_fem_extraction(payload)
        assert modal.source == "fem"
        assert modal.frequencies_hz == (1.2, 3.3)
        assert modal.damping_ratios[0] == pytest.approx(1.0e-2)
        assert modal.metadata["damping_ratio_defaults_used"] == 1
        assert modal.mode_shape_quarter_curvature == (0.0, 2.0e-3)