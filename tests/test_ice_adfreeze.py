"""Tests for the ice/structure adfreeze and breakout module.

Run from the repository root::

    python -m pytest tests/test_ice_adfreeze.py -q

The suite mixes (i) physics verification against analytic results (Stefan
solution, cohesive-law energy, momentum theory), (ii) calibration residuals
against the Frederking (1984) strength data, and (iii) the two machine
acceptance criteria of the W3-N02 node (strength error < 15 % at four
temperatures, breakout/contact-area consistency < 10 % for three ship types).
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

_SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from tensorphysics.ice import (  # noqa: E402
    FREDERKING_ADFREEZE,
    SALINE_TO_FRESH_RATIO,
    SHIPS,
    DEFAULT_FRICTION_COEFF,
    adfreeze_strength,
    adfreeze_strength_state,
    bollard_thrust_mn,
    breakout_force_mn,
    brine_salinity,
    brine_volume_fraction,
    calibrate_reference_strength,
    cohesive_law,
    contact_band_height,
    contact_time_to_saturation,
    critical_slip,
    empirical_contact_area,
    evaluate_breakout,
    frederking_strengths,
    frederking_temperatures,
    frozen_area_fraction,
    hull_ice_contact_area,
    peel_softening,
    reference_strength_mpa,
    required_debonded_area,
    roughness_factor,
    salinity_degradation_factor,
    salinity_factor,
    seawater_freezing_point,
    stefan_frozen_thickness,
    thermal_degradation_factor,
)
from tensorphysics.ice import adfreeze as adf  # noqa: E402


# ---------------------------------------------------------------------------
# Acceptance criterion (a): strength vs Frederking data, four temperatures
# ---------------------------------------------------------------------------
class TestAdfreezeStrengthValidation:
    def test_matches_frederking_within_15_percent(self):
        """Model vs the four digitised Frederking temperatures (< 15 % error)."""
        errors = {}
        for rec in FREDERKING_ADFREEZE:
            model = adfreeze_strength(
                rec["T_C"],
                contact_time_s=1.0e5,  # saturated contact
                salinity_gkg=0.0,  # freshwater ice
                roughness_um=0.0,  # polished steel reference surface
            )
            errors[rec["T_C"]] = 100.0 * (model / rec["tau_MPa"] - 1.0)
            assert abs(errors[rec["T_C"]]) < 15.0, (rec["T_C"], model, errors)
        # the calibration spread is much tighter than the criterion
        assert max(abs(e) for e in errors.values()) < 6.0

    def test_matches_frederking_with_hull_roughness(self):
        """Same check with as-received hull roughness (Ra = 2.5 um) still < 15 %."""
        for rec in FREDERKING_ADFREEZE:
            model = adfreeze_strength(
                rec["T_C"], contact_time_s=1.0e5, salinity_gkg=0.0, roughness_um=2.5
            )
            err = 100.0 * (model / rec["tau_MPa"] - 1.0)
            assert abs(err) < 15.0, (rec["T_C"], err)

    def test_calibration_is_reproducible_from_reference_data(self):
        """Fitted (tau_sat, d_half) reproduce the module constants."""
        tau_sat, d_half, rms = calibrate_reference_strength(
            frederking_temperatures(), frederking_strengths()
        )
        assert rms < 0.05
        assert tau_sat == pytest.approx(adf.TAU_SAT_MPA, rel=1e-4)
        assert d_half == pytest.approx(adf.D_HALF_K, rel=1e-4)

    def test_strength_increases_with_undercooling(self):
        temps = [-1.0, -2.0, -5.0, -10.0, -20.0, -30.0]
        taus = [adfreeze_strength(t, 1.0e5, 0.0, 2.5) for t in temps]
        assert all(b > a for a, b in zip(taus, taus[1:]))

    def test_strength_bounded_by_ice_shear_strength(self):
        """A bond cannot be stronger than the ice it is made of."""
        assert reference_strength_mpa(200.0) <= adf.TAU_ICE_CAP_MPA + 1e-12

    def test_zero_strength_at_freezing_point(self):
        assert adfreeze_strength(0.0) == 0.0
        assert adfreeze_strength(0.5) == 0.0  # above the liquidus: no bond

    def test_state_decomposition_is_multiplicative(self):
        st = adfreeze_strength_state(-10.0, 3.0e3, 0.0, 2.5, 5.0e-3)
        product = (
            st.tau_reference_mpa * st.frozen_fraction * st.roughness_factor * st.salinity_factor
        )
        assert st.tau_adfreeze_mpa == pytest.approx(product, rel=1e-12)
        assert st.undercooling_k == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# Freezing kinetics: Stefan solution and contact-time build-up
# ---------------------------------------------------------------------------
class TestFreezingKinetics:
    def test_stefan_sqrt_time_law(self):
        """h(t) ~ sqrt(t): quadrupling the time doubles the frozen thickness."""
        h1 = stefan_frozen_thickness(10.0, 100.0)
        h4 = stefan_frozen_thickness(10.0, 400.0)
        assert h4 / h1 == pytest.approx(2.0, rel=1e-12)

    def test_stefan_matches_analytic_value(self):
        """h = sqrt(2*k*dT*t/(rho*L)) checked against a hand calculation."""
        k, rho, Lf = 2.2, 917.0, 3.34e5
        expected = np.sqrt(2.0 * k * 10.0 * 60.0 / (rho * Lf))
        assert stefan_frozen_thickness(10.0, 60.0) == pytest.approx(expected, rel=1e-12)

    def test_frozen_area_fraction_saturates(self):
        assert frozen_area_fraction(10.0, 0.0) == 0.0
        eta_short = frozen_area_fraction(10.0, 60.0)
        eta_long = frozen_area_fraction(10.0, 1.0e6)
        assert 0.0 < eta_short < eta_long
        assert eta_long > 0.999

    def test_colder_joint_freezes_faster(self):
        assert frozen_area_fraction(20.0, 300.0) > frozen_area_fraction(2.0, 300.0)

    def test_contact_time_to_saturation_inverts_kinetics(self):
        for temp in (-2.0, -5.0, -10.0, -20.0):
            t_sat = contact_time_to_saturation(temp, 0.0, 0.95)
            assert frozen_area_fraction(-temp, t_sat) == pytest.approx(0.95, rel=1e-6)


# ---------------------------------------------------------------------------
# Roughness and salinity closures
# ---------------------------------------------------------------------------
class TestSurfaceAndSalinity:
    def test_roughness_factor_limits(self):
        assert roughness_factor(0.0) == pytest.approx(1.0)
        assert roughness_factor(5.0) > 1.0
        assert roughness_factor(1.0e4) < 1.0 + adf.ROUGH_K + 1e-12
        vals = [roughness_factor(ra) for ra in (0.0, 1.0, 5.0, 20.0, 100.0)]
        assert all(b >= a for a, b in zip(vals, vals[1:]))

    def test_seawater_freezing_point(self):
        assert seawater_freezing_point(0.0) == 0.0
        assert seawater_freezing_point(32.0) == pytest.approx(-1.83, abs=0.15)
        # freezing point depression is monotone in salinity
        assert seawater_freezing_point(15.0) > seawater_freezing_point(32.0)

    def test_brine_volume_fraction(self):
        assert brine_volume_fraction(-10.0, 0.0) == 0.0
        vb = brine_volume_fraction(-10.0, 32.0)
        assert 0.15 < vb < 0.26  # published sea-ice brine volume at -10 degC
        # colder ice holds less brine
        assert brine_volume_fraction(-20.0, 32.0) < vb
        # brine salinity decreases the colder the ice
        assert brine_salinity(-20.0) > brine_salinity(-5.0)

    def test_salinity_penalty_matches_comparative_datum(self):
        ratio = salinity_factor(SALINE_TO_FRESH_RATIO["T_C"], SALINE_TO_FRESH_RATIO["salinity_gkg"])
        assert ratio == pytest.approx(SALINE_TO_FRESH_RATIO["ratio"], abs=0.08)

    def test_no_bond_above_saline_liquidus(self):
        # -1 degC seawater ice is above its liquidus (-1.83 degC) -> no bond
        assert salinity_factor(-1.0, 32.0) == 0.0
        assert adfreeze_strength(-1.0, 1.0e5, 32.0, 2.5) == 0.0


# ---------------------------------------------------------------------------
# Degradation mechanisms
# ---------------------------------------------------------------------------
class TestDegradation:
    def test_cohesive_law_energy_equals_interface_gc(self):
        """Integral of traction over slip equals G_c (energy consistency)."""
        g_c = 5.0
        tau = 0.7
        slips = np.linspace(0.0, 1.5 * critical_slip(tau, g_c), 20001)
        traction = cohesive_law(tau, slips, g_c)
        energy = np.trapezoid(traction * 1.0e6, slips)
        assert energy == pytest.approx(g_c, rel=1e-3)

    def test_peel_softening_reaches_zero(self):
        tau = 0.7
        dc = critical_slip(tau)
        assert peel_softening(tau, 0.0) == pytest.approx(1.0)
        assert peel_softening(tau, 0.5 * dc) == pytest.approx(0.5)
        assert peel_softening(tau, dc) == 0.0
        assert peel_softening(tau, 3.0 * dc) == 0.0

    def test_thermal_degradation_factors(self):
        # warmer joint -> weaker bond; colder -> stronger
        assert thermal_degradation_factor(-10.0, -5.0) < 1.0
        assert thermal_degradation_factor(-10.0, -20.0) == pytest.approx(1.5, abs=0.3)
        # melting to the liquidus destroys the bond
        assert thermal_degradation_factor(-10.0, 0.0) == 0.0

    def test_salinity_degradation_is_reversible(self):
        f_brine = salinity_degradation_factor(-10.0, 32.0, 0.0)
        f_fresh = salinity_degradation_factor(-10.0, 0.0, 32.0)
        assert f_brine > 1.0 and f_fresh < 1.0
        assert f_brine * f_fresh == pytest.approx(1.0, rel=1e-9)


# ---------------------------------------------------------------------------
# Geometry, thrust and the breakout criterion
# ---------------------------------------------------------------------------
class TestBreakoutCriterion:
    def test_contact_area_matches_closed_form_for_three_ships(self):
        """Acceptance criterion (b): geometric integration vs closed form < 10 %."""
        for ship in SHIPS:
            area = hull_ice_contact_area(ship)
            area_emp = empirical_contact_area(ship)
            dev = 100.0 * (area / area_emp - 1.0)
            assert abs(dev) < 10.0, (ship.name, area, area_emp, dev)

    def test_contact_area_responds_to_ice_thickness(self):
        ship = SHIPS[1]
        thin = hull_ice_contact_area(ship, 0.5)
        thick = hull_ice_contact_area(ship, 1.2)
        assert thick > thin

    def test_contact_band_height_limited_by_flare(self):
        ship = SHIPS[0]
        vertical = contact_band_height(ship, 1.0, 0.0)
        flared = contact_band_height(ship, 1.0, 30.0)
        assert vertical == pytest.approx(1.0)
        assert flared < vertical

    def test_bollard_thrust_momentum_theory_consistency(self):
        """Recompute T from the induced n and check the momentum-theory identity."""
        power_kw, d, count = 18000.0, 4.2, 2
        t_mn = bollard_thrust_mn(power_kw, d, count)
        p_per = power_kw * 1e3 / count
        n = (p_per / (2.0 * np.pi * 0.060 * 1025.0 * d**5)) ** (1.0 / 3.0)
        assert t_mn == pytest.approx(count * 0.45 * 1025.0 * n**2 * d**4 / 1e6, rel=1e-12)
        # more power / bigger propeller -> more thrust
        assert bollard_thrust_mn(30000.0, 4.2, 2) > t_mn
        assert bollard_thrust_mn(18000.0, 5.0, 2) > t_mn

    def test_breakout_force_is_linear_in_area_and_strength(self):
        f1 = breakout_force_mn(100.0, 0.5, 0.0, 0.0)[0]
        f2 = breakout_force_mn(200.0, 0.5, 0.0, 0.0)[0]
        f3 = breakout_force_mn(100.0, 1.0, 0.0, 0.0)[0]
        assert f2 == pytest.approx(2.0 * f1)
        assert f3 == pytest.approx(2.0 * f1)
        assert f1 == pytest.approx(50.0)  # 100 m^2 * 0.5 MPa = 50 MN

    def test_friction_term_is_small_against_adhesion(self):
        res = evaluate_breakout(SHIPS[1], -10.0, contact_time_s=1.0e5)
        assert res.friction_force_mn < 0.05 * res.breakout_force_mn

    def test_full_joint_breakout_exceeds_thrust(self):
        """Physical conclusion: a fully developed joint cannot be sheared by thrust."""
        for ship in SHIPS:
            res = evaluate_breakout(ship, -10.0, contact_time_s=1.0e5)
            assert res.margin < 0.1
            assert 0.0 <= res.critical_area_fraction < 0.05
            peeled = required_debonded_area(
                res.thrust_minus_friction_mn, res.tau_adfreeze_mpa, res.contact_area_m2
            )
            assert peeled / res.contact_area_m2 == pytest.approx(
                1.0 - res.critical_area_fraction, abs=1e-6
            )

    def test_warm_or_saline_joint_reduces_breakout_force(self):
        ship = SHIPS[0]
        cold = evaluate_breakout(ship, -20.0, contact_time_s=1.0e5)
        warm = evaluate_breakout(ship, -2.0, contact_time_s=1.0e5)
        saline = evaluate_breakout(ship, -10.0, salinity_gkg=32.0, contact_time_s=1.0e5)
        fresh = evaluate_breakout(ship, -10.0, salinity_gkg=0.0, contact_time_s=1.0e5)
        assert warm.breakout_force_mn < cold.breakout_force_mn
        assert saline.breakout_force_mn < fresh.breakout_force_mn

    def test_short_contact_time_weakens_the_bond(self):
        ship = SHIPS[0]
        fresh_joint = evaluate_breakout(ship, -10.0, contact_time_s=30.0)
        aged_joint = evaluate_breakout(ship, -10.0, contact_time_s=1.0e5)
        assert fresh_joint.tau_adfreeze_mpa < aged_joint.tau_adfreeze_mpa


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------
class TestValidation:
    @pytest.mark.parametrize(
        "call",
        [
            lambda: reference_strength_mpa(-1.0),
            lambda: stefan_frozen_thickness(-1.0, 10.0),
            lambda: stefan_frozen_thickness(1.0, -10.0),
            lambda: frozen_area_fraction(10.0, 10.0, 0.0),
            lambda: roughness_factor(-1.0),
            lambda: salinity_factor(-10.0, -5.0),
            lambda: brine_volume_fraction(-5.0, -1.0),
            lambda: seawater_freezing_point(-1.0),
            lambda: brine_salinity(0.5),
            lambda: contact_time_to_saturation(-10.0, 0.0, 1.5),
            lambda: peel_softening(0.0, 1e-6),
            lambda: peel_softening(0.5, -1e-6),
            lambda: hull_ice_contact_area(SHIPS[0], 1.0, 0.0),
            lambda: hull_ice_contact_area(SHIPS[0], 1.0, 1.5),
            lambda: contact_band_height(SHIPS[0], 0.0, 5.0),
            lambda: bollard_thrust_mn(0.0, 4.0),
            lambda: bollard_thrust_mn(1000.0, 0.0),
            lambda: bollard_thrust_mn(1000.0, 4.0, 0),
            lambda: breakout_force_mn(-1.0, 0.5),
            lambda: breakout_force_mn(10.0, -0.5),
            lambda: salinity_degradation_factor(0.5, 0.0, 0.0),
        ],
    )
    def test_invalid_inputs_raise(self, call):
        with pytest.raises(ValueError):
            call()

    def test_gc_must_be_positive(self):
        with pytest.raises(ValueError):
            critical_slip(0.5, 0.0)