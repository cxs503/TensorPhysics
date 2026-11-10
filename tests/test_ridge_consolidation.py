"""Tests for the ice-ridge consolidation thermodynamics module (IN13-W3-N05).

Every test is a physics test: an analytic identity, a comparison against an
independent numerical method, or a limiting case of a previously verified law.
"""

from __future__ import annotations

import math

import pytest

from tensorphysics.ice.resistance import (
    DomainLimits,
    ShipGeometry,
    level_ice_resistance,
    ridge_resistance,
)
from tensorphysics.ice_contact.stefan import neumann_lambda, solve_stefan_1d
from tensorphysics.ice_mechanics.constitutive import compressive_strength
from tensorphysics.ice_mechanics.ridge_consolidation import (
    K_BRINE,
    K_ICE,
    L_FUSION,
    RHO_ICE,
    ConsolidationState,
    consolidation_history,
    consolidation_state,
    consolidation_time_to_degree,
    consolidation_time_to_thickness,
    consolidated_layer_temperature,
    consolidated_strength,
    keel_properties,
    keel_resistance_coupled,
    maxwell_eucken_conductivity,
    quasi_steady_constant,
    ridge_total_resistance,
    sail_resistance_coupled,
    solid_ice_strength,
    stefan_constant,
    stefan_number,
)
from tensorphysics.ice_mechanics.ridge_verification import run_all_criteria

DAY = 86_400.0
SHIP = ShipGeometry()


# ── (a) effective properties of the porous keel ────────────────────────


def test_effective_conductivity_geometric_mean_and_bounds():
    for phi in (0.30, 0.35, 0.40, 0.45, 0.50):
        props = keel_properties(phi)
        # closed form of the model
        assert props.k_eff == pytest.approx(K_ICE ** (1 - phi) * K_BRINE**phi, rel=1e-14)
        # mixing law must sit inside the Voigt/Reuss bracket
        assert props.k_reuss < props.k_eff < props.k_voigt
        # volumetric averages are exact mixture rules
        assert props.rho_eff == pytest.approx(
            (1 - phi) * RHO_ICE + phi * 1025.0, rel=1e-14
        )
        assert props.latent_vol == pytest.approx(phi * RHO_ICE * L_FUSION, rel=1e-14)
        assert props.kappa_eff == pytest.approx(props.k_eff / props.cvol, rel=1e-14)


def test_effective_conductivity_monotone_and_maxwell_bound():
    values = [keel_properties(p).k_eff for p in (0.30, 0.35, 0.40, 0.45, 0.50)]
    assert all(a > b for a, b in zip(values, values[1:]))
    # the Maxwell-Eucken bound of an ice matrix with spherical pores is an upper
    # bound for the geometric mean and a lower bound for the Voigt average
    for phi in (0.3, 0.4, 0.5):
        props = keel_properties(phi)
        assert props.k_reuss < props.k_eff < props.k_maxwell < props.k_voigt
        assert props.k_maxwell == pytest.approx(maxwell_eucken_conductivity(phi), rel=1e-14)


def test_porosity_range_is_enforced():
    with pytest.raises(ValueError):
        keel_properties(0.20)
    with pytest.raises(ValueError):
        keel_properties(0.60)
    with pytest.raises(ValueError):
        keel_properties(-0.1)


def test_freezing_point_from_liquidus():
    props = keel_properties(0.4, salinity=32.0)
    assert props.freezing_point == pytest.approx(-0.054 * 32.0, rel=1e-12)


# ── (b) modified Stefan law ────────────────────────────────────────────


def test_stefan_constant_matches_independent_bisection():
    for ste in (0.02, 0.1, 0.3, 0.5346, 0.9):
        # independent solve of lambda exp(lambda^2) erf(lambda) = Ste/sqrt(pi)
        target = ste / math.sqrt(math.pi)
        lo, hi = 0.0, 3.0
        for _ in range(300):
            mid = 0.5 * (lo + hi)
            f = mid * math.exp(mid * mid) * math.erf(mid) - target
            if f < 0.0:
                lo = mid
            else:
                hi = mid
        lam = 0.5 * (lo + hi)
        assert stefan_constant(ste) == pytest.approx(2.0 * lam, rel=1e-10)
        assert stefan_constant(ste) == pytest.approx(2.0 * neumann_lambda(ste), rel=1e-12)


def test_quasi_steady_is_accurate_only_for_small_stefan():
    for ste in (0.02, 0.05, 0.1):
        err = abs(quasi_steady_constant(ste) - stefan_constant(ste)) / stefan_constant(ste)
        assert err < 0.02
    # documented degradation for large Ste (heat stored in the frozen layer)
    err = abs(quasi_steady_constant(0.5346) - stefan_constant(0.5346)) / stefan_constant(0.5346)
    assert 0.05 < err < 0.10


def test_stefan_number_scales_with_temperature_and_latent_heat():
    props = keel_properties(0.4)
    ste_20 = stefan_number(20.0, props)
    ste_40 = stefan_number(40.0, props)
    assert ste_40 == pytest.approx(2.0 * ste_20, rel=1e-14)
    assert ste_20 == pytest.approx(props.cvol * 20.0 / props.latent_vol, rel=1e-14)
    # Only the pore water freezes.  Raising the porosity from 0.4 to 0.5 grows the
    # latent heat per gross volume by 25 % but the composite volumetric heat
    # capacity by only 8 % (brine has the larger rho*cp), so Ste *decreases*:
    assert stefan_number(20.0, keel_properties(0.5)) < ste_20
    # ... and with k_eff falling as well (kappa_eff = k_eff/C_vol drops by 19 %),
    # a more porous keel consolidates more slowly at equal time:
    slow = consolidation_state(30 * DAY, keel_depth=8.0, air_temperature=-25.0, porosity=0.5)
    fast = consolidation_state(30 * DAY, keel_depth=8.0, air_temperature=-25.0, porosity=0.3)
    assert slow.h_upper < fast.h_upper


def test_front_grows_as_sqrt_of_time():
    h1 = consolidation_state(10 * DAY, keel_depth=8.0, air_temperature=-25.0).h_upper
    h4 = consolidation_state(40 * DAY, keel_depth=8.0, air_temperature=-25.0).h_upper
    assert h4 / h1 == pytest.approx(2.0, rel=1e-12)


def test_front_matches_independent_enthalpy_solver():
    props = keel_properties(0.40)
    t_end = 60 * DAY
    out = solve_stefan_1d(
        length=20.0,
        n_cells=600,
        t_end=t_end,
        melting_temperature=props.freezing_point,
        wall_temperature=-25.0,
        initial_temperature=props.freezing_point,
        rho=1.0,
        c=props.cvol,
        k=props.k_eff,
        latent_heat=props.latent_vol,
        mode="freezing",
        far_boundary="adiabatic",
        sample_times=[t_end],
    )
    analytic = consolidation_state(
        t_end, keel_depth=20.0, air_temperature=-25.0
    ).h_upper
    assert out["front"][0] == pytest.approx(analytic, rel=0.05)


def test_two_fronts_mirror_symmetric():
    state = consolidation_state(
        30 * DAY, keel_depth=8.0, air_temperature=-20.0, lower_boundary_temperature=-20.0
    )
    assert state.h_upper == pytest.approx(state.h_lower, rel=1e-12)
    assert state.h_consolidated == pytest.approx(2.0 * state.h_upper, rel=1e-12)
    # with the ocean at the pore-water liquidus the lower front cannot grow
    realistic = consolidation_state(30 * DAY, keel_depth=8.0, air_temperature=-20.0)
    assert realistic.h_lower == 0.0
    assert "LOWER_FRONT_STAGNANT" in realistic.flags


def test_supercooled_water_drives_the_lower_front():
    base = consolidation_state(30 * DAY, keel_depth=8.0, air_temperature=-25.0)
    supercooled = consolidation_state(
        30 * DAY,
        keel_depth=8.0,
        air_temperature=-25.0,
        lower_boundary_temperature=base.lower_boundary_temperature - 0.02,
    )
    assert supercooled.h_lower > 0.0
    # sqrt(dT) scaling of the similarity front: 0.02 K against 23.3 K
    expected = math.sqrt(0.02 / (base.props.freezing_point - (-25.0)))
    assert supercooled.h_lower / supercooled.h_upper < 0.05
    assert supercooled.h_lower / supercooled.h_upper > 0.5 * expected


def test_consolidation_closes_at_the_keel_depth():
    state = consolidation_state(
        400 * DAY, keel_depth=0.5, air_temperature=-40.0, lower_boundary_temperature=-40.0
    )
    assert state.h_consolidated == pytest.approx(0.5, rel=1e-12)
    assert state.degree == pytest.approx(1.0, rel=1e-12)
    assert "CONSOLIDATION_CLOSED" in state.flags


def test_invalid_inputs_are_rejected():
    with pytest.raises(ValueError):
        consolidation_state(-1.0, keel_depth=8.0, air_temperature=-20.0)
    with pytest.raises(ValueError):
        consolidation_state(10.0, keel_depth=8.0, air_temperature=5.0)
    with pytest.raises(ValueError):
        consolidation_state(10.0, keel_depth=-1.0, air_temperature=-20.0)
    with pytest.raises(ValueError):
        keel_resistance_coupled(8.0, 1.0, SHIP, degree=1.5, sigma_c=1e6)
    with pytest.raises(ValueError):
        consolidated_strength(-0.1, 0.4, -10.0)


def test_snow_cover_delays_consolidation():
    bare = consolidation_state(30 * DAY, keel_depth=8.0, air_temperature=-25.0).h_upper
    snowy = consolidation_state(
        30 * DAY, keel_depth=8.0, air_temperature=-25.0, snow_thickness=0.3
    ).h_upper
    assert snowy < bare
    # 0.3 m of snow at k=0.25 is 1.2 m^2K/W, i.e. a resistance comparable to ~1.5 m
    # of rubble ice: the front must be substantially reduced but not stalled
    assert 0.3 * bare < snowy < 0.9 * bare


# ── (c) strength evolution ─────────────────────────────────────────────


def test_strength_endpoints_and_monotonicity():
    t_mean = consolidated_layer_temperature(-25.0, 0.40)
    sigma_ice = solid_ice_strength(t_mean)
    assert consolidated_strength(0.0, 0.40, t_mean) == 0.0
    assert consolidated_strength(1.0, 0.40, t_mean) == pytest.approx(
        sigma_ice * 0.60, rel=1e-12
    )
    vals = [consolidated_strength(c, 0.40, t_mean) for c in (0.1, 0.2, 0.4, 0.6, 0.8, 1.0)]
    assert all(a < b for a, b in zip(vals, vals[1:]))


def test_strength_exponent_sensitivity():
    t_mean = -13.0
    s1 = consolidated_strength(0.5, 0.4, t_mean, exponent=1.0)
    s2 = consolidated_strength(0.5, 0.4, t_mean, exponent=0.5)
    s3 = consolidated_strength(0.5, 0.4, t_mean, exponent=2.0)
    assert s2 > s1 > s3
    assert consolidated_strength(1.0, 0.4, t_mean, exponent=2.0) == pytest.approx(
        consolidated_strength(1.0, 0.4, t_mean, exponent=1.0), rel=1e-12
    )


def test_solid_ice_strength_increases_with_cold():
    warm = solid_ice_strength(-5.0)
    cold = solid_ice_strength(-30.0)
    assert cold > warm > 0.0
    # brine-volume reduction is monotone: colder ice is less porous
    from tensorphysics.ice_mechanics.thermodynamics import brine_volume_fraction

    assert brine_volume_fraction(8.0, -5.0) > brine_volume_fraction(8.0, -30.0)
    # cross-check the constitutive model is the one being used
    vb = brine_volume_fraction(8.0, -10.0)
    assert solid_ice_strength(-10.0) == pytest.approx(float(compressive_strength(-10.0, vb)), rel=1e-12)


def test_consolidated_strength_inside_published_ridge_band():
    t_mean = consolidated_layer_temperature(-25.0, 0.40)
    sigma_ice = solid_ice_strength(t_mean)
    ratios = {c: consolidated_strength(c, 0.40, t_mean) / sigma_ice for c in (0.3, 0.5, 0.8)}
    assert ratios[0.3] == pytest.approx(0.18, rel=1e-12)
    assert 0.2 <= ratios[0.5] <= 0.5
    assert 0.2 <= ratios[0.8] <= 0.5


# ── (d) consolidation -> resistance coupling ───────────────────────────


def test_zero_degree_recovers_the_existing_ridge_law():
    reference = ridge_resistance(3.0, 1.0, SHIP, porosity=0.40, strict=False)
    coupled = keel_resistance_coupled(3.0, 1.0, SHIP, degree=0.0, sigma_c=1.0e6, porosity=0.40)
    assert coupled["R_keel_N"] == pytest.approx(reference["R_total_N"], rel=1e-12)
    assert coupled["R_plate_N"] == 0.0


def test_plate_term_reuses_lindqvist():
    ref = level_ice_resistance(0.8, 1.0, SHIP, sigma_c=2.5e6)
    out = keel_resistance_coupled(
        5.0, 1.0, SHIP, degree=0.16, sigma_c=2.5e6, porosity=0.999
    )
    assert out["h_consolidated_m"] == pytest.approx(0.8, rel=1e-12)
    assert out["R_plate_N"] == pytest.approx(ref["R_total_N"], rel=1e-12)


def test_keel_resistance_monotone_in_degree():
    values = []
    for degree in (0.0, 0.1, 0.2, 0.4, 0.6):
        sigma_c = consolidated_strength(degree, 0.4, -13.4)
        values.append(
            keel_resistance_coupled(5.0, 1.0, SHIP, degree=degree, sigma_c=sigma_c)["R_keel_N"]
        )
    assert all(a < b for a, b in zip(values, values[1:]))


def test_ridge_total_is_sail_plus_keel():
    out = ridge_total_resistance(8.0, 1.0, SHIP, degree=0.3, air_temperature=-25.0)
    assert out["R_ridge_N"] == pytest.approx(out["R_sail_N"] + out["R_keel_N"], rel=1e-15)
    assert out["sail_height_m"] == pytest.approx(1.6, rel=1e-14)
    assert out["sigma_c_Pa"] > 0.0


def test_sail_blend_endpoints():
    rubble = sail_resistance_coupled(2.0, 1.0, SHIP, degree=0.0)
    bonded = sail_resistance_coupled(2.0, 1.0, SHIP, degree=1.0)
    assert rubble["R_sail_bending_N"] == 0.0
    assert bonded["R_sail_rubble_N"] == 0.0
    assert rubble["R_sail_N"] == pytest.approx(rubble["R_sail_rubble_N"], rel=1e-15)
    assert bonded["R_sail_N"] == pytest.approx(bonded["R_sail_bending_N"], rel=1e-15)


def test_out_of_domain_requests_are_flagged_not_capped():
    sigma_c = consolidated_strength(0.3, 0.4, -13.4)
    deep = keel_resistance_coupled(20.0, 1.0, SHIP, degree=0.3, sigma_c=sigma_c)
    assert any(f.startswith("KEEL_BEYOND_RIDGE_LAW_LIMIT") for f in deep["flags"])
    # the deep keel is reported at its requested depth, not truncated
    assert deep["keel_depth_m"] == 20.0
    thick_plate = keel_resistance_coupled(
        10.0, 1.0, SHIP, degree=0.9, sigma_c=sigma_c, limits=DomainLimits()
    )
    assert any(f.startswith("PLATE_BEYOND_LEVEL_ICE_LIMIT") for f in thick_plate["flags"])


# ── (e) time scales ────────────────────────────────────────────────────


def test_time_to_thickness_round_trip():
    for temp in (-10.0, -25.0, -40.0):
        for h in (0.3, 0.5, 1.0):
            t = consolidation_time_to_thickness(h, air_temperature=temp)
            back = consolidation_state(t, keel_depth=1000.0, air_temperature=temp).h_upper
            assert back == pytest.approx(h, rel=1e-9)
    # snow case inverts the series-resistance closed form
    t = consolidation_time_to_thickness(0.5, air_temperature=-25.0, snow_thickness=0.3)
    back = consolidation_state(
        t, keel_depth=1000.0, air_temperature=-25.0, snow_thickness=0.3
    ).h_upper
    assert back == pytest.approx(0.5, rel=1e-9)


def test_time_scales_span_days_to_weeks():
    fast = consolidation_time_to_degree(0.5, keel_depth=1.0, air_temperature=-40.0) / DAY
    slow = consolidation_time_to_degree(0.5, keel_depth=8.0, air_temperature=-10.0) / DAY
    assert 1.0 < fast < 10.0            # thin keel, cold: days
    assert 30.0 < slow < 5000.0         # thick keel, warm: weeks to months
    assert slow > fast
    # full consolidation of a thick keel is a multi-season process, not weeks
    full = consolidation_time_to_degree(1.0, keel_depth=10.0, air_temperature=-10.0) / DAY
    assert full > 5 * 365.0


def test_history_is_monotone():
    times = [i * 5 * DAY for i in range(1, 13)]
    history = consolidation_history(times, keel_depth=8.0, air_temperature=-25.0)
    degrees = [s.degree for s in history]
    assert all(a < b for a, b in zip(degrees, degrees[1:]))
    assert all(isinstance(s, ConsolidationState) for s in history)


# ── (f) the acceptance harness itself ──────────────────────────────────


def test_acceptance_criteria_all_pass():
    report = run_all_criteria()
    assert report["n_criteria"] == 9
    failed = [c["criterion"] for c in report["criteria"] if not c["passed"]]
    assert failed == []
    assert report["all_passed"] is True