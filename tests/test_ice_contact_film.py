"""Melt-film thickness ODE: steady states, transients and energy bookkeeping."""

import pytest

from tensorphysics.ice_contact.film import ContactState, FilmModel
from tensorphysics.ice_contact.pressure_melting import melt_point_depression, melt_temperature
from tensorphysics.ice_contact.properties import IceWaterProperties

PROPS = IceWaterProperties()
MODEL = FilmModel()


def test_pure_friction_melting_steady_state_matches_closed_form():
    """Ice at its melting point: every Watt melts ice, h_ss = mu p L / (rho Lf)."""
    for pressure, velocity in ((1.0e6, 1.0), (1.0e6, 4.0), (4.0e6, 2.0)):
        t_melt = melt_temperature(pressure)
        state = ContactState(
            pressure=pressure,
            velocity=velocity,
            ice_temperature=t_melt,
            structure_temperature=t_melt,
        )
        analytic = MODEL.steady_state_analytic(state, friction_coefficient_value=0.05)
        reference = (
            0.05 * pressure * state.contact_length / PROPS.latent_heat_per_volume_ice
        )
        assert analytic == pytest.approx(reference, rel=1e-9)


def test_coupled_steady_state_matches_the_quadratic_solution():
    state = ContactState(
        pressure=1.0e6,
        velocity=4.0,
        ice_temperature=263.15,
        structure_temperature=melt_temperature(1.0e6),
    )
    steady = MODEL.steady_state(state)
    assert steady["exists"]
    analytic = MODEL.steady_state_analytic(state, steady["friction_coefficient"])
    assert abs(steady["film_thickness"] - analytic) / analytic < 0.01


def test_transient_integration_relaxes_onto_the_analytic_steady_state():
    state = ContactState(
        pressure=1.0e6,
        velocity=4.0,
        ice_temperature=263.15,
        structure_temperature=melt_temperature(1.0e6),
    )
    steady = MODEL.steady_state(state)
    analytic = MODEL.steady_state_analytic(state, steady["friction_coefficient"])
    for initial in (5.0e-8, 5.0e-7, 2.0e-6):
        history = MODEL.integrate(state, initial_thickness=initial, t_end=1.0, n_samples=21)
        final = history["film_thickness"][-1]
        assert abs(final - analytic) / analytic < 0.01


def test_cold_contact_has_no_persistent_film():
    """1 MPa / 0.1 m/s / -10 degC: refreezing beats friction melting."""
    state = ContactState(
        pressure=1.0e6, velocity=0.1, ice_temperature=263.15, structure_temperature=263.15
    )
    steady = MODEL.steady_state(state)
    assert not steady["exists"]
    assert steady["dh_dt_at_floor"] < 0.0
    assert steady["regime"] == "boundary"


def test_refreezing_collapses_an_initial_film():
    state = ContactState(
        pressure=1.0e6, velocity=0.1, ice_temperature=263.15, structure_temperature=263.15
    )
    history = MODEL.integrate(state, initial_thickness=1.0e-6, t_end=5.0e-3, n_samples=51)
    assert history["film_thickness"][0] == pytest.approx(1.0e-6)
    assert history["film_thickness"][-1] <= MODEL.film_floor * 1.0000001
    assert all(b <= a for a, b in zip(history["film_thickness"], history["film_thickness"][1:]))
    assert history["dry_branch"][-1] is True


def test_energy_and_mass_bookkeeping_closes():
    """dh/dt must equal the sum of the decomposed mechanism rates minus drainage."""
    state = ContactState(
        pressure=2.0e6,
        velocity=4.0,
        ice_temperature=263.15,
        structure_temperature=melt_temperature(2.0e6),
    )
    for thickness in (1.0e-7, 5.0e-7, 2.0e-6):
        rates = MODEL.decompose(thickness, state)
        total = (
            rates["friction_melting"]
            + rates["pressure_melting"]
            + rates["passive_melting"]
            - rates["refreezing"]
            - rates["drain_rate"]
        )
        assert total == pytest.approx(MODEL.rhs(thickness, state), rel=1e-9)


def test_flux_signs_follow_the_thermal_state():
    cold = ContactState(
        pressure=1.0e6, velocity=1.0, ice_temperature=263.15, structure_temperature=263.15
    )
    warm = ContactState(
        pressure=1.0e6, velocity=1.0, ice_temperature=263.15, structure_temperature=283.15
    )
    cold_fluxes = MODEL.fluxes(1.0e-6, cold)
    warm_fluxes = MODEL.fluxes(1.0e-6, warm)
    assert cold_fluxes["q_ice"] < 0.0 and cold_fluxes["q_structure"] < 0.0
    assert warm_fluxes["q_structure"] > 0.0
    assert cold_fluxes["q_friction"] > 0.0


def test_pressure_melting_raises_the_melting_temperature_budget():
    """Higher pressure depresses T_m, which must reduce the refreezing sink."""
    low = ContactState(
        pressure=1.0e6, velocity=1.0, ice_temperature=263.15, structure_temperature=263.15
    )
    high = ContactState(
        pressure=5.0e7, velocity=1.0, ice_temperature=263.15, structure_temperature=263.15
    )
    assert MODEL.melting_temperature(high) < MODEL.melting_temperature(low)
    low_rates = MODEL.decompose(1.0e-6, low)
    high_rates = MODEL.decompose(1.0e-6, high)
    # the pressure-melting rate is proportional to the melting point depression
    depression_ratio = (
        melt_point_depression(5.0e7) / melt_point_depression(1.0e6)
    )
    assert depression_ratio > 40.0
    assert high_rates["pressure_melting"] / low_rates["pressure_melting"] == pytest.approx(
        depression_ratio, rel=1e-6
    )


def test_invalid_contact_state_is_rejected():
    with pytest.raises(ValueError):
        ContactState(pressure=0.0, velocity=1.0, ice_temperature=263.15,
                     structure_temperature=263.15)
    with pytest.raises(ValueError):
        ContactState(pressure=1.0e6, velocity=-1.0, ice_temperature=263.15,
                     structure_temperature=263.15)