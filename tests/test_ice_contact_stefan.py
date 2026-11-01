"""Stefan moving-boundary reference solutions and the enthalpy solver."""

from math import exp, erf, sqrt, pi

import pytest

from tensorphysics.ice_contact.properties import IceWaterProperties
from tensorphysics.ice_contact.stefan import (
    neumann_front,
    neumann_lambda,
    neumann_lambda_small_stefan,
    quasi_steady_front,
    solidification_time,
    solve_stefan_1d,
)

PROPS = IceWaterProperties()


def _freezing_run(n_cells: int, t_end: float = 5.0e-4) -> dict:
    return solve_stefan_1d(
        length=4.0e-5,
        n_cells=n_cells,
        t_end=t_end,
        melting_temperature=PROPS.melting_temperature,
        wall_temperature=PROPS.melting_temperature - 10.0,
        initial_temperature=PROPS.melting_temperature,
        rho=PROPS.rho_water,
        c=PROPS.c_water,
        k=PROPS.k_water,
        latent_heat=PROPS.latent_heat_fusion,
        mode="freezing",
        far_boundary="adiabatic",
        sample_times=[t_end * f for f in (0.2, 0.4, 0.6, 0.8, 1.0)],
    )


def test_neumann_lambda_satisfies_the_transcendental_equation():
    for stefan_number in (0.01, 0.1255, 1.0, 5.0):
        lam = neumann_lambda(stefan_number)
        residual = lam * exp(lam * lam) * erf(lam) - stefan_number / sqrt(pi)
        assert abs(residual) < 1e-9


def test_neumann_lambda_matches_the_small_stefan_asymptote():
    for stefan_number in (1e-4, 1e-3):
        lam = neumann_lambda(stefan_number)
        asymptote = neumann_lambda_small_stefan(stefan_number)
        assert abs(lam - asymptote) / asymptote < 0.01


def test_enthalpy_solver_reproduces_the_neumann_front():
    """Refreezing front vs X(t) = 2 lambda sqrt(alpha t): error < 5 %."""
    stefan_number = PROPS.c_water * 10.0 / PROPS.latent_heat_fusion
    lam = neumann_lambda(stefan_number)
    alpha = PROPS.alpha_water
    solution = _freezing_run(300)
    for time, front in zip(solution["times"], solution["front"]):
        reference = neumann_front(time, alpha, lam)
        assert abs(front - reference) / reference < 0.05


def test_enthalpy_solver_front_error_converges_with_the_grid():
    stefan_number = PROPS.c_water * 10.0 / PROPS.latent_heat_fusion
    lam = neumann_lambda(stefan_number)
    alpha = PROPS.alpha_water
    errors = []
    for cells in (150, 300):
        solution = _freezing_run(cells)
        reference = [neumann_front(t, alpha, lam) for t in solution["times"]]
        errors.append(max(abs(f - r) / r for f, r in zip(solution["front"], reference)))
    assert errors[1] < errors[0]
    assert errors[1] < 0.05


def test_isothermal_melting_wall_is_rejected():
    with pytest.raises(ValueError):
        solve_stefan_1d(
            length=2.0e-5,
            n_cells=100,
            t_end=1.0e-4,
            melting_temperature=PROPS.melting_temperature,
            wall_temperature=PROPS.melting_temperature,
            initial_temperature=PROPS.melting_temperature - 10.0,
            rho=PROPS.rho_ice,
            c=PROPS.c_ice,
            k=PROPS.k_ice,
            latent_heat=PROPS.latent_heat_fusion,
            mode="melting",
            far_boundary="dirichlet",
        )


def test_quasi_steady_front_bounds_the_exact_solution():
    """The quasi-steady front is first order in Ste; it must bracket the exact one."""
    delta_t = 10.0
    stefan_number = PROPS.c_water * delta_t / PROPS.latent_heat_fusion
    lam = neumann_lambda(stefan_number)
    time = 1.0e-3
    exact = neumann_front(time, PROPS.alpha_water, lam)
    quasi = quasi_steady_front(
        time, PROPS.k_water, PROPS.rho_water, PROPS.latent_heat_fusion, delta_t
    )
    assert quasi == pytest.approx(exact, rel=0.2)
    freeze_time = solidification_time(
        1.0e-5, PROPS.k_water, PROPS.rho_water, PROPS.latent_heat_fusion, delta_t
    )
    assert 1.0e-4 < freeze_time < 1.0e-1


def test_invalid_front_arguments_are_rejected():
    with pytest.raises(ValueError):
        neumann_lambda(0.0)
    with pytest.raises(ValueError):
        neumann_front(-1.0, 1e-7, 0.3)