"""Acceptance-criteria harness for the ice/structure contact phase-change model.

Each criterion is a *comparison*, never a self-assessment: the model under test and
an independent reference (closed form, similarity solution, or a second numerical
method) are evaluated separately and their difference is reported as a relative
error together with the threshold from the task acceptance criteria.

Criteria
--------
(a) pressure melting: RK4 integration of the Clausius-Clapeyron ODE vs the exact
    integrated form, error < 1 %; plus the slope vs the literature value
    (0.0742 K/MPa).
(b) pure friction melting: steady film thickness from the ODE fixed point vs the
    closed-form quadratic solution, error < 5 %.
(c) refreezing: 1-D enthalpy solver vs the Neumann similarity solution for the
    freezing rate, error < 5 % (melting of subcooled ice reported as well).
(d) three-mechanism competition: physical direction gates G1-G5.
(e) end-to-end case: ice/steel contact at 1 MPa, 0.1 m/s, -10 degC with film and
    friction time histories, plus two companion cases that exercise the
    friction-melting and pressure-melting branches.
(f) test suite: executed by the caller (the benchmark runner parses pytest).

The harness returns plain JSON-serialisable data; the benchmark runner writes it
to ``results.json`` so that every number quoted in the report is machine-read
from the artefact.
"""

from math import sqrt

from .adhesion import IceAdhesionModel
from .competition import direction_gates, melting_onset_velocity, melting_onset_velocity_analytic
from .film import ContactState, FilmModel
from .pressure_melting import (
    clapeyron_slope,
    integrate_melt_temperature,
    melt_point_depression,
    melt_temperature,
    pressure_melting_threshold,
)
from .properties import IceWaterProperties
from .stefan import neumann_front, neumann_lambda, solve_stefan_1d

__all__ = [
    "verify_pressure_melting",
    "verify_friction_melting_steady",
    "verify_refreezing_stefan",
    "verify_competition_gates",
    "end_to_end_case",
    "case_studies",
    "run_all_criteria",
]

LITERATURE_CLAPEYRON_SLOPE = -0.0742  # K/MPa, ice Ih melting line at the triple point


def _relative_error(model: float, reference: float) -> float:
    if reference == 0.0:
        return abs(model)
    return abs(model - reference) / abs(reference)


def verify_pressure_melting(pressures: tuple[float, ...] = (1.0e6, 1.0e7, 5.0e7, 1.0e8)) -> dict:
    """Criterion (a): Clausius-Clapeyron pressure melting vs analytic solution."""
    props = IceWaterProperties()
    rows = []
    worst = 0.0
    for pressure in pressures:
        p_rk4, t_rk4 = integrate_melt_temperature(pressure, 200, props)
        t_analytic = melt_temperature(pressure, props)
        error = _relative_error(t_rk4[-1], t_analytic)
        worst = max(worst, error)
        rows.append(
            {
                "pressure": pressure,
                "melt_temperature_rk4": t_rk4[-1],
                "melt_temperature_analytic": t_analytic,
                "relative_error": error,
                "depression": melt_point_depression(pressure, props),
                "melt_temperature_linear": props.melting_temperature + clapeyron_slope(props) * pressure,
                "steps": len(p_rk4) - 1,
            }
        )
    slope_mpa = clapeyron_slope(props) * 1.0e6
    slope_error = _relative_error(abs(slope_mpa), abs(LITERATURE_CLAPEYRON_SLOPE))
    threshold = pressure_melting_threshold(263.15, props)
    return {
        "criterion": "a_pressure_melting",
        "threshold": 0.01,
        "error": worst,
        "passed": bool(worst < 0.01 and slope_error < 0.01),
        "rows": rows,
        "slope_K_per_MPa": slope_mpa,
        "literature_slope_K_per_MPa": LITERATURE_CLAPEYRON_SLOPE,
        "slope_relative_error": slope_error,
        "melting_threshold_pressure_at_263K": threshold,
        "detail": "RK4 march of dT/dp = T dv/Lf against T(p) = T0 exp(dv p / Lf)",
    }


def verify_friction_melting_steady() -> dict:
    """Criterion (b): steady melt film from friction heating vs closed form."""
    model = FilmModel()
    row_sets = []

    # (i) pure friction melting: ice at its own melting point and a structure at
    #     the melting point, so both Stefan sinks vanish and every watt of
    #     frictional power goes into latent heat.  The closed form is then
    #     h_ss = mu p L / (c_d rho_ice L_f), independent of the sliding speed.
    worst_constant = 0.0
    cases = [(1.0e6, 1.0), (1.0e6, 4.0), (2.0e6, 2.0), (4.0e6, 8.0)]
    for pressure, velocity in cases:
        t_melt_here = melt_temperature(pressure)
        state = ContactState(
            pressure=pressure,
            velocity=velocity,
            ice_temperature=t_melt_here,
            structure_temperature=t_melt_here,
        )
        mu = 0.05
        analytic = model.steady_state_analytic(state, friction_coefficient_value=mu)
        reference = (
            mu * pressure * state.contact_length
            / (model.drainage_coefficient * model.props.latent_heat_per_volume_ice)
        )
        error = _relative_error(analytic, reference)
        worst_constant = max(worst_constant, error)
        row_sets.append(
            {
                "mode": "pure_friction_melting_closed_form",
                "pressure": pressure,
                "velocity": velocity,
                "ice_temperature": t_melt_here,
                "friction_coefficient": mu,
                "analytic_thickness": analytic,
                "reference_thickness": reference,
                "relative_error": error,
            }
        )

    # (ii) fully coupled model: ODE fixed point (bracketing) vs the closed-form
    #      quadratic evaluated at the converged friction coefficient.
    worst_coupled = 0.0
    coupled_cases = [
        (1.0e6, 1.0, 263.15),
        (1.0e6, 4.0, 263.15),
        (2.0e6, 2.0, 268.15),
        (4.0e6, 8.0, 253.15),
    ]
    for pressure, velocity, temperature in coupled_cases:
        state = ContactState(
            pressure=pressure,
            velocity=velocity,
            ice_temperature=temperature,
            structure_temperature=melt_temperature(pressure),
        )
        steady = model.steady_state(state)
        if not steady["exists"]:
            row_sets.append(
                {
                    "mode": "coupled",
                    "pressure": pressure,
                    "velocity": velocity,
                    "ice_temperature": temperature,
                    "exists": False,
                    "relative_error": None,
                }
            )
            continue
        analytic = model.steady_state_analytic(state, steady["friction_coefficient"])
        error = _relative_error(steady["film_thickness"], analytic)
        worst_coupled = max(worst_coupled, error)
        row_sets.append(
            {
                "mode": "coupled",
                "pressure": pressure,
                "velocity": velocity,
                "ice_temperature": temperature,
                "exists": True,
                "friction_coefficient": steady["friction_coefficient"],
                "ode_fixed_point": steady["film_thickness"],
                "analytic_thickness": analytic,
                "relative_error": error,
            }
        )

    # (iii) transient verification: the ODE integrated with RK4 from different
    #       initial thicknesses must converge onto the analytic steady state.
    transient = []
    worst_transient = 0.0
    state = ContactState(
        pressure=1.0e6,
        velocity=4.0,
        ice_temperature=263.15,
        structure_temperature=melt_temperature(1.0e6),
    )
    steady = model.steady_state(state)
    analytic = model.steady_state_analytic(state, steady["friction_coefficient"])
    for initial in (5.0e-8, 2.0e-7, 1.0e-6):
        history = model.integrate(state, initial_thickness=initial, t_end=2.0, n_samples=41)
        final = history["film_thickness"][-1]
        error = _relative_error(final, analytic)
        worst_transient = max(worst_transient, error)
        transient.append(
            {
                "initial_thickness": initial,
                "final_thickness": final,
                "analytic_thickness": analytic,
                "relative_error": error,
            }
        )

    worst = max(worst_constant, worst_coupled, worst_transient)
    return {
        "criterion": "b_friction_melting_steady",
        "threshold": 0.05,
        "error": worst,
        "passed": bool(worst < 0.05),
        "rows": row_sets,
        "transient": transient,
        "detail": "ODE fixed point / RK4 transient against the closed-form quadratic steady state",
    }


def verify_refreezing_stefan(
    n_cells: tuple[int, ...] = (150, 300, 600),
    t_end: float = 1.0e-3,
) -> dict:
    """Criterion (c): refreezing rate vs the Neumann similarity solution."""
    props = IceWaterProperties()
    delta_t = 10.0
    t_melt = props.melting_temperature
    wall_temperature = t_melt - delta_t
    rho, c, k, latent = props.rho_water, props.c_water, props.k_water, props.latent_heat_fusion
    stefan_number = c * delta_t / latent
    lam = neumann_lambda(stefan_number)
    alpha = k / (rho * c)
    sample_times = [t_end * f for f in (0.2, 0.4, 0.6, 0.8, 1.0)]

    rows = []
    worst = 0.0
    for cells in n_cells:
        length = 4.0e-5
        solution = solve_stefan_1d(
            length=length,
            n_cells=cells,
            t_end=t_end,
            melting_temperature=t_melt,
            wall_temperature=wall_temperature,
            initial_temperature=t_melt,
            rho=rho,
            c=c,
            k=k,
            latent_heat=latent,
            mode="freezing",
            far_boundary="adiabatic",
            sample_times=sample_times,
            dt_safety=0.4,
        )
        reference = [neumann_front(t, alpha, lam) for t in solution["times"]]
        errors = [_relative_error(x, r) for x, r in zip(solution["front"], reference)]
        worst = max(worst, max(errors))
        rows.append(
            {
                "n_cells": cells,
                "dx": solution["dx"],
                "dt": solution["dt"],
                "n_steps": solution["n_steps"],
                "times": solution["times"],
                "front_numerical": solution["front"],
                "front_neumann": reference,
                "relative_errors": errors,
                "max_relative_error": max(errors),
            }
        )

    finest = rows[-1]["max_relative_error"]
    return {
        "criterion": "c_refreezing_stefan",
        "threshold": 0.05,
        "error": finest,
        "error_coarsest_grid": worst,
        "grid_convergence": [row["max_relative_error"] for row in rows],
        "passed": bool(finest < 0.05),
        "stefan_number_freezing": stefan_number,
        "neumann_lambda_freezing": lam,
        "rows": rows,
        "detail": "1-D enthalpy solver vs X(t) = 2 lambda sqrt(alpha t); the grid "
        "sequence documents first-order convergence of the front position",
    }


def verify_competition_gates() -> dict:
    """Criterion (d): three-mechanism competition and physical direction gates."""
    model = FilmModel(adhesion=IceAdhesionModel())
    gates = direction_gates(model)
    state = ContactState(pressure=1.0e6, velocity=1.0, ice_temperature=263.15, structure_temperature=263.15)
    equilibrated = ContactState(
        pressure=1.0e6,
        velocity=1.0,
        ice_temperature=263.15,
        structure_temperature=melt_temperature(1.0e6),
    )
    onset_numeric = melting_onset_velocity(model, equilibrated)
    onset_analytic = melting_onset_velocity_analytic(model, equilibrated)
    onset_error = _relative_error(onset_numeric, onset_analytic)
    gate_values = {name: gate["passed"] for name, gate in gates.items()}
    # the closed form uses the dry boundary friction coefficient at the film
    # floor, while the closure adds a small viscous contribution there, so 5 % is
    # the honest agreement tolerance for the onset-speed comparison
    passed = all(gate_values.values()) and onset_error < 0.05
    return {
        "criterion": "d_competition_gates",
        "threshold": 0.05,
        "error": onset_error,
        "passed": bool(passed),
        "gates": gates,
        "gate_flags": gate_values,
        "onset_velocity_numeric": onset_numeric,
        "onset_velocity_analytic": onset_analytic,
        "onset_relative_error": onset_error,
        "cold_structure_onset": melting_onset_velocity(model, state),
        "detail": "gates G1-G5 plus melting-onset speed against its closed form",
    }


def end_to_end_case(
    *,
    pressure: float = 1.0e6,
    velocity: float = 0.1,
    temperature: float = 263.15,
    contact_length: float = 0.1,
    equilibrated_structure: bool = False,
    initial_thickness: float = 1.0e-6,
    t_end: float = 5.0e-3,
    n_samples: int = 201,
    name: str = "specified_case",
) -> dict:
    """One end-to-end ice/steel contact case with film and friction histories."""
    model = FilmModel(adhesion=IceAdhesionModel())
    probe = ContactState(
        pressure=pressure,
        velocity=velocity,
        ice_temperature=temperature,
        structure_temperature=temperature,
        contact_length=contact_length,
    )
    t_melt = model.melting_temperature(probe)
    state = ContactState(
        pressure=pressure,
        velocity=velocity,
        ice_temperature=temperature,
        structure_temperature=t_melt if equilibrated_structure else temperature,
        contact_length=contact_length,
    )
    steady = model.steady_state(state)
    history = model.integrate(state, initial_thickness=initial_thickness, t_end=t_end, n_samples=n_samples)
    lifetime = None
    for t, h in zip(history["time"], history["film_thickness"]):
        if h <= model.film_floor * 1.0000001:
            lifetime = t
            break
    rates = model.decompose(
        max(steady["film_thickness"], model.film_floor), state
    )
    return {
        "name": name,
        "inputs": {
            "pressure": pressure,
            "velocity": velocity,
            "ice_temperature": temperature,
            "structure_temperature": state.structure_temperature,
            "contact_length": contact_length,
            "equilibrated_structure": equilibrated_structure,
            "initial_thickness": initial_thickness,
            "t_end": t_end,
        },
        "melting_temperature": t_melt,
        "melt_point_depression": melt_point_depression(pressure, model.props),
        "adhesion_strength": IceAdhesionModel().adhesion_strength(temperature),
        "steady_state": steady,
        "steady_rates": rates,
        "film_lifetime": lifetime,
        "history": history,
        "onset_velocity_equilibrated": melting_onset_velocity(
            model,
            ContactState(
                pressure=pressure,
                velocity=velocity,
                ice_temperature=temperature,
                structure_temperature=t_melt,
                contact_length=contact_length,
            ),
        ),
    }


def case_studies() -> list[dict]:
    """The specified case plus two companion cases for the other two branches."""
    specified = end_to_end_case(name="specified_1MPa_0p1mps_253K", t_end=5.0e-3, n_samples=201)
    friction = end_to_end_case(
        name="friction_melting_1MPa_1mps_253K",
        velocity=1.0,
        equilibrated_structure=True,
        initial_thickness=5.0e-8,
        t_end=0.5,
        n_samples=201,
    )
    pressure = end_to_end_case(
        name="pressure_melting_20MPa_5mmps_272p5K",
        pressure=2.0e7,
        velocity=5.0e-3,
        temperature=272.5,
        equilibrated_structure=True,
        initial_thickness=1.0e-6,
        t_end=1.0,
        n_samples=201,
    )
    refreezing = end_to_end_case(
        name="refreezing_1MPa_1mps_253K_cold_slider",
        velocity=1.0,
        equilibrated_structure=False,
        initial_thickness=1.0e-6,
        t_end=5.0e-3,
        n_samples=201,
    )
    return [specified, friction, pressure, refreezing]


def run_all_criteria() -> dict:
    """Run criteria (a)-(e) and return the machine-checkable summary.

    Criterion (f) (the pytest count) is filled in by the benchmark runner, which
    is the only component that executes the test suite.
    """
    results = {
        "a_pressure_melting": verify_pressure_melting(),
        "b_friction_melting_steady": verify_friction_melting_steady(),
        "c_refreezing_stefan": verify_refreezing_stefan(),
        "d_competition_gates": verify_competition_gates(),
    }
    cases = case_studies()
    specified = cases[0]
    end_to_end_passed = bool(
        specified["history"]["film_thickness"]
        and all(map(lambda v: v == v, specified["history"]["film_thickness"]))
        and 0.0 < specified["steady_state"]["friction_coefficient"] <= 1.0
        and all(case["steady_state"]["friction_coefficient"] > 0.0 for case in cases)
    )
    results["e_end_to_end"] = {
        "criterion": "e_end_to_end",
        "threshold": None,
        "error": None,
        "passed": end_to_end_passed,
        "specified_case": {
            key: specified[key]
            for key in (
                "name",
                "inputs",
                "melting_temperature",
                "melt_point_depression",
                "adhesion_strength",
                "steady_state",
                "steady_rates",
                "film_lifetime",
                "onset_velocity_equilibrated",
            )
        },
        "cases": [
            {
                key: case[key]
                for key in (
                    "name",
                    "inputs",
                    "melting_temperature",
                    "melt_point_depression",
                    "adhesion_strength",
                    "steady_state",
                    "steady_rates",
                    "film_lifetime",
                    "onset_velocity_equilibrated",
                )
            }
            for case in cases
        ],
        "detail": "film thickness and friction time histories at 1 MPa / 0.1 m/s / -10 degC"
        " plus friction-melting, pressure-melting and refreezing companion cases",
    }
    summary = {
        name: bool(results[name]["passed"])
        for name in results
    }
    results["summary"] = {
        "criteria": summary,
        "all_passed": all(summary.values()),
    }
    return results