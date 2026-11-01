"""Competition of the three phase-change mechanisms in the ice/structure contact.

The mechanisms are

* **friction melting** -- dissipation ``tau V`` converted into latent heat;
* **pressure melting** -- the Clausius-Clapeyron depression of ``T_melt``, which
  weakens the refreezing sinks and, above a threshold pressure, superheats the
  ice so that it melts without any friction;
* **refreezing** -- the Stefan sinks into the cold ice body and the cold
  structure, which are strongest exactly when the film is thin (the structure
  resistance is the film itself).

This module provides the mechanism decomposition at a point (:meth:`FilmModel`
already exposes it), the *melting onset velocity* where friction melting overtakes
the sinks, and a parameter-space scanner that labels which mechanism dominates.
The scanner is what the acceptance criterion "physical direction gates" checks:
the sign of every partial derivative must match the physics, not merely be
non-NaN.

Physical direction gates (implemented in :func:`direction_gates`)
----------------------------------------------------------------
G1  friction melting: film thickness increases with sliding speed (saturating at
    ``mu p L / (c_d rho_ice L_f)``);
G2  load: film thickness increases with contact pressure;
G3  thermal: film thickness decreases as the ice gets colder;
G4  pressure melting: the depression grows with pressure and, at -10 degC, a film
    appears with no frictional heating above ~135 MPa;
G5  mechanism coverage: all three mechanisms dominate somewhere in the scanned
    space (otherwise the "competition" is not demonstrated).
"""

from math import sqrt

from .film import ContactState, FilmModel
from .pressure_melting import melt_point_depression, pressure_melting_threshold, superheat

__all__ = [
    "mechanism_decomposition",
    "dominant_mechanism",
    "melting_onset_velocity",
    "melting_onset_velocity_analytic",
    "scan_mechanism_map",
    "direction_gates",
]

_MECHANISMS = ("friction_melting", "pressure_melting", "refreezing")


def mechanism_decomposition(model: FilmModel, state: ContactState, film_thickness: float) -> dict:
    """Rate decomposition [m/s] with the dominant mechanism labelled."""
    rates = model.decompose(film_thickness, state)
    return rates


def dominant_mechanism(model: FilmModel, state: ContactState, film_thickness: float) -> str:
    """Label of the mechanism with the largest rate magnitude at this state."""
    rates = model.decompose(film_thickness, state)
    magnitudes = {name: abs(rates[name]) for name in _MECHANISMS}
    return max(magnitudes, key=magnitudes.get)


def melting_onset_velocity(
    model: FilmModel,
    state: ContactState,
    lower: float = 1.0e-4,
    upper: float = 100.0,
    tolerance: float = 1.0e-10,
) -> float | None:
    """Sliding speed [m/s] above which a steady melt film can exist.

    Bisection on the sign of the ODE right-hand side evaluated at the film floor:
    when it is positive the interface can melt even from a vanishing film, when it
    is negative refreezing wins and no film survives.  Returns ``None`` when the
    bracket does not change sign.
    """

    def rate(velocity: float) -> float:
        trial = ContactState(
            pressure=state.pressure,
            velocity=velocity,
            ice_temperature=state.ice_temperature,
            structure_temperature=state.structure_temperature,
            contact_length=state.contact_length,
        )
        return model.rhs(model.film_floor, trial)

    if rate(lower) > 0.0:
        return lower
    if rate(upper) < 0.0:
        return None
    lo, hi = lower, upper
    for _ in range(400):
        mid = 0.5 * (lo + hi)
        if rate(mid) > 0.0:
            hi = mid
        else:
            lo = mid
        if hi - lo <= tolerance * hi:
            break
    return 0.5 * (lo + hi)


def melting_onset_velocity_analytic(
    model: FilmModel,
    state: ContactState,
    friction_coefficient_value: float | None = None,
) -> float:
    """Closed-form melting onset speed [m/s], independent of any root finding.

    At the film floor both Stefan sinks scale as ``sqrt(V)`` while the friction
    heating scales as ``V``:

        N(V) = mu p V + (Delta_ice + Delta_struct) sqrt(V),
        Delta_ice    = k_ice (T_ice    - T_melt) / sqrt(alpha_ice    L),
        Delta_struct = k_s   (T_struct - T_melt) / sqrt(alpha_struct L),

    so the existence boundary of a steady film is

        V_c = ( -(Delta_ice + Delta_struct) / (mu p) )^2,

    clamped at zero when the sum of the sinks is non-negative (the interface melts
    even at vanishing speed).  Returns ``0.0`` in that case and ``inf`` when the
    sinks cannot be overcome at any speed.
    """
    mu = (
        model.tribology.mu_boundary
        if friction_coefficient_value is None
        else float(friction_coefficient_value)
    )
    t_melt = model.melting_temperature(state)
    delta_ice = sqrt(model.props.alpha_ice * state.contact_length)
    delta_structure = sqrt(model.structure.alpha * state.contact_length)
    delta_ice_sink = model.props.k_ice * (state.ice_temperature - t_melt) / delta_ice
    delta_structure_sink = (
        model.structure.k * (state.structure_temperature - t_melt) / delta_structure
    )
    total_sink = delta_ice_sink + delta_structure_sink
    if total_sink >= 0.0:
        return 0.0
    return (min(-total_sink, 1.0e300) / (mu * state.pressure)) ** 2


def scan_mechanism_map(
    model: FilmModel,
    pressures: list[float],
    velocities: list[float],
    temperatures: list[float],
    contact_length: float = 0.1,
    equilibrated_structure: bool = True,
) -> list[dict]:
    """Steady-state scan over (pressure, velocity, temperature).

    ``equilibrated_structure=True`` sets the structure surface temperature to the
    local melting point (a slider that has already been heated by the contact),
    which isolates the ice-side Stefan sink; ``False`` keeps a cold structure and
    activates the structure-side refreezing sink.  Every row reports the
    self-consistent steady state, its friction coefficient, the mechanism rates
    and the dominant mechanism, so the map is auditable row by row.
    """
    rows: list[dict] = []
    for pressure in pressures:
        for velocity in velocities:
            for temperature in temperatures:
                probe = ContactState(
                    pressure=pressure,
                    velocity=velocity,
                    ice_temperature=temperature,
                    structure_temperature=temperature,
                    contact_length=contact_length,
                )
                t_melt = model.melting_temperature(probe)
                structure_temperature = t_melt if equilibrated_structure else temperature
                state = ContactState(
                    pressure=pressure,
                    velocity=velocity,
                    ice_temperature=temperature,
                    structure_temperature=structure_temperature,
                    contact_length=contact_length,
                )
                steady = model.steady_state(state)
                thickness = steady["film_thickness"]
                probe_thickness = thickness if thickness > 0.0 else model.film_floor
                rates = model.decompose(probe_thickness, state)
                rows.append(
                    {
                        "pressure": pressure,
                        "velocity": velocity,
                        "ice_temperature": temperature,
                        "structure_temperature": structure_temperature,
                        "film_exists": steady["exists"],
                        "film_thickness": thickness,
                        "friction_coefficient": steady["friction_coefficient"],
                        "regime": steady["regime"],
                        "melting_temperature": t_melt,
                        "melt_point_depression": melt_point_depression(pressure, model.props),
                        "superheat": superheat(pressure, temperature, model.props),
                        "friction_melting": rates["friction_melting"],
                        "pressure_melting": rates["pressure_melting"],
                        "refreezing": rates["refreezing"],
                        "dominant_mechanism": dominant_mechanism(model, state, probe_thickness),
                        "equilibrated_structure": equilibrated_structure,
                    }
                )
    return rows


def direction_gates(model: FilmModel, contact_length: float = 0.1) -> dict:
    """Check the physical direction gates G1-G5 and return the evidence.

    Each gate returns ``{"passed": bool, "values": [...], "detail": str}`` so the
    report can quote the numbers that were compared, not only the verdict.
    """
    gates: dict[str, dict] = {}

    def steady_thickness(pressure: float, velocity: float, temperature: float,
                         equilibrated: bool = True) -> float | None:
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
            structure_temperature=t_melt if equilibrated else temperature,
            contact_length=contact_length,
        )
        result = model.steady_state(state)
        return result["film_thickness"] if result["exists"] else None

    # G1: friction melting increases the film with speed
    velocities = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
    thicknesses = [steady_thickness(1.0e6, v, 263.15) for v in velocities]
    existing = [(v, h) for v, h in zip(velocities, thicknesses) if h is not None]
    monotone = all(
        existing[i][1] <= existing[i + 1][1] * (1.0 + 1e-9) for i in range(len(existing) - 1)
    )
    gates["G1_friction_melting_V"] = {
        "passed": bool(len(existing) >= 3 and monotone and existing[-1][1] > existing[0][1]),
        "values": [{"velocity": v, "film_thickness": h} for v, h in zip(velocities, thicknesses)],
        "detail": "film thickness must increase with sliding speed above the melting onset",
    }

    # G2: thicker film at higher contact pressure
    pressures = [0.25e6, 0.5e6, 1.0e6, 2.0e6, 4.0e6]
    thicknesses = [steady_thickness(p, 4.0, 263.15) for p in pressures]
    existing = [(p, h) for p, h in zip(pressures, thicknesses) if h is not None]
    monotone = all(
        existing[i][1] <= existing[i + 1][1] * (1.0 + 1e-9) for i in range(len(existing) - 1)
    )
    gates["G2_load_p"] = {
        "passed": bool(len(existing) >= 3 and monotone and existing[-1][1] > existing[0][1]),
        "values": [{"pressure": p, "film_thickness": h} for p, h in zip(pressures, thicknesses)],
        "detail": "film thickness must increase with contact pressure",
    }

    # G3: colder ice -> thinner film and higher friction.  The thickness gate is
    # stated over the melting-relevant range (-5 ... -30 degC); below that the
    # mixed-lubrication feedback (thinner film -> more asperity load sharing ->
    # more frictional heat) flattens the thickness while friction keeps rising,
    # which is a modelled effect and is asserted separately.
    def steady_mu(pressure: float, velocity: float, temperature: float) -> float | None:
        probe = ContactState(
            pressure=pressure,
            velocity=velocity,
            ice_temperature=temperature,
            structure_temperature=temperature,
            contact_length=contact_length,
        )
        state = ContactState(
            pressure=pressure,
            velocity=velocity,
            ice_temperature=temperature,
            structure_temperature=model.melting_temperature(probe),
            contact_length=contact_length,
        )
        result = model.steady_state(state)
        return result["friction_coefficient"] if result["exists"] else None

    temperatures = [268.15, 263.15, 253.15, 243.15]
    thicknesses = [steady_thickness(1.0e6, 8.0, t) for t in temperatures]
    existing = [(t, h) for t, h in zip(temperatures, thicknesses) if h is not None]
    monotone = all(
        existing[i][1] >= existing[i + 1][1] * (1.0 - 1e-9) for i in range(len(existing) - 1)
    )
    cold_temperatures = [268.15, 263.15, 253.15, 243.15, 233.15, 223.15]
    friction_values = [steady_mu(1.0e6, 8.0, t) for t in cold_temperatures]
    friction_existing = [(t, mu) for t, mu in zip(cold_temperatures, friction_values) if mu is not None]
    friction_monotone = all(
        friction_existing[i][1] <= friction_existing[i + 1][1] * (1.0 + 1e-9)
        for i in range(len(friction_existing) - 1)
    )
    gates["G3_thermal_T"] = {
        "passed": bool(
            len(existing) >= 3
            and monotone
            and existing[0][1] > existing[-1][1]
            and len(friction_existing) >= 4
            and friction_monotone
            and friction_existing[-1][1] > friction_existing[0][1]
        ),
        "values": {
            "film_thickness_vs_temperature": [
                {"ice_temperature": t, "film_thickness": h} for t, h in zip(temperatures, thicknesses)
            ],
            "friction_vs_temperature": [
                {"ice_temperature": t, "friction_coefficient": mu}
                for t, mu in zip(cold_temperatures, friction_values)
            ],
        },
        "detail": "colder ice must thin the steady film (Range -5..-30 degC) and raise friction "
        "monotonically; the thickness plateau below -30 degC is the friction self-regulation",
    }

    # G4: pressure melting threshold at -10 degC
    threshold = pressure_melting_threshold(263.15, model.props)
    depression_low = melt_point_depression(1.0e6, model.props)
    depression_high = melt_point_depression(threshold, model.props)
    superheat_high = superheat(threshold * 1.05, 263.15, model.props)
    superheat_low = superheat(1.0e6, 263.15, model.props)
    gates["G4_pressure_melting"] = {
        "passed": bool(
            abs(depression_high - 10.0) < 0.05
            and depression_low < 0.1
            and superheat_low < 0.0 < superheat_high
        ),
        "values": {
            "threshold_pressure_Pa": threshold,
            "depression_at_1MPa_K": depression_low,
            "depression_at_threshold_K": depression_high,
            "superheat_at_1MPa_K": superheat_low,
            "superheat_at_1.05x_threshold_K": superheat_high,
        },
        "detail": "pressure melting must be negligible at 1 MPa and active above ~135 MPa at -10 degC",
    }

    # G5: all three mechanisms dominate somewhere
    rows = scan_mechanism_map(
        model,
        pressures=[0.5e6, 1.0e6, 4.0e6, 2.0e7, 5.0e7],
        velocities=[0.002, 0.005, 0.02, 0.05, 0.5, 5.0],
        temperatures=[272.9, 272.5, 270.0, 268.15, 263.15, 248.15],
        contact_length=contact_length,
        equilibrated_structure=True,
    )
    cold_rows = scan_mechanism_map(
        model,
        pressures=[1.0e6],
        velocities=[0.05, 0.5, 5.0],
        temperatures=[263.15],
        contact_length=contact_length,
        equilibrated_structure=False,
    )
    labels = {row["dominant_mechanism"] for row in rows + cold_rows}
    gates["G5_mechanism_coverage"] = {
        "passed": set(_MECHANISMS).issubset(labels),
        "values": sorted(labels),
        "detail": "pressure melting, friction melting and refreezing must each dominate somewhere",
    }
    return gates