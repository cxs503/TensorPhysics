"""Wet / dry hull-girder modal basis for the ice-induced vibration chain.

This sub-module is the *modal input* of the chain
``ice milling torque -> hull girder response -> equipment base -> pipe stress ->
fatigue life``.  It does not re-implement a hull FEM: it reuses the long-ship
Timoshenko idealisation and the 2-D strip fluid-structure solver of
:mod:`tensorphysics.hull_hydroelastic` and exposes the pair of modal sets the
chain needs:

``dry``
    Free-free vertical bending modes in air (two rigid-body modes removed).
``wet``
    The wet eigenproblem solved in the dry modal basis.  The generalised mass
    becomes ``M = I + A`` with ``A_ij = int a33(x) phi_i phi_j dx`` (the
    infinite-frequency 2-D strip added mass) and the stiffness becomes
    ``K = diag(omega_dry^2) + G`` with ``G_ij = int rho g b(x) phi_i phi_j dx``
    (linearised buoyancy restoring stiffness).  The wet modes are therefore
    coupled (the added mass matrix is not diagonal) and the frequency ratio
    ``f_wet / f_dry < 1`` is the first physical acceptance criterion.

Geometry discipline (owner standing rule: geometry before physics): the hull
form is validated against a published reference displacement *before* any modal
or fatigue number is produced, so that a wrong ``Cb``/``L``/``B`` cannot
silently propagate into the fatigue life.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from ..hull_hydroelastic.presets import HullCase, hull_cases
from ..hull_hydroelastic.timoshenko import TimoshenkoBeam
from ..hull_hydroelastic.wet import (
    added_mass_distribution,
    ship_beam_properties,
    wet_modes,
)

__all__ = [
    "GEOMETRY_REFERENCE",
    "GeometryGate",
    "HullModalBasis",
    "geometry_gate",
    "wet_dry_modal_basis",
]

#: Published bare-hull references used by the geometry gate.
#: ``displacement_t`` is the moulded displacement at the design draught.
GEOMETRY_REFERENCE: dict[str, dict[str, float]] = {
    "KCS": {"displacement_t": 52030.0, "cb": 0.6505, "lpp_m": 230.0, "breadth_m": 32.2},
    "KVLCC2": {"displacement_t": 320650.0, "cb": 0.8098, "lpp_m": 320.0, "breadth_m": 58.0},
}


@dataclass
class GeometryGate:
    """Result of the hull-form consistency check (Cb / L / B / T)."""

    case_name: str
    length_m: float
    breadth_m: float
    draught_m: float
    depth_m: float
    block_coefficient: float
    displacement_t: float
    displacement_reference_t: float | None
    displacement_rel_error: float | None
    l_over_b: float
    b_over_t: float
    relative_tolerance: float
    passed: bool
    checks: dict[str, Any] = field(default_factory=dict)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "case": self.case_name,
            "length_m": self.length_m,
            "breadth_m": self.breadth_m,
            "draught_m": self.draught_m,
            "depth_m": self.depth_m,
            "block_coefficient": self.block_coefficient,
            "displacement_t": self.displacement_t,
            "displacement_reference_t": self.displacement_reference_t,
            "displacement_rel_error": self.displacement_rel_error,
            "L_over_B": self.l_over_b,
            "B_over_T": self.b_over_t,
            "relative_tolerance": self.relative_tolerance,
            "passed": self.passed,
            "checks": dict(self.checks),
        }


def geometry_gate(case: HullCase, rel_tol: float = 0.03) -> GeometryGate:
    """Validate Cb / L / B / T before any structural number is produced.

    Three independent checks (all must pass):

    1. ``|Cb - Cb_ref| / Cb_ref < rel_tol`` against the published block
       coefficient of the reference hull;
    2. ``|displacement - displacement_ref| / displacement_ref < 3 * rel_tol``
       (the displacement integrates ``Cb L B T``, so it catches a wrong
       combination of principal dimensions);
    3. principal-dimension ratios inside wide physical ranges
       (``2.5 < L/B < 12``, ``1.5 < B/T < 8``, ``0.40 < Cb < 0.90``).
    """
    ship = case.girder()
    ref = GEOMETRY_REFERENCE.get(case.name)
    displacement_t = ship.displacement_mass / 1000.0
    displacement_ref = None if ref is None else float(ref["displacement_t"])
    rel_error = (
        None
        if displacement_ref is None
        else float(abs(displacement_t - displacement_ref) / displacement_ref)
    )
    l_over_b = float(case.length / case.breadth)
    b_over_t = float(case.breadth / case.draught)
    checks = {
        "cb_relative": True
        if ref is None
        else bool(abs(case.block - ref["cb"]) / ref["cb"] < rel_tol),
        "displacement_relative": True if rel_error is None else bool(rel_error < 3.0 * rel_tol),
        "principal_ratios": bool(2.5 < l_over_b < 12.0 and 1.5 < b_over_t < 8.0),
        "block_coefficient_range": bool(0.40 < case.block < 0.90),
        "reference_available": ref is not None,
    }
    return GeometryGate(
        case_name=case.name,
        length_m=float(case.length),
        breadth_m=float(case.breadth),
        draught_m=float(case.draught),
        depth_m=float(case.depth),
        block_coefficient=float(case.block),
        displacement_t=float(displacement_t),
        displacement_reference_t=displacement_ref,
        displacement_rel_error=rel_error,
        l_over_b=l_over_b,
        b_over_t=b_over_t,
        relative_tolerance=float(rel_tol),
        passed=bool(all(v for k, v in checks.items() if k != "reference_available")),
        checks=checks,
    )


@dataclass
class HullModalBasis:
    """Dry and wet modal sets of one hull, plus the geometry gate."""

    case_name: str
    geometry: GeometryGate
    x_m: np.ndarray
    dry_frequencies_hz: np.ndarray
    wet_frequencies_hz: np.ndarray
    added_mass_ratio: np.ndarray
    dry_shapes: np.ndarray            # (n_modes, n_nodes), unit-max normalised
    wet_shapes: np.ndarray            # (n_modes, n_nodes), wet-mass normalised
    zeta: np.ndarray
    modal_mass_kg: np.ndarray         # wet generalised mass (1 kg by normalisation)
    added_mass_distribution_kg_m: np.ndarray
    mass_per_length_kg_m: np.ndarray
    dry_generalised_mass_kg: np.ndarray
    buoyancy_share: np.ndarray
    x_foundation_m: float
    x_excitation_m: float
    x_reaction_m: float
    metadata: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ views
    @property
    def n_modes(self) -> int:
        return int(self.wet_frequencies_hz.size)

    @property
    def frequency_ratio(self) -> np.ndarray:
        return self.wet_frequencies_hz / self.dry_frequencies_hz

    @property
    def all_wet_below_dry(self) -> bool:
        return bool(np.all(self.frequency_ratio < 1.0))

    def shape_at(self, x: float, wet: bool = True) -> np.ndarray:
        """Mode-shape ordinates interpolated at station ``x`` (m)."""
        shapes = self.wet_shapes if wet else self.dry_shapes
        return np.array([float(np.interp(x, self.x_m, phi)) for phi in shapes])

    def table(self) -> list[dict[str, Any]]:
        rows = []
        for i in range(self.n_modes):
            rows.append(
                {
                    "mode": i + 1,
                    "f_dry_hz": float(self.dry_frequencies_hz[i]),
                    "f_wet_hz": float(self.wet_frequencies_hz[i]),
                    "ratio_wet_over_dry": float(self.frequency_ratio[i]),
                    "added_mass_ratio": float(self.added_mass_ratio[i]),
                    "zeta": float(self.zeta[i]),
                    "buoyancy_stiffness_share": float(self.buoyancy_share[i]),
                    "phi_foundation_wet": float(self.shape_at(self.x_foundation_m)[i]),
                }
            )
        return rows

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "case": self.case_name,
            "geometry_gate": self.geometry.to_json_dict(),
            "stations_m": {
                "foundation": self.x_foundation_m,
                "excitation": self.x_excitation_m,
                "reaction": self.x_reaction_m,
            },
            "modes": self.table(),
            "wet_below_dry_all_modes": self.all_wet_below_dry,
            "n_modes": self.n_modes,
            "metadata": dict(self.metadata),
        }


def wet_dry_modal_basis(
    case: HullCase,
    n_modes: int = 8,
    n_elements: int = 120,
    zeta_structural: float = 0.02,
    n_stations: int = 13,
    n_panels: int = 64,
    x_foundation_fraction: float = 0.20,
    x_excitation_fraction: float = -0.30,
    x_reaction_fraction: float = -0.45,
) -> HullModalBasis:
    """Solve the dry and wet vertical-bending modal sets of ``case``.

    ``x_foundation_fraction`` is the longitudinal station of the equipment
    foundation (fraction of ``L``, ``+`` forward), ``x_excitation_fraction`` the
    station where the shaft-line reaction of the ice torque enters the hull and
    ``x_reaction_fraction`` the companion (opposite-sign) bearing station, so
    that the excitation is a *self-equilibrated couple* and cannot excite the
    rigid-body modes of the free-free hull.
    """
    gate = geometry_gate(case)
    if not gate.passed:
        raise ValueError(f"geometry gate failed for {case.name}: {gate.to_json_dict()}")

    ship = case.girder()
    beam = TimoshenkoBeam(ship_beam_properties(ship, n_elements=n_elements))
    am = added_mass_distribution(
        ship,
        n_stations=n_stations,
        section_exponent=case.section_exponent,
        n_panels=n_panels,
        x=beam.props.x,
    )
    wet = wet_modes(
        beam,
        am["a33"],
        n_modes=n_modes,
        zeta_structural=zeta_structural,
        b_waterplane=ship.waterplane_width(beam.props.x),
    )
    x = np.asarray(wet.x, dtype=float)
    dry_shapes = np.array(wet.dry.shapes, dtype=float)
    # unit-max normalisation for reporting / plotting (keeps the physics in the
    # mass-normalised wet shapes used by the response solver)
    dry_shapes_unit = dry_shapes / np.max(np.abs(dry_shapes), axis=1, keepdims=True)
    return HullModalBasis(
        case_name=case.name,
        geometry=gate,
        x_m=x,
        dry_frequencies_hz=np.asarray(wet.dry.frequencies_hz, dtype=float),
        wet_frequencies_hz=np.asarray(wet.wet_frequencies_hz, dtype=float),
        added_mass_ratio=np.asarray(wet.added_mass_ratio, dtype=float),
        dry_shapes=dry_shapes_unit,
        wet_shapes=np.array(wet.wet_shapes, dtype=float),
        zeta=np.asarray(wet.zeta, dtype=float),
        modal_mass_kg=np.asarray(wet.generalised_mass, dtype=float),
        added_mass_distribution_kg_m=np.asarray(am["a33"], dtype=float),
        mass_per_length_kg_m=np.asarray(beam.props.m, dtype=float),
        dry_generalised_mass_kg=np.ones(n_modes, dtype=float),
        buoyancy_share=np.asarray(wet.buoyancy_share, dtype=float),
        x_foundation_m=float(x_foundation_fraction) * float(case.length),
        x_excitation_m=float(x_excitation_fraction) * float(case.length),
        x_reaction_m=float(x_reaction_fraction) * float(case.length),
        metadata={
            "n_elements": int(n_elements),
            "n_stations_added_mass": int(n_stations),
            "n_panels_2d": int(n_panels),
            "zeta_structural": float(zeta_structural),
            "added_mass_model": "2-D strip, infinite-frequency heave added mass a33(x)",
            "wet_eigenproblem": "dry modal basis, generalised M = I + A, K = diag(w_dry^2) + buoyancy",
            "mass_normalisation": "phi^T M phi = 1 kg (generalised mass = 1 kg)",
            "excitation_model": "self-equilibrated shaft-line reaction couple (x_exc, x_react)",
        },
    )