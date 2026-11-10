"""Acceptance harness for the J2 ice-indentation node (AC1 ... AC6).

Each criterion is computed from a *programmatic* quantity and compared with its
threshold; nothing is hard-coded as a pass.  The harness returns a plain dict
so that the benchmark driver can dump it as JSON (zero hand-copied numbers).

Criteria
--------
AC1  uniaxial stress path vs the closed-form J2 + linear hardening solution   <= 1 %
AC2  pure-shear first yield vs ``sigma_y / sqrt(3)``                          <= 1 %
AC3  indentation force-displacement vs the exact Hertz elastic solution       <= 15 %
     (the elastoplastic branch is reported separately: no experimental or
      closed-form reference for J2 ice indentation exists in this worktree,
      so that sub-criterion is declared ``WITHHELD`` instead of being faked)
AC4  contact force under mesh refinement: 3 meshes, monotonically decreasing
     error against the finest mesh (elastoplastic case)
AC5  energy balance ``work of the contact loads = elastic energy + plastic
     dissipation`` residual                                                    <= 1 %
AC6  pytest case count for this node                                          >= 12

Model problem (quarter block with symmetry planes ``x = y = 0``):

=================  ============================================================
ice block          ``half_width x half_width x height`` = 10 x 10 x 6 mm
mesh               power-graded Hex8 (``n**-2`` spacing towards the contact)
base / far field   ``u_z = 0`` on the base, rollers on the two outer faces
indenter           rigid frictionless sphere ``R = 10 mm``
ice constitutive   J2, ``E = 9 GPa``, ``nu = 0.33``, ``sigma_y = 5 MPa``
                   (elastic-perfectly-plastic, ``H = 0``)
loading            displacement control, ``delta_max = 15 um`` in 6 steps
                   (``delta proportional to s**3`` load ramp)
contact            penalty, facet integrated, ``beta = 10``
=================  ============================================================
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from ..solid_fem.material import J2Material, PlasticState

from .analytics import (
    hertz_force,
    hertz_mean_pressure,
    hertz_yield_onset,
    johnson_mean_pressure,
    johnson_validity_ratio,
    reduced_modulus,
)
from .constitutive import shear_yield_stress, tangent_error, uniaxial_stress_response
from .contact import SphericalIndenter
from .indentation import simulate_indentation
from .mesh import quarter_block_mesh

__all__ = [
    "AcceptanceConfig",
    "default_config",
    "check_ac1",
    "check_ac2",
    "check_ac3",
    "check_ac4",
    "check_ac5",
    "check_ac6",
    "run_indentation_study",
    "run_elastic_hertz_study",
    "acceptance_report",
]


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AcceptanceConfig:
    """Material, geometry and mesh-family parameters of the acceptance study."""

    # constitutive (ice at about -10 degC)
    E: float = 9.0e9
    nu: float = 0.33
    sigma_y: float = 5.0e6
    hardening: float = 0.0

    # AC1 / AC2 paths
    uniaxial_max_strain: float = 0.01
    uniaxial_steps: int = 400
    ac1_tolerance: float = 0.01
    ac2_tolerance: float = 0.01

    # indentation model problem
    half_width: float = 10.0e-3
    height: float = 6.0e-3
    indenter_radius: float = 10.0e-3
    delta_max: float = 15.0e-6
    n_steps: int = 6
    power: float = 2.0
    vertical_power: float = 1.6
    penalty_factor: float = 10.0
    newton_tol: float = 1e-5
    newton_max_iter: int = 40
    contact_model: str = "facet"
    ramp: str = "power3"

    # mesh family (refinement by increasing divisions, fixed grading)
    mesh_dims: tuple = ((9, 9, 4), (11, 11, 5), (14, 14, 6))

    # elastic Hertz comparison (same block/indenter, no yielding)
    hertz_deltas: tuple = (5.0e-6, 10.0e-6, 15.0e-6)
    hertz_sigma_y: float = 1.0e12
    # the elastic verification uses a finer mesh than the plastic family: with
    # yielding off there is no plastic-zone depth to resolve, so the extra
    # elements are best spent on the contact patch
    hertz_dims: tuple = (18, 18, 9)

    # dedicated energy-balance run (dense linear load path so that the
    # trapezoidal path integral itself is accurate to << 1 %)
    energy_steps: int = 16
    energy_ramp: str = "linear"

    # thresholds
    ac3_tolerance: float = 0.15
    ac5_tolerance: float = 0.01
    ac6_min_tests: int = 12

    def material(self, sigma_y: float | None = None,
                 hardening: float | None = None) -> J2Material:
        return J2Material(
            E=self.E, nu=self.nu,
            sigma_y0=self.sigma_y if sigma_y is None else sigma_y,
            hardening=self.hardening if hardening is None else hardening,
        )


def default_config(**kwargs) -> AcceptanceConfig:
    return replace(AcceptanceConfig(), **kwargs)


# --------------------------------------------------------------------------- #
# AC1 / AC2 -- material point
# --------------------------------------------------------------------------- #
def _curve(resp) -> dict:
    return {
        "strain": np.asarray(resp.strain_xx).tolist(),
        "stress_numeric": np.asarray(resp.stress_xx).tolist(),
        "stress_analytic": np.asarray(resp.analytic).tolist(),
    }


def check_ac1(config: AcceptanceConfig) -> dict:
    """Uniaxial stress path vs the closed form (tension / compression / hardening)."""
    out = {"criterion": "AC1",
           "description": "uniaxial stress path vs analytic J2 + linear hardening",
           "tolerance": config.ac1_tolerance}
    cases: dict[str, dict] = {}
    curves: dict[str, dict] = {}
    material = config.material()
    for name, signed in (("tension", 1.0), ("compression", -1.0)):
        resp = uniaxial_stress_response(material, signed * config.uniaxial_max_strain,
                                        n_steps=config.uniaxial_steps)
        cases[name] = {
            "max_relative_error": resp.max_relative_error,
            "yield_strain_analytic": material.sigma_y0 / material.E,
            "yield_strain_numeric": resp.plastic_onset_strain(),
            "sigma_max_Pa": float(resp.stress_xx[-1]),
            "analytic_max_Pa": float(resp.analytic[-1]),
        }
        curves[name] = _curve(resp)
    material_h = config.material(hardening=5.0e8)
    resp_h = uniaxial_stress_response(material_h, config.uniaxial_max_strain,
                                      n_steps=config.uniaxial_steps)
    cases["tension_hardening"] = {
        "max_relative_error": resp_h.max_relative_error,
        "hardening_Pa": material_h.hardening,
        "sigma_max_Pa": float(resp_h.stress_xx[-1]),
    }
    curves["tension_hardening"] = _curve(resp_h)
    for case in cases.values():
        case["passed"] = bool(case["max_relative_error"] <= config.ac1_tolerance)
    out["cases"] = cases
    out["curves"] = curves
    out["max_relative_error"] = max(c["max_relative_error"] for c in cases.values())
    out["passed"] = all(c["passed"] for c in cases.values())
    return out


def check_ac2(config: AcceptanceConfig) -> dict:
    """Pure-shear first yield vs ``sigma_y / sqrt(3)`` + consistent tangent check."""
    material = config.material()
    measured = shear_yield_stress(material)
    analytic = material.sigma_y0 / np.sqrt(3.0)
    err = abs(measured - analytic) / analytic
    strain = np.zeros(6)
    strain[3] = 0.01
    t_err = tangent_error(material, strain, PlasticState())
    return {
        "criterion": "AC2",
        "description": "pure shear yield stress = sigma_y / sqrt(3)",
        "measured_Pa": float(measured),
        "analytic_Pa": float(analytic),
        "relative_error": float(err),
        "tolerance": config.ac2_tolerance,
        "consistent_tangent_relative_error": float(t_err),
        "passed": bool(err <= config.ac2_tolerance),
    }


# --------------------------------------------------------------------------- #
# Indentation studies
# --------------------------------------------------------------------------- #
def _run_one(config: AcceptanceConfig, dims, sigma_y: float | None = None,
             delta_max: float | None = None, n_steps: int | None = None,
             penalty_factor: float | None = None,
             ramp: str | None = None) -> tuple:
    mesh = quarter_block_mesh(
        half_width=config.half_width, height=config.height, dims=dims,
        power=config.power, vertical_power=config.vertical_power,
    )
    material = config.material(sigma_y=sigma_y)
    indenter = SphericalIndenter(radius=config.indenter_radius, apex_z=config.height)
    result = simulate_indentation(
        mesh, material, indenter,
        delta_max=config.delta_max if delta_max is None else delta_max,
        n_steps=config.n_steps if n_steps is None else n_steps,
        penalty_factor=(config.penalty_factor if penalty_factor is None
                        else penalty_factor),
        symmetry_factor=4,
        tol=config.newton_tol,
        max_iter=config.newton_max_iter,
        contact_model=config.contact_model,
        ramp=config.ramp if ramp is None else ramp,
    )
    return mesh, material, result


def run_indentation_study(config: AcceptanceConfig, mesh_dims=None) -> dict:
    """Elastoplastic indentation on every mesh of the refinement family."""
    dims_list = config.mesh_dims if mesh_dims is None else mesh_dims
    runs = []
    for dims in dims_list:
        mesh, material, result = _run_one(config, dims)
        runs.append({
            "dims": tuple(int(d) for d in dims),
            "n_elem": int(mesh.n_elem),
            "n_dof": int(mesh.n_dof),
            "h_contact_m": float(np.min(np.diff(np.unique(mesh.nodes[:, 0])))),
            "force_N": float(result.force[-1]),
            "converged": bool(result.converged),
            "contact_radius_m": float(result.contact_radius[-1]),
            "mean_pressure_Pa": float(result.mean_pressure[-1]),
            "max_alpha": float(result.max_alpha[-1]),
            "max_von_mises_Pa": float(result.max_von_mises[-1]),
            "newton_iterations": int(sum(i.get("iterations", 0)
                                         for i in result.newton_iterations)),
            "energy_balance": result.energy_balance(),
            "delta": result.delta.tolist(),
            "force_history": result.force.tolist(),
            "_result": result,
        })
    return {"runs": runs}


def run_elastic_hertz_study(config: AcceptanceConfig, dims=None) -> dict:
    """Elastic indentation (yield switched off) vs the exact Hertz solution."""
    dims = config.hertz_dims if dims is None else dims
    E_star = reduced_modulus(config.E, config.nu)
    deltas, forces, hertz, areas = [], [], [], []
    for delta in config.hertz_deltas:
        mesh, material, result = _run_one(config, dims, sigma_y=config.hertz_sigma_y,
                                          delta_max=delta, n_steps=1)
        deltas.append(float(delta))
        forces.append(float(result.force[-1]))
        hertz.append(float(hertz_force(E_star, config.indenter_radius, delta)))
        areas.append(float(result.contact_area[-1]))
    forces = np.asarray(forces)
    hertz = np.asarray(hertz)
    rel = np.abs(forces - hertz) / hertz
    return {
        "dims": tuple(int(d) for d in dims),
        "deltas_m": deltas,
        "force_fem_N": forces.tolist(),
        "force_hertz_N": hertz.tolist(),
        "relative_error": rel.tolist(),
        "max_relative_error": float(np.max(rel)),
        "contact_area_m2": areas,
    }


# --------------------------------------------------------------------------- #
# AC3 ... AC5
# --------------------------------------------------------------------------- #
def check_ac3(config: AcceptanceConfig, elastic: dict, plastic_run: dict,
              material: J2Material) -> dict:
    """Force-displacement verification against the exact Hertz elastic solution.

    Honest note on the benchmark: the node asks for a comparison with an
    *experimental* ice-indentation force-displacement curve.  No such dataset
    exists inside this repository and fabricating one is not acceptable, so

    * the elastic branch is verified against the exact Hertz solution
      (``P = 4/3 E* sqrt(R) delta**1.5``) on the same mesh, block and indenter
      -- this is a genuine independent reference and the acceptance gate;
    * the elastoplastic branch (``sigma_y = 5 MPa``) is reported with its
      numerical uncertainty taken from the mesh-refinement study (AC4) and is
      flagged ``WITHHELD`` against experiment.  The Johnson cavity-expansion
      mean pressure is reported as a diagnostic together with its validity
      ratio ``E* a/(3 sigma_y)``, which for this ice-like material lies far
      below the asymptotic window ``>> 1``, so it is *not* used as a gate.
    """
    worst = elastic["max_relative_error"]
    E_star = reduced_modulus(config.E, config.nu)
    a = np.sqrt(config.indenter_radius * config.delta_max)
    p_m = float(plastic_run["mean_pressure_Pa"])
    out = {
        "criterion": "AC3",
        "description": "indentation F(delta) vs exact Hertz elastic solution",
        "tolerance": config.ac3_tolerance,
        "elastic_reference": elastic,
        "max_relative_error": worst,
        "mean_relative_error": float(np.mean(elastic["relative_error"])),
        "elastic_passed": bool(worst <= config.ac3_tolerance),
        "plastic_branch": {
            "delta_max_m": float(plastic_run["delta"][-1]),
            "force_N": float(plastic_run["force_N"]),
            "mean_pressure_Pa": p_m,
            "constraint_factor_p_m_over_sigma_y": p_m / material.sigma_y0,
            "hertz_mean_pressure_at_delta_max_Pa": float(
                hertz_mean_pressure(E_star, config.indenter_radius,
                                    config.delta_max)),
            "johnson_validity_ratio": float(
                johnson_validity_ratio(config.E, config.nu, material.sigma_y0, a)),
            "johnson_mean_pressure_Pa": (
                float(johnson_mean_pressure(config.E, config.nu,
                                            material.sigma_y0, a))
                if johnson_validity_ratio(config.E, config.nu,
                                          material.sigma_y0, a) > 1.0 else None),
            "yield_onset": hertz_yield_onset(config.E, config.nu,
                                             material.sigma_y0,
                                             config.indenter_radius),
            "reference": "WITHHELD",
            "reference_note": (
                "no experimental indentation F(delta) dataset and no closed-form "
                "J2 indentation solution is available in this worktree; the "
                "elastoplastic branch is therefore reported with its "
                "mesh-refinement uncertainty (AC4) instead of an error against "
                "an invented reference"
            ),
        },
        "experimental_subcriterion": "WITHHELD",
    }
    out["passed"] = bool(out["elastic_passed"])
    return out


def check_ac4(config: AcceptanceConfig, study: dict) -> dict:
    """Contact force under mesh refinement: 3 meshes, monotonically decreasing error."""
    runs = study["runs"]
    forces = np.array([r["force_N"] for r in runs], dtype=float)
    reference = forces[-1]
    errs = np.abs(forces - reference) / reference
    deltas = np.diff(errs)
    monotone = bool(np.all(deltas < 0.0))
    return {
        "criterion": "AC4",
        "description": "mesh refinement (3 meshes): force error decreases monotonically",
        "meshes": [r["dims"] for r in runs],
        "n_elem": [r["n_elem"] for r in runs],
        "h_contact_m": [r["h_contact_m"] for r in runs],
        "forces_N": forces.tolist(),
        "relative_errors": errs.tolist(),
        "error_decrements": deltas.tolist(),
        "converged_flags": [r["converged"] for r in runs],
        "monotone_decrease": monotone,
        "observed_order": _observed_order(
            np.array([r["h_contact_m"] for r in runs]), errs[:-1] + 1e-16),
        "passed": bool(monotone and all(r["converged"] for r in runs)),
    }


def _observed_order(h: np.ndarray, err: np.ndarray) -> float | None:
    """Least-squares slope of ``log(err)`` against ``log(h)`` (observed order)."""
    h = np.asarray(h, dtype=float)[: err.size]
    if h.size < 2 or np.any(err <= 0.0):
        return None
    slope, _intercept = np.polyfit(np.log(h), np.log(err), 1)
    return float(slope)


def check_ac5(config: AcceptanceConfig, result) -> dict:
    """Work of the contact loads = elastic energy + plastic dissipation.

    The gate is the residual of the body balance ``W = U_elastic + D_plastic``.
    The penalty energy is *not* part of the body balance (it lives in the
    artificial spring) and is reported as a diagnostic together with the
    indenter-work closure ``W_indenter = U_el + D_pl + U_penalty``.
    """
    bal = result.energy_balance()
    return {
        "criterion": "AC5",
        "description": "work of contact loads = elastic + plastic dissipation",
        "tolerance": config.ac5_tolerance,
        **bal,
        "penalty_energy_fraction": abs(bal["penalty_fraction"]),
        "passed": bool(abs(bal["relative_residual"]) <= config.ac5_tolerance),
    }


def check_ac6(config: AcceptanceConfig, n_tests: int) -> dict:
    return {
        "criterion": "AC6",
        "description": "pytest cases for the ice-indentation node",
        "required": config.ac6_min_tests,
        "collected": int(n_tests),
        "passed": bool(n_tests >= config.ac6_min_tests),
    }


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def acceptance_report(config: AcceptanceConfig | None = None,
                      mesh_dims=None, n_tests: int | None = None,
                      verbose: bool = False, return_objects: bool = False):
    """Run every criterion and return the machine-checkable report dict.

    With ``return_objects=True`` a tuple ``(report, objects)`` is returned where
    ``objects`` holds the raw :class:`IndentationResult` instances (for figures
    and field plots); they are not JSON serialisable and are kept out of the
    report dict.
    """
    cfg = default_config() if config is None else config
    ac1 = check_ac1(cfg)
    ac2 = check_ac2(cfg)
    if verbose:
        print(f"AC1 max rel err = {ac1['max_relative_error']:.3e}")
        print(f"AC2 rel err     = {ac2['relative_error']:.3e}")
    study = run_indentation_study(cfg, mesh_dims)
    runs = study["runs"]
    elastic = run_elastic_hertz_study(cfg)
    if verbose:
        print(f"Hertz max rel err = {elastic['max_relative_error']:.3e}")
        print("forces:", [round(r["force_N"], 3) for r in runs])
    ac3 = check_ac3(cfg, elastic, runs[-1], runs[-1]["_result"].material)
    ac4 = check_ac4(cfg, study)
    _, _, energy_result = _run_one(cfg, cfg.mesh_dims[0],
                                   n_steps=cfg.energy_steps,
                                   ramp=cfg.energy_ramp)
    ac5 = check_ac5(cfg, energy_result)
    ac5["run"] = {"dims": list(cfg.mesh_dims[0]), "n_steps": cfg.energy_steps,
                  "ramp": cfg.energy_ramp,
                  "converged": bool(energy_result.converged),
                  "force_N": float(energy_result.force[-1])}
    ac6 = check_ac6(cfg, n_tests if n_tests is not None else 0)
    criteria = {"AC1": ac1, "AC2": ac2, "AC3": ac3, "AC4": ac4, "AC5": ac5,
                "AC6": ac6}
    summary = {k: {"passed": bool(v["passed"]), "key_metric": _key_metric(k, v)}
               for k, v in criteria.items()}
    report = {
        "node": "p233w1-N06",
        "title": "J2 elastoplastic ice indentation with penalty contact",
        "config": {
            "E_Pa": cfg.E, "nu": cfg.nu, "sigma_y_Pa": cfg.sigma_y,
            "hardening_Pa": cfg.hardening,
            "indenter_radius_m": cfg.indenter_radius,
            "half_width_m": cfg.half_width, "height_m": cfg.height,
            "delta_max_m": cfg.delta_max, "n_steps": cfg.n_steps,
            "penalty_factor": cfg.penalty_factor, "contact_model": cfg.contact_model,
            "mesh_dims": [list(d) for d in (cfg.mesh_dims if mesh_dims is None
                                            else mesh_dims)],
            "newton_tol": cfg.newton_tol,
            "energy_steps": cfg.energy_steps, "energy_ramp": cfg.energy_ramp,
        },
        "criteria": criteria,
        "summary": summary,
        "all_passed": all(bool(v["passed"]) for v in criteria.values()),
        "runs": [{k: v for k, v in r.items() if k != "_result"} for r in runs],
    }
    if return_objects:
        return report, {"study": study, "elastic": elastic, "energy": energy_result}
    return report


def _key_metric(name: str, block: dict) -> dict:
    if name == "AC1":
        return {"max_relative_error": block["max_relative_error"],
                "tolerance": block["tolerance"]}
    if name == "AC2":
        return {"relative_error": block["relative_error"],
                "tolerance": block["tolerance"]}
    if name == "AC3":
        return {"max_relative_error_elastic": block["max_relative_error"],
                "tolerance": block["tolerance"],
                "experimental_subcriterion": block["experimental_subcriterion"],
                "plastic_constraint_factor":
                    block["plastic_branch"]["constraint_factor_p_m_over_sigma_y"]}
    if name == "AC4":
        return {"relative_errors": block["relative_errors"],
                "monotone_decrease": block["monotone_decrease"],
                "observed_order": block["observed_order"]}
    if name == "AC5":
        return {"relative_residual": block["relative_residual"],
                "penalty_energy_fraction": block["penalty_energy_fraction"],
                "tolerance": block["tolerance"]}
    return {"collected": block["collected"], "required": block["required"]}