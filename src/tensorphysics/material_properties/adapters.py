"""Solver-side adapters: MPDB -> the parameters each solver actually needs.

These are the *integration samples* required by the asset spec.  Each function
is a thin, tested translation layer that returns a plain dict (JSON-serialisable)
plus the provenance of every property that entered it, so a solver run can quote
its own inputs.

Covered integrations (>= 2 solver families)
-------------------------------------------
``lbm_scaling``        TensorLBM-style lattice scaling: (rho0, nu_lat, tau, omega,
                       dx, dt, Re, Ma) from seawater/fresh-water state.
``fvm_air_state``      TensorFVM/DG-style gas state: (rho, mu, nu, k, cp, Pr, a)
                       and a ``reynolds`` helper.
``fem_ice_material``   TensorFEM-style ice plate material: (E, nu, rho, sigma_f)
                       with the brine-volume drive, plus ``plate_rigidity``.
``ship_resistance_env`` ship-performance-style towing/full-scale environment:
                       (rho, nu) + ITTC-57 friction line + Reynolds number.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional

from .correlations import (
    brine_volume_fraction,
    sea_ice_flexural_strength,
    seawater_density,
    seawater_dynamic_viscosity,
)
from .query import MaterialDB, PropertyValue

__all__ = [
    "lbm_scaling",
    "fvm_air_state",
    "fem_ice_material",
    "ship_resistance_env",
    "plate_rigidity",
    "ittc57_cf",
    "provenance_ledger",
]


def _prov(value: PropertyValue) -> Dict[str, Any]:
    return {
        "row_ids": list(value.row_ids),
        "source_id": value.provenance.get("source_id"),
        "locator": value.provenance.get("locator"),
        "evidence_class": value.provenance.get("evidence_class"),
        "unit": value.unit,
        "uncertainty_rel": value.uncertainty_rel,
    }


def _db(db: Optional[MaterialDB]) -> MaterialDB:
    return db if db is not None else MaterialDB.load()


# --------------------------------------------------------------------------- #
# LBM (kinetic solver)
# --------------------------------------------------------------------------- #
def lbm_scaling(substance: str = "water", T_C: float = 15.0, salinity: Optional[float] = None,
                L_phys: float = 1.0, U_phys: float = 1.0, n_lattice: int = 128,
                u_lattice: float = 0.05, tau_min: float = 0.51,
                db: Optional[MaterialDB] = None) -> Dict[str, Any]:
    """Diffusive-scaling LBM parameters consistent with the real fluid state.

    ``Re = U L / nu`` is preserved exactly; the lattice viscosity follows from
    the chosen lattice velocity and resolution, ``nu_lat = u_lat N / Re``, and
    ``tau = 3 nu_lat + 1/2`` (BGK, ``c_s^2 = 1/3``).  The scheme is refused when
    ``tau < tau_min`` (default 0.51): the strict BGK limit is 0.5, but within
    0.01 of it the operator is numerically unusable for engineering runs.

    Returns a dict with ``rho0, nu_phys, Re, nu_lat, tau, omega, dx, dt, Ma`` and
    ``provenance``.
    """
    database = _db(db)
    if salinity is None and substance == "seawater":
        salinity = 35.0
    cond = {"T_C": T_C}
    if salinity is not None:
        cond["salinity"] = salinity
    rho = database.get(substance, "density", **cond)
    nu = database.get(substance, "kinematic_viscosity", **cond)

    Re = U_phys * L_phys / float(nu)
    nu_lat = u_lattice * n_lattice / Re
    tau = 3.0 * nu_lat + 0.5
    if tau < tau_min:
        raise ValueError(
            f"BGK unstable: tau = {tau:.5f} < tau_min = {tau_min} "
            "(raise u_lattice or n_lattice, or lower the physical Reynolds number)")
    dx = L_phys / n_lattice
    dt = dx * u_lattice / U_phys
    cs_lat = math.sqrt(1.0 / 3.0)
    out = {
        "substance": substance,
        "T_K": cond["T_C"] + 273.15,
        "rho0": float(rho),
        "nu_phys": float(nu),
        "Re": Re,
        "nu_lat": nu_lat,
        "tau": tau,
        "omega": 1.0 / tau,
        "dx": dx,
        "dt": dt,
        "u_lattice": u_lattice,
        "Ma": u_lattice / cs_lat,
        "provenance": {"density": _prov(rho), "kinematic_viscosity": _prov(nu)},
    }
    return out


# --------------------------------------------------------------------------- #
# FVM / DG (finite-volume gas)
# --------------------------------------------------------------------------- #
def fvm_air_state(T_C: float = 20.0, p: float = 101325.0,
                  db: Optional[MaterialDB] = None) -> Dict[str, Any]:
    """Thermodynamic + transport state of dry air for a convective solver."""
    database = _db(db)
    rho = database.get("air", "density", T_C=T_C, p=p)
    mu = database.get("air", "dynamic_viscosity", T_C=T_C)
    nu = database.get("air", "kinematic_viscosity", T_C=T_C)
    k = database.get("air", "thermal_conductivity", T_C=T_C)
    cp = database.get("air", "specific_heat")
    pr = database.get("air", "prandtl_number", T_C=T_C)
    a = database.get("air", "sound_speed", T_C=T_C)
    return {
        "T_K": T_C + 273.15,
        "p_Pa": p,
        "rho": float(rho),
        "mu": float(mu),
        "nu": float(nu),
        "k": float(k),
        "cp": float(cp),
        "Pr": float(pr),
        "a": float(a),
        "R": 287.058,
        "gamma": 1.4,
        "provenance": {
            "density": _prov(rho), "dynamic_viscosity": _prov(mu),
            "kinematic_viscosity": _prov(nu), "thermal_conductivity": _prov(k),
            "specific_heat": _prov(cp), "prandtl_number": _prov(pr),
            "sound_speed": _prov(a),
        },
    }


# --------------------------------------------------------------------------- #
# FEM (ice plate)
# --------------------------------------------------------------------------- #
def fem_ice_material(T_C: float = -10.0, salinity: float = 5.0,
                     variant: str = "effective", db: Optional[MaterialDB] = None) -> Dict[str, Any]:
    """Sea-ice elastic + strength inputs for an FEM ice-plate model.

    ``youngs_modulus`` is taken from the tabulated state (``initial`` or
    ``effective``); ``flexural_strength`` is evaluated with the Timco & Weeks
    brine-volume fit at the requested temperature/salinity, which is *inside*
    the model's validity because the brine volume is computed with the same
    (T, S) pair.
    """
    database = _db(db)
    E = database.get("ice_sea", "youngs_modulus", variant=variant)
    nu = database.get("ice_sea", "poisson_ratio")
    rho = database.get("ice_sea", "bulk_density", variant="first_year")
    vb = brine_volume_fraction(T_C + 273.15, salinity)
    sigma_f = sea_ice_flexural_strength(vb)
    ref = database.get("ice_sea", "flexural_strength", T_C=T_C, salinity=salinity)
    return {
        "T_K": T_C + 273.15,
        "salinity_g_per_kg": salinity,
        "E": float(E),
        "nu": float(nu),
        "rho": float(rho),
        "brine_volume": vb,
        "sigma_f": sigma_f,
        "sigma_f_mpdb_grid": float(ref),
        "variant": variant,
        "provenance": _prov(E),
        "provenance_note": (
            "sigma_f from Timco & Weeks (2010) eq. (8) evaluated at the requested (T,S); "
            "the MPDB grid value at the nearest sampled state is given for cross-check"
        ),
    }


def plate_rigidity(E: float, nu: float, h: float) -> Dict[str, float]:
    """Plate flexural rigidity D = E h^3 / (12 (1 - nu^2)) and section modulus."""
    D = E * h ** 3 / (12.0 * (1.0 - nu ** 2))
    return {"D": D, "section_modulus": h ** 2 / 6.0, "I_per_width": h ** 3 / 12.0}


# --------------------------------------------------------------------------- #
# Ship resistance / towing tank environment
# --------------------------------------------------------------------------- #
def ittc57_cf(Re: float) -> float:
    """ITTC-57 model-ship friction coefficient Cf = 0.075 / (log10 Re - 2)^2."""
    if Re <= 0:
        raise ValueError("Re must be positive")
    return 0.075 / (math.log10(Re) - 2.0) ** 2


def ship_resistance_env(T_C: float = 15.0, salinity: Optional[float] = 35.0,
                        L: Optional[float] = None, V: Optional[float] = None,
                        db: Optional[MaterialDB] = None) -> Dict[str, Any]:
    """Water state for a resistance/self-propulsion calculation (+ optional Re, Cf)."""
    database = _db(db)
    if salinity is None:
        substance = "water"
        cond: Dict[str, Any] = {"T_C": T_C}
    else:
        substance = "seawater"
        cond = {"T_C": T_C, "salinity": salinity}
    rho = database.get(substance, "density", **cond)
    nu = database.get(substance, "kinematic_viscosity", **cond)
    out: Dict[str, Any] = {
        "substance": substance,
        "T_K": T_C + 273.15,
        "salinity_g_per_kg": salinity,
        "rho": float(rho),
        "nu": float(nu),
        "provenance": {"density": _prov(rho), "kinematic_viscosity": _prov(nu)},
    }
    if L is not None and V is not None:
        Re = V * L / float(nu)
        out["Re"] = Re
        out["Cf_ittc57"] = ittc57_cf(Re)
    return out


# --------------------------------------------------------------------------- #
# Ledger helper
# --------------------------------------------------------------------------- #
def provenance_ledger(entries: Mapping[str, Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Flatten adapter outputs into a run-ledger list (one row per consumed property)."""
    rows: List[Dict[str, Any]] = []
    for owner, blob in entries.items():
        prov = blob.get("provenance") if isinstance(blob, Mapping) else None
        if isinstance(prov, Mapping) and "source_id" in prov:
            rows.append({"consumer": owner, **dict(prov)})
            continue
        for key, item in (prov or {}).items():
            if isinstance(item, Mapping) and item.get("source_id"):
                rows.append({"consumer": f"{owner}.{key}", **dict(item)})
    return rows


def seawater_consistency_check(T_C: float = 15.0, salinity: float = 35.0,
                               db: Optional[MaterialDB] = None) -> Dict[str, float]:
    """Cross-check the MPDB seawater rows against the encoded correlations.

    Used by the tests and by ``cli validate``: a non-zero deviation means the
    packaged rows and the correlation code have drifted apart.
    """
    database = _db(db)
    T = T_C + 273.15
    rho_db = float(database.get("seawater", "density", T_C=T_C, salinity=salinity))
    mu_db = float(database.get("seawater", "dynamic_viscosity", T_C=T_C, salinity=salinity))
    rho_ref = seawater_density(T, salinity)
    mu_ref = seawater_dynamic_viscosity(T, salinity)
    return {
        "rho_rel_dev": abs(rho_db - rho_ref) / rho_ref,
        "mu_rel_dev": abs(mu_db - mu_ref) / mu_ref,
    }