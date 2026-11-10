"""Ice-going ship digital twin: parameterised forward model + model-form ensemble.

Scope
-----
This package owns the **model-validation and uncertainty-quantification** layer
of the ice-going-ship digital twin:

* Bayesian calibration of the twin's semi-empirical coefficients
  (:mod:`tensorphysics.ice_digital_twin.bayes`),
* variance-based global sensitivity of the twin response
  (:mod:`tensorphysics.ice_digital_twin.sensitivity`),
* model-form uncertainty through Bayesian model averaging over a structural
  ensemble of the component-resolved level-ice law
  (:mod:`tensorphysics.ice_digital_twin.model_form`),
* prediction-interval coverage validation by leave-one-out cross-validation
  (:mod:`tensorphysics.ice_digital_twin.validation`),
* streaming (online) re-calibration with a per-step timing budget
  (:mod:`tensorphysics.ice_digital_twin.online`).

The *physics* itself is not duplicated here: the level-ice law is
:func:`tensorphysics.ice.resistance.level_ice_resistance` (Lindqvist 1989,
component-resolved, REAL). This module re-expresses that closed form in a
vectorised, parameter-vectorised way so that tens of thousands of evaluations
inside an MCMC loop cost milliseconds, and it is **anchored** to the reference
implementation by an exact-agreement test (see
``tests/test_p223w3_n09_digital_twin_uq.py::test_vectorised_forward_matches_reference_law``).

Calibration parameterisation
----------------------------
Four physical coefficients, each a **dimensionless multiplier on a literature
reference value** so that ``theta = 1`` reproduces the published law exactly and
relative errors are directly comparable with the literature:

======================  ======================  =================  ===============
name                    reference value          bounds             source
======================  ======================  =================  ===============
``sigma_c_scale``       sigma_c = 5.0 MPa       [0.50, 2.00]       ITTC(2017)/ISO 19906 2-8 MPa
``sigma_f_scale``       sigma_f = 0.5 MPa       [0.50, 2.00]       Timco & Weeks (2010) 0.35-0.7 MPa
``mu_hull_ice``         mu = 0.15               [0.05, 0.30]       Riska (1997) 0.05-0.25
``C_sub_scale``         C_s = 0.15+0.3*Fr_h     [0.50, 2.00]       Lindqvist (1989)
======================  ======================  =================  ===============

A fifth calibration parameter ``kappa`` (log-scale coefficient of variation of
the multiplicative model-inadequacy term, bounds ``[0, 0.30]``) is owned by the
likelihood (:func:`tensorphysics.ice_digital_twin.bayes.log_posterior_factory`)
and is *not* a physical property of the ship/ice system; it carries the
model-form error of the ensemble member being calibrated (Kennedy & O'Hagan
2001 style multiplicative discrepancy).

Evidence grading (TensorPhysics vocabulary)
-------------------------------------------
* forward model      : REAL -- closed-form Lindqvist law, anchored to
  :func:`tensorphysics.ice.resistance.level_ice_resistance` (identical to 1e-12).
* calibration layer  : REAL -- adaptive Metropolis MCMC with split-R, ESS.
* measurement data   : SYNTHETIC/REAL? -> ``WITHHELD`` for real full-scale data:
  no digitised full-scale ice-resistance record exists in this repository, so
  every numerical exercise here runs on a **synthetic validation corpus**
  generated from a declared truth (negative-control design). Coverage and
  calibration numbers therefore establish *machinery correctness*, not
  full-scale accuracy.
* model-form ensemble : STRUCTURAL VARIANTS -- see :data:`FORMS`; the members
  are component subsets of the same closed form, not independent literature
  laws. They span the *structural incompleteness* axis of model-form
  uncertainty only.

References
----------
* Lindqvist, G. (1989) POAC'89, Vol. 2, 722-735.
* Kennedy, M.C. & O'Hagan, A. (2001) "Bayesian calibration of computer models",
  JRSS-B 63(3), 425-464.
* Haario, H., Saksman, E. & Tamminen, J. (2001) "An adaptive Metropolis
  algorithm", Bernoulli 7(2), 223-242.
* Gelman, A. et al. (2013) "Bayesian Data Analysis", 3rd ed., ch. 11.
* Yao, Y. et al. (2018) "Using stacking to average Bayesian predictive
  distributions", Bayesian Analysis 13(3), 917-1007.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..ice.resistance import (
    G,
    MU_HULL_ICE,
    PARAMETER_PROVENANCE,
    RHO_ICE,
    SIGMA_C_REF,
    SIGMA_F_REF,
    DomainLimits,
    ShipGeometry,
    level_ice_resistance,
)

__all__ = [
    "PARAM_NAMES",
    "PARAM_BOUNDS",
    "PARAM_REFERENCE",
    "PARAM_SOURCE",
    "KAPPA_BOUNDS",
    "KAPPA_REFERENCE",
    "ModelForm",
    "FORMS",
    "DEFAULT_FORM",
    "resistance_components",
    "resistance_kN",
    "resistance_kN_batch",
    "unit_to_physical",
    "physical_to_unit",
    "parameter_reference_table",
    "velocity_max_ms",
]

# ---------------------------------------------------------------------------
# calibration parameterisation
# ---------------------------------------------------------------------------

PARAM_NAMES: Tuple[str, ...] = (
    "sigma_c_scale",
    "sigma_f_scale",
    "mu_hull_ice",
    "C_sub_scale",
)

#: Box constraints of the physical multipliers/albedo (see module docstring).
PARAM_BOUNDS: np.ndarray = np.array(
    [
        [0.50, 2.00],  # sigma_c multiplier  (5 MPa -> 2.5 .. 10 MPa)
        [0.50, 2.00],  # sigma_f multiplier  (0.5 MPa -> 0.25 .. 1.0 MPa)
        [0.05, 0.30],  # hull-ice friction coefficient
        [0.50, 2.00],  # submergence coefficient multiplier
    ],
    dtype=float,
)

PARAM_REFERENCE: Dict[str, float] = {
    "sigma_c_scale": 1.0,
    "sigma_f_scale": 1.0,
    "mu_hull_ice": MU_HULL_ICE,
    "C_sub_scale": 1.0,
}

PARAM_SOURCE: Dict[str, str] = {
    "sigma_c_scale": "ITTC(2017)/ISO 19906:2019 compressive strength 2-8 MPa, ref 5 MPa",
    "sigma_f_scale": "Timco & Weeks (2010) flexural strength 0.35-0.7 MPa, ref 0.5 MPa",
    "mu_hull_ice": "Riska (1997) steel-ice friction 0.05-0.25, ref 0.15",
    "C_sub_scale": "Lindqvist (1989) submergence coefficient C_s = 0.15+0.3 Fr_h, ref scale 1",
}

#: Model-inadequacy (multiplicative log-scale) parameter: prior bounds and the
#: value used when the discrepancy term is switched off.
KAPPA_BOUNDS: Tuple[float, float] = (0.0, 0.30)
KAPPA_REFERENCE: float = 0.10


def k_bend_reference() -> float:
    """Bending-shape constant used by the reference law (from provenance table)."""
    return float(PARAMETER_PROVENANCE["k_bend"]["value"])


def parameter_reference_table() -> List[Dict[str, Any]]:
    """Tabulate the calibration parameters, bounds, reference values and sources."""
    rows: List[Dict[str, Any]] = []
    for i, name in enumerate(PARAM_NAMES):
        rows.append(
            {
                "index": i,
                "name": name,
                "reference": PARAM_REFERENCE[name],
                "lower": float(PARAM_BOUNDS[i, 0]),
                "upper": float(PARAM_BOUNDS[i, 1]),
                "unit": "dimensionless multiplier" if name != "mu_hull_ice" else "-",
                "source": PARAM_SOURCE[name],
            }
        )
    return rows


# ---------------------------------------------------------------------------
# unit-cube <-> physical mapping (Sobol' design lives on the unit cube)
# ---------------------------------------------------------------------------


def unit_to_physical(unit: Any, bounds: Optional[np.ndarray] = None) -> np.ndarray:
    """Map unit-cube coordinates ``[0,1]^d`` to the physical box ``bounds``."""
    u = np.asarray(unit, dtype=float)
    b = PARAM_BOUNDS if bounds is None else np.asarray(bounds, dtype=float)
    if u.shape[-1] != b.shape[0]:
        raise ValueError(f"unit last dimension {u.shape[-1]} != {b.shape[0]} bounds")
    lo, hi = b[:, 0], b[:, 1]
    return lo + u * (hi - lo)


def physical_to_unit(phys: Any, bounds: Optional[np.ndarray] = None) -> np.ndarray:
    """Inverse of :func:`unit_to_physical` (clipped to ``[0, 1]``)."""
    x = np.asarray(phys, dtype=float)
    b = PARAM_BOUNDS if bounds is None else np.asarray(bounds, dtype=float)
    lo, hi = b[:, 0], b[:, 1]
    u = (x - lo) / (hi - lo)
    if np.any(u < -1e-9) or np.any(u > 1.0 + 1e-9):
        raise ValueError("physical parameters outside the declared box")
    return np.clip(u, 0.0, 1.0)


def velocity_max_ms(ship: ShipGeometry, limits: Optional[DomainLimits] = None) -> float:
    """Froude-implied top speed of the reference law's validity domain."""
    return (limits or DomainLimits()).v_limit(ship)


# ---------------------------------------------------------------------------
# vectorised component-resolved forward model
# ---------------------------------------------------------------------------


def resistance_components(
    sigma_c_scale: float,
    sigma_f_scale: float,
    mu: float,
    c_sub_scale: float,
    h: Any,
    V: Any,
    ship: ShipGeometry,
    *,
    k_bend: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Lindqvist (1989) components ``(R_crush, R_break, R_submersion)`` in newton.

    This is line-for-line the algebra of
    :func:`tensorphysics.ice.resistance.level_ice_resistance`, only rewritten so
    that ``h``/``V`` may be numpy arrays and the coefficients are explicit
    multipliers.  It does **no** domain clipping: callers must stay inside
    :class:`tensorphysics.ice.resistance.DomainLimits` (the campaign corpus does).

        R_c = 0.5 (tan a + mu / cos a) sigma_c h^2
        R_b = 0.5 (sin(phi/2) + mu cos(phi/2)) sigma_f h B / k_bend
        R_s = C_sub_scale (0.15 + 0.3 Fr_h) rho_i V^2 h B,  Fr_h = V / sqrt(g h)
    """
    hh = np.asarray(h, dtype=float)
    vv = np.asarray(V, dtype=float)
    if np.any(hh <= 0.0):
        raise ValueError("ice thickness must be strictly positive")
    if np.any(vv < 0.0):
        raise ValueError("ship speed must be non-negative")

    kb = k_bend_reference() if k_bend is None else float(k_bend)
    alpha = ship.flare_rad
    phi = ship.bow_angle_rad

    c_crush = 0.5 * (math.tan(alpha) + float(mu) / math.cos(alpha))
    r_crush = c_crush * (float(sigma_c_scale) * SIGMA_C_REF) * np.square(hh)

    c_break = 0.5 * (math.sin(0.5 * phi) + float(mu) * math.cos(0.5 * phi)) / kb
    r_break = c_break * (float(sigma_f_scale) * SIGMA_F_REF) * hh * ship.B

    fr_h = vv / np.sqrt(G * hh)
    c_sub = float(c_sub_scale) * (0.15 + 0.3 * fr_h)
    r_sub = c_sub * RHO_ICE * np.square(vv) * hh * ship.B

    return r_crush, r_break, r_sub


# ---------------------------------------------------------------------------
# model-form ensemble (structural variants of the same component-resolved law)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelForm:
    """One ensemble member: a structural variant of the component-resolved law.

    ``components`` selects which of the three Lindqvist mechanisms are retained;
    ``param_names`` lists the coefficients that remain identifiable once a
    mechanism is dropped (dropping a mechanism removes its coefficient from the
    inverse problem instead of leaving an unidentifiable parameter wandering in
    the prior -- an unidentifiable coefficient would inflate the predictive
    variance for purely mathematical reasons).
    """

    key: str
    label: str
    components: Tuple[str, ...]
    param_names: Tuple[str, ...]
    note: str = ""

    @property
    def n_phys(self) -> int:
        return len(self.param_names)

    def full_vector(self, sub: Sequence[float]) -> np.ndarray:
        """Expand a member-specific physical vector into the full 4-vector."""
        if len(sub) != self.n_phys:
            raise ValueError(
                f"{self.key}: expected {self.n_phys} parameters, got {len(sub)}"
            )
        full = np.array([PARAM_REFERENCE[n] for n in PARAM_NAMES], dtype=float)
        for value, name in zip(sub, self.param_names):
            full[PARAM_NAMES.index(name)] = float(value)
        return full

    def bounds(self) -> np.ndarray:
        return np.array([PARAM_BOUNDS[PARAM_NAMES.index(n)] for n in self.param_names])

    def as_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "components": list(self.components),
            "param_names": list(self.param_names),
            "note": self.note,
        }


#: Structural ensemble.  ``full`` is the reference law (all three mechanisms);
#: the other members drop one mechanism each and stand for the simplifying
#: assumptions one finds in the engineering literature (submergence neglected at
#: low speed, crushing neglected for thin ice / strong bending regime, bending
#: neglected in the "crushing + friction" simplification).  They are *not*
#: independent literature laws -- see the module docstring.
FORMS: Dict[str, ModelForm] = {
    "full": ModelForm(
        key="full",
        label="Lindqvist full (crush + bending + submergence)",
        components=("crush", "break", "submersion"),
        param_names=("sigma_c_scale", "sigma_f_scale", "mu_hull_ice", "C_sub_scale"),
        note="reference structural form, all three mechanisms retained",
    ),
    "crush_break": ModelForm(
        key="crush_break",
        label="crush + bending (submergence dropped)",
        components=("crush", "break"),
        param_names=("sigma_c_scale", "sigma_f_scale", "mu_hull_ice"),
        note="low-speed / low-Froude simplification: broken-ice submersion neglected",
    ),
    "crush_sub": ModelForm(
        key="crush_sub",
        label="crush + submergence (bending dropped)",
        components=("crush", "submersion"),
        param_names=("sigma_c_scale", "mu_hull_ice", "C_sub_scale"),
        note="crushing-and-friction simplification with hydrodynamic submergence",
    ),
    "break_sub": ModelForm(
        key="break_sub",
        label="bending + submergence (crush dropped)",
        components=("break", "submersion"),
        param_names=("sigma_f_scale", "mu_hull_ice", "C_sub_scale"),
        note="thin-ice / bending-dominated simplification",
    ),
}

DEFAULT_FORM: str = "full"


def get_form(form: Any) -> ModelForm:
    """Resolve a form key or :class:`ModelForm` to a member of :data:`FORMS`."""
    if isinstance(form, ModelForm):
        return form
    try:
        return FORMS[str(form)]
    except KeyError as exc:  # pragma: no cover - defensive
        raise KeyError(f"unknown model form {form!r}; known: {sorted(FORMS)}") from exc


def resistance_kN(
    form: Any,
    sub_params: Sequence[float],
    h: Any,
    V: Any,
    ship: ShipGeometry,
) -> np.ndarray:
    """Ensemble-member resistance in kN for a member-specific parameter vector."""
    f = get_form(form)
    sigma_c_scale, sigma_f_scale, mu, c_sub = f.full_vector(sub_params)
    rc, rb, rs = resistance_components(
        sigma_c_scale, sigma_f_scale, mu, c_sub, h, V, ship
    )
    parts = {"crush": rc, "break": rb, "submersion": rs}
    total = np.zeros_like(np.asarray(h, dtype=float))
    for name in f.components:
        total = total + parts[name]
    return total / 1e3


def resistance_kN_batch(
    form: Any,
    phys: np.ndarray,
    h: float,
    V: float,
    ship: ShipGeometry,
) -> np.ndarray:
    """Twin resistance (kN) for a **batch of parameter vectors at one condition**.

    ``phys`` has shape ``(n, n_phys)`` with ``n_phys = get_form(form).n_phys``;
    ``h`` and ``V`` are scalars (a single ice condition).  Used by the Sobol'
    design and by the particle-based online calibrator, where tens of thousands of
    parameter draws must be evaluated at one condition per call.
    """
    f = get_form(form)
    th = np.atleast_2d(np.asarray(phys, dtype=float))
    if th.shape[1] != f.n_phys:
        raise ValueError(f"expected (n, {f.n_phys}) parameters, got {th.shape}")
    n = th.shape[0]
    full = np.tile(
        np.array([PARAM_REFERENCE[k] for k in PARAM_NAMES], dtype=float), (n, 1)
    )
    full[:, [PARAM_NAMES.index(k) for k in f.param_names]] = th
    sigma_c = full[:, 0] * SIGMA_C_REF
    sigma_f = full[:, 1] * SIGMA_F_REF
    mu = full[:, 2]
    c_sub = full[:, 3]

    alpha, phi = ship.flare_rad, ship.bow_angle_rad
    kb = k_bend_reference()
    r_crush = 0.5 * (math.tan(alpha) + mu / math.cos(alpha)) * sigma_c * h**2
    r_break = (
        0.5 * (math.sin(0.5 * phi) + mu * math.cos(0.5 * phi)) / kb
        * sigma_f * h * ship.B
    )
    r_sub = c_sub * (0.15 + 0.3 * V / math.sqrt(G * h)) * RHO_ICE * V**2 * h * ship.B
    parts = {"crush": r_crush, "break": r_break, "submersion": r_sub}
    total = np.zeros(n)
    for name in f.components:
        total = total + parts[name]
    return total / 1e3


def component_breakdown_kN(
    sub_params: Sequence[float],
    h: Any,
    V: Any,
    ship: ShipGeometry,
) -> Dict[str, np.ndarray]:
    """Per-mechanism resistance breakdown (kN) for reporting."""
    sigma_c_scale, sigma_f_scale, mu, c_sub = get_form("full").full_vector(sub_params)
    rc, rb, rs = resistance_components(
        sigma_c_scale, sigma_f_scale, mu, c_sub, h, V, ship
    )
    return {
        "R_crush_kN": np.asarray(rc) / 1e3,
        "R_break_kN": np.asarray(rb) / 1e3,
        "R_submersion_kN": np.asarray(rs) / 1e3,
        "R_total_kN": (np.asarray(rc) + np.asarray(rb) + np.asarray(rs)) / 1e3,
    }


def reference_law_kN(
    h: Any, V: Any, ship: ShipGeometry, **kwargs: Any
) -> np.ndarray:
    """Call the repository's reference law (``ice.resistance``) for anchoring.

    Kept as a thin shim so the agreement test compares the vectorised twin
    against the independently maintained implementation, not against itself.
    """
    hh = np.atleast_1d(np.asarray(h, dtype=float))
    vv = np.atleast_1d(np.asarray(V, dtype=float))
    out = np.empty(hh.shape[0])
    for i in range(hh.shape[0]):
        res = level_ice_resistance(float(hh[i]), float(vv[i]), ship, **kwargs)
        out[i] = float(res["R_total_kN"])
    return out