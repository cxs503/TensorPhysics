"""Measured-data uncertainty-quantification face of the ice-going digital twin.

Node ``p223w2b-n03-physics-uq-digital-twin`` (TensorPhysics).

What is new relative to :mod:`tensorphysics.ice_digital_twin` (node
``p223w3-n09``, which validated the UQ *machinery* on a synthetic corpus)
-----------------------------------------------------------------------
This module runs the whole UQ stack on the **measured** level-ice resistance
corpus already registered in this repository
(:mod:`tensorphysics.ice.measured_anchor`, series ``MTUIKKU-HZ2016``: 10
full-scale points of the MT Uikku ice-basin campaign, Hu & Zhou 2016,
DOI 10.1016/j.ijnaoe.2016.01.004).  Predictive statements here are therefore
graded ``REAL (measured)`` rather than ``WITHHELD``, and the *model-form* error
is treated explicitly with a Kennedy-O'Hagan (2001) discrepancy function instead
of being hidden inside an inflated noise term.

The five faces
--------------
1. **Bayesian calibration** (:func:`calibrate`): the twin's four physical
   coefficients (multipliers on the *measured* per-point ice strengths) plus a
   power-law model-inadequacy function ``delta(h) = exp(beta0) (h/h_bar)^beta1``
   and a residual inadequacy coefficient ``kappa``.  Adaptive Metropolis
   (Haario et al. 2001) with over-dispersed chains, split-R-hat and ESS.
2. **Identifiability diagnostics** (:func:`prior_sensitivity`): the same
   posterior under *flat* priors, exposing how much of the posterior location is
   data information versus literature anchoring - required because ten points
   cannot identify four coefficients plus a discrepancy amplitude.
3. **Variance-based global sensitivity** (:func:`sobol_exact`,
   :func:`sobol_mc`): total-order Sobol' indices of the twin response, with an
   *exact* tensor-quadrature ANOVA decomposition as the reference against which
   the pick-freeze Monte-Carlo estimator is checked (closure test).
4. **Predictive-interval coverage** (:func:`loo_predictive`,
   :func:`negative_control`): exact leave-one-out refit predictive intervals on
   the measured points, plus a *negative control* (literature coefficients, no
   discrepancy) that must **fail** coverage - the falsification test that gives
   the coverage criterion its teeth.
5. **Model-form averaging** (:func:`bma_published_laws`): pseudo-BMA
   (Yao et al. 2018) over the four independent literature level-ice laws whose
   predictions are transcribed in the same measured asset (Lindqvist, Riska,
   Jeong, Keinonen, as tabulated by Hu & Zhou 2016).

Evidence grading
----------------
* measured corpus : REAL (``MEASURED-PRIMARY``, sha256 pinned in
  ``data/measured/ice_resistance/registry.json``).
* forward law : REAL (Lindqvist 1989 closed form, algebra identical to
  :func:`tensorphysics.ice_digital_twin.twin.resistance_components`, pinned by
  ``tests/test_p223w2b_n03_measured_uq.py::test_vectorised_forward_matches_reference``);
  its adequacy against the measured points is **poor** and is quantified, not
  repaired, here.
* calibration / sensitivity / coverage / BMA : REAL mechanisms on REAL data.
* absolute-accuracy claim : inherits the honest FAIL of node ``p273w3-n07``
  (median 198 % against the same measured series); this node attributes it to a
  single multiplicative + thickness-power model-form error (posterior
  ``exp(beta0)`` well below one) rather than to ice-strength parameters.

Geometry / convention
---------------------
The measured asset declares ``bow_angle_deg = 21`` as the *plan-view stem
half-angle* argument (``phi/2``) of the bending closure, while the twin's
forward law applies ``0.5 * ship.bow_angle_rad``.  :func:`documented_ship`
therefore builds the geometry with ``bow_angle_deg = 2 * 21``; ``flare_deg = 30``
is used as the plating angle to the horizontal (asset assumption A1).

References
----------
* Lindqvist, G. (1989) POAC'89, vol. 2, 722-735.
* Hu, Z. & Zhou, L. (2016) Int. J. Naval Arch. Ocean Eng. 8(2), 169-176.
* Kennedy, M.C. & O'Hagan, A. (2001) JRSS-B 63(3), 425-464.
* Haario, H., Saksman, E. & Tamminen, J. (2001) Bernoulli 7(2), 223-242.
* Saltelli, A. et al. (2010) Computer Physics Communications 181, 259-270.
* Yao, Y. et al. (2018) Bayesian Analysis 13(3), 917-1007.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..ice.resistance import G, RHO_ICE, ShipGeometry
from .bayes import ChainSet, coverage_rate, digest, ess_min, predictive_interval, split_rhat
from .model_form import pseudo_bma_weights
from .twin import k_bend_reference

__all__ = [
    "Discrepancy",
    "LITERATURE_REFERENCE",
    "MeasuredCampaign",
    "MeasuredPoint",
    "PARAM_NAMES_MEASURED",
    "PRIOR_BOUNDS",
    "PRIOR_LITERATURE",
    "PUBLISHED_LAWS",
    "bma_published_laws",
    "calibrate",
    "discrepancy_factor",
    "documented_ship",
    "forward_components_kN",
    "forward_kN",
    "load_measured_campaign",
    "load_published_laws",
    "log_posterior_factory",
    "loo_predictive",
    "negative_control",
    "posterior_summary",
    "prior_sensitivity",
    "run_measured_uq",
    "run_metropolis",
    "sobol_exact",
    "sobol_mc",
    "write_report",
]

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SERIES_ID = "MTUIKKU-HZ2016"
SERIES_REL = "data/measured/ice_resistance/series/mtuikku_hz2016/series.json"
PASS_A_REL = "data/measured/ice_resistance/series/mtuikku_hz2016/pass_a.json"
REGISTRY_REL = "data/measured/ice_resistance/registry.json"

#: Literature law predictions transcribed in the measured asset
#: (Hu & Zhou 2016, Table 8: experimental reference + published predictions).
PUBLISHED_LAWS: Tuple[str, ...] = ("Lindqvist", "Riska", "Jeong", "Keinonen")

#: Calibration parameters, ordered.  The first four are physical multipliers
#: acting on the *measured* per-point quantities (``1.0`` = "trust the
#: measurement"); the last three carry the model-form (Kennedy-O'Hagan) structure.
PARAM_NAMES_MEASURED: Tuple[str, ...] = (
    "sigma_c_scale",
    "sigma_f_scale",
    "mu_hull_ice",
    "C_sub_scale",
    "beta0",
    "beta1",
    "kappa",
)

PRIOR_BOUNDS: np.ndarray = np.array(
    [
        [0.50, 2.00],    # sigma_c multiplier on the measured crush strength
        [0.50, 2.00],    # sigma_f multiplier on the measured flexural strength
        [0.05, 0.30],    # hull-ice friction coefficient (Riska 1997: 0.05-0.25)
        [0.50, 2.00],    # submergence/friction coefficient multiplier
        [-3.00, 3.00],   # log multiplicative discrepancy amplitude
        [-1.50, 1.50],   # log-thickness exponent of the discrepancy function
        [0.01, 0.60],    # residual inadequacy coefficient of variation
    ],
    dtype=float,
)

#: Literature-anchored prior specification.  ``lognormal`` is Gaussian in log
#: space about the reference value (``sigma`` in dex), ``normal`` Gaussian in
#: physical units, ``uniform`` the box.  Log-scales follow the cited ranges:
#: crush strength 2-8 MPa about 5 MPa, flexural strength 0.35-0.7 MPa about
#: 0.5 MPa, friction 0.05-0.25 about 0.15.
PRIOR_LITERATURE: Dict[str, Dict[str, Any]] = {
    "sigma_c_scale": {"kind": "lognormal", "center": 1.0, "sigma": 0.18},
    "sigma_f_scale": {"kind": "lognormal", "center": 1.0, "sigma": 0.18},
    "mu_hull_ice": {"kind": "normal", "center": 0.15, "sigma": 0.05},
    "C_sub_scale": {"kind": "lognormal", "center": 1.0, "sigma": 0.18},
    "beta0": {"kind": "normal", "center": 0.0, "sigma": 0.70},
    "beta1": {"kind": "normal", "center": 0.0, "sigma": 0.30},
    "kappa": {"kind": "uniform", "center": 0.0, "sigma": 0.0},
}

#: Reference (literature) values used by the "posterior vs literature" criterion.
LITERATURE_REFERENCE: Dict[str, float] = {
    "sigma_c_scale": 1.0,
    "sigma_f_scale": 1.0,
    "mu_hull_ice": 0.15,
    "C_sub_scale": 1.0,
}

#: Fractional standard uncertainty imputed where the source table reports no
#: force band (3 of the 10 points); equals the median relative band of the 7
#: reported points.  The imputation is flagged per point and reported.
UNC_IMPUTATION_FRACTION = 0.23

MAP_START_DEFAULT = np.array([1.0, 1.0, 0.15, 1.0, -1.0, 0.0, 0.25], dtype=float)


# ---------------------------------------------------------------------------
# measured corpus
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MeasuredPoint:
    """One full-scale measured level-ice resistance point."""

    point_id: str
    h_m: float
    v_ms: float
    sigma_c_pa: float
    sigma_f_pa: float
    r_meas_kN: float
    sigma_r_kN: float
    sigma_r_imputed: bool
    r_max_kN: Optional[float] = None
    r_min_kN: Optional[float] = None
    published_kN: Mapping[str, float] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "point_id": self.point_id,
            "h_m": self.h_m,
            "V_ms": self.v_ms,
            "sigma_c_MPa": self.sigma_c_pa / 1e6,
            "sigma_f_MPa": self.sigma_f_pa / 1e6,
            "R_meas_kN": self.r_meas_kN,
            "sigma_R_kN": self.sigma_r_kN,
            "sigma_R_imputed": self.sigma_r_imputed,
            "published_kN": dict(self.published_kN),
        }


@dataclass(frozen=True)
class MeasuredCampaign:
    """A measured resistance campaign plus its provenance digest."""

    points: Tuple[MeasuredPoint, ...]
    ship: ShipGeometry
    series_id: str
    sha256: str
    source_path: str
    doi: Optional[str] = None

    def __len__(self) -> int:
        return len(self.points)

    @property
    def h_m(self) -> np.ndarray:
        return np.array([p.h_m for p in self.points], dtype=float)

    @property
    def v_ms(self) -> np.ndarray:
        return np.array([p.v_ms for p in self.points], dtype=float)

    @property
    def sigma_c_pa(self) -> np.ndarray:
        return np.array([p.sigma_c_pa for p in self.points], dtype=float)

    @property
    def sigma_f_pa(self) -> np.ndarray:
        return np.array([p.sigma_f_pa for p in self.points], dtype=float)

    @property
    def r_meas_kN(self) -> np.ndarray:
        return np.array([p.r_meas_kN for p in self.points], dtype=float)

    @property
    def sigma_r_kN(self) -> np.ndarray:
        return np.array([p.sigma_r_kN for p in self.points], dtype=float)

    @property
    def point_ids(self) -> Tuple[str, ...]:
        return tuple(p.point_id for p in self.points)

    def subset(self, index: Sequence[int]) -> "MeasuredCampaign":
        idx = np.asarray(index, dtype=int).reshape(-1)
        return MeasuredCampaign(
            points=tuple(self.points[int(i)] for i in idx),
            ship=self.ship,
            series_id=self.series_id,
            sha256=self.sha256,
            source_path=self.source_path,
            doi=self.doi,
        )

    def published_matrix(
        self, laws: Iterable[str] = PUBLISHED_LAWS
    ) -> Tuple[Tuple[str, ...], np.ndarray]:
        keys = tuple(laws)
        mat = np.array(
            [[p.published_kN.get(k, float("nan")) for k in keys] for p in self.points],
            dtype=float,
        )
        return keys, mat

    def fingerprint(self) -> Dict[str, Any]:
        return {
            "series_id": self.series_id,
            "sha256": self.sha256,
            "source_path": self.source_path,
            "n_points": len(self.points),
            "point_ids": list(self.point_ids),
            "doi": self.doi,
            "ship": self.ship.as_dict(),
        }


def documented_ship(asset: Mapping[str, Any]) -> ShipGeometry:
    """Build the twin geometry under the measured asset's declared convention.

    The asset reports the waterline entrance angle (21 deg) as the argument
    ``phi/2`` of the bending closure, while the twin's law applies
    ``0.5 * bow_angle_rad``; the geometry is therefore built with
    ``bow_angle_deg = 2 * value``.  The mapping lives in one place so it can be
    tested.
    """
    ship_asset = asset["ship"]
    half_angle_deg = float(ship_asset["bow_angle_deg"])
    return ShipGeometry(
        name=str(ship_asset.get("name", SERIES_ID)),
        L_pp=float(ship_asset["L_pp_m"]),
        B=float(ship_asset["B_m"]),
        T=float(ship_asset["T_m"]),
        bow_angle_deg=2.0 * half_angle_deg,
        flare_deg=float(ship_asset["flare_deg"]),
    )


def _sha256_of(path: str) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def load_measured_campaign(
    series_path: Optional[str] = None,
    registry_path: Optional[str] = None,
    *,
    verify_sha256: bool = True,
) -> MeasuredCampaign:
    """Load the registered measured level-ice series (REAL data, sha256 pinned).

    The file read is the canonical ``series.json`` of series ``MTUIKKU-HZ2016``,
    whose digest is pinned in ``data/measured/ice_resistance/registry.json``.
    Per-point force bands, measured ice strengths and the published literature
    predictions are all taken from the same digest-verified record.
    """
    sp = series_path or os.path.join(REPO_ROOT, SERIES_REL)
    rp = registry_path or os.path.join(REPO_ROOT, REGISTRY_REL)
    with open(sp, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    sha = _sha256_of(sp)
    if verify_sha256:
        with open(rp, "r", encoding="utf-8") as fh:
            reg = json.load(fh)
        entry = next((s for s in reg["series"] if s["series_id"] == SERIES_ID), None)
        if entry is None:
            raise ValueError(f"{SERIES_ID} missing from registry {rp}")
        if entry["sha256"] != sha:
            raise ValueError(
                f"measured series digest mismatch: registry {entry['sha256']} != file {sha}"
            )
    ship = documented_ship(raw)
    points: List[MeasuredPoint] = []
    baselines = {"lindqvist": "Lindqvist", "riska": "Riska", "jeong": "Jeong", "keinonen": "Keinonen"}
    for rec in raw["points"]:
        r_meas = float(rec["R_meas_kN"])
        band = rec.get("R_unc_kN")
        imputed = band is None
        sigma = float(band) if not imputed else UNC_IMPUTATION_FRACTION * abs(r_meas)
        lit = rec.get("literature_baselines_kN") or {}
        published = {
            law: (float(lit[key]) if lit.get(key) is not None else float("nan"))
            for key, law in baselines.items()
        }
        points.append(
            MeasuredPoint(
                point_id=str(rec["point_id"]),
                h_m=float(rec["h_m"]),
                v_ms=float(rec["V_ms"]),
                sigma_c_pa=float(rec["sigma_c_kPa"]) * 1e3,
                sigma_f_pa=float(rec["sigma_f_kPa"]) * 1e3,
                r_meas_kN=r_meas,
                sigma_r_kN=sigma,
                sigma_r_imputed=imputed,
                r_max_kN=None if rec.get("R_max_kN") is None else float(rec["R_max_kN"]),
                r_min_kN=None if rec.get("R_min_kN") is None else float(rec["R_min_kN"]),
                published_kN=published,
            )
        )
    return MeasuredCampaign(
        points=tuple(points),
        ship=ship,
        series_id=str(raw.get("series_id", SERIES_ID)),
        sha256=sha,
        source_path=os.path.relpath(sp, REPO_ROOT),
        doi=(raw.get("provenance") or {}).get("doi"),
    )


def double_entry_check(
    campaign: Optional[MeasuredCampaign] = None,
    *,
    pass_a_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Cross-check the canonical series against the independent pass-A transcript.

    The measured asset ships two independent read orders (pass A / pass B) of the
    same source tables.  This re-reads the pass-A file and compares the measured
    forces point by point: a mismatch would invalidate the corpus.
    """
    camp = campaign or load_measured_campaign()
    path = pass_a_path or os.path.join(REPO_ROOT, PASS_A_REL)
    if not os.path.exists(path):
        return {"available": False, "path": path}
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    checked = 0
    mismatches: List[Dict[str, Any]] = []
    for p in camp.points:
        rec = raw.get(p.point_id)
        if not isinstance(rec, dict) or rec.get("R_meas_kN") is None:
            continue
        checked += 1
        if abs(float(rec["R_meas_kN"]) - p.r_meas_kN) > 1e-9:
            mismatches.append(
                {"point_id": p.point_id, "series_kN": p.r_meas_kN,
                 "pass_a_kN": float(rec["R_meas_kN"])}
            )
    return {
        "available": True,
        "path": os.path.relpath(path, REPO_ROOT),
        "sha256": _sha256_of(path),
        "n_points_checked": checked,
        "n_mismatch": len(mismatches),
        "mismatches": mismatches,
        "agreement": checked > 0 and not mismatches,
    }


def load_published_laws(campaign: Optional[MeasuredCampaign] = None) -> Dict[str, float]:
    """Median ratio ``published / measured`` per literature law (diagnostic)."""
    camp = campaign or load_measured_campaign()
    keys, mat = camp.published_matrix()
    y = camp.r_meas_kN
    return {k: float(np.median(mat[:, i] / y)) for i, k in enumerate(keys)}


# ---------------------------------------------------------------------------
# forward model (vectorised over the campaign, measured per-point strengths)
# ---------------------------------------------------------------------------


def _geometry_factors(ship: ShipGeometry) -> Tuple[float, float, float]:
    return k_bend_reference(), ship.flare_rad, ship.bow_angle_rad


def forward_components_kN(
    campaign: MeasuredCampaign,
    sigma_c_scale: float,
    sigma_f_scale: float,
    mu: float,
    c_sub_scale: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(R_crush, R_break, R_submersion)`` in kN for every measured point.

    Line-for-line the algebra of
    :func:`tensorphysics.ice_digital_twin.twin.resistance_components`
    (Lindqvist 1989)::

        R_c = 0.5 (tan a + mu / cos a) sigma_c h^2
        R_b = 0.5 (sin(phi/2) + mu cos(phi/2)) sigma_f h B / k_bend
        R_s = C_sub_scale (0.15 + 0.3 Fr_h) rho_i V^2 h B,  Fr_h = V / sqrt(g h)

    vectorised over the campaign, with the **measured** per-point ice strengths
    scaled by the calibration multipliers.
    """
    kb, alpha, phi = _geometry_factors(campaign.ship)
    h = campaign.h_m
    v = campaign.v_ms
    sigma_c = float(sigma_c_scale) * campaign.sigma_c_pa
    sigma_f = float(sigma_f_scale) * campaign.sigma_f_pa
    c_crush = 0.5 * (math.tan(alpha) + float(mu) / math.cos(alpha))
    c_break = 0.5 * (math.sin(0.5 * phi) + float(mu) * math.cos(0.5 * phi)) / kb
    r_crush = c_crush * sigma_c * np.square(h)
    r_break = c_break * sigma_f * h * campaign.ship.B
    r_sub = (
        float(c_sub_scale)
        * (0.15 + 0.3 * v / np.sqrt(G * h))
        * RHO_ICE
        * np.square(v)
        * h
        * campaign.ship.B
    )
    return r_crush / 1e3, r_break / 1e3, r_sub / 1e3


def discrepancy_factor(beta0: float, beta1: float, h_m: Any, h_ref_m: float) -> np.ndarray:
    """Kennedy-O'Hagan multiplicative discrepancy ``exp(beta0) (h/h_ref)^beta1``.

    A thickness power law is the physically motivated structure: the error of
    the classic level-ice closures is concentrated in the bending term (node
    ``p273w3-n07``: 198 % median error, 62 % of the prediction carried by
    ``R_break``), and that term scales with ``h``; the log-log (power-law)
    description is the lowest-order structure that captures it.
    """
    hh = np.asarray(h_m, dtype=float)
    return np.exp(float(beta0)) * np.power(hh / float(h_ref_m), float(beta1))


class Discrepancy:
    """Container describing the discrepancy function actually used."""

    kind = "koh_multiplicative_power_law"

    def __init__(self, beta0: float, beta1: float, h_ref_m: float) -> None:
        self.beta0 = float(beta0)
        self.beta1 = float(beta1)
        self.h_ref_m = float(h_ref_m)

    def factor(self, h_m: Any) -> np.ndarray:
        return discrepancy_factor(self.beta0, self.beta1, h_m, self.h_ref_m)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "beta0": self.beta0,
            "beta1": self.beta1,
            "h_ref_m": self.h_ref_m,
            "amplitude_exp_beta0": math.exp(self.beta0),
        }


def forward_kN(
    campaign: MeasuredCampaign,
    theta: Sequence[float],
    *,
    with_discrepancy: bool = True,
) -> np.ndarray:
    """Total resistance (kN) implied by the parameter vector ``theta``."""
    t = np.asarray(theta, dtype=float).reshape(-1)
    rc, rb, rs = forward_components_kN(campaign, t[0], t[1], t[2], t[3])
    total = rc + rb + rs
    if with_discrepancy:
        total = total * discrepancy_factor(
            t[4], t[5], campaign.h_m, float(np.mean(campaign.h_m))
        )
    return total


# ---------------------------------------------------------------------------
# posterior, sampler, calibration
# ---------------------------------------------------------------------------


def _prior_logpdf(theta: np.ndarray, spec: Mapping[str, Mapping[str, Any]]) -> float:
    lp = 0.0
    for i, name in enumerate(PARAM_NAMES_MEASURED):
        s = spec[name]
        kind = s["kind"]
        x = float(theta[i])
        if kind == "uniform":
            continue
        if kind == "lognormal":
            if x <= 0.0:
                return -math.inf
            z = (math.log(x) - math.log(float(s["center"]))) / float(s["sigma"])
            lp -= 0.5 * z * z
        elif kind == "normal":
            z = (x - float(s["center"])) / float(s["sigma"])
            lp -= 0.5 * z * z
        else:  # pragma: no cover - defensive
            raise ValueError(f"unknown prior kind {kind!r}")
    return lp


def log_posterior_factory(
    campaign: MeasuredCampaign,
    *,
    prior: str = "literature",
    with_discrepancy: bool = True,
) -> Tuple[Callable[[np.ndarray], float], np.ndarray, List[str]]:
    """Build ``log p(theta | D)`` for the measured campaign.

    ``prior`` is ``"literature"`` (see :data:`PRIOR_LITERATURE`) or ``"flat"``
    (uniform on the box: the identifiability diagnostic).  With
    ``with_discrepancy=False`` the discrepancy is frozen to
    ``beta0 = beta1 = 0`` (negative-control configuration).
    """
    if prior not in {"literature", "flat"}:
        raise ValueError(f"unknown prior {prior!r}")
    bounds = PRIOR_BOUNDS.copy()
    names = list(PARAM_NAMES_MEASURED)
    y = campaign.r_meas_kN
    sigma2 = np.square(campaign.sigma_r_kN)
    log2pi = math.log(2.0 * math.pi)

    def log_post(theta: np.ndarray) -> float:
        t = np.asarray(theta, dtype=float).reshape(-1)
        if t.size != bounds.shape[0]:
            raise ValueError(f"expected {bounds.shape[0]} parameters, got {t.size}")
        if np.any(t < bounds[:, 0]) or np.any(t > bounds[:, 1]):
            return -math.inf
        tt = t.copy()
        if not with_discrepancy:
            tt[4] = 0.0
            tt[5] = 0.0
        model = forward_kN(campaign, tt, with_discrepancy=with_discrepancy)
        var = sigma2 + np.square(tt[6] * model)
        if np.any(var <= 0.0):
            return -math.inf
        resid = y - model
        ll = -0.5 * float(np.sum(log2pi + np.log(var) + np.square(resid) / var))
        if prior == "flat":
            return ll
        if not with_discrepancy:
            # the discrepancy parameters are inactive: their prior is constant
            t_prior = t.copy()
            t_prior[4] = 0.0
            t_prior[5] = 0.0
            return ll + _prior_logpdf(t_prior, PRIOR_LITERATURE)
        return ll + _prior_logpdf(t, PRIOR_LITERATURE)

    return log_post, bounds, names


def run_metropolis(
    log_post: Callable[[np.ndarray], float],
    bounds: np.ndarray,
    *,
    inits: Sequence[Sequence[float]],
    n_steps: int = 12000,
    burn: int = 3000,
    seed: int = 20261,
    n_adapt: int = 2000,
    scale: float = 2.38,
) -> ChainSet:
    """Adaptive Metropolis with the proposal frozen after ``n_adapt`` steps.

    Proposal covariance ``scale**2/d * Cov(samples)`` (Haario et al. 2001),
    re-estimated every ``n_adapt // 4`` steps during the adaptation window and
    **frozen** afterwards, so that adaptation cannot hide non-convergence from
    split-R-hat.  Chain starts are supplied by the caller (MAP start plus an
    over-dispersed spread) to keep split-R-hat a real test.
    """
    b = np.asarray(bounds, dtype=float)
    d = b.shape[0]
    init = np.asarray(inits, dtype=float)
    if init.ndim != 2 or init.shape[1] != d:
        raise ValueError("inits must be (n_chains, d)")
    if np.any(init < b[:, 0]) or np.any(init > b[:, 1]):
        raise ValueError("inits must lie inside the bounds")
    n_chains = init.shape[0]
    rng = np.random.default_rng(seed)
    span = b[:, 1] - b[:, 0]
    cov = np.diag(np.square(0.05 * span))
    samples = np.empty((n_chains, n_steps, d))
    logps = np.empty((n_chains, n_steps))
    chunk = max(10, n_adapt // 4)
    for c in range(n_chains):
        x = init[c].copy()
        lp = float(log_post(x))
        if not np.isfinite(lp):
            raise ValueError(f"initial point of chain {c} has non-finite log posterior")
        frozen: Optional[np.ndarray] = None
        hist: List[np.ndarray] = []
        for step in range(n_steps):
            x_new = x + rng.multivariate_normal(np.zeros(d), cov)
            # mirror once into the box; anything still outside is rejected by
            # the uniform box boundary (log_post returns -inf)
            x_new = np.where(x_new < b[:, 0], 2 * b[:, 0] - x_new, x_new)
            x_new = np.where(x_new > b[:, 1], 2 * b[:, 1] - x_new, x_new)
            lp_new = float(log_post(x_new))
            if math.log(rng.random()) < lp_new - lp:
                x, lp = x_new, lp_new
            samples[c, step, :] = x
            logps[c, step] = lp
            if frozen is None:
                hist.append(x.copy())
                if (step + 1) % chunk == 0 and len(hist) > 2 * d:
                    arr = np.asarray(hist[-n_adapt:], dtype=float)
                    est = np.atleast_2d(np.cov(arr, rowvar=False))
                    cov = (scale * scale / d) * est + 1e-12 * np.eye(d)
                if step + 1 >= n_adapt:
                    frozen = cov.copy()
    return ChainSet(
        samples=samples,
        logp=logps,
        bounds=b,
        names=[f"x{i}" for i in range(d)],
        burn=burn,
        seeds=[int(seed)],
        proposal_scale=float(scale),
    )


def _post_burn_array(chains: ChainSet) -> np.ndarray:
    post = chains.post_burn
    if callable(post):  # pragma: no cover - compatibility with older ChainSet API
        post = post()
    return np.asarray(post, dtype=float)


def posterior_summary(
    chains: ChainSet,
    names: Sequence[str],
    *,
    reference: Optional[Mapping[str, float]] = None,
    level: float = 0.95,
) -> Dict[str, Any]:
    """Posterior means, central intervals, split-R-hat, ESS and reference deltas."""
    pb = _post_burn_array(chains)
    rhat = split_rhat(pb)
    ess, ess_worst = ess_min(pb)
    flat = pb.reshape(-1, pb.shape[-1])
    alpha = 0.5 * (1.0 - level) * 100.0
    out: Dict[str, Any] = {
        "names": list(names),
        "level": level,
        "rhat_max": float(np.max(rhat)),
        "ess_min": int(ess_worst),
        "parameters": {},
    }
    for j, name in enumerate(names):
        col = flat[:, j]
        lo, hi = np.percentile(col, [alpha, 100.0 - alpha])
        rec: Dict[str, Any] = {
            "mean": float(col.mean()),
            "sd": float(col.std(ddof=1)),
            "median": float(np.median(col)),
            "ci_low": float(lo),
            "ci_high": float(hi),
            "rhat": float(rhat[j]),
            "ess": float(ess[j]),
        }
        if reference is not None and name in reference:
            ref = float(reference[name])
            rec["literature"] = ref
            rec["relative_error"] = float(abs(rec["mean"] - ref) / abs(ref))
        out["parameters"][name] = rec
    return out


def _map_start(
    log_post: Callable[[np.ndarray], float],
    bounds: np.ndarray,
    x0: Optional[Sequence[float]] = None,
) -> np.ndarray:
    from scipy.optimize import minimize

    b = np.asarray(bounds, dtype=float)
    start = (b[:, 0] + b[:, 1]) / 2.0 if x0 is None else np.asarray(x0, dtype=float)
    cache: Dict[Tuple[float, ...], float] = {}

    def obj(t: np.ndarray) -> float:
        key = tuple(np.round(t, 12))
        if key not in cache:
            val = float(log_post(t))
            cache[key] = -val if np.isfinite(val) else 1e9
        return cache[key]

    res = minimize(
        obj,
        start,
        method="Nelder-Mead",
        options={"maxiter": 60000, "xatol": 1e-5, "fatol": 1e-7},
    )
    cand = np.clip(np.asarray(res.x, dtype=float), b[:, 0] + 1e-9, b[:, 1] - 1e-9)
    if not np.isfinite(log_post(cand)):
        cand = np.clip(start, b[:, 0] + 1e-9, b[:, 1] - 1e-9)
    return cand


def calibrate(
    campaign: Optional[MeasuredCampaign] = None,
    *,
    prior: str = "literature",
    n_chains: int = 4,
    n_steps: int = 12000,
    burn: int = 3000,
    seed: int = 20261,
    with_discrepancy: bool = True,
) -> Dict[str, Any]:
    """Bayesian calibration of the twin on the measured campaign."""
    camp = campaign or load_measured_campaign()
    log_post, bounds, names = log_posterior_factory(
        camp, prior=prior, with_discrepancy=with_discrepancy
    )
    map_x = _map_start(log_post, bounds, MAP_START_DEFAULT)
    span = bounds[:, 1] - bounds[:, 0]
    frac = np.linspace(-0.12, 0.12, n_chains).reshape(-1, 1)
    inits = np.clip(
        map_x.reshape(1, -1) + frac * span, bounds[:, 0] + 1e-9, bounds[:, 1] - 1e-9
    )
    chains = run_metropolis(
        log_post, bounds, inits=inits, n_steps=n_steps, burn=burn, seed=seed
    )
    summary = posterior_summary(chains, names, reference=LITERATURE_REFERENCE)
    return {
        "prior": prior,
        "with_discrepancy": bool(with_discrepancy),
        "map": {n: float(v) for n, v in zip(names, map_x)},
        "map_log_post": float(log_post(map_x)),
        "summary": summary,
        "chains": chains,
        "campaign_fingerprint": camp.fingerprint(),
    }


def prior_sensitivity(
    campaign: Optional[MeasuredCampaign] = None,
    *,
    n_steps: int = 8000,
    burn: int = 2500,
    seed: int = 20262,
) -> Dict[str, Any]:
    """Flat-prior calibration: how much of the posterior location is data-driven?

    The literature anchors shrink the coefficients; with ten points and a free
    discrepancy amplitude the data alone are nearly degenerate (strength scale
    and discrepancy amplitude trade off).  This run exposes that trade-off
    instead of hiding it behind the anchor.
    """
    camp = campaign or load_measured_campaign()
    flat = calibrate(
        camp, prior="flat", n_chains=4, n_steps=n_steps, burn=burn, seed=seed
    )
    names = list(PARAM_NAMES_MEASURED)
    summary = flat["summary"]
    at_bounds = {
        n: bool(
            summary["parameters"][n]["ci_low"] <= PRIOR_BOUNDS[i, 0] + 1e-6
            or summary["parameters"][n]["ci_high"] >= PRIOR_BOUNDS[i, 1] - 1e-6
        )
        for i, n in enumerate(names)
    }
    return {
        "prior": "flat",
        "summary": summary,
        "map": flat["map"],
        "at_bounds": at_bounds,
        "campaign_fingerprint": camp.fingerprint(),
    }


# ---------------------------------------------------------------------------
# leave-one-out predictive coverage
# ---------------------------------------------------------------------------


def _predictive_draws_at(
    campaign_full: MeasuredCampaign,
    fold: int,
    theta_draws: np.ndarray,
    *,
    with_discrepancy: bool = True,
    n_draw: int = 3000,
    seed: int = 20271,
) -> np.ndarray:
    """Posterior predictive draws for the held-out point ``fold``.

    Integrates parameter uncertainty, the discrepancy amplitude and the
    residual-inadequacy coefficient (the observation-noise term).
    """
    rng = np.random.default_rng(seed + fold)
    idx = rng.integers(0, theta_draws.shape[0], size=n_draw)
    held = campaign_full.subset([fold])
    means = np.array(
        [
            forward_kN(held, theta_draws[i], with_discrepancy=with_discrepancy)[0]
            for i in idx
        ]
    )
    kappa = theta_draws[idx, 6]
    return means + kappa * means * rng.standard_normal(n_draw)


def loo_predictive(
    campaign: Optional[MeasuredCampaign] = None,
    *,
    n_steps: int = 5000,
    burn: int = 1500,
    seed: int = 20271,
    level: float = 0.95,
    with_discrepancy: bool = True,
) -> Dict[str, Any]:
    """Leave-one-out refit predictive intervals on the measured points.

    Each fold re-runs the whole calibration on the remaining points (exact
    refit, no importance-sampling approximation) and then evaluates the
    predictive distribution of the held-out point.
    """
    camp = campaign or load_measured_campaign()
    n = len(camp)
    rows: List[Dict[str, Any]] = []
    for i in range(n):
        rest = camp.subset([j for j in range(n) if j != i])
        fit = calibrate(
            rest,
            n_chains=3,
            n_steps=n_steps,
            burn=burn,
            seed=seed + 101 * i,
            with_discrepancy=with_discrepancy,
        )
        draws = _post_burn_array(fit["chains"]).reshape(-1, len(PARAM_NAMES_MEASURED))
        pred = _predictive_draws_at(
            camp, i, draws, with_discrepancy=with_discrepancy, seed=seed + 7 * i
        )
        lo, hi = predictive_interval(pred.reshape(-1, 1), level)
        y = float(camp.r_meas_kN[i])
        rows.append(
            {
                "point_id": camp.point_ids[i],
                "h_m": float(camp.h_m[i]),
                "V_ms": float(camp.v_ms[i]),
                "observed_kN": y,
                "pred_mean_kN": float(np.mean(pred)),
                "pred_sd_kN": float(np.std(pred, ddof=1)),
                "pi_low_kN": float(lo[0]),
                "pi_high_kN": float(hi[0]),
                "pi_width_kN": float(hi[0] - lo[0]),
                "inside": bool(lo[0] <= y <= hi[0]),
                "z_score": float((y - np.mean(pred)) / max(np.std(pred, ddof=1), 1e-9)),
                "fold_kappa": float(np.mean(draws[:, 6])),
                "fold_exp_beta0": float(math.exp(np.mean(draws[:, 4]))),
            }
        )
    inside_n = int(sum(r["inside"] for r in rows))
    return {
        "level": level,
        "with_discrepancy": bool(with_discrepancy),
        "n_points": n,
        "inside": inside_n,
        "coverage": inside_n / n,
        "mean_pi_width_kN": float(np.mean([r["pi_width_kN"] for r in rows])),
        "mean_abs_z": float(np.mean([abs(r["z_score"]) for r in rows])),
        "rows": rows,
    }


def negative_control(
    campaign: Optional[MeasuredCampaign] = None, *, level: float = 0.95
) -> Dict[str, Any]:
    """Falsification: literature coefficients, **no** discrepancy, must under-cover.

    The twin is frozen at its literature calibration (coefficients ``theta = 1``,
    ``beta0 = beta1 = 0``) and the interval comes only from the reported
    measurement band.  If this control covered at the nominal rate, the coverage
    criterion would be vacuous.
    """
    camp = campaign or load_measured_campaign()
    theta = np.array([1.0, 1.0, 0.15, 1.0, 0.0, 0.0, 0.0])
    pred = forward_kN(camp, theta, with_discrepancy=False)
    lo = pred - 1.959963984540054 * camp.sigma_r_kN
    hi = pred + 1.959963984540054 * camp.sigma_r_kN
    inside, n, rate = coverage_rate(lo, hi, camp.r_meas_kN)
    return {
        "level": level,
        "inside": inside,
        "n_points": n,
        "coverage": rate,
        "median_ratio_model_over_measured": float(np.median(pred / camp.r_meas_kN)),
        "rows": [
            {
                "point_id": camp.point_ids[i],
                "observed_kN": float(camp.r_meas_kN[i]),
                "predicted_kN": float(pred[i]),
                "pi_low_kN": float(lo[i]),
                "pi_high_kN": float(hi[i]),
                "inside": bool(lo[i] <= camp.r_meas_kN[i] <= hi[i]),
            }
            for i in range(n)
        ],
    }


# ---------------------------------------------------------------------------
# global sensitivity: exact quadrature ANOVA vs pick-freeze Monte Carlo
# ---------------------------------------------------------------------------


def _outer_weight(weights: Sequence[np.ndarray]) -> np.ndarray:
    out = np.array(1.0)
    for wi in weights:
        out = np.multiply.outer(out, np.asarray(wi, dtype=float))
    return out


def _gauss_legendre_unit(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre nodes/weights normalised to the unit interval (sum = 1)."""
    nodes, weights = np.polynomial.legendre.leggauss(n)
    return 0.5 * (nodes + 1.0), 0.5 * weights


def sensitivity_point(
    campaign: MeasuredCampaign,
    *,
    h_m: Optional[float] = None,
    v_ms: Optional[float] = None,
) -> Callable[[np.ndarray], np.ndarray]:
    """Twin response at a frozen operating point, as a function of the coefficients.

    ``x`` has shape ``(N, 4)`` and holds ``(sigma_c_scale, sigma_f_scale, mu,
    C_sub_scale)``; ice strengths are held at their campaign means and the
    operating condition is frozen - the well-posed question "how much of the
    response variance at a design point comes from which coefficient".  Defaults:
    median measured thickness and the top measured speed (the condition at which
    the submergence term is least negligible).
    """
    kb, alpha, phi = _geometry_factors(campaign.ship)
    ship = campaign.ship
    sigma_c_ref = float(np.mean(campaign.sigma_c_pa))
    sigma_f_ref = float(np.mean(campaign.sigma_f_pa))
    h = float(np.median(campaign.h_m) if h_m is None else h_m)
    v = float(np.max(campaign.v_ms) if v_ms is None else v_ms)
    fr = v / math.sqrt(G * h)

    def evaluate(x: np.ndarray) -> np.ndarray:
        arr = np.atleast_2d(np.asarray(x, dtype=float))
        c_crush = 0.5 * (np.tan(alpha) + arr[:, 2] / math.cos(alpha))
        c_break = 0.5 * (np.sin(0.5 * phi) + arr[:, 2] * np.cos(0.5 * phi)) / kb
        r_c = c_crush * arr[:, 0] * sigma_c_ref * h * h
        r_b = c_break * arr[:, 1] * sigma_f_ref * h * ship.B
        r_s = arr[:, 3] * (0.15 + 0.3 * fr) * RHO_ICE * v * v * h * ship.B
        return (r_c + r_b + r_s) / 1e3

    return evaluate


def sobol_exact(
    func: Callable[[np.ndarray], np.ndarray],
    bounds: np.ndarray,
    *,
    order: int = 16,
) -> Dict[str, Any]:
    """Exact (tensor-quadrature) Sobol' indices ``S_i`` and ``S_Ti``.

    Uniform inputs on ``bounds``.  The ANOVA decomposition
    ``f = f0 + sum_i f_i + sum_{i<j} f_ij + ...`` is integrated with nested
    Gauss-Legendre quadrature, so the indices are exact to quadrature accuracy
    and give the reference for the Monte-Carlo estimator.  Closure identities:
    ``sum_i S_i + interaction share = 1`` and
    ``sum_i S_Ti = 1 + interaction share``.
    """
    b = np.asarray(bounds, dtype=float)
    d = b.shape[0]
    u, w = _gauss_legendre_unit(order)
    grids = [b[i, 0] + u * (b[i, 1] - b[i, 0]) for i in range(d)]
    mesh = np.meshgrid(*grids, indexing="ij")
    pts = np.stack([m.reshape(-1) for m in mesh], axis=1)
    vals = np.asarray(func(pts), dtype=float).reshape([order] * d)
    prob = _outer_weight([w] * d)
    f0 = float(np.sum(vals * prob))
    var = float(np.sum(np.square(vals - f0) * prob))
    s_i: List[float] = []
    s_ti: List[float] = []
    for i in range(d):
        axes = tuple(j for j in range(d) if j != i)
        wo = _outer_weight([w] * len(axes))
        cond = np.tensordot(vals, wo, axes=(axes, tuple(range(len(axes)))))
        v_i = float(np.sum(np.square(cond - f0) * w))
        keep = np.tensordot(vals, w, axes=(i, 0))
        v_compl = float(np.sum(np.square(keep - f0) * wo))
        s_i.append(v_i / max(var, 1e-300))
        s_ti.append(1.0 - v_compl / max(var, 1e-300))
    return {
        "mean": f0,
        "variance": var,
        "sd": math.sqrt(max(var, 0.0)),
        "S_i": s_i,
        "S_Ti": s_ti,
        "sum_S_i": float(sum(s_i)),
        "sum_S_Ti": float(sum(s_ti)),
        "interaction_share": float(1.0 - sum(s_i)),
        "order": int(order),
    }


def sobol_mc(
    func: Callable[[np.ndarray], np.ndarray],
    bounds: np.ndarray,
    *,
    n_base: int = 1 << 14,
    seed: int = 20281,
) -> Dict[str, Any]:
    """Saltelli/Jansen pick-freeze Monte-Carlo Sobol' indices (estimator check)."""
    b = np.asarray(bounds, dtype=float)
    d = b.shape[0]
    rng = np.random.default_rng(seed)
    a = b[:, 0] + rng.random((n_base, d)) * (b[:, 1] - b[:, 0])
    bm = b[:, 0] + rng.random((n_base, d)) * (b[:, 1] - b[:, 0])
    fa = np.asarray(func(a), dtype=float).reshape(-1)
    fb = np.asarray(func(bm), dtype=float).reshape(-1)
    var = float(np.var(np.concatenate([fa, fb]), ddof=1))
    s_i: List[float] = []
    s_ti: List[float] = []
    for i in range(d):
        ab = a.copy()
        ab[:, i] = bm[:, i]
        fab = np.asarray(func(ab), dtype=float).reshape(-1)
        s_i.append(float(np.mean(fb * (fab - fa))) / var)
        s_ti.append(float(np.mean(np.square(fa - fab))) / (2.0 * var))
    return {
        "S_i": s_i,
        "S_Ti": s_ti,
        "sum_S_i": float(sum(s_i)),
        "sum_S_Ti": float(sum(s_ti)),
        "interaction_share": float(1.0 - sum(s_i)),
        "variance": var,
        "n_base": int(n_base),
        "model_evaluations": int(2 * n_base + n_base * d),
    }


# ---------------------------------------------------------------------------
# model-form averaging over the published literature laws
# ---------------------------------------------------------------------------


LAW_BOUNDS = np.array([[-2.0, 2.0], [0.01, 0.60]], dtype=float)


def _law_log_posterior(
    y: np.ndarray, base: np.ndarray, sigma: np.ndarray
) -> Callable[[np.ndarray], float]:
    """Two-parameter posterior (log-bias, inadequacy) for one literature law."""

    def log_post(t: np.ndarray) -> float:
        beta, kappa = float(t[0]), float(t[1])
        if not (-2.0 <= beta <= 2.0 and 0.01 <= kappa <= 0.60):
            return -math.inf
        m = math.exp(beta) * base
        var = np.square(sigma) + np.square(kappa * m)
        resid = y - m
        ll = -0.5 * float(np.sum(math.log(2.0 * math.pi) + np.log(var) + np.square(resid) / var))
        ll -= 0.5 * (beta / 0.70) ** 2
        return ll

    return log_post


def _fit_law(
    y: np.ndarray,
    base: np.ndarray,
    sigma: np.ndarray,
    *,
    n_steps: int,
    burn: int,
    seed: int,
    n_chains: int = 3,
) -> np.ndarray:
    """Posterior draws of ``(beta, kappa)`` for one law on a data subset."""
    log_post = _law_log_posterior(y, base, sigma)
    map_x = _map_start(log_post, LAW_BOUNDS, np.array([0.0, 0.15]))
    span = LAW_BOUNDS[:, 1] - LAW_BOUNDS[:, 0]
    frac = np.linspace(-0.15, 0.15, n_chains).reshape(-1, 1)
    inits = np.clip(
        map_x.reshape(1, -1) + frac * span, LAW_BOUNDS[:, 0] + 1e-9, LAW_BOUNDS[:, 1] - 1e-9
    )
    chains = run_metropolis(
        log_post, LAW_BOUNDS, inits=inits, n_steps=n_steps, burn=burn, seed=seed
    )
    return _post_burn_array(chains).reshape(-1, 2)


def _law_predictive(
    base_kN: float, sigma_kN: float, draws: np.ndarray, *, n_draw: int, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, draws.shape[0], size=n_draw)
    m = np.exp(draws[idx, 0]) * base_kN
    sd = np.sqrt(np.square(sigma_kN) + np.square(draws[idx, 1] * m))
    return m + sd * rng.standard_normal(n_draw)


def _logmeanexp(v: np.ndarray) -> float:
    m = float(np.max(v))
    if not np.isfinite(m):
        return -math.inf
    return m + math.log(float(np.mean(np.exp(v - m))))


def bma_published_laws(
    campaign: Optional[MeasuredCampaign] = None,
    *,
    level: float = 0.95,
    n_steps: int = 4000,
    burn: int = 1200,
    n_steps_loo: int = 2000,
    burn_loo: int = 700,
    seed: int = 20291,
    n_draw: int = 3000,
    reference_width_kN: Optional[float] = None,
) -> Dict[str, Any]:
    """Pseudo-BMA over the four published literature level-ice laws (REAL data).

    Each law keeps its own log-bias and inadequacy coefficient, so the members
    are calibrated *before* averaging; the weights are stacked on the exact
    leave-one-out log predictive score (Yao et al. 2018 pseudo-BMA).
    """
    camp = campaign or load_measured_campaign()
    keys, mat = camp.published_matrix()
    y = camp.r_meas_kN
    sigma = camp.sigma_r_kN
    n = len(camp)
    members: Dict[str, Dict[str, Any]] = {}
    lpd: Dict[str, float] = {}
    widths: Dict[str, float] = {}
    draws_by_law: Dict[str, np.ndarray] = {}
    for j, law in enumerate(keys):
        base = mat[:, j]
        if not np.all(np.isfinite(base)):
            raise ValueError(f"law {law!r} has missing published predictions")
        draws = _fit_law(y, base, sigma, n_steps=n_steps, burn=burn, seed=seed + 31 * j)
        loo_score = 0.0
        for i in range(n):
            rest = [k for k in range(n) if k != i]
            sub = _fit_law(
                y[rest],
                base[rest],
                sigma[rest],
                n_steps=n_steps_loo,
                burn=burn_loo,
                seed=seed + 1000 + 37 * j + i,
                n_chains=2,
            )
            pred = _law_predictive(
                float(base[i]),
                float(sigma[i]),
                sub,
                n_draw=n_draw,
                seed=seed + 7000 + 41 * j + i,
            )
            sd = np.sqrt(np.square(float(sigma[i])) + np.square(sub[:, 1] * np.exp(sub[:, 0]) * base[i]))
            from scipy.stats import norm as _norm

            logp = _norm.logpdf(y[i], loc=np.exp(sub[:, 0]) * base[i], scale=sd)
            loo_score += _logmeanexp(logp)
        pred_full = _law_predictive(
            base, sigma, draws, n_draw=n_draw, seed=seed + 555 + j
        ) if False else np.stack(
            [
                _law_predictive(
                    float(base[i]), float(sigma[i]), draws,
                    n_draw=n_draw, seed=seed + 555 + 13 * j + i,
                )
                for i in range(n)
            ],
            axis=1,
        )
        lo, hi = predictive_interval(pred_full, level)
        lpd[law] = float(loo_score)
        widths[law] = float(np.mean(hi - lo))
        draws_by_law[law] = pred_full
        members[law] = {
            "loo_log_score": float(loo_score),
            "mean_pi_width_kN": widths[law],
            "beta_mean": float(np.mean(draws[:, 0])),
            "kappa_mean": float(np.mean(draws[:, 1])),
            "exp_beta_mean": float(np.exp(np.mean(draws[:, 0]))),
            "median_ratio_published_over_measured": float(np.median(base / y)),
            "bias_corrected_mean_kN": float(np.mean(np.exp(draws[:, 0].mean()) * base)),
        }
    weights = pseudo_bma_weights(lpd)
    wsum = float(sum(weights.values()))
    rng = np.random.default_rng(seed + 9999)
    mixture_cols: List[np.ndarray] = []
    member_keys = sorted(draws_by_law)
    for i in range(n):
        pick = rng.choice(len(member_keys), size=n_draw, p=[weights[k] for k in member_keys])
        col = np.empty(n_draw)
        for m_i, key in enumerate(member_keys):
            mask = pick == m_i
            if not np.any(mask):
                continue
            src = draws_by_law[key][:, i]
            col[mask] = src[rng.integers(0, src.size, size=int(np.sum(mask)))]
        mixture_cols.append(col)
    mixture = np.stack(mixture_cols, axis=1)
    lo, hi = predictive_interval(mixture, level)
    bma_width = float(np.mean(hi - lo))
    inside, npts, rate = coverage_rate(lo, hi, y)
    best_law = min(members, key=lambda k: -members[k]["loo_log_score"])
    out: Dict[str, Any] = {
        "level": level,
        "laws": list(keys),
        "members": members,
        "weights": weights,
        "weights_sum": wsum,
        "bma_mean_pi_width_kN": bma_width,
        "bma_coverage": rate,
        "bma_inside": inside,
        "n_points": npts,
        "best_single_law": best_law,
        "best_single_law_width_kN": widths[best_law],
        "narrowest_single_law": min(widths, key=lambda k: widths[k]),
        "narrowest_single_law_width_kN": min(widths.values()),
        "bma_rows": [
            {
                "point_id": camp.point_ids[i],
                "observed_kN": float(y[i]),
                "pi_low_kN": float(lo[i]),
                "pi_high_kN": float(hi[i]),
                "inside": bool(lo[i] <= y[i] <= hi[i]),
            }
            for i in range(n)
        ],
    }
    if reference_width_kN is not None and reference_width_kN > 0:
        out["reference_single_model_width_kN"] = float(reference_width_kN)
        out["width_ratio_vs_reference"] = bma_width / float(reference_width_kN)
        out["narrowing_vs_reference"] = 1.0 - bma_width / float(reference_width_kN)
    out["width_ratio_vs_best_single_law"] = bma_width / widths[best_law]
    out["narrowing_vs_best_single_law"] = 1.0 - bma_width / widths[best_law]
    return out


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------


def _criterion(
    cid: str, statement: str, value: Any, threshold: str, status: str, evidence: str
) -> Dict[str, Any]:
    return {
        "id": cid,
        "statement": statement,
        "value": value,
        "threshold": threshold,
        "status": status,
        "evidence": evidence,
    }


def run_measured_uq(
    *,
    level: float = 0.95,
    calib_steps: int = 12000,
    calib_burn: int = 3000,
    loo_steps: int = 5000,
    loo_burn: int = 1500,
    sob_order: int = 16,
    sob_n_base: int = 1 << 14,
    seed: int = 20261,
    campaign: Optional[MeasuredCampaign] = None,
) -> Dict[str, Any]:
    """Run every face of the measured-data UQ study and evaluate the criteria."""
    camp = campaign or load_measured_campaign()

    # --- baseline diagnostics (no calibration) -----------------------------
    theta_lit = np.array([1.0, 1.0, 0.15, 1.0, 0.0, 0.0, 0.0])
    rc, rb, rs = forward_components_kN(camp, 1.0, 1.0, 0.15, 1.0)
    total_model = rc + rb + rs
    baseline = {
        "predicted_kN": total_model.tolist(),
        "component_share": {
            "crush": (rc / total_model).tolist(),
            "break": (rb / total_model).tolist(),
            "submersion": (rs / total_model).tolist(),
        },
        "ratio_model_over_measured": (total_model / camp.r_meas_kN).tolist(),
        "median_ratio_model_over_measured": float(np.median(total_model / camp.r_meas_kN)),
        "median_break_share": float(np.median(rb / total_model)),
        "observed_kN": camp.r_meas_kN.tolist(),
        "published_ratio": load_published_laws(camp),
    }

    # --- (a)(b) calibration ------------------------------------------------
    calib = calibrate(
        camp,
        prior="literature",
        n_steps=calib_steps,
        burn=calib_burn,
        seed=seed,
    )
    summary = calib["summary"]
    physical = ["sigma_c_scale", "sigma_f_scale", "mu_hull_ice", "C_sub_scale"]
    rel_err = {k: summary["parameters"][k]["relative_error"] for k in physical}
    max_rel = float(max(rel_err.values()))
    crit_a = _criterion(
        "a",
        "posterior mean vs literature value, per physical coefficient",
        {"relative_error": rel_err, "max_relative_error": max_rel},
        "< 0.20",
        "PASS" if max_rel < 0.20 else "FAIL",
        "calibration.summary.parameters",
    )
    rhat_max = float(summary["rhat_max"])
    crit_b = _criterion(
        "b",
        "Gelman-Rubin split R-hat over 4 chains",
        {"rhat_max": rhat_max, "ess_min": summary["ess_min"],
         "rhat_per_parameter": {k: v["rhat"] for k, v in summary["parameters"].items()}},
        "< 1.1",
        "PASS" if rhat_max < 1.1 else "FAIL",
        "calibration.summary.rhat_max",
    )

    # --- identifiability ---------------------------------------------------
    ident = prior_sensitivity(camp, n_steps=max(4000, calib_steps // 2), seed=seed + 1)

    # --- (c) global sensitivity -------------------------------------------
    bounds4 = PRIOR_BOUNDS[:4].copy()
    point = sensitivity_point(camp)
    exact = sobol_exact(point, bounds4, order=sob_order)
    mc = sobol_mc(point, bounds4, n_base=sob_n_base, seed=seed + 2)
    dev_total = abs(exact["sum_S_Ti"] - 1.0)
    dev_mc = abs(mc["sum_S_Ti"] - exact["sum_S_Ti"])
    max_index_diff = float(max(
        max(abs(a - b) for a, b in zip(exact["S_i"], mc["S_i"])),
        max(abs(a - b) for a, b in zip(exact["S_Ti"], mc["S_Ti"])),
    ))
    crit_c = _criterion(
        "c",
        "variance decomposition closure: sum of total-order Sobol indices = 1",
        {
            "sum_S_Ti_exact": exact["sum_S_Ti"],
            "sum_S_i_exact": exact["sum_S_i"],
            "interaction_share_exact": exact["interaction_share"],
            "sum_S_Ti_mc": mc["sum_S_Ti"],
            "max_abs_index_difference_mc_vs_exact": max_index_diff,
            "S_i_exact": exact["S_i"],
            "S_Ti_exact": exact["S_Ti"],
        },
        "|sum_S_Ti - 1| < 0.05 and MC matches exact",
        "PASS" if (dev_total < 0.05 and dev_mc < 0.05 and max_index_diff < 0.05) else "FAIL",
        "sobol.exact / sobol.mc",
    )

    # --- (d) predictive-interval coverage ---------------------------------
    loo = loo_predictive(
        camp, n_steps=loo_steps, burn=loo_burn, level=level, seed=seed + 3
    )
    ncg = negative_control(camp, level=level)
    crit_d = _criterion(
        "d",
        f"{int(level*100)}% leave-one-out predictive interval coverage on measured points",
        {
            "coverage": loo["coverage"],
            "inside": loo["inside"],
            "n_points": loo["n_points"],
            "mean_pi_width_kN": loo["mean_pi_width_kN"],
            "mean_abs_z": loo["mean_abs_z"],
            "negative_control_coverage": ncg["coverage"],
        },
        "0.90 <= coverage <= 1.00 (negative control must fail)",
        "PASS" if (0.90 <= loo["coverage"] <= 1.0 and ncg["coverage"] < 0.90) else "FAIL",
        "coverage.loo / coverage.negative_control",
    )

    # --- (e) model-form averaging -----------------------------------------
    bma = bma_published_laws(
        camp,
        level=level,
        seed=seed + 4,
        reference_width_kN=loo["mean_pi_width_kN"],
    )
    width_ok = bma.get("width_ratio_vs_reference", 1.0) <= 0.80
    weight_ok = abs(bma["weights_sum"] - 1.0) < 1e-6
    crit_e = _criterion(
        "e",
        "BMA weights sum to one and the BMA interval is >20% narrower than the single-model interval",
        {
            "weights": bma["weights"],
            "weights_sum": bma["weights_sum"],
            "bma_mean_pi_width_kN": bma["bma_mean_pi_width_kN"],
            "single_model_width_kN": bma.get("reference_single_model_width_kN"),
            "width_ratio_vs_single_model": bma.get("width_ratio_vs_reference"),
            "width_ratio_vs_best_single_law": bma.get("width_ratio_vs_best_single_law"),
        },
        "sum(w)=1 and width ratio <= 0.80",
        "PASS" if (weight_ok and width_ok) else "FAIL",
        "bma.weights / bma.bma_mean_pi_width_kN",
    )

    criteria = {c["id"]: c for c in (crit_a, crit_b, crit_c, crit_d, crit_e)}
    n_pass = sum(1 for c in criteria.values() if c["status"] == "PASS")

    report: Dict[str, Any] = {
        "node": "p223w2b-n03-physics-uq-digital-twin",
        "module": "tensorphysics.ice_digital_twin.measured_uq",
        "series": camp.fingerprint(),
        "level": level,
        "baseline": baseline,
        "calibration": {
            "prior": "literature",
            "map": calib["map"],
            "summary": summary,
        },
        "identifiability": ident,
        "sobol": {
            "reference_condition": {
                "h_m": float(np.median(camp.h_m)),
                "V_ms": float(np.max(camp.v_ms)),
                "sigma_c_mean_MPa": float(np.mean(camp.sigma_c_pa) / 1e6),
                "sigma_f_mean_MPa": float(np.mean(camp.sigma_f_pa) / 1e6),
                "inputs": list(PARAM_NAMES_MEASURED[:4]),
                "bounds": bounds4.tolist(),
            },
            "exact": exact,
            "monte_carlo": mc,
        },
        "coverage": {"loo": loo, "negative_control": ncg},
        "bma": bma,
        "criteria": criteria,
        "criteria_summary": {"n_pass": n_pass, "n_total": len(criteria),
                             "all_pass": n_pass == len(criteria)},
        "provenance": {
            "series_sha256": camp.sha256,
            "series_path": camp.source_path,
            "measured_dataset_owner": "tensorphysics.ice.measured_anchor",
            "doi": camp.doi,
            "unc_imputation_fraction": UNC_IMPUTATION_FRACTION,
            "unc_imputed_points": [p.point_id for p in camp.points if p.sigma_r_imputed],
        },
    }
    report["digest"] = digest({k: v for k, v in report.items() if k != "digest"})
    return report


def write_report(report: Mapping[str, Any], outdir: str) -> str:
    """Write the study report JSON (deterministic key order) and return its path."""
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "report.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1, sort_keys=True, default=float)
    return path
