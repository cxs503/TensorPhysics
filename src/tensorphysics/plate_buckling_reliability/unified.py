"""Unified deterministic + probabilistic buckling/reliability assessment.

``UnifiedBucklingReliability`` is the single interface promised by W4-N03: it
takes one panel specification (geometry, material, loads), and returns

* ``sigma_ultimate`` -- the deterministic limit stress from the arc-length
  nonlinear finite-element trace (W3-N03 capability),
* ``beta`` -- the Hasofer-Lind reliability index of the same panel with the
  deterministic capacity embedded in the limit state (W3-N04 capability),
* ``P_f`` -- the corresponding failure probability, together with the SORM
  curvature correction, the Ditlevsen system bounds and an optional Monte
  Carlo check,
* the inverse design thickness for a target index (``beta = 4.0``, ISO 19906).

The capacity used inside the limit state is a response surface *fitted to* the
deterministic solver (:mod:`.surrogate`); the fit residual against the finite
element limit points is always reported with the result, so the chain never
presents a surrogate value as if it were a direct FE result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from .deterministic import DeterministicAssessment
from .limit_states import DEFAULT_MODES, MODE_ULTIMATE, PanelLimitStates
from .panel import PanelSpec
from .reliability import ProbabilisticAssessment, VariableModel
from .surrogate import (
    UltimateStrengthSurrogate,
    faulkner_reference,
    load_default_surrogate,
)

__all__ = ["UnifiedBucklingReliability"]

#: Slenderness values of the FE calibration of the capacity surrogate (the
#: six-sigma table of the W3-N03 campaign; mesh 8x4, imperfection "average").
DEFAULT_CALIBRATION_BETAS = (1.0, 1.25, 1.5, 2.0, 2.5, 3.0)


@dataclass
class UnifiedBucklingReliability:
    """One panel, one call: deterministic limit load + probabilistic assessment."""

    panel: PanelSpec
    variables: Optional[VariableModel] = None
    surrogate: Optional[UltimateStrengthSurrogate] = None
    ds0: float = 6.0e-4
    max_steps: int = 18
    max_iter: int = 25
    deflection_limit_ratio: float = 0.01
    calibration_log: List[Dict[str, object]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.geometry_check = self.panel.validate()
        if self.variables is None:
            self.variables = VariableModel(fy_mean=self.panel.sigma_y / 1e6)
        if self.surrogate is None:
            self.surrogate = load_default_surrogate()
        self.deterministic = DeterministicAssessment(
            self.panel, ds0=self.ds0, max_steps=self.max_steps, max_iter=self.max_iter)

    # ------------------------------------------------------------- calibration
    def calibrate(self, betas: Sequence[float] = DEFAULT_CALIBRATION_BETAS,
                  save_path=None) -> UltimateStrengthSurrogate:
        """Fit the capacity surrogate to fresh arc-length limit-point traces.

        Each calibration panel keeps the specification's aspect ratio, mesh,
        boundary conditions and imperfection level and only changes the
        thickness so that ``beta`` hits the requested value at the nominal
        yield stress.
        """
        points = []
        b = self.panel.b
        coef = b * np.sqrt(self.panel.sigma_y / self.panel.E)
        for beta in betas:
            t = coef / float(beta)
            trial = PanelSpec(**{**self.panel.__dict__, "t": float(t)})
            trial.validate()
            record = DeterministicAssessment(
                trial, ds0=self.ds0, max_steps=self.max_steps,
                max_iter=self.max_iter).ultimate()
            record["beta_target"] = float(beta)
            points.append({k: v for k, v in record.items() if k != "path"})
            self.calibration_log.append(record)
        self.surrogate = UltimateStrengthSurrogate.fit(
            points, aspect=self.panel.a / self.panel.b,
            imperfection_level=self.panel.imperfection_level)
        if save_path is not None:
            self.surrogate.save(save_path)
        return self.surrogate

    # ------------------------------------------------------------------ pieces
    def limit_states(self) -> PanelLimitStates:
        if self.surrogate is None:
            raise RuntimeError(
                "no capacity surrogate available: call calibrate() or pass "
                "surrogate=/calibration= to the constructor")
        return PanelLimitStates(self.panel, self.surrogate,
                                deflection_limit_ratio=self.deflection_limit_ratio)

    def probabilistic(self, samples: int = 0, system_samples: int = 0,
                      modes: Sequence[str] = DEFAULT_MODES,
                      sensitivity: bool = True) -> Dict[str, object]:
        assessment = ProbabilisticAssessment(self.limit_states(), self.variables)
        report = assessment.report(modes=modes, samples=samples,
                                   sensitivity=sensitivity,
                                   system_samples=system_samples)
        report["surrogate"] = {
            "model": "sigma_u/sigma_y = c0 + c1/beta + c2/beta^2",
            "coefficients": list(self.surrogate.coefficients()),
            "beta_window": [self.surrogate.beta_min, self.surrogate.beta_max],
            "accuracy": self.surrogate.accuracy(),
            "source": self.surrogate.source,
        }
        return report

    # ------------------------------------------------------------- unified call
    def assess(self, run_fem: bool = False, samples: int = 0,
               system_samples: int = 0, target_beta: Optional[float] = 4.0,
               sensitivity: bool = True) -> Dict[str, object]:
        """Complete deterministic -> probabilistic chain of the panel.

        Parameters
        ----------
        run_fem:
            Also run the full nonlinear finite-element trace of *this* panel
            (tens of seconds).  When ``False`` the deterministic block reports
            the surrogate capacity at the net mean thickness and flags it as
            surrogate-derived.
        samples, system_samples:
            Monte Carlo sample counts (0 disables the simulation cross-check).
        target_beta:
            Target index for the inverse thickness design (``None`` to skip).
        """
        prob = self.probabilistic(samples=samples, system_samples=system_samples,
                                  sensitivity=sensitivity)
        surrogate_capacity = float(self.surrogate.sigma_u(
            self.panel.t - self.variables.dt_mean * 1e-3,
            self.panel.sigma_y, self.panel.b, self.panel.E))

        if run_fem:
            det = self.deterministic.report(with_sweep=False)
            sigma_ultimate = float(det["ultimate"]["sigma_u_over_sigma_y"]) * self.panel.sigma_y
            det["capacity_source"] = "arc-length nonlinear FE limit point"
        else:
            det = {
                "linear_buckling": self.deterministic.linear_buckling(),
                "geometry_check": self.geometry_check,
                "panel": self.panel.as_dict(),
                "capacity_source": "FE-calibrated response surface (no FE run "
                                   "in this call)",
            }
            sigma_ultimate = surrogate_capacity
        det["sigma_ultimate_MPa"] = float(sigma_ultimate / 1e6)
        det["sigma_ultimate_over_sigma_y"] = float(sigma_ultimate / self.panel.sigma_y)

        form_ult = prob["form"][MODE_ULTIMATE]
        out: Dict[str, object] = {
            "panel": self.panel.as_dict(),
            "geometry_check": self.geometry_check,
            "deterministic": det,
            "probabilistic": prob,
            "sigma_ultimate_MPa": float(sigma_ultimate / 1e6),
            "beta": float(form_ult["beta"]),
            "pf": float(form_ult["pf"]),
            "beta_sorm": float(prob["sorm"][MODE_ULTIMATE]["beta_sorm"]),
            "pf_sorm": float(prob["sorm"][MODE_ULTIMATE]["pf_sorm"]),
            "beta_system": float(prob["system_ditlevsen"]["beta_point"]),
            "pf_system": float(prob["system_ditlevsen"]["pf_point"]),
            "design_point": {
                "u": list(form_ult["design_point_u"]),
                "x": list(form_ult["design_point_x"]),
                "variables": list(self.variables.variable_set().names),
            },
        }
        if target_beta is not None:
            try:
                out["target_thickness"] = ProbabilisticAssessment(
                    self.limit_states(), self.variables).required_thickness(target_beta)
            except Exception as exc:                      # pragma: no cover - diagnostic
                out["target_thickness"] = {"error": str(exc)}
        return out

    # ------------------------------------------------------------- design help
    def required_thickness(self, target_beta: float = 4.0,
                           mode: str = MODE_ULTIMATE) -> Dict[str, object]:
        return ProbabilisticAssessment(self.limit_states(),
                                       self.variables).required_thickness(target_beta, mode)

    def beta_vs_thickness(self, thicknesses: Sequence[float],
                          mode: str = MODE_ULTIMATE) -> List[Dict[str, float]]:
        rows = []
        for t in thicknesses:
            states = self.limit_states().with_thickness(float(t))
            trial = ProbabilisticAssessment(states, self.variables)
            try:
                form = trial.form(mode)
                beta = float(form.beta)
                pf = float(form.pf)
            except Exception:                             # pragma: no cover
                beta, pf = float("nan"), float("nan")
            rows.append({"t_mm": float(t * 1e3), "beta": beta, "pf": pf})
        return rows

    def faulkner_ratio(self) -> float:
        """Independent analytic ultimate-strength ratio at the panel slenderness."""
        return float(faulkner_reference(self.panel.beta))

    # ------------------------------------------------------------------- json
    def to_dict(self, run_fem: bool = False, samples: int = 0) -> Dict[str, object]:
        return self.assess(run_fem=run_fem, samples=samples)

    @classmethod
    def from_payload(cls, payload: Dict[str, object]) -> "UnifiedBucklingReliability":
        """Build from an API payload (panel / variables / calibration blocks)."""
        panel_data = payload.get("panel") or payload.get("geometry") or {}
        panel = PanelSpec.from_dict(dict(panel_data))
        var_data = dict(payload.get("variables") or {})
        variables = VariableModel(**{k: float(v) for k, v in var_data.items()
                                     if k in VariableModel.__dataclass_fields__})
        calibration = payload.get("calibration")
        surrogate = (UltimateStrengthSurrogate.from_dict(dict(calibration))
                     if calibration else None)
        options = dict(payload.get("options") or {})
        return cls(panel=panel, variables=variables, surrogate=surrogate,
                   ds0=float(options.get("ds0", 6e-4)),
                   max_steps=int(options.get("max_steps", 18)),
                   deflection_limit_ratio=float(options.get("deflection_limit_ratio", 0.01)))