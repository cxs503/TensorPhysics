"""Ultimate-strength surrogate calibrated on the arc-length finite-element runs.

The reliability problem needs the ultimate capacity ``sigma_u`` at thousands of
random realisations of the thickness and the yield stress, which is impossible
with the full nonlinear FE model (tens of seconds per limit-point trace).  We
therefore calibrate a *response surface* in the plate-slenderness variable
``beta = (b/t) sqrt(sigma_y/E)``:

.. math::
    \\sigma_u/\\sigma_y = c_0 + c_1/\\beta + c_2/\\beta^2

which is the von Karman / Faulkner effective-width family (Faulkner's formula
is the special case ``c = (0, 2, -1)``) that plate ultimate-strength empirical
fits, including the Paik-type ALPS fits, are built on.  The coefficients are
fitted by least squares to the *finite-element* limit points -- the surrogate
is a fit, never a replacement, and its residual against the FE calibration
points is reported with the model (``accuracy``) and used as a documented
source of model uncertainty.

The fit is in the slenderness variable only, so it transfers across panels with
the same aspect ratio ``a/b`` and the same imperfection level; this is stated as
a limitation, and the calibration record carries the aspect ratio so callers
can check it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

__all__ = ["UltimateStrengthSurrogate", "faulkner_reference"]

#: von Karman / Faulkner effective-width curve (independent analytic reference).
def faulkner_reference(beta: float) -> float:
    """``sigma_u/sigma_y = 2/beta - 1/beta^2`` for ``beta >= 1``."""
    beta = np.asarray(beta, dtype=float)
    return np.where(beta >= 1.0, 2.0 / beta - 1.0 / beta**2, 1.0)


@dataclass(frozen=True)
class UltimateStrengthSurrogate:
    """``sigma_u/sigma_y = c0 + c1/beta + c2/beta**2`` with a validity window."""

    c0: float
    c1: float
    c2: float
    beta_min: float = 1.0
    beta_max: float = 4.0
    aspect: float = 3.0
    imperfection_level: str = "average"
    calibration: List[Dict[str, float]] = field(default_factory=list)
    source: str = "calibrated"

    # ------------------------------------------------------------- evaluation
    def ratio(self, beta) -> np.ndarray:
        """``sigma_u/sigma_y`` at slenderness ``beta`` (vectorised)."""
        beta = np.asarray(beta, dtype=float)
        safe = np.where(beta > 0.0, beta, np.nan)
        return self.c0 + self.c1 / safe + self.c2 / safe**2

    def sigma_u(self, t: float, sigma_y: float, b: float, E: float) -> float:
        """Ultimate compressive stress (Pa) of a panel of net thickness ``t``."""
        t = float(t)
        if t <= 0.0:
            raise ValueError("net thickness must be positive")
        beta = (b / t) * np.sqrt(sigma_y / E)
        return float(sigma_y * float(self.ratio(beta)))

    def sigma_u_batch(self, t, sigma_y, b: float, E: float) -> np.ndarray:
        """Vectorised :meth:`sigma_u` for ``DesignPoint``/Monte-Carlo batches."""
        t = np.asarray(t, dtype=float)
        if np.any(t <= 0.0):
            raise ValueError("net thickness must be positive")
        beta = (b / t) * np.sqrt(np.asarray(sigma_y, dtype=float) / E)
        return np.asarray(sigma_y, dtype=float) * self.ratio(beta)

    def extrapolated(self, beta) -> np.ndarray:
        """Boolean mask of the points outside the calibration window."""
        beta = np.asarray(beta, dtype=float)
        return (beta < self.beta_min) | (beta > self.beta_max)

    # ------------------------------------------------------------- calibration
    @classmethod
    def fit(cls, points: Iterable[Dict[str, float]],
            aspect: float = 3.0, imperfection_level: str = "average",
            source: str = "calibrated") -> "UltimateStrengthSurrogate":
        """Least-squares fit to finite-element limit points.

        ``points`` entries need ``beta`` and ``sigma_u_over_sigma_y`` (the
        fields written by :meth:`DeterministicAssessment.ultimate`).
        """
        rows = list(points)
        if len(rows) < 3:
            raise ValueError("at least three calibration points are required")
        beta = np.array([float(r["beta"]) for r in rows])
        ratio = np.array([float(r["sigma_u_over_sigma_y"]) for r in rows])
        design = np.column_stack([np.ones_like(beta), 1.0 / beta, 1.0 / beta**2])
        coeffs, *_ = np.linalg.lstsq(design, ratio, rcond=None)
        return cls(c0=float(coeffs[0]), c1=float(coeffs[1]), c2=float(coeffs[2]),
                   beta_min=float(beta.min()), beta_max=float(beta.max()),
                   aspect=float(aspect), imperfection_level=imperfection_level,
                   calibration=rows, source=source)

    def accuracy(self) -> Dict[str, float]:
        """Residual of the fit against its own FE calibration points."""
        if not self.calibration:
            return {}
        beta = np.array([float(r["beta"]) for r in self.calibration])
        fe = np.array([float(r["sigma_u_over_sigma_y"]) for r in self.calibration])
        fit = self.ratio(beta)
        rel = fit / fe - 1.0
        return {
            "n_points": int(beta.size),
            "max_rel_error": float(np.max(np.abs(rel))),
            "mean_rel_error": float(np.mean(rel)),
            "rms_rel_error": float(np.sqrt(np.mean(rel**2))),
        }

    # ------------------------------------------------------------ persistence
    def to_dict(self) -> Dict[str, object]:
        return {
            "model": "sigma_u/sigma_y = c0 + c1/beta + c2/beta^2",
            "c0": self.c0, "c1": self.c1, "c2": self.c2,
            "beta_min": self.beta_min, "beta_max": self.beta_max,
            "aspect": self.aspect,
            "imperfection_level": self.imperfection_level,
            "source": self.source,
            "accuracy": self.accuracy(),
            "calibration": [dict(r) for r in self.calibration],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "UltimateStrengthSurrogate":
        rows = [dict(r) for r in data.get("calibration", [])]
        return cls(c0=float(data["c0"]), c1=float(data["c1"]), c2=float(data["c2"]),
                   beta_min=float(data.get("beta_min", 1.0)),
                   beta_max=float(data.get("beta_max", 4.0)),
                   aspect=float(data.get("aspect", 3.0)),
                   imperfection_level=str(data.get("imperfection_level", "average")),
                   calibration=rows,
                   source=str(data.get("source", "package_data")))

    def save(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path) -> "UltimateStrengthSurrogate":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    # ------------------------------------------------------------------ misc
    def coefficients(self) -> Sequence[float]:
        return (self.c0, self.c1, self.c2)

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return (f"sigma_u/sigma_y = {self.c0:.4f} + {self.c1:.4f}/beta "
                f"+ {self.c2:.4f}/beta^2  (beta in "
                f"[{self.beta_min:.2f}, {self.beta_max:.2f}])")


def default_surrogate_path() -> Path:
    """Location of the packaged calibration produced by the verification run."""
    return Path(__file__).with_name("data") / "fem_calibration.json"


def load_default_surrogate() -> Optional[UltimateStrengthSurrogate]:
    """Packaged FE calibration, or ``None`` when the artifact is absent."""
    path = default_surrogate_path()
    if not path.exists():
        return None
    return UltimateStrengthSurrogate.load(path)