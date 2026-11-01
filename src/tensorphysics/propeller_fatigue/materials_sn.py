"""S-N curve description (Basquin) and mean-stress correction.

Reference fatigue data (fully reversed, R = -1) are literature-approximate
points for propeller-relevant alloys.  They are stored with explicit provenance
so the report can distinguish measured/digitised data from engineering table
values; the fit error reported by :meth:`BasquinSN.fit_error` is computed against
these points and nothing else.

PROVENANCE = "LIT_TABLE" : typical values quoted in propeller-design and
fatigue handbooks for cast Ni-Al-bronze and 6xxx/2xxx aluminium; they are not
digitised from a single original paper and carry ~+-5 % table scatter.  This is
stated in the report as a limitation of criterion (d).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np


@dataclass(frozen=True)
class Material:
    key: str
    name: str
    E: float          # Young's modulus [Pa]
    rho: float        # density [kg/m^3]
    sigma_uts: float  # ultimate tensile strength [Pa]
    sigma_y: float    # 0.2 % proof stress [Pa]
    sn_points: Tuple[Tuple[float, float], ...]  # ((N, sigma_a_R=-1 [Pa]), ...)


#: Reference S-N points: (cycles to failure, stress amplitude at R = -1).
_MATERIALS: Dict[str, Material] = {
    "NAB": Material(
        key="NAB",
        name="Cast Ni-Al-bronze (CuAl9Ni5Fe4Mn, propeller grade)",
        E=118.0e9,
        rho=7640.0,
        sigma_uts=650.0e6,
        sigma_y=280.0e6,
        sn_points=(
            (1.0e4, 340.0e6),
            (1.0e5, 290.0e6),
            (1.0e6, 220.0e6),
            (1.0e7, 175.0e6),
        ),
    ),
    "AL6082": Material(
        key="AL6082",
        name="Al-Mg-Si 6082-T6 (blade / bracket reference alloy)",
        E=70.0e9,
        rho=2700.0,
        sigma_uts=310.0e6,
        sigma_y=260.0e6,
        sn_points=(
            (1.0e4, 240.0e6),
            (1.0e5, 160.0e6),
            (1.0e6, 110.0e6),
            (1.0e7, 85.0e6),
        ),
    ),
}

MATERIALS = _MATERIALS


@dataclass
class SNReference:
    """A named S-N reference data set (material points + provenance)."""

    material_key: str
    points: np.ndarray = field(init=False)
    provenance: str = "LIT_TABLE"

    def __post_init__(self) -> None:
        m = MATERIALS[self.material_key]
        self.points = np.asarray(m.sn_points, dtype=float)
        self.material = m

    @property
    def N(self) -> np.ndarray:
        return self.points[:, 0]

    @property
    def sigma_a(self) -> np.ndarray:
        return self.points[:, 1]


class BasquinSN:
    """Basquin law ``sigma_a = sigma_f' (2 N)^b`` fitted in log-log space.

    Parameters are obtained by ordinary least squares on
    ``log10(sigma_a)`` vs ``log10(2N)`` of the reference points, i.e. the fit is
    *predictive* (it interpolates/extrapolates the table) and its residual is an
    honest measure of how well the Basquin form describes the data.
    """

    def __init__(self, sigma_f: float, b: float, material_key: str = "", sigma_uts: float = 650.0e6):
        self.sigma_f = float(sigma_f)
        self.b = float(b)
        self.material_key = material_key
        self.sigma_uts = float(sigma_uts)

    # ------------------------------------------------------------------ fit
    @classmethod
    def fit(cls, ref: SNReference) -> "BasquinSN":
        x = np.log10(2.0 * ref.N)
        y = np.log10(ref.sigma_a)
        b, log_sf = np.polyfit(x, y, 1)
        return cls(sigma_f=10.0 ** log_sf, b=b, material_key=ref.material_key,
                   sigma_uts=ref.material.sigma_uts)

    # --------------------------------------------------------------- query
    def sigma_a(self, N):
        """Stress amplitude [Pa] giving failure at ``N`` cycles (scalar or array)."""
        N = np.asarray(N, dtype=float)
        out = self.sigma_f * (2.0 * N) ** self.b
        return float(out) if out.ndim == 0 else out

    def cycles(self, sigma_a: float) -> float:
        """Cycles to failure at stress amplitude ``sigma_a`` [Pa]."""
        sigma_a = np.asarray(sigma_a, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            n = 0.5 * (sigma_a / self.sigma_f) ** (1.0 / self.b)
        return float(n) if np.isscalar(sigma_a) else n

    # ------------------------------------------------------------ quality
    def fit_error(self, ref: SNReference) -> Dict[str, float]:
        """Relative residuals of the fit on the reference points themselves."""
        pred = self.sigma_a(ref.N)
        rel = (pred - ref.sigma_a) / ref.sigma_a
        return {
            "max_abs_rel": float(np.max(np.abs(rel))),
            "rms_rel": float(np.sqrt(np.mean(rel ** 2))),
            "residuals": rel.tolist(),
        }

    def exponent_sigma_range(self) -> float:
        """Inverse Basquin exponent 1/|b| (slope of the S-N line in decades)."""
        return 1.0 / abs(self.b)

    def as_dict(self) -> Dict[str, float]:
        return {
            "material": self.material_key,
            "sigma_f_Pa": self.sigma_f,
            "b": self.b,
            "inverse_slope": self.exponent_sigma_range(),
            "sigma_uts_Pa": self.sigma_uts,
        }


def goodman_amplitude(sigma_a, sigma_m, sigma_uts):
    """Goodman mean-stress correction: ``sigma_a_eff = sigma_a / (1 - sigma_m/UTS)``.

    Clipped at ``sigma_m = 0.95 UTS`` to keep the far tail finite.
    """
    sigma_a = np.asarray(sigma_a, dtype=float)
    sigma_m = np.asarray(sigma_m, dtype=float)
    ratio = np.clip(sigma_m / sigma_uts, -0.95, 0.95)
    return sigma_a / (1.0 - ratio)


def material_list() -> List[str]:
    return sorted(MATERIALS)