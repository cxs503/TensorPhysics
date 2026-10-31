"""Panel specification and geometry gate for the unified buckling/reliability chain.

The unified framework (W4-N03) couples two previously separate capabilities:
the deterministic arc-length nonlinear buckling solver
(:mod:`tensorphysics.plate_buckling`, W3-N03) and the FORM/SORM structural
reliability core (:mod:`tensorphysics.structural`, W3-N04).  The coupling is
done on a *panel specification*: one immutable description of the geometry,
material, mesh, imperfection level and secondary load state that is consumed by
both halves of the chain.  Keeping the specification in one place is what makes
the deterministic limit load and the probabilistic limit state refer to the
same physical panel -- the classic failure mode of such couplings is a silent
geometry/unit mismatch between the two halves.

Units
-----
SI everywhere in this module (m, Pa, N): the finite-element model and the
distributions are strictly SI.  The public API/JSON layer converts to MPa/mm
for readability (see :mod:`.api` and the report renderer).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Tuple

import numpy as np

from ..plate_buckling import PlateGeometry
from ..plate_buckling.core import GeometryError as _CoreGeometryError

#: Reference material of the W3-N03 plate-buckling campaign (ice-grade NV/AH32).
DEFAULT_E = 2.06e11
DEFAULT_NU = 0.3
DEFAULT_SIGMA_Y = 313.0e6

__all__ = ["PanelSpec", "GeometryGateError", "DEFAULT_E", "DEFAULT_NU",
           "DEFAULT_SIGMA_Y"]


class GeometryGateError(ValueError):
    """Raised when the panel fails the pre-analysis geometry gate."""


@dataclass(frozen=True)
class PanelSpec:
    """One ice-belt plate panel: geometry, material, mesh, load state.

    Parameters
    ----------
    a, b, t:
        Panel length (x, longitudinal), breadth (y, between frames) and
        thickness in metres.
    E, nu, sigma_y, E_t:
        Elastic modulus, Poisson ratio, yield stress and hardening modulus.
    nx, ny:
        Finite-element mesh of the deterministic solver.
    bc:
        Boundary condition code of the plate model (``SSSS``, ``CCCC``,
        ``SCSC``, ``CSCS``).
    imperfection_level, sigma_r_ratio:
        Standard welding-distortion level (``slight``/``average``/``severe``)
        and residual-stress amplitude as a fraction of the yield stress.
    sigma_global:
        Secondary/global membrane compression carried by the panel (Pa).
    psi_membrane:
        Membrane amplification of the local ice pressure (Pa per Pa).
    beta_range, aspect_range, max_elem_aspect:
        Design envelope used by the geometry gate.  ``beta_range`` is wider
        than the W3-N03 verification envelope because the reliability chain
        evaluates capacities down to the target-reliability thickness.
    """

    a: float = 3.0
    b: float = 1.0
    t: float = 19.49e-3
    E: float = DEFAULT_E
    nu: float = DEFAULT_NU
    sigma_y: float = DEFAULT_SIGMA_Y
    E_t: float = 0.0
    nx: int = 8
    ny: int = 4
    bc: str = "SSSS"
    imperfection_level: str = "average"
    sigma_r_ratio: float = 0.0
    sigma_global: float = 95.0e6
    psi_membrane: float = 1.5
    beta_range: Tuple[float, float] = (1.0, 4.5)
    aspect_range: Tuple[float, float] = (2.0, 6.0)
    max_elem_aspect: float = 3.0

    # ------------------------------------------------------------------ props
    @property
    def beta(self) -> float:
        """Plate slenderness ``beta = (b/t) sqrt(sigma_y/E)``."""
        return (self.b / self.t) * np.sqrt(self.sigma_y / self.E)

    def geometry(self) -> PlateGeometry:
        """Return the finite-element geometry of this panel."""
        return PlateGeometry(a=self.a, b=self.b, t=self.t, E=self.E, nu=self.nu,
                             sigma_y=self.sigma_y, E_t=self.E_t,
                             nx=self.nx, ny=self.ny)

    def validate(self) -> Dict[str, float]:
        """Run the geometry gate *before* any analysis (verify-before-solve).

        The gate rejects a panel whose aspect ratio, slenderness or mesh
        quality is outside the envelope in which the deterministic solver and
        the capacity surrogate were verified.  Raises
        :class:`GeometryGateError` on failure; returns the checked numbers so
        they can be persisted as provenance instead of being re-typed.
        """
        try:
            checks = self.geometry().validate(
                aspect_range=self.aspect_range,
                beta_range=self.beta_range,
                max_elem_aspect=self.max_elem_aspect,
            )
        except _CoreGeometryError as exc:
            # single exception type for every geometry-gate failure
            raise GeometryGateError(str(exc)) from exc
        b_over_t = self.b / self.t
        checks["b_over_t"] = float(b_over_t)
        checks["t_mm"] = float(self.t * 1e3)
        checks["sigma_y_MPa"] = float(self.sigma_y / 1e6)
        if b_over_t > 200.0:
            raise GeometryGateError(
                f"b/t={b_over_t:.1f} > 200: plate outside the stocky-panel domain "
                "of the capacity surrogate")
        return checks

    def with_thickness(self, t: float) -> "PanelSpec":
        """Same panel with a different as-built thickness (design iteration)."""
        new = replace(self, t=float(t))
        new.validate()
        return new

    def as_dict(self) -> Dict[str, object]:
        return {
            "a_m": float(self.a),
            "b_m": float(self.b),
            "t_mm": float(self.t * 1e3),
            "E_GPa": float(self.E / 1e9),
            "nu": float(self.nu),
            "sigma_y_MPa": float(self.sigma_y / 1e6),
            "E_t_MPa": float(self.E_t / 1e6),
            "mesh": [int(self.nx), int(self.ny)],
            "bc": self.bc,
            "imperfection_level": self.imperfection_level,
            "sigma_r_ratio": float(self.sigma_r_ratio),
            "sigma_global_MPa": float(self.sigma_global / 1e6),
            "psi_membrane": float(self.psi_membrane),
            "beta": float(self.beta),
            "a_over_b": float(self.a / self.b),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "PanelSpec":
        """Build from an API payload.

        ``t``/``t_mm``, ``sigma_y`` (Pa) / ``sigma_y_MPa`` and ``E`` (Pa) /
        ``E_GPa`` are both accepted; a value below ``1e6`` in a stress field is
        interpreted as MPa.
        """
        d = dict(data)

        def stress(keys, default):
            for key in keys:
                if key in d:
                    value = float(d.pop(key))
                    if key.endswith("_MPa") or (0.0 < value < 1.0e6):
                        value *= 1e6
                    return value
            return float(default)

        t = float(d.pop("t", d.pop("t_mm", 19.49) * 1e-3))
        sigma_y = stress(("sigma_y", "sigma_y_MPa"), DEFAULT_SIGMA_Y)
        E = float(d.pop("E", d.pop("E_GPa", 206.0) * 1e9))
        E_t = stress(("E_t", "E_t_MPa"), 0.0) if ("E_t" in d or "E_t_MPa" in d) else 0.0
        sigma_global = stress(("sigma_global", "sigma_global_MPa"), 0.0)
        mesh = d.pop("mesh", None)
        nx = int(d.pop("nx", mesh[0] if mesh else 8))
        ny = int(d.pop("ny", mesh[1] if mesh else 4))
        fields = {
            "a": float(d.pop("a", d.pop("a_m", 3.0))),
            "b": float(d.pop("b", d.pop("b_m", 1.0))),
            "t": t, "E": E, "sigma_y": sigma_y, "E_t": E_t,
            "sigma_global": sigma_global, "nx": nx, "ny": ny,
        }
        allowed = set(cls.__dataclass_fields__)
        for key in list(d):
            if key in allowed:
                fields[key] = d.pop(key)
        return cls(**fields)