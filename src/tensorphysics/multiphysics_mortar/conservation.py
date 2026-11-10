"""Mass / momentum / energy conservation residuals of an interface transfer.

Three integrated quantities are tracked separately because they fail separately:

``mass``      ``int_Gamma rho dGamma``            -- leaks appear as a net source/sink at the interface
``momentum``  ``int_Gamma rho u dGamma`` (vector)  -- a spurious interface force (added mass / drag bias)
``energy``    ``int_Gamma rho E dGamma``          -- a spurious interface work (heating or cooling)

with the caloric stand-in ``E = c_v T + |u|^2 / 2``.  For a weakly compressible
liquid ``c_v ~= c_p`` is the usual engineering choice; ``specific_heat`` is a
parameter so the tests state the value they use instead of hiding it.

Residual definition (identical for every quantity ``q``):

    R_q = || I_tgt - I_src || / || I_src ||,   I = int_Gamma q dGamma ,

with a machine-scale floor on the denominator so a zero-mean payload (e.g. a
purely tangential velocity field) gives ``R_q = 0`` instead of ``0/0``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "MaterialState",
    "QuantityResidual",
    "QUANTITIES",
    "conservation_residuals",
    "integrate",
    "local_balance_defect",
]

QUANTITIES = ("mass", "momentum", "energy")
_FLOOR = 1e-300


@dataclass(frozen=True)
class MaterialState:
    """Nodal state on an interface trace space (SI units)."""

    density: np.ndarray
    velocity: np.ndarray
    temperature: np.ndarray
    specific_heat: float = 4182.0

    def __post_init__(self) -> None:
        rho = np.asarray(self.density, dtype=float).ravel()
        vel = np.asarray(self.velocity, dtype=float)
        if vel.ndim == 1:
            vel = vel[:, None]
        temp = np.asarray(self.temperature, dtype=float).ravel()
        if vel.shape[0] != rho.size or temp.size != rho.size:
            raise ValueError("density / velocity / temperature size mismatch")
        object.__setattr__(self, "density", rho)
        object.__setattr__(self, "velocity", vel)
        object.__setattr__(self, "temperature", temp)

    @property
    def dimension(self) -> int:
        return int(self.velocity.shape[1])

    @property
    def momentum(self) -> np.ndarray:
        return self.density[:, None] * self.velocity

    @property
    def energy(self) -> np.ndarray:
        kinetic = 0.5 * np.sum(self.velocity ** 2, axis=1)
        return self.density * (self.specific_heat * self.temperature + kinetic)

    def quantities(self) -> dict[str, np.ndarray]:
        return {"mass": self.density[:, None],
                "momentum": self.momentum,
                "energy": self.energy[:, None]}


@dataclass(frozen=True)
class QuantityResidual:
    """Integrated balance of one quantity across the interface."""

    quantity: str
    source_integral: np.ndarray
    target_integral: np.ndarray
    residual: float

    @property
    def absolute(self) -> np.ndarray:
        return self.target_integral - self.source_integral

    def to_dict(self) -> dict:
        return {
            "quantity": self.quantity,
            "source_integral": [float(v) for v in np.atleast_1d(self.source_integral)],
            "target_integral": [float(v) for v in np.atleast_1d(self.target_integral)],
            "absolute_deficit": [float(v) for v in np.atleast_1d(self.absolute)],
            "residual": self.residual,
        }


def integrate(space, values: np.ndarray) -> np.ndarray:
    """``int_Gamma q dGamma`` from the space's own (independent) measures."""
    vals = np.asarray(values, dtype=float)
    if vals.ndim == 1:
        vals = vals[:, None]
    if vals.shape[0] != space.n:
        raise ValueError(f"payload has {vals.shape[0]} DOFs, space has {space.n}")
    return space.measures() @ vals


def conservation_residuals(src_space, src_state: MaterialState,
                           tgt_space, tgt_state: MaterialState
                           ) -> dict[str, QuantityResidual]:
    """Residuals of the three conservative quantities across one interface."""
    q_src = src_state.quantities()
    q_tgt = tgt_state.quantities()
    out: dict[str, QuantityResidual] = {}
    for name in QUANTITIES:
        i_src = integrate(src_space, q_src[name])
        i_tgt = integrate(tgt_space, q_tgt[name])
        den = max(float(np.linalg.norm(i_src)), _FLOOR)
        out[name] = QuantityResidual(quantity=name, source_integral=i_src,
                                     target_integral=i_tgt,
                                     residual=float(np.linalg.norm(i_tgt - i_src)) / den)
    return out


def local_balance_defect(assembly, src_state: MaterialState,
                         tgt_state: MaterialState) -> dict[str, np.ndarray]:
    """Per-receiving-DOF balance defect, normalised by the interface flux.

    ``d_i = | A_i q_i^tgt - int_Gamma q_src phi_i^tgt | / sum_i | int_Gamma q_src phi_i^tgt |``

    With a non-matching interface a *globally* conservative projection cannot make
    every ``d_i`` vanish -- the coarse cell cannot reproduce a fine-cell
    distribution it does not resolve.  The global residual is therefore the gate,
    and this field is the diagnostic map (max and L1 norms are reported).
    """
    q_src = src_state.quantities()
    q_tgt = tgt_state.quantities()
    out: dict[str, np.ndarray] = {}
    for name in QUANTITIES:
        flux = assembly.C @ q_src[name]                       # (n_tgt, k)
        own = assembly.tgt.measures()[:, None] * q_tgt[name]   # (n_tgt, k)
        den = max(float(np.sum(np.abs(flux))), _FLOOR)
        out[name] = np.abs(own - flux).sum(axis=1) / den
    return out