"""Versioned field-exchange contracts for the fluid / structure / ice coupling.

The campaign node consumes the upstream interface contracts produced by

* ``n01`` -- FEM constitutive hand-off (stress/strain state carried by the
  structural solver), and
* ``n02`` -- DEM particle hand-off (ice fragment packets produced by the
  fracture front end).

Both are reduced here to the minimum surface that the three-field coupler
needs, and every packet is validated against the conservation law that makes
the exchange *admissible*: mass for the DEM hand-off, momentum for the
fluid/structure hand-off, and energy for the fracture ``FEM -> DEM`` hand-off.

Rationale for validating rather than trusting: a hand-off that silently
duplicates or drops mass/momentum produces a coupling that looks converged
while diverging from the physics.  These gates are cheap and fail loudly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

CONTRACT_VERSION = "fsi_ice.field_exchange.v1"

#: Relative tolerance for the mass hand-off (JSON float64 round trip).
MASS_TOL = 1e-12
#: Relative tolerance for momentum hand-off (sum of nodal forces vs. resultant).
MOMENTUM_TOL = 1e-10
#: Relative tolerance for the fracture energy hand-off.
ENERGY_TOL = 1e-8


class FieldExchangeError(ValueError):
    """A coupling packet violates its conservation contract."""


def _as_float_array(name: str, value: Any, ndim: int) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim != ndim:
        raise FieldExchangeError(f"{name} must be {ndim}-D, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise FieldExchangeError(f"{name} contains non-finite entries")
    return arr


@dataclass(frozen=True)
class FluidFieldPacket:
    """Fluid -> structure/particles hand-off (one coupling instant).

    ``velocity`` and ``pressure`` are sampled on the *coupling grid* (the
    LBM lattice in lattice units) so that no solver-internal array layout
    leaks across the interface.
    """

    time: float
    velocity: np.ndarray  # (nz, ny, nx, 3)
    pressure: np.ndarray  # (nz, ny, nx)
    solid_mask: np.ndarray  # (nz, ny, nx) bool
    hull_force: np.ndarray  # (3,) momentum-exchange force on the body
    density: float
    schema: str = CONTRACT_VERSION

    def validate(self) -> None:
        v = _as_float_array("velocity", self.velocity, 4)
        p = _as_float_array("pressure", self.pressure, 3)
        if v.shape[:3] != p.shape:
            raise FieldExchangeError("velocity/pressure grid mismatch")
        if v.shape[3] != 3:
            raise FieldExchangeError("velocity must carry 3 components")
        if not np.isfinite(self.time):
            raise FieldExchangeError("time must be finite")
        _as_float_array("hull_force", self.hull_force, 1)


@dataclass(frozen=True)
class StructurePacket:
    """Structure -> fluid hand-off: the wetted boundary motion.

    ``node_positions``/``node_velocities`` are the *hull* surface samples in
    lattice units, used to prescribe the moving bounce-back wall velocity.
    ``loads`` is the nodal contact load vector handed to the FEM solver.
    """

    time: float
    node_positions: np.ndarray  # (n, 3)
    node_velocities: np.ndarray  # (n, 3)
    loads: np.ndarray  # (n, 3)
    schema: str = CONTRACT_VERSION

    def validate(self) -> None:
        pos = _as_float_array("node_positions", self.node_positions, 2)
        vel = _as_float_array("node_velocities", self.node_velocities, 2)
        loads = _as_float_array("loads", self.loads, 2)
        if not (pos.shape == vel.shape == loads.shape):
            raise FieldExchangeError("structure packet arrays must share a shape")
        if pos.shape[1] != 3:
            raise FieldExchangeError("structure packet must be 3-D per node")

    @property
    def resultant(self) -> np.ndarray:
        return self.loads.sum(axis=0)


@dataclass(frozen=True)
class ParticlePacket:
    """DEM particle hand-off consumed by the fluid coupling (n02 contract).

    ``radius``/``mass``/``position``/``velocity`` are the minimum state the
    coupler needs to (a) compute drag from the fluid field and (b) map contact
    loads onto the structure.
    """

    time: float
    position: np.ndarray  # (n, 3)
    velocity: np.ndarray  # (n, 3)
    radius: np.ndarray  # (n,)
    mass: np.ndarray  # (n,)
    schema: str = CONTRACT_VERSION

    def validate(self) -> None:
        pos = _as_float_array("position", self.position, 2)
        vel = _as_float_array("velocity", self.velocity, 2)
        rad = _as_float_array("radius", self.radius, 1)
        mass = _as_float_array("mass", self.mass, 1)
        if pos.shape != vel.shape or pos.shape[1] != 3:
            raise FieldExchangeError("particle position/velocity shape mismatch")
        if rad.shape != pos.shape[:1] or mass.shape != pos.shape[:1]:
            raise FieldExchangeError("particle radius/mass must match particle count")
        if np.any(rad <= 0.0) or np.any(mass <= 0.0):
            raise FieldExchangeError("particle radius and mass must be positive")

    @property
    def total_mass(self) -> float:
        return float(self.mass.sum())


def check_mass_conservation(
    fragments: Mapping[str, Any],
    packet: ParticlePacket,
    *,
    tol: float = MASS_TOL,
) -> float:
    """Assert that a FEM->DEM fragment hand-off preserves mass.

    ``fragments`` follows the n02 contract: ``{"volume": array, "density": rho}``.
    Returns the relative error that was verified.
    """
    volume = _as_float_array("fragment volume", fragments["volume"], 1)
    rho = float(fragments["density"])
    expected = float(volume.sum() * rho)
    if expected <= 0.0:
        raise FieldExchangeError("fragment hand-off carries no mass")
    packet.validate()
    rel = abs(packet.total_mass - expected) / expected
    if rel > tol:
        raise FieldExchangeError(
            f"mass hand-off error {rel:.3e} exceeds tolerance {tol:.3e}"
        )
    return rel


def check_momentum_balance(
    packet: StructurePacket,
    resultant: Sequence[float],
    *,
    tol: float = MOMENTUM_TOL,
) -> float:
    """Assert that the nodal load vector reproduces the applied resultant."""
    packet.validate()
    target = _as_float_array("resultant", resultant, 1)
    if target.shape[0] != 3:
        raise FieldExchangeError("resultant must be a 3-vector")
    scale = max(float(np.linalg.norm(target)), 1e-30)
    rel = float(np.linalg.norm(packet.resultant - target)) / scale
    if rel > tol:
        raise FieldExchangeError(
            f"momentum hand-off error {rel:.3e} exceeds tolerance {tol:.3e}"
        )
    return rel


def check_energy_handoff(
    release_energies: Iterable[float],
    packet_energies: Iterable[float],
    *,
    tol: float = ENERGY_TOL,
) -> float:
    """Assert that fracture surface energy survives the FEM -> DEM hand-off."""
    a = _as_float_array("release energies", np.asarray(list(release_energies), float), 1)
    b = _as_float_array("packet energies", np.asarray(list(packet_energies), float), 1)
    if a.shape != b.shape:
        raise FieldExchangeError("energy hand-off shapes differ")
    scale = max(float(np.abs(a).sum()), 1e-30)
    rel = float(np.abs(a - b).sum()) / scale
    if rel > tol:
        raise FieldExchangeError(
            f"energy hand-off error {rel:.3e} exceeds tolerance {tol:.3e}"
        )
    return rel