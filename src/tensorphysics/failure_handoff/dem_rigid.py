"""Minimal rigid-fragment DEM integrator used to close the energy chain.

Scope: translational rigid bodies (each fragment = one polyhedron with mass ``m``,
equivalent-sphere contact radius ``r``) under gravity, with a planar ground.  The
integrator is semi-implicit Euler *with exact contact-time splitting*: when a body
would cross the ground inside a step, the step is split at the analytic contact
time, the normal velocity is reflected with restitution ``e`` and the Coulomb
tangential impulse is applied.  That removes the classic "penetration then push
out" energy artefact, so an elastic (``e = 1``, ``mu = 0``) run must conserve
``KE + PE + internal`` to round-off — a property the acceptance criterion (C4)
uses and the tests assert.

Rotation is **not** integrated (fragments are treated as point masses for the
energy ledger).  For a gravity-only release from rest with no inter-fragment
contacts the rotational degree of freedom carries no energy, so the energy chain
is unaffected; this limitation is stated in the report.  Fragment–fragment
collisions are likewise not resolved in this verification driver.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .dem_fragments import FragmentSet

__all__ = [
    "RigidFragmentState",
    "RigidDEM",
    "EnergyHistory",
]


@dataclass
class RigidFragmentState:
    fragment_id: str
    mass_kg: float
    radius_m: float
    position_m: np.ndarray
    velocity_m_s: np.ndarray
    internal_energy_j: float


@dataclass
class EnergyHistory:
    times: list[float] = field(default_factory=list)
    kinetic: list[float] = field(default_factory=list)
    potential: list[float] = field(default_factory=list)
    internal: list[float] = field(default_factory=list)
    total: list[float] = field(default_factory=list)
    dissipated: list[float] = field(default_factory=list)
    min_height: list[float] = field(default_factory=list)
    contact_events: int = 0

    @property
    def initial_total(self) -> float:
        return self.total[0]

    def max_relative_deviation(self) -> float:
        """``max |E(t) - E(0)| / E(0)`` over the recorded history."""
        if not self.total:
            return 0.0
        e0 = self.total[0]
        if e0 == 0.0:
            return 0.0
        return max(abs(value - e0) / e0 for value in self.total)

    def closure_with_dissipation(self) -> float:
        """``max |E(t) + dissipated(t) - E(0)| / E(0)`` — the true closure."""
        if not self.total:
            return 0.0
        e0 = self.total[0]
        if e0 == 0.0:
            return 0.0
        return max(
            abs(total + diss - e0) / e0
            for total, diss in zip(self.total, self.dissipated)
        )

    def as_dict(self) -> dict:
        return {
            "times": self.times,
            "kinetic_j": self.kinetic,
            "potential_j": self.potential,
            "internal_j": self.internal,
            "total_j": self.total,
            "dissipated_j": self.dissipated,
            "min_height_m": self.min_height,
            "contact_events": self.contact_events,
            "initial_total_j": self.initial_total,
            "max_relative_deviation": self.max_relative_deviation(),
            "closure_with_dissipation": self.closure_with_dissipation(),
        }


class RigidDEM:
    """Gravity-driven rigid-fragment integrator with a planar ground."""

    def __init__(
        self,
        fragment_set: FragmentSet,
        *,
        gravity_m_s2: float = 9.81,
        ground_y_m: float = 0.0,
        restitution: float = 1.0,
        friction: float = 0.0,
        dt_s: float = 1.0e-4,
    ) -> None:
        if not 0.0 <= restitution <= 1.0:
            raise ValueError("restitution must be in [0, 1]")
        if friction < 0.0:
            raise ValueError("friction must be non-negative")
        if dt_s <= 0.0:
            raise ValueError("dt_s must be positive")
        self.fragment_set = fragment_set
        self.gravity = float(gravity_m_s2)
        self.ground = float(ground_y_m)
        self.restitution = float(restitution)
        self.friction = float(friction)
        self.dt = float(dt_s)
        self.states: list[RigidFragmentState] = [
            RigidFragmentState(
                fragment_id=fragment.fragment_id,
                mass_kg=fragment.mass_kg,
                radius_m=fragment.equivalent_radius_m,
                position_m=np.array(fragment.centroid_m, dtype=float),
                velocity_m_s=np.zeros(3),
                internal_energy_j=fragment.fracture_energy_j,
            )
            for fragment in fragment_set.fragments
        ]
        self.time = 0.0
        self.dissipated_j = 0.0
        self.contact_events = 0
        self._initial_shift = 0.0

    # -- frame helpers ---------------------------------------------------
    def _arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        masses = np.array([s.mass_kg for s in self.states])
        radii = np.array([s.radius_m for s in self.states])
        positions = np.array([s.position_m for s in self.states])
        velocities = np.array([s.velocity_m_s for s in self.states])
        internal = np.array([s.internal_energy_j for s in self.states])
        return masses, radii, positions, velocities, internal

    # -- energy ledger ---------------------------------------------------
    def energies(self) -> dict[str, float]:
        masses, _radii, positions, velocities, internal = self._arrays()
        kinetic = float(0.5 * np.sum(masses * np.sum(velocities**2, axis=1)))
        potential = float(self.gravity * np.sum(masses * (positions[:, 1] - self.ground)))
        return {
            "kinetic_j": kinetic,
            "potential_j": potential,
            "internal_j": float(np.sum(internal)),
            "total_j": kinetic + potential + float(np.sum(internal)),
            "dissipated_j": self.dissipated_j,
        }

    # -- integration -----------------------------------------------------
    def step(self, dt: float | None = None) -> None:
        """Advance by ``dt`` with exact contact-time splitting."""
        remaining = self.dt if dt is None else float(dt)
        if remaining <= 0.0:
            return
        masses, radii, positions, velocities, _internal = self._arrays()
        g = self.gravity
        while remaining > 1e-18:
            # height above the contact plane per body
            height = positions[:, 1] - radii - self.ground
            vel_y = velocities[:, 1]
            contact = height <= 1e-12
            # analytic time to reach the contact plane for bodies still in flight
            flight = ~contact
            t_hit = np.full(len(height), np.inf)
            if np.any(flight):
                h0 = height[flight]
                v0 = vel_y[flight]
                disc = v0**2 + 2.0 * g * h0
                ok = disc >= 0.0
                root = np.full(h0.shape, np.inf)
                safe = np.maximum(disc, 0.0)
                t_plus = (v0 + np.sqrt(safe)) / g
                t_minus = (v0 - np.sqrt(safe)) / g
                candidates = np.where(t_plus > 0.0, t_plus, np.inf)
                candidates = np.minimum(candidates, np.where(t_minus > 0.0, t_minus, np.inf))
                root = np.where(ok & (candidates > 0.0), candidates, np.inf)
                t_hit[flight] = root
            h = min(remaining, float(np.min(t_hit)) if np.min(t_hit) < np.inf else remaining)
            if not np.isfinite(h) or h <= 0.0:
                h = remaining
            # free flight for h
            positions = positions + velocities * h + np.array([0.0, -0.5 * g * h * h, 0.0])
            velocities = velocities + np.array([0.0, -g * h, 0.0])
            remaining -= h
            # resolve contacts (bodies whose height has collapsed onto the plane)
            height = positions[:, 1] - radii - self.ground
            hit = (height <= 1e-10) & (velocities[:, 1] < 0.0)
            if np.any(hit):
                before = 0.5 * float(np.sum(masses[hit] * np.sum(velocities[hit] ** 2, axis=1)))
                positions[hit, 1] = self.ground + radii[hit]
                vn = velocities[hit, 1]
                velocities[hit, 1] = -self.restitution * vn
                if self.friction > 0.0:
                    tangential = velocities[hit][:, [0, 2]].copy()
                    magnitude = np.linalg.norm(tangential, axis=1)
                    budget = self.friction * (1.0 + self.restitution) * np.abs(vn)
                    scale = np.where(
                        magnitude > 0.0, np.clip(budget / np.maximum(magnitude, 1e-30), 0.0, 1.0), 0.0
                    )
                    velocities[hit, 0] -= scale * tangential[:, 0]
                    velocities[hit, 2] -= scale * tangential[:, 1]
                after = 0.5 * float(np.sum(masses[hit] * np.sum(velocities[hit] ** 2, axis=1)))
                self.dissipated_j += max(before - after, 0.0)
                self.contact_events += int(np.count_nonzero(hit))
            if remaining <= 1e-18:
                break
        self.time += self.dt if dt is None else float(dt)
        for index, state in enumerate(self.states):
            state.position_m = positions[index].copy()
            state.velocity_m_s = velocities[index].copy()

    def run(self, duration_s: float, *, sample_every: int = 1) -> EnergyHistory:
        """Integrate ``duration_s`` and record the energy ledger."""
        if duration_s <= 0.0:
            raise ValueError("duration_s must be positive")
        history = EnergyHistory()
        self._record(history)
        steps = int(round(duration_s / self.dt))
        if steps < 1:
            raise ValueError("duration_s is shorter than one step")
        for index in range(1, steps + 1):
            self.step()
            if index % sample_every == 0:
                self._record(history)
        history.contact_events = self.contact_events
        return history

    def _record(self, history: EnergyHistory) -> None:
        ledger = self.energies()
        history.times.append(self.time)
        history.kinetic.append(ledger["kinetic_j"])
        history.potential.append(ledger["potential_j"])
        history.internal.append(ledger["internal_j"])
        history.total.append(ledger["total_j"])
        history.dissipated.append(ledger["dissipated_j"])
        masses, radii, positions, _v, _i = self._arrays()
        history.min_height.append(float(np.min(positions[:, 1] - radii - self.ground)))

    # -- convenience -----------------------------------------------------
    def set_velocities(self, velocities) -> None:
        """Overwrite the fragment translational velocities (handoff initial condition).

        The FEM→DEM handoff inherits a velocity field; the rigid integrator is
        otherwise initialised at rest.
        """
        values = np.asarray(velocities, dtype=float)
        if values.shape != (len(self.states), 3):
            raise ValueError(
                f"expected {len(self.states)} velocities of shape (3,), got {values.shape}"
            )
        for state, velocity in zip(self.states, values):
            state.velocity_m_s = np.array(velocity, dtype=float)

    def lift_to(self, release_height_m: float) -> None:
        """Translate every fragment so the lowest one starts at ``release_height_m``.

        The rigid-body release frame is the handoff frame shifted vertically; the
        internal energy and the masses are untouched, so the FEM→DEM energy chain
        is unchanged by the lift (a pure potential-energy offset).
        """
        if release_height_m <= 0.0:
            raise ValueError("release_height_m must be positive")
        masses, radii, positions, velocities, _i = self._arrays()
        lowest = float(np.min(positions[:, 1] - radii - self.ground))
        shift = release_height_m - lowest
        positions[:, 1] += shift
        for index, state in enumerate(self.states):
            state.position_m = positions[index].copy()
            state.velocity_m_s = velocities[index].copy()
        self._initial_shift = shift