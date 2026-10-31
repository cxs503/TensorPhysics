"""Co-rotational 2-node beam element for finite-rotation hull analysis.

Formulation
-----------
The element is the classical co-rotational (Crisfield-type) planar beam: an initially
straight 2-node element with three degrees of freedom per node, ``(u_y, u_z, theta)``,
whose deformation is split into

* an axial stretch ``ubar = l - l0`` measured along the *current* chord, and
* two local end rotations ``theta_bar_a = (theta_a - theta_a0) - (beta - beta0)``,
  i.e. the nodal rotations measured relative to the rotation of the chord.

``beta`` is the current chord angle and ``beta0`` its reference value.  Because the
rigid-body rotation ``beta - beta0`` is subtracted, the element is objective: any finite
rigid rotation of the element produces exactly zero strain energy.  The element strain
energy is

    U_e = 1/2 * (EA / l0) * ubar^2 + (2 EI / l0) * (tb1^2 + tb1*tb2 + tb2^2)

whose local gradient reproduces the standard linear-beam end moments
``M1 = (2EI/l0)(2 tb1 + tb2)``, ``M2 = (2EI/l0)(tb1 + 2 tb2)`` and axial force
``N = EA * ubar / l0``.  The second term is *not* a small-rotation approximation of the
geometry: it acts on rotations measured in the co-rotated frame, so the mesh may rotate
by any angle (the 15 deg roll of the hull, or a 60 deg elastic bend) while the strain
measure stays small and objective -- the standard large-rotation / small-strain regime
of ship structures.

Why the energy functional is central
------------------------------------
Internal forces are evaluated as the **exact** gradient ``f = dU/dq`` and the tangent
stiffness as the **exact** Hessian ``K = d2U/dq2`` by algorithmic differentiation
(``torch.autograd``, float64).  The consequence is that

* Newton's method uses a tangent that is consistent with the force by construction, and
* an explicit/leapfrog (symplectic) integrator becomes energy consistent, so total
  energy conservation can be used as a machine-checkable quality gate of the
  geometric-nonlinear update (see :mod:`tensorphysics.structure.hull_roll`).

Mass is lumped (translational mass + rotary inertia) so that the same ledger can be used
for static roll cases and for the conservative large-rotation transient.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

__all__ = ["Section", "CorotationalBeamModel", "coRotationalElementEnergy"]

_DTYPE = torch.float64


@dataclass(frozen=True)
class Section:
    """Beam/shell-strip section properties (SI units)."""

    area: float
    inertia: float
    young_modulus: float
    fibre_distance: float
    density: float = 7850.0

    def __post_init__(self) -> None:
        if self.area <= 0.0 or self.inertia <= 0.0:
            raise ValueError("area and inertia must be positive")
        if self.fibre_distance <= 0.0:
            raise ValueError("fibre_distance must be positive")

    @property
    def ea(self) -> float:
        return self.area * self.young_modulus

    @property
    def ei(self) -> float:
        return self.inertia * self.young_modulus

    @property
    def section_modulus(self) -> float:
        """Elastic section modulus ``W = I / c``."""
        return self.inertia / self.fibre_distance

    @property
    def mass_per_length(self) -> float:
        return self.area * self.density


def coRotationalElementEnergy(ubar: float, tb1: float, tb2: float, l0: float,
                              ea: float, ei: float) -> float:
    """Element strain energy (scalar version, used for documentation / tests)."""
    return 0.5 * ea / l0 * ubar ** 2 + (2.0 * ei / l0) * (
        tb1 ** 2 + tb1 * tb2 + tb2 ** 2
    )


@dataclass
class CorotationalBeamModel:
    """Assembly of co-rotational beam elements with an explicit energy functional."""

    reference_nodes: np.ndarray
    elements: np.ndarray
    section: Section
    node_mass: np.ndarray | None = None
    node_rotary_inertia: np.ndarray | None = None
    fixed_dofs: tuple[int, ...] = ()
    dof_springs: tuple[tuple[int, float], ...] = ()
    _l0: np.ndarray = field(init=False, repr=False)
    _beta0: np.ndarray = field(init=False, repr=False)
    _mass: np.ndarray = field(init=False, repr=False)
    _inertia: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        nodes = np.asarray(self.reference_nodes, dtype=float)
        elements = np.asarray(self.elements, dtype=int)
        if nodes.ndim != 2 or nodes.shape[1] != 2:
            raise ValueError("reference_nodes must be (n_nodes, 2)")
        if elements.ndim != 2 or elements.shape[1] != 2:
            raise ValueError("elements must be (n_elements, 2)")
        if elements.min() < 0 or elements.max() >= len(nodes):
            raise ValueError("element connectivity out of range")
        self.reference_nodes = nodes.copy()
        self.elements = elements.copy()

        d = nodes[elements[:, 1]] - nodes[elements[:, 0]]
        self._l0 = np.linalg.norm(d, axis=1)
        if np.any(self._l0 <= 0.0):
            raise ValueError("degenerate element with zero reference length")
        self._beta0 = np.arctan2(d[:, 1], d[:, 0])
        self._elem_i = torch.as_tensor(elements[:, 0], dtype=torch.long)
        self._elem_j = torch.as_tensor(elements[:, 1], dtype=torch.long)
        self._l0_t = torch.as_tensor(self._l0, dtype=_DTYPE)
        self._beta0_t = torch.as_tensor(self._beta0, dtype=_DTYPE)

        n = len(nodes)
        if self.node_mass is None:
            self._mass = np.zeros(n)
            for e, (i, j) in enumerate(self.elements):
                half = 0.5 * self.section.mass_per_length * self._l0[e]
                self._mass[i] += half
                self._mass[j] += half
        else:
            self._mass = np.asarray(self.node_mass, dtype=float).copy()
            if self._mass.shape != (n,):
                raise ValueError("node_mass must have one entry per node")
        if self.node_rotary_inertia is None:
            self._inertia = np.zeros(n)
            rot_density = self.section.density * self.section.inertia
            for e, (i, j) in enumerate(self.elements):
                half = 0.5 * rot_density * self._l0[e]
                self._inertia[i] += half
                self._inertia[j] += half
            # guarantee a positive-definite (diagonal) rotational mass
            self._inertia = np.maximum(self._inertia, 1e-9 * max(self._mass.mean(), 1.0))
        else:
            self._inertia = np.asarray(self.node_rotary_inertia, dtype=float).copy()
            if self._inertia.shape != (n,):
                raise ValueError("node_rotary_inertia must have one entry per node")

    # ------------------------------------------------------------------ helpers
    @property
    def n_nodes(self) -> int:
        return len(self.reference_nodes)

    @property
    def n_dof(self) -> int:
        return 3 * self.n_nodes

    def pack(self, x: np.ndarray, theta: np.ndarray) -> torch.Tensor:
        """Flatten to the node-major DOF vector ``[x_y, x_z, theta]_i``.

        The node-major layout is shared by the element assembly, the external load
        vector and ``fixed_dofs`` (node ``i`` owns DOFs ``3i, 3i+1, 3i+2``), so a load
        applied to ``3*i + comp`` acts on component ``comp`` of node ``i``.
        """
        x_t = torch.as_tensor(np.asarray(x, dtype=float), dtype=_DTYPE)
        th_t = torch.as_tensor(np.asarray(theta, dtype=float), dtype=_DTYPE)
        return torch.cat([x_t, th_t.reshape(-1, 1)], dim=1).reshape(-1)

    def unpack(self, q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        n = self.n_nodes
        q3 = q.reshape(n, 3)
        return q3[:, :2], q3[:, 2]

    def _element_energy(self, e: int, x: torch.Tensor, theta: torch.Tensor
                        ) -> torch.Tensor:
        i, j = self.elements[e]
        d = x[j] - x[i]
        l0 = self._l0[e]
        l = torch.sqrt(torch.sum(d * d) + 1e-30)
        beta = torch.atan2(d[1], d[0])
        dbeta = torch.atan2(torch.sin(beta - self._beta0[e]),
                            torch.cos(beta - self._beta0[e]))
        ubar = l - l0
        tb1 = theta[i] - dbeta
        tb2 = theta[j] - dbeta
        ea, ei = self.section.ea, self.section.ei
        return 0.5 * ea / l0 * ubar ** 2 + (2.0 * ei / l0) * (
            tb1 ** 2 + tb1 * tb2 + tb2 ** 2)

    def element_energy_from_flat(self, q: torch.Tensor) -> torch.Tensor:
        """Strain energy stored in the beam elements (vectorised over elements)."""
        x, theta = self.unpack(q)
        i, j = self._elem_i, self._elem_j
        d = x[j] - x[i]
        l0 = self._l0_t
        load = torch.sqrt(torch.sum(d * d, dim=1) + 1e-30)
        beta = torch.atan2(d[:, 1], d[:, 0])
        dbeta = torch.atan2(torch.sin(beta - self._beta0_t),
                            torch.cos(beta - self._beta0_t))
        ubar = load - l0
        # local (co-rotated) rotations, wrapped to (-pi, pi] so that arbitrarily large
        # nodal rotations remain objective (a 180 deg rigid roll must give tb == 0)
        tb1 = torch.atan2(torch.sin(theta[i] - dbeta), torch.cos(theta[i] - dbeta))
        tb2 = torch.atan2(torch.sin(theta[j] - dbeta), torch.cos(theta[j] - dbeta))
        ea = self.section.ea
        ei = self.section.ei
        energy_axial = 0.5 * ea / l0 * ubar ** 2
        energy_bend = (2.0 * ei / l0) * (tb1 ** 2 + tb1 * tb2 + tb2 ** 2)
        return torch.sum(energy_axial + energy_bend)

    def restoring_potential(self, theta) -> float:
        """Potential energy of the generalised (roll) springs, ``1/2 k theta^2``."""
        th = torch.as_tensor(np.asarray(theta, dtype=float), dtype=_DTYPE)
        total = 0.0
        for dof, k in self.dof_springs:
            node, comp = divmod(int(dof), 3)
            if comp != 2:
                raise ValueError("dof_springs may only act on rotational DOFs")
            total += 0.5 * float(k) * float(th[node]) ** 2
        return float(total)

    def energy_from_flat(self, q: torch.Tensor) -> torch.Tensor:
        """Total potential energy: element strain energy + generalised springs."""
        total = self.element_energy_from_flat(q)
        if self.dof_springs:
            _, theta = self.unpack(q)
            for dof, k in self.dof_springs:
                node, comp = divmod(int(dof), 3)
                if comp != 2:
                    raise ValueError("dof_springs may only act on rotational DOFs")
                total = total + 0.5 * float(k) * theta[node] ** 2
        return total

    # ---------------------------------------------------------------- mechanics
    def strain_energy(self, x, theta=None) -> float:
        """Elastic strain energy of the beam elements only."""
        theta = np.zeros(self.n_nodes) if theta is None else theta
        q = self.pack(x, theta)
        return float(self.element_energy_from_flat(q))

    def internal_force(self, x, theta=None) -> tuple[np.ndarray, np.ndarray]:
        """Exact gradient of the strain energy: ``f_int = dU/dq``."""
        theta = np.zeros(self.n_nodes) if theta is None else theta
        q = self.pack(x, theta).requires_grad_(True)
        energy = self.energy_from_flat(q)
        grad = torch.autograd.grad(energy, q)[0]
        f_x, f_theta = self.unpack(grad)
        return f_x.detach().numpy().copy(), f_theta.detach().numpy().copy()

    def tangent_stiffness(self, x, theta=None) -> np.ndarray:
        """Exact Hessian of the strain energy: ``K = d2U/dq2``."""
        theta = np.zeros(self.n_nodes) if theta is None else theta
        q = self.pack(x, theta)
        hess = torch.autograd.functional.hessian(self.energy_from_flat, q)
        return hess.detach().numpy().copy()

    def element_section_forces(self, x, theta=None) -> np.ndarray:
        """Local section resultants ``(N, M1, M2)`` of every element.

        ``N`` is the axial force (positive = tension), ``M1`` / ``M2`` are the end
        moments in the co-rotated frame, both in N and N*m.
        """
        theta = np.zeros(self.n_nodes) if theta is None else theta
        x = np.asarray(x, dtype=float)
        theta = np.asarray(theta, dtype=float)
        ea, ei = self.section.ea, self.section.ei
        out = np.zeros((len(self.elements), 3))
        for e, (i, j) in enumerate(self.elements):
            d = x[j] - x[i]
            l0 = self._l0[e]
            l = float(np.linalg.norm(d))
            beta = float(np.arctan2(d[1], d[0]))
            dbeta = float(np.arctan2(np.sin(beta - self._beta0[e]),
                                     np.cos(beta - self._beta0[e])))
            ubar = l - l0
            tb1 = float(np.arctan2(np.sin(theta[i] - dbeta), np.cos(theta[i] - dbeta)))
            tb2 = float(np.arctan2(np.sin(theta[j] - dbeta), np.cos(theta[j] - dbeta)))
            n_axial = ea * ubar / l0
            m1 = (2.0 * ei / l0) * (2.0 * tb1 + tb2)
            m2 = (2.0 * ei / l0) * (tb1 + 2.0 * tb2)
            out[e] = (n_axial, m1, m2)
        return out

    def max_beam_stress(self, x, theta=None) -> dict:
        """Maximum direct stress ``|N|/A + max(|M1|,|M2|)/W`` over all elements."""
        forces = self.element_section_forces(x, theta)
        sec = self.section
        sigma_axial = np.abs(forces[:, 0]) / sec.area
        sigma_bend = np.abs(forces[:, 1:3]).max(axis=1) / sec.section_modulus
        sigma = sigma_axial + sigma_bend
        e_max = int(np.argmax(sigma))
        return {
            "sigma_max_pa": float(sigma[e_max]),
            "element": e_max,
            "axial_pa": float(sigma_axial[e_max]),
            "bending_pa": float(sigma_bend[e_max]),
            "section_forces": forces,
        }

    # ----------------------------------------------------------------- solvers
    def solve_static(self, external_force: np.ndarray, x0=None, theta0=None,
                     max_iter: int = 60, tol: float = 1e-8,
                     load_steps: int = 1) -> dict:
        """Newton-Raphson solution of ``f_int(q) - f_ext = 0``.

        ``external_force`` is a flat vector of length ``3 * n_nodes`` in world
        coordinates (it is *not* rotated during the solve: a world-fixed ice load is a
        dead load applied at the current configuration of the nodes).  Fixed degrees of
        freedom (``fixed_dofs``) are eliminated; prescribed values enter through
        ``x0`` / ``theta0``.
        """
        f_ext = np.asarray(external_force, dtype=float).reshape(-1)
        if f_ext.shape != (self.n_dof,):
            raise ValueError("external_force must have length 3*n_nodes")
        x = np.asarray(self.reference_nodes if x0 is None else x0, dtype=float).copy()
        theta = np.zeros(self.n_nodes) if theta0 is None else np.asarray(
            theta0, dtype=float).copy()

        free = np.ones(self.n_dof, dtype=bool)
        free[list(self.fixed_dofs)] = False
        q = self.pack(x, theta)
        history = []
        steps = max(1, int(load_steps))
        for step in range(1, steps + 1):
            scale = step / steps
            res_norm = np.inf
            for _ in range(max_iter):
                q = q.detach().requires_grad_(True)
                energy = self.energy_from_flat(q)
                grad = torch.autograd.grad(energy, q)[0]
                residual = (grad - scale * torch.as_tensor(f_ext, dtype=_DTYPE))
                r_free = residual.detach().numpy()[free]
                res_norm = float(np.linalg.norm(r_free))
                history.append(res_norm)
                if res_norm <= tol * max(1.0, float(np.linalg.norm(scale * f_ext))):
                    break
                hess = torch.autograd.functional.hessian(self.energy_from_flat,
                                                         q.detach())
                k_ff = hess.detach().numpy()[np.ix_(free, free)]
                dq = np.linalg.solve(k_ff, -r_free)
                q_free = q.detach().numpy()[free] + dq
                q_next = q.detach().numpy().copy()
                q_next[free] = q_free
                q = torch.as_tensor(q_next, dtype=_DTYPE)
        x_end, theta_end = self.unpack(q.detach())
        return {
            "x": x_end.numpy().copy(),
            "theta": theta_end.numpy().copy(),
            "iterations": len(history),
            "residual": res_norm,
            "q": q.detach().numpy().copy(),
        }

    def transient(self, v0: np.ndarray, w0: np.ndarray, dt: float, n_steps: int,
                  x0=None, theta0=None, gravity: float | None = None,
                  sample_every: int = 1) -> dict:
        """Explicit leapfrog (velocity-Verlet) integration of the large-rotation motion.

        Internal forces are the exact energy gradient, therefore the scheme is
        symplectic for a conservative system and total energy is bounded (no secular
        drift).  ``gravity`` (m/s^2, acting along -z) enters through the conservative
        potential ``V = sum m g z`` which is included in the reported ledger.
        """
        x = np.asarray(self.reference_nodes if x0 is None else x0, dtype=float).copy()
        theta = np.zeros(self.n_nodes) if theta0 is None else np.asarray(
            theta0, dtype=float).copy()
        v = np.asarray(v0, dtype=float).copy()
        w = np.asarray(w0, dtype=float).copy()
        if v.shape != (self.n_nodes, 2) or w.shape != (self.n_nodes,):
            raise ValueError("v0 must be (n_nodes, 2) and w0 (n_nodes,)")

        m = self._mass
        inertia = self._inertia
        fixed_mask = np.zeros(self.n_dof, dtype=bool)
        fixed_mask[list(self.fixed_dofs)] = True
        v = v.copy()
        w = w.copy()
        # zero the velocity of constrained DOFs only (a node may be partly constrained)
        v[fixed_mask[0::3], 0] = 0.0
        v[fixed_mask[1::3], 1] = 0.0
        w[fixed_mask[2::3]] = 0.0

        def accelerations(xx, th):
            f_x, f_th = self.internal_force(xx, th)
            acc = np.zeros(self.n_dof)
            acc[0::3] = -f_x[:, 0] / m
            acc[1::3] = -f_x[:, 1] / m
            acc[2::3] = -f_th / inertia
            if gravity is not None:
                acc[1::3] += -gravity  # world-fixed weight
            acc[fixed_mask] = 0.0
            return acc

        def ledger(xx, th, vv, ww):
            ke = 0.5 * float(np.sum(m[:, None] * vv ** 2) + np.sum(inertia * ww ** 2))
            u = self.strain_energy(xx, th) + self.restoring_potential(th)
            pot = 0.0 if gravity is None else float(np.sum(m * gravity * xx[:, 1]))
            return ke, u, pot

        ke0, u0, pot0 = ledger(x, theta, v, w)
        e0 = ke0 + u0 + pot0
        times = [0.0]
        kin = [ke0]
        strain = [u0]
        pot_hist = [pot0]
        total = [e0]
        def chord_roll_deg(xx):
            """Finite-rotation roll angle of the section: inclination of the chord."""
            chord = xx[-1] - xx[0]
            return float(np.degrees(np.arctan2(-chord[0], chord[1])))

        roll = [chord_roll_deg(x)]

        acc = accelerations(x, theta)
        for step in range(1, n_steps + 1):
            v[:, 0] = v[:, 0] + 0.5 * dt * acc[0::3]
            v[:, 1] = v[:, 1] + 0.5 * dt * acc[1::3]
            w = w + 0.5 * dt * acc[2::3]
            x = x + dt * v
            theta = theta + dt * w
            acc = accelerations(x, theta)
            v[:, 0] = v[:, 0] + 0.5 * dt * acc[0::3]
            v[:, 1] = v[:, 1] + 0.5 * dt * acc[1::3]
            w = w + 0.5 * dt * acc[2::3]
            if step % sample_every == 0 or step == n_steps:
                ke, u, pot = ledger(x, theta, v, w)
                times.append(step * dt)
                kin.append(ke)
                strain.append(u)
                pot_hist.append(pot)
                total.append(ke + u + pot)
                roll.append(chord_roll_deg(x))
        total_arr = np.asarray(total)
        drift = float(np.max(np.abs(total_arr - e0)) / max(abs(e0), 1e-30))
        return {
            "time": np.asarray(times),
            "kinetic": np.asarray(kin),
            "strain": np.asarray(strain),
            "potential": np.asarray(pot_hist),
            "total": total_arr,
            "energy0": float(e0),
            "energy_drift_ratio": drift,
            "roll_amplitude_deg": float(np.max(np.abs(np.asarray(roll)))),
            "roll_angle_deg_trace": np.asarray(roll),
            "x": x,
            "theta": theta,
            "v": v,
            "w": w,
        }