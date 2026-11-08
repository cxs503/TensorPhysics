"""Interface stabilisation for the added-mass-unstable (light body) coupling.

Physics
-------
For a partitioned (staggered) fluid-structure iteration the fluid hands the
structure a force that depends on the interface acceleration through the added
mass ``A``.  The naive Gauss-Seidel fixed point

    a^{k+1} = M_s^{-1} ( f_ext - A a^k )

has the iteration matrix ``-M_s^{-1} A``, so it *diverges* as soon as the
structural mass is smaller than the added mass (``rho_s/rho_f < 1`` for a light
floating body) - the classical added-mass instability.  Two stabilisations are
implemented here and both must rescue the iteration:

* ``aitken``     dynamic under-relaxation (Aitken's Delta-squared, vector form),
* ``iqn-ils``    interface quasi-Newton with an inverse least-squares Jacobian
                 built from the stored residual/solution differences
                 (Degroote et al. 2009).

This module deliberately knows nothing about waves or beams: it is the generic
interface solver, exercised here on the chain's (heave, pitch) interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["InterfaceProblem", "CouplingResult", "solve_interface", "spectral_radius"]


@dataclass(frozen=True)
class InterfaceProblem:
    """Linear two-way interface: fluid added mass + structural mass.

    Parameters
    ----------
    structural_mass : np.ndarray
        ``(n, n)`` structural mass matrix on the interface DOF.
    added_mass : np.ndarray
        ``(n, n)`` fluid added-mass matrix on the interface DOF.
    external_force : np.ndarray
        ``(n,)`` all other generalised forces (excitation, restoring, ...).
    """

    structural_mass: np.ndarray
    added_mass: np.ndarray
    external_force: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "structural_mass", np.atleast_2d(np.asarray(self.structural_mass, dtype=float)))
        object.__setattr__(self, "added_mass", np.atleast_2d(np.asarray(self.added_mass, dtype=float)))
        object.__setattr__(self, "external_force", np.atleast_1d(np.asarray(self.external_force, dtype=float)))

    @property
    def n_dof(self) -> int:
        return int(self.structural_mass.shape[0])

    def fluid_force(self, acceleration: np.ndarray) -> np.ndarray:
        """Fluid reaction to the interface acceleration (added-mass operator)."""
        return -self.added_mass @ np.asarray(acceleration, dtype=float)

    def structure_acceleration(self, acceleration: np.ndarray) -> np.ndarray:
        """Structure acceleration implied by a given interface acceleration."""
        return np.linalg.solve(
            self.structural_mass, self.external_force + self.fluid_force(acceleration)
        )

    def residual(self, acceleration: np.ndarray) -> np.ndarray:
        """Fixed-point residual ``M_s^{-1}(f_ext - A a) - a``."""
        a = np.asarray(acceleration, dtype=float)
        return self.structure_acceleration(a) - a


@dataclass
class CouplingResult:
    """Outcome of one interface (sub-)iteration count."""

    converged: bool
    divergent: bool
    iterations: int
    residual_history: list = field(default_factory=list)
    relaxation_history: list = field(default_factory=list)
    acceleration: np.ndarray | None = None
    scheme: str = "none"

    @property
    def final_residual(self) -> float:
        return float(self.residual_history[-1]) if self.residual_history else float("nan")


def spectral_radius(problem: InterfaceProblem) -> float:
    """Spectral radius of the naive fixed-point iteration matrix ``-M_s^{-1} A``."""
    M = np.linalg.solve(problem.structural_mass, problem.added_mass)
    return float(np.max(np.abs(np.linalg.eigvals(M))))


def solve_interface(
    problem: InterfaceProblem,
    scheme: str = "none",
    tol: float = 1.0e-6,
    max_iterations: int = 60,
    initial: np.ndarray | None = None,
    divergence_factor: float = 1.0e4,
    aitken_omega0: float = 0.5,
    iqn_bootstrap_omega: float = 0.5,
    iqn_memory: int = 20,
) -> CouplingResult:
    """Iterate the interface to convergence, with an optional stabilisation.

    Parameters
    ----------
    scheme : {"none", "aitken", "iqn-ils"}
        ``"none"`` is the plain Gauss-Seidel fixed point and is the negative
        control: it diverges whenever :func:`spectral_radius` exceeds 1.
    divergence_factor : float
        Residual growth over the initial residual that is declared divergence.
    """
    a = np.zeros(problem.n_dof) if initial is None else np.asarray(initial, dtype=float).copy()
    r = problem.residual(a)
    r0 = max(float(np.linalg.norm(r)), 1.0e-30)
    hist = [float(np.linalg.norm(r))]
    relax: list[float] = []

    # secant storage (IQN-ILS) and previous iterate (Aitken)
    V = np.zeros((problem.n_dof, 0))
    W = np.zeros((problem.n_dof, 0))
    prev_r = r.copy()
    prev_a = a.copy()
    omega = float(aitken_omega0)

    if float(np.linalg.norm(r)) <= tol:
        return CouplingResult(True, False, 0, hist, relax, a, scheme)

    for k in range(1, int(max_iterations) + 1):
        # --- secant information *before* the step (the differences of the last
        # two iterates; zero on the first iteration, which is why it is skipped)
        dr = r - prev_r
        da_prev = a - prev_a
        prev_r = r.copy()
        prev_a = a.copy()

        if scheme == "none":
            da = r
            relax.append(1.0)
        elif scheme == "aitken":
            if k > 1:
                denom = float(dr @ dr)
                if denom > 1e-300:
                    omega = float(np.clip(-omega * float(prev_r_prev @ dr) / denom, -4.0, 4.0))
            relax.append(omega)
            da = omega * r
            prev_r_prev = r.copy()
        elif scheme == "iqn-ils":
            if np.linalg.norm(dr) > 0.0:
                V = np.column_stack([V, dr])
                W = np.column_stack([W, da_prev])
                if V.shape[1] > iqn_memory:
                    V = V[:, -iqn_memory:]
                    W = W[:, -iqn_memory:]
            if V.shape[1] == 0:
                da = float(iqn_bootstrap_omega) * r
                relax.append(float(iqn_bootstrap_omega))
            else:
                c, *_ = np.linalg.lstsq(V, r, rcond=None)
                remainder = r - V @ c
                da = -(W @ c)
                if np.linalg.norm(remainder) > tol * r0:
                    da = da - float(iqn_bootstrap_omega) * remainder
                relax.append(1.0)
        else:
            raise ValueError(f"unknown scheme {scheme!r}")

        a = a + da
        r = problem.residual(a)
        rn = float(np.linalg.norm(r))
        hist.append(rn)
        if not np.isfinite(rn) or rn > divergence_factor * r0 or not np.all(np.isfinite(a)):
            return CouplingResult(False, True, k, hist, relax, a, scheme)
        if rn <= tol:
            return CouplingResult(True, False, k, hist, relax, a, scheme)

    return CouplingResult(False, False, int(max_iterations), hist, relax, a, scheme)


def divergence_detection_rate(
    problems: list[InterfaceProblem],
    scheme: str = "none",
    **kwargs,
) -> dict:
    """Fraction of unstable interfaces that get flagged as divergent.

    The negative control of the chain requires this to be exactly 1.0: silence
    about a diverging coupling is the failure mode the gate must exclude.
    """
    flagged = 0
    detail = []
    for i, p in enumerate(problems):
        res = solve_interface(p, scheme=scheme, **kwargs)
        unstable = spectral_radius(p) > 1.0
        flagged += int(bool(res.divergent) or (unstable and not res.converged))
        detail.append(
            {
                "case": i,
                "spectral_radius": spectral_radius(p),
                "divergent": bool(res.divergent),
                "converged": bool(res.converged),
                "iterations": int(res.iterations),
                "final_residual": res.final_residual,
            }
        )
    return {
        "scheme": scheme,
        "n_cases": len(problems),
        "detected": flagged,
        "detection_rate": float(flagged / max(len(problems), 1)),
        "detail": detail,
    }