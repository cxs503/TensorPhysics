"""Regularisation chains for SIMP topology optimisation.

The package already ships Sigmund's *linear hat* density filter
(:mod:`.density_filter`).  This module adds the two regularisers that the
literature actually uses for manufacturable, mesh-independent designs, and
composes them into an auditable chain ``design density -> filter -> projection``:

``PdeFilter``
    Helmholtz-type implicit filter (Kawamoto et al. 2011, Lazarov & Sigmund
    2011).  The filtered field solves

    .. math::   -r^2 \\nabla^2 \\tilde\\rho + \\tilde\\rho = \\rho
                \\quad\\text{with}\\quad \\partial_n\\tilde\\rho = 0 ,

    discretised with the same bilinear Q1 basis as the mechanical problem
    (natural Neumann boundary conditions, no ghost layer needed).  The filter
    radius is set to ``r = r_min / (2*sqrt(3))`` so that the PDE kernel has the
    same second moment as the hat filter of radius ``r_min``.  The advantage
    over the hat filter is that the *design* variables stay element-wise while
    the filtering operator is a well-conditioned sparse solve -- the same code
    path scales to 3-D and to unstructured meshes.

``HeavisideProjection``
    Smooth Heaviside projection of the filtered field (Guest et al. 2004),
    ``beta`` controlling the sharpness and ``eta`` the threshold.  As
    ``beta -> inf`` the field becomes black-and-white, which is what removes
    the intermediate ("grey") material a pure linear filter leaves behind.

Both are differentiable, and both expose :meth:`apply` / :meth:`apply_transpose`
so that the adjoint sensitivity of the compliance factorises exactly::

    dC/drho = (d rho_bar / d rho)^T  dC/drho_bar

:class:`FilterChain` composes the two (filter first, then projection) and
:func:`run_simp_chain` drives the optimality-criteria optimisation through a
chain, including a ``beta``-continuation schedule.
"""

from __future__ import annotations

import math
import multiprocessing as mp
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .density_filter import DensityFilter, checkerboard_index
from .fem2d import Q4Mesh, compliance_and_displacements
from .problems import SIMPProblem, oc_update

__all__ = [
    "FilterChain",
    "HeavisideProjection",
    "IdentityFilter",
    "PdeFilter",
    "adjoint_sensitivity_chain",
    "fd_sensitivity_chain",
    "make_filter_chain",
    "pde_radius_from_hat_radius",
    "run_simp_chain",
]

_GAUSS = 1.0 / math.sqrt(3.0)
_GP = ((-_GAUSS, -_GAUSS), (_GAUSS, -_GAUSS), (_GAUSS, _GAUSS), (-_GAUSS, _GAUSS))


# ---------------------------------------------------------------------------
# filter / projection operators
# ---------------------------------------------------------------------------
class IdentityFilter:
    """No regularisation (the pathological baseline, kept for symmetry)."""

    kind = "none"

    def __init__(self, n_elems: int) -> None:
        self.n_elems = int(n_elems)

    def apply(self, rho: np.ndarray) -> np.ndarray:
        flat = np.asarray(rho, dtype=float).ravel()
        if flat.size != self.n_elems:
            raise ValueError(f"expected {self.n_elems} densities, got {flat.size}")
        return flat

    def apply_transpose(self, g: np.ndarray) -> np.ndarray:
        return np.asarray(g, dtype=float).ravel().copy()


class HatFilter:
    """Adapter exposing :class:`DensityFilter` through the chain protocol."""

    kind = "density"

    def __init__(self, nelx: int, nely: int, radius: float) -> None:
        if radius <= 0.0:
            raise ValueError("hat filter radius must be positive")
        self._filter = DensityFilter(nelx, nely, radius)
        self.n_elems = self._filter.n_elems
        self.radius = float(radius)

    def apply(self, rho: np.ndarray) -> np.ndarray:
        return self._filter.apply(rho)

    def apply_transpose(self, g: np.ndarray) -> np.ndarray:
        return self._filter.matrix.T @ np.asarray(g, dtype=float).ravel()

    @property
    def matrix(self) -> sp.csr_matrix:
        return self._filter.matrix


def pde_radius_from_hat_radius(r_min: float) -> float:
    """PDE filter radius with the same second moment as a hat filter of ``r_min``.

    ``r = r_min / (2*sqrt(3))`` (Lazarov & Sigmund 2011).  The relation is
    checked numerically by :func:`effective_radius` and in the test suite.
    """
    if r_min <= 0.0:
        raise ValueError("filter radius must be positive")
    return float(r_min) / (2.0 * math.sqrt(3.0))


def _q1_matrices(hx: float, hy: float) -> tuple[np.ndarray, np.ndarray]:
    """Bilinear mass and Laplacian element matrices on an ``hx x hy`` cell."""
    M = np.zeros((4, 4), dtype=float)
    K = np.zeros((4, 4), dtype=float)
    for xi, eta in _GP:
        N = np.array(
            [
                0.25 * (1.0 - xi) * (1.0 - eta),
                0.25 * (1.0 + xi) * (1.0 - eta),
                0.25 * (1.0 + xi) * (1.0 + eta),
                0.25 * (1.0 - xi) * (1.0 + eta),
            ]
        )
        dN_dxi = 0.25 * np.array(
            [
                [-(1.0 - eta), -(1.0 - xi)],
                [(1.0 - eta), -(1.0 + xi)],
                [(1.0 + eta), (1.0 + xi)],
                [-(1.0 + eta), (1.0 - xi)],
            ]
        )
        dN = dN_dxi / np.array([0.5 * hx, 0.5 * hy])  # d/dx, d/dy (J diagonal)
        w = 0.25 * hx * hy
        M += np.outer(N, N) * w
        K += dN @ dN.T * w
    return M, K


class PdeFilter:
    """Helmholtz (implicit) density filter on an ``(nely, nelx)`` element grid.

    ``rho_tilde = E_avg A^{-1} B rho`` with ``A = r^2 K + M`` (nodal Q1
    operators), ``B`` the element->node mass operator and ``E_avg`` the
    node->element averaging operator, so that the public interface stays
    element-wise like every other filter in this package.
    """

    kind = "pde"

    def __init__(self, nelx: int, nely: int, radius: float,
                 hx: float = 1.0, hy: float = 1.0) -> None:
        if radius <= 0.0:
            raise ValueError("PDE filter radius must be positive")
        self.nelx = int(nelx)
        self.nely = int(nely)
        self.radius = float(radius)
        self.hx = float(hx)
        self.hy = float(hy)
        self.n_elems = self.nelx * self.nely
        self._n_nodes = (self.nelx + 1) * (self.nely + 1)
        self._build()

    # -- assembly -----------------------------------------------------------
    def _build(self) -> None:
        nelx, nely = self.nelx, self.nely
        nhx = nelx + 1
        n1 = np.arange(self.n_elems)
        # node ids of element e (bottom-left ccw), element id = ey*nelx + ex
        ey = n1 // nelx
        ex = n1 % nelx
        nid = np.column_stack([
            ey * nhx + ex,
            ey * nhx + ex + 1,
            (ey + 1) * nhx + ex + 1,
            (ey + 1) * nhx + ex,
        ])
        Me, Ke = _q1_matrices(self.hx, self.hy)

        rows = np.repeat(nid, 4, axis=1).ravel()
        cols = np.tile(nid, (1, 4)).ravel()
        M = sp.coo_matrix(
            (np.tile(Me.ravel(), self.n_elems), (rows, cols)),
            shape=(self._n_nodes, self._n_nodes),
        ).tocsr()
        K = sp.coo_matrix(
            (np.tile(Ke.ravel(), self.n_elems), (rows, cols)),
            shape=(self._n_nodes, self._n_nodes),
        ).tocsr()
        A = (self.radius ** 2) * K + M
        self._A = A.tocsr()
        self._M = M
        # element (piecewise constant density) -> nodal load operator
        w = 0.25 * self.hx * self.hy
        self._B = sp.coo_matrix(
            (np.full(self.n_elems * 4, w),
             (nid.ravel(), np.repeat(n1, 4))),
            shape=(self._n_nodes, self.n_elems),
        ).tocsr()
        self._E = sp.coo_matrix(
            (np.full(self.n_elems * 4, 0.25),
             (np.repeat(n1, 4), nid.ravel())),
            shape=(self.n_elems, self._n_nodes),
        ).tocsr()
        self._lu = spla.splu(self._A.tocsc())

    # -- public API ---------------------------------------------------------
    def apply(self, rho: np.ndarray) -> np.ndarray:
        flat = np.asarray(rho, dtype=float).ravel()
        if flat.size != self.n_elems:
            raise ValueError(f"expected {self.n_elems} densities, got {flat.size}")
        nodal = self._lu.solve(self._B @ flat)
        return self._E @ nodal

    def apply_transpose(self, g: np.ndarray) -> np.ndarray:
        """``F^T g`` where ``F = E A^{-1} B`` (``A`` symmetric)."""
        g = np.asarray(g, dtype=float).ravel()
        if g.size != self.n_elems:
            raise ValueError(f"expected {self.n_elems} gradients, got {g.size}")
        nodal = self._lu.solve(self._E.T @ g)
        return self._B.T @ nodal

    def nodal_field(self, rho: np.ndarray) -> np.ndarray:
        """Filtered *nodal* field (useful for plotting / post-processing)."""
        flat = np.asarray(rho, dtype=float).ravel()
        return self._lu.solve(self._B @ flat)

    def effective_radius(self, at: int | None = None) -> float:
        """Second-moment radius of the discrete kernel (element units).

        Computed from the response to a unit impulse at element ``at``
        (default: grid centre), ``sqrt(sum w_i d_i^2 / sum w_i)`` with
        ``d_i`` the element-centre distance.
        """
        if at is None:
            at = (self.nely // 2) * self.nelx + self.nelx // 2
        e = np.zeros(self.n_elems)
        e[at] = 1.0
        w = self.apply(e)
        ys, xs = np.meshgrid(np.arange(self.nely), np.arange(self.nelx), indexing="ij")
        ey, ex = divmod(at, self.nelx)
        x, y = xs.ravel() - ex, ys.ravel() - ey
        tot = w.sum()
        return float(np.sqrt((w * (x * x + y * y)).sum() / tot))


class HeavisideProjection:
    """Smooth Heaviside projection (Guest, Prévost & Belytschko 2004).

    ``rho_bar = (tanh(beta*eta) + tanh(beta*(rho - eta)))
                / (tanh(beta*eta) + tanh(beta*(1 - eta)))``

    with derivative ``d rho_bar / d rho = beta (1 - tanh^2(beta (rho-eta)))
    / (tanh(beta*eta) + tanh(beta*(1-eta)))``.
    """

    kind = "heaviside"

    def __init__(self, beta: float = 8.0, eta: float = 0.5) -> None:
        if beta <= 0.0:
            raise ValueError("beta must be positive")
        if not 0.0 < eta < 1.0:
            raise ValueError("eta must lie in (0, 1)")
        self.beta = float(beta)
        self.eta = float(eta)
        self._denom = math.tanh(self.beta * self.eta) + math.tanh(
            self.beta * (1.0 - self.eta)
        )

    def apply(self, rho: np.ndarray) -> np.ndarray:
        r = np.asarray(rho, dtype=float).ravel()
        return (
            math.tanh(self.beta * self.eta) + np.tanh(self.beta * (r - self.eta))
        ) / self._denom

    def derivative(self, rho: np.ndarray) -> np.ndarray:
        r = np.asarray(rho, dtype=float).ravel()
        t = np.tanh(self.beta * (r - self.eta))
        return self.beta * (1.0 - t * t) / self._denom

    def apply_transpose(self, g: np.ndarray) -> np.ndarray:
        g = np.asarray(g, dtype=float).ravel()
        # the projection is point-wise, so the transpose needs the state; the
        # chain supplies it through ``set_state`` before calling.
        if self._state is None:
            raise RuntimeError("HeavisideProjection.apply_transpose needs a state "
                               "(use FilterChain, which records it)")
        return self.derivative(self._state) * g

    _state: np.ndarray | None = None


class FilterChain:
    """``design density -> filter -> projection`` with exact adjoints.

    ``apply`` is forward, ``apply_transpose`` is the reverse-mode chain rule.
    The intermediate state is cached so the point-wise projection derivative is
    evaluated at the same field the forward pass used (a single optimisation
    step is: ``rho_bar = chain.apply(rho)``; ``dc = chain.apply_transpose(dc_bar)``).
    """

    def __init__(self, stages: Sequence[Any], *, label: str | None = None) -> None:
        if not stages:
            raise ValueError("a filter chain needs at least one stage")
        self.stages = list(stages)
        self.label = label or "+".join(getattr(s, "kind", "stage") for s in self.stages)
        n = getattr(self.stages[0], "n_elems", None)
        self.n_elems = n
        self._cache: list[np.ndarray] = []

    # convenience accessors -------------------------------------------------
    @property
    def filter_stage(self):
        return self.stages[0]

    @property
    def projection_stage(self) -> HeavisideProjection | None:
        return self.stages[1] if len(self.stages) > 1 else None

    @property
    def matrix(self) -> sp.csr_matrix:
        """Linear part of the chain (only meaningful without a projection)."""
        if self.projection_stage is not None:
            raise ValueError("chain with a projection is non-linear: no matrix")
        return self.stages[0].matrix

    # forward / reverse -----------------------------------------------------
    def apply(self, rho: np.ndarray) -> np.ndarray:
        x = np.asarray(rho, dtype=float).ravel()
        self._cache = [x]
        for st in self.stages:
            x = st.apply(x)
            self._cache.append(x)
        if isinstance(self.stages[-1], HeavisideProjection):
            self.stages[-1]._state = self._cache[-2]
        return x

    def apply_transpose(self, g: np.ndarray) -> np.ndarray:
        x = np.asarray(g, dtype=float).ravel()
        for st in reversed(self.stages):
            x = st.apply_transpose(x)
        return x

    def volume_gradient(self) -> np.ndarray:
        """``d(mean rho_tilde)/d rho`` -- the volume-constraint gradient."""
        n = self.n_elems if self.n_elems is not None else self.stages[0].n_elems
        ones = np.ones(n)
        # constraint is imposed on the filtered (pre-projection) field
        return self.stages[0].apply_transpose(ones) / n


def make_filter_chain(
    kind: str,
    nelx: int,
    nely: int,
    *,
    radius: float = 1.5,
    beta: float | None = 8.0,
    eta: float = 0.5,
    hx: float = 1.0,
    hy: float = 1.0,
) -> FilterChain:
    """Build a chain by name.

    ``kind`` is one of ``none``, ``density``, ``pde``, ``density+heaviside``,
    ``pde+heaviside``.  The hat/PDE radius is expressed in *element lengths*
    (``radius`` must exceed ``1`` -- the element diagonal/edge, i.e.
    ``r_min > h`` -- otherwise the regularisation cannot suppress the
    element-wise checkerboard mode; that requirement is enforced by
    :func:`check_radius_above_mesh`).
    """
    kind = kind.lower()
    if kind == "none":
        return FilterChain([IdentityFilter(nelx * nely)], label="none")
    base = {
        "density": lambda: HatFilter(nelx, nely, radius),
        "pde": lambda: PdeFilter(nelx, nely, pde_radius_from_hat_radius(radius), hx, hy),
    }
    if kind in base:
        return FilterChain([base[kind]()], label=kind)
    if kind.endswith("+heaviside"):
        stem = kind[: -len("+heaviside")]
        if stem not in base:
            raise ValueError(f"unknown filter chain kind {kind!r}")
        return FilterChain([base[stem](), HeavisideProjection(beta or 8.0, eta)],
                           label=kind)
    raise ValueError(f"unknown filter chain kind {kind!r}")


def check_radius_above_mesh(radius: float, h: float = 1.0) -> None:
    """Enforce the well-posedness requirement ``r_min > h``."""
    if radius <= h:
        raise ValueError(
            f"filter radius r_min={radius} must exceed the element size h={h} "
            "to suppress the element-wise checkerboard mode"
        )


# ---------------------------------------------------------------------------
# sensitivities through a chain
# ---------------------------------------------------------------------------
def _element_energies(mesh: Q4Mesh, u: np.ndarray) -> np.ndarray:
    dofs = mesh.element_dofs()
    k = mesh.unit_stiffness()
    ue = u[dofs]
    return np.einsum("ei,ij,ej->e", ue, k, ue)


def adjoint_sensitivity_chain(
    mesh: Q4Mesh,
    rho: np.ndarray,
    F: np.ndarray,
    fixed: Sequence[int],
    chain: FilterChain,
    *,
    penal: float = 3.0,
    E0: float = 1.0,
    Emin: float = 1e-9,
) -> tuple[float | None, np.ndarray | None, np.ndarray | None]:
    """Compliance and ``dC/drho`` through ``chain`` (filter + projection)."""
    rho_d = np.asarray(rho, dtype=float).ravel()
    rho_p = chain.apply(rho_d)
    C, u = compliance_and_displacements(
        mesh, rho_p, F, fixed, penal=penal, E0=E0, Emin=Emin
    )
    if u is None:
        return None, None, None
    slope = penal * np.power(np.maximum(rho_p, 0.0), penal - 1.0) * (E0 - Emin)
    dc_bar = -slope * _element_energies(mesh, u)
    return float(C), dc_bar, chain.apply_transpose(dc_bar)


_FD_CHAIN: dict[str, Any] = {}


def _fd_worker(payload):
    elems, delta = payload
    st = _FD_CHAIN
    mesh, rho, F, fixed, chain = (
        st["mesh"], st["rho"], st["F"], st["fixed"], st["chain"],
    )
    penal, E0, Emin = st["penal"], st["E0"], st["Emin"]
    out = []
    for e in elems:
        vals = []
        for sign in (+1.0, -1.0):
            x = rho.copy()
            x[e] += sign * delta
            rho_p = chain.apply(x)
            c, _ = compliance_and_displacements(
                mesh, rho_p, F, fixed, penal=penal, E0=E0, Emin=Emin
            )
            if c is None:
                raise ValueError("perturbed design is singular")
            vals.append(c)
        out.append((vals[0] - vals[1]) / (2.0 * delta))
    return out


def fd_sensitivity_chain(
    mesh: Q4Mesh,
    rho: np.ndarray,
    F: np.ndarray,
    fixed: Sequence[int],
    chain: FilterChain,
    *,
    delta: float = 1e-6,
    penal: float = 3.0,
    E0: float = 1.0,
    Emin: float = 1e-9,
    elements: Sequence[int] | None = None,
    workers: int = 1,
) -> np.ndarray:
    """Central-difference ``dC/drho`` through the chain.

    Unlike the linear-only path in :mod:`.sensitivity` this works with the
    non-linear Heaviside projection as well (the perturbed design is pushed
    through the full chain).  ``workers > 1`` forks: the factorised PDE filter
    (a SuperLU object) cannot be pickled, so ``fork`` is required.
    """
    rho_d = np.asarray(rho, dtype=float).ravel()
    idx = (np.arange(mesh.n_elems) if elements is None
           else np.asarray(list(elements), dtype=np.int64))
    _FD_CHAIN.clear()
    _FD_CHAIN.update(mesh=mesh, rho=rho_d, F=np.asarray(F, float),
                     fixed=tuple(int(d) for d in fixed), chain=chain,
                     penal=float(penal), E0=float(E0), Emin=float(Emin))
    if workers <= 1 or idx.size < 4:
        return np.asarray(_fd_worker((idx, delta)), dtype=float)
    chunks = [c for c in np.array_split(idx, min(workers * 2, idx.size)) if c.size]
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=workers) as pool:
        res = pool.map(_fd_worker, [(c, delta) for c in chunks])
    return np.asarray([v for chunk in res for v in chunk], dtype=float)


# ---------------------------------------------------------------------------
# optimisation driver
# ---------------------------------------------------------------------------
@dataclass
class ChainResult:
    problem: str
    nelx: int
    nely: int
    volfrac: float
    kind: str
    radius: float | None
    beta: float | None
    iterations: int
    converged: bool
    compliance: float
    checkerboard_index: float
    grey_fraction: float
    volume_fraction: float
    rho: np.ndarray = field(repr=False)
    rho_physical: np.ndarray = field(repr=False)
    history: list[dict[str, float]] = field(default_factory=list, repr=False)

    def as_dict(self) -> dict[str, Any]:
        return {
            "problem": self.problem,
            "nelx": self.nelx,
            "nely": self.nely,
            "aspect_ratio": self.nelx / self.nely,
            "volfrac": self.volfrac,
            "kind": self.kind,
            "radius": self.radius,
            "beta": self.beta,
            "iterations": self.iterations,
            "converged": self.converged,
            "compliance": self.compliance,
            "checkerboard_index": self.checkerboard_index,
            "grey_fraction": self.grey_fraction,
            "volume_fraction": self.volume_fraction,
        }


def _beta_schedule(beta_max: float, n_steps: int = 5) -> list[float]:
    """Doubling continuation ``1, 2, 4, ... beta_max``."""
    schedule = [1.0]
    while schedule[-1] < beta_max:
        schedule.append(min(2.0 * schedule[-1], float(beta_max)))
    if len(schedule) > n_steps:
        schedule = schedule[-n_steps:]
    return schedule


def run_simp_chain(
    problem: SIMPProblem,
    *,
    kind: str = "pde+heaviside",
    radius: float = 1.5,
    beta: float | None = 8.0,
    eta: float = 0.5,
    penal: float = 3.0,
    iters: int = 200,
    move: float = 0.2,
    tol: float = 1e-3,
    E0: float = 1.0,
    Emin: float = 1e-9,
    rho0: np.ndarray | None = None,
    allow_radius_below_mesh: bool = False,
) -> ChainResult:
    """SIMP compliance minimisation through a regularisation chain.

    With a Heaviside projection the continuation runs ``beta = 1, 2, 4, ...``
    up to ``beta``; the volume constraint is imposed on the *filtered* field
    (``mean(rho_tilde) = volfrac``), which keeps the constraint differentiable
    even when the projection is almost a step function.
    """
    mesh = problem.mesh
    n = mesh.n_elems
    rho = (np.full(n, problem.volfrac) if rho0 is None
           else np.asarray(rho0, dtype=float).ravel().copy())

    if kind.lower() == "none":
        beta_schedule = [None]
    elif "heaviside" in kind.lower():
        if not allow_radius_below_mesh:
            check_radius_above_mesh(radius)
        beta_schedule = _beta_schedule(beta or 8.0)
    else:
        if not allow_radius_below_mesh:
            check_radius_above_mesh(radius)
        beta_schedule = [None]

    per_stage = max(8, int(math.ceil(iters / len(beta_schedule))))
    history: list[dict[str, float]] = []
    converged = False
    total = 0
    C = None
    chain = None
    for stage_idx, stage_beta in enumerate(beta_schedule):
        last_stage = stage_idx == len(beta_schedule) - 1
        chain = make_filter_chain(kind, mesh.nelx, mesh.nely, radius=radius,
                                  beta=stage_beta, eta=eta, hx=mesh.hx, hy=mesh.hy)
        dv = chain.volume_gradient()
        for _ in range(per_stage):
            total += 1
            C, _, dc = adjoint_sensitivity_chain(
                mesh, rho, problem.F, problem.fixed_dofs, chain,
                penal=penal, E0=E0, Emin=Emin,
            )
            if dc is None:
                raise RuntimeError("stiffness matrix became singular during optimisation")
            rho_new = oc_update(rho, dc, dv, problem.volfrac, move=move)
            change = float(np.max(np.abs(rho_new - rho)))
            rho = rho_new
            rho_p = chain.apply(rho)
            history.append({
                "iteration": float(total),
                "beta": float(stage_beta if stage_beta is not None else 0.0),
                "compliance": float(C),
                "change": change,
                "volume_fraction": float(np.mean(chain.stages[0].apply(rho))),
                "checkerboard_index": checkerboard_index(rho_p.reshape(mesh.shape)),
            })
            # the eta-continuation must run to completion: stopping at a low
            # beta would freeze a grey design and hide the projection's effect
            if change < tol and last_stage:
                converged = True
                break
        if converged:
            break

    assert chain is not None
    C, _, _ = adjoint_sensitivity_chain(
        mesh, rho, problem.F, problem.fixed_dofs, chain,
        penal=penal, E0=E0, Emin=Emin,
    )
    rho_p = chain.apply(rho)
    grey = float(np.mean((rho_p > 0.1) & (rho_p < 0.9)))
    return ChainResult(
        problem=problem.name,
        nelx=mesh.nelx,
        nely=mesh.nely,
        volfrac=problem.volfrac,
        kind=kind,
        radius=radius if kind.lower() != "none" else None,
        beta=beta if "heaviside" in kind.lower() else None,
        iterations=total,
        converged=converged,
        compliance=float(C),
        checkerboard_index=checkerboard_index(rho_p.reshape(mesh.shape)),
        grey_fraction=grey,
        volume_fraction=float(np.mean(chain.stages[0].apply(rho))),
        rho=rho,
        rho_physical=rho_p,
        history=history,
    )