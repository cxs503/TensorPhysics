"""Adjoint vs finite-difference sensitivity for SIMP compliance.

For ``C = F^T u`` with ``K(rho) u = F`` and the SIMP interpolation
``E_e = Emin + rho_e**p (E0 - Emin)`` the self-adjoint derivative is::

    dC/drho_e = -(p * rho_e**(p-1) * (E0 - Emin)) * u_e^T k0 u_e

which is exactly minus the element strain energy scaled by the interpolation
slope.  With a linear density filter ``rho_tilde = P rho`` the chain rule gives
``dC/drho = P^T dC/drho_tilde``.

The finite-difference references re-solve the FE problem with one design
variable perturbed, so agreement between the two is a genuine cross-check of
the adjoint implementation (not a tautology): the adjoint uses the *reaction*
of the whole displacement field, the FD uses the true perturbed solution.

Three FD variants are provided:

``forward``        ``(C(rho+drho) - C(rho)) / drho`` -- truncation ``O(drho)``;
``central``        ``(C(rho+drho) - C(rho-drho)) / (2 drho)`` -- ``O(drho^2)``
                   but a higher cancellation floor;
``complex_step``   ``Im C(rho + i drho) / drho`` -- no cancellation at all, so
                   it probes the analytic derivative to machine precision and
                   is used as an independent third opinion.
"""

from __future__ import annotations

import concurrent.futures
import os
from typing import Iterable, Sequence

import multiprocessing as mp

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .density_filter import DensityFilter, filter_sensitivity
from .fem2d import Q4Mesh, assemble_stiffness, compliance_and_displacements

__all__ = [
    "adjoint_sensitivity",
    "complex_step_sensitivity",
    "fd_sensitivity",
    "sensitivity_error_stats",
    "step_convergence_study",
]


def _flat(x: np.ndarray, mesh: Q4Mesh) -> np.ndarray:
    x = np.asarray(x, dtype=float).ravel()
    if x.size != mesh.n_elems:
        raise ValueError(f"expected {mesh.n_elems} element values, got {x.size}")
    return x


def _element_energies(mesh: Q4Mesh, u: np.ndarray) -> np.ndarray:
    """``u_e^T k0 u_e`` for every element (unit modulus)."""
    dofs = mesh.element_dofs()
    k = mesh.unit_stiffness()
    ue = u[dofs]  # (n_elems, 8)
    return np.einsum("ei,ij,ej->e", ue, k, ue)


def adjoint_sensitivity(
    mesh: Q4Mesh,
    rho: np.ndarray,
    F: np.ndarray,
    fixed: Iterable[int],
    *,
    penal: float = 3.0,
    E0: float = 1.0,
    Emin: float = 1e-9,
    dens_filter: DensityFilter | None = None,
) -> tuple[float | None, np.ndarray | None, np.ndarray | None]:
    """Analytic compliance and adjoint sensitivity ``dC/drho``.

    Returns ``(C, dC_drho_tilde, dC_drho)``; the first two are ``None`` when the
    reduced system is singular.  ``dC_drho`` already includes the filter chain
    rule when ``dens_filter`` is given.
    """
    rho_d = _flat(rho, mesh)
    rho_p = dens_filter.apply(rho_d) if dens_filter is not None else rho_d
    C, u = compliance_and_displacements(
        mesh, rho_p, F, fixed, penal=penal, E0=E0, Emin=Emin
    )
    if u is None:
        return None, None, None
    slope = penal * np.power(rho_p, penal - 1.0) * (E0 - Emin)
    dc_tilde = -slope * _element_energies(mesh, u)
    return float(C), dc_tilde, filter_sensitivity(dc_tilde, dens_filter)


_FD_STATE: dict = {}


def _fd_init(mesh, rho, F, fixed, penal, E0, Emin, dens_filter) -> None:
    _FD_STATE.clear()
    _FD_STATE.update(
        mesh=mesh, rho=np.asarray(rho, dtype=float), F=np.asarray(F, dtype=float),
        fixed=tuple(fixed), penal=float(penal), E0=float(E0), Emin=float(Emin),
        dens_filter=dens_filter,
    )


def _fd_batch(payload):
    """Worker: finite-difference sensitivities for a batch of elements.

    Relies on the module-level ``_FD_STATE`` (inherited through ``fork``) so the
    mesh / design are not pickled for every task.
    """
    elems, delta, scheme = payload
    st = _FD_STATE
    mesh = st["mesh"]
    rho = st["rho"]
    F = st["F"]
    fixed = st["fixed"]
    dens = st["dens_filter"]
    penal, E0, Emin = st["penal"], st["E0"], st["Emin"]

    def C_of(x: np.ndarray) -> float:
        rho_p = dens.apply(x) if dens is not None else x
        C, _ = compliance_and_displacements(
            mesh, rho_p, F, fixed, penal=penal, E0=E0, Emin=Emin
        )
        if C is None:
            raise ValueError("perturbed design has a singular stiffness matrix")
        return C

    c0 = C_of(rho) if scheme == "forward" else None
    out = []
    for e in elems:
        if scheme == "complex":
            x = rho.astype(complex)
            x[e] += 1j * delta
            rho_p = dens.matrix @ x if dens is not None else x
            out.append(_solve_complex(mesh, rho_p, F, fixed, penal, E0, Emin).imag / delta)
            continue
        xp = rho.copy()
        xp[e] += delta
        cp = C_of(xp)
        if scheme == "forward":
            out.append((cp - c0) / delta)
        else:
            xm = rho.copy()
            xm[e] -= delta
            out.append((cp - C_of(xm)) / (2.0 * delta))
    return out


def fd_sensitivity(
    mesh: Q4Mesh,
    rho: np.ndarray,
    F: np.ndarray,
    fixed: Iterable[int],
    *,
    delta: float = 1e-6,
    scheme: str = "central",
    penal: float = 3.0,
    E0: float = 1.0,
    Emin: float = 1e-9,
    dens_filter: DensityFilter | None = None,
    elements: Sequence[int] | None = None,
    workers: int | None = None,
) -> np.ndarray:
    """Finite-difference ``dC/drho`` on the *design* variables.

    With a density filter the perturbation is applied to ``rho`` and the
    filtered field is recomputed, i.e. the FD measures the same projected
    quantity as :func:`adjoint_sensitivity`.

    ``workers`` parallelises the element-wise re-solves (forked processes, one
    FE solve per element is independent); ``workers=1`` is the sequential path.
    """
    rho_d = _flat(rho, mesh)
    if delta <= 0.0:
        raise ValueError("delta must be positive")
    scheme = scheme.lower()
    if scheme not in {"forward", "central"}:
        raise ValueError("scheme must be 'forward' or 'central'")
    fixed = tuple(int(d) for d in fixed)
    idx = (
        np.arange(mesh.n_elems)
        if elements is None
        else np.asarray(list(elements), dtype=np.int64)
    )

    if workers is None:
        workers = min(16, os.cpu_count() or 1)
    workers = max(1, int(workers))

    if workers == 1 or idx.size <= 2:
        _fd_init(mesh, rho_d, F, fixed, penal, E0, Emin, dens_filter)
        return np.asarray(_fd_batch((idx, delta, scheme)), dtype=float)

    nchunks = min(workers * 2, idx.size)
    chunks = [c for c in np.array_split(idx, nchunks) if c.size]
    payloads = [(c, delta, scheme) for c in chunks]
    ctx = mp.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
        # initialise worker state through a task that does the work itself
        futures = [pool.submit(_fd_batch_with_state,
                               (mesh, rho_d, F, fixed, penal, E0, Emin, dens_filter),
                               pl) for pl in payloads]
        results = [f.result() for f in futures]
    flat = [v for chunk in results for v in chunk]
    return np.asarray(flat, dtype=float)


def _fd_batch_with_state(state, payload):
    """Worker entry point: install the FE state once, then run ``_fd_batch``."""
    if not _FD_STATE:
        mesh, rho, F, fixed, penal, E0, Emin, dens_filter = state
        _fd_init(mesh, rho, F, fixed, penal, E0, Emin, dens_filter)
    return _fd_batch(payload)


def _solve_complex(mesh: Q4Mesh, rho_p: np.ndarray, F: np.ndarray,
                   fixed: Iterable[int], penal: float, E0: float, Emin: float) -> complex:
    E = Emin + np.power(rho_p.astype(complex), penal) * (E0 - Emin)
    dofs = mesh.element_dofs()
    k = mesh.unit_stiffness()
    ke = (E[:, None, None] * k[None, :, :]).ravel()
    rows = np.repeat(dofs, 8, axis=1).ravel()
    cols = np.tile(dofs, (1, 8)).ravel()
    K = sp.coo_matrix((ke, (rows, cols)), shape=(mesh.n_dof, mesh.n_dof)).tocsr()
    fixed = np.asarray(sorted(set(int(d) for d in fixed)), dtype=np.int64)
    free = np.setdiff1d(np.arange(mesh.n_dof), fixed)
    ur = spla.spsolve(K[free][:, free].tocsc(), np.asarray(F, dtype=complex)[free])
    u = np.zeros(mesh.n_dof, dtype=complex)
    u[free] = ur
    return complex(np.asarray(F, dtype=complex) @ u)


def complex_step_sensitivity(
    mesh: Q4Mesh,
    rho: np.ndarray,
    F: np.ndarray,
    fixed: Iterable[int],
    *,
    delta: float = 1e-30,
    penal: float = 3.0,
    E0: float = 1.0,
    Emin: float = 1e-9,
    dens_filter: DensityFilter | None = None,
    elements: Sequence[int] | None = None,
    workers: int | None = None,
) -> np.ndarray:
    """Complex-step derivative of the compliance (no subtractive cancellation)."""
    rho_d = _flat(rho, mesh)
    if delta <= 0.0:
        raise ValueError("delta must be positive")
    fixed = tuple(int(d) for d in fixed)
    idx = (
        np.arange(mesh.n_elems)
        if elements is None
        else np.asarray(list(elements), dtype=np.int64)
    )
    if workers is None:
        workers = min(16, os.cpu_count() or 1)
    workers = max(1, int(workers))
    if workers == 1 or idx.size <= 2:
        _fd_init(mesh, rho_d, F, fixed, penal, E0, Emin, dens_filter)
        return np.asarray(_fd_batch((idx, delta, "complex")), dtype=float)
    nchunks = min(workers * 2, idx.size)
    chunks = [c for c in np.array_split(idx, nchunks) if c.size]
    payloads = [(c, delta, "complex") for c in chunks]
    ctx = mp.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
        futures = [pool.submit(_fd_batch_with_state,
                               (mesh, rho_d, F, fixed, penal, E0, Emin, dens_filter),
                               pl) for pl in payloads]
        results = [f.result() for f in futures]
    flat = [v for chunk in results for v in chunk]
    return np.asarray(flat, dtype=float)


def sensitivity_error_stats(analytic: np.ndarray, numeric: np.ndarray) -> dict[str, float]:
    """Absolute / relative error statistics between two sensitivity fields.

    Void elements have ``rho = 0`` and therefore a vanishing SIMP slope, so
    their sensitivity is exactly 0 and a pointwise relative error there is
    meaningless (0/0).  Relative statistics are therefore restricted to
    elements with ``|analytic| > 1e-6 * max|analytic|`` and are accompanied by
    the scale-free ``l2_rel`` norm.
    """
    a = np.asarray(analytic, dtype=float).ravel()
    n = np.asarray(numeric, dtype=float).ravel()
    if a.shape != n.shape:
        raise ValueError("analytic and numeric arrays must have the same shape")
    abs_err = np.abs(a - n)
    ref = float(np.abs(a).max()) if a.size else 0.0
    mask = np.abs(a) > max(1e-6 * ref, 1e-30)
    max_rel = float((abs_err[mask] / np.abs(a[mask])).max()) if mask.any() else float("nan")
    mean_rel = float((abs_err[mask] / np.abs(a[mask])).mean()) if mask.any() else float("nan")
    denom = float(np.linalg.norm(a))
    l2_rel = float(np.linalg.norm(abs_err) / denom) if denom > 0.0 else float("nan")
    # relative error on the "structurally active" elements only (>=1% of the
    # peak sensitivity): the only place where a pointwise relative error is a
    # meaningful number.
    top = np.abs(a) >= 0.01 * ref if ref > 0.0 else np.zeros_like(a, dtype=bool)
    max_rel_top1 = float((abs_err[top] / np.abs(a[top])).max()) if top.any() else float("nan")
    mean_rel_top1 = float((abs_err[top] / np.abs(a[top])).mean()) if top.any() else float("nan")
    return {
        "n": float(a.size),
        "n_rel": float(mask.sum()),
        "max_abs": float(abs_err.max()),
        "mean_abs": float(abs_err.mean()),
        "max_rel": max_rel,
        "mean_rel": mean_rel,
        "max_rel_top1pct": max_rel_top1,
        "mean_rel_top1pct": mean_rel_top1,
        "n_top1pct": float(top.sum()),
        "l2_rel": l2_rel,
        "worst_index": int(np.argmax(abs_err)),
    }


def step_convergence_study(
    mesh: Q4Mesh,
    rho: np.ndarray,
    F: np.ndarray,
    fixed: Iterable[int],
    deltas: Sequence[float] = (1e-4, 1e-5, 1e-6, 1e-7, 1e-8),
    *,
    scheme: str = "forward",
    penal: float = 3.0,
    E0: float = 1.0,
    Emin: float = 1e-9,
    dens_filter: DensityFilter | None = None,
    elements: Sequence[int] | None = None,
    workers: int | None = None,
) -> list[dict[str, float]]:
    """FD step-refinement study against the analytic adjoint sensitivity.

    Returns one row per ``delta`` with ``max_abs``/``mean_abs`` error, the
    log-log convergence rate to the previous row, and a ``monotone`` flag
    describing the whole sequence (``monotone`` is filled on all rows).
    """
    _, _, dc_analytic = adjoint_sensitivity(
        mesh, rho, F, fixed, penal=penal, E0=E0, Emin=Emin, dens_filter=dens_filter
    )
    if dc_analytic is None:
        raise ValueError("base design has a singular stiffness matrix")
    idx = np.arange(mesh.n_elems) if elements is None else np.asarray(list(elements), dtype=np.int64)
    ref = dc_analytic[idx]

    rows: list[dict[str, float]] = []
    prev: dict[str, float] | None = None
    for d in deltas:
        if scheme == "complex":
            fd = complex_step_sensitivity(
                mesh, rho, F, fixed, delta=d, penal=penal, E0=E0, Emin=Emin,
                dens_filter=dens_filter, elements=idx, workers=workers,
            )
        else:
            fd = fd_sensitivity(
                mesh, rho, F, fixed, delta=d, scheme=scheme, penal=penal,
                E0=E0, Emin=Emin, dens_filter=dens_filter, elements=idx, workers=workers,
            )
        stats = sensitivity_error_stats(ref, fd)
        row = {"delta": float(d), **stats}
        if prev is not None and prev["max_abs"] > 0.0 and stats["max_abs"] > 0.0:
            row["rate_max"] = float(
                np.log(prev["max_abs"] / stats["max_abs"]) / np.log(prev["delta"] / d)
            )
        else:
            row["rate_max"] = float("nan")
        rows.append(row)
        prev = row

    monotone = all(
        rows[i + 1]["max_abs"] <= rows[i]["max_abs"] for i in range(len(rows) - 1)
    )
    for row in rows:
        row["monotone"] = float(monotone)
    return rows