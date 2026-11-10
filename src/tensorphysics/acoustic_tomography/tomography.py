"""Acoustic travel-time tomography: ART / SIRT algebraic reconstruction.

Forward model
-------------
Travel time along ray ``i`` is the path integral of slowness ``s = 1/c``

    t_i = int_{Gamma_i} s dl  ~=  sum_j a_ij s_j ,      a_ij = int_{Gamma_i cap cell_j} dl

The unknown vector ``s`` lives on the nodes of a coarse vertical-plane grid, and
the path integral is discretised with a bilinear (node) quadrature: every
resampled path segment contributes its length to the four surrounding nodes with
the bilinear weights.  Node-based unknowns keep the retrieved field continuous,
so it can be sampled at arbitrary CTD positions without stair-casing.

Reconstruction
--------------
``art``   -- Kaczmarz / Algebraic Reconstruction Technique (row-by-row update)
``sirt``  -- Simultaneous Iterative Reconstruction Technique with optional
             Laplacian smoothing (Tikhonov-type regularisation of the
             underdetermined null space)

Both solve for the *slowness anomaly* relative to a declared background model,
which is the standard ocean-acoustic-tomography formulation: the background
(climatology / a reference CTD cast) removes the range-independent part, and the
data resolve the horizontal structure.

Uncertainty
-----------
``monte_carlo_uncertainty`` re-inverts an ensemble of travel times perturbed by
the measurement noise and returns the ensemble standard deviation of the
retrieved field, i.e. an empirical posterior width sigma_c(r, z).  No analytic
covariance approximation is used, so the estimate also captures the bias of the
regulariser.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Grid helpers
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class NodeGrid:
    """Rectangular node grid ``(nr, nz)`` in the (range, depth) plane."""

    r_nodes: np.ndarray
    z_nodes: np.ndarray

    def __post_init__(self):
        object.__setattr__(self, "r_nodes", np.asarray(self.r_nodes, dtype=float))
        object.__setattr__(self, "z_nodes", np.asarray(self.z_nodes, dtype=float))

    @property
    def shape(self):
        return (self.r_nodes.size, self.z_nodes.size)

    @property
    def size(self):
        return self.r_nodes.size * self.z_nodes.size

    def mesh(self):
        return np.meshgrid(self.r_nodes, self.z_nodes, indexing="ij")

    def flatten(self, values):
        return np.asarray(values, dtype=float).reshape(self.size)

    def unflatten(self, vector):
        return np.asarray(vector, dtype=float).reshape(self.shape)


def bilinear_quadrature(r_s, z_s, nodes: NodeGrid):
    """Bilinear interpolation weights of sample points on a node grid.

    Returns ``(index, weight)`` arrays of shape ``(4, n_samples)`` where
    ``index`` are flat node indices and ``weight`` the bilinear weights.
    """
    r_nodes, z_nodes = nodes.r_nodes, nodes.z_nodes
    r = np.clip(np.asarray(r_s, dtype=float), r_nodes[0], r_nodes[-1])
    z = np.clip(np.asarray(z_s, dtype=float), z_nodes[0], z_nodes[-1])
    ir = np.clip(np.searchsorted(r_nodes, r, side="right") - 1, 0, r_nodes.size - 2)
    iz = np.clip(np.searchsorted(z_nodes, z, side="right") - 1, 0, z_nodes.size - 2)
    tr = (r - r_nodes[ir]) / (r_nodes[ir + 1] - r_nodes[ir])
    tz = (z - z_nodes[iz]) / (z_nodes[iz + 1] - z_nodes[iz])
    nz = z_nodes.size
    idx = np.stack(
        [
            ir * nz + iz,
            (ir + 1) * nz + iz,
            ir * nz + iz + 1,
            (ir + 1) * nz + iz + 1,
        ]
    )
    w = np.stack(
        [
            (1 - tr) * (1 - tz),
            tr * (1 - tz),
            (1 - tr) * tz,
            tr * tz,
        ]
    )
    return idx, w


def interpolate_nodes(values, nodes: NodeGrid, r_s, z_s):
    """Sample a node-based field at arbitrary points."""
    idx, w = bilinear_quadrature(r_s, z_s, nodes)
    flat = np.asarray(values, dtype=float).reshape(-1)
    return np.sum(w * flat[idx], axis=0)


def path_matrix(paths, nodes: NodeGrid, ds: float = 10.0):
    """Bilinear path-integral matrix ``A`` for a list of polyline paths.

    Parameters
    ----------
    paths : sequence of (N, 2) arrays
        Rays as ``[r, z]`` polylines in metres.
    nodes : NodeGrid
    ds : float
        Resampling step along the path [m].

    Returns
    -------
    A : ndarray, shape (n_paths, nodes.size)
        Path length per node [m].
    lengths : ndarray, shape (n_paths,)
        Geometric path length [m].
    """
    n_paths = len(paths)
    A = np.zeros((n_paths, nodes.size), dtype=float)
    lengths = np.zeros(n_paths, dtype=float)
    for k, path in enumerate(paths):
        path = np.asarray(path, dtype=float)
        if path.ndim != 2 or path.shape[0] < 2:
            raise ValueError("each path must be an (N, 2) array with N >= 2")
        seg = np.hypot(np.diff(path[:, 0]), np.diff(path[:, 1]))
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        total = cum[-1]
        lengths[k] = total
        if total <= 0.0:
            continue
        n_s = max(int(np.ceil(total / ds)), 1)
        s_target = np.linspace(0.0, total, n_s + 1)
        r_s = np.interp(s_target, cum, path[:, 0])
        z_s = np.interp(s_target, cum, path[:, 1])
        r_mid = 0.5 * (r_s[:-1] + r_s[1:])
        z_mid = 0.5 * (z_s[:-1] + z_s[1:])
        ds_vec = np.diff(s_target)
        idx, w = bilinear_quadrature(r_mid, z_mid, nodes)
        contrib = w * ds_vec[None, :]
        flat = np.bincount(
            idx.reshape(-1), weights=contrib.reshape(-1), minlength=nodes.size
        )
        A[k] = flat
    return A, lengths


# ---------------------------------------------------------------------------
# Regularisation
# ---------------------------------------------------------------------------
def laplacian(vector, nodes: NodeGrid, physical: bool = False):
    """Five-point Laplacian of a node field (Neumann at the boundary).

    With ``physical=True`` the two directions are divided by the squared node
    spacing (``dr^2``, ``dz^2``), so the operator approximates
    ``d2/dr2 + d2/dz2`` in SI units and the explicit smoothing step used by
    :func:`sirt` obeys the usual diffusion stability limit (diffusion number
    <= 0.2 for this five-point stencil).  The default (``physical=False``)
    returns the raw index-space stencil, which is what a regularisation matrix
    expects.
    """
    g = nodes.unflatten(vector)
    out = np.zeros_like(g)
    nr, nz = g.shape
    if physical:
        hr = np.gradient(nodes.r_nodes)
        hz = np.gradient(nodes.z_nodes)
        wr = 1.0 / np.maximum(hr, 1e-12) ** 2 if nr > 2 else np.zeros(nr)
        wz = 1.0 / np.maximum(hz, 1e-12) ** 2 if nz > 2 else np.zeros(nz)
    else:
        wr = np.ones(nr)
        wz = np.ones(nz)
    if nr > 2:
        out[1:-1, :] += (g[2:, :] + g[:-2, :] - 2.0 * g[1:-1, :]) * wr[1:-1, None]
        out[0, :] += (g[1, :] - g[0, :]) * wr[0]
        out[-1, :] += (g[-2, :] - g[-1, :]) * wr[-1]
    if nz > 2:
        out[:, 1:-1] += (g[:, 2:] + g[:, :-2] - 2.0 * g[:, 1:-1]) * wz[None, 1:-1]
        out[:, 0] += (g[:, 1] - g[:, 0]) * wz[0]
        out[:, -1] += (g[:, -2] - g[:, -1]) * wz[-1]
    return out.reshape(-1)


# ---------------------------------------------------------------------------
# Iterative solvers
# ---------------------------------------------------------------------------
def sirt(A, t, n_iter: int = 200, relaxation: float = 1.0, smooth: float = 0.0,
         nodes: NodeGrid | None = None, initial=None,
         target_residual: float | None = None, check_every: int = 25):
    """Simultaneous Iterative Reconstruction Technique.

    ``s <- s + relaxation * C A^T R (t - A s)`` with ``R = 1/rowsum(A)`` and
    ``C = 1/colsum(A)``, followed by an optional physical Laplacian smoothing
    step of diffusion number ``smooth`` (stability limit 0.2).

    SIRT is semi-convergent: the data residual falls monotonically while the
    field error first decreases and then grows as the timing noise is fitted.
    Passing ``target_residual`` applies the *discrepancy principle* -- the
    iteration stops at the first check (every ``check_every`` iterations) whose
    residual RMS is at or below that value, which for a known noise level
    ``sigma_t`` is the standard regularisation choice and needs no tuning
    against the truth.  Returns a dict with the solution, the residual history
    and the number of iterations actually performed.
    """
    A = np.asarray(A, dtype=float)
    t = np.asarray(t, dtype=float)
    n_paths, n_nodes = A.shape
    row = A.sum(axis=1)
    col = A.sum(axis=0)
    valid_row = row > 0
    R = np.zeros(n_paths)
    R[valid_row] = 1.0 / row[valid_row]
    C = np.zeros(n_nodes)
    nz_col = col > 0
    C[nz_col] = 1.0 / col[nz_col]
    s = np.zeros(n_nodes) if initial is None else np.array(initial, dtype=float)
    history = []
    n_done = 0
    for k in range(int(n_iter)):
        resid = t - A @ s
        history.append(float(np.sqrt(np.mean(resid[valid_row] ** 2)) if np.any(valid_row) else np.nan))
        s = s + relaxation * C * (A.T @ (R * resid))
        if smooth > 0.0 and nodes is not None:
            s = s + smooth * laplacian(s, nodes, physical=True)
        n_done = k + 1
        if target_residual is not None and (k + 1) % max(int(check_every), 1) == 0:
            if np.sqrt(np.mean((t - A @ s)[valid_row] ** 2)) <= target_residual:
                break
    resid = t - A @ s
    history.append(float(np.sqrt(np.mean(resid[valid_row] ** 2))))
    return {
        "slowness": s,
        "residual_history": np.asarray(history),
        "residual": resid,
        "n_iter": int(n_done),
    }


def art(A, t, n_iter: int = 20, relaxation: float = 1.0, initial=None):
    """Kaczmarz / Algebraic Reconstruction Technique (row-by-row update)."""
    A = np.asarray(A, dtype=float)
    t = np.asarray(t, dtype=float)
    n_paths, n_nodes = A.shape
    s = np.zeros(n_nodes) if initial is None else np.array(initial, dtype=float)
    norms = np.einsum("ij,ij->i", A, A)
    history = []
    for _ in range(int(n_iter)):
        for i in range(n_paths):
            if norms[i] <= 0.0:
                continue
            s = s + relaxation * A[i] * (t[i] - A[i] @ s) / norms[i]
        resid = t - A @ s
        history.append(float(np.sqrt(np.mean(resid**2))))
    return {"slowness": s, "residual_history": np.asarray(history), "residual": t - A @ s}


# ---------------------------------------------------------------------------
# Tomographic inversion driver
# ---------------------------------------------------------------------------
def invert_sound_speed(
    paths,
    travel_times,
    nodes: NodeGrid,
    background_slowness,
    n_iter: int = 300,
    relaxation: float = 1.0,
    smooth: float = 0.02,
    method: str = "sirt",
    ds: float = 10.0,
    target_residual: float | None = None,
    check_every: int = 25,
):
    """Invert travel times for a node-based sound-speed field.

    Parameters
    ----------
    paths : sequence of (N, 2) arrays
        Ray geometries used by the inversion (straight or traced polylines).
    travel_times : array_like
        Observed travel times [s].  An empty sequence returns an all-NaN field
        (campaign negative control d1: no acoustic data -> no inversion).
    nodes : NodeGrid
    background_slowness : ndarray
        Prior slowness at the nodes [s/m]; the inversion updates the anomaly.

    Returns
    -------
    dict with ``sound_speed`` (node field [m/s]), ``slowness``, ``residual``,
    ``rms_residual_s``, ``path_matrix``, ``path_lengths``, ``n_observations``.
    """
    tt = np.asarray(travel_times, dtype=float).reshape(-1)
    bg = np.asarray(background_slowness, dtype=float).reshape(-1)
    nan_field = np.full(nodes.size, np.nan)
    if tt.size == 0 or not np.any(np.isfinite(tt)) or len(paths) == 0:
        return {
            "sound_speed": nan_field.reshape(nodes.shape),
            "slowness": nan_field.reshape(nodes.shape),
            "residual": np.array([]),
            "rms_residual_s": float("nan"),
            "path_matrix": np.zeros((0, nodes.size)),
            "path_lengths": np.zeros(0),
            "n_observations": 0,
            "data_variance_reduction": float("nan"),
        }
    if tt.size != len(paths):
        raise ValueError("travel_times and paths must have the same length")
    good = np.isfinite(tt)
    A_all, lengths_all = path_matrix([p for p, g in zip(paths, good) if g], nodes, ds=ds)
    t_bg = A_all @ bg
    dt = tt[good] - t_bg
    if method == "art":
        sol = art(A_all, dt, n_iter=max(n_iter // 40, 5), relaxation=relaxation)
    elif method == "sirt":
        sol = sirt(
            A_all,
            dt,
            n_iter=n_iter,
            relaxation=relaxation,
            smooth=smooth,
            nodes=nodes,
            target_residual=target_residual,
            check_every=check_every,
        )
    else:
        raise ValueError(f"unknown method {method!r}")
    s = bg + sol["slowness"]
    resid = tt[good] - A_all @ s
    var_data = float(np.var(tt[good])) if tt[good].size > 1 else float("nan")
    return {
        "sound_speed": (1.0 / s).reshape(nodes.shape),
        "slowness": s.reshape(nodes.shape),
        "residual": resid,
        "rms_residual_s": float(np.sqrt(np.mean(resid**2))),
        "path_matrix": A_all,
        "path_lengths": lengths_all,
        "n_observations": int(good.sum()),
        "n_iter": int(sol.get("n_iter", n_iter)),
        "data_variance_reduction": float(1.0 - np.var(resid) / var_data) if var_data > 0 else float("nan"),
    }


def monte_carlo_uncertainty(
    paths,
    travel_times,
    nodes: NodeGrid,
    background_slowness,
    noise_sigma: float,
    n_realisations: int = 100,
    seed: int = 0,
    **solve_kwargs,
):
    """Ensemble spread (1 sigma) of the retrieved field under timing noise.

    Returns a dict with ``sigma_sound_speed`` [m/s], ``sigma_mean`` and the
    ensemble mean field.
    """
    tt = np.asarray(travel_times, dtype=float)
    if tt.size == 0:
        return {
            "sigma_sound_speed": np.full(nodes.shape, np.nan),
            "sigma_mean": float("nan"),
            "ensemble_mean": np.full(nodes.shape, np.nan),
            "n_realisations": 0,
        }
    rng = np.random.default_rng(seed)
    good = np.isfinite(tt)
    if not np.any(good):
        return {
            "sigma_sound_speed": np.full(nodes.shape, np.nan),
            "sigma_mean": float("nan"),
            "ensemble_mean": np.full(nodes.shape, np.nan),
            "n_realisations": 0,
        }
    paths_good = [p for p, g in zip(paths, good) if g]
    A, _ = path_matrix(paths_good, nodes, ds=float(solve_kwargs.pop("ds", 10.0)))
    bg = np.asarray(background_slowness, dtype=float).reshape(-1)
    dt0 = tt[good] - A @ bg
    method = solve_kwargs.pop("method", "sirt")
    target_residual = solve_kwargs.pop("target_residual", None)
    check_every = solve_kwargs.pop("check_every", 25)
    fields = []
    for _ in range(int(n_realisations)):
        perturbed = dt0 + rng.normal(0.0, noise_sigma, size=dt0.shape)
        if method == "art":
            sol = art(A, perturbed, n_iter=max(int(solve_kwargs.get("n_iter", 200)) // 40, 5),
                      relaxation=solve_kwargs.get("relaxation", 1.0))
        else:
            sol = sirt(
                A,
                perturbed,
                n_iter=int(solve_kwargs.get("n_iter", 200)),
                relaxation=solve_kwargs.get("relaxation", 1.0),
                smooth=float(solve_kwargs.get("smooth", 0.0)),
                nodes=nodes,
                target_residual=target_residual,
                check_every=check_every,
            )
        fields.append(1.0 / (bg + sol["slowness"]))
    stack = np.asarray(fields).reshape(int(n_realisations), *nodes.shape)
    sigma = np.nanstd(stack, axis=0, ddof=1)
    return {
        "sigma_sound_speed": sigma,
        "sigma_mean": float(np.nanmean(sigma)),
        "sigma_max": float(np.nanmax(sigma)),
        "ensemble_mean": np.nanmean(stack, axis=0),
        "n_realisations": int(n_realisations),
    }