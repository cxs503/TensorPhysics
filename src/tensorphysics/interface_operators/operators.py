"""Three interface operators + deliberately broken negative controls.

The operators
-------------
``MortarProjection``
    Area-weighted (mortar) projection: target cell mean = integral of the
    piecewise-linear reconstruction of the donor data over the overlap, divided
    by the target cell measure.  Conservative by construction for *any* data
    because the overlap integrals partition every source cell exactly.

``ConservativeScatter``
    RBF / radial-basis scatter: Gaussian kernel with local linear (MLS)
    augmentation for consistency, followed by an explicit global conservative
    deficit correction so that the received integral matches the sent one.

``LBMDEMMomentumExchange``
    LBM-DEM momentum exchange (ME): node momentum is gathered onto particles
    with a kernel normalised *per node* (so every node's momentum is fully
    handed out), and particle reaction forces are spread back to nodes with a
    kernel normalised *per particle* (so every particle's impulse is fully
    handed back).  Those two normalisations are exactly what closes the
    fluid+particle momentum budget.

Negative controls (``broken=...``)
----------------------------------
Each operator ships one deliberately wrong variant whose only purpose is to be
*red* under the acceptance gates (campaign criterion C6):

* mortar ``broken="midpoint"``    -- flux by point sampling, non-conservative on
  non-matching meshes (the classic "first-order flux on a staggered grid" bug);
* scatter ``broken="no_fix"``     -- RBF gather without the conservative
  correction, loses/gains mass at boundaries and on steep fields;
* ME ``broken="per_node_norm"``   -- spread kernel normalised per node instead
  of per particle, which silently discards part of the particle impulse.
"""
from __future__ import annotations

import numpy as np

from .core import (InterfaceOperator, PointSet, SegmentMesh, measure)

__all__ = ["MortarProjection", "ConservativeScatter",
           "LBMDEMMomentumExchange"]


_EPS = 1e-15


def _overlaps(src: SegmentMesh, tgt: SegmentMesh):
    """Exact 1D overlap decomposition of two partitions of the same interval."""
    a, b = src.edges, tgt.edges
    if abs(a[0] - b[0]) > 1e-12 or abs(a[-1] - b[-1]) > 1e-12:
        raise ValueError(f"interface extents differ: src{src.bounds} tgt{tgt.bounds}")
    i = j = 0
    out = []
    while i < src.n and j < tgt.n:
        lo = max(a[i], b[j])
        hi = min(a[i + 1], b[j + 1])
        if hi - lo > _EPS:
            out.append((i, j, lo, hi))
        if a[i + 1] <= b[j + 1]:
            i += 1
        else:
            j += 1
    return out


def _slope_matrix(mesh: SegmentMesh) -> np.ndarray:
    """Matrix S with (S u)_i = 2nd-order slope estimate of cell mean u_i."""
    n = mesh.n
    S = np.zeros((n, n))
    if n == 1:
        return S
    xc = mesh.centers
    if n == 2:
        d = xc[1] - xc[0]
        S[0, 0], S[0, 1] = -1.0 / d, 1.0 / d
        S[1, 1], S[1, 0] = 1.0 / d, -1.0 / d
        return S
    d = xc[1] - xc[0]
    S[0, 1], S[0, 0] = 1.0 / d, -1.0 / d                       # one-sided (2nd order)
    d = xc[-1] - xc[-2]
    S[-1, -1], S[-1, -2] = 1.0 / d, -1.0 / d
    daw = xc[2:] - xc[:-2]
    idx = np.arange(1, n - 1)
    S[idx, idx - 1] = -1.0 / daw
    S[idx, idx + 1] = 1.0 / daw
    return S


class MortarProjection(InterfaceOperator):
    """Conservative mortar (finite-volume) projection between 1D interfaces."""

    name = "mortar-area-weighted"

    def __init__(self, order: int = 2, broken: str | None = None) -> None:
        if order not in (1, 2):
            raise ValueError("order must be 1 or 2")
        if broken not in (None, "midpoint", "clip"):
            raise ValueError(f"unknown broken mode {broken!r}")
        self.order = order
        self.broken = broken
        if broken is not None:
            self.name = f"mortar-area-weighted[{broken}-NEGATIVE-CONTROL]"

    # -- linear representation -------------------------------------------
    def matrix(self, src: SegmentMesh, tgt: SegmentMesh) -> np.ndarray:
        ns, nt = src.n, tgt.n
        D = np.zeros((nt, ns))          # pure area weighting
        G = np.zeros((nt, ns))          # first moment about the source centroid
        xc, W = src.centers, tgt.widths
        for i, j, lo, hi in _overlaps(src, tgt):
            d = hi - lo
            D[j, i] += d / W[j]
            G[j, i] += (0.5 * (lo + hi) - xc[i]) * d / W[j]
        if self.order == 1:
            return D
        return D + G @ _slope_matrix(src)

    # -- the map ---------------------------------------------------------
    def transfer(self, src: SegmentMesh, src_values: np.ndarray, tgt: SegmentMesh) -> np.ndarray:
        u = np.asarray(src_values, dtype=float)
        if self.broken == "midpoint":
            # point-sample the reconstruction at the target centroid
            k = np.clip(np.searchsorted(src.edges, tgt.centers, side="right") - 1,
                        0, src.n - 1)
            s = _slope_matrix(src) @ u
            return u[k] + s[k] * (tgt.centers - src.centers[k])
        v = self.matrix(src, tgt) @ u
        if self.broken == "clip":       # "positivity fix" that silently destroys mass
            v = np.maximum(v, 0.0)
        return v


class ConservativeScatter(InterfaceOperator):
    """RBF scatter with linear (MLS) augmentation + conservative correction."""

    name = "rbf-scatter"

    def __init__(self, epsilon: float = 1.8, linear_augment: bool = True,
                 broken: str | None = None) -> None:
        if broken not in (None, "no_fix"):
            raise ValueError(f"unknown broken mode {broken!r}")
        self.epsilon = float(epsilon)
        self.linear_augment = bool(linear_augment)
        self.broken = broken
        if broken is not None:
            self.name = f"rbf-scatter[{broken}-NEGATIVE-CONTROL]"
        self.conservative = broken is None

    def transfer(self, src, src_values: np.ndarray, tgt) -> np.ndarray:
        u = np.asarray(src_values, dtype=float)
        xc = np.asarray(src.centers, dtype=float)          # scatter: source centres
        yc = np.asarray(tgt.centers, dtype=float)          # gather: target centres
        eps = self.epsilon * float(np.mean(measure(src)))
        D = xc[None, :] - yc[:, None]                       # d_ji = x_i - y_j
        w = np.exp(-((D / eps) ** 2))                       # (nt, ns)
        S0 = w.sum(axis=1)
        R0 = w @ u
        w = np.maximum(w, 1e-300)
        if self.linear_augment:
            S1 = np.sum(w * D, axis=1)
            S2 = np.sum(w * D * D, axis=1)
            R1 = (w * D) @ u
            det = S0 * S2 - S1 * S1
            good = np.abs(det) > 1e-14 * np.maximum(S0 * S2, 1e-300)
            v = np.where(good, (R0 * S2 - R1 * S1) / np.where(good, det, 1.0),
                         R0 / np.maximum(S0, 1e-300))
        else:
            v = R0 / np.maximum(S0, 1e-300)
        if self.conservative:
            sent = float(np.sum(u * measure(src)))
            recv = float(np.sum(v * measure(tgt)))
            mt = measure(tgt)
            denom = float(np.sum(mt * mt)) or 1.0
            # measure-proportional deficit redistribution: int(delta v) = sent-recv
            v = v + (sent - recv) * mt / denom
        return v


class LBMDEMMomentumExchange(InterfaceOperator):
    """LBM-DEM momentum exchange: gather node momentum, spread particle reaction.

    Side conventions
    ----------------
    * node data are *momenta per unit volume* on the lattice cells (measure =
      cell width), particles are PointSets whose ``measures`` carry the
      effective exchange volume ``V_p = sum_l lambda_lp w_l``;
    * :meth:`gather` returns the particle force *density* (impulse / V_p), so
      that ``sum_p v_p * V_p`` is the momentum handed to the DEM side;
    * :meth:`spread` returns the reaction *density* on the nodes, i.e. the
      momentum per unit volume handed back to the lattice.
    """

    name = "lbm-dem-momentum-exchange"

    def __init__(self, epsilon: float = 1.2, moment_correct: bool = True,
                 broken: str | None = None) -> None:
        if broken not in (None, "per_node_norm"):
            raise ValueError(f"unknown broken mode {broken!r}")
        self.epsilon = float(epsilon)
        self.moment_correct = bool(moment_correct)
        self.broken = broken
        if broken is not None:
            self.name = f"lbm-dem-momentum-exchange[{broken}-NEGATIVE-CONTROL]"

    # -- kernels ---------------------------------------------------------
    def _kernel(self, nodes: SegmentMesh, particles: PointSet) -> np.ndarray:
        eps = self.epsilon * float(np.mean(nodes.widths))
        xl = nodes.centers[:, None]
        yp = np.asarray(particles.centers)[None, :]
        return np.exp(-(((yp - xl) / eps) ** 2))            # (nl, np)

    def _gather_weights(self, nodes: SegmentMesh, particles: PointSet) -> np.ndarray:
        """lambda[l, p]: normalised *per node* (each node hands out its momentum).

        Row constraints of the second-order (moment-preserving) variant:
        ``sum_p lambda = 1`` and ``sum_p lambda (x_p - x_l) = 0``.  They are
        imposed on the normalised kernel ``q`` through a weighted-variance
        correction ``lambda = q (1 + gamma (d - dbar))`` which is algebraically
        well conditioned (the denominator is the kernel-weighted variance of the
        particle offset), unlike the raw 2x2 normal-equation solve.
        """
        w = np.maximum(self._kernel(nodes, particles), 1e-300)
        q = w / w.sum(axis=1, keepdims=True)
        if not self.moment_correct:
            return q
        d = np.asarray(particles.centers)[None, :] - nodes.centers[:, None]
        dbar = np.sum(q * d, axis=1, keepdims=True)
        var = np.sum(q * (d - dbar) ** 2, axis=1, keepdims=True)
        ref = float(np.mean(nodes.widths))
        # Tikhonov-regularised + bounded slope correction: exact for a linear
        # field whenever the local particle cloud is well spread, and *bounded*
        # (never catastrophic) when a node only sees one or two particles.
        gamma = -dbar / (var + 0.0625 * ref * ref)
        gamma = np.clip(gamma, -1.0 / ref, 1.0 / ref)
        lam = q * (1.0 + gamma * (d - dbar))
        return lam / lam.sum(axis=1, keepdims=True)

    def particle_measure(self, nodes: SegmentMesh, particles: PointSet) -> np.ndarray:
        """Effective exchange volume of each particle: V_p = sum_l lambda_lp w_l."""
        lam = self._gather_weights(nodes, particles)
        return lam.T @ nodes.widths

    def dem_pointset(self, nodes: SegmentMesh, centers) -> PointSet:
        """Build the DEM side of the interface with consistent exchange volumes."""
        base = PointSet(np.asarray(centers, dtype=float), None, label="dem-particles",
                        kind="particle")
        return PointSet(base.centers, self.particle_measure(nodes, base),
                        label="dem-particles", kind="particle")

    # -- gather / spread --------------------------------------------------
    def gather(self, nodes: SegmentMesh, node_momentum: np.ndarray,
               particles: PointSet) -> np.ndarray:
        m = np.asarray(node_momentum, dtype=float) * nodes.widths   # momentum per node
        lam = self._gather_weights(nodes, particles)
        impulse = lam.T @ m
        return impulse / np.maximum(np.asarray(particles.measures), 1e-300)

    def spread(self, particles: PointSet, particle_density: np.ndarray,
               nodes: SegmentMesh) -> np.ndarray:
        """Node reactions (momentum density) for the given particle densities.

        The spread kernel is normalised *per particle* so that
        ``sum_l reac_l * w_l == -sum_p F_p``: the fluid strictly receives the
        negative of the particle impulse (closed-system momentum conservation).
        """
        imp = np.asarray(particle_density, dtype=float) * np.asarray(particles.measures)
        w = np.maximum(self._kernel(nodes, particles), 1e-300)      # (nl, np)
        if self.broken == "per_node_norm":
            mu = w / w.sum(axis=1, keepdims=True)                   # WRONG: per node
        else:
            mu = w / w.sum(axis=0, keepdims=True)                   # per particle
        return -(mu @ imp) / nodes.widths

    # -- the contract map --------------------------------------------------
    def transfer(self, src, src_values: np.ndarray, tgt) -> np.ndarray:
        if isinstance(tgt, PointSet) and tgt.kind == "particle":
            return self.gather(src, src_values, tgt)
        if isinstance(src, PointSet) and src.kind == "particle":
            return self.spread(src, src_values, tgt)
        raise TypeError("LBMDEMMomentumExchange needs one particle PointSet side")
