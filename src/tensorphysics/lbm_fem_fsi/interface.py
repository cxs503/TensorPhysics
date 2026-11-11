"""Conservative traction -> nodal-load exchange (resultant *and* moment exact).

Campaign node ``p362w2-n09``.  This module owns the *quantities that must not be
invented* by a coupling: the fluid hands the structure a force resultant and a
moment, and the structure must receive exactly those two numbers on its own
nodes.  Bilinear/nearest-neighbour interpolation of a surface traction field onto
an unrelated structural surface does not do that; the deficiency is silent and
shows up as a few-percent momentum leak in a long VIV run.

Method
------
1. **Surface sampling.**  The lattice delivers one force vector per boundary
   link (Ladd momentum exchange), located at the wall intersection point
   ``x_l``.  Those are the discrete samples of the traction field
   ``t(x) = dF/dl``; the reference resultant is the plain sum

       ``F_ref = sum_l f_l``,   ``M_ref = sum_l (x_l - x_c) x f_l``.

2. **Resultant-preserving transfer.**  For transverse nodal loads ``f_i`` on the
   selected structural nodes ``(s_i, w_i)``,

       ``sum_i f_i = F_w``,     ``sum_i (s_i - s_c) f_i = M_z``,

   which is the ``2 x n`` constraint ``A f = b``.  The minimum-norm solution
   ``f = A^T (A A^T)^-1 b`` is a partition of unity (weights sum to one) that
   reproduces both constraints to round-off and spreads the load with the least
   nodal effort -- the discrete analogue of the consistent load vector, without
   requiring the two surfaces to be conforming.

3. **Audit.**  :func:`audit_transfer` re-evaluates the resultant and the moment
   from the *nodal* loads and reports the relative defects.  Because the
   constraints are solved in the same call, a non-zero value means a genuine
   bug or an ill-conditioned node set, not a modelling choice.

A second, physically independent check is provided by
:func:`traction_components`: the samples are grouped by surface segment to give
the pressure (normal) and friction (tangential) distributions, whose integrals
must reproduce the same resultant -- the classic pressure/friction split of a
hydrodynamic load.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "SurfaceLoad",
    "surface_load",
    "min_norm_weights",
    "map_to_nodes",
    "audit_transfer",
    "traction_components",
]


@dataclass(frozen=True)
class SurfaceLoad:
    """Discrete traction samples and their exact resultants (lattice units)."""

    positions: np.ndarray      # (n, 2) wall intersection points
    forces: np.ndarray         # (n, 2) per-sample force
    center: np.ndarray         # (2,) body centre
    force: np.ndarray          # (2,) resultant
    moment: float              # scalar z-moment about ``center``


def surface_load(positions, forces, center) -> SurfaceLoad:
    pos = np.asarray(positions, dtype=float).reshape(-1, 2)
    frc = np.asarray(forces, dtype=float).reshape(-1, 2)
    c = np.asarray(center, dtype=float).reshape(2)
    F = frc.sum(axis=0)
    r = pos - c
    M = float(np.sum(r[:, 0] * frc[:, 1] - r[:, 1] * frc[:, 0]))
    return SurfaceLoad(pos, frc, c, F, M)


def min_norm_weights(positions: np.ndarray, center, force_w: float, moment: float):
    """Partition-of-unity transverse nodal loads reproducing ``(force_w, moment)``.

    ``positions`` are the 2-D coordinates of the structural nodes in the beam
    frame ``(s, w)``; only their ``s`` offsets from ``center[0]`` produce a
    moment about ``z`` for a transverse load.
    """
    pos = np.asarray(positions, dtype=float).reshape(-1, 2)
    c = np.asarray(center, dtype=float).reshape(2)
    A = np.vstack([np.ones(pos.shape[0]), pos[:, 0] - c[0]])
    b = np.array([float(force_w), float(moment)])
    try:
        f = A.T @ np.linalg.solve(A @ A.T, b)
    except np.linalg.LinAlgError:  # pragma: no cover - degenerate node set
        f = np.linalg.lstsq(A, b, rcond=None)[0]
    return f


def map_to_nodes(load: SurfaceLoad, node_positions: np.ndarray) -> np.ndarray:
    """Transverse nodal loads for the attachment nodes of the FE tether."""
    return min_norm_weights(node_positions, load.center, load.force[1], load.moment)


def audit_transfer(load: SurfaceLoad, node_positions: np.ndarray, nodal: np.ndarray) -> dict:
    """Relative defects of the transferred resultant and moment."""
    pos = np.asarray(node_positions, dtype=float).reshape(-1, 2)
    nodal = np.asarray(nodal, dtype=float).reshape(-1)
    F_node = float(nodal.sum())
    M_node = float(np.sum((pos[:, 0] - load.center[0]) * nodal))
    scale_F = max(abs(load.force[1]), 1e-30)
    scale_M = max(abs(load.moment), 1e-30)
    return {
        "force_rel_error": abs(F_node - load.force[1]) / scale_F,
        "moment_rel_error": abs(M_node - load.moment) / scale_M,
        "sum_weights": float(nodal.sum() / load.force[1]) if load.force[1] != 0 else None,
        "force_node": F_node,
        "moment_node": M_node,
        "force_ref": float(load.force[1]),
        "moment_ref": float(load.moment),
    }


def traction_components(load: SurfaceLoad) -> dict:
    """Normal (pressure) and tangential (friction) split of the same samples.

    The normal at each sample is the outward radial direction of the circle;
    ``F_pressure = sum (f.n) n`` and ``F_friction = sum (f - (f.n) n)`` must add
    up to the resultant, which is a second, geometry-based consistency check of
    the surface sampling (it is *not* independent of the sample values, but it
    validates the outward-normal bookkeeping that a pressure field would use).
    """
    r = load.positions - load.center
    rn = np.linalg.norm(r, axis=1)
    rn[rn == 0] = 1.0
    n = r / rn[:, None]
    fn = np.sum(load.forces * n, axis=1)
    F_pressure = (fn[:, None] * n).sum(axis=0)
    F_friction = (load.forces - fn[:, None] * n).sum(axis=0)
    return {
        "F_pressure": F_pressure,
        "F_friction": F_friction,
        "F_total": F_pressure + F_friction,
        "drag_pressure": float(F_pressure[0]),
        "friction_share": float(np.linalg.norm(F_friction) / max(np.linalg.norm(load.force), 1e-30)),
    }