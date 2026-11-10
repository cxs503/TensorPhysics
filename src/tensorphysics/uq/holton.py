"""Holton-type high-dimensional linear + interaction anchor with closed forms.

The Ishigami anchor of :mod:`tensorphysics.uq.benchmarks` is three-dimensional
and exercises a strong interaction ``x3^4 sin(x1)``.  It does **not** exercise
the situation this anchor is for: a *high-dimensional* model whose variance is
dominated by a few linear directions plus a band of pairwise interactions, i.e.
the structure that sparse polynomial chaos is supposed to recover cheaply.

Model
-----
For ``x_i ~ U(0, 1)`` independent and coefficients ``a_i`` (linear) and
``b_ij`` (pairwise interaction)::

    Y = sum_i a_i x_i + sum_{i<j} b_ij x_i x_j

Centring ``x_i = 1/2 + u_i`` with ``u_i ~ U(-1/2, 1/2)`` (``Var u = 1/12``)::

    Y = c0 + sum_i c_i u_i + sum_{i<j} b_ij u_i u_j
    c0 = sum_i a_i / 2 + sum_{i<j} b_ij / 4
    c_i = a_i + (1/2) sum_{j != i} b_ij

The centred terms are mutually orthogonal (independent, symmetric ``u_i``):
``E[u_i] = 0``, ``E[u_i u_j] = 0``, ``E[u_i^2 u_j] = 0`` and
``E[u_i u_j u_k u_l] = 0`` unless the indices pair up.  Hence the exact
Sobol' decomposition is available in closed form::

    Var(Y)    = sum_i c_i^2 / 12 + sum_{i<j} b_ij^2 / 144
    S_i       = (c_i^2 / 12) / Var(Y)
    S_Ti      = (c_i^2 / 12 + sum_{j != i} b_ij^2 / 144) / Var(Y)
    S_ij      = (b_ij^2 / 144) / Var(Y)

and ``sum_i S_Ti = 1 + sum_{i<j} S_ij`` exactly for a model with no
three-way term, which is a useful machine-checkable invariant (and the reason
the total indices of the default anchor sum to about 1.18, not 1.0).

Naming note
-----------
The campaign node calls this the "Holton-type" anchor.  No published function
named Holton is reproduced here; what is used is the campaign's own
specification -- *high-dimensional linear + interaction terms* -- instantiated
as the model above so that the indices are known analytically.  The
interpretation is recorded here rather than presented as a literature citation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from .sample_spec import Marginal, SampleSpec

__all__ = [
    "HoltonBenchmark",
    "holton",
    "holton_reference_indices",
    "make_holton",
    "holton_analytic",
    "holton_spec",
    "holton_default_coefficients",
]

_UNIFORM_VARIANCE = 1.0 / 12.0


def _incident_sums(b: np.ndarray) -> np.ndarray:
    """``sum_j b_ij`` for a pairwise coefficient matrix stored on one triangle."""
    b = np.asarray(b, dtype=float)
    return b.sum(axis=1) + b.sum(axis=0) - np.diag(b)


def holton_default_coefficients(
    dim: int, *, interaction_band: int = 2, interaction_scale: float = 1.4
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic default coefficients for the anchor.

    The defaults are specified in the **centred** parametrisation, because that
    is the spectrum the anchor is supposed to test:

    * centred linear coefficients ``c_i = 0.5 / (1 + i)`` -- the first inputs
      dominate and the tail is a genuine sparsity test;
    * banded pairwise interactions ``b_ij = scale / (1 + |i - j|)^2`` for
      ``0 < |i - j| <= interaction_band`` (default band 2, ``scale=1.4``),
      which contributes an interaction share of about 18 % of ``Var(Y)`` -- large
      enough that a first-order-only analysis is provably insufficient, small
      enough that the first-order indices still carry most of the variance;
    * the physical coefficients are then ``a_i = c_i - (1/2) sum_j b_ij`` so that
      :func:`holton` reproduces ``c_i`` exactly when evaluated on ``[0, 1]^d``.
    """
    if dim < 2:
        raise ValueError(f"holton needs dim >= 2, got {dim}")
    if interaction_band < 1:
        raise ValueError(f"interaction_band must be >= 1, got {interaction_band}")
    c = np.array([0.5 / (1.0 + i) for i in range(dim)], dtype=float)
    b = np.zeros((dim, dim), dtype=float)
    for i in range(dim):
        for j in range(i + 1, dim):
            sep = j - i
            if sep <= interaction_band:
                b[i, j] = interaction_scale / float(1 + sep) ** 2
    a = c - 0.5 * _incident_sums(b)
    return a, b


@dataclass(frozen=True)
class HoltonBenchmark:
    """The high-dimensional anchor: callable + specification + closed forms."""

    dim: int
    linear: np.ndarray = field(repr=False)
    interaction: np.ndarray = field(repr=False)

    def __post_init__(self) -> None:
        if self.linear.shape != (self.dim,) or self.interaction.shape != (self.dim, self.dim):
            raise ValueError("linear/interaction shapes do not match dim")

    # -- model -------------------------------------------------------------
    def __call__(self, points: np.ndarray) -> np.ndarray:
        return holton(points, linear=self.linear, interaction=self.interaction)

    def spec(self) -> SampleSpec:
        return holton_spec(self.dim)

    def analytic(self) -> dict[str, Any]:
        return holton_analytic(
            self.dim, linear=self.linear, interaction=self.interaction
        )

    def as_dict(self) -> dict[str, Any]:
        a = self.linear
        b = [(i, j, float(self.interaction[i, j])) for i in range(self.dim)
             for j in range(i + 1, self.dim) if self.interaction[i, j] != 0.0]
        return {
            "model": "holton_linear_plus_pairwise_interaction",
            "dim": int(self.dim),
            "linear": [float(v) for v in a],
            "interaction": [{"i": int(i), "j": int(j), "b": float(v)} for i, j, v in b],
            "support": "[0, 1]^dim, independent uniform marginals",
            "analytic": {
                "mean": float(self.analytic()["mean"]),
                "variance": float(self.analytic()["variance"]),
                "first": [float(v) for v in self.analytic()["first"]],
                "total": [float(v) for v in self.analytic()["total"]],
            },
        }


def make_holton(
    dim: int = 8, *, interaction_band: int = 2, interaction_scale: float = 1.4
) -> HoltonBenchmark:
    """Build the default anchor for ``dim`` inputs."""
    a, b = holton_default_coefficients(
        dim, interaction_band=interaction_band, interaction_scale=interaction_scale
    )
    return HoltonBenchmark(dim=int(dim), linear=a, interaction=b)


def holton(
    points: np.ndarray,
    *,
    linear: Sequence[float] | np.ndarray | None = None,
    interaction: Sequence[Sequence[float]] | np.ndarray | None = None,
    dim: int | None = None,
) -> np.ndarray:
    """``Y = sum_i a_i x_i + sum_{i<j} b_ij x_i x_j`` on ``[0, 1]^d``.

    ``linear``/``interaction`` default to :func:`holton_default_coefficients`
    for the column count of ``points`` (or ``dim`` when given).
    """
    x = np.asarray(points, dtype=float)
    if x.ndim != 2:
        raise ValueError(f"holton expects (n, d) points, got {x.shape}")
    n, d = x.shape
    if dim is not None and int(dim) != d:
        raise ValueError(f"dim={dim} does not match the {d} columns of points")
    a_def, b_def = holton_default_coefficients(d)
    a = a_def if linear is None else np.asarray(linear, dtype=float).reshape(-1)
    b = b_def if interaction is None else np.asarray(interaction, dtype=float)
    if a.shape != (d,):
        raise ValueError(f"linear must have {d} entries, got {a.shape}")
    if b.shape != (d, d):
        raise ValueError(f"interaction must be ({d}, {d}), got {b.shape}")
    y = x @ a
    # pairwise term; the upper triangle is used so b_ij is not double counted
    iu, ju = np.triu_indices(d, k=1)
    if iu.size:
        y = y + (x[:, iu] * x[:, ju] * b[iu, ju]).sum(axis=1)
    return y


def holton_spec(dim: int = 8) -> SampleSpec:
    """Independent ``U(0, 1)`` marginals (Legendre chaos coordinates)."""
    if dim < 2:
        raise ValueError(f"holton needs dim >= 2, got {dim}")
    return SampleSpec(tuple(Marginal.uniform(f"x{i + 1}", 0.0, 1.0) for i in range(dim)))


def holton_analytic(
    dim: int = 8,
    *,
    linear: Sequence[float] | np.ndarray | None = None,
    interaction: Sequence[Sequence[float]] | np.ndarray | None = None,
) -> dict[str, Any]:
    """Closed-form variance decomposition of :func:`holton`."""
    a_def, b_def = holton_default_coefficients(dim)
    a = a_def if linear is None else np.asarray(linear, dtype=float).reshape(-1)
    b = b_def if interaction is None else np.asarray(interaction, dtype=float)
    if a.shape != (dim,) or b.shape != (dim, dim):
        raise ValueError("linear/interaction shapes do not match dim")

    # centred coefficients c_i = a_i + (1/2) sum_{j != i} b_ij
    b_sym = b + b.T  # pairs are stored once; symmetrise for the shift
    # x_i = 1/2 + u_i gives each pair a shift 1/2 b_ij on *each* of i and j,
    # i.e. c_i = a_i + (1/2) sum_j b_ij.  The band is stored once (i < j), so
    # the incident sum must cover both triangles.
    c = a + 0.5 * _incident_sums(b)
    var_linear = float(np.sum(c**2) / (12.0))
    pair_variance = 0.0
    second: dict[tuple[int, int], float] = {}
    for i in range(dim):
        for j in range(i + 1, dim):
            v = float(b[i, j] ** 2) * _UNIFORM_VARIANCE**2
            if v:
                second[(i, j)] = v
                pair_variance += v
    variance = var_linear + pair_variance
    if variance <= 0.0:
        raise ValueError("degenerate coefficients: zero output variance")
    first = (c**2) / (12.0) / variance
    total = first.copy()
    for (i, j), v in second.items():
        total[i] += v / variance
        total[j] += v / variance
    mean = float(np.sum(a) / 2.0 + sum(b[i, j] for i in range(dim) for j in range(i + 1, dim)) / 4.0)
    return {
        "mean": mean,
        "variance": variance,
        "first": first,
        "total": total,
        "second": {k: v / variance for k, v in second.items()},
        "linear_variance_share": var_linear / variance,
        "interaction_variance_share": pair_variance / variance,
        "total_sum": float(np.sum(total)),
        "dim": int(dim),
    }


def holton_reference_indices(dim: int = 8) -> dict[str, Any]:
    """Alias kept for symmetry with :func:`ishigami_analytic` call sites."""
    out = holton_analytic(dim)
    out["names"] = tuple(f"x{i + 1}" for i in range(dim))
    out["sqrt"] = math.sqrt(out["variance"])
    return out