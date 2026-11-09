"""Variance-based (Saltelli / Jansen) Sobol' indices from direct model runs.

This is the **independent second opinion** to the polynomial-chaos route in
:mod:`tensorphysics.uq.pce`: it makes no smoothness assumption and only needs
model evaluations, so it can arbitrate a chaos expansion that has quietly
over-fitted or under-resolved an interaction.

Design
------
Two base matrices ``A`` and ``B`` of ``n_base`` unit-hypercube points each are
built from a scrambled Sobol' sequence, randomly permuted and split so that the
rows of ``A`` and ``B`` are *not* paired (two separately seeded Sobol' sets are
correlated through the shared direction numbers -- an easy way to get
N-independent, wrong indices; the random permutation removes that coupling).
For each input ``i`` the matrix ``A_B^i`` copies column ``i`` of ``B`` into
``A``; for each pair ``(i, j)`` the matrix ``A_B^{ij}`` copies both columns.

Estimators (``V`` = variance of ``f(A)`` and ``f(B)`` pooled, ``f`` centred
implicitly through the products):

* first order, Jansen (default):   ``S_i = 1 - E[(f(B) - f(A_B^i))**2] / (2 V)``
* first order, Saltelli (2010):    ``S_i = E[f(B) (f(A_B^i) - f(A))] / V``
* total order, Jansen:             ``S_T,i = E[(f(A) - f(A_B^i))**2] / (2 V)``
* second order:                    ``S_ij = E[f(B)(f(A_B^ij) - f(A_B^i) - f(A_B^j) + f(A))] / V``

Both first-order variants are unbiased and are reported side by side; the
Jansen form is the one used for the cross-check because it was the more
accurate of the two on the Ishigami anchor (see the UQ report).

The ``evaluate`` callable receives **unit-hypercube** points: the pick-freeze
construction is only meaningful in the independent coordinates of the design,
so the caller maps them to physical parameters (``spec.transform``) before
calling the solver.  Passing physical points directly would silently replace
the design's independence structure.

Cost: ``2 + d + d(d-1)/2`` model blocks of ``n_base`` runs each.  For ``d = 6``
and ``n_base = 2**16`` that is 23 blocks (~1.5 M black-box calls), which is why
the UQ workflow lets the adapter fan these blocks out over cluster shards.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

from .sampler import draw_unit

__all__ = ["SobolMCResult", "saltelli_indices", "saltelli_design"]

_EVALUATE = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True, eq=False)
class SobolMCResult:
    """Monte-Carlo (pick-freeze) Sobol' indices with optional bootstrap CIs."""

    n_base: int
    n_model_evaluations: int
    mean: float
    variance: float
    first: np.ndarray = field(repr=False)
    first_saltelli: np.ndarray = field(repr=False)
    total: np.ndarray = field(repr=False)
    second: dict[tuple[int, int], float] = field(default_factory=dict)
    first_ci: tuple[np.ndarray, np.ndarray] | None = field(default=None, repr=False)
    total_ci: tuple[np.ndarray, np.ndarray] | None = field(default=None, repr=False)
    seed: int = 0
    rule: str = "sobol"
    bootstrap: int = 0

    @property
    def dim(self) -> int:
        return int(self.first.size)

    @property
    def first_sum(self) -> float:
        return float(np.sum(self.first))

    @property
    def estimator_spread(self) -> np.ndarray:
        """``|S_i(Jansen) - S_i(Saltelli)|`` -- internal estimator disagreement."""
        return np.abs(self.first - self.first_saltelli)

    def as_dict(self, names: Sequence[str] | None = None) -> dict[str, Any]:
        dim = self.dim
        names = tuple(names) if names is not None else tuple(f"x{i}" for i in range(dim))
        out: dict[str, Any] = {
            "n_base": int(self.n_base),
            "n_model_evaluations": int(self.n_model_evaluations),
            "mean": self.mean,
            "variance": self.variance,
            "seed": int(self.seed),
            "rule": self.rule,
            "bootstrap": int(self.bootstrap),
            "first_jansen": {names[i]: float(self.first[i]) for i in range(dim)},
            "first_saltelli": {names[i]: float(self.first_saltelli[i]) for i in range(dim)},
            "total": {names[i]: float(self.total[i]) for i in range(dim)},
            "second": {f"{names[i]}*{names[j]}": float(v) for (i, j), v in self.second.items()},
            "first_sum": self.first_sum,
        }
        if self.first_ci is not None:
            lo, hi = self.first_ci
            out["first_ci95"] = {names[i]: [float(lo[i]), float(hi[i])] for i in range(dim)}
        if self.total_ci is not None:
            lo, hi = self.total_ci
            out["total_ci95"] = {names[i]: [float(lo[i]), float(hi[i])] for i in range(dim)}
        return out


def saltelli_design(
    dim: int, n_base: int, *, seed: int = 20240, rule: str = "sobol"
) -> dict[str, np.ndarray]:
    """Return the unit-cube matrices ``A``, ``B`` and the derived ``A_B`` blocks.

    Exposed so that a cluster worker can be handed exactly the same design and
    only evaluate model blocks; the mapping from block name to matrix is stable.
    """
    if dim < 1:
        raise ValueError(f"dim must be >= 1, got {dim}")
    if n_base < 2:
        raise ValueError(f"n_base must be >= 2, got {n_base}")
    unit = draw_unit(2 * n_base, dim, rule=rule, seed=seed)
    rng = np.random.default_rng(seed + 1)
    perm = rng.permutation(2 * n_base)
    A = unit[perm[:n_base]]
    B = unit[perm[n_base:]]
    blocks: dict[str, np.ndarray] = {"A": A, "B": B}
    for i in range(dim):
        M = A.copy()
        M[:, i] = B[:, i]
        blocks[f"A_B_{i}"] = M
    for i in range(dim):
        for j in range(i + 1, dim):
            M = A.copy()
            M[:, i] = B[:, i]
            M[:, j] = B[:, j]
            blocks[f"A_B_{i}_{j}"] = M
    return blocks


def saltelli_indices(
    evaluate: _EVALUATE,
    dim: int,
    *,
    n_base: int = 1 << 16,
    seed: int = 20240,
    rule: str = "sobol",
    second_order: bool = True,
    bootstrap: int = 0,
    bootstrap_seed: int | None = None,
) -> SobolMCResult:
    """Estimate first/total/second-order Sobol' indices from model runs.

    Parameters
    ----------
    evaluate:
        ``evaluate(unit_points) -> values`` black box, ``unit_points`` of shape
        ``(n_base, dim)`` in ``[0, 1)^dim`` (map to physical parameters inside,
        e.g. with :meth:`tensorphysics.uq.sample_spec.SampleSpec.transform`).
    dim:
        Number of inputs.
    n_base:
        Base sample size per matrix (model evaluations = ``2 + d + d(d-1)/2``
        blocks of ``n_base`` for second order).
    seed:
        Seed of the pick-freeze design (recorded, part of the evidence).
    second_order:
        Also estimate the ``d(d-1)/2`` closed second-order indices.
    bootstrap:
        Number of bootstrap resamples for percentile 95% intervals of the
        first/total indices (0 disables).
    """
    if not callable(evaluate):
        raise TypeError("evaluate must be callable")
    spec_blocks = saltelli_design(dim, n_base, seed=seed, rule=rule)
    A, B = spec_blocks["A"], spec_blocks["B"]

    def _run(points: np.ndarray) -> np.ndarray:
        values = np.asarray(evaluate(points), dtype=float).reshape(-1)
        if values.size != points.shape[0]:
            raise ValueError(
                f"evaluate returned {values.size} values for {points.shape[0]} points"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError(
                "evaluate returned non-finite values; the UQ verdict must be withheld"
            )
        return values

    yA = _run(A)
    yB = _run(B)
    n_eval = 2 * n_base
    yAi: list[np.ndarray] = []
    for i in range(dim):
        yAi.append(_run(spec_blocks[f"A_B_{i}"]))
        n_eval += n_base
    second: dict[tuple[int, int], float] = {}
    yAij: dict[tuple[int, int], np.ndarray] = {}
    if second_order and dim > 1:
        for i in range(dim):
            for j in range(i + 1, dim):
                key = (i, j)
                yAij[key] = _run(spec_blocks[f"A_B_{i}_{j}"])
                n_eval += n_base

    V = float(np.var(np.concatenate([yA, yB])))
    if V <= 0.0:
        raise ValueError("model output has zero variance; Sobol indices are undefined")
    # Sobol' indices are invariant under the shift f -> f + c.  The Jansen
    # estimators below only use *differences* of model outputs and are therefore
    # shift-invariant by construction.  The Saltelli (2010) first-order estimator
    # and the closed second-order estimator multiply such a difference by a raw
    # model value; with a large output offset (a ship resistance of O(1e3) kN, a
    # pressure of O(1e5) Pa) that factor is a pure-noise amplifier: the exact
    # identity E[f_{A_i} - f_A] = 0 holds only in expectation, so the offset
    # leaks into the estimate as c * O(sigma / sqrt(N)).  Centring the multiplier
    # on the grand mean removes the leak exactly and leaves the estimator
    # unchanged for a zero-mean output.
    grand = float(np.mean(np.concatenate([yA, yB])))
    centred_b = yB - grand
    first_j = np.empty(dim)
    first_s = np.empty(dim)
    total = np.empty(dim)
    for i in range(dim):
        diff = yB - yAi[i]
        first_j[i] = 1.0 - float(np.mean(np.square(diff))) / (2.0 * V)
        first_s[i] = float(np.mean(centred_b * (yAi[i] - yA))) / V
        total[i] = float(np.mean(np.square(yA - yAi[i]))) / (2.0 * V)
    for (i, j), yij in yAij.items():
        second[(i, j)] = (
            float(np.mean(centred_b * (yij - yAi[i] - yAi[j] + yA))) / V
        )

    first_ci = None
    total_ci = None
    if bootstrap:
        if bootstrap < 10:
            raise ValueError(f"bootstrap must be 0 or >= 10, got {bootstrap}")
        rng = np.random.default_rng(
            bootstrap_seed if bootstrap_seed is not None else seed + 7
        )
        # sufficient statistics for the two index families
        prod_first = np.column_stack(
            [1.0 - np.square(yB - yAi[i]) / (2.0 * V) for i in range(dim)]
        )
        prod_total = np.column_stack(
            [np.square(yA - yAi[i]) / (2.0 * V) for i in range(dim)]
        )
        f_rep = np.empty((bootstrap, dim))
        t_rep = np.empty((bootstrap, dim))
        for b in range(bootstrap):
            take = rng.integers(0, n_base, size=n_base)
            f_rep[b] = prod_first[take].mean(axis=0)
            t_rep[b] = prod_total[take].mean(axis=0)
        first_ci = (np.percentile(f_rep, 2.5, axis=0), np.percentile(f_rep, 97.5, axis=0))
        total_ci = (np.percentile(t_rep, 2.5, axis=0), np.percentile(t_rep, 97.5, axis=0))

    return SobolMCResult(
        n_base=int(n_base),
        n_model_evaluations=int(n_eval),
        mean=float(0.5 * (yA.mean() + yB.mean())),
        variance=V,
        first=first_j,
        first_saltelli=first_s,
        total=total,
        second=second,
        first_ci=first_ci,
        total_ci=total_ci,
        seed=int(seed),
        rule=rule,
        bootstrap=int(bootstrap),
    )