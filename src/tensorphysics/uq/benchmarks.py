"""Analytic UQ benchmarks with closed-form Sobol' indices (self-verification).

Every UQ capability must be verifiable against a problem whose variance
decomposition is known exactly; otherwise "the indices converged" only means
"the code agrees with itself".  Two classical benchmarks are provided.

Ishigami function (Ishigami & Homma 1990)
-----------------------------------------
``Y = sin(x1) + a sin^2(x2) + b x3^4 sin(x1)``, ``x_i ~ U(-pi, pi)``, ``a=7``,
``b=0.1``.  With ``V1 = 0.5 (1 + b pi^4 / 5)^2``, ``V2 = a^2/8``,
``V13 = b^2 pi^8 (1/18 - 1/50)`` and ``V = V1 + V2 + V13``:

``S1 = (V1, V2, 0)/V``, ``ST = (V1 + V13, V2, V13)/V``, ``S13 = V13/V``.

The strengths of the benchmark are exactly what a sparse PCE has to get right:
``x3`` has **zero** first-order effect but a large total effect, ``x1`` has a
non-polynomial (sine) effect, and ``x3`` enters through a degree-4 interaction --
so a low-order or over-truncated expansion cannot fake success.

Sobol' g-function
-----------------
``g = prod_i (|4 x_i - 2| + a_i)/(1 + a_i)``, ``x_i ~ U(0, 1)`` with
``V_i = (1/3)/(1 + a_i)^2`` and ``Var(g) = prod_i (1 + V_i) - 1``, giving
``S1_i = V_i / Var(g)`` and ``ST_i = V_i prod_{j != i}(1 + V_j) / Var(g)``.
With the classical decay coefficients ``a = (0, 1, 4.5, 9, 99, 99, 99, 99)`` the
effective dimension is ~3 in 8 inputs, which is the case that justifies sparse
regression over a full tensor basis.
"""
from __future__ import annotations

import math
from typing import Dict, Sequence, Tuple

import numpy as np

from .distributions import InputModel, Uniform
from .indices import SobolIndices

__all__ = [
    "ISHIGAMI_A",
    "ISHIGAMI_B",
    "G_FUNCTION_A",
    "ishigami_input_model",
    "ishigami_function",
    "ishigami_analytic_indices",
    "g_function_input_model",
    "g_function",
    "g_function_analytic_indices",
    "ishigami_literature_values",
]

ISHIGAMI_A = 7.0
ISHIGAMI_B = 0.1
G_FUNCTION_A: Tuple[float, ...] = (0.0, 1.0, 4.5, 9.0, 99.0, 99.0, 99.0, 99.0)


def ishigami_input_model(a: float = ISHIGAMI_A, b: float = ISHIGAMI_B) -> InputModel:
    """``x_i ~ U(-pi, pi)`` for the three Ishigami inputs (order: x1, x2, x3)."""
    del a, b  # the distribution does not depend on the coefficients
    return InputModel(
        ["x1", "x2", "x3"],
        [Uniform(-math.pi, math.pi), Uniform(-math.pi, math.pi), Uniform(-math.pi, math.pi)],
    )


def ishigami_function(X: np.ndarray, a: float = ISHIGAMI_A, b: float = ISHIGAMI_B) -> np.ndarray:
    """Vectorised Ishigami function; ``X`` has shape ``(n, 3)``."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    if X.shape[1] != 3:
        raise ValueError(f"Ishigami needs 3 columns, got {X.shape[1]}")
    x1, x2, x3 = X[:, 0], X[:, 1], X[:, 2]
    return np.sin(x1) + a * np.sin(x2) ** 2 + b * x3**4 * np.sin(x1)


def ishigami_analytic_indices(
    a: float = ISHIGAMI_A, b: float = ISHIGAMI_B, *, ci_level: float = 0.95
) -> SobolIndices:
    """Exact Ishigami Sobol' indices from the closed-form variances.

    The literature values usually quoted (``S1 = 0.3139, 0.4424, 0``,
    ``ST = 0.5576, 0.4424, 0.2439``) are these numbers rounded to four decimals;
    the returned values are exact so that an absolute-error criterion is not
    contaminated by the rounding of the reference.
    """
    pi = math.pi
    V1 = 0.5 * (1.0 + b * pi**4 / 5.0) ** 2
    V2 = a * a / 8.0
    V13 = b * b * pi**8 * (1.0 / 18.0 - 1.0 / 50.0)
    V = V1 + V2 + V13
    S1 = np.array([V1 / V, V2 / V, 0.0])
    ST = np.array([(V1 + V13) / V, V2 / V, V13 / V])
    return SobolIndices(
        names=["x1", "x2", "x3"],
        S1=S1,
        ST=ST,
        variance=V,
        estimator="analytic",
        ci_level=ci_level,
        S2={"x1-x3": V13 / V},
        meta={"V1": V1, "V2": V2, "V13": V13, "a": float(a), "b": float(b)},
    )


def ishigami_literature_values() -> Dict[str, Sequence[float]]:
    """The rounded values published for ``a=7, b=0.1`` (used as a cross-check)."""
    return {
        "S1": (0.3139, 0.4424, 0.0),
        "ST": (0.5576, 0.4424, 0.2439),
    }


def g_function_input_model(a: Sequence[float] = G_FUNCTION_A) -> InputModel:
    """Independent ``U(0, 1)`` inputs for the Sobol' g-function."""
    return InputModel(
        [f"x{i + 1}" for i in range(len(a))],
        [Uniform(0.0, 1.0) for _ in a],
    )


def g_function(X: np.ndarray, a: Sequence[float] = G_FUNCTION_A) -> np.ndarray:
    """Sobol' g-function ``prod_i (|4 x_i - 2| + a_i)/(1 + a_i)``."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    av = np.asarray(a, dtype=float)
    if X.shape[1] != av.size:
        raise ValueError(f"g-function expects {av.size} columns, got {X.shape[1]}")
    return np.prod((np.abs(4.0 * X - 2.0) + av) / (1.0 + av), axis=1)


def g_function_analytic_indices(
    a: Sequence[float] = G_FUNCTION_A, *, ci_level: float = 0.95
) -> SobolIndices:
    """Exact Sobol' indices of the g-function (``V_i = (1/3)/(1+a_i)^2``)."""
    av = np.asarray(a, dtype=float)
    Vi = (1.0 / 3.0) / (1.0 + av) ** 2
    total = float(np.prod(1.0 + Vi) - 1.0)
    S1 = Vi / total
    ST = np.array([Vi[i] * np.prod(np.delete(1.0 + Vi, i)) / total for i in range(av.size)])
    return SobolIndices(
        names=[f"x{i + 1}" for i in range(av.size)],
        S1=S1,
        ST=ST,
        variance=total,
        estimator="analytic",
        ci_level=ci_level,
        meta={"a": [float(v) for v in av], "V_i": [float(v) for v in Vi]},
    )