"""Analytic and domain UQ benchmarks used to certify the UQ workflow.

Three families, in increasing order of "no reference available":

``ishigami``
    The standard Ishigami function on ``[-pi, pi]^3`` with closed-form indices
    (Sobol' & Levitan 1999; Saltelli et al. 2008, eq. 4.19-4.25).  This is the
    primary anchor of the acceptance test: the analytic indices are known, so
    the polynomial-chaos and Saltelli routes can be scored *absolutely*.

``sobol_g``
    Sobol's ``G``-function, ``f = prod_i (|4 x_i - 2| + a_i) / (1 + a_i)`` on
    ``[0, 1]^d``, which has closed-form indices for any dimension and any
    coefficient vector ``a``.  Used as an *extensible family* regression test:
    raising ``a`` makes the non-additive (interaction) share arbitrary, so the
    same anchor exercises both near-additive and strongly interacting regimes.

``holtrop_type``
    A ship-resistance benchmark built by *calling* the existing Holtrop-1984
    kernel of this repository (read-only, no modification, no copy of its
    physics): main particulars and speed are uncertain, the quantity of
    interest is the calm-water total resistance ``R_T`` in kN.  No analytic
    indices exist for it, so it is certified by the PCE <-> Saltelli
    cross-check plus the surrogate LOO error, which is exactly the situation a
    real ship-design UQ run is in.

Naming note: the campaign node refers to a "Holton-type" benchmark; in this
repository the only matching reference model is the Holtrop-1984 resistance
kernel (:mod:`tensorphysics.physics_kernels_holtrop1984`), so that kernel is
used and the interpretation is recorded here rather than silently relabelled.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .sample_spec import Marginal, SampleSpec

__all__ = [
    "ISHIGAMI_QUOTED",
    "HoltropTypeBenchmark",
    "ishigami",
    "ishigami_analytic",
    "ishigami_spec",
    "holtrop_type_benchmark",
    "sobol_g",
    "sobol_g_analytic",
    "sobol_g_spec",
]

#: Values quoted in the campaign node (S1 = 0.3139, S2 = 0.4424, ST = 0.5574).
#: The closed forms below give 0.31391 / 0.44241 / 0.55759; the 4-decimal
#: campaign values are therefore consistent to < 2e-4 and both are reported.
ISHIGAMI_QUOTED = {"S1": 0.3139, "S2": 0.4424, "ST": 0.5574}


# ---------------------------------------------------------------------------
# Ishigami
# ---------------------------------------------------------------------------
def ishigami(points: np.ndarray, *, a: float = 7.0, b: float = 0.1) -> np.ndarray:
    """``f = sin(x1) + a sin^2(x2) + b x3^4 sin(x1)`` on ``[-pi, pi]^3``."""
    x = np.asarray(points, dtype=float)
    if x.ndim != 2 or x.shape[1] != 3:
        raise ValueError(f"ishigami expects (n, 3) points, got {x.shape}")
    return np.sin(x[:, 0]) + a * np.sin(x[:, 1]) ** 2 + b * x[:, 2] ** 4 * np.sin(x[:, 0])


def ishigami_spec() -> SampleSpec:
    """Uniform ``[-pi, pi]`` marginals for the three Ishigami inputs."""
    return SampleSpec(
        tuple(Marginal.uniform(f"x{i + 1}", -math.pi, math.pi) for i in range(3))
    )


def ishigami_analytic(*, a: float = 7.0, b: float = 0.1) -> dict[str, Any]:
    """Closed-form variance decomposition of :func:`ishigami`.

    With ``x_i ~ U(-pi, pi)``::

        V1  = 0.5 (1 + b pi^4 / 5)^2
        V2  = a^2 / 8
        V13 = b^2 pi^8 (1/18 - 1/50)
        V   = V1 + V2 + V13
    """
    pi = math.pi
    v1 = 0.5 * (1.0 + b * pi**4 / 5.0) ** 2
    v2 = a * a / 8.0
    v13 = b * b * pi**8 * (1.0 / 18.0 - 1.0 / 50.0)
    v = v1 + v2 + v13
    return {
        "variance": v,
        "first": np.array([v1 / v, v2 / v, 0.0]),
        "total": np.array([(v1 + v13) / v, v2 / v, v13 / v]),
        "second": {(0, 2): v13 / v, (0, 1): 0.0, (1, 2): 0.0},
        "components": {"V1": v1, "V2": v2, "V13": v13, "V": v},
        "quoted": dict(ISHIGAMI_QUOTED),
    }


# ---------------------------------------------------------------------------
# Sobol' G-function
# ---------------------------------------------------------------------------
def sobol_g(points: np.ndarray, a: np.ndarray | list[float]) -> np.ndarray:
    """Sobol' G-function on ``[0, 1]^d``."""
    x = np.asarray(points, dtype=float)
    a = np.asarray(a, dtype=float)
    if x.ndim != 2 or x.shape[1] != a.size:
        raise ValueError(f"sobol_g expects (n, {a.size}) points, got {x.shape}")
    if np.any(a < 0.0):
        raise ValueError("G-function coefficients must be non-negative")
    return np.prod((np.abs(4.0 * x - 2.0) + a) / (1.0 + a), axis=1)


def sobol_g_spec(dim: int) -> SampleSpec:
    """Independent uniform ``[0, 1]`` inputs for the G-function."""
    return SampleSpec(tuple(Marginal.uniform(f"x{i + 1}", 0.0, 1.0) for i in range(dim)))


def sobol_g_analytic(a: np.ndarray | list[float], *, order: int = 2) -> dict[str, Any]:
    """Closed-form G-function indices: ``V_i = 1/(3 (1+a_i)^2)``, ``V_u = prod V_i``."""
    a = np.asarray(a, dtype=float)
    if np.any(a < 0.0):
        raise ValueError("G-function coefficients must be non-negative")
    v = 1.0 / (3.0 * np.square(1.0 + a))
    total_variance = float(np.prod(1.0 + v) - 1.0)
    dim = a.size
    first = v / total_variance
    tot = np.empty(dim)
    for i in range(dim):
        others = np.prod(1.0 + np.delete(v, i))
        tot[i] = v[i] * others / total_variance
    second: dict[tuple[int, int], float] = {}
    if order >= 2:
        for i in range(dim):
            for j in range(i + 1, dim):
                second[(i, j)] = float(v[i] * v[j] / total_variance)
    return {
        "variance": total_variance,
        "first": first,
        "total": tot,
        "second": second,
        "components": v,
    }


# ---------------------------------------------------------------------------
# Holtrop-1984 ship-resistance benchmark
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class HoltropTypeBenchmark:
    """Ship-resistance UQ benchmark: uncertain particulars -> ``R_T`` [kN]."""

    spec: SampleSpec
    resistance_kn: Callable[[np.ndarray], np.ndarray]
    reference_case: dict[str, float]
    notes: str = ""

    @property
    def dim(self) -> int:
        return self.spec.dim


def holtrop_type_benchmark(
    *,
    length_m: float = 180.0,
    beam_m: float = 30.0,
    draft_m: float = 11.0,
    cb: float = 0.62,
    speed_kn: float = 14.0,
    rel_uncertainty: float = 0.01,
    seed_scale: float = 1.0,
) -> HoltropTypeBenchmark:
    """Build the ship-resistance benchmark around the Holtrop-1984 kernel.

    Five inputs are uncertain with independent uniform distributions centred on
    the reference case, each spanning ``+-rel_uncertainty`` of the base value
    (``rel_uncertainty=0.01`` means +-1%).  The kernel is called exactly as
    shipped -- no physics is re-implemented here and nothing in
    :mod:`tensorphysics.physics_kernels_holtrop1984` is modified.

    Parameters
    ----------
    speed_kn:
        Reference speed in knots (converted to m/s internally).
    rel_uncertainty:
        Half-width of every marginal as a fraction of its reference value.
    seed_scale:
        Multiplier applied to ``rel_uncertainty`` for the *draft* and ``Cb``
        marginals, used by the acceptance run to demonstrate the FAIL branch
        with a genuinely wider error band.
    """
    from ..physics_kernels_holtrop1984 import (  # local import: keep module light
        HullGeometry,
        estimate_cm,
        estimate_cwp,
        holtrop1984_resistance,
    )

    if rel_uncertainty <= 0.0:
        raise ValueError("rel_uncertainty must be positive")
    base = {
        "L": float(length_m),
        "B": float(beam_m),
        "T": float(draft_m),
        "Cb": float(cb),
        "Vs": float(speed_kn),
    }
    widths = {
        "L": rel_uncertainty * base["L"],
        "B": rel_uncertainty * base["B"],
        "T": rel_uncertainty * seed_scale * base["T"],
        "Cb": rel_uncertainty * seed_scale * base["Cb"],
        "Vs": rel_uncertainty * base["Vs"],
    }
    spec = SampleSpec(
        (
            Marginal.uniform("L", base["L"] - widths["L"], base["L"] + widths["L"]),
            Marginal.uniform("B", base["B"] - widths["B"], base["B"] + widths["B"]),
            Marginal.uniform("T", base["T"] - widths["T"], base["T"] + widths["T"]),
            Marginal.uniform("Cb", base["Cb"] - widths["Cb"], base["Cb"] + widths["Cb"]),
            Marginal.uniform("Vs", base["Vs"] - widths["Vs"], base["Vs"] + widths["Vs"]),
        )
    )

    def resistance_kn(points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 5:
            raise ValueError(f"expected (n, 5) points, got {pts.shape}")
        out = np.empty(pts.shape[0])
        for k, (loa, beam, draught, block, knots) in enumerate(pts):
            cm = estimate_cm(float(block))
            cwp = estimate_cwp(float(block))
            geom = HullGeometry(
                name="uq_benchmark",
                l=float(loa),
                b=float(beam),
                t=float(draught),
                cb=float(block),
                cm=float(cm),
                cwp=float(cwp),
            )
            res = holtrop1984_resistance(geom, float(knots) * 0.514444)
            out[k] = res.rt / 1000.0
        return out

    reference_points = np.array(
        [[base["L"], base["B"], base["T"], base["Cb"], base["Vs"]]]
    )
    reference_rt = float(resistance_kn(reference_points)[0])
    return HoltropTypeBenchmark(
        spec=spec,
        resistance_kn=resistance_kn,
        reference_case={**base, "RT_kN": reference_rt},
        notes=(
            "Holtrop-1984 kernel used read-only; inputs independent uniform; "
            "QoI = calm-water total resistance R_T [kN]; "
            "no analytic Sobol reference exists, cross-checked PCE vs Saltelli."
        ),
    )