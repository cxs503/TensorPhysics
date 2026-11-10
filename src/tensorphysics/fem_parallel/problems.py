"""Verification problems for the parallel FEM infrastructure.

Every case is defined by analytic fields so that the discretisation error can be
measured and the parallel solution can be compared against both a serial run and
the exact solution.

``poisson_manufactured``
    2-D Poisson, non-separable manufactured solution::

        u = sin(pi x) sin(pi y) + 0.5 x(1-x) y(1-y)
        f = 2 pi^2 sin sin + [x(1-x) + y(1-y)]

    The polynomial term matters: the pure sine mode is an *exact* eigenvector of
    the Q1 stiffness stencil on a uniform grid (the 9-point stencil is
    separable), so CG would converge in a single iteration and every solver
    comparison would be degenerate.
``poisson_unit_source``
    Poisson benchmark with the same manufactured data; used for the
    strong/weak scaling study because its Krylov dimension is ``O(1/h)``.
``heat_variable_k``
    ``-div(k grad T) = f`` with ``k = 1 + 0.5 sin(pi x)`` and the manufactured
    ``T``; ``f`` is evaluated analytically.
``elasticity_patch``
    Linear displacement field on a plane-strain patch: Q1 elements reproduce it
    exactly, so the parallel solution must match the analytic field to machine
    precision (assembly + Dirichlet bookkeeping audit).
``elasticity_body_force``
    Plane strain with a sinusoidal body force (no closed form); used for the
    parallel-vs-serial equivalence check and for a mesh-refinement study.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np

from .discretisation import ElementKernel, StructuredGrid

__all__ = ["Problem", "make_problem", "PROBLEMS"]


def _sin_sin(c):
    return np.sin(np.pi * c[:, 0]) * np.sin(np.pi * c[:, 1])


def _poly(c):
    x, y = c[:, 0], c[:, 1]
    return x * (1.0 - x) * y * (1.0 - y)


def _poly3(c):
    x, y, z = c[:, 0], c[:, 1], c[:, 2]
    return x * (1 - x) * y * (1 - y) * z * (1 - z)


def _lap_poly3(c):
    x, y, z = c[:, 0], c[:, 1], c[:, 2]
    return (-2.0 * y * (1 - y) * z * (1 - z)
            - 2.0 * x * (1 - x) * z * (1 - z)
            - 2.0 * x * (1 - x) * y * (1 - y))


@dataclass
class Problem:
    name: str
    kind: str
    grid: StructuredGrid
    kernel: ElementKernel
    ncomp: int
    exact: Optional[Callable[[np.ndarray], np.ndarray]] = None
    notes: str = ""

    def dirichlet(self, coords: np.ndarray) -> np.ndarray:
        """Boundary data ``(n, ncomp)`` (homogeneous when no exact field)."""
        if self.exact is None:
            return np.zeros((coords.shape[0], self.ncomp))
        v = np.asarray(self.exact(coords), dtype=float)
        if v.ndim == 1:
            v = v.reshape(-1, 1)
        return v.reshape(coords.shape[0], self.ncomp)


def make_problem(name: str, n: Tuple[int, ...] = (33, 33), lengths=None) -> Problem:
    """Instantiate a verification problem on an ``n``-node tensor grid."""
    grid = StructuredGrid(tuple(n), tuple(lengths) if lengths else None)
    if name == "poisson_manufactured":
        if grid.ndim != 2:
            raise ValueError("poisson_manufactured is 2-D")

        def u2(c):
            return _sin_sin(c) + 0.5 * _poly(c)

        def f2(c):
            x, y = c[:, 0], c[:, 1]
            return 2.0 * np.pi ** 2 * _sin_sin(c) + (x * (1.0 - x) + y * (1.0 - y))

        kernel = ElementKernel("poisson", 2, grid.h, {}, source=f2)
        return Problem(name, "poisson", grid, kernel, 1, exact=u2,
                       notes="non-separable manufactured Poisson solution")

    if name == "poisson_unit_source":
        if grid.ndim == 2:
            def u2(c):
                return _sin_sin(c) + 0.5 * _poly(c)

            def f2(c):
                x, y = c[:, 0], c[:, 1]
                return 2.0 * np.pi ** 2 * _sin_sin(c) + (x * (1.0 - x) + y * (1.0 - y))

            exact, src = u2, f2
        else:
            def u3(c):
                return (np.sin(np.pi * c[:, 0]) * np.sin(np.pi * c[:, 1])
                        * np.sin(np.pi * c[:, 2]) + 0.5 * _poly3(c))

            def f3(c):
                x, y, z = c[:, 0], c[:, 1], c[:, 2]
                return (3.0 * np.pi ** 2 * np.sin(np.pi * x) * np.sin(np.pi * y)
                        * np.sin(np.pi * z) + 0.5 * _lap_poly3(c))

            exact, src = u3, f3

        kernel = ElementKernel("poisson", grid.ndim, grid.h, {}, source=src)
        return Problem(name, "poisson", grid, kernel, 1, exact=exact,
                       notes="Poisson benchmark used for the scaling study")

    if name == "heat_variable_k":
        if grid.ndim != 2:
            raise ValueError("heat_variable_k is 2-D")

        def kfun(c):
            return 1.0 + 0.5 * np.sin(np.pi * c[:, 0])

        def uh(c):
            return _sin_sin(c) + 0.5 * _poly(c)

        def fh(c):
            x, y = c[:, 0], c[:, 1]
            s, co = np.sin(np.pi * x), np.cos(np.pi * x)
            sy = np.sin(np.pi * y)
            k = 1.0 + 0.5 * s
            kx = 0.5 * np.pi * co
            Tx = np.pi * co * sy + 0.5 * (1.0 - 2.0 * x) * y * (1.0 - y)
            Txx = -np.pi ** 2 * s * sy - y * (1.0 - y)
            Tyy = -np.pi ** 2 * s * sy - x * (1.0 - x)
            return -(kx * Tx + k * (Txx + Tyy))

        kernel = ElementKernel("heat", 2, grid.h, {}, source=fh, conductivity=kfun)
        return Problem(name, "heat", grid, kernel, 1, exact=uh,
                       notes="variable-conductivity diffusion, non-separable solution")

    if name == "elasticity_patch":
        if grid.ndim == 2:
            def up(c):
                x, y = c[:, 0], c[:, 1]
                return np.stack([1e-3 * x + 2e-3 * y, -1.5e-3 * x + 0.5e-3 * y], axis=1)
        else:
            def up(c):
                x, y, z = c[:, 0], c[:, 1], c[:, 2]
                return np.stack([1e-3 * x + 2e-3 * y, -1.5e-3 * x + 0.5e-3 * z,
                                 0.8e-3 * y - 0.4e-3 * z], axis=1)

        kernel = ElementKernel(
            "elasticity", grid.ndim, grid.h,
            {"E": 210e9, "nu": 0.3, "mode": "plane_strain"},
        )
        return Problem(name, "elasticity", grid, kernel, grid.ndim, exact=up,
                       notes="constant-strain patch test (Q1 exact)")

    if name == "elasticity_body_force":
        if grid.ndim != 2:
            raise ValueError("elasticity_body_force is 2-D")

        def fb(c):
            x, y = c[:, 0], c[:, 1]
            return 1e6 * np.stack([np.sin(np.pi * x) * np.sin(np.pi * y),
                                   x * (1.0 - x) * y * (1.0 - y)], axis=1)

        kernel = ElementKernel(
            "elasticity", 2, grid.h, {"E": 210e9, "nu": 0.3, "mode": "plane_strain"},
            source=fb,
        )
        return Problem(name, "elasticity", grid, kernel, 2, exact=None,
                       notes="plane-strain body force (parallel-vs-serial audit)")

    raise ValueError(f"unknown problem {name!r}; choose from {sorted(PROBLEMS)}")


PROBLEMS = (
    "poisson_manufactured",
    "poisson_unit_source",
    "heat_variable_k",
    "elasticity_patch",
    "elasticity_body_force",
)
