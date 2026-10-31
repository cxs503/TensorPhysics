"""Small dense linear-algebra helpers used by the structural reliability core.

The reliability module deliberately depends on the Python standard library only,
so the few matrix operations it needs (Cholesky factorisation of a correlation
matrix, forward substitution, symmetric eigen-decomposition for SORM
curvatures) are implemented here for the small dimensions (n <= ~10) that
component reliability problems use.
"""
from __future__ import annotations

from math import hypot, sqrt
from typing import Sequence

Matrix = Sequence[Sequence[float]]
Vector = Sequence[float]


def zeros_like(matrix: Matrix) -> list[list[float]]:
    return [[0.0] * len(row) for row in matrix]


def transpose(matrix: Matrix) -> list[list[float]]:
    return [list(column) for column in zip(*matrix)]


def matmul(a: Matrix, b: Matrix) -> list[list[float]]:
    n, m, p = len(a), len(b), len(b[0])
    out = [[0.0] * p for _ in range(n)]
    for i in range(n):
        row = out[i]
        ai = a[i]
        for k in range(m):
            aik = ai[k]
            if aik == 0.0:
                continue
            bk = b[k]
            for j in range(p):
                row[j] += aik * bk[j]
    return out


def matvec(matrix: Matrix, vector: Vector) -> list[float]:
    return [sum(a * v for a, v in zip(row, vector)) for row in matrix]


def outer(n: Vector) -> list[list[float]]:
    return [[a * b for b in n] for a in n]


def identity(n: int) -> list[list[float]]:
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def cholesky(matrix: Matrix, tolerance: float = 1e-12) -> list[list[float]]:
    """Lower-triangular factor ``L`` with ``L L^T = matrix``.

    Raises ``ValueError`` when the matrix is not symmetric positive definite,
    which for a correlation matrix means the correlation structure is invalid.
    """
    n = len(matrix)
    if any(len(row) != n for row in matrix):
        raise ValueError("matrix must be square")
    lower = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = matrix[i][j]
            if abs(s - matrix[j][i]) > 1e-10:
                raise ValueError("matrix must be symmetric")
            s -= sum(lower[i][k] * lower[j][k] for k in range(j))
            if i == j:
                if s <= tolerance:
                    raise ValueError(
                        "matrix is not positive definite (correlation matrix must be "
                        "positive definite; check for |rho| = 1 or linearly dependent rows)"
                    )
                lower[i][i] = sqrt(s)
            else:
                lower[i][j] = s / lower[j][j]
    return lower


def forward_solve(lower: Matrix, rhs: Vector) -> list[float]:
    """Solve ``L y = rhs`` for lower-triangular ``L``."""
    n = len(lower)
    y = [0.0] * n
    for i in range(n):
        s = rhs[i]
        for k in range(i):
            s -= lower[i][k] * y[k]
        y[i] = s / lower[i][i]
    return y


def backward_solve(upper: Matrix, rhs: Vector) -> list[float]:
    """Solve ``U y = rhs`` for upper-triangular ``U``."""
    n = len(upper)
    y = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = rhs[i]
        for k in range(i + 1, n):
            s -= upper[i][k] * y[k]
        y[i] = s / upper[i][i]
    return y


def inverse_from_cholesky(lower: Matrix) -> list[list[float]]:
    """Inverse of ``L L^T`` given ``L`` (used for Nataf back-transformation)."""
    n = len(lower)
    inv = [[0.0] * n for _ in range(n)]
    for i in range(n):
        e = [0.0] * n
        e[i] = 1.0
        y = forward_solve(lower, e)
        upper = transpose(lower)
        column = backward_solve(upper, y)
        for row in range(n):
            inv[row][i] = column[row]
    return inv


def jacobi_eigen(
    matrix: Matrix, tolerance: float = 1e-14, max_sweeps: int = 100
) -> tuple[list[float], list[list[float]]]:
    """Cyclic Jacobi eigen-decomposition of a real symmetric matrix.

    Returns ``(eigenvalues, eigenvectors)`` where ``eigenvectors[k]`` is the
    unit eigenvector belonging to ``eigenvalues[k]``. Eigenvalues are returned
    in ascending order.
    """
    n = len(matrix)
    a = [[0.5 * (matrix[i][j] + matrix[j][i]) for j in range(n)] for i in range(n)]
    v = identity(n)
    for _ in range(max_sweeps):
        off = sqrt(sum(a[i][j] ** 2 for i in range(n) for j in range(n) if i != j))
        if off < tolerance:
            break
        for p in range(n - 1):
            for q in range(p + 1, n):
                apq = a[p][q]
                if abs(apq) < 1e-300:
                    continue
                theta = (a[q][q] - a[p][p]) / (2.0 * apq)
                t = (1.0 if theta >= 0 else -1.0) / (abs(theta) + hypot(1.0, theta))
                c = 1.0 / hypot(1.0, t)
                s = t * c
                for k in range(n):
                    akp, akq = a[k][p], a[k][q]
                    a[k][p] = c * akp - s * akq
                    a[k][q] = s * akp + c * akq
                for k in range(n):
                    apk, aqk = a[p][k], a[q][k]
                    a[p][k] = c * apk - s * aqk
                    a[q][k] = s * apk + c * aqk
                for k in range(n):
                    vkp, vkq = v[k][p], v[k][q]
                    v[k][p] = c * vkp - s * vkq
                    v[k][q] = s * vkp + c * vkq
    values = [a[i][i] for i in range(n)]
    vectors = [[v[i][j] for i in range(n)] for j in range(n)]
    order = sorted(range(n), key=lambda j: values[j])
    return [values[j] for j in order], [vectors[j] for j in order]


def tangent_basis(normal: Vector, max_attempts: int | None = None) -> list[list[float]]:
    """Orthonormal basis of the hyper-plane orthogonal to ``normal``.

    Modified Gram-Schmidt applied to the standard basis; the returned vectors
    are orthogonal to ``normal`` and to each other (dimension n-1).
    """
    n = len(normal)
    norm = sqrt(sum(x * x for x in normal))
    if norm <= 0.0:
        raise ValueError("normal vector must be non-zero")
    unit = [x / norm for x in normal]
    basis: list[list[float]] = []
    attempts = max_attempts if max_attempts is not None else n
    for axis in range(n):
        if len(basis) == n - 1:
            break
        candidate = [0.0] * n
        candidate[axis] = 1.0
        for v in basis:
            proj = sum(a * b for a, b in zip(candidate, v))
            for i in range(n):
                candidate[i] -= proj * v[i]
        proj = sum(a * b for a, b in zip(candidate, unit))
        for i in range(n):
            candidate[i] -= proj * unit[i]
        length = sqrt(sum(x * x for x in candidate))
        if length > 1e-8:
            basis.append([x / length for x in candidate])
    if len(basis) != n - 1:
        raise ValueError("failed to build tangent basis")
    del attempts
    return basis