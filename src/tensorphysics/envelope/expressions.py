"""Domain-free expression trees: how a constraint model reaches the engine.

The envelope engine must never know what a parameter *means*.  Constraint
models therefore arrive as data -- a small expression tree over named
parameters -- and this module is the only place that turns such a tree into a
callable.  Names, spans, coefficients and tolerances are all injected by
configuration; there is no built-in physical vocabulary anywhere in the
package.

Operator table (``"op"`` key)
-----------------------------
``const``      ``{"op": "const", "value": v}``
``var``        ``{"op": "var", "name": "x"}`` -- resolved from the parameter map
``sum``        ``{"op": "sum", "terms": [...]}``
``product``    ``{"op": "product", "factors": [...]}``
``sub``        ``{"op": "sub", "a": ..., "b": ...}``
``div``        ``{"op": "div", "numerator": ..., "denominator": ...}`` (zero-guarded)
``reciprocal`` ``{"op": "reciprocal", "arg": ...}`` (zero-guarded)
``neg`` / ``scale`` / ``abs`` / ``square`` / ``sqrt``
``max`` / ``min``  ``{"op": "max", "args": [...]}``
``table``      ``{"op": "table", "arg": ..., "x": [...], "y": [...]}`` piecewise-linear,
               refuses to extrapolate outside the knot span.

Numerical policy
----------------
Every operator except ``table`` and ``sqrt`` is written with plain arithmetic
(``max(a, b)`` is ``(a + b + |a - b|) / 2``, ``sqrt`` dispatches on the operand
type).  A single code path therefore serves both scalar evaluation -- the
production path, pure standard library, no third-party dependency -- and
vectorised evaluation on an array-like operand, which is how the brute-force
cross-check in the acceptance script covers millions of points without
duplicating the constraint algebra.  ``table`` imports the array backend lazily
and only when it is handed a non-scalar operand.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Iterable, Mapping, Sequence, Set

__all__ = [
    "ExpressionError",
    "compile_expression",
    "evaluate_expression",
    "free_names",
    "OPERATORS",
    "validate_expression",
]


class ExpressionError(ValueError):
    """Raised for a malformed or non-evaluable expression tree."""


# ---------------------------------------------------------------------------
# arithmetic helpers that work on Python scalars *and* array-likes
# ---------------------------------------------------------------------------

def _is_scalar(value: Any) -> bool:
    return isinstance(value, (int, float))


def _sqrt(value: Any) -> Any:
    if _is_scalar(value):
        return math.sqrt(value)
    return value ** 0.5


def _amax(a: Any, b: Any) -> Any:
    return 0.5 * (a + b + abs(a - b))


def _amin(a: Any, b: Any) -> Any:
    return 0.5 * (a + b - abs(a - b))


def _guarded_reciprocal(value: Any) -> Any:
    if _is_scalar(value) and value == 0.0:
        raise ExpressionError("reciprocal of zero")
    return 1.0 / value


def _eval_table(arg: Any, xs: Sequence[float], ys: Sequence[float]) -> Any:
    if len(xs) != len(ys) or len(xs) < 2:
        raise ExpressionError("table needs matching x/y knots and at least two of them")
    for i in range(len(xs) - 1):
        if not xs[i] < xs[i + 1]:
            raise ExpressionError("table knots must be strictly increasing")
    lo, hi = xs[0], xs[-1]
    if not _is_scalar(arg):
        # array path: lazy backend import keeps the production path stdlib-only
        import numpy as np  # noqa: PLC0415

        out = np.asarray(arg, dtype=float)
        if np.any(out < lo) or np.any(out > hi):
            raise ExpressionError("table argument outside the knot span (no extrapolation)")
        return np.interp(out, np.asarray(xs, dtype=float), np.asarray(ys, dtype=float))
    value = float(arg)
    if value < lo or value > hi:
        raise ExpressionError("table argument outside the knot span (no extrapolation)")
    # binary search, then linear interpolation inside the bracketing segment
    left, right = 0, len(xs) - 1
    while right - left > 1:
        mid = (left + right) // 2
        if xs[mid] <= value:
            left = mid
        else:
            right = mid
    x0, x1 = xs[left], xs[right]
    if x1 == x0:
        return float(ys[left])
    weight = (value - x0) / (x1 - x0)
    return float(ys[left] + weight * (ys[right] - ys[left]))


def _reduce_pairwise(values: Iterable[Any], op: Callable[[Any, Any], Any]) -> Any:
    iterator = iter(values)
    try:
        acc = next(iterator)
    except StopIteration as exc:  # pragma: no cover - guarded by validate
        raise ExpressionError("empty argument list") from exc
    for item in iterator:
        acc = op(acc, item)
    return acc


# ---------------------------------------------------------------------------
# operator registry
# ---------------------------------------------------------------------------

def _op_const(spec: Mapping[str, Any], params: Mapping[str, Any]) -> Any:
    return float(spec["value"])


def _op_var(spec: Mapping[str, Any], params: Mapping[str, Any]) -> Any:
    name = spec["name"]
    try:
        return params[name]
    except KeyError as exc:
        raise ExpressionError(f"unknown parameter {name!r}") from exc


def _op_sum(spec: Mapping[str, Any], params: Mapping[str, Any]) -> Any:
    terms = spec["terms"]
    acc: Any = 0.0
    for term in terms:
        acc = acc + _eval(term, params)
    return acc


def _op_product(spec: Mapping[str, Any], params: Mapping[str, Any]) -> Any:
    acc: Any = 1.0
    for factor in spec["factors"]:
        acc = acc * _eval(factor, params)
    return acc


def _op_sub(spec: Mapping[str, Any], params: Mapping[str, Any]) -> Any:
    return _eval(spec["a"], params) - _eval(spec["b"], params)


def _op_div(spec: Mapping[str, Any], params: Mapping[str, Any]) -> Any:
    return _eval(spec["numerator"], params) * _guarded_reciprocal(
        _eval(spec["denominator"], params)
    )


def _op_reciprocal(spec: Mapping[str, Any], params: Mapping[str, Any]) -> Any:
    return _guarded_reciprocal(_eval(spec["arg"], params))


def _op_neg(spec: Mapping[str, Any], params: Mapping[str, Any]) -> Any:
    return -_eval(spec["arg"], params)


def _op_scale(spec: Mapping[str, Any], params: Mapping[str, Any]) -> Any:
    return float(spec["factor"]) * _eval(spec["arg"], params)


def _op_abs(spec: Mapping[str, Any], params: Mapping[str, Any]) -> Any:
    return abs(_eval(spec["arg"], params))


def _op_square(spec: Mapping[str, Any], params: Mapping[str, Any]) -> Any:
    value = _eval(spec["arg"], params)
    return value * value


def _op_sqrt(spec: Mapping[str, Any], params: Mapping[str, Any]) -> Any:
    return _sqrt(_eval(spec["arg"], params))


def _op_max(spec: Mapping[str, Any], params: Mapping[str, Any]) -> Any:
    return _reduce_pairwise((_eval(a, params) for a in spec["args"]), _amax)


def _op_min(spec: Mapping[str, Any], params: Mapping[str, Any]) -> Any:
    return _reduce_pairwise((_eval(a, params) for a in spec["args"]), _amin)


def _op_table(spec: Mapping[str, Any], params: Mapping[str, Any]) -> Any:
    return _eval_table(_eval(spec["arg"], params), spec["x"], spec["y"])


OPERATORS: Mapping[str, Callable[[Mapping[str, Any], Mapping[str, Any]], Any]] = {
    "const": _op_const,
    "var": _op_var,
    "sum": _op_sum,
    "product": _op_product,
    "sub": _op_sub,
    "div": _op_div,
    "reciprocal": _op_reciprocal,
    "neg": _op_neg,
    "scale": _op_scale,
    "abs": _op_abs,
    "square": _op_square,
    "sqrt": _op_sqrt,
    "max": _op_max,
    "min": _op_min,
    "table": _op_table,
}

_REQUIRED_KEYS: Mapping[str, tuple[str, ...]] = {
    "const": ("value",),
    "var": ("name",),
    "sum": ("terms",),
    "product": ("factors",),
    "sub": ("a", "b"),
    "div": ("numerator", "denominator"),
    "reciprocal": ("arg",),
    "neg": ("arg",),
    "scale": ("factor", "arg"),
    "abs": ("arg",),
    "square": ("arg",),
    "sqrt": ("arg",),
    "max": ("args",),
    "min": ("args",),
    "table": ("arg", "x", "y"),
}

_CHILD_KEYS: Mapping[str, tuple[str, ...]] = {
    "sum": ("terms",),
    "product": ("factors",),
    "sub": ("a", "b"),
    "div": ("numerator", "denominator"),
    "reciprocal": ("arg",),
    "neg": ("arg",),
    "scale": ("arg",),
    "abs": ("arg",),
    "square": ("arg",),
    "sqrt": ("arg",),
    "max": ("args",),
    "min": ("args",),
    "table": ("arg",),
}


def validate_expression(spec: Any, path: str = "expression") -> None:
    """Structural validation; raises :class:`ExpressionError` with a located path."""

    if not isinstance(spec, Mapping):
        raise ExpressionError(f"{path}: expected a mapping, got {type(spec).__name__}")
    op = spec.get("op")
    if op not in OPERATORS:
        raise ExpressionError(f"{path}: unknown op {op!r} (known: {sorted(OPERATORS)})")
    for key in _REQUIRED_KEYS[op]:
        if key not in spec:
            raise ExpressionError(f"{path}: op {op!r} missing key {key!r}")
    if op == "const":
        if not isinstance(spec["value"], (int, float)):
            raise ExpressionError(f"{path}: const value must be numeric")
    if op == "var":
        if not isinstance(spec["name"], str) or not spec["name"]:
            raise ExpressionError(f"{path}: var name must be a non-empty string")
    if op == "scale":
        if not isinstance(spec["factor"], (int, float)):
            raise ExpressionError(f"{path}: scale factor must be numeric")
    if op == "table":
        xs, ys = spec["x"], spec["y"]
        if not isinstance(xs, Sequence) or not isinstance(ys, Sequence):
            raise ExpressionError(f"{path}: table knots must be sequences")
        if len(xs) != len(ys) or len(xs) < 2:
            raise ExpressionError(f"{path}: table needs len(x) == len(y) >= 2")
        for i in range(len(xs) - 1):
            if not float(xs[i]) < float(xs[i + 1]):
                raise ExpressionError(f"{path}: table x knots must strictly increase")
    for key in _CHILD_KEYS.get(op, ()):
        child = spec[key]
        if isinstance(child, Sequence) and not isinstance(child, (str, bytes)):
            for index, item in enumerate(child):
                validate_expression(item, f"{path}.{key}[{index}]")
        else:
            validate_expression(child, f"{path}.{key}")


def free_names(spec: Mapping[str, Any]) -> Set[str]:
    """Parameter names an expression depends on."""

    found: Set[str] = set()

    def walk(node: Mapping[str, Any]) -> None:
        op = node["op"]
        if op == "var":
            found.add(node["name"])
            return
        for key in _CHILD_KEYS.get(op, ()):
            child = node[key]
            if isinstance(child, Sequence) and not isinstance(child, (str, bytes)):
                for item in child:
                    walk(item)
            else:
                walk(child)

    walk(spec)
    return found


def _eval(spec: Mapping[str, Any], params: Mapping[str, Any]) -> Any:
    try:
        handler = OPERATORS[spec["op"]]
    except KeyError as exc:
        raise ExpressionError(f"unknown op in {spec!r}") from exc
    return handler(spec, params)


def evaluate_expression(spec: Mapping[str, Any], params: Mapping[str, Any]) -> Any:
    """Evaluate ``spec`` on ``params``; scalar input gives scalar output."""

    return _eval(spec, params)


def compile_expression(spec: Mapping[str, Any]) -> Callable[[Mapping[str, Any]], Any]:
    """Validate once, return a callable -- the form the engine stores."""

    validate_expression(spec)
    frozen = spec

    def evaluate(params: Mapping[str, Any]) -> Any:
        return evaluate_expression(frozen, params)

    return evaluate