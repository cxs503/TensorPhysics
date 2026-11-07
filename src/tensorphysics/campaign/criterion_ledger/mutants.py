"""Mutant classes M1..M5: generic single-site AST rewrites.

These are *generic* mutation operators: they act on numeric literals and
comparisons of any Python module, with no knowledge of what the module
computes.  Their physical reading (documented, not encoded):

======  ================================================================  ============
class   rewrite                                                             reading
======  ================================================================  ============
M1      ``v -> v*(1+eps)``, ``eps in {0.01, 0.03, 0.15}``                  numeric perturbation
M2      ``v -> v*f``, ``f in {1e3, 1e-3, 3600, 1/3600}``                   unit / dimension mismatch
M3      ``v -> v*10``                                                      constant substitution
M4      ``v -> -v``                                                        sample disguise (sign)
M5      ``x <op> L  ->  x <op> x`` (L a numeric literal)                   reference replaced by the quantity itself (identity / self-reference)
======  ================================================================  ============

Site selection mirrors the upstream audit tool's conventions: numeric
literals only (bool excluded), ``0`` and ``1`` excluded (identity
elements, not constants), deterministic DFS pre-order.

M5 is the tautology operator: the reference value of a comparison is
replaced by a copy of the compared quantity, so the check compares the
quantity against itself.  A gate that survives M5 is self-certifying.
"""
from __future__ import annotations

import ast
import copy
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

#: M1 perturbation levels (fractional).
M1_EPS = (0.01, 0.03, 0.15)
#: M2 unit-conversion factors (m<->mm, s<->h and inverses).
M2_FACTORS = (1.0e3, 1.0e-3, 3600.0, 1.0 / 3600.0)
#: M3 substitution scale.
M3_SCALE = 10.0

NATIVE = "native"
M1 = "M1_numeric_perturb"
M2 = "M2_unit_mismatch"
M3 = "M3_constant_substitute"
M4 = "M4_sample_disguise"
M5 = "M5_reference_selfref"

ALL_CLASSES = (NATIVE, M1, M2, M3, M4, M5)


@dataclass
class Mutant:
    """One single-site mutant (same field layout as the upstream tool)."""

    op: str            # class id (M1..M5 or a native operator class)
    index: int         # site index inside the class
    line: int          # 1-based source line of the mutated node
    snippet: str       # source text of the mutated node (pre-mutation)
    patch: str = ""    # short description of the change
    killed: bool = False
    detail: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def name(self) -> str:
        return f"{self.op}[{self.index}]"


# ---------------------------------------------------------------------------
# site collection (deterministic DFS pre-order)
# ---------------------------------------------------------------------------

def _snip(src: str, node: ast.AST) -> str:
    try:
        return ast.get_source_segment(src, node) or ""
    except Exception:  # pragma: no cover - defensive
        return ""


def _is_numeric(node: ast.AST) -> bool:
    if not isinstance(node, ast.Constant):
        return False
    v = node.value
    return isinstance(v, (int, float)) and not isinstance(v, bool) and v not in (0, 1)


def _collect_numeric(tree: ast.AST) -> List[Tuple[int, ast.AST]]:
    sites: List[Tuple[int, ast.AST]] = []

    class _C(ast.NodeVisitor):
        def visit(self, node):  # noqa: N802 - visitor protocol
            if _is_numeric(node):
                sites.append((len(sites), node))
            self.generic_visit(node)

    _C().visit(tree)
    return sites


def _collect_selfref(tree: ast.AST) -> List[Tuple[int, ast.AST, int]]:
    """Comparisons ``x <op> L`` / ``L <op> x`` with a numeric literal.

    Returns ``(site_index, compare_node, literal_position)`` where
    ``literal_position`` indexes into ``[left] + comparators``.
    """
    out: List[Tuple[int, ast.AST, int]] = []

    class _C(ast.NodeVisitor):
        def visit_Compare(self, node):  # noqa: N802
            operands = [node.left] + list(node.comparators)
            for pos, _op in enumerate(node.ops):
                a, b = operands[pos], operands[pos + 1]
                if _is_numeric(a) and not _is_numeric(b):
                    out.append((len(out), node, pos))
                elif _is_numeric(b) and not _is_numeric(a):
                    out.append((len(out), node, pos + 1))
            self.generic_visit(node)

    _C().visit(tree)
    return out


def collect_numeric_sites(src: str) -> List[Tuple[int, ast.AST]]:
    """All mutable numeric-literal sites in deterministic DFS pre-order."""
    return _collect_numeric(ast.parse(src))


def collect_selfref_sites(src: str) -> List[Tuple[int, ast.AST, int]]:
    """All M5 sites (see :func:`_collect_selfref`)."""
    return _collect_selfref(ast.parse(src))


# ---------------------------------------------------------------------------
# rewriters (replace exactly one target node)
# ---------------------------------------------------------------------------

class _ReplaceNode(ast.NodeTransformer):
    def __init__(self, target: ast.AST, replacement: ast.AST):
        self.target = target
        self.replacement = replacement
        self.done = False

    def generic_visit(self, node):  # noqa: N802
        if node is self.target and not self.done:
            self.done = True
            return self.replacement
        return super().generic_visit(node)


def _scaled(value: float, factor: float) -> ast.AST:
    return ast.Constant(value * factor)


# ---------------------------------------------------------------------------
# the five classes
# ---------------------------------------------------------------------------

def enumerate_m1(src: str, eps: Optional[float] = None) -> List[Mutant]:
    """M1 numeric perturbation: ``v -> v*(1+eps)`` for each eps level."""
    eps_levels = (eps,) if eps is not None else M1_EPS
    out: List[Mutant] = []
    for idx, node in collect_numeric_sites(src):
        v = node.value
        for e in eps_levels:
            out.append(Mutant(
                op=M1, index=idx, line=node.lineno, snippet=_snip(src, node),
                patch=f"v={v!r} -> v*{1.0 + e:g} (eps={e:g})",
                extra={"value": v, "eps": e},
            ))
    return out


def enumerate_m2(src: str) -> List[Mutant]:
    """M2 unit / dimension mismatch: ``v -> v*f`` for each factor."""
    out: List[Mutant] = []
    for idx, node in collect_numeric_sites(src):
        v = node.value
        for f in M2_FACTORS:
            out.append(Mutant(
                op=M2, index=idx, line=node.lineno, snippet=_snip(src, node),
                patch=f"v={v!r} -> v*{f:g} (unit factor)",
                extra={"value": v, "factor": f},
            ))
    return out


def enumerate_m3(src: str) -> List[Mutant]:
    """M3 constant substitution: ``v -> v*10``."""
    out: List[Mutant] = []
    for idx, node in collect_numeric_sites(src):
        v = node.value
        out.append(Mutant(
            op=M3, index=idx, line=node.lineno, snippet=_snip(src, node),
            patch=f"v={v!r} -> v*{M3_SCALE:g} (constant substitute)",
            extra={"value": v, "scale": M3_SCALE},
        ))
    return out


def enumerate_m4(src: str) -> List[Mutant]:
    """M4 sample disguise: ``v -> -v`` (magnitude kept, sign flipped)."""
    out: List[Mutant] = []
    for idx, node in collect_numeric_sites(src):
        v = node.value
        out.append(Mutant(
            op=M4, index=idx, line=node.lineno, snippet=_snip(src, node),
                patch=f"v={v!r} -> -v (sample disguise)",
                extra={"value": v},
            ))
    return out


def enumerate_m5(src: str) -> List[Mutant]:
    """M5 reference self-reference: ``x <op> L  ->  x <op> x``."""
    out: List[Mutant] = []
    for idx, node, _pos in collect_selfref_sites(src):
        out.append(Mutant(
            op=M5, index=idx, line=node.lineno, snippet=_snip(src, node),
            patch="reference literal -> copy of compared quantity (self-ref)",
        ))
    return out


def apply_mutant(src: str, m: Mutant) -> str:
    """Apply one M1..M5 mutant to the module source (single parse)."""
    tree = ast.parse(src)
    if m.op == M5:
        sites = _collect_selfref(tree)
        _idx, node, pos = sites[m.index]
        operands = [node.left] + list(node.comparators)
        other = operands[pos - 1] if pos > 0 else operands[pos + 1]
        rep = ast.copy_location(copy.deepcopy(other), other)
        if pos == 0:
            node.left = rep
        else:
            node.comparators[pos - 1] = rep
        return ast.unparse(ast.fix_missing_locations(tree)) + "\n"
    sites = _collect_numeric(tree)
    _idx, target = sites[m.index]
    if m.op == M1:
        rep = _scaled(m.extra["value"], 1.0 + m.extra["eps"])
    elif m.op == M2:
        rep = _scaled(m.extra["value"], m.extra["factor"])
    elif m.op == M3:
        rep = _scaled(m.extra["value"], M3_SCALE)
    elif m.op == M4:
        rep = ast.Constant(-m.extra["value"])
    else:  # pragma: no cover - guarded by the registry
        raise ValueError(f"unknown mutant class: {m.op}")
    new_tree = _ReplaceNode(target, rep).visit(tree)
    return ast.unparse(ast.fix_missing_locations(new_tree)) + "\n"


#: class id -> enumerator (no-arg form; M1 uses all eps levels)
ENUMERATORS = {
    M1: enumerate_m1,
    M2: enumerate_m2,
    M3: enumerate_m3,
    M4: enumerate_m4,
    M5: enumerate_m5,
}
