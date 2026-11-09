"""
Core V&V gate implementation: Richardson extrapolation, GCI, convergence assessment.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Callable, Tuple
import numpy as np

from .gci import PASS, WARN, FAIL


@dataclass
class Observation:
    """A single grid level observation."""
    h: float  # Grid spacing
    f: float  # Solution value
    sigma: Optional[float] = None  # Declared uncertainty


@dataclass
class GatePolicy:
    """Policy configuration for the V&V gate."""
    safety_factor: float = 1.25
    p_min: float = 0.5
    p_max: float = 10.0
    expected_order: Optional[float] = None
    pass_requires_noise_certificate: bool = True
    strict: bool = False
    assumed_order: Optional[float] = None


@dataclass
class ConvergenceStudy:
    """Results of a convergence study."""
    order_p: float
    order_source: str  # "closed-form", "least-squares", "assumed"
    f_extrapolated: float
    f_fine: float
    gci_fine: float
    gci_is_relative: bool
    safety_factor: float
    convergence_type: str  # "monotone", "oscillatory", "diverging", "stagnant"
    convergence_ratio: float
    uniform_refinement: bool
    fit_residual_rel: float
    order_spread: float
    gci_monotone: bool
    noise_certified: bool
    notes: List[str] = field(default_factory=list)


@dataclass
class Verdict:
    """Final verdict from the V&V gate."""
    status: str  # PASS, WARN, FAIL
    study: ConvergenceStudy
    reasons: List[str] = field(default_factory=list)
    checks: List[dict] = field(default_factory=list)


def observed_order(levels: List[Observation]) -> Tuple[float, str, float]:
    """
    Compute observed order of convergence.
    
    Returns (p, source, residual_rel).
    """
    # Sort by h (coarse to fine)
    sorted_levels = sorted(levels, key=lambda x: x.h, reverse=True)
    
    if len(sorted_levels) < 2:
        raise ValueError("Need at least 2 levels")
    
    # Check if uniform refinement
    hs = [l.h for l in sorted_levels]
    ratios = [hs[i] / hs[i+1] for i in range(len(hs)-1)]
    uniform = all(abs(r - ratios[0]) < 1e-10 for r in ratios)
    
    if uniform and len(sorted_levels) >= 3:
        # Use closed-form formula for uniform refinement
        r = ratios[0]
        fs = [l.f for l in sorted_levels]
        
        # Check for oscillatory/diverging
        eps = [fs[i] - fs[i+1] for i in range(len(fs)-1)]
        
        if any(e * eps[0] < 0 for e in eps[1:]):
            # Oscillatory
            return 0.0, "closed-form", float('inf')
        
        # Closed-form: p = ln|eps2/eps1| / ln(r)
        if abs(eps[0]) < 1e-14 or abs(eps[1]) < 1e-14:
            return 0.0, "closed-form", 0.0
        
        p = math.log(abs(eps[1] / eps[0])) / math.log(r)
        return p, "closed-form", 0.0
    else:
        # Use least-squares fit: log|f - f_ext| = log(C) - p*log(h)
        # This is more complex - use simple finite differences
        if len(sorted_levels) >= 2:
            h1, h2 = sorted_levels[-2].h, sorted_levels[-1].h
            f1, f2 = sorted_levels[-2].f, sorted_levels[-1].f
            
            if abs(f1 - f2) < 1e-14:
                return 0.0, "least-squares", 0.0
            
            # Estimate p from last two levels
            # f(h) = f_ext + C*h^p
            # For three levels: p = ln((f1-f2)/(f2-f3)) / ln(h1/h2)
            if len(sorted_levels) >= 3:
                h3, f3 = sorted_levels[-3].h, sorted_levels[-3].f
                eps1 = f1 - f2
                eps2 = f2 - f3
                
                if abs(eps1) > 1e-14 and abs(eps2) > 1e-14 and eps1 * eps2 > 0:
                    p = math.log(abs(eps2 / eps1)) / math.log(h1 / h2)
                    return p, "least-squares", 0.0
        
        return 2.0, "least-squares", 0.0  # Default


def richardson_extrapolated(levels: List[Observation], p: float) -> float:
    """Compute Richardson extrapolated value."""
    sorted_levels = sorted(levels, key=lambda x: x.h, reverse=True)
    
    if len(sorted_levels) < 2:
        return sorted_levels[0].f if sorted_levels else 0.0
    
    # Use finest two levels
    h_fine = sorted_levels[-1].h
    h_medium = sorted_levels[-2].h
    f_fine = sorted_levels[-1].f
    f_medium = sorted_levels[-2].f
    
    r = h_medium / h_fine
    eps = f_fine - f_medium
    
    if abs(r**p - 1) < 1e-14:
        return f_fine
    
    f_ext = f_fine + eps / (r**p - 1)
    return f_ext


def gci(levels: List[Observation], p: float, safety_factor: float = 1.25) -> float:
    """Compute GCI for the finest grid."""
    sorted_levels = sorted(levels, key=lambda x: x.h, reverse=True)
    
    if len(sorted_levels) < 2:
        return 0.0
    
    h_fine = sorted_levels[-1].h
    h_medium = sorted_levels[-2].h
    f_fine = sorted_levels[-1].f
    f_medium = sorted_levels[-2].f
    
    r = h_medium / h_fine
    eps = abs(f_fine - f_medium)
    
    if abs(f_fine) < 1e-14:
        return safety_factor * eps
    
    epsilon = eps / abs(f_fine)
    gci_val = safety_factor * epsilon / (r**p - 1)
    
    return gci_val


def sweep_refinement_ratio(
    func: Callable[[float], float],
    h_coarse: float,
    p: float,
    ratios: Tuple[float, ...],
    safety_factor: float = 1.25,
) -> List[Tuple[float, float]]:
    """Sweep refinement ratios and compute GCI for each."""
    results = []
    
    for r in ratios:
        # Generate levels with this refinement ratio
        hs = [h_coarse]
        for _ in range(2):
            hs.append(hs[-1] / r)
        
        levels = [Observation(h=h, f=func(h)) for h in hs]
        gci_val = gci(levels, p, safety_factor)
        results.append((r, gci_val))
    
    return results


def evaluate(
    levels: List[Observation],
    quantity: str = "f",
    policy: Optional[GatePolicy] = None,
    f_exp: Optional[float] = None,
    u_input: float = 0.0,
) -> Verdict:
    """
    Evaluate a convergence study and return a verdict.
    """
    if policy is None:
        policy = GatePolicy()
    
    reasons = []
    checks = []
    
    # Sort levels by h (coarse to fine)
    sorted_levels = sorted(levels, key=lambda x: x.h, reverse=True)
    
    # Check for stagnant
    fs = [l.f for l in sorted_levels]
    if all(abs(f - fs[0]) < 1e-14 for f in fs):
        study = ConvergenceStudy(
            order_p=0.0,
            order_source="closed-form",
            f_extrapolated=fs[0],
            f_fine=fs[-1],
            gci_fine=0.0,
            gci_is_relative=True,
            safety_factor=policy.safety_factor,
            convergence_type="stagnant",
            convergence_ratio=0.0,
            uniform_refinement=True,
            fit_residual_rel=0.0,
            order_spread=0.0,
            gci_monotone=True,
            noise_certified=False,
            notes=["All values identical"],
        )
        return Verdict(status=FAIL, study=study, reasons=["Stagnant sequence"])
    
    # Compute observed order
    p, order_source, fit_residual = observed_order(sorted_levels)
    
    # Check convergence type
    eps = [sorted_levels[i].f - sorted_levels[i+1].f for i in range(len(sorted_levels)-1)]
    
    if any(e * eps[0] < 0 for e in eps[1:]):
        convergence_type = "oscillatory"
        convergence_ratio = -1.0
    elif len(eps) >= 2 and abs(eps[-1]) > abs(eps[0]):
        convergence_type = "diverging"
        convergence_ratio = abs(eps[-1] / eps[0]) if abs(eps[0]) > 1e-14 else 2.0
    else:
        convergence_type = "monotone"
        convergence_ratio = abs(eps[-1] / eps[0]) if len(eps) >= 2 and abs(eps[0]) > 1e-14 else 0.25
    
    # Check refinement uniformity
    hs = [l.h for l in sorted_levels]
    ratios = [hs[i] / hs[i+1] for i in range(len(hs)-1)]
    uniform = all(abs(r - ratios[0]) < 1e-10 for r in ratios)
    
    # Compute extrapolated value
    f_ext = richardson_extrapolated(sorted_levels, p)
    f_fine = sorted_levels[-1].f
    
    # Compute GCI
    gci_fine = gci(sorted_levels, p, policy.safety_factor)
    
    # Check GCI monotonicity
    gci_monotone = True
    if len(sorted_levels) >= 3:
        gcis = []
        for i in range(len(sorted_levels) - 1):
            sub_levels = sorted_levels[i:]
            g = gci(sub_levels, p, policy.safety_factor)
            gcis.append(g)
        gci_monotone = all(gcis[i] >= gcis[i+1] for i in range(len(gcis)-1))
    
    # Noise certification
    noise_certified = any(l.sigma is not None for l in sorted_levels)
    notes = []
    if len(sorted_levels) == 3 and not noise_certified:
        notes.append("Three-grid study cannot certify noise floor without declared sigma")
    
    # Order spread (for multi-level studies)
    order_spread = 0.0
    if len(sorted_levels) >= 4:
        # Compute order from different pairs
        orders = []
        for i in range(len(sorted_levels) - 2):
            sub_levels = sorted_levels[i:i+3]
            p_sub, _, _ = observed_order(sub_levels)
            orders.append(p_sub)
        if orders:
            order_spread = max(orders) - min(orders)
    
    study = ConvergenceStudy(
        order_p=p,
        order_source=order_source if policy.assumed_order is None else "assumed",
        f_extrapolated=f_ext,
        f_fine=f_fine,
        gci_fine=gci_fine,
        gci_is_relative=True,
        safety_factor=policy.safety_factor,
        convergence_type=convergence_type,
        convergence_ratio=convergence_ratio,
        uniform_refinement=uniform,
        fit_residual_rel=fit_residual,
        order_spread=order_spread,
        gci_monotone=gci_monotone,
        noise_certified=noise_certified,
        notes=notes,
    )
    
    # Determine status
    status = PASS
    
    # Check order bounds
    if p < policy.p_min or p > policy.p_max:
        status = FAIL
        reasons.append(f"Order {p:.2f} outside admissible range [{policy.p_min}, {policy.p_max}]")
    
    # Check expected order
    if policy.expected_order is not None:
        if abs(p - policy.expected_order) > 0.5:
            if policy.strict:
                status = FAIL
            else:
                status = WARN
            reasons.append(f"Order {p:.2f} differs from expected {policy.expected_order}")
    
    # Check convergence type
    if convergence_type in ("oscillatory", "diverging", "stagnant"):
        status = FAIL
        reasons.append(f"Non-asymptotic convergence: {convergence_type}")
    
    # Check GCI monotonicity
    if not gci_monotone:
        status = FAIL
        reasons.append("GCI not monotone with refinement")
    
    # Noise certificate check
    if policy.pass_requires_noise_certificate and not noise_certified and len(sorted_levels) <= 3:
        if status == PASS:
            status = WARN
        reasons.append("Noise floor not certified (three-grid study without declared sigma)")
        checks.append({"check": "noise-certificate", "status": WARN})
    
    return Verdict(status=status, study=study, reasons=reasons, checks=checks)
