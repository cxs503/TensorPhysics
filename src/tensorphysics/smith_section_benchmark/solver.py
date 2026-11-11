"""Fiber-based Smith-method solvers for a general plated section.

Two solvers share one element-curve kernel:

:class:`SmithSectionSolver`
    **pure bending** of the section.  For every curvature the axial strain at
    the geometric neutral axis, ``s0``, is the root of the axial equilibrium
    ``N(s0, kappa) = 0`` and the bending moment is
    ``M(kappa) = sum_j A_j sigma_j(z_j) (z_j - z_NA)`` (positive sagging).  The
    ultimate moment is the maximum of ``M(kappa)`` over the sweep, refined by a
    golden-section search in the neighbourhood of the discrete peak so that the
    value is independent of the curvature step.

:func:`axial_ultimate_strength`
    **uniform axial compression** of the section (a stiffened panel loaded
    end-to-end): ``sigma_avg(s) = sum_j A_j sigma_j(s) / A`` and
    ``sigma_u = max sigma_avg``.  This is the quantity compared with CSR-type
    stiffened-panel design curves.

Element load–end shortening curves (compression positive)
---------------------------------------------------------
* **Plating** (:func:`plate_element_stress`) — elastic up to the plate buckling
  stress ``sigma_cr``, von Karman post-buckling branch
  ``sqrt(sigma_cr E s)`` up to edge yielding, then a linear softening branch
  down to a residual plateau (this is what produces a limit point).
* **Stiffener** — J2 uniaxial elasto-plastic stress capped by the *element
  strength* (min of squash / weak-axis column / tripping, see
  :func:`section.stiffener_strength_caps`).
* **Tension side** — elastic up to yield, perfectly plastic afterwards (no
  buckling, no fracture).

Failure-mode attribution
------------------------
At the ultimate state every fiber is scored by its **utilisation**
``sigma_j / sigma_cap,j`` (compression fibers only).  The governing group is the
one with the highest utilisation; the mode follows from the governing element's
strength mechanism and slenderness — see :meth:`SmithSectionSolver.classify`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .section import SectionFiber, SectionModel

SATURATION_TOL = 0.95
SHARE_TOL = 0.05

__all__ = [
    "ElementArrays",
    "element_stress",
    "SmithSectionResult",
    "SmithSectionSolver",
    "AxialPanelResult",
    "axial_ultimate_strength",
]


# ---------------------------------------------------------------------------
# vectorised element-curve kernel
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ElementArrays:
    """Per-fiber parameter arrays driving the element-curve kernel."""

    area: np.ndarray
    z: np.ndarray
    sigma_y: np.ndarray
    sigma_cr: np.ndarray
    sigma_cap: np.ndarray
    is_plate: np.ndarray
    groups: Tuple[str, ...]
    group_index: np.ndarray
    cap_source: Tuple[str, ...]

    @property
    def n(self) -> int:
        return int(self.area.size)


def build_element_arrays(fibers: Sequence[SectionFiber]) -> ElementArrays:
    groups: List[str] = []
    idx: List[int] = []
    sources: List[str] = []
    for f in fibers:
        if f.group not in groups:
            groups.append(f.group)
        idx.append(groups.index(f.group))
        sources.append(f.cap_source)
    return ElementArrays(
        area=np.array([f.area for f in fibers], dtype=float),
        z=np.array([f.z for f in fibers], dtype=float),
        sigma_y=np.array([f.sigma_y for f in fibers], dtype=float),
        sigma_cr=np.array([f.sigma_cr for f in fibers], dtype=float),
        sigma_cap=np.array([f.sigma_cap for f in fibers], dtype=float),
        is_plate=np.array([f.kind == "plate" for f in fibers], dtype=bool),
        groups=tuple(groups),
        group_index=np.array(idx, dtype=int),
        cap_source=tuple(sources),
    )


def element_stress(
    s: np.ndarray,
    el: ElementArrays,
    E: float,
    softening_modulus_ratio: float = 0.02,
    residual_stress_ratio: float = 0.2,
) -> np.ndarray:
    """Element stress [Pa, compression positive] for shortening ``s`` [-].

    ``s`` may be an array of the same length as the fiber arrays, or a scalar
    broadcast over all fibers.  Both length and stiffness of the branches are
    vectorised so the whole section is evaluated in one pass (this is what makes
    the three-level grid-convergence study cheap enough to run in the test
    suite).
    """
    s = np.asarray(s, dtype=float)
    if s.ndim == 0:
        s = np.full(el.n, float(s))
    elif s.shape != (el.n,):
        raise ValueError("strain array length must match the fiber count")

    E = float(E)
    sig_el = E * s
    # The plateau strain s_p is the strain at which the element reaches its
    # strength:
    #   * post-buckling regime (sigma_cap >= sigma_cr): the von Karman branch
    #     sqrt(sigma_cr E s) reaches sigma_cap at s_p = sigma_cap^2/(sigma_cr E);
    #   * imperfection-limited regime (sigma_cap < sigma_cr): the plate reaches
    #     its strength elastically, s_p = sigma_cap/E.
    # Below s_p the stress is min(E s, sqrt(sigma_cr E s)) (elastic / von Karman),
    # above s_p it softens linearly to the residual plateau.  This keeps the curve
    # monotone and capped at the element strength in *both* regimes - the
    # "sigma_cap < sigma_cr" case is the normal one for a Faulkner-law plate whose
    # imperfection-inclusive strength lies below its elastic buckling stress.
    with np.errstate(invalid="ignore", divide="ignore"):
        s_buckle = np.where(el.sigma_cr > 0.0, el.sigma_cap ** 2 / (el.sigma_cr * E),
                            np.inf)
        s_plateau = np.minimum(el.sigma_cap / E, s_buckle)
        s_vk = np.sqrt(np.maximum(el.sigma_cr * E * np.minimum(s, s_plateau), 0.0))
        sig_post = np.maximum(
            el.sigma_cap - softening_modulus_ratio * E * (s - s_plateau),
            residual_stress_ratio * el.sigma_cap)
        sig_plate = np.where(s <= s_plateau, np.minimum(sig_el, s_vk), sig_post)
    slender = el.is_plate & (el.sigma_cr < el.sigma_y)
    # stiffener / stocky-plate branch: J2 stub stress capped at element strength
    sig_stub = np.minimum(np.minimum(sig_el, el.sigma_y), el.sigma_cap)
    sig_comp = np.where(slender, sig_plate, sig_stub)
    # tension: elastic then perfectly plastic, never buckling limited
    sig_ten = np.maximum(sig_el, -el.sigma_y)
    return np.where(s > 0.0, sig_comp, sig_ten)


# ---------------------------------------------------------------------------
# bending solver
# ---------------------------------------------------------------------------
@dataclass
class SmithSectionResult:
    """Result of a Smith-method bending analysis."""

    M_u: float                                  # ultimate moment [N*m], sign follows bending
    kappa_u: float                              # curvature at ultimate [1/m]
    moment_curvature: List[Tuple[float, float]]  # [(kappa, M)]
    failure_mode: str
    governing_group: str
    utilisation: Dict[str, float]               # max sigma/sigma_cap per group at M_u
    stress_at_ultimate: Dict[str, float]        # max compressive stress per group [Pa]
    z_neutral_axis: float
    converged: bool
    n_fibers: int
    diagnostics: Dict[str, object] = field(default_factory=dict)

    @property
    def M_u_MNm(self) -> float:
        return float(self.M_u / 1e6)


class SmithSectionSolver:
    """Incremental-curvature Smith solver for an arbitrary plated section.

    Parameters
    ----------
    section : SectionModel
    n_fibers : int
        Fibers through each vertical plating strip (mesh-refinement parameter).
    n_web_fibers : int
        Fibers through each stiffener web.
    n_kappa : int
        Number of curvature steps of the sweep used to bracket the peak.
    kappa_max : float or None
        Largest curvature of the sweep [1/m]; ``None`` derives it from the first
        yield curvature of the extreme compression fiber (``20 x`` that value).
    bending : str
        "sagging" (deck in compression, positive moment) or "hogging".
    """

    def __init__(
        self,
        section: SectionModel,
        n_fibers: int = 2,
        n_web_fibers: int = 2,
        n_kappa: int = 60,
        kappa_max: float | None = None,
        bending: str = "sagging",
        softening_modulus_ratio: float = 0.02,
        residual_stress_ratio: float = 0.2,
    ) -> None:
        if bending not in ("sagging", "hogging"):
            raise ValueError("bending must be 'sagging' or 'hogging'")
        if n_kappa < 8:
            raise ValueError("n_kappa must be >= 8")
        self.section = section
        self.bending = bending
        self.sign = 1.0 if bending == "sagging" else -1.0
        self.n_kappa = int(n_kappa)
        self.soft = float(softening_modulus_ratio)
        self.resid = float(residual_stress_ratio)
        self.fibers = section.fibers(n_fibers=n_fibers, n_web_fibers=n_web_fibers)
        self.el = build_element_arrays(self.fibers)
        self.E = float(section.material.E)
        self.z_na = float(section.neutral_axis)
        self.area_total = float(self.el.area.sum())
        self.kappa_max = float(kappa_max if kappa_max is not None
                               else self._default_kappa_max())

    # -- strain field ------------------------------------------------------
    def _strain(self, s0: float, kappa: float) -> np.ndarray:
        """Shortening field ``s(z) = s0 + sign*kappa (z - z_NA)``."""
        return s0 + self.sign * kappa * (self.el.z - self.z_na)

    def _axial_force(self, s0: float, kappa: float) -> float:
        sig = element_stress(self._strain(s0, kappa), self.el, self.E,
                             self.soft, self.resid)
        return float(np.dot(self.el.area, sig))

    def _moment(self, kappa: float, s0: float | None = None) -> Tuple[float, float]:
        """Moment [N*m] and neutral-axis offset strain at curvature ``kappa``.

        The returned moment is **signed** (positive sagging); the sign convention
        is fixed by ``self.sign`` so that a hogging analysis reports a negative
        moment.  The peak search always maximises ``|M|``.
        """
        if s0 is None:
            s0 = self._solve_s0(kappa)
        sig = element_stress(self._strain(s0, kappa), self.el, self.E,
                             self.soft, self.resid)
        m = float(np.dot(self.el.area * sig, self.el.z - self.z_na))
        return m, s0

    def _objective(self, kappa: float) -> float:
        """``|M|`` at curvature ``kappa`` — the quantity the sweep maximises."""
        return self.sign * self._moment(kappa)[0]

    def _solve_s0(self, kappa: float) -> float:
        """Root of the axial equilibrium ``N(s0, kappa) = 0`` (bisection)."""
        lo, hi = -0.05, 0.05
        f_lo = self._axial_force(lo, kappa)
        f_hi = self._axial_force(hi, kappa)
        if f_lo > 0.0 or f_hi < 0.0:
            raise RuntimeError(
                "axial equilibrium is not bracketed: the section cannot carry the "
                "imposed curvature within |s0| <= 5%")
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            f_mid = self._axial_force(mid, kappa)
            if f_mid < 0.0:
                lo = mid
            else:
                hi = mid
            if hi - lo < 1e-14:
                break
        return 0.5 * (lo + hi)

    def _default_kappa_max(self) -> float:
        z_top = float(self.el.z.max())
        depth = max(z_top - self.z_na, 1e-6)
        eps_y = float(self.el.sigma_y.min()) / self.E
        return float(20.0 * eps_y / depth)

    MAX_EXTENSIONS = 3

    # -- sweep -------------------------------------------------------------
    def solve(self) -> SmithSectionResult:
        # The sweep is extended (kappa_max doubled, at most MAX_EXTENSIONS times)
        # whenever the peak falls on the last sample, i.e. the section was still
        # gaining moment at the end of the sweep.  This makes the peak location -
        # and therefore M_u - independent of the guessed curvature range.
        path: List[Tuple[float, float]] = []
        ms: List[float] = []
        arr = np.zeros(1)
        for _attempt in range(self.MAX_EXTENSIONS + 1):
            kappas = np.linspace(self.kappa_max / self.n_kappa, self.kappa_max,
                                 self.n_kappa)
            path = []
            ms = []
            for kap in kappas:
                m, _ = self._moment(float(kap))
                path.append((float(kap), float(m)))
                ms.append(float(m))
            arr = np.array(ms)
            magnitudes = self.sign * arr
            i_peak = int(np.argmax(magnitudes))
            if i_peak < len(arr) - 1:
                break
            self.kappa_max *= 2.0
        kappas = np.array([k for k, _ in path])
        magnitudes = self.sign * arr
        i_peak = int(np.argmax(magnitudes))
        converged = 0 < i_peak < len(arr) - 1
        kappa_u, M_u = float(kappas[i_peak]), float(arr[i_peak])
        if converged:
            kappa_u, M_u = self._refine_peak(float(kappas[i_peak - 1]),
                                             float(kappas[i_peak + 1]))
        s0_u = self._solve_s0(kappa_u)
        sig_u = element_stress(self._strain(s0_u, kappa_u), self.el, self.E,
                               self.soft, self.resid)
        util, stress, mode_info = self._attribute(sig_u, arm=self.el.z - self.z_na)
        return SmithSectionResult(
            M_u=M_u,
            kappa_u=kappa_u,
            moment_curvature=path,
            failure_mode=mode_info["mode"],
            governing_group=mode_info["governing_group"],
            utilisation=util,
            stress_at_ultimate=stress,
            z_neutral_axis=self.z_na,
            converged=bool(converged),
            n_fibers=len(self.fibers),
            diagnostics={
                "bending": self.bending,
                "sigma_y_min": float(self.el.sigma_y.min()),
                "kappa_y_first_yield": float(self._default_kappa_max() / 20.0),
                "moment_at_first_yield": float(arr[0]),
                "peak_index": i_peak,
                "peak_is_interior": bool(converged),
                "plate_slenderness": sorted(
                    {f.group: round(f.beta, 4) for f in self.fibers
                     if f.kind == "plate"}.items()),
                "cap_sources": dict(zip(self.el.groups, self.el.cap_source)),
                "post_peak_drop": float(1.0 - arr[-1] / max(arr.max(), 1e-30)),
                **mode_info,
            },
        )

    def _refine_peak(self, lo: float, hi: float, n_iter: int = 60) -> Tuple[float, float]:
        """Golden-section maximisation of ``M(kappa)`` on ``[lo, hi]``."""
        gr = (np.sqrt(5.0) - 1.0) / 2.0
        a, b = lo, hi
        c = b - gr * (b - a)
        d = a + gr * (b - a)
        mc = self._objective(c)
        md = self._objective(d)
        for _ in range(n_iter):
            if b - a < 1e-12 * max(1.0, b):
                break
            if mc > md:
                b, d, md = d, c, mc
                c = b - gr * (b - a)
                mc = self._objective(c)
            else:
                a, c, mc = c, d, md
                d = a + gr * (b - a)
                md = self._objective(d)
        kap = 0.5 * (a + b)
        return float(kap), float(self._moment(kap)[0])

    # -- failure attribution ----------------------------------------------
    def _collapse_strains(self) -> Dict[str, float]:
        """Shortening at which each group's element reaches its own strength [-].

        For a plating group in the post-buckling regime the strength is reached at
        ``sigma_cap^2/(sigma_cr E)``; otherwise (stocky plating, stiffeners) at
        ``sigma_cap / E``.  This is the strain that defines "first element to
        collapse" in the Smith sense.
        """
        out: Dict[str, float] = {}
        for gi, g in enumerate(self.el.groups):
            m = self.el.group_index == gi
            cap = float(self.el.sigma_cap[m][0])
            scr = float(self.el.sigma_cr[m][0])
            is_plate = bool(self.el.is_plate[m][0])
            sy = float(self.el.sigma_y[m][0])
            e = self.E
            if is_plate and scr < sy:
                # post-buckling regime: the von Karman branch reaches the plate
                # strength at cap^2/(sigma_cr E)
                out[g] = float(cap ** 2 / (scr * e))
            else:
                # stocky plating / stiffener: the element reaches its strength at
                # the elastic strain cap/E
                out[g] = float(cap / e)
        return out

    def _attribute(self, sig: np.ndarray,
                   arm: np.ndarray | None = None) -> Tuple[Dict[str, float],
                                                           Dict[str, float],
                                                           Dict[str, object]]:
        """Max utilisation, governing mechanism and failure mode at the peak.

        Attribution rule (declared): among the *saturated* element groups
        (utilisation within 1% of the maximum) the governing group is the one
        with the **lowest element strength** — i.e. the weakest link that set the
        ceiling of the section capacity.  Two saturated groups whose strengths
        are within 2% of each other *and* whose mechanisms differ are reported as
        ``interactive``; two groups saturated at the yield cap are ``yielding``
        (general section yielding).  Without this "weakest saturated link" rule a
        panel whose plating and stiffener are both at their own caps would be
        mislabelled as interactive although the plating strength is what limits
        the panel.
        """
        comp = sig > 0.0
        util = np.where(comp, sig / self.el.sigma_cap, 0.0)
        if arm is None:
            arm = np.ones_like(sig)
        contribution = np.abs(self.el.area * sig * arm)
        util_by_group: Dict[str, float] = {}
        stress_by_group: Dict[str, float] = {}
        cap_by_group: Dict[str, float] = {}
        source_by_group: Dict[str, str] = {}
        share: Dict[str, float] = {}
        total_contrib = float(contribution.sum())
        for gi, g in enumerate(self.el.groups):
            mask = (self.el.group_index == gi) & comp
            gmask = self.el.group_index == gi
            share[g] = float(contribution[gmask].sum() / total_contrib) if total_contrib > 0 else 0.0
            sources = [self.el.cap_source[i] for i in np.where(gmask)[0]]
            source_by_group[g] = max(set(sources), key=sources.count)
            cap_by_group[g] = float(self.el.sigma_cap[gmask][0]) if np.any(gmask) else float("inf")
            if not np.any(mask):
                util_by_group[g] = 0.0
                stress_by_group[g] = 0.0
                continue
            util_by_group[g] = float(util[mask].max())
            stress_by_group[g] = float(sig[mask].max())
        ordered = sorted(util_by_group.items(), key=lambda kv: kv[1], reverse=True)
        u_max = ordered[0][1] if ordered else 0.0
        # "saturated" = effectively at the element limit at the section peak; the
        # 5% tolerance is needed because the plating of a panel is already just
        # past its own plate strength when the last element (the stiffener) yields
        # The classical Smith attribution: the *first* element to reach its own
        # ultimate (smallest collapse strain) sets the failure mode, but only
        # elements that actually carry the section (share of the resultant above
        # SHARE_TOL) are eligible - an element at its cap next to the neutral axis
        # or a small appendage does not govern the section.
        strain_ult = self._collapse_strains()
        priority = {g: util_by_group[g] * share[g] for g in util_by_group}
        saturated = [g for g, u in ordered if u > 0.0 and u >= SATURATION_TOL * u_max]
        eligible = [g for g in saturated if share[g] >= SHARE_TOL] or saturated
        eligible.sort(key=lambda g: (strain_ult[g], -share[g]))
        governing = eligible[0] if eligible else (ordered[0][0] if ordered else "")
        governing_strain = strain_ult.get(governing, float("nan"))
        second = saturated[1] if len(saturated) > 1 else (ordered[1][0], 0.0)[0:1][0] if len(ordered) > 1 else ""
        source = source_by_group.get(governing, "yield")
        sigma_cap = cap_by_group.get(governing, float("nan"))
        betas = [f.beta for f in self.fibers if f.group == governing]
        beta = float(max(betas)) if betas else float("nan")
        second_source = source_by_group.get(second, "") if second else ""
        caps_comparable = (second != "" and
                           abs(cap_by_group[second] - sigma_cap) <= 0.02 * sigma_cap)
        mode = self.classify(source=source, beta=beta,
                             u1=util_by_group.get(governing, 0.0),
                             u2=util_by_group.get(second, 0.0) if second else 0.0,
                             second_group=second,
                             second_source=second_source,
                             caps_comparable=caps_comparable)
        info = {
            "governing_group": governing,
            "moment_share": {g: round(v, 4) for g, v in share.items()},
            "collapse_strain": {g: round(v, 8) for g, v in strain_ult.items()},
            "governing_collapse_strain": governing_strain,
            "eligible_groups": eligible,
            "priority_score": {g: round(v, 4) for g, v in priority.items()},
            "governing_cap_source": source,
            "governing_beta": beta,
            "governing_sigma_cap": sigma_cap,
            "mode": mode,
            "saturated_groups": saturated,
            "second_group": second,
            "second_cap_source": second_source,
            "utilisation_ratio_first_second": float(
                util_by_group[governing] / util_by_group[second]) if second
            and util_by_group[second] > 0 else float("inf"),
        }
        return util_by_group, stress_by_group, info

    @staticmethod
    def classify(source: str, beta: float, u1: float, u2: float,
                 second_group: str | None = None,
                 second_source: str = "",
                 caps_comparable: bool = False) -> str:
        """Map the governing element mechanism to a failure-mode label.

        Rules (declared, see the report for the validation of the 3-case set):

        * governing element at the yield cap (and so is the runner-up) ->
          ``yielding`` (general section yielding; the whole section is plastic);
        * two eligible groups with comparable strengths and different mechanisms ->
          ``interactive``;
        * governing plating with ``beta > 1.25`` -> ``plate_buckling``;
        * governing stiffener capped by torsional strength ->
          ``stiffener_tripping``;
        * governing stiffener capped by the weak-axis column curve ->
          ``stiffener_column_buckling``.
        """
        if source == "yield" and second_source in ("yield", ""):
            # the governing element sits at its yield cap: the section collapses by
            # general yielding, not by an element instability
            return "yielding"
        if caps_comparable and second_source != source:
            return "interactive"
        if source == "stiffener_tripping":
            return "stiffener_tripping"
        if source == "stiffener_column":
            return "stiffener_column_buckling"
        if source == "plate_von_karman":
            return "plate_buckling"
        if beta is not None and np.isfinite(beta) and beta > 1.25:
            return "plate_buckling"
        return "yielding"


# ---------------------------------------------------------------------------
# axial (panel) solver
# ---------------------------------------------------------------------------
@dataclass
class AxialPanelResult:
    """Result of a uniform axial compression analysis of a panel section."""

    sigma_u: float
    eps_u: float
    sigma_y: float
    load_shortening: List[Tuple[float, float]]
    failure_mode: str
    governing_group: str
    utilisation: Dict[str, float]
    n_fibers: int
    diagnostics: Dict[str, object] = field(default_factory=dict)

    @property
    def sigma_u_over_sigma_y(self) -> float:
        return float(self.sigma_u / self.sigma_y)


def axial_ultimate_strength(
    section: SectionModel,
    n_fibers: int = 2,
    n_web_fibers: int = 2,
    n_steps: int = 200,
    max_strain: float = 0.02,
    softening_modulus_ratio: float = 0.02,
    residual_stress_ratio: float = 0.2,
) -> AxialPanelResult:
    """Ultimate average compressive stress of a stiffened panel section.

    The prescribed shortening is uniform over the section (pure axial load, no
    bending), ``sigma_avg(s) = sum A_j sigma_j(s) / A``; the ultimate strength is
    the peak of that curve.  This is the quantity that CSR-type design curves
    predict, so it is used for the stiffened-panel benchmark family.
    """
    fibers = section.fibers(n_fibers=n_fibers, n_web_fibers=n_web_fibers)
    el = build_element_arrays(fibers)
    E = float(section.material.E)
    area_total = float(el.area.sum())
    strains = np.linspace(max_strain / n_steps, max_strain, n_steps)
    curve: List[Tuple[float, float]] = []
    sigmas: List[float] = []
    for s in strains:
        sig = element_stress(float(s), el, E, softening_modulus_ratio,
                             residual_stress_ratio)
        sig_avg = float(np.dot(el.area, sig) / area_total)
        curve.append((float(s), sig_avg))
        sigmas.append(sig_avg)
    arr = np.array(sigmas)
    i_peak = int(np.argmax(arr))
    solver = SmithSectionSolver(section, n_fibers=n_fibers,
                                n_web_fibers=n_web_fibers, n_kappa=8)
    sig_at_peak = element_stress(float(strains[i_peak]), el, E,
                                 softening_modulus_ratio, residual_stress_ratio)
    util, _stress, info = solver._attribute(sig_at_peak, arm=np.ones(el.n))
    sigma_y = float(min(el.sigma_y.min(), np.inf))
    return AxialPanelResult(
        sigma_u=float(arr[i_peak]),
        eps_u=float(strains[i_peak]),
        sigma_y=sigma_y,
        load_shortening=curve,
        failure_mode=info["mode"],
        governing_group=info["governing_group"],
        utilisation=util,
        n_fibers=len(fibers),
        diagnostics={
            "sigma_cap_sources": dict(zip(el.groups, el.cap_source)),
            "sigma_y_by_group": {g: float(el.sigma_y[el.group_index == i].max())
                                 for i, g in enumerate(el.groups)},
            **info,
        },
    )