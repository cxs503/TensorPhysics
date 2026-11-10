"""Node acceptance surface ``IN157-N05``: SIMP density filtering vs. checkerboard modes.

The node asks for a *checkerboard-focused* acceptance of the density filter on
the MBB beam (100x50, volume fraction 0.5, SIMP penalty p = 3):

===========  ==================================================================
criterion    statement
===========  ==================================================================
C1           ``C_no_filter / C_filter >= 5``
C2           checkerboard index (alternating 2x2 cells / total cells):
             ``> 0.3`` without filter, ``< 0.05`` with filter (r = 1.5)
C3           radius scan ``r in [1.0, 3.0]``: compliance monotone in ``r``,
             Spearman ``rho > 0.9``
C4           negative control ``r -> 0``: the filtered run degrades onto the
             unfiltered run (deviation ``<= 1 %``)
C5           ``>= 8`` pytest cases covering the standard case, the
             checkerboard detector, the radius scan and the negative control
===========  ==================================================================

C1 cannot hold for a compliance-minimisation problem, and this module says so
instead of redefining the criterion away:

* filtered physical fields form the *linear image* ``{P rho : rho in [0,1]^n}``,
  which is a subset of the admissible densities ``[0,1]^n`` of the unfiltered
  problem (``P`` has unit row sums and non-negative entries, so ``P rho`` stays
  in the box);
* hence the unfiltered optimum is a minimum over a *superset* and
  ``C_no_filter <= C_filter``, i.e. the ratio is bounded by 1 from above.

The measured ratio is therefore reported as an honest FAIL for the literal
``>= 5`` threshold, together with the quantity the threshold presumably aimed
at -- the checkerboard *suppression factor* (unfiltered index / filtered index),
which does exceed 5, and the checkerboard index itself (C2).

Detectors
---------
``alternating_cell_fraction``
    Literal campaign detector: a cell counts as alternating if it belongs to a
    2x2 element block whose two diagonals agree and whose off-diagonals differ
    (contrast ``>= band``). Index = alternating cells / total cells, so a pure
    checkerboard scores 1 and a smooth/constant field scores 0.
``density_filter.checkerboard_index``
    The continuous companion diagnostic (mean absolute mixed second difference
    normalised by its maximum amplitude) from the filter module; it reacts to
    short-wavelength roughness even when the field is grey, so both numbers are
    recorded for the same designs.

Everything the verdict uses is computed here and written to ``verification.json``
from the computed fields (no hand-transcribed numbers).
"""

from __future__ import annotations

import ast
import concurrent.futures
import json
import multiprocessing as mp
import pathlib
import time
from typing import Any, Iterable, Sequence

import numpy as np

from .density_filter import DensityFilter, checkerboard_index
from .problems import build_mbb_beam, evaluate_binary_design, run_simp

__all__ = [
    "MBB_SETTINGS",
    "RADIUS_SCAN",
    "alternating_cell_fraction",
    "spearman_rho",
    "count_test_cases",
    "run_acceptance",
]

MBB_SETTINGS: dict[str, Any] = {
    "problem": "mbb_beam",
    "nelx": 100,
    "nely": 50,
    "volfrac": 0.5,
    "penal": 3.0,
    "radius": 1.5,
}
RADIUS_SCAN: tuple[float, ...] = (1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0)
_SEED_AMPLITUDE = 0.1  # +-5 % symmetric break of the uniform start


# --------------------------------------------------------------------------
# detectors
# --------------------------------------------------------------------------
def alternating_cell_fraction(
    rho: np.ndarray, *, threshold: float = 0.5, band: float = 0.5
) -> float:
    """Fraction of cells sitting in an alternating 2x2 block.

    Parameters
    ----------
    rho
        ``(nely, nelx)`` element-density field.
    threshold
        Level used when the field is grey: the local contrast test below also
        separates values across this level, so binary *and* grey designs are
        handled by the same rule.
    band
        Minimum contrast ``|rho_a - rho_b|`` for two neighbours to count as
        "differing" (default 0.5: solid vs. void on a 0-1 field).

    Returns
    -------
    float in ``[0, 1]``.  A perfect checkerboard returns 1.0, uniform or
    layered (stripe) fields return values below 1 because a stripe has no
    diagonal agreement (``rho_ij == rho_i+1,j+1`` fails).

    Notes
    -----
    A block is alternating iff both diagonals agree and the other pair differs,
    i.e. ``|r00 - r11| < band``, ``|r01 - r10| < band`` and
    ``|r00 - r01| >= band``.  Every cell of a qualifying block is flagged, and
    the index is the flagged fraction -- so it is insensitive to how many
    qualifying blocks overlap a given cell.
    """
    r = np.asarray(rho, dtype=float)
    if r.ndim != 2:
        raise ValueError("alternating_cell_fraction expects an (nely, nelx) array")
    if r.shape[0] < 3 or r.shape[1] < 3:
        raise ValueError("checkerboard detector needs at least a 3x3 element grid")
    if not 0.0 < band <= 1.0:
        raise ValueError("band must lie in (0, 1]")

    a = r[:-1, :-1]
    b = r[1:, 1:]
    c = r[:-1, 1:]
    d = r[1:, :-1]
    block = (np.abs(a - b) < band) & (np.abs(c - d) < band) & (np.abs(a - c) >= band)
    if not block.any():
        return 0.0
    mask = np.zeros_like(r, dtype=bool)
    mask[:-1, :-1] |= block
    mask[1:, 1:] |= block
    mask[:-1, 1:] |= block
    mask[1:, :-1] |= block
    return float(mask.mean())


def _rank(values: Iterable[float]) -> np.ndarray:
    """Average ranks (1-based) with ties handled."""
    v = np.asarray(list(values), dtype=float).ravel()
    n = v.size
    order = np.argsort(v, kind="mergesort")
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(1, n + 1, dtype=float)
    sv = v[order]
    i = 0
    while i < n:
        j = i + 1
        while j < n and sv[j] == sv[i]:
            j += 1
        if j - i > 1:
            ranks[order[i:j]] = 0.5 * (i + 1 + j)
        i = j
    return ranks


def spearman_rho(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman rank correlation (ties averaged); 0.0 for a constant sample."""
    rx = _rank(x)
    ry = _rank(y)
    if rx.size != ry.size:
        raise ValueError("samples must have equal length")
    if rx.size < 2:
        raise ValueError("Spearman correlation needs at least two samples")
    sx = rx - rx.mean()
    sy = ry - ry.mean()
    denom = float(np.sqrt((sx * sx).sum() * (sy * sy).sum()))
    if denom == 0.0:
        return 0.0
    return float((sx * sy).sum() / denom)


def count_test_cases(pattern: str = "tests/test_simp_in157*.py") -> int:
    """Count ``def test_*`` functions in the IN157 test module(s) via AST."""
    total = 0
    for path in sorted(pathlib.Path(".").glob(pattern)):
        tree = ast.parse(path.read_text())
        total += sum(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name.startswith("test_")
            for node in ast.walk(tree)
        )
    return total


# --------------------------------------------------------------------------
# optimisation jobs
# --------------------------------------------------------------------------
def _run_job(job: tuple) -> dict[str, Any]:
    """One SIMP optimisation (module-level so that it pickles under spawn)."""
    tag, seed, radius, nelx, nely, volfrac, penal, iters, tol = job
    prob = build_mbb_beam(nelx, nely, volfrac)
    rho0 = None
    if seed is not None:
        rng = np.random.default_rng(seed)
        rho0 = volfrac + _SEED_AMPLITUDE * (rng.random(prob.n_elems) - 0.5)
    t0 = time.time()
    res = run_simp(prob, radius=radius, penal=penal, iters=iters, tol=tol, rho0=rho0)
    c_binary, solid = evaluate_binary_design(prob, res["rho"])
    raw = res["rho"].reshape(prob.mesh.shape)
    phys = res["rho_physical"].reshape(prob.mesh.shape)
    return {
        "tag": tag,
        "seed": seed,
        "radius": 0.0 if radius is None else float(radius),
        "filtered": radius is not None,
        "nelx": nelx,
        "nely": nely,
        "volfrac": volfrac,
        "penal": penal,
        "iters": iters,
        "tol": tol,
        "iterations": res["iterations"],
        "converged": res["converged"],
        "compliance": float(res["compliance"]),
        "compliance_binary": float(c_binary) if c_binary is not None else None,
        "solid_fraction_binary": float(np.mean(solid)),
        "alternating_fraction_raw": alternating_cell_fraction(raw),
        "alternating_fraction_physical": alternating_cell_fraction(phys),
        "checkerboard_index_raw": float(checkerboard_index(raw)),
        "checkerboard_index_physical": float(checkerboard_index(phys)),
        "grey_fraction": float(res["grey_fraction"]),
        "wall_seconds": float(time.time() - t0),
        "history": res["history"],
        "rho": res["rho"],
        "rho_physical": res["rho_physical"],
    }


def _parallel(jobs: Sequence[tuple], workers: int) -> list[dict[str, Any]]:
    if workers <= 1:
        return [_run_job(job) for job in jobs]
    ctx = mp.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
        return list(pool.map(_run_job, jobs))


def _strip(rec: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in rec.items() if k not in {"rho", "rho_physical"}}


# --------------------------------------------------------------------------
# acceptance driver
# --------------------------------------------------------------------------
def run_acceptance(
    outdir: str | pathlib.Path | None = None,
    *,
    workers: int | None = None,
    iters: int = 300,
    tol: float = 5e-3,
    seeds: Sequence[int] = (0, 1),
    radii: Sequence[float] = RADIUS_SCAN,
    negative_radius: float = 1e-12,
    nelx: int | None = None,
    nely: int | None = None,
    quick: bool = False,
) -> dict[str, Any]:
    """Run C1-C5 end to end and return the machine-readable record."""
    t_start = time.time()
    workers = workers or min(8, mp.cpu_count() or 1)
    nx = int(nelx if nelx is not None else (12 if quick else MBB_SETTINGS["nelx"]))
    ny = int(nely if nely is not None else (6 if quick else MBB_SETTINGS["nely"]))
    volfrac = MBB_SETTINGS["volfrac"]
    penal = MBB_SETTINGS["penal"]
    r_ref = MBB_SETTINGS["radius"]
    if quick:
        iters = min(iters, 8)
        radii = tuple(r for r in radii if r <= 2.5)[:3] or (1.0, r_ref)

    base = (nx, ny, volfrac, penal, iters, tol)
    jobs: list[tuple] = []
    for s in seeds:
        jobs.append(("unfiltered", s, None, *base))
    for s in seeds[:1]:
        jobs.append((f"filtered_r{r_ref:g}", s, r_ref, *base))
    for r in radii:
        jobs.append((f"scan_r{r:g}", seeds[0], float(r), *base))
    jobs.append(("negative_control", seeds[0], float(negative_radius), *base))

    print(f"[in157] {len(jobs)} SIMP runs on {nx}x{ny} ({workers} worker(s)) ...", flush=True)
    results = _parallel(jobs, workers)
    print(f"[in157] optimisation finished in {time.time() - t_start:.1f}s", flush=True)

    by_tag = {rec["tag"]: rec for rec in results}
    unf = [rec for rec in results if rec["tag"] == "unfiltered"]
    fil = by_tag[f"filtered_r{r_ref:g}"]
    scan = [by_tag[f"scan_r{r:g}"] for r in radii]
    neg = by_tag["negative_control"]

    record: dict[str, Any] = {
        "node": "IN157-N05",
        "module": "tensorphysics.topology_optimization.simp",
        "problem": dict(MBB_SETTINGS, nelx=nx, nely=ny, iters=iters, tol=tol, seeds=list(seeds)),
        "radius_scan": list(radii),
        "negative_control_radius": float(negative_radius),
        "detectors": {
            "alternating_cell_fraction": (
                "alternating 2x2 cells / total cells (band 0.5, threshold 0.5)"
            ),
            "checkerboard_index": (
                "mean |mixed second difference| / 2 on the physical density field"
            ),
        },
        "runs": [_strip(rec) for rec in results],
        "criteria": {},
    }

    # ------------------------------------------------------------------ C1
    detector_floor = 1.0 / float(nx * ny)  # alternating index resolution
    ci_unf = [rec["alternating_fraction_raw"] for rec in unf]
    ci_fil = fil["alternating_fraction_physical"]
    mean_alt_unf = float(np.mean(ci_unf))
    mean_ci_unf = float(np.mean([rec["checkerboard_index_raw"] for rec in unf]))
    c_unf = [rec["compliance"] for rec in unf]
    ratio = c_unf[0] / fil["compliance"]
    ratios = [c / fil["compliance"] for c in c_unf]
    record["criteria"]["C1"] = {
        "statement": "C_no_filter / C_filter >= 5",
        "compliance_no_filter": c_unf,
        "compliance_filtered": fil["compliance"],
        "ratio": ratio,
        "ratios_per_seed": ratios,
        "ratio_mean": float(np.mean(ratios)),
        "threshold": 5.0,
        "status": "PASS" if float(np.mean(ratios)) >= 5.0 else "FAIL",
        "theoretical_bound": (
            "C_no_filter / C_filter <= 1: filtered physical fields P rho (unit row "
            "sums, non-negative weights) stay inside [0,1]^n, so the filtered "
            "problem minimises over a subset of the unfiltered feasible set."
        ),
        "note": (
            "The >= 5 threshold is not attainable for compliance minimisation; the "
            "measured ratio is recorded as an honest FAIL and the quantity the "
            "criterion presumably targets is reported as checkerboard_suppression "
            "below (and C2)."
        ),
        "checkerboard_suppression": float(
            mean_alt_unf / max(ci_fil, detector_floor)
        ),
        "checkerboard_suppression_note": (
            "unfiltered index / filtered index; the denominator is floored at the "
            "detector resolution 1/(nelx*nely) because the index is a cell count "
            "and can be exactly zero"
        ),
        "compliance_ratio_binary_designs": (
            float(np.mean([rec["compliance_binary"] for rec in unf]) / fil["compliance_binary"])
            if fil["compliance_binary"]
            else None
        ),
    }

    # ------------------------------------------------------------------ C2
    c2_ok = mean_alt_unf > 0.3 and ci_fil < 0.05
    record["criteria"]["C2"] = {
        "statement": "checkerboard index > 0.3 (no filter), < 0.05 (r = 1.5)",
        "index_no_filter": ci_unf,
        "index_no_filter_mean": mean_alt_unf,
        "index_filtered": ci_fil,
        "threshold_no_filter": 0.3,
        "threshold_filtered": 0.05,
        "suppression_ratio": float(mean_alt_unf / max(ci_fil, detector_floor)),
        "detector_resolution": detector_floor,
        "continuous_index_no_filter": [rec["checkerboard_index_raw"] for rec in unf],
        "continuous_index_filtered": fil["checkerboard_index_physical"],
        "continuous_suppression_ratio": float(
            mean_ci_unf / max(fil["checkerboard_index_physical"], detector_floor)
        ),
        "status": "PASS" if c2_ok else "FAIL",
    }

    # ------------------------------------------------------------------ C3
    c_scan = [rec["compliance"] for rec in scan]
    rho_s = spearman_rho(radii, c_scan)
    steps = np.diff(c_scan)
    record["criteria"]["C3"] = {
        "statement": "compliance monotone in r over [1.0, 3.0]; Spearman rho > 0.9",
        "radii": list(radii),
        "compliance": c_scan,
        "spearman_rho": rho_s,
        "threshold": 0.9,
        "monotone_increasing": bool(np.all(steps > 0.0)),
        "n_decreasing_steps": int(np.sum(steps <= 0.0)),
        "min_relative_step": float(np.min(steps / np.asarray(c_scan[:-1]))) if len(steps) else None,
        "status": "PASS" if rho_s > 0.9 else "FAIL",
        "note": (
            "Compliance must not decrease with the filter radius: a larger radius "
            "restricts the attainable fields further, so C(r) is monotonically "
            "non-decreasing in r. A strictly negative step is a numerical red flag."
        ),
    }

    # ------------------------------------------------------------------ C4
    c_ref = unf[0]["compliance"]
    dev = abs(neg["compliance"] - c_ref) / c_ref
    d_rho = float(np.max(np.abs(neg["rho"] - unf[0]["rho"])))
    record["criteria"]["C4"] = {
        "statement": "r -> 0 degrades to the unfiltered run (deviation <= 1 %)",
        "radius": float(negative_radius),
        "compliance": neg["compliance"],
        "compliance_reference": c_ref,
        "relative_deviation": dev,
        "max_density_difference": d_rho,
        "threshold": 0.01,
        "status": "PASS" if dev <= 0.01 else "FAIL",
        "note": (
            "The negative control must reproduce the unfiltered run, not fail to "
            "run: P(r -> 0) -> I, so both the design and the compliance have to "
            "collapse onto the r = None branch."
        ),
    }

    # ------------------------------------------------------------------ C5
    n_tests = count_test_cases()
    record["criteria"]["C5"] = {
        "statement": ">= 8 pytest cases (standard case / detector / scan / negative control)",
        "value": n_tests,
        "threshold": 8,
        "status": "PASS" if n_tests >= 8 else "FAIL",
    }

    statuses = {k: v["status"] for k, v in record["criteria"].items()}
    failed = [k for k, v in statuses.items() if v == "FAIL"]
    record["verdict"] = {
        "criteria_status": statuses,
        "n_pass": sum(v == "PASS" for v in statuses.values()),
        "n_total": len(statuses),
        "failed": failed,
        "honest_note": (
            "C1 fails as worded: the filtered optimum cannot be stiffer than the "
            "unfiltered one (measured ratio "
            f"{ratio:.4f} <= 1, theoretical bound <= 1). C2/C3/C4/C5 carry the "
            "physical claim (checkerboard suppression + radius monotonicity + "
            "negative control)."
            if "C1" in failed
            else "all criteria met"
        ),
    }
    record["wall_seconds"] = time.time() - t_start

    if outdir is not None:
        out = pathlib.Path(outdir)
        out.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, np.ndarray] = {}
        for rec in results:
            key = f"{rec['tag']}_seed{rec['seed']}"
            arrays[f"{key}_rho"] = rec["rho"].reshape(ny, nx)
            arrays[f"{key}_evidence"] = np.array(
                [
                    rec["compliance"],
                    rec["alternating_fraction_raw"],
                    rec["alternating_fraction_physical"],
                    rec["checkerboard_index_raw"],
                    rec["checkerboard_index_physical"],
                ]
            )
        np.savez_compressed(out / "designs.npz", **arrays)
        (out / "verification.json").write_text(
            json.dumps(json.loads(json.dumps(record, default=float)), indent=2)
        )
        record["artefacts"] = [
            str(out / "verification.json"),
            str(out / "designs.npz"),
        ]
    return record


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="IN157-N05 SIMP filter acceptance (C1-C5)")
    parser.add_argument("--outdir", default="artifacts/simp_in157_n05")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--iters", type=int, default=300)
    parser.add_argument("--tol", type=float, default=5e-3)
    parser.add_argument("--quick", action="store_true", help="scaled-down smoke run")
    args = parser.parse_args(argv)
    rec = run_acceptance(
        args.outdir, workers=args.workers, iters=args.iters, tol=args.tol, quick=args.quick
    )
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk in
                          {"status", "value", "ratio", "ratio_mean", "spearman_rho",
                           "index_no_filter_mean", "index_filtered",
                           "suppression_ratio", "relative_deviation"}}
                      for k, v in rec["criteria"].items()}, indent=2, default=float))
    print("verdict:", json.dumps(rec["verdict"], indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())