#!/usr/bin/env python3
"""End-to-end UQ -> validation-gate chain on a ship-resistance quantity.

This example is the "wiring" proof of the UQ contract:

1. a **black-box** resistance solver (the Holtrop-1984 kernel of this
   repository, called as-is -- nothing in it is imported by the UQ workflow
   itself and nothing is modified);
2. uncertain main particulars and speed -- a
   :class:`~tensorphysics.uq.sample_spec.SampleSpec`;
3. :func:`~tensorphysics.uq.workflow.run_uq_study` -- sampling, solver
   fan-out, polynomial chaos, analytic + sampled Sobol' indices, cross-check
   and the three-state verdict;
4. the resulting 95 % output band becomes ``U_input`` of the existing ASME
   V&V 20 gate :func:`tensorphysics.vv.validation.validate`, i.e. the
   uncertainty budget is consumed by the verification workflow exactly like a
   discretisation or experimental uncertainty.

Run with ``python examples/uq_end_to_end.py`` (about 10 s).
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from tensorphysics.uq import holtrop_type_benchmark, run_uq_study  # noqa: E402
from tensorphysics.vv.validation import validate  # noqa: E402


def main() -> int:
    bench = holtrop_type_benchmark()
    print("uncertain inputs:", ", ".join(bench.spec.names))
    print(f"reference case   : R_T = {bench.reference_case['RT_kN']:.2f} kN")
    print(f"spec fingerprint : {bench.spec.fingerprint()[:16]}...")

    study = run_uq_study(
        bench.resistance_kn,
        bench.spec,
        n_samples=256,
        degree=4,
        n_boot=100,
        n_base_mc=1 << 13,
        bootstrap_mc=50,
        budget_n_mc=10000,
        direct_budget_n_mc=2048,
    )

    print("\n-- certificate -------------------------------------------------")
    print(f"sample-set hash  : {study.samples.hash}")
    print(f"surrogate        : degree {study.surrogate.degree}, "
          f"{study.surrogate.n_terms} terms, eps_LOO = {study.surrogate.eps_loo:.2e}, "
          f"R2 = {study.surrogate.r2:.6f}")
    for name, s_i, st_i in zip(bench.spec.names, study.pce_sobol.first, study.pce_sobol.total):
        print(f"  S[{name:>3}] = {s_i:+.4f}   S_T[{name:>3}] = {st_i:+.4f}")
    if study.mc is not None:
        print(f"  Monte-Carlo cross-check: material indices (|S| >= 5%) agree to "
              f"{study.cross_check.max_material_relative_deviation:.2%}; "
              f"worst over all reported indices {study.cross_check.max_relative_deviation:.2%} "
              f"({study.cross_check.n_noise_limited} index/indices limited by MC noise)")
    print(f"95% band of R_T  : [{study.budget.lower95:.2f}, {study.budget.upper95:.2f}] kN "
          f"-> U/|mean| = {study.budget.relative_width:.3%}")
    if study.budget_direct is not None:
        print(f"direct-MC band   : U/|mean| = {study.budget_direct.relative_width:.3%} "
              f"(surrogate vs direct: {abs(study.budget.relative_width - study.budget_direct.relative_width):.3%})")
    print(f"verdict          : {study.status} "
          f"(threshold {study.verdict.threshold:.0%}, reason codes {study.verdict.reason_codes or 'none'})")

    # ---- hand the UQ band to the existing V&V 20 gate --------------------
    nominal = np.array([[bench.reference_case[k] for k in bench.spec.names]])
    _, coords = bench.spec.from_latent(bench.spec.physical_to_latent(nominal))
    f_surrogate = float(study.surrogate.evaluate(coords)[0])
    f_solver = float(bench.resistance_kn(nominal)[0])
    result = validate(
        f_surrogate,
        f_solver,
        u_num=study.surrogate.eps_loo_abs,
        u_input=0.5 * study.budget.width,
        label="holtrop1984-UQ-budget",
    )
    print("\n-- ASME V&V 20 gate (U_input from the UQ budget) -----------------")
    print(json.dumps(result.as_dict(), indent=1))
    print(f"gate status      : {result.status} (|E|/U_val = {result.ratio:.3f})")

    payload = {
        "uq_status": study.status,
        "uq_reason_codes": list(study.verdict.reason_codes),
        "relative_width": study.budget.relative_width,
        "cross_check_material_deviation": study.cross_check.max_material_relative_deviation,
        "cross_check_max_deviation": study.cross_check.max_relative_deviation,
        "cross_check_noise_limited": study.cross_check.n_noise_limited,
        "sample_hash": study.samples.hash,
        "spec_hash": bench.spec.fingerprint(),
        "sobol_first": dict(zip(bench.spec.names, study.pce_sobol.first.tolist())),
        "sobol_total": dict(zip(bench.spec.names, study.pce_sobol.total.tolist())),
        "vv_status": result.status,
        "vv_ratio": result.ratio,
    }
    out = pathlib.Path("artifacts/uq/evidence/end_to_end_example.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nwritten: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())