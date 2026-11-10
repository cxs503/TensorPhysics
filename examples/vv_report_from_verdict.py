"""Render a V&V report from a gate verdict and attach the §4 figures.

Mission m8633: the report body is injected from
:func:`tensorphysics.vv.gci.evaluate` (zero hand-copy), the log-log
error-vs-grid-spacing figure is generated from the same verdict, and both are
written to ``artifacts/vv_report_m8633/``.

Run::

    python examples/vv_report_from_verdict.py

Outputs
-------
``report.md``                 rendered six-element report (markdown)
``report.json``               machine ledger (``VVReport.as_dict()``)
``fig_convergence.png``       log-log |f - f_ext| vs h, fitted order, GCI band
``fig_validation.png``        |E| / U_val vs the acceptance line
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tensorphysics.vv.gci import Observation, evaluate  # noqa: E402
from tensorphysics.vv.report import build_report  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "artifacts" / "vv_report_m8633"

# Synthetic manufactured sequence: f(h) = f_inf + c * h**p with known p = 1.8,
# f_inf = 2.0 (a stand-in for a hull-resistance coefficient on a refinement
# ladder; the report generator is solver-neutral and takes any gate verdict).
F_INF, C, P = 2.0, 3.0, 1.8
LEVELS = [Observation(h=h, f=F_INF + C * h**P, label=f"h={h:g}") for h in (0.4, 0.2, 0.1, 0.05)]
F_REF, U_INPUT = 2.002, 0.004  # reference value and its declared uncertainty


def _figures(verdict) -> dict[str, str]:
    """Generate the two §4 figures straight from the verdict."""
    OUT.mkdir(parents=True, exist_ok=True)
    study = verdict.study
    hs = [lvl.h for lvl in study.levels]
    errs = [abs(lvl.f - study.f_extrapolated) for lvl in study.levels]

    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    ax.loglog(hs, errs, "o-", label=" |f_h - f_ext|  (gate levels)")
    href = [min(hs) * 0.5, max(hs) * 2.0]
    scale = errs[0] / (hs[0] ** study.order_p)
    ax.loglog(href, [scale * h ** study.order_p for h in href], "--",
              label=f"slope p = {study.order_p:.4f} (observed)")
    ax.set_xlabel("characteristic spacing h")
    ax.set_ylabel("|f_h - f_ext|")
    ax.set_title("Grid convergence of the key quantity")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    f1 = OUT / "fig_convergence.png"
    fig.tight_layout()
    fig.savefig(f1, dpi=130)
    plt.close(fig)

    val = verdict.validation
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    ax.bar(["|E| / U_val"], [val.ratio], color="#2b7bba", width=0.45)
    ax.axhline(1.0, color="crimson", linestyle="--", label="acceptance |E| = U_val")
    ax.set_ylim(0, max(1.25, val.ratio * 1.25))
    ax.set_ylabel("acceptance ratio")
    ax.set_title(
        f"ASME V&V 20: {val.status} (U_val = {val.u_val:.4g}, "
        f"|E| = {val.error:.4g})"
    )
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    f2 = OUT / "fig_validation.png"
    fig.tight_layout()
    fig.savefig(f2, dpi=130)
    plt.close(fig)
    return {"convergence": str(f1), "validation": str(f2)}


def main() -> int:
    verdict = evaluate(LEVELS, quantity="Cd", f_exp=F_REF, u_input=U_INPUT, u_ref=U_INPUT)
    report = build_report(
        verdict,
        title="Manufactured-convergence V&V report (m8633 example)",
        metadata={
            "case_name": "manufactured sequence f(h) = 2 + 3 h^1.8, four levels",
            "reference_source": "manufactured solution (exact reference, not an experiment)",
            "parameters": [
                {"name": "f_inf", "value": F_INF, "unit": "-", "uncertainty": 0.0},
                {"name": "p (manufactured)", "value": P, "unit": "-", "uncertainty": 0.0},
                {"name": "f_ref", "value": F_REF, "unit": "-", "uncertainty": U_INPUT},
            ],
            "notes": "levels 0.4 / 0.2 / 0.1 / 0.05, refinement ratio 2 (uniform)",
        },
    )
    figs = _figures(verdict)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.md").write_text(report.markdown, encoding="utf-8")
    (OUT / "report.json").write_text(
        json.dumps(report.as_dict(), indent=2, sort_keys=True), encoding="utf-8"
    )

    print(f"verdict status   : {verdict.status}")
    print(f"report status    : {report.status}")
    print(f"observed order p : {verdict.study.order_p!r} (manufactured {P})")
    print(f"f_ext            : {verdict.study.f_extrapolated!r} (exact {F_INF})")
    rel = abs(verdict.validation.f_sim - verdict.validation.f_exp) / abs(verdict.validation.f_exp)
    print(f"relative error   : {rel * 100:.4f} % (acceptance < 3 %)")
    print(f"|E| / U_val      : {verdict.validation.ratio:.6g}  -> {verdict.validation.status}")
    print(f"coverage         : {report.coverage['ratio']:.2f} "
          f"(withheld required: {report.coverage['required_withheld']})")
    print(f"provenance       : {report.provenance['source_digest'][:16]}… / "
          f"{report.provenance['body_digest'][:16]}…")
    for name, path in figs.items():
        print(f"figure {name:11s}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())