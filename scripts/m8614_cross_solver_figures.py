"""Generate report figures + numbers for the cross-solver agreement gate (m8614).

Every number in the report is produced here by importing the module under test
and calling its public API -- no hand-typed results.
"""
from __future__ import annotations

import base64
import io
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tensorphysics.vv.cross_solver import (
    AgreementPolicy,
    SolverSample,
    agreement_spread,
    consensus,
    cross_solver_agreement,
)

OUT = pathlib.Path("runs/m8614_cross_solver")
OUT.mkdir(parents=True, exist_ok=True)


def S(solver, f, u=None):
    return SolverSample(solver=solver, f=f, u=u)


# ---------------------------------------------------------------------------
# Case A: four solvers on a drag coefficient (the integration case)
# ---------------------------------------------------------------------------
drag = [
    S("TensorLBM", 0.0120, 0.0004),
    S("TensorFVM", 0.0120, 0.0005),
    S("TensorFEM", 0.0121, 0.0003),
    S("TensorBEM", 0.0120, 0.0006),
]
res_a = cross_solver_agreement(drag, quantity="Cd")
f_c, u_c = consensus(drag)
spread, spread_rel, widest = agreement_spread(drag)

fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=150)
names = [s.solver for s in drag]
xs = range(len(drag))
ax.errorbar(
    xs,
    [s.f for s in drag],
    yerr=[s.u for s in drag],
    fmt="o",
    capsize=5,
    ms=7,
    color="#1f77b4",
    ecolor="#1f77b4",
    alpha=0.85,
    label="solver value (1-sigma)",
)
ax.axhline(f_c, color="#d62728", lw=2, ls="--", label=f"consensus = {f_c:.5f}")
ax.axhspan(f_c - u_c, f_c + u_c, color="#d62728", alpha=0.12, label=f"consensus +/- u_c = {u_c:.2e}")
ax.set_xticks(list(xs))
ax.set_xticklabels(names, rotation=12)
ax.set_ylabel("drag coefficient $C_d$")
ax.set_title(f"Cross-solver agreement, $C_d$  ->  {res_a.status}")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "fig1_consensus.png")
plt.close(fig)

# ---------------------------------------------------------------------------
# Case B: sweep one solver's disagreement -> PASS/WARN/FAIL transitions
# ---------------------------------------------------------------------------
policy = AgreementPolicy()
base = [S("TensorLBM", 1.00, 0.01), S("TensorFVM", 1.00, 0.01), S("TensorFEM", 1.00, 0.01)]
sweeps = []
for delta in [i * 0.001 for i in range(0, 81)]:
    samples = base + [S("TensorBEM", 1.00 + delta, 0.01)]
    r = cross_solver_agreement(samples, quantity="Cd", policy=policy)
    sweeps.append({"delta": delta, "status": r.status, "spread_rel": r.spread_rel, "consensus": r.consensus})

status_color = {"PASS": "#2ca02c", "WARN": "#ff7f0e", "FAIL": "#d62728"}
fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=150)
for s in sweeps:
    ax.scatter(s["delta"] * 100, 0, s=26, color=status_color[s["status"]], zorder=3)
ax.axvline(policy.warn_spread * 100, color="#ff7f0e", ls=":", lw=1.5)
ax.axvline(policy.fail_spread * 100, color="#d62728", ls=":", lw=1.5)
ax.text(policy.warn_spread * 100 + 0.15, 0.28, f"warn {policy.warn_spread:.0%}", color="#ff7f0e", fontsize=8)
ax.text(policy.fail_spread * 100 + 0.15, 0.28, f"fail {policy.fail_spread:.0%}", color="#d62728", fontsize=8)
ax.set_xlim(-1, 81)
ax.set_ylim(-0.6, 0.6)
ax.set_yticks([])
ax.set_xlabel("TensorBEM offset from the other three  (%)")
ax.set_title("Agreement-gate verdict vs. one solver's offset (3 anchors at 1.00)")
handles = [plt.Line2D([0], [0], marker="o", ls="", color=status_color[k], label=k) for k in ("PASS", "WARN", "FAIL")]
ax.legend(handles=handles, fontsize=8, loc="upper left")
ax.grid(alpha=0.3, axis="x")
fig.tight_layout()
fig.savefig(OUT / "fig2_gate_sweep.png")
plt.close(fig)

# ---------------------------------------------------------------------------
# numbers for the report (programmatic, zero hand-copy)
# ---------------------------------------------------------------------------
# find the delta where the verdict first leaves PASS and first hits FAIL
first_warn = next((s["delta"] for s in sweeps if s["status"] != "PASS"), None)
first_fail = next((s["delta"] for s in sweeps if s["status"] == "FAIL"), None)

numbers = {
    "case_A": {
        "quantity": "Cd",
        "solvers": {s.solver: {"f": s.f, "u": s.u} for s in drag},
        "consensus": f_c,
        "consensus_u": u_c,
        "spread": spread,
        "spread_rel": spread_rel,
        "widest_pair": list(widest),
        "status": res_a.status,
        "outliers": list(res_a.outliers),
    },
    "case_B": {
        "policy": policy.as_dict(),
        "first_non_pass_delta": first_warn,
        "first_fail_delta": first_fail,
        "n_sweeps": len(sweeps),
    },
}
(OUT / "numbers.json").write_text(json.dumps(numbers, indent=2))

# base64 for the report payload
b64 = {}
for name in ("fig1_consensus.png", "fig2_gate_sweep.png"):
    b64[name] = base64.b64encode((OUT / name).read_bytes()).decode()

print(json.dumps(numbers, indent=2))
print("figures:", sorted(p.name for p in OUT.glob("*.png")))
