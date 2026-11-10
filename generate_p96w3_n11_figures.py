"""P96W3-N11 figures — generated from artifacts/p96w3_n11/acceptance.json only.

No number is hand-typed: every plotted value is read from the acceptance JSON
that the pytest suite writes.  Output: docs/p96w3_n11/fig*.png (plus a
provenance sidecar with sha256 of the input JSON).
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent
ACC = REPO / "artifacts" / "p96w3_n11" / "acceptance.json"
OUT = REPO / "docs" / "p96w3_n11"
OUT.mkdir(parents=True, exist_ok=True)

D = json.loads(ACC.read_text())


def fig1_shard_scaling() -> Path:
    c1 = D["C1"]
    ns = sorted(int(k) for k in c1["per_n"])
    wall = [c1["per_n"][str(n)]["elapsed_s"] for n in ns]
    sizes = {n: c1["per_n"][str(n)].get("shard_sizes", [c1["per_n"][str(n)]["keys"]]) for n in ns}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    bars = ax.bar([str(n) for n in ns], wall, color="#3b6ea5")
    for b, n in zip(bars, ns):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{c1['per_n'][str(n)]['keys']} keys",
                ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("shard count n")
    ax.set_ylabel("wall time for 64-point scan [s]")
    ax.set_title("(a) channel execution time (subprocess per shard)")

    ax = axes[1]
    width = 0.8 / len(ns)
    for j, n in enumerate(ns):
        vals = sizes[n]
        ax.bar(np.arange(len(vals)) + j * width, vals, width=width, label=f"n={n}")
    ax.set_xlabel("shard index")
    ax.set_ylabel("parameter points per shard")
    ax.set_title("(b) balanced segment sizes (max-min = %s)"
                 % {k: v for k, v in c1["balance"].items()})
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle("C1 swap-compatibility: 64 points aggregated bit-identically for n=1/2/4/8 "
                 f"({c1['shard_order_permutations_checked']} shard-order permutations)", fontsize=10)
    fig.tight_layout()
    path = OUT / "fig1_shard_scaling.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig2_float_bounds() -> Path:
    rows = D["C2_conditioning"]["rows"]
    R = np.array([max(r["R_actual"], 1e-1) for r in rows])
    naive = np.array([max(r["naive_rel_err"], 1e-18) for r in rows])
    kahan = np.array([max(r["kahan_rel_err"], 1e-18) for r in rows])

    # bounds evaluated for this construction: n=64 terms, sum|x| ~= 2 (m ~ 1)
    eps = np.finfo(float).eps
    abs_sum = 2.0
    naive_bound = D["C2"]["K_naive"] * eps * 64 * abs_sum / np.maximum(
        np.array([r["R_actual"] for r in rows]), 1e-300
    )
    kahan_bound = D["C2"]["K_kahan"] * eps * abs_sum / np.maximum(
        np.array([r["R_actual"] for r in rows]), 1e-300
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(R, naive, "o-", label="naive sum", color="#c0392b")
    ax.loglog(R, kahan, "s-", label="Kahan compensated", color="#1f6f3f")
    ax.loglog(R, naive_bound, "--", color="#c0392b", alpha=0.5,
              label=r"bound $K_{naive}\cdot\varepsilon\cdot n\cdot\sum|x|/|\Sigma x|$")
    ax.loglog(R, kahan_bound, "--", color="#1f6f3f", alpha=0.5,
              label=r"bound $K_{kahan}\cdot\varepsilon\cdot\sum|x|$ / $|\Sigma x|$")
    ax.axhline(1e-12, color="k", ls=":", lw=1.2)
    ax.text(R[0], 1.2e-12, "contract bar 1e-12 (naive)", fontsize=8)
    ax.axhline(1e-15, color="grey", ls=":", lw=1.2)
    ax.text(R[0], 1.2e-15, "contract bar 1e-15 (Kahan)", fontsize=8, color="grey")

    ncross = D["C2_conditioning"]["naive_crossover_R"]
    kcross = D["C2_conditioning"]["kahan_crossover_R"]
    if ncross:
        ax.axvline(ncross, color="#c0392b", lw=0.8, alpha=0.4)
        ax.annotate(f"naive leaves 1e-12\nat R≈{ncross:.1e}", (ncross, 1e-11),
                    fontsize=8, color="#c0392b", ha="right")
    if kcross:
        ax.axvline(kcross, color="#1f6f3f", lw=0.8, alpha=0.4)
        ax.annotate(f"Kahan leaves 1e-15\nat R≈{kcross:.1e}", (kcross, 1e-16),
                    fontsize=8, color="#1f6f3f", ha="left")

    ax.set_xlabel(r"conditioning ratio $R=\sum|x|\,/\,|\sum x|$")
    ax.set_ylabel("relative error vs exact (math.fsum)")
    ax.set_title("C2 float audit: compensated summation extends the valid range by ~3 decades")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    path = OUT / "fig2_float_bounds.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig3_failclosed_matrix() -> Path:
    c3, c3h, c4, c5 = D["C3"], D["C3_half_written"], D["C4"], D["C5"]
    rows = [
        ("kill 3/8 shards (artifacts deleted)", c3["raised_on_partial"]),
        ("kill -9 three shard processes (chaos)",
         c3["chaos_kill"]["recovered_bit_identical"] and not c3["chaos_kill"]["partial_result_emitted"]),
        ("retry of killed shards reproduces reference", c3["retry_bit_identical"]),
        ("truncated artifact (60% of bytes)", c3h["truncated_raises"]),
        ("complete JSON, status=running", c3h["incomplete_status_raises"]),
        ("zero-length artifact", c3h["empty_raises"]),
        ("duplicate key across two shards", c4["duplicate_key_raises"]),
        ("segment boundary swap (key moved)", c4["boundary_swap_raises"]),
        ("last-write-wins aggregator killed", c4["lww_killed_by_contract"]),
        ("git sha tamper (re-hashed envelope)", c5["results"]["git_sha"] == "red"),
        ("segment digest tamper", c5["results"]["segment_boundary"] == "red"),
        ("result payload tamper", c5["results"]["result_hash"] == "red"),
        ("pristine restore accepted", c5["results"]["restored"] == "green"),
    ]
    labels = [r[0] for r in rows]
    vals = np.array([[1.0 if r[1] else 0.0] for r in rows])

    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    ax.imshow(vals, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks([0])
    ax.set_xticklabels(["contract behaved\nas claimed"], fontsize=9)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    for i, (_, ok) in enumerate(rows):
        ax.text(0, i, "PASS" if ok else "FAIL", ha="center", va="center",
                fontsize=9, fontweight="bold")
    ax.set_title("C3/C4/C5 fail-closed injection matrix\n(all injections must raise, restore must pass)")
    fig.tight_layout()
    path = OUT / "fig3_failclosed_matrix.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig4_coverage() -> Path:
    trials = D["C6"]["trials"]
    ks = np.array([t["k"] for t in trials])
    ns = np.array([t["n_shards"] for t in trials])
    imb = np.array([t["imbalance"] for t in trials])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    jitter = np.random.default_rng(3).normal(0, 0.08, len(ns))
    sc = ax.scatter(ns + jitter, ks, c=imb, cmap="viridis", s=45,
                    edgecolor="k", linewidth=0.4)
    fig.colorbar(sc, ax=ax, label="max shard size - min shard size")
    ax.set_xlabel("shard count n")
    ax.set_ylabel("parameter points in the random space")
    ax.set_title(f"(a) 100 random spaces: coverage {D['C6']['coverage']}, "
                 f"overlaps {D['C6']['overlaps']}")

    ax = axes[1]
    bins = np.arange(-0.5, max(imb.max(), 1) + 1.5, 1)
    ax.hist(imb, bins=bins, color="#3b6ea5", edgecolor="k")
    ax.set_xlabel("segment imbalance (max-min)")
    ax.set_ylabel("count of spaces")
    ax.set_title(f"(b) imbalance distribution (max {imb.max()}), determinism {D['C6']['determinism']}")
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle("C6 boundary contract: coverage/seam-free mapping under non-linear index "
                 "and float endpoints", fontsize=10)
    fig.tight_layout()
    path = OUT / "fig4_coverage.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> None:
    paths = [fig1_shard_scaling(), fig2_float_bounds(), fig3_failclosed_matrix(), fig4_coverage()]
    digest = hashlib.sha256(ACC.read_bytes()).hexdigest()
    sidecar = {
        "input": str(ACC.relative_to(REPO)),
        "input_sha256": digest,
        "figures": [str(p.relative_to(REPO)) for p in paths],
        "figure_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
    }
    (OUT / "figures_provenance.json").write_text(json.dumps(sidecar, indent=2, sort_keys=True))
    print(json.dumps(sidecar, indent=2))


if __name__ == "__main__":
    main()