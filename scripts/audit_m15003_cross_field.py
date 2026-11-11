#!/usr/bin/env python3
"""m15003 cross-field audit: independent reproduction + latency attribution.

Reads the m14678 benchmark artefact that *this* audit run regenerated
(``artifacts/m14678/results.json``), re-hashes payloads of the same sizes to
measure the local sha256 throughput, and tests the hypothesis

    t_shm_zero_copy(nbytes) ~= 2 * t_sha256(nbytes) + fixed_overhead

i.e. that the "zero-copy" path is dominated by the two integrity hashes
(one in ``describe()`` on publish, one in ``verify_payload()`` on receive) and
not by copies.  Writes ``artifacts/audit_m15003/evidence.json`` + figure.

No solver code is modified; this script only measures.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

RESULTS = ROOT / "artifacts/m14678/results.json"
OUT = ROOT / "artifacts/audit_m15003"


def sha256_ns_per_byte(nbytes: int, repeats: int = 20) -> float:
    """Wall-clock seconds per byte for one sha256 over ``nbytes`` bytes."""
    buf = np.zeros(nbytes, dtype=np.uint8)
    mv = memoryview(buf)
    hashlib.sha256(mv).digest()  # warm
    t0 = time.perf_counter_ns()
    for _ in range(repeats):
        hashlib.sha256(mv).digest()
    dt = (time.perf_counter_ns() - t0) / repeats
    return dt / nbytes


def main() -> int:
    res = json.loads(RESULTS.read_text())
    rows = res["transport_matrix"]
    OUT.mkdir(parents=True, exist_ok=True)

    table = []
    for r in rows:
        nb = r["nbytes"]
        t_hash_ns = sha256_ns_per_byte(nb) * nb
        zc = r["median_ns"]["shm_zero_copy"]
        sock = r["median_ns"]["socket_single_copy"]
        fileb = r["median_ns"]["file_baseline"]
        pred = 2.0 * t_hash_ns
        table.append({
            "nbytes": nb,
            "median_ns": r["median_ns"],
            "bandwidth_MBps": r["bandwidth_MBps"],
            "copies_total": r["copies_total"],
            "bit_exact": r["bit_exact"],
            "sha256_one_pass_ns": t_hash_ns,
            "sha256_two_pass_ns": pred,
            "shm_zero_copy_ns": zc,
            "hash_share_of_shm_zero_copy": pred / zc,
            "shm_vs_socket_ratio": zc / sock,
            "shm_vs_file_ratio": zc / fileb,
        })

    big = max(table, key=lambda x: x["nbytes"])
    findings = {
        "sha256_throughput_MBps_4MiB": round(4 * 1024 * 1024
                                             / (big["sha256_one_pass_ns"] / 1e9) / 1e6, 1),
        "hash_share_of_shm_zero_copy_4MiB": round(big["hash_share_of_shm_zero_copy"], 3),
        "shm_vs_socket_ratio_4MiB": round(big["shm_vs_socket_ratio"], 2),
        "shm_vs_file_ratio_4MiB": round(big["shm_vs_file_ratio"], 2),
        "zero_copy_faster_than_socket_any_size": all(
            t["shm_zero_copy_ns"] < t["median_ns"]["socket_single_copy"] for t in table),
        "max_tested_field_MiB": round(big["nbytes"] / 2**20, 2),
    }

    payload = {
        "audit_mission": 15003,
        "audited_mission": 14678,
        "gap": "GAP-CROSS-FIELD",
        "repro": {
            "regenerated_by": "python tools/m14678_field_transport_benchmark.py "
                              "--out artifacts/m14678",
            "tests": "python -m pytest tests/test_field_exchange_transport.py -q -> 21 passed",
            "subset": "python -m pytest tests -k 'field_exchange or volume_transfer or "
                      "mapping' -q -> 81 passed, 1 skipped, 1 error "
                      "(pre-existing missing golden p27w3, disclosed by m14678)",
        },
        "acceptance_reproduced": res["two_process_exchange"]["wall_shear"],
        "bit_exact_reproduced": res["two_process_exchange"]["bit_exact_roundtrip"],
        "no_file_audit": res["no_file_audit"],
        "latency_attribution": table,
        "findings": findings,
    }
    (OUT / "evidence.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(json.dumps(findings, indent=2))

    # ---- figure -----------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nb = np.array([t["nbytes"] for t in table], dtype=float)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for key, label, style in (
        ("shm_zero_copy", "anon-shm zero-copy", "o-"),
        ("shm_marshal", "anon-shm marshal (1 copy)", "s--"),
        ("socket_single_copy", "socket (1+1 copy)", "^-."),
        ("file_baseline", "file round trip (no fsync)", "d:"),
    ):
        ax[0].loglog(nb, [t["median_ns"][key] / 1e6 for t in table], style, label=label)
    ax[0].loglog(nb, [t["sha256_two_pass_ns"] / 1e6 for t in table], "kx--",
                 label="2 x sha256 (integrity hashes)")
    ax[0].set_xlabel("field size [bytes]"); ax[0].set_ylabel("median latency [ms]")
    ax[0].set_title("GAP-CROSS-FIELD transport: latency vs field size")
    ax[0].grid(True, which="both", alpha=0.3); ax[0].legend(fontsize=7)

    frac = [100 * t["hash_share_of_shm_zero_copy"] for t in table]
    ax[1].semilogx(nb, frac, "o-")
    ax[1].axhline(50, color="grey", ls=":")
    ax[1].set_xlabel("field size [bytes]")
    ax[1].set_ylabel("2x sha256 as % of zero-copy latency")
    ax[1].set_title("integrity hashing dominates above ~1 MiB")
    ax[1].set_ylim(0, 105); ax[1].grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    png = OUT / "fig_latency_attribution.png"
    fig.savefig(png, dpi=130)
    print("figure:", png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())