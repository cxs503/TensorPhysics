"""Real three-grid MMS convergence run through the V&V gate: TensorLBM DG advection.

This tool *uses* TensorLBM as a library (zero intrusion: nothing inside that
repository is modified or patched).  It reproduces the manufactured-solution
case that TensorLBM ships in ``tests/test_dg_advection.py`` -- a smooth
``sin(x)`` profile advected once around a periodic ``[0, 2*pi)`` domain by the
nodal-DG advection kernel ``tensorlbm.dg_advection`` -- on three (or more)
uniform grids, collects the discrete L2 error, writes the numbers in
TensorLBM's own benchmark dialect, and normalises them with the cross-repository
adapter so the shared CLI can adjudicate them.

Theory: P1 nodal DG with an upwind flux and SSP-RK3 in time is second order in
the L2 norm, ``||e||_2 = O(h^2)``.  A gate PASS therefore requires
``|p_observed - 2| < 0.2`` with a decreasing GCI.

Usage::

    python tools/vv_crossrepo_lbm_dgmms.py --out artifacts/vv_gate_n07/tensorlbm_dgmms
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DEFAULT_TENSORLBM = Path("/opt/mm-verify/TensorLBM")


def _import_dg(root: Path):
    src = root / "src"
    if not src.is_dir():
        raise SystemExit(f"TensorLBM checkout not found at {root} (use --tensorlbm-root)")
    sys.path.insert(0, str(src))
    import torch  # noqa: PLC0415

    from tensorlbm.dg_advection import (  # noqa: PLC0415
        dg_advect,
        get_ops,
        lobatto_nodes,
    )

    return torch, dg_advect, get_ops, lobatto_nodes


def _lobatto_weights(degree: int):
    """Lobatto quadrature weights on [-1, 1] for ``degree + 1`` nodes.

    Identical to the helper of TensorLBM's own MMS test::
        w_k = 2 / (n (n - 1)) / P_{n-1}(x_k)^2,   n = degree + 1
    """
    import numpy as np  # noqa: PLC0415
    from numpy.polynomial import legendre as leg  # noqa: PLC0415

    from tensorlbm.dg_advection import lobatto_nodes as _nodes  # noqa: PLC0415

    n = degree + 1
    nodes = _nodes(degree)
    w = np.zeros(n)
    for k in range(n):
        pk = leg.Legendre.basis(n - 1)(nodes[k])
        w[k] = 2.0 / (n * (n - 1)) / (pk * pk)
    return w


def advect_periodic_1d(nx: int, degree: int = 1, n_periods: float = 1.0, cfl: float = 0.2) -> float:
    """Advect ``sin(x)`` once around a periodic domain; return the L2 error.

    Byte-for-byte the discretisation of TensorLBM's own MMS test (P1 DG, upwind
    flux, SSP-RK3, Lobatto quadrature), re-expressed here so that the grid is a
    free parameter and the error can be fed to the gate.
    """
    torch, dg_advect, get_ops, lobatto_nodes = _import_dg(Path(SETTINGS["root"]))
    dt_t = torch.float64
    L = 2.0 * math.pi
    dx = L / nx
    ops = get_ops(degree=degree, dx=dx, dtype=dt_t)
    r = torch.tensor(lobatto_nodes(degree), dtype=dt_t)
    i_idx = torch.arange(nx, dtype=dt_t).view(nx, 1)
    x_node = (i_idx * dx + (dx / 2.0) * (1.0 + r.view(1, -1))) % L
    f = torch.sin(x_node).unsqueeze(0)
    vel = torch.tensor([[1.0]], dtype=dt_t)
    T = n_periods * L
    n_steps = int(round(T / (cfl * dx)))
    dt = T / n_steps
    for _ in range(n_steps):
        f = dg_advect(f, vel, ops, ndim_spatial=1, dt=dt, n_substeps=1, scheme="rk3")
    exact = torch.sin(x_node)
    w = _lobatto_weights(degree)
    err2 = ((f.squeeze(0) - exact) ** 2 * torch.tensor(w, dtype=dt_t)).sum().item()
    err2 *= dx / 2.0
    return math.sqrt(err2 / L)


SETTINGS = {"root": str(DEFAULT_TENSORLBM)}


def run(grids: list[int], degree: int = 1, out: Path | None = None) -> dict:
    errors = [advect_periodic_1d(nx, degree=degree) for nx in grids]
    doc = {
        "schema": "tensorlbm.convergence/1.0",
        "case": "dg_advection_mms_periodic_sin",
        "quantity": "l2_error",
        "note": "nodal P1 DG advection of sin(x), one period, CFL=0.2, SSP-RK3",
        "expected_order": 2.0,
        "levels": [
            {"resolution": nx, "value": err, "label": f"nx={nx}"}
            for nx, err in zip(grids, errors)
        ],
        "oracle": "||e||_2 = O(h^2) for P1 DG (theory order 2)",
    }
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        (out / "tensorlbm_dg_mms_native.json").write_text(
            json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return doc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "vv_gate_n07" / "tensorlbm_dgmms"))
    ap.add_argument("--tensorlbm-root", default=str(DEFAULT_TENSORLBM))
    ap.add_argument("--grids", default="32,64,128", help="comma separated grid counts")
    ap.add_argument("--degree", type=int, default=1)
    args = ap.parse_args(argv)
    SETTINGS["root"] = args.tensorlbm_root
    grids = [int(g) for g in args.grids.split(",") if g.strip()]
    doc = run(grids, degree=args.degree, out=Path(args.out))
    print(json.dumps({k: v for k, v in doc.items() if k != "levels"}, indent=2))
    for lvl in doc["levels"]:
        print(f"  nx={lvl['resolution']:>4}  l2_error={lvl['value']:.12e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())