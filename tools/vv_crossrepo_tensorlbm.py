"""Cross-repository V&V run: TensorLBM Poiseuille verification case through the gate.

This script is *zero-intrusion*: it imports ``tensorlbm`` (read-only) and
reproduces, in this repository, the exact solver call sequence of
``TensorLBM/tests/test_verification.py::_run_poiseuille`` -- BGK collision,
periodic streaming, half-way bounce-back on the two channel walls and the
Guo-style body-force term of that file -- so the numbers are directly
comparable with the repository's own O(h^2) claim while nothing in TensorLBM is
touched.

Three grids (``ny = 16, 32, 64``) are run and three observables of the same
fields are reported:

* ``rel_l2_error``  -- relative L2 error of the axial velocity profile;
* ``flow_rate_error`` -- relative error of the integrated flow rate ``sum(ux)``;
* ``peak_velocity_error`` -- relative error of the centreline velocity.

``--emit-dir`` receives one gate observation JSON per observable plus the raw
record, so the CLI can be run on each of them.

Usage::

    python tools/vv_crossrepo_tensorlbm.py --emit-dir artifacts/vv_gate
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

DEFAULT_SRC = os.environ.get("TENSORLBM_SRC", "/root/TensorLBM/src")


def _import_tensorlbm(src: str):
    sys.path.insert(0, src)
    import torch  # noqa: PLC0415

    torch.set_default_dtype(torch.float64)
    import tensorlbm  # noqa: PLC0415
    from tensorlbm import OPPOSITE, collide_bgk, equilibrium, macroscopic, stream  # noqa: PLC0415

    return torch, tensorlbm, dict(
        OPPOSITE=OPPOSITE, collide_bgk=collide_bgk, equilibrium=equilibrium,
        macroscopic=macroscopic, stream=stream,
    )


def run_poiseuille(torch, api, ny: int, *, nu: float, fx: float, n_steps: int):
    """Faithful transcription of TensorLBM's own 2-D Poiseuille verification case."""
    nx = 4
    rho0 = torch.ones((ny, nx))
    zero = torch.zeros_like(rho0)
    f = api["equilibrium"](rho0, zero, zero)
    tau = 3.0 * nu + 0.5
    opp = api["OPPOSITE"]
    for _ in range(n_steps):
        f = api["collide_bgk"](f, tau=tau)
        f = api["stream"](f)
        f_bb = f.clone()
        f_bb[:, 0, :] = f[opp][:, 0, :]
        f_bb[:, -1, :] = f[opp][:, -1, :]
        f = f_bb
        rho, ux, uy = api["macroscopic"](f)
        ux = ux + 2.0 * fx * tau
        feq_new = api["equilibrium"](rho, ux, uy)
        feq_old = api["equilibrium"](rho, ux - 2.0 * fx * tau, uy)
        f = f + (feq_new - feq_old) * (1.0 - 0.5 / tau)
    _, ux_f, _ = api["macroscopic"](f)
    return ux_f[:, 0]


def analytic(torch, ny: int, nu: float, fx: float):
    y = torch.arange(ny, dtype=torch.float64)
    return fx / (2.0 * nu) * y * (ny - 1 - y)


def metrics(torch, ux, ref):
    nx_inner = slice(1, -1)
    num = torch.sqrt(((ux[nx_inner] - ref[nx_inner]) ** 2).sum())
    den = torch.sqrt((ref[nx_inner] ** 2).sum())
    rel_l2 = float(num / den)
    q_num, q_ref = float(ux.sum()), float(ref.sum())
    flow_rate = abs(q_num - q_ref) / abs(q_ref)
    peak = abs(float(ux.max()) - float(ref.max())) / abs(float(ref.max()))
    return {"rel_l2_error": rel_l2, "flow_rate_error": flow_rate,
            "peak_velocity_error": peak}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tensorlbm-src", default=DEFAULT_SRC)
    parser.add_argument("--emit-dir", default="artifacts/vv_gate")
    parser.add_argument("--resolutions", type=int, nargs="+", default=[16, 32, 64])
    parser.add_argument("--nu", type=float, default=1.0 / 6.0)
    parser.add_argument("--fx", type=float, default=1e-4)
    parser.add_argument("--steps-per-cell", type=float, default=4.0,
                        help="n_steps = steps_per_cell * ny**2 (the repo test uses 4)")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args(argv)

    torch, tensorlbm, api = _import_tensorlbm(args.tensorlbm_src)
    emit = Path(args.emit_dir)
    emit.mkdir(parents=True, exist_ok=True)

    records = []
    profiles = []
    analytic_profile: list[float] = []
    for ny in args.resolutions:
        steps = int(args.steps_per_cell * ny * ny)
        t0 = time.time()
        ux = run_poiseuille(torch, api, ny, nu=args.nu, fx=args.fx, n_steps=steps)
        ref = analytic(torch, ny, args.nu, args.fx)
        met = metrics(torch, ux, ref)
        elapsed = time.time() - t0
        records.append({"ny": ny, "h": 1.0 / ny, "steps": steps,
                        "seconds": round(elapsed, 2), **met})
        profiles.append({
            "ny": ny,
            "ux": [float(v) for v in ux],
            "error": [float(a - b) for a, b in zip(ux, ref)],
        })
        analytic_profile = [float(v) for v in ref]
        print(f"ny={ny:4d} steps={steps:7d} " +
              " ".join(f"{k}={v:.6e}" for k, v in met.items()) +
              f"  t={elapsed:.1f}s", flush=True)

    record = {
        "schema": "tensorphysics.vv.crossrepo/1",
        "repo": "TensorLBM",
        "case": "bidirectional_channel_poiseuille_verification",
        "solver_version": getattr(tensorlbm, "__version__", "unknown"),
        "parameters": {"nu": args.nu, "fx": args.fx, "nx": 4,
                       "steps_per_cell": args.steps_per_cell,
                       "wall_treatment": "half-way bounce-back",
                       "forcing": "Guo-style body force (repo sequence)",
                       "oracle": "ux(y) = fx/(2 nu) y (ny-1-y)"},
        "levels": records,
    }
    out = Path(args.json_out) if args.json_out else emit / "tensorlbm_poiseuille_raw.json"
    out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    (emit / "lbm_profiles.json").write_text(json.dumps({
        "case": "poiseuille_verification",
        "finite_difference_note": "profiles plotted against the analytic channel height",
        "analytic": analytic_profile,
        "analytic_y": [i / (len(analytic_profile) - 1) for i in range(len(analytic_profile))],
        "profiles": profiles,
    }, indent=1) + "\n", encoding="utf-8")

    for observable in ("rel_l2_error", "flow_rate_error", "peak_velocity_error"):
        payload = {
            "schema": "tensorphysics.vv.observations/1",
            "quantity": observable,
            "expected_order": 2.0,
            "source": {"repo": "TensorLBM", "path": str(out),
                       "study": "poiseuille_verification",
                       "metadata": {"solver_version": record["solver_version"]}},
            "levels": [
                {"h": row["h"], "f": row[observable], "label": f"ny={row['ny']}"}
                for row in records
            ],
        }
        path = emit / f"obs_tensorlbm_poiseuille_{observable}.json"
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {path}")
    # the profile name expected by the test-suite / report
    canonical = emit / "obs_tensorlbm_poiseuille.json"
    canonical.write_text(
        (emit / "obs_tensorlbm_poiseuille_rel_l2_error.json").read_text(encoding="utf-8"),
        encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())