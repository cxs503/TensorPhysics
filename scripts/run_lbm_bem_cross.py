"""Drive the LBM <-> BEM cross-validation: solve the shared scenarios, compare, report.

Usage
-----
    python scripts/run_lbm_bem_cross.py --stage lbm --case re100 --max-seconds 100
    python scripts/run_lbm_bem_cross.py --stage compare --case re100
    python scripts/run_lbm_bem_cross.py --stage report

Stages
------
``lbm``      advance the lattice-Boltzmann cylinder case (checkpointed, so the
             stage can be called repeatedly with a wall-clock budget);
``bem``      solve the BEM + integral-boundary-layer side of the case;
``compare``  cross-solver metrics, criteria evaluation and the diff-report JSON;
``report``   PNG figures + the consolidated criteria JSON.

All artifacts land in ``artifacts/lbm_bem_cross/``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

ART = ROOT / "artifacts" / "lbm_bem_cross"
ART.mkdir(parents=True, exist_ok=True)

from tensorphysics.lbm_bem_cross import (  # noqa: E402
    CrossValidation, SCENARIOS, geometry_contract_check, load_geometry,
    run_case, summarize_case,
)
from tensorphysics.lbm_bem_cross.lbm import CylinderLBM, LBMConfig  # noqa: E402


def _cfg(sc) -> LBMConfig:
    return LBMConfig(
        re=sc.re, diameter=sc.diameter, u_in=sc.u_in,
        up_d=sc.up_d, down_d=sc.down_d, half_width_d=sc.half_width_d,
        lateral_bc="freestream", les_cs=sc.les_cs, init="potential",
        label=sc.name,
    )


def stage_lbm(name: str, max_seconds: float) -> dict:
    sc = SCENARIOS[name]
    cfg = _cfg(sc)
    ckpt = ART / f"ckpt_{name}.npz"
    lb = CylinderLBM(cfg)
    if ckpt.exists():
        lb.load_state_dict(np.load(ckpt))
    t0 = time.time()
    n = 0
    while lb.step_count < sc.total_steps and time.time() - t0 < max_seconds:
        lb.run(500, probe_every=500)
        n += 500
        if lb.step_count % 5000 == 0:
            np.savez(ckpt, **lb.state_dict())
    np.savez(ckpt, **lb.state_dict())
    done = lb.step_count >= sc.total_steps
    out = {"case": name, "steps": int(lb.step_count), "target": sc.total_steps,
           "complete": bool(done), "wall_s": round(time.time() - t0, 1)}
    if done:
        u, v = lb.velocity_field()
        th_w, cp_w = lb.wall_pressure()
        th_r, cp_r = lb.surface_pressure()
        cd_hist = np.asarray([p["cd"] for p in lb.probe])
        cl_hist = np.asarray([p["cl"] for p in lb.probe])
        np.savez(ART / f"lbm_{name}.npz", u=u.astype(np.float32), v=v.astype(np.float32),
                 rho=lb.rho.astype(np.float32), theta_wall=th_w, cp_wall=cp_w,
                 theta_ring=th_r, cp_ring=cp_r, cd_hist=cd_hist, cl_hist=cl_hist,
                 x=lb.x.ravel(), y=lb.y.ravel(), cx0=np.array(cfg.cx0),
                 cy0=np.array(cfg.cy0), diameter=np.array(cfg.diameter),
                 u_in=np.array(cfg.u_in), re=np.array(cfg.re),
                 step=np.array(lb.step_count),
                 sep_angle=np.array(lb.separation_angle()))
        summ = lb.summary()
        summ["sep_angle_deg"] = float(np.degrees(lb.separation_angle()))
        fin = cd_hist[np.isfinite(cd_hist)]
        summ["cd_mean_last_half"] = float(np.mean(fin[len(fin) // 2:])) if fin.size else float("nan")
        finc = cl_hist[np.isfinite(cl_hist)]
        summ["cl_rms_last_half"] = float(np.sqrt(np.mean(finc[len(finc) // 2:] ** 2))) if finc.size else float("nan")
        summ["strouhal"] = strouhal_number(cd_hist, cl_hist, cfg, probe_every=500)
        summ["cd_history"] = [None if not np.isfinite(c) else float(c) for c in cd_hist]
        summ["solution_valid"] = bool(np.isfinite(u).all())
        (ART / f"lbm_{name}_summary.json").write_text(json.dumps(summ, indent=1))
        out["summary"] = summ
        ckpt.unlink(missing_ok=True)
    return out


def strouhal_number(cd_hist: np.ndarray, cl_hist: np.ndarray, cfg: LBMConfig,
                    probe_every: int) -> float:
    hist = cl_hist + cd_hist  # lift dominates; fall back on cd if lift is flat
    src = cl_hist if np.std(cl_hist) > 1e-3 else cd_hist
    if np.std(src) < 1e-6:
        return float("nan")
    fft = np.fft.rfft(src - src.mean())
    freqs = np.fft.rfftfreq(src.size, d=probe_every)
    k = int(np.argmax(np.abs(fft[1:])) + 1)
    f_lattice = freqs[k]
    return float(f_lattice * cfg.diameter / cfg.u_in)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="lbm",
                    choices=["lbm", "bem", "compare", "report", "all"])
    ap.add_argument("--case", default=None)
    ap.add_argument("--max-seconds", type=float, default=90.0)
    args = ap.parse_args()

    cases = [args.case] if args.case else list(SCENARIOS)
    if args.stage in ("lbm",):
        for c in cases:
            print(json.dumps(stage_lbm(c, args.max_seconds)), flush=True)
    elif args.stage == "bem":
        for c in cases:
            print(json.dumps(run_case(c, ART), default=float), flush=True)
    elif args.stage == "compare":
        for c in cases:
            print(json.dumps(summarize_case(c, ART), default=float), flush=True)
    elif args.stage == "report":
        from tensorphysics.lbm_bem_cross.report import build_report
        print(json.dumps(build_report(ART), indent=1, default=float))
    elif args.stage == "all":
        for c in cases:
            while not stage_lbm(c, args.max_seconds)["complete"]:
                pass
            run_case(c, ART)
            summarize_case(c, ART)
        from tensorphysics.lbm_bem_cross.report import build_report
        print(json.dumps(build_report(ART), indent=1, default=float))


if __name__ == "__main__":
    main()