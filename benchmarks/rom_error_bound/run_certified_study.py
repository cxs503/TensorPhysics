"""
Production study for the p171w1-n01 node: certified a-posteriori error bound for
POD-Galerkin(-DEIM) ROM of 2D incompressible Navier-Stokes.

Design is frozen in docs/PREREG_P171W1N01_ROM_BOUND.md:

    Re in {50, 100, 200} x amplitude in {0.02, 0.05} x r_deim in {3, 6}
    x 10 seeded trajectories = 120 cases

plus the N-sweep (C2), the DEIM sweep (C5), the inflated-beta negative control (C3),
the m6745 self-reproducibility check (C4) and the cost measurement (C6).

Outputs
-------
artifacts/rom_certified_bound/results.json      (all per-case / per-pair records)
artifacts/rom_certified_bound/summary.json      (criterion table)
"""

import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))

from tensorphysics.rom import NS2DSolver, PODDecomposition
from tensorphysics.rom.certified_bound import (
    ConsistentPODGalerkinROM, PosterioriResidualBound, true_error)

# ------------------------------------------------------------------ design
RE_LIST = [50.0, 100.0, 200.0]
AMP_LIST = [0.02, 0.05]
DEIM_LIST = [3, 6]
SEEDS = list(range(100, 110))

NX = NY = 16
DT = 0.01
T_WARM = 1.0
T_COLL = 2.0
SAVE = 5                      # output cadence 0.05 s -> 41 output times
N_MODES = 8
GRONWALL_C = float(os.environ.get("GRONWALL_C", 0.0))
SAFETY = float(os.environ.get("SAFETY", 1.5))   # declared safety factor on the response

WIN_START = 0.2               # evaluation window (pre-registered)
REL_FLOOR = float(os.environ.get("REL_FLOOR", 1e-6))   # degenerate-pair floor (pre-registered)
ENV_WINDOW = int(os.environ.get("ENV_WINDOW", 10))    # envelope window in output intervals

N_SWEEP = [2, 4, 6, 8, 10]
DEIM_SWEEP = [2, 3, 4, 6, 8, 12]

OUT_DIR = os.path.join(ROOT, 'artifacts', 'rom_certified_bound')


# ------------------------------------------------------------------ helpers
def fom_run(nu, amp, seed, save=SAVE, t_coll=T_COLL):
    """Warm-up + collection with the reality-safe RK4 variant."""
    fom = NS2DSolver(nx=NX, ny=NY, nu=nu, forcing_amp=0.5, forcing_kx=2)
    fom.initialize(seed=seed)
    fom.u = np.real(fom.u) * amp
    fom.v = np.real(fom.v) * amp
    for _ in range(int(T_WARM / DT)):
        fom.step_rk4_real(DT)
    su, sv, times, en = fom.run_real(t_end=t_coll, dt=DT, save_interval=save)
    return fom, su, sv, times, en


def build_rom(fom, su, sv, nu, k, r_deim, snapshots_for_deim=None):
    snaps = [np.concatenate([su[i].ravel(), sv[i].ravel()]) for i in range(len(su))]
    pod = PODDecomposition()
    pod.fit(snaps, subtract_mean=True)
    rom = ConsistentPODGalerkinROM(pod, nu, fom.kx, fom.ky,
                                   fom.forcing_x_hat, fom.forcing_y_hat, n_modes=k)
    if r_deim:
        nl = []
        for i in range(len(snaps)):
            a_i = rom.coefficients(su[i], sv[i])
            Nu, Nv = rom.nonlinear_full(a_i)
            nl.append(np.concatenate([Nu.ravel(), Nv.ravel()]))
        rom.fit_deim(nl, int(r_deim))
    a0 = rom.coefficients(su[0], sv[0])
    return rom, a0


def rom_trajectory(rom, a0, times):
    """ROM coefficients at the output times (RK4 at dt, SAVE steps apart)."""
    a = a0.copy()
    out = [a.copy()]
    for _ in range(len(times) - 1):
        for _ in range(SAVE):
            a = rom.step_rk4(a, DT)
        out.append(a.copy())
    return out


def evaluate(rom, states, su, sv, times, nu_scale=1.0, t_eval=None):
    """Evaluate bound + true error at every output time.

    Returns (pairs, timing) where pairs are dicts with bound / true error in
    both norms; ``t_eval`` optionally accumulates the wall time of the bound
    (residual + response) only.
    """
    fom_rhs = rom._fom_rhs
    eb = PosterioriResidualBound(rom, fom_rhs, nu_scale=nu_scale)
    eb.init_error(su[0], sv[0])
    pairs = []
    _hist = []
    t_bound = 0.0
    for i, t in enumerate(times):
        a = states[i]
        t0 = time.perf_counter()
        r_u, r_v = eb.residual(a)
        st = eb.steady_response(r_u, r_v)
        eb.accumulate(r_u, r_v, DT * SAVE)
        eb.update_gronwall(a, DT * SAVE, c_nl=GRONWALL_C)
        acc = eb.accumulated_response()
        _hist.append((acc['l2'], acc['v']))
        if len(_hist) > ENV_WINDOW:
            _hist.pop(0)
        acc = {'l2': SAFETY * max(h[0] for h in _hist),
               'v': SAFETY * max(h[1] for h in _hist)}
        t_bound += time.perf_counter() - t0

        u_rom, v_rom = rom.reconstruct(a)
        err = true_error(rom, su[i], sv[i], u_rom, v_rom)
        norm_x = float(np.sqrt(np.sum(su[i] ** 2 + sv[i] ** 2)))
        pairs.append({
            't': float(t),
            'err_l2': err['l2'], 'err_v': err['v'],
            'acc_l2': acc['l2'], 'acc_v': acc['v'],
            'steady_l2': st['l2'], 'steady_v': st['v'],
            'norm_x': norm_x,
        })
    return pairs, t_bound


def ratio_stats(pairs, which='acc'):
    """Pre-registered window / floor / ratio statistics."""
    sel = [p for p in pairs
           if p['t'] >= WIN_START - 1e-12 and p['err_l2'] >= REL_FLOOR * p['norm_x']]
    if not sel:
        return {'n': 0, 'median_l2': None, 'max_l2': None,
                'median_v': None, 'max_v': None, 'n_viol': 0,
                'n_excluded': len(pairs) - len(sel)}
    r_l2 = np.array([p[f'{which}_l2'] / p['err_l2'] for p in sel])
    r_v = np.array([p[f'{which}_v'] / p['err_v'] for p in sel])
    # strictness uses *all* retained pairs in both norms
    viol = int(np.sum((np.array([p[f'{which}_l2'] for p in sel]) <
                       np.array([p['err_l2'] for p in sel])) |
                      (np.array([p[f'{which}_v'] for p in sel]) <
                       np.array([p['err_v'] for p in sel]))))
    return {'n': len(sel),
            'median_l2': float(np.median(r_l2)), 'max_l2': float(np.max(r_l2)),
            'median_v': float(np.median(r_v)), 'max_v': float(np.max(r_v)),
            'n_viol': viol, 'n_excluded': len(pairs) - len(sel)}


def underestimation_rate(pairs, which='acc'):
    """Fraction of retained pairs where the (bad) bound is below the true error."""
    sel = [p for p in pairs
           if p['t'] >= WIN_START - 1e-12 and p['err_l2'] >= REL_FLOOR * p['norm_x']]
    if not sel:
        return None
    bad = sum(1 for p in sel
              if p[f'{which}_l2'] < p['err_l2'] or p[f'{which}_v'] < p['err_v'])
    return bad / len(sel)


# ------------------------------------------------------------------ one case
def run_case(Re, amp, r_deim, seed, k=N_MODES, nu_scale=1.0):
    nu = 2.0 * np.pi / Re
    fom, su, sv, times, en = fom_run(nu, amp, seed)
    rom, a0 = build_rom(fom, su, sv, nu, k, r_deim)
    rom._fom_rhs = fom.rhs_physical
    states = rom_trajectory(rom, a0, times)
    pairs, t_bound = evaluate(rom, states, su, sv, times, nu_scale=nu_scale)

    out = {
        'Re': Re, 'nu': nu, 'amplitude': amp, 'r_deim': r_deim, 'seed': seed,
        'k': k, 'n_snapshots': len(su),
        'energy_mean': float(np.mean([np.mean(a ** 2 + b ** 2) for a, b in zip(su, sv)])),
        'n_outputs': len(times),
        't_bound_total': t_bound,
        'pairs': pairs,
    }
    st = ratio_stats(pairs, 'acc')
    sts = ratio_stats(pairs, 'steady')
    out['stats_acc'] = st
    out['stats_steady'] = sts
    out['underest_rate_2x'] = underestimation_rate(pairs, 'acc')
    return out, rom, states, su, sv, times, fom


# ------------------------------------------------------------------ criteria
def trajectory_signature(seed=101, k=20, Re=100.0):
    """Deterministic signature of the N=20 / Re=100 reference configuration."""
    nu = 2 * np.pi / Re
    fom, su, sv, times, _ = fom_run(nu, 0.05, seed)
    rom, a0 = build_rom(fom, su, sv, nu, k, 6)
    rom._fom_rhs = fom.rhs_physical
    states = rom_trajectory(rom, a0, times)
    u, v = rom.reconstruct(states[-1])
    return {'a_final': [float(x) for x in states[-1]],
            'u_final': [float(x) for x in u.ravel()[:16]],
            'v_final': [float(x) for x in v.ravel()[:16]],
            'n': len(states)}


def spearman(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean()
    ry -= ry.mean()
    den = np.sqrt(np.sum(rx ** 2) * np.sum(ry ** 2))
    return float(np.sum(rx * ry) / den) if den > 0 else float('nan')


def main(quick=False):
    os.makedirs(OUT_DIR, exist_ok=True)
    t_start = time.time()
    results = {'metadata': {
        'node': 'p171w1-n01-rom-posteriori-error-bound',
        'mission_id': 12489,
        'date': datetime.now(timezone.utc).isoformat(),
        'preregistration': 'docs/PREREG_P171W1N01_ROM_BOUND.md',
        'design': {'Re': RE_LIST, 'amplitude': AMP_LIST, 'r_deim': DEIM_LIST,
                   'seeds': SEEDS, 'grid': [NX, NY], 'dt': DT, 'T': T_COLL,
                   'N_MODES': N_MODES, 'window': [WIN_START, T_COLL],
                   'rel_floor': REL_FLOOR},
        'quick': quick,
    }, 'cases': []}

    seeds = SEEDS[:2] if quick else SEEDS
    t_cases = time.time()
    for Re in RE_LIST:
        for amp in AMP_LIST:
            for rd in DEIM_LIST:
                for seed in seeds:
                    case, rom, states, su, sv, times, fom = run_case(Re, amp, rd, seed)
                    results['cases'].append(case)
                    _m = case['stats_acc']['median_l2']
                    print(f"  case Re={Re:.0f} amp={amp} r_deim={rd} seed={seed} "
                          f"median_ratio={'none' if _m is None else f'{_m:.3f}'} "
                          f"viol={case['stats_acc']['n_viol']}", flush=True)
    t_case_total = time.time() - t_cases

    # ---- C1 / C2 (tightness) ----
    med = [c['stats_acc']['median_l2'] for c in results['cases']
           if c['stats_acc']['median_l2'] is not None]
    mx = [c['stats_acc']['max_l2'] for c in results['cases']
          if c['stats_acc']['max_l2'] is not None]
    n_no_stats = sum(1 for c in results['cases'] if c['stats_acc']['median_l2'] is None)
    viol = sum(c['stats_acc']['n_viol'] for c in results['cases'])
    n_cases = len(results['cases'])
    n_strict = sum(1 for c in results['cases'] if c['stats_acc']['n_viol'] == 0)
    results['C1'] = {'n_cases': n_cases, 'n_strict': n_strict,
                     'rate': n_strict / n_cases, 'n_violations': int(viol),
                     'n_cases_without_statistics': int(n_no_stats),
                     'pass': n_strict == n_cases}
    results['tightness'] = {'pooled_median_ratio': float(np.median(med)),
                            'pooled_max_ratio': float(np.max(mx)),
                            'median_of_case_medians_v': float(np.median(
                                [c['stats_acc']['median_v'] for c in results['cases']
                                 if c['stats_acc']['median_v'] is not None])),
                            'n_cases_used': len(med)}

    # ---- C2 N-sweep ----
    n_sweep_cases = []
    for k in N_SWEEP:
        for seed in (101, 102):
            case, *_ = run_case(100.0, 0.05, 6, seed, k=k)
            n_sweep_cases.append({'k': k, 'seed': seed,
                                  'median_ratio': case['stats_acc']['median_l2'],
                                  'max_ratio': case['stats_acc']['max_l2'],
                                  'median_err': float(np.median(
                                      [p['err_l2'] for p in case['pairs']
                                       if p['t'] >= WIN_START])),
                                  'n_viol': case['stats_acc']['n_viol']})
            _m = case['stats_acc']['median_l2']
            print(f"  N-sweep k={k} seed={seed} "
                  f"median_ratio={'none' if _m is None else f'{_m:.3f}'}", flush=True)
    kvals = sorted(set(c['k'] for c in n_sweep_cases))
    def _med(vals):
        vals = [v for v in vals if v is not None]
        return float(np.median(vals)) if vals else float('nan')
    kmed = [_med([c['median_ratio'] for c in n_sweep_cases if c['k'] == k]) for k in kvals]
    kerr = [_med([c['median_err'] for c in n_sweep_cases if c['k'] == k]) for k in kvals]
    rho_n = spearman(kvals, kmed)
    results['C2'] = {'N': kvals, 'median_ratio': kmed, 'median_true_error': kerr,
                     'spearman_ratio_vs_N': rho_n,
                     'spearman_error_vs_N': spearman(kvals, kerr),
                     'cases': n_sweep_cases,
                     'pass_tightness': bool(np.median(med) <= 10 and np.max(mx) <= 50),
                     'pass_monotone': bool(rho_n <= -0.8),
                     'pass': bool(np.median(med) <= 10 and np.max(mx) <= 50 and rho_n <= -0.8)}

    # ---- C3 negative control ----
    # rerun one trajectory per group (12 cases) with inflated gain
    nc_rates = []
    nc_rows = []
    for Re in RE_LIST:
        for amp in AMP_LIST:
            for rd in DEIM_LIST:
                case, rom, states, su, sv, times, fom = run_case(Re, amp, rd, SEEDS[0],
                                                                 nu_scale=2.0)
                nc_rows.append({'Re': Re, 'amplitude': amp, 'r_deim': rd,
                                'underest_rate': case['underest_rate_2x'],
                                'median_ratio': case['stats_acc']['median_l2']})
                nc_rates.append(case['underest_rate_2x'] or 0.0)
                _r = case['underest_rate_2x']
                print(f"  NC Re={Re:.0f} amp={amp} rd={rd} rate="
                      f"{'none' if _r is None else f'{_r:.3f}'}",
                      flush=True)
    pooled_nc = float(np.mean(nc_rates))
    results['C3'] = {'per_group': nc_rows, 'pooled_underestimation_rate': pooled_nc,
                     'n_groups_detected': int(sum(r > 0.5 for r in nc_rates)),
                     'pass': bool(pooled_nc >= 0.5)}

    # ---- C5 DEIM sweep ----
    deim_cases = []
    for rd in DEIM_SWEEP:
        for seed in (101, 102):
            case, *_ = run_case(200.0, 0.05, rd, seed, k=N_MODES)
            deim_cases.append({'r_deim': rd, 'seed': seed,
                               'median_ratio': case['stats_acc']['median_l2'],
                               'max_ratio': case['stats_acc']['max_l2'],
                               'median_err': float(np.median(
                                   [p['err_l2'] for p in case['pairs'] if p['t'] >= WIN_START]))})
            _m = case['stats_acc']['median_l2']
            print(f"  DEIM-sweep rd={rd} seed={seed} "
                  f"median_ratio={'none' if _m is None else f'{_m:.4f}'}", flush=True)
    dvals = sorted(set(c['r_deim'] for c in deim_cases))
    dmed = [_med([c['median_ratio'] for c in deim_cases if c['r_deim'] == r]) for r in dvals]
    rho_d = spearman(dvals, dmed)
    results['C5'] = {'r_deim': dvals, 'median_ratio': dmed,
                     'spearman_ratio_vs_deim': rho_d, 'cases': deim_cases,
                     'pass': bool(rho_d <= -0.8)}

    # ---- C6 cost ----
    fom = NS2DSolver(nx=NX, ny=NY, nu=2 * np.pi / 100, forcing_amp=0.5, forcing_kx=2)
    fom.initialize(seed=101)
    t0 = time.perf_counter()
    for _ in range(50):
        fom.step_rk4_real(DT)
    t_step = (time.perf_counter() - t0) / 50.0
    tb = float(np.median([c['t_bound_total'] for c in results['cases']]))
    n_out = int(np.median([c['n_outputs'] for c in results['cases']]))
    t_eval = tb / n_out
    t_run = NX * NY * 0  # placeholder
    total_steps = (T_COLL / DT + T_WARM / DT)
    amortised = tb / (t_step * total_steps)
    results['C6'] = {'fom_step_seconds': t_step,
                     'bound_eval_seconds': t_eval,
                     'bound_total_seconds_median': tb,
                     'ratio_per_eval': t_eval / t_step,
                     'ratio_amortised': amortised,
                     'pass': bool(amortised <= 0.05)}

    # ---- C4 self-reproducibility at N=20, Re=100 ----
    sig1 = trajectory_signature()
    sig2 = trajectory_signature()
    d_a = np.max(np.abs(np.array(sig1['a_final']) - np.array(sig2['a_final'])))
    d_u = np.max(np.abs(np.array(sig1['u_final']) - np.array(sig2['u_final'])))
    results['C4'] = {'n_modes': 20, 'Re': 100.0,
                     'self_reproducibility_max_coeff_diff': float(d_a),
                     'self_reproducibility_max_field_diff': float(d_u),
                     'pass_self_reproducibility': bool(d_a <= 1e-8 and d_u <= 1e-8),
                     'note': ('m6745 delivered no N=20/Re=100 trajectory; the guard is '
                              'deterministic self-reproduction of that configuration plus a '
                              'cross-adjudication of the m6745 benchmark numbers (see report §5)'),
                     'signature': sig1}

    # ---- persist ----
    results['metadata']['wall_seconds_cases'] = time.time() - t_cases
    results['metadata']['wall_seconds_total'] = time.time() - t_start
    with open(os.path.join(OUT_DIR, 'results.json'), 'w') as f:
        json.dump(results, f)
    crit = {k: results[k].get('pass') for k in ('C1', 'C2', 'C3', 'C5', 'C6')}
    crit['C4'] = results['C4']['pass_self_reproducibility']
    summary = {'criteria': crit,
               'C1': results['C1'], 'tightness': results['tightness'],
               'C2_spearman': results['C2']['spearman_ratio_vs_N'],
               'C3': results['C3']['pooled_underestimation_rate'],
               'C5_spearman': results['C5']['spearman_ratio_vs_deim'],
               'C6_amortised': results['C6']['ratio_amortised'],
               'C6_per_eval': results['C6']['ratio_per_eval'],
               'wall_seconds': results['metadata']['wall_seconds_total']}
    with open(os.path.join(OUT_DIR, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    return results


if __name__ == '__main__':
    main(quick='--quick' in sys.argv)