"""Pareto study: safety factor vs strictness vs negative-control power (C3)."""
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'benchmarks', 'rom_error_bound'))
sys.path.insert(0, os.path.join(ROOT, 'src'))

CFGS = [(50.0, 0.02, 3, 100), (50.0, 0.05, 6, 101), (100.0, 0.02, 3, 101),
        (100.0, 0.05, 6, 100), (200.0, 0.02, 6, 100), (200.0, 0.05, 3, 101)]


def run_for(safety):
    import importlib
    import run_certified_study as S
    S.SAFETY = safety
    rows = []
    for cfg in CFGS:
        case, *_ = S.run_case(*cfg)
        kept = [p for p in case['pairs'] if p['t'] >= 0.2 - 1e-12]
        r = np.array([p['acc_l2'] / max(p['err_l2'], 1e-300) for p in kept])
        rv = np.array([p['acc_v'] / max(p['err_v'], 1e-300) for p in kept])
        bad, *_ = S.run_case(*cfg, nu_scale=2.0)
        rows.append({'cfg': list(cfg),
                     'n_viol': int(np.sum((r < 1) | (rv < 1))),
                     'min_ratio': float(min(r.min(), rv.min())),
                     'median_ratio': float(np.median(r)),
                     'nc_rate': bad['underest_rate_2x']})
    return rows


if __name__ == '__main__':
    out = {}
    for s in (1.0, 1.25, 1.5, 2.0):
        rows = run_for(s)
        out[str(s)] = rows
        print('safety', s, 'strict_cases', sum(1 for r in rows if r['n_viol'] == 0),
              '/', len(rows),
              'median_eff %.2f' % np.median([r['median_ratio'] for r in rows]),
              'nc %.3f' % np.mean([r['nc_rate'] for r in rows if r['nc_rate'] is not None]), flush=True)
    path = os.path.join(ROOT, 'artifacts', 'rom_certified_bound', 'tradeoff.json')
    with open(path, 'w') as f:
        json.dump(out, f, indent=2)
    print('written', path)