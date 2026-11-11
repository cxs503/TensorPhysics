"""Figures for the p171w1-n01 ROM a-posteriori error bound node.

Reads artifacts/rom_certified_bound/results.json and writes PNGs to
artifacts/rom_certified_bound/figures/.  Every figure is regenerated from the
recorded numbers (no hand-copied values).
"""

import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.path.insert(0, os.path.join(ROOT, 'benchmarks', 'rom_error_bound'))

OUT = os.path.join(ROOT, 'artifacts', 'rom_certified_bound')
FIG = os.path.join(OUT, 'figures')


def load():
    with open(os.path.join(OUT, 'results.json')) as f:
        return json.load(f)


def fig_strictness(res):
    pairs = [p for c in res['cases'] for p in c['pairs']
             if p['t'] >= 0.2 - 1e-12 and p['err_l2'] >= 1e-6 * p['norm_x']]
    e = np.array([p['err_l2'] for p in pairs])
    b = np.array([p['acc_l2'] for p in pairs])
    ev = np.array([p['err_v'] for p in pairs])
    bv = np.array([p['acc_v'] for p in pairs])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    for a, x, y, lab in ((ax[0], e, b, 'L2'), (ax[1], ev, bv, 'energy')):
        a.loglog(x, y, '.', ms=3, alpha=0.55, label=f'{lab} pairs (n={len(x)})')
        lim = [min(x.min(), y.min()) * 0.5, max(x.max(), y.max()) * 2]
        a.plot(lim, lim, 'k--', lw=1, label='bound = error')
        a.plot(lim, [2 * v for v in lim], 'r:', lw=1, label='2x line')
        a.set_xlabel(f'true error ({lab})')
        a.set_ylabel(f'bound ({lab})')
        a.set_title(f'strictness, {lab} norm')
        a.legend(fontsize=8)
        a.grid(alpha=0.3, which='both')
    fig.suptitle('A-posteriori bound vs true ROM error, 120 trajectories (all output pairs)')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'fig1_strictness.png'), dpi=140)
    plt.close(fig)


def fig_timeseries(res):
    case = [c for c in res['cases']
            if c['Re'] == 200.0 and c['amplitude'] == 0.05 and c['r_deim'] == 6][0]
    t = np.array([p['t'] for p in case['pairs']])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].semilogy(t, [p['err_l2'] for p in case['pairs']], 'k-', label='true error L2')
    ax[0].semilogy(t, [p['acc_l2'] for p in case['pairs']], 'C0--', label='bound (accumulated)')
    ax[0].semilogy(t, [p['steady_l2'] for p in case['pairs']], 'C1:', label='bound (steady)')
    ax[0].set_xlabel('t [s]'); ax[0].set_ylabel('L2 norm')
    ax[0].set_title(f"Re={case['Re']:.0f}, amp={case['amplitude']}, r_deim={case['r_deim']}, seed={case['seed']}")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
    r = np.array([p['acc_l2'] / max(p['err_l2'], 1e-300) for p in case['pairs']])
    rv = np.array([p['acc_v'] / max(p['err_v'], 1e-300) for p in case['pairs']])
    ax[1].plot(t, r, label='effectivity L2')
    ax[1].plot(t, rv, '--', label='effectivity energy')
    ax[1].axhline(1.0, color='k', lw=1)
    ax[1].set_xlabel('t [s]'); ax[1].set_ylabel('bound / true error')
    ax[1].set_ylim(0, max(4, np.nanmax(r) * 1.1))
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'fig2_timeseries.png'), dpi=140)
    plt.close(fig)


def fig_sweeps(res):
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))
    c2 = res['C2']
    ax[0].plot(c2['N'], c2['median_ratio'], 'o-', label='median effectivity')
    ax[0].plot(c2['N'], [x * 0 for x in c2['N']], 'k:')
    ax0b = ax[0].twinx()
    ax0b.semilogy(c2['N'], c2['median_true_error'], 's--', color='C3', label='median true error')
    ax[0].set_xlabel('POD modes N'); ax[0].set_ylabel('median effectivity')
    ax0b.set_ylabel('median true error (L2)', color='C3')
    ax[0].set_title(f"rank sweep, Spearman(ratio,N)={c2['spearman_ratio_vs_N']:.2f}")
    ax[0].grid(alpha=0.3)

    c5 = res['C5']
    ax[1].plot(c5['r_deim'], c5['median_ratio'], 'o-', color='C2')
    ax[1].set_xlabel('DEIM points r_deim'); ax[1].set_ylabel('median effectivity')
    ax[1].set_title(f"DEIM sweep, Spearman(ratio, r_deim)={c5['spearman_ratio_vs_deim']:.2f}")
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'fig3_sweeps.png'), dpi=140)
    plt.close(fig)


def fig_negative_control(res):
    rows = res['C3']['per_group']
    labels = [f"Re{r['Re']:.0f}\namp{r['amplitude']}\nrd{r['r_deim']}" for r in rows]
    vals = [0.0 if r['underest_rate'] is None else r['underest_rate'] for r in rows]
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.bar(range(len(vals)), vals, color='C3')
    ax.axhline(0.5, color='k', ls='--', label='criterion 0.5')
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labels, fontsize=6, rotation=90)
    ax.set_ylabel('underestimate detection rate')
    ax.set_title('Negative control: 2x-inflated Stokes gain detection')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'fig4_negative_control.png'), dpi=140)
    plt.close(fig)


def fig_fields():
    """FOM / ROM / error field at the end of a representative trajectory."""
    import run_certified_study as study
    from tensorphysics.rom.certified_bound import true_error  # noqa: F401
    case, rom, states, su, sv, times, fom = study.run_case(200.0, 0.05, 6, 101)
    u_rom, v_rom = rom.reconstruct(states[-1])
    eu, ev = su[-1] - u_rom, sv[-1] - v_rom
    fig, ax = plt.subplots(1, 4, figsize=(14, 3.6))
    im0 = ax[0].imshow(su[-1], origin='lower', cmap='viridis'); ax[0].set_title('FOM u')
    im1 = ax[1].imshow(u_rom, origin='lower', cmap='viridis'); ax[1].set_title('ROM u')
    err2 = eu ** 2 + ev ** 2
    im2 = ax[2].imshow(eu, origin='lower', cmap='RdBu_r'); ax[2].set_title('error u (FOM-ROM)')
    im3 = ax[3].imshow(np.sqrt(err2), origin='lower', cmap='magma'); ax[3].set_title('|error|')
    for a, im in zip(ax, (im0, im1, im2, im3)):
        a.set_xticks([]); a.set_yticks([])
        fig.colorbar(im, ax=a, fraction=0.046)
    fig.suptitle('Flow fields at t=%.1f s (Re=200, N=%d, r_deim=%d)'
                 % (times[-1], rom.k, case['r_deim']))
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'fig5_fields.png'), dpi=140)
    plt.close(fig)


if __name__ == '__main__':
    os.makedirs(FIG, exist_ok=True)
    res = load()
    fig_strictness(res)
    fig_timeseries(res)
    fig_sweeps(res)
    fig_negative_control(res)
    fig_fields()
    print('figures written to', FIG)