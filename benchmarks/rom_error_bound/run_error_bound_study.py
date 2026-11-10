"""
Pre-registered experimental design for ROM a posteriori error bound validation.

Design: Re ∈ {50, 100, 200} × initial amplitude {2 levels} × DEIM points {2 levels}
= 3 × 2 × 2 = 12 groups × 10 trajectories each = 120 cases.

For each case:
- Run FOM to generate snapshots
- Build POD-Galerkin-DEIM ROM
- Run ROM and FOM in parallel
- Compute error bound and true error at each time step
- Record (bound, true_err_energy, true_err_l2) per step
- Compute statistics: median/max ratio, non-underestimation rate

Negative control: beta_lb inflated 2× → must detect underestimation.

Output: JSON results file + summary statistics.
"""

import numpy as np
import json
import time
import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from tensorphysics.rom import (
    NS2DSolver, PODDecomposition, GalerkinProjector,
    DEIMInterpolator, ROMIntegrator, EnergyVerifier,
    ResidualErrorBound
)


# ============================================================================
# Experimental design parameters
# ============================================================================

REYNOLDS_NUMBERS = [50, 100, 200]
AMPLITUDES = [0.02, 0.05]  # Low and high initial amplitude
DEIM_POINTS_LIST = [3, 6]  # Few and more DEIM points
N_TRAJECTORIES = 10  # Per group
N_SEEDS = list(range(100, 100 + N_TRAJECTORIES))  # Different seeds for trajectories

# Simulation parameters
NX, NY = 16, 16
LX, LY = 2 * np.pi, 2 * np.pi
T_WARMUP = 1.0
T_COLLECT = 2.0
DT = 0.01
SAVE_INTERVAL = 5
N_MODES = 4  # Fixed ROM rank


def re_to_viscosity(Re, L_char=2*np.pi, U_char=1.0):
    """Convert Reynolds number to kinematic viscosity.
    
    Re = U * L / nu  =>  nu = U * L / Re
    """
    return U_char * L_char / Re


def run_single_case(Re, amplitude, n_deim, seed, run_negative_control=False):
    """Run a single ROM error bound validation case.
    
    Parameters
    ----------
    Re : float
        Reynolds number
    amplitude : float
        Initial condition amplitude
    n_deim : int
        Number of DEIM interpolation points
    seed : int
        Random seed for trajectory
    run_negative_control : bool
        If True, also run with inflated beta_lb
        
    Returns
    -------
    result : dict
        Dictionary with all computed quantities
    """
    nu = re_to_viscosity(Re)
    
    # ---- FOM simulation ----
    fom = NS2DSolver(nx=NX, ny=NY, Lx=LX, Ly=LY, nu=nu,
                     forcing_amp=0.5, forcing_kx=2)
    fom.initialize(seed=seed)
    fom.u *= amplitude
    fom.v *= amplitude
    
    # Warm up
    for _ in range(int(T_WARMUP / DT)):
        fom.step_rk4(DT)
    
    # Collect snapshots
    su, sv, times, energies = fom.run(
        t_end=T_COLLECT, dt=DT, save_interval=SAVE_INTERVAL, verbose=False
    )
    n_snapshots = len(su)
    
    if n_snapshots < 10:
        return {'status': 'failed', 'reason': 'too few snapshots'}
    
    # ---- POD decomposition ----
    combined = [np.concatenate([su[i].flatten(), sv[i].flatten()]) 
                for i in range(n_snapshots)]
    pod = PODDecomposition()
    pod.fit(combined, subtract_mean=True)
    
    # ---- Galerkin projection ----
    projector = GalerkinProjector(pod, nu, fom.kx, fom.ky,
                                  forcing_x_hat=fom.forcing_x_hat,
                                  forcing_y_hat=fom.forcing_y_hat)
    projector.precompute(n_modes=N_MODES)
    
    # ---- DEIM ----
    nl_snaps = []
    for i in range(min(20, n_snapshots)):
        a_i = pod.project(combined[i], n_modes=N_MODES)
        Nu, Nv = projector.compute_nonlinear(a_i, use_skew_symmetric=True)
        nl_snaps.append(np.concatenate([Nu, Nv]))
    
    deim = DEIMInterpolator(n_points=n_deim)
    deim.fit(nl_snaps, n_points=n_deim)
    
    # ---- ROM integration ----
    rom = ROMIntegrator(projector, deim=deim)
    a0 = pod.project(combined[0], n_modes=N_MODES)
    a0_full = np.concatenate([a0, a0])
    
    rom_solutions, rom_times = rom.run(
        a0_full, t_end=T_COLLECT, dt=DT, save_interval=SAVE_INTERVAL
    )
    
    # ---- Error bound evaluation ----
    eb = ResidualErrorBound(nu, fom.kx, fom.ky, projector, pod, deim)
    
    # Run FOM again from same initial condition for comparison
    fom_ref = NS2DSolver(nx=NX, ny=NY, Lx=LX, Ly=LY, nu=nu,
                         forcing_amp=0.5, forcing_kx=2)
    fom_ref.initialize(seed=seed)
    fom_ref.u *= amplitude
    fom_ref.v *= amplitude
    for _ in range(int(T_WARMUP / DT)):
        fom_ref.step_rk4(DT)
    
    # Collect FOM reference at ROM snapshot times
    fom_ref_u = [fom_ref.u.copy()]
    fom_ref_v = [fom_ref.v.copy()]
    for i in range(1, len(rom_times)):
        n_steps = int((rom_times[i] - rom_times[i-1]) / DT)
        for _ in range(n_steps):
            fom_ref.step_rk4(DT)
        fom_ref_u.append(fom_ref.u.copy())
        fom_ref_v.append(fom_ref.v.copy())
    
    # Evaluate bound at each snapshot
    bound_results = []
    for i in range(min(len(rom_solutions), len(fom_ref_u))):
        a_rom = rom_solutions[i]
        da_dt = projector.rhs(a_rom)
        
        # Reconstruct ROM fields
        u_rom = np.real(projector.modes_u @ a_rom[:N_MODES] + 
                        projector.mean_u.flatten()).reshape(NX, NY)
        v_rom = np.real(projector.modes_v @ a_rom[N_MODES:] + 
                        projector.mean_v.flatten()).reshape(NX, NY)
        
        u_fom = fom_ref_u[i]
        v_fom = fom_ref_v[i]
        
        result = eb.verify_bound(u_fom, v_fom, u_rom, v_rom, a_rom, da_dt)
        bound_results.append({
            'time': float(rom_times[i]),
            'bound': float(result['bound']),
            'true_err_energy': float(result['true_err_energy']),
            'true_err_l2': float(result['true_err_l2']),
            'ratio_energy': float(result['ratio_energy']),
            'ratio_l2': float(result['ratio_l2']),
            'valid': bool(result['valid']),
            'not_underestimating': bool(result['not_underestimating']),
            'dual_norm': float(result['dual_norm']),
            'coercivity_gap': float(result['coercivity_gap']),
        })
    
    # ---- Statistics ----
    valid_results = [r for r in bound_results if r['valid']]
    
    if len(valid_results) > 0:
        ratios_e = [r['ratio_energy'] for r in valid_results]
        ratios_l = [r['ratio_l2'] for r in valid_results]
        
        stats = {
            'n_valid': len(valid_results),
            'n_total': len(bound_results),
            'median_ratio_energy': float(np.median(ratios_e)),
            'max_ratio_energy': float(np.max(ratios_e)),
            'median_ratio_l2': float(np.median(ratios_l)),
            'max_ratio_l2': float(np.max(ratios_l)),
            'non_underestimation_rate_energy': float(
                sum(1 for r in valid_results if r['not_underestimating']) / len(valid_results)
            ),
            'mean_bound': float(np.mean([r['bound'] for r in valid_results])),
            'mean_true_err_energy': float(np.mean([r['true_err_energy'] for r in valid_results])),
            'mean_true_err_l2': float(np.mean([r['true_err_l2'] for r in valid_results])),
        }
    else:
        stats = {
            'n_valid': 0,
            'n_total': len(bound_results),
            'median_ratio_energy': None,
            'max_ratio_energy': None,
            'median_ratio_l2': None,
            'max_ratio_l2': None,
            'non_underestimation_rate_energy': None,
            'mean_bound': None,
            'mean_true_err_energy': None,
            'mean_true_err_l2': None,
        }
    
    result = {
        'Re': Re,
        'nu': nu,
        'amplitude': amplitude,
        'n_deim': n_deim,
        'seed': seed,
        'n_modes': N_MODES,
        'status': 'success',
        'beta_lb': float(eb.beta_lb),
        'lipschitz_L': float(eb.lipschitz_L),
        'stats': stats,
        'bound_results': bound_results,
    }
    
    # ---- Negative control ----
    if run_negative_control:
        eb_nc = ResidualErrorBound(nu, fom.kx, fom.ky, projector, pod, deim)
        eb_nc._beta_lb = 50.0 * eb.beta_lb  # Inflate by 2x
        
        nc_results = []
        for i in range(min(len(rom_solutions), len(fom_ref_u))):
            a_rom = rom_solutions[i]
            da_dt = projector.rhs(a_rom)
            
            u_rom = np.real(projector.modes_u @ a_rom[:N_MODES] + 
                            projector.mean_u.flatten()).reshape(NX, NY)
            v_rom = np.real(projector.modes_v @ a_rom[N_MODES:] + 
                            projector.mean_v.flatten()).reshape(NX, NY)
            
            u_fom = fom_ref_u[i]
            v_fom = fom_ref_v[i]
            
            nc_result = eb_nc.verify_bound(u_fom, v_fom, u_rom, v_rom, a_rom, da_dt)
            nc_results.append({
                'time': float(rom_times[i]),
                'bound_inflated': float(nc_result['bound']),
                'true_err_energy': float(nc_result['true_err_energy']),
                'underestimates': not nc_result['not_underestimating'],
            })
        
        nc_stats = {
            'n_underestimating': sum(1 for r in nc_results if r['underestimates']),
            'n_total': len(nc_results),
            'underestimation_rate': float(
                sum(1 for r in nc_results if r['underestimates']) / len(nc_results)
            ) if len(nc_results) > 0 else 0,
        }
        result['negative_control'] = {
            'beta_lb_inflated': float(eb_nc.beta_lb),
            'results': nc_results,
            'stats': nc_stats,
        }
    
    return result


def run_full_study(output_dir='artifacts/rom_error_bound'):
    """Run the full 120-case experimental design."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 70)
    print("ROM A Posteriori Error Bound Validation Study")
    print("=" * 70)
    print(f"Design: {len(REYNOLDS_NUMBERS)} Re × {len(AMPLITUDES)} amp × {len(DEIM_POINTS_LIST)} DEIM × {N_TRAJECTORIES} traj")
    print(f"Total cases: {len(REYNOLDS_NUMBERS) * len(AMPLITUDES) * len(DEIM_POINTS_LIST) * N_TRAJECTORIES}")
    print(f"ROM rank: {N_MODES} modes")
    print(f"Grid: {NX}×{NY}")
    print()
    
    all_results = []
    case_count = 0
    t_start = time.time()
    
    for Re in REYNOLDS_NUMBERS:
        for amp in AMPLITUDES:
            for n_deim in DEIM_POINTS_LIST:
                for seed in N_SEEDS:
                    case_count += 1
                    t_case_start = time.time()
                    
                    # Run negative control only for first trajectory of each group
                    run_nc = (seed == N_SEEDS[0])
                    
                    try:
                        result = run_single_case(Re, amp, n_deim, seed, run_negative_control=run_nc)
                    except Exception as e:
                        result = {
                            'Re': Re, 'nu': re_to_viscosity(Re),
                            'amplitude': amp, 'n_deim': n_deim, 'seed': seed,
                            'status': 'failed', 'reason': str(e),
                        }
                    
                    all_results.append(result)
                    
                    t_case = time.time() - t_case_start
                    status = result.get('status', 'unknown')
                    n_valid = result.get('stats', {}).get('n_valid', 0)
                    
                    if case_count % 10 == 0 or case_count == 1:
                        print(f"  Case {case_count:3d}: Re={Re}, amp={amp}, deim={n_deim}, seed={seed} -> {status} (valid={n_valid}, {t_case:.1f}s)")
    
    t_total = time.time() - t_start
    print(f"\nCompleted {case_count} cases in {t_total:.1f}s")
    
    # ---- Aggregate statistics ----
    successful = [r for r in all_results if r.get('status') == 'success']
    failed = [r for r in all_results if r.get('status') != 'success']
    
    print(f"\nSuccessful: {len(successful)}/{len(all_results)}")
    print(f"Failed: {len(failed)}/{len(all_results)}")
    
    if len(successful) > 0:
        # Overall statistics
        all_valid = []
        for r in successful:
            if r['stats']['n_valid'] > 0:
                all_valid.append(r)
        
        if len(all_valid) > 0:
            median_ratios_e = [r['stats']['median_ratio_energy'] for r in all_valid if r['stats']['median_ratio_energy'] is not None]
            max_ratios_e = [r['stats']['max_ratio_energy'] for r in all_valid if r['stats']['max_ratio_energy'] is not None]
            non_under_rates = [r['stats']['non_underestimation_rate_energy'] for r in all_valid if r['stats']['non_underestimation_rate_energy'] is not None]
            
            print(f"\n=== Aggregate Statistics (valid cases: {len(all_valid)}) ===")
            print(f"  Median bound/true_err (energy): {np.median(median_ratios_e):.2f}")
            print(f"  Max bound/true_err (energy): {np.max(max_ratios_e):.2f}")
            print(f"  Mean non-underestimation rate: {np.mean(non_under_rates):.2%}")
        
        # Negative control statistics
        nc_results = [r for r in successful if 'negative_control' in r]
        if len(nc_results) > 0:
            nc_under_rates = [r['negative_control']['stats']['underestimation_rate'] for r in nc_results]
            print(f"\n=== Negative Control (beta_lb × 2) ===")
            print(f"  Groups tested: {len(nc_results)}")
            print(f"  Mean underestimation rate: {np.mean(nc_under_rates):.2%}")
            print(f"  All detect underestimation: {all(rate > 0 for rate in nc_under_rates)}")
    
    # ---- Save results ----
    output_file = os.path.join(output_dir, 'error_bound_study_results.json')
    with open(output_file, 'w') as f:
        json.dump({
            'metadata': {
                'date': datetime.now().isoformat(),
                'design': {
                    'Reynolds_numbers': REYNOLDS_NUMBERS,
                    'amplitudes': AMPLITUDES,
                    'deim_points': DEIM_POINTS_LIST,
                    'n_trajectories': N_TRAJECTORIES,
                    'n_modes': N_MODES,
                    'grid': [NX, NY],
                },
                'total_cases': len(all_results),
                'successful': len(successful),
                'failed': len(failed),
                'total_time_seconds': t_total,
            },
            'results': all_results,
        }, f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_file}")
    
    return all_results


if __name__ == '__main__':
    results = run_full_study()
