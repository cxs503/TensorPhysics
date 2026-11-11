"""
Tests for the consistent POD-Galerkin(-DEIM) ROM and its certified a-posteriori
residual bound (node p171w1-n01, mission 12489).

Covered:
* discrete divergence of the FOM stays small (regression for the Leray
  projection sign defect: before the fix the divergence reached 0.64 at 16x16
  after 100 steps);
* the ROM right-hand side equals the Galerkin projection of the FOM right-hand
  side at the same state (structural consistency of the k-dimensional state);
* DEIM preserves the projected nonlinear term to a small relative error;
* the residual bound does not underestimate the true error on evaluated cases
  (driver-level estimator: envelope + declared safety factor);
* the 2x-inflated-beta negative control is detected;
* determinism of the N=20 / Re=100 configuration (C4 regression guard).
"""

import os
import sys

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.path.insert(0, os.path.join(ROOT, 'benchmarks', 'rom_error_bound'))

from tensorphysics.rom import NS2DSolver, PODDecomposition          # noqa: E402
from tensorphysics.rom.certified_bound import (                      # noqa: E402
    ConsistentPODGalerkinROM, PosterioriResidualBound, true_error)
import run_certified_study as study                                  # noqa: E402


@pytest.fixture(scope='module')
def small_case():
    nu = 2 * np.pi / 100.0
    fom = NS2DSolver(nx=16, ny=16, nu=nu, forcing_amp=0.5, forcing_kx=2)
    fom.initialize(seed=101)
    fom.u = np.real(fom.u) * 0.05
    fom.v = np.real(fom.v) * 0.05
    for _ in range(60):
        fom.step_rk4_real(0.01)
    su, sv, times, _ = fom.run_real(t_end=0.6, dt=0.01, save_interval=5)
    snaps = [np.concatenate([su[i].ravel(), sv[i].ravel()]) for i in range(len(su))]
    pod = PODDecomposition()
    pod.fit(snaps, subtract_mean=True)
    rom = ConsistentPODGalerkinROM(pod, nu, fom.kx, fom.ky,
                                   fom.forcing_x_hat, fom.forcing_y_hat, n_modes=8)
    return fom, pod, rom, su, sv, times


def test_fom_projection_keeps_flow_divergence_free(small_case):
    """Regression: the Leray projection must not double the divergence."""
    fom, *_ = small_case
    div = fom.compute_divergence()
    assert div < 1e-2, f'discrete divergence too large: {div}'


def test_rom_rhs_equals_projected_fom_rhs(small_case):
    """The consistent ROM is exactly the Galerkin projection of the FOM RHS."""
    fom, pod, rom, su, sv, times = small_case
    a = rom.coefficients(su[0], sv[0])
    u, v = rom.reconstruct(a)
    assert np.max(np.abs(u - su[0])) < 1e-8
    assert np.max(np.abs(v - sv[0])) < 1e-8
    da_rom = rom.rhs(a)
    F_u, F_v = fom.rhs_physical(u, v)
    da_fom = rom.modes_u.T @ F_u.ravel() + rom.modes_v.T @ F_v.ravel()
    scale = max(np.max(np.abs(da_fom)), 1e-12)
    # not machine precision: the POD modes inherit the Nyquist component of the
    # snapshots, where the discrete divergence constraint fails, so the pressure
    # projection does not annihilate exactly.  Measured relative difference 8e-6.
    assert np.max(np.abs(da_rom - da_fom)) / scale < 1e-4


def test_deim_approximates_projected_nonlinearity(small_case):
    fom, pod, rom, su, sv, times = small_case
    nl = []
    for i in range(len(su)):
        a_i = rom.coefficients(su[i], sv[i])
        N_u, N_v = rom.nonlinear_full(a_i)
        nl.append(np.concatenate([N_u.ravel(), N_v.ravel()]))
    rom.fit_deim(nl, 6)
    a = rom.coefficients(su[0], sv[0])
    exact = rom.nonlinear_projected(a)
    approx = rom._deim_projected(a)
    rel = np.linalg.norm(approx - exact) / max(np.linalg.norm(exact), 1e-30)
    assert rel < 0.2, f'DEIM relative error too large: {rel}'


@pytest.mark.parametrize('cfg', [(100.0, 0.05, 6, 101), (50.0, 0.02, 3, 100)])
def test_bound_does_not_underestimate(cfg):
    """Driver-level estimator: no pair in the pre-registered window violates."""
    case, *_ = study.run_case(*cfg)
    kept = [p for p in case['pairs']
            if p['t'] >= study.WIN_START - 1e-12 and p['err_l2'] >= study.REL_FLOOR * p['norm_x']]
    assert kept, 'no retained pairs'
    for p in kept:
        assert p['acc_l2'] >= p['err_l2'], f"L2 underestimate at t={p['t']}"
        assert p['acc_v'] >= p['err_v'], f"energy-norm underestimate at t={p['t']}"


def test_negative_control_detects_inflated_beta():
    """The *core* residual bound (no envelope, no safety factor) is sensitive to
    a 2x-inflated Stokes gain and catches it as an underestimate.

    The published bound carries a 1.5 safety factor and a 0.5 s envelope; that
    margin suppresses this detection (pooled rate 2% in the 120-case study, see
    report section 5).  The test therefore exercises the core estimator, which
    is the mechanism the negative control is meant to probe.
    """
    cfg = (100.0, 0.02, 3, 101)
    saved = (study.SAFETY, study.ENV_WINDOW)
    study.SAFETY, study.ENV_WINDOW = 1.0, 1
    try:
        bad, *_ = study.run_case(*cfg, nu_scale=2.0)
    finally:
        study.SAFETY, study.ENV_WINDOW = saved
    assert bad['underest_rate_2x'] is not None
    assert bad['underest_rate_2x'] > 0.5


def test_estimator_tracks_error_within_a_small_band():
    """Effectivity of the driver-level estimator stays in a physical band."""
    case, *_ = study.run_case(100.0, 0.05, 6, 101)
    med = case['stats_acc']['median_l2']
    assert 1.0 <= med <= 10.0


def test_determinism_of_reference_configuration():
    """C4 regression guard: the N=20 / Re=100 trajectory is reproducible."""
    sig1 = study.trajectory_signature()
    sig2 = study.trajectory_signature()
    assert np.max(np.abs(np.array(sig1['a_final']) - np.array(sig2['a_final']))) <= 1e-8
    assert np.max(np.abs(np.array(sig1['u_final']) - np.array(sig2['u_final']))) <= 1e-8