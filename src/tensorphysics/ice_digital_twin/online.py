"""Streaming (online) re-calibration of the twin with a per-step time budget.

Why a separate online path
--------------------------
A digital twin is re-calibrated every time a new full-scale or model-test record
arrives, and the operator needs that update *inside* the control loop.  Running a
fresh 3-chain MCMC per record is not an option, so this module implements
**sequential Monte Carlo with MCMC rejuvenation** (resample-move, Gilks & Berzuini
2001):

1. the posterior is represented by ``n_particles`` weighted particles over the
   calibration box;
2. a new record multiplies the weights by its likelihood (one vectorised pass);
3. if the effective particle count drops below ``resample_fraction * N`` the
   particles are systematically resampled;
4. after (re)sampling, ``n_sweeps`` joint random-walk Metropolis sweeps with the
   *full data seen so far* as target restore particle diversity.  Because the
   target is the exact posterior of every record seen so far (no forgetting
   window, no Laplace approximation), the online estimate converges to the batch
   MCMC posterior -- which is the check reported in the campaign.

Cost per update is ``O(n_particles * n_records_seen * n_sweeps)`` elementary
vector operations plus ``O(n_particles)`` likelihood evaluations for the new
record, i.e. milliseconds for the campaign sizes used here; the measured
per-step wall time is reported against the 1 s budget.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..ice.resistance import G, RHO_ICE, SIGMA_C_REF, SIGMA_F_REF, ShipGeometry
from .bayes import CalibrationData, ChainSet, explained_prior, log_likelihood
from .twin import (
    KAPPA_BOUNDS,
    PARAM_BOUNDS,
    PARAM_NAMES,
    get_form,
    resistance_kN_batch,
)

__all__ = ["OnlineCalibrator", "run_online_calibration"]


def _resistance_particles(
    form: Any,
    params: np.ndarray,
    h: float,
    V: float,
    ship: ShipGeometry,
) -> np.ndarray:
    """Twin resistance (kN) for a **batch of particles** at one ice condition."""
    return resistance_kN_batch(form, params, h, V, ship)


@dataclass
class OnlineCalibrator:
    """Resample-move SMC calibrator over the twin parameter box."""

    ship: ShipGeometry
    form: Any = "full"
    n_particles: int = 400
    use_discrepancy: bool = True
    prior_kind: str = "uniform"
    n_sweeps: int = 2
    resample_fraction: float = 0.5
    seed: int = 20237

    particles: np.ndarray = field(init=False)
    logw: np.ndarray = field(init=False)
    records: List[Tuple[float, float, float, float]] = field(default_factory=list, init=False)
    history: List[Dict[str, Any]] = field(default_factory=list, init=False)
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        f = get_form(self.form)
        self.form = f
        bnd = f.bounds()
        if self.use_discrepancy:
            bnd = np.vstack([bnd, np.asarray([KAPPA_BOUNDS], dtype=float)])
        self.bounds = bnd
        if self.n_particles < 50:
            raise ValueError("n_particles must be >= 50 for a meaningful ESS")
        self._rng = np.random.default_rng(self.seed)
        lo, hi = bnd[:, 0], bnd[:, 1]
        self.particles = lo + (hi - lo) * self._rng.random((self.n_particles, bnd.shape[0]))
        self.logw = np.full(self.n_particles, -math.log(self.n_particles))

    # -- internals ---------------------------------------------------------
    def _log_prior(self, x: np.ndarray) -> np.ndarray:
        f = get_form(self.form)
        out = np.zeros(x.shape[0])
        for j, name in enumerate(f.param_names):
            i = PARAM_NAMES.index(name)
            lo, hi = PARAM_BOUNDS[i]
            val = x[:, j]
            ok = (val >= lo) & (val <= hi)
            out = out + np.where(ok, 0.0, -np.inf)
        if self.use_discrepancy:
            klo, khi = KAPPA_BOUNDS
            k = x[:, -1]
            out = out + np.where((k >= klo) & (k <= khi), 0.0, -np.inf)
        return out

    def _log_data_lik(self, x: np.ndarray, records: Sequence[Tuple[float, float, float, float]]) -> np.ndarray:
        """Log likelihood of every stored record, vectorised over particles."""
        ll = np.zeros(x.shape[0])
        if not records:
            return ll
        f = get_form(self.form)
        kappa = x[:, -1] if self.use_discrepancy else np.zeros(x.shape[0])
        for (h, v, y, sigma) in records:
            mu = _resistance_particles(f, x[:, : f.n_phys], h, v, self.ship)
            var = sigma**2 + np.square(kappa * mu)
            ll = ll - 0.5 * (np.log(2.0 * math.pi * var) + np.square(y - mu) / var)
        return ll

    def _weights(self) -> np.ndarray:
        m = float(np.max(self.logw))
        w = np.exp(self.logw - m)
        return w / w.sum()

    def ess(self) -> float:
        w = self._weights()
        return float(1.0 / np.sum(np.square(w)))

    def _resample(self) -> None:
        w = self._weights()
        n = self.particles.shape[0]
        positions = (self._rng.random() + np.arange(n)) / n
        cum = np.cumsum(w)
        idx = np.searchsorted(cum, positions)
        idx = np.clip(idx, 0, n - 1)
        self.particles = self.particles[idx]
        self.logw = np.full(n, -math.log(n))

    def _rejuvenate(self) -> int:
        """``n_sweeps`` joint random-walk Metropolis sweeps; returns accepted count."""
        n, d = self.particles.shape
        cov = np.cov(self.particles.T) + 1e-10 * np.eye(d)
        try:
            chol = np.linalg.cholesky(cov)
        except np.linalg.LinAlgError:  # pragma: no cover - defensive
            chol = np.diag(np.sqrt(np.maximum(np.diag(cov), 1e-10)))
        scale = 2.38 / math.sqrt(d)
        lp_current = self._log_prior(self.particles) + self._log_data_lik(
            self.particles, self.records
        )
        accepted = 0
        lo, hi = self.bounds[:, 0], self.bounds[:, 1]
        for _ in range(max(0, int(self.n_sweeps))):
            prop = self.particles + scale * (self._rng.standard_normal((n, d)) @ chol.T)
            prop = np.clip(prop, lo, hi)
            lp_prop = self._log_prior(prop) + self._log_data_lik(prop, self.records)
            log_alpha = lp_prop - lp_current
            u = np.log(self._rng.random(n))
            take = u < log_alpha
            self.particles[take] = prop[take]
            lp_current[take] = lp_prop[take]
            accepted += int(np.sum(take))
        return accepted

    # -- public ------------------------------------------------------------
    def update(self, h: float, V: float, y: float, sigma: float) -> Dict[str, Any]:
        """Fold one new record into the posterior; returns timing + summary."""
        t0 = time.perf_counter()
        rec = (float(h), float(V), float(y), float(sigma))
        f = get_form(self.form)
        phys = self.particles[:, : f.n_phys]
        kappa = self.particles[:, -1] if self.use_discrepancy else np.zeros(self.particles.shape[0])
        mu = _resistance_particles(f, phys, rec[0], rec[1], self.ship)
        var = rec[3] ** 2 + np.square(kappa * mu)
        self.logw = self.logw - 0.5 * (np.log(2.0 * math.pi * var) + np.square(rec[2] - mu) / var)
        self.records.append(rec)

        ess_pre = self.ess()
        resampled = False
        if ess_pre < self.resample_fraction * self.n_particles:
            self._resample()
            resampled = True
        n_accept = self._rejuvenate()
        w = self._weights()
        mean = np.sum(w[:, None] * self.particles, axis=0)
        var_post = np.sum(w[:, None] * np.square(self.particles - mean), axis=0)
        dt = time.perf_counter() - t0
        row = {
            "step": len(self.records),
            "n_records": len(self.records),
            "ess_before_resample": float(ess_pre),
            "resampled": bool(resampled),
            "rejuvenation_acceptance": float(n_accept) / max(
                1, self.n_particles * max(1, self.n_sweeps)
            ),
            "time_s": float(dt),
            "mean": {n: float(mean[i]) for i, n in enumerate(self._names())},
            "sd": {
                n: float(math.sqrt(max(var_post[i], 0.0)))
                for i, n in enumerate(self._names())
            },
            "record": {"h_m": rec[0], "V_ms": rec[1], "R_kN": rec[2], "sigma_kN": rec[3]},
        }
        self.history.append(row)
        return row

    def _names(self) -> List[str]:
        f = get_form(self.form)
        names = list(f.param_names)
        if self.use_discrepancy:
            names.append("kappa")
        return names

    def posterior_mean(self) -> Dict[str, float]:
        w = self._weights()
        mean = np.sum(w[:, None] * self.particles, axis=0)
        return {n: float(mean[i]) for i, n in enumerate(self._names())}

    def posterior_sd(self) -> Dict[str, float]:
        w = self._weights()
        mean = np.sum(w[:, None] * self.particles, axis=0)
        var = np.sum(w[:, None] * np.square(self.particles - mean), axis=0)
        return {
            n: float(math.sqrt(max(var[i], 0.0))) for i, n in enumerate(self._names())
        }


def run_online_calibration(
    data: CalibrationData,
    ship: ShipGeometry,
    *,
    form: Any = "full",
    n_particles: int = 400,
    n_sweeps: int = 2,
    seed: int = 20237,
    batch_chains: Optional[ChainSet] = None,
) -> Dict[str, Any]:
    """Stream the campaign through :class:`OnlineCalibrator` record by record."""
    cal = OnlineCalibrator(
        ship=ship, form=form, n_particles=n_particles, n_sweeps=n_sweeps, seed=seed
    )
    order = np.argsort(data.h_m)  # records arrive in increasing thickness
    rows: List[Dict[str, Any]] = []
    for i in order:
        rows.append(
            cal.update(
                float(data.h_m[i]),
                float(data.V_ms[i]),
                float(data.R_kN[i]),
                float(data.sigma_kN[i]),
            )
        )
    times = np.array([r["time_s"] for r in rows], dtype=float)
    means = np.array([[r["mean"][n] for n in cal._names()] for r in rows], dtype=float)
    final = cal.posterior_mean()
    out: Dict[str, Any] = {
        "n_steps": int(times.size),
        "n_particles": int(n_particles),
        "n_sweeps": int(n_sweeps),
        "seed": int(seed),
        "time_per_step_s": {
            "mean": float(times.mean()),
            "max": float(times.max()),
            "median": float(np.median(times)),
            "total": float(times.sum()),
        },
        "final_posterior_mean": final,
        "final_posterior_sd": cal.posterior_sd(),
        "mean_trajectory": {
            n: means[:, i].tolist() for i, n in enumerate(cal._names())
        },
        "ess_history": [r["ess_before_resample"] for r in rows],
        "time_series_s": [r["time_s"] for r in rows],
        "resample_count": int(sum(1 for r in rows if r["resampled"])),
        "history": rows,
        "criteria": {
            "time_per_step_under_1s": bool(times.max() < 1.0),
            "mean_time_per_step_s": float(times.mean()),
        },
    }
    if batch_chains is not None:
        batch = {n: float(v) for n, v in zip(cal._names(), batch_chains.mean())}
        err = {
            n: float(abs(final[n] - batch[n]) / max(abs(batch[n]), 1e-12))
            for n in batch
        }
        out["batch_comparison"] = {
            "batch_mean": batch,
            "relative_difference": err,
            "max_relative_difference": float(max(err.values())),
        }
    return out