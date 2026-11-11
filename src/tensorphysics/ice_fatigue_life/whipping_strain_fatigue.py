"""Ice-induced whipping fatigue: measured-strain Bayesian updating → rainflow → Miner → voyage life.

This module is the *chain* layer that joins four pieces which already exist as
stand-alone capabilities elsewhere in the package:

======  ===============================================================  ==========================
stage   what it does                                                     implementation used
======  ===============================================================  ==========================
1       whipping bending-moment time history of the hull girder driven     ``ice_vibration.integrators.newmark_mdof``
        by a *continuous* ice-load spectrum (not a single half-sine)        + ``ice_vibration.modal.ModalData``
2       Bayesian updating of the whipping-response model scale from         conjugate Gaussian (arithmetic
        measured deck strain (gauge record) → posterior moment ensemble     verified against a numerical grid)
3       rainflow cycle counting on the posterior moment histories →         ``fatigue.rainflow`` / ``fatigue.sn_curve``
        stress ranges at the detail → S-N / Palmgren-Miner damage           / ``fatigue.miner``
4       ice-zone voyage life: damage per exposure block × annual blocks     closed-form inversion
======  ===============================================================  ==========================

Why a separate chain layer
--------------------------
Each piece was already REAL on its own, but nothing connected them, and two
modelling gaps were open:

* ``ice_vibration.whiplash.whiplash_response`` integrates *one* half-sine bow
  impact.  Real ice transit delivers a **train** of impacts / continuous
  crushing, whose moment history carries amplitude scatter and superposed
  modes.  :func:`whipping_moment_history` accepts an arbitrary force record.
* The whipping response amplitude is the dominant uncertainty of the whole
  fatigue chain (it enters the damage to the power of the S-N slope ``m``).
  A strain-gauge record on deck measures exactly that amplitude, so the model
  scale is updated with Bayes instead of being guessed.

Uncertainty model (kept deliberately auditable)
-----------------------------------------------
The whipping moment history is ``M(t) = theta * M_model(t)`` where ``theta`` is
a single non-negative scale factor.  Deck strain is converted to bending moment
with the section modulus ``W`` and Young's modulus ``E``:
``eps(t) = M(t) / (W E)`` (a section modulus alone yields *stress*, not strain —
that unit trap is why ``youngs_modulus_pa`` is a mandatory argument).  With Gaussian strain
noise ``N(0, sigma_eps^2)`` and a Gaussian prior ``theta ~ N(mu_0, sigma_0^2)``
the posterior is Gaussian in closed form::

    tau_1 = 1/sigma_0^2 + sum_k (eps_model,k)^2 / sigma_eps^2
    mu_1  = (mu_0/sigma_0^2 + sum_k eps_model,k eps_obs,k / sigma_eps^2) / tau_1
    sigma_1^2 = 1/tau_1

That closed form is *not* taken on trust: :func:`bayesian_posterior_grid`
recomputes the same posterior by brute-force numerical integration of the
unnormalised posterior on a fine ``theta`` grid (an independent code path), and
the acceptance metric compares the two.

Only ``theta`` (amplitude/load-path uncertainty) is updated.  Modal damping and
frequencies are *not* inferred from a single strain channel; a phase-resolved
modal updating would need multiple channels and is out of scope (stated as a
limitation, not hidden).

Units: SI internally (m, N, N*m, Pa); strain is reported in microstrain and
stress in MPa because that is the S-N curve convention.

References
----------
- DNV-RP-C203 (2021) *Fatigue design of offshore steel structures* — S-N curves,
  thickness correction, design fatigue factors.
- ASTM E1049-85 — rainflow cycle counting.
- Palmgren (1924) / Miner (1945) — linear cumulative damage.
- ISO 19906 (2019) — ice action and ice-induced vibration of offshore structures.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..fatigue.miner import MinerResult, miner_damage
from ..fatigue.rainflow import RainflowResult, rainflow_counting
from ..fatigue.sn_curve import DNCurve
from ..ice_vibration.integrators import newmark_mdof
from ..ice_vibration.modal import ModalData
from ..ice_vibration.whiplash import quasi_static_midship_moment

#: Young's modulus of hull structural steel [Pa]; used only where a caller does
#: not override it, and always recorded in the output JSON.
DEFAULT_STEEL_YOUNGS_MODULUS_PA = 2.06e11

__all__ = [
    "DEFAULT_STEEL_YOUNGS_MODULUS_PA",
    "WhippingMomentHistory",
    "whipping_moment_history",
    "half_sine_force_signal",
    "StrainBayesPosterior",
    "bayesian_response_scale_update",
    "bayesian_posterior_grid",
    "DetailFatigueResult",
    "moment_ranges_to_stress_ranges_mpa",
    "rainflow_miner_damage_from_moment",
    "damage_vs_scale_samples",
    "VoyageFatigueLife",
    "voyage_fatigue_life",
    "ice_exposure_blocks_per_year",
]


# ---------------------------------------------------------------------------
# Stage 1 — whipping bending-moment time history for an arbitrary ice-load record
# ---------------------------------------------------------------------------


@dataclass
class WhippingMomentHistory:
    """Whipping response of the hull girder to an arbitrary bow force record.

    ``midship_moment_nm`` / ``quarter_moment_nm`` are the physical bending
    moments obtained from modal superposition,
    ``M(x, t) = EI * sum_n phi_n''(x) q_n(t)``.
    """

    time_s: np.ndarray
    force_n: np.ndarray
    modal_coordinates: np.ndarray          # (n_steps, n_modes) generalised coords q_n(t)
    midship_moment_nm: np.ndarray
    quarter_moment_nm: np.ndarray
    bending_stiffness_nm2: float
    dt_s: float
    modal: ModalData
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    # -- derived ------------------------------------------------------------
    @property
    def duration_s(self) -> float:
        return float(self.time_s[-1] - self.time_s[0]) if self.time_s.size else 0.0

    @property
    def peak_midship_moment_nm(self) -> float:
        return float(np.max(np.abs(self.midship_moment_nm))) if self.midship_moment_nm.size else 0.0

    @property
    def peak_quarter_moment_nm(self) -> float:
        return float(np.max(np.abs(self.quarter_moment_nm))) if self.quarter_moment_nm.size else 0.0

    @property
    def rms_midship_moment_nm(self) -> float:
        if not self.midship_moment_nm.size:
            return 0.0
        return float(np.sqrt(np.mean(self.midship_moment_nm**2)))

    @property
    def time_of_peak_s(self) -> float:
        if not self.midship_moment_nm.size:
            return float("nan")
        return float(self.time_s[int(np.argmax(np.abs(self.midship_moment_nm)))])

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "class": "WhippingMomentHistory",
            "n_steps": int(self.time_s.size),
            "dt_s": self.dt_s,
            "duration_s": self.duration_s,
            "bending_stiffness_nm2": self.bending_stiffness_nm2,
            "peak_midship_moment_nm": self.peak_midship_moment_nm,
            "peak_quarter_moment_nm": self.peak_quarter_moment_nm,
            "rms_midship_moment_nm": self.rms_midship_moment_nm,
            "time_of_peak_s": self.time_of_peak_s,
            "diagnostics": dict(self.diagnostics),
        }


def half_sine_force_signal(
    amplitude_n: float,
    pulse_duration_s: float,
    duration_s: float,
    dt_s: float,
) -> np.ndarray:
    """Single half-sine bow impact ``F(t) = F0 sin(pi t / T_p)`` sampled on ``dt_s``.

    Provided so that a chain-level result can be compared against the
    single-impact reference path in :mod:`tensorphysics.ice_vibration.whiplash`.
    """
    if amplitude_n <= 0.0 or pulse_duration_s <= 0.0 or duration_s <= 0.0 or dt_s <= 0.0:
        raise ValueError("amplitude, pulse duration, duration and dt must all be positive")
    n_steps = int(round(duration_s / dt_s)) + 1
    t = np.arange(n_steps) * dt_s
    force = np.zeros(n_steps)
    mask = t <= pulse_duration_s
    force[mask] = amplitude_n * np.sin(math.pi * t[mask] / pulse_duration_s)
    return force


def whipping_moment_history(
    modal: ModalData,
    bending_stiffness_nm2: float,
    force_n: Sequence[float],
    dt_s: float,
    bow_mode_shape: Optional[Sequence[float]] = None,
) -> WhippingMomentHistory:
    """Modal-superposition whipping moment history for an arbitrary bow force record.

    Solves the diagonal generalised systems

        q_n'' + 2 zeta_n w_n q_n' + w_n^2 q_n = phi_n(x_bow) F(t) / m_n

    with Newmark-beta (average acceleration) and maps the generalised
    coordinates to midship / quarter bending moments via the modal curvatures.
    Any force record is accepted, so the same entry point covers a single
    half-sine impact (reproduction check against ``whiplash_response``), a
    quasi-periodic crushing train and a Poisson impact train.

    Parameters
    ----------
    modal : ModalData
        Modal set; must carry midship curvature ordinates (and quarter
        curvature ordinates for the quarter moment).
    bending_stiffness_nm2 : float
        Hull-girder vertical bending stiffness EI [N*m^2].
    force_n : sequence of float
        Bow force time history [N] on a uniform grid.
    dt_s : float
        Sampling interval [s].
    bow_mode_shape : sequence of float, optional
        Override the modal bow ordinates (default: ``modal.mode_shape_bow``).
    """
    f = np.asarray(force_n, dtype=float)
    if f.ndim != 1 or f.size < 2:
        raise ValueError("force_n must be a 1-D record with at least two samples")
    if not np.all(np.isfinite(f)):
        raise ValueError("force_n contains non-finite values")
    if not (np.isfinite(dt_s) and dt_s > 0.0):
        raise ValueError("dt_s must be positive and finite")
    if modal.n_modes < 1:
        raise ValueError("modal data must contain at least one mode")
    if bending_stiffness_nm2 <= 0.0:
        raise ValueError("bending stiffness EI must be positive")
    if len(modal.mode_shape_midship_curvature) != modal.n_modes:
        raise ValueError("modal data must carry midship curvature ordinates for every mode")

    phi_bow = np.asarray(
        modal.mode_shape_bow if bow_mode_shape is None else bow_mode_shape, dtype=float
    )
    if phi_bow.size != modal.n_modes:
        raise ValueError(f"bow mode shape has {phi_bow.size} ordinates, expected {modal.n_modes}")
    phi_mid_pp = np.asarray(modal.mode_shape_midship_curvature, dtype=float)
    phi_quarter_pp = np.asarray(modal.mode_shape_quarter_curvature, dtype=float)

    n_steps = f.size
    duration_s = (n_steps - 1) * dt_s
    force_matrix = np.outer(f, phi_bow)  # Q_n(t) = phi_n(bow) F(t)

    marched = newmark_mdof(
        stiffness=np.diag(modal.generalized_stiffness()),
        mass=np.diag(np.asarray(modal.modal_masses_kg, dtype=float)),
        damping=np.diag(modal.generalized_damping()),
        force=force_matrix,
        duration_s=duration_s,
        dt_s=dt_s,
    )
    q = np.asarray(marched.displacement, dtype=float)
    if q.ndim == 1:
        q = q[:, None]

    mid_moment = bending_stiffness_nm2 * (q @ phi_mid_pp)
    if phi_quarter_pp.size == modal.n_modes:
        quarter_moment = bending_stiffness_nm2 * (q @ phi_quarter_pp)
    else:
        quarter_moment = np.zeros_like(mid_moment)

    # Quasi-static reference evaluated at the peak force of *this* record, so the
    # dynamic amplification ratio is comparable with the single-impact one.
    peak_force = float(np.max(np.abs(f)))
    static_moment = (
        quasi_static_midship_moment(modal, bending_stiffness_nm2, peak_force) if peak_force > 0 else 0.0
    )
    peak_moment = float(np.max(np.abs(mid_moment)))
    diagnostics = {
        "method_marched": marched.method,
        "n_modes": modal.n_modes,
        "n_steps": int(n_steps),
        "dt_s": float(dt_s),
        "duration_s": float(duration_s),
        "modal_source": modal.source,
        "peak_force_n": peak_force,
        "quasi_static_midship_moment_at_peak_force_nm": static_moment,
        "whiplash_dynamic_amplification": (
            peak_moment / static_moment if static_moment > 0.0 else float("nan")
        ),
    }
    return WhippingMomentHistory(
        time_s=np.arange(n_steps) * dt_s,
        force_n=f,
        modal_coordinates=q,
        midship_moment_nm=mid_moment,
        quarter_moment_nm=quarter_moment,
        bending_stiffness_nm2=float(bending_stiffness_nm2),
        dt_s=float(dt_s),
        modal=modal,
        diagnostics=diagnostics,
    )


# ---------------------------------------------------------------------------
# Stage 2 — Bayesian updating of the whipping response scale from measured strain
# ---------------------------------------------------------------------------


@dataclass
class StrainBayesPosterior:
    """Gaussian posterior of the whipping-response scale ``theta``.

    ``theta`` multiplies the model moment history, so the posterior predictive
    moment history is ``theta_i * M_model(t)`` with ``theta_i`` drawn from the
    posterior.  Because rainflow ranges scale linearly with the signal, the
    cycle count is computed **once** and only the stress ranges are rescaled —
    :func:`damage_vs_scale_samples` exploits that exact property.
    """

    prior_mean: float
    prior_std: float
    posterior_mean: float
    posterior_std: float
    noise_std_microstrain: float
    section_modulus_m3: float
    youngs_modulus_pa: float
    sample_indices: np.ndarray
    observed_strain_microstrain: np.ndarray
    model_strain_microstrain: np.ndarray
    scale_samples: np.ndarray
    credible_level: float
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    # -- derived ------------------------------------------------------------
    @property
    def credible_interval(self) -> Tuple[float, float]:
        """Equal-tailed credible interval of ``theta`` (Gaussian quantile)."""
        nd = statistics.NormalDist()
        half = 0.5 * (1.0 + self.credible_level)
        z = nd.inv_cdf(half)
        return (self.posterior_mean - z * self.posterior_std, self.posterior_mean + z * self.posterior_std)

    @property
    def shrinkage(self) -> float:
        """Posterior/prior standard-deviation ratio (< 1 means data reduced the spread)."""
        if self.prior_std <= 0.0:
            return float("nan")
        return self.posterior_std / self.prior_std

    @property
    def shift_from_prior(self) -> float:
        """Relative shift of the mean, ``(mu_1 - mu_0)/mu_0``."""
        if self.prior_mean == 0.0:
            return float("nan")
        return (self.posterior_mean - self.prior_mean) / self.prior_mean

    def posterior_moment_ensemble(self, model_moment_nm: Sequence[float]) -> np.ndarray:
        """Posterior moment histories ``(n_samples, n_steps)``, ``theta_i * M_model``."""
        m = np.asarray(model_moment_nm, dtype=float)
        return self.scale_samples[:, None] * m[None, :]

    def peak_moment_distribution(self, model_moment_nm: Sequence[float]) -> np.ndarray:
        """Posterior distribution of the peak |moment| [N*m]."""
        m = np.asarray(model_moment_nm, dtype=float)
        return self.scale_samples * float(np.max(np.abs(m)))

    def to_json_dict(self) -> Dict[str, Any]:
        lo, hi = self.credible_interval
        return {
            "class": "StrainBayesPosterior",
            "prior_mean": self.prior_mean,
            "prior_std": self.prior_std,
            "posterior_mean": self.posterior_mean,
            "posterior_std": self.posterior_std,
            "posterior_sd_ratio": self.shrinkage,
            "posterior_mean_shift_relative": self.shift_from_prior,
            "credible_level": self.credible_level,
            "credible_interval": [lo, hi],
            "n_observations": int(self.sample_indices.size),
            "noise_std_microstrain": self.noise_std_microstrain,
            "section_modulus_m3": self.section_modulus_m3,
            "youngs_modulus_pa": self.youngs_modulus_pa,
            "prior_precision": (1.0 / self.prior_std**2) if self.prior_std > 0 else None,
            "posterior_precision": (1.0 / self.posterior_std**2) if self.posterior_std > 0 else None,
            "diagnostics": dict(self.diagnostics),
        }


def _model_strain_microstrain(
    model_moment_nm: np.ndarray,
    section_modulus_m3: float,
    youngs_modulus_pa: float,
    sample_indices: np.ndarray,
) -> np.ndarray:
    """Deck strain [microstrain] from the bending moment: ``eps = M / (W E)``."""
    return model_moment_nm[sample_indices] / (section_modulus_m3 * youngs_modulus_pa) * 1.0e6


def bayesian_response_scale_update(
    model_moment_nm: Sequence[float],
    observed_strain_microstrain: Sequence[float],
    section_modulus_m3: float,
    youngs_modulus_pa: float,
    noise_std_microstrain: float,
    sample_indices: Optional[Sequence[int]] = None,
    prior_mean: float = 1.0,
    prior_std: float = 0.25,
    n_samples: int = 4000,
    credible_level: float = 0.95,
    seed: int = 20260101,
) -> StrainBayesPosterior:
    """Conjugate Gaussian update of the whipping-response scale from deck strain.

    Measurement model::

        eps_obs,k = theta * M_model(t_k) / W + noise,   noise ~ N(0, sigma_eps^2)

    Parameters
    ----------
    model_moment_nm : sequence of float
        Model whipping moment history at the strain-gauge station [N*m].
    observed_strain_microstrain : sequence of float
        Measured strain samples [microstrain] (one per ``sample_indices`` entry).
    section_modulus_m3 : float
        Deck section modulus at the gauge station [m^3].
    youngs_modulus_pa : float
        Young's modulus of the deck material [Pa]; ``eps = M/(W E)``.
    noise_std_microstrain : float
        Strain measurement noise standard deviation [microstrain] (> 0).
    sample_indices : sequence of int, optional
        Indices of the model history that the measurements correspond to
        (default: evenly spread over the record).
    prior_mean, prior_std : float
        Gaussian prior of ``theta`` (``prior_std > 0``).
    n_samples : int
        Number of posterior samples drawn for the predictive ensemble.
    """
    m = np.asarray(model_moment_nm, dtype=float)
    if m.ndim != 1 or m.size < 2:
        raise ValueError("model_moment_nm must be a 1-D history with at least two samples")
    obs = np.asarray(observed_strain_microstrain, dtype=float)
    if obs.ndim != 1 or obs.size < 1:
        raise ValueError("observed strain must be a non-empty 1-D array")
    if not np.all(np.isfinite(obs)):
        raise ValueError("observed strain contains non-finite values")
    if section_modulus_m3 <= 0.0:
        raise ValueError("section modulus must be positive")
    if not (youngs_modulus_pa > 0.0):
        raise ValueError("Young's modulus must be positive")
    if not (noise_std_microstrain > 0.0):
        raise ValueError("strain noise standard deviation must be positive")
    if not (prior_std > 0.0):
        raise ValueError("prior standard deviation must be positive")
    if prior_mean <= 0.0:
        raise ValueError("prior mean of the positive scale factor must be positive")
    if not 0.0 < credible_level < 1.0:
        raise ValueError("credible_level must lie in (0, 1)")

    if sample_indices is None:
        idx = np.linspace(0, m.size - 1, obs.size).round().astype(int)
    else:
        idx = np.asarray(sample_indices, dtype=int)
        if idx.ndim != 1 or idx.size != obs.size:
            raise ValueError("sample_indices must be 1-D and match the number of observations")
        if idx.min() < 0 or idx.max() >= m.size:
            raise ValueError("sample_indices out of range of the moment history")

    eps_model = _model_strain_microstrain(m, section_modulus_m3, youngs_modulus_pa, idx)

    # Conjugate Gaussian arithmetic (closed form).
    sigma2 = float(noise_std_microstrain) ** 2
    prior_precision = 1.0 / float(prior_std) ** 2
    info_data = float(np.sum(eps_model**2)) / sigma2
    post_precision = prior_precision + info_data
    post_mean = (float(prior_mean) * prior_precision + float(np.sum(eps_model * obs)) / sigma2) / post_precision
    post_var = 1.0 / post_precision
    post_std = math.sqrt(post_var)

    # Marginal log-likelihood (log evidence) of the observation set — Bayesian
    # model-comparison diagnostic, also used by the numerical cross-check.
    resid = obs  # theta is integrated out analytically below
    log_evidence = _gaussian_log_evidence(obs, eps_model, sigma2, float(prior_mean), float(prior_std))

    rng = np.random.default_rng(seed)
    scale_samples = rng.normal(post_mean, post_std, size=int(n_samples))

    diagnostics = {
        "method": "conjugate_gaussian_scale_update",
        "data_information": info_data,
        "prior_information": prior_precision,
        "log_evidence": log_evidence,
        "note": (
            "theta is the multiplicative scale of the model whipping moment; "
            "modal damping/frequency are not inferred from a single strain channel; "
            "strain-noise is assumed iid Gaussian (correlated gauge calibration bias is not modelled)"
        ),
        "youngs_modulus_pa": float(youngs_modulus_pa),
    }
    return StrainBayesPosterior(
        prior_mean=float(prior_mean),
        prior_std=float(prior_std),
        posterior_mean=float(post_mean),
        posterior_std=float(post_std),
        noise_std_microstrain=float(noise_std_microstrain),
        section_modulus_m3=float(section_modulus_m3),
        youngs_modulus_pa=float(youngs_modulus_pa),
        sample_indices=idx,
        observed_strain_microstrain=obs,
        model_strain_microstrain=eps_model,
        scale_samples=scale_samples,
        credible_level=float(credible_level),
        diagnostics=diagnostics,
    )


def _gaussian_log_evidence(
    obs: np.ndarray,
    eps_model: np.ndarray,
    noise_var: float,
    prior_mean: float,
    prior_std: float,
) -> float:
    """log p(obs) with theta integrated out (closed form, up to constants kept explicit)."""
    prior_var = prior_std**2
    tau0 = 1.0 / prior_var
    tau1 = tau0 + float(np.sum(eps_model**2)) / noise_var
    mu1 = (prior_mean * tau0 + float(np.sum(eps_model * obs)) / noise_var) / tau1
    quad = float(np.sum(obs**2)) / noise_var + prior_mean**2 / prior_var - mu1**2 * tau1
    log_two_pi = math.log(2.0 * math.pi)
    log_normaliser = (
        -0.5 * obs.size * log_two_pi
        - 0.5 * obs.size * math.log(noise_var)
        - 0.5 * log_two_pi
        - 0.5 * math.log(prior_var)
        + 0.5 * log_two_pi
        + 0.5 * math.log(tau1)
    )
    return log_normaliser - 0.5 * quad


def bayesian_posterior_grid(
    model_moment_nm: Sequence[float],
    observed_strain_microstrain: Sequence[float],
    section_modulus_m3: float,
    youngs_modulus_pa: float,
    noise_std_microstrain: float,
    sample_indices: Optional[Sequence[int]] = None,
    prior_mean: float = 1.0,
    prior_std: float = 0.25,
    n_grid: int = 400001,
    n_sigma: float = 8.0,
) -> Dict[str, float]:
    """Independent, brute-force recomputation of the posterior (verification path).

    The unnormalised log posterior is evaluated on a uniform ``theta`` grid and
    integrated with the trapezoidal rule.  None of the conjugate algebra of
    :func:`bayesian_response_scale_update` is reused, so agreement of the two
    results is a genuine cross-check of the Bayesian update (it also returns
    the normalisation integral ``Z`` so a non-converged quadrature cannot hide).
    """
    m = np.asarray(model_moment_nm, dtype=float)
    obs = np.asarray(observed_strain_microstrain, dtype=float)
    if sample_indices is None:
        idx = np.linspace(0, m.size - 1, obs.size).round().astype(int)
    else:
        idx = np.asarray(sample_indices, dtype=int)
    eps_model = _model_strain_microstrain(m, section_modulus_m3, youngs_modulus_pa, idx)
    sigma2 = float(noise_std_microstrain) ** 2

    centre = float(prior_mean)
    span = float(n_sigma) * float(prior_std)
    theta = np.linspace(max(centre - span, 1e-9), centre + span, int(n_grid))
    log_post = -0.5 * (theta - prior_mean) ** 2 / prior_std**2
    # residuals for all grid points at once: (n_grid, n_obs)
    resid = obs[None, :] - theta[:, None] * eps_model[None, :]
    log_post = log_post - 0.5 * np.sum(resid**2, axis=1) / sigma2
    log_post -= float(np.max(log_post))
    post = np.exp(log_post)
    Z = float(np.trapezoid(post, theta))
    mean = float(np.trapezoid(theta * post, theta) / Z)
    var = float(np.trapezoid((theta - mean) ** 2 * post, theta) / Z)
    return {
        "posterior_mean": mean,
        "posterior_std": math.sqrt(max(var, 0.0)),
        "log_normalisation": math.log(Z),
        "n_grid": int(n_grid),
        "grid_span": [float(theta[0]), float(theta[-1])],
    }


# ---------------------------------------------------------------------------
# Stage 3 — rainflow + S-N / Miner damage on moment histories
# ---------------------------------------------------------------------------


@dataclass
class DetailFatigueResult:
    """Rainflow → S-N → Palmgren-Miner result at one structural detail."""

    stress_ranges_mpa: np.ndarray
    counts: np.ndarray
    moment_ranges_nm: np.ndarray
    total_cycles: float
    damage: float
    sn_curve_class: str
    sn_curve_environment: str
    section_modulus_m3: float
    stress_concentration_factor: float
    thickness_factor: float
    damage_per_cycle: np.ndarray = field(default_factory=lambda: np.zeros(0))
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_json_dict(self) -> Dict[str, Any]:
        if self.stress_ranges_mpa.size:
            order = np.argsort(-self.stress_ranges_mpa)
            top = [
                {
                    "stress_range_mpa": float(self.stress_ranges_mpa[i]),
                    "counts": float(self.counts[i]),
                    "damage": float(self.damage_per_cycle[i]) if self.damage_per_cycle.size else None,
                }
                for i in order[:10]
            ]
            max_range = float(np.max(self.stress_ranges_mpa))
            mean_range = float(np.average(self.stress_ranges_mpa, weights=self.counts)) if self.counts.sum() else 0.0
        else:
            top, max_range, mean_range = [], 0.0, 0.0
        return {
            "class": "DetailFatigueResult",
            "sn_curve_class": self.sn_curve_class,
            "sn_curve_environment": self.sn_curve_environment,
            "section_modulus_m3": self.section_modulus_m3,
            "stress_concentration_factor": self.stress_concentration_factor,
            "thickness_factor": self.thickness_factor,
            "n_cycles_total": self.total_cycles,
            "n_bins": int(self.stress_ranges_mpa.size),
            "max_stress_range_mpa": max_range,
            "cycle_weighted_mean_range_mpa": mean_range,
            "damage": self.damage,
            "top_bins": top,
            "diagnostics": dict(self.diagnostics),
        }


def moment_ranges_to_stress_ranges_mpa(
    moment_ranges_nm: Sequence[float],
    section_modulus_m3: float,
    stress_concentration_factor: float = 1.0,
    thickness_mm: Optional[float] = None,
    thickness_ref_mm: float = 25.0,
    thickness_exponent: float = 0.25,
) -> np.ndarray:
    """Convert moment ranges [N*m] to hot-spot stress ranges [MPa].

    ``d_sigma = K_s * (t/t_ref)^k * dM / W`` with the DNV-RP-C203 thickness
    correction applied for ``t > t_ref`` (the same convention as
    :func:`tensorphysics.ice_fatigue_life.sn_dnv_rp_c203.thickness_correction`).
    """
    if section_modulus_m3 <= 0.0:
        raise ValueError("section modulus must be positive")
    if stress_concentration_factor <= 0.0:
        raise ValueError("stress concentration factor must be positive")
    d_m = np.abs(np.asarray(moment_ranges_nm, dtype=float))
    t_factor = 1.0
    if thickness_mm is not None:
        if thickness_mm <= 0.0:
            raise ValueError("plate thickness must be positive")
        if thickness_mm > thickness_ref_mm:
            t_factor = (thickness_mm / thickness_ref_mm) ** thickness_exponent
    return d_m / section_modulus_m3 * stress_concentration_factor * t_factor / 1.0e6


def rainflow_miner_damage_from_moment(
    moment_history_nm: Sequence[float],
    section_modulus_m3: float,
    sn_curve: Optional[DNCurve] = None,
    stress_concentration_factor: float = 1.0,
    thickness_mm: Optional[float] = None,
    thickness_ref_mm: float = 25.0,
    thickness_exponent: float = 0.25,
) -> DetailFatigueResult:
    """Rainflow-count a moment history and accumulate Miner damage at a detail.

    The counting is done on the moment record (counting is invariant under a
    positive rescaling) and the resulting ranges are converted to stress ranges
    at the detail, so the same count can be reused for every posterior draw.
    """
    if sn_curve is None:
        sn_curve = DNCurve(curve_class="F", environment="seawater_cp")
    m = np.asarray(moment_history_nm, dtype=float)
    if m.ndim != 1 or m.size < 3:
        raise ValueError("moment history must be a 1-D record with at least three samples")
    rf: RainflowResult = rainflow_counting(m)
    if rf.ranges.size == 0:
        raise ValueError("rainflow counting produced no cycles for this record")

    stress = moment_ranges_to_stress_ranges_mpa(
        rf.ranges,
        section_modulus_m3,
        stress_concentration_factor=stress_concentration_factor,
        thickness_mm=thickness_mm,
        thickness_ref_mm=thickness_ref_mm,
        thickness_exponent=thickness_exponent,
    )
    miner: MinerResult = miner_damage(stress, rf.counts, sn_curve)
    t_factor = 1.0
    if thickness_mm is not None and thickness_mm > thickness_ref_mm:
        t_factor = (thickness_mm / thickness_ref_mm) ** thickness_exponent

    diagnostics = {
        "rainflow_n_full": int(rf.n_full),
        "rainflow_n_half": int(rf.n_half),
        "miner_is_safe": bool(miner.is_safe),
        "n_to_failure_per_block_min": (
            float(np.min(miner.n_to_failure_per_block[np.isfinite(miner.n_to_failure_per_block)]))
            if np.any(np.isfinite(miner.n_to_failure_per_block))
            else float("inf")
        ),
    }
    return DetailFatigueResult(
        stress_ranges_mpa=stress,
        counts=rf.counts,
        moment_ranges_nm=np.abs(rf.ranges),
        total_cycles=float(rf.total_cycles),
        damage=float(miner.total_damage),
        sn_curve_class=sn_curve.curve_class,
        sn_curve_environment=sn_curve.environment,
        section_modulus_m3=float(section_modulus_m3),
        stress_concentration_factor=float(stress_concentration_factor),
        thickness_factor=float(t_factor),
        damage_per_cycle=np.asarray(miner.damage_per_block, dtype=float),
        diagnostics=diagnostics,
    )


def damage_vs_scale_samples(
    base: DetailFatigueResult,
    scale_samples: Sequence[float],
    sn_curve: Optional[DNCurve] = None,
) -> np.ndarray:
    """Damage for every posterior scale draw, reusing one rainflow count.

    Rainflow ranges scale linearly with the amplitude of the record, so
    ``D(theta) = sum_k n_k / N(theta * d_sigma_k)`` — exact, no re-counting and
    no Monte-Carlo noise beyond the posterior draws themselves.
    """
    if sn_curve is None:
        sn_curve = DNCurve(curve_class=base.sn_curve_class, environment=base.sn_curve_environment)
    scales = np.asarray(scale_samples, dtype=float)
    if scales.ndim != 1 or scales.size == 0:
        raise ValueError("scale samples must be a non-empty 1-D array")
    dmg = np.empty(scales.size)
    for i, theta in enumerate(scales):
        stress = theta * base.stress_ranges_mpa
        # n_to_failure is vectorised over the bins
        n_fail = np.array([sn_curve.n_to_failure(s) for s in stress])
        ok = np.isfinite(n_fail) & (n_fail > 0.0)
        dmg[i] = float(np.sum(base.counts[ok] / n_fail[ok]))
    return dmg


# ---------------------------------------------------------------------------
# Stage 4 — ice-zone voyage life
# ---------------------------------------------------------------------------


@dataclass
class VoyageFatigueLife:
    """Voyage/design life from per-block damage and annual exposure."""

    damage_per_block: float
    blocks_per_year: float
    annual_damage: float
    design_fatigue_factor: float
    critical_damage: float
    years_to_failure: float
    utilization_per_year: float
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "class": "VoyageFatigueLife",
            "damage_per_block": self.damage_per_block,
            "blocks_per_year": self.blocks_per_year,
            "annual_damage": self.annual_damage,
            "design_fatigue_factor": self.design_fatigue_factor,
            "critical_damage": self.critical_damage,
            "years_to_failure": self.years_to_failure,
            "utilization_per_year": self.utilization_per_year,
            "diagnostics": dict(self.diagnostics),
        }


def ice_exposure_blocks_per_year(
    operating_days_per_year: float,
    transit_hours_per_day: float,
    ice_fraction_of_transit: float,
    block_duration_s: float,
) -> float:
    """Number of ``block_duration_s`` exposure blocks per year spent in ice.

    Args:
        operating_days_per_year: sea days of the ship per year.
        transit_hours_per_day: hours per sea day actually transiting.
        ice_fraction_of_transit: fraction of the transit time spent in ice.
        block_duration_s: length of the simulated load block.

    The count is deterministic and states its exposure assumption explicitly
    (no hidden utilisation factor).
    """
    for name, value in (
        ("operating_days_per_year", operating_days_per_year),
        ("transit_hours_per_day", transit_hours_per_day),
        ("block_duration_s", block_duration_s),
    ):
        if not (np.isfinite(value) and value > 0.0):
            raise ValueError(f"{name} must be positive and finite")
    if not 0.0 <= ice_fraction_of_transit <= 1.0:
        raise ValueError("ice_fraction_of_transit must lie in [0, 1]")
    return (
        operating_days_per_year * transit_hours_per_day * 3600.0 * ice_fraction_of_transit / block_duration_s
    )


def voyage_fatigue_life(
    damage_per_block: float,
    blocks_per_year: float,
    design_fatigue_factor: float = 2.0,
    exposure_note: Optional[str] = None,
) -> VoyageFatigueLife:
    """Years to reach the allowable damage ``D_allow = 1 / DFF``.

    DNV-RP-C203 requires ``D <= 1/DFF`` with ``DFF`` the design fatigue factor
    (2.0 for inspectable ice-strengthened structure, 3.0 where inspection is
    not possible).  ``years_to_failure`` inverts the linear accumulation exactly:
    ``t = (1/DFF) / (D_block * n_blocks_per_year)``.
    """
    if damage_per_block < 0.0:
        raise ValueError("damage per block must be non-negative")
    if blocks_per_year < 0.0:
        raise ValueError("blocks per year must be non-negative")
    if design_fatigue_factor <= 0.0:
        raise ValueError("design fatigue factor must be positive")
    annual = float(damage_per_block) * float(blocks_per_year)
    critical = 1.0 / float(design_fatigue_factor)
    years = float("inf") if annual <= 0.0 else critical / annual
    return VoyageFatigueLife(
        damage_per_block=float(damage_per_block),
        blocks_per_year=float(blocks_per_year),
        annual_damage=annual,
        design_fatigue_factor=float(design_fatigue_factor),
        critical_damage=critical,
        years_to_failure=years,
        utilization_per_year=(annual / critical if critical > 0 else float("inf")),
        diagnostics={
            "formula": "years = (1/DFF) / (D_block * blocks_per_year)",
            "exposure_note": exposure_note,
        },
    )