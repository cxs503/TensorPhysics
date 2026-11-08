"""Physics invariants of a transmitted RAO table (model-independent gates).

The transport check in :mod:`tensorphysics.integration.rao` only proves that the
receiver reads the payload the way the sender wrote it.  It says nothing about
whether the *curve itself* is physical.  The invariants below are the ones that
hold for any linear potential-flow seakeeping solution, independently of which
strip theory or panel method produced it, so they can be asserted on a payload
received from a foreign repository:

============================  ==============================================
invariant                     statement
============================  ==============================================
long-wave heave               ``kL -> 0`` : ``RAO_heave -> 1`` (body follows
                              the free surface), Newman (1977) §6.10
long-wave pitch               ``kL -> 0`` : ``RAO_pitch -> 1`` (the hull aligns
                              with the wave slope; the slope normalisation
                              ``rad/(k a)`` makes the limit exactly one)
short-wave decay              ``kL >> 1`` : ``RAO -> 0`` (the hull cannot
                              follow short waves)
symmetry of head seas         ``beta = 180 deg`` : roll excitation vanishes by
                              symmetry, so ``RAO_roll = 0``
deep-water dispersion         ``k = omega^2 / g`` exactly (used by both sides)
encounter frequency           ``omega_e = omega - k U cos(beta)`` with the sign
                              kept (following seas can reverse it)
============================  ==============================================
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .interchange import RAOTransfer
from .rao import G, pchip_interp

__all__ = [
    "InvariantResult",
    "check_invariants",
    "encounter_frequency_model",
    "jonswap_omega",
    "table_fidelity",
]

#: Frequency below which the long-wave limit is asserted (kL <= 0.01 for a
#: 100 m hull at omega = 0.03 rad/s; the peer grid reaches 0.02 rad/s).
LONG_WAVE_KL = 0.02
#: Short-wave probe: kL >= 20 means the wavelength is under a twentieth of the hull.
SHORT_WAVE_KL = 20.0


@dataclass(frozen=True)
class InvariantResult:
    name: str
    value: float
    reference: float
    rel_dev: float
    passed: bool
    note: str

    def as_dict(self) -> dict:
        return {"name": self.name, "value": self.value, "reference": self.reference,
                "rel_dev": self.rel_dev, "passed": self.passed, "note": self.note}


def encounter_frequency_model(omega: float, speed_ms: float,
                              heading_deg: float, g: float = G) -> float:
    """``omega_e = omega - k U cos(beta)``, sign kept (head seas = 180 deg).

    With ``beta = 180 deg`` the wave travels against the ship, ``cos(beta) = -1``
    and the encounter frequency rises; in following seas the ship can overtake
    the waves and ``omega_e`` becomes zero or negative, at which point the
    frequency-domain solution is meaningless and the receiver must say so
    instead of folding the sign (that is what the sender's
    ``encounter_frequencies_valid`` flag is for).
    """
    k = omega * omega / g
    return float(omega - k * speed_ms * np.cos(np.radians(heading_deg)))


def check_invariants(transfer: RAOTransfer, reference: dict | None = None) -> list[InvariantResult]:
    """Evaluate the model-independent invariants on the transmitted curve."""
    L = float(transfer.hull_params["L"])
    out: list[InvariantResult] = []

    # ---- long-wave limit (lowest transmitted frequency) -------------------
    w_lo = float(transfer.omega[0])
    kl_lo = w_lo * w_lo / G * L

    def dev(v: float, ref: float) -> float:
        return abs(v - ref) / abs(ref) if ref else abs(v - ref)

    if kl_lo <= LONG_WAVE_KL:
        h = float(transfer.heave_rao[0])
        p = float(transfer.pitch_rao[0])
        out.append(InvariantResult(
            "long_wave_heave_rao_to_unity", h, 1.0, dev(h, 1.0), dev(h, 1.0) < 0.05,
            f"kL={kl_lo:.4f} at omega={w_lo:.3f} rad/s (limit: body follows the wave)"))
        out.append(InvariantResult(
            "long_wave_pitch_rao_to_unity", p, 1.0, dev(p, 1.0), dev(p, 1.0) < 0.05,
            f"kL={kl_lo:.4f}; rad/(k a) normalisation makes the limit exactly 1"))

    # ---- short-wave decay -------------------------------------------------
    w_hi = float(transfer.omega[-1])
    kl_hi = w_hi * w_hi / G * L
    if kl_hi >= SHORT_WAVE_KL:
        h_hi = float(transfer.heave_rao[-1])
        out.append(InvariantResult(
            "short_wave_heave_rao_decay", h_hi, 0.0, h_hi, h_hi < 0.05,
            f"kL={kl_hi:.2f} at omega={w_hi:.3f} rad/s (hull cannot follow)"))

    # ---- head-sea symmetry ------------------------------------------------
    if abs(abs(transfer.heading_deg) - 180.0) < 1e-9:
        roll_max = float(np.max(transfer.roll_rao))
        out.append(InvariantResult(
            "head_sea_roll_symmetry", roll_max, 0.0, roll_max, roll_max < 1e-6,
            "roll excitation vanishes at beta = 180 deg by symmetry"))

    # ---- encounter frequency (forward speed) -----------------------------
    if reference and "speed_case" in reference:
        case = reference["speed_case"]
        speed_ms = float(case["speed_kn"]) * 1852.0 / 3600.0
        heading = float(case.get("heading_deg", 180.0))
        worst = 0.0
        for omega, we_sender in zip(case["omega"], case["omega_encounter"]):
            we = encounter_frequency_model(float(omega), speed_ms, heading)
            worst = max(worst, abs(we - float(we_sender)) / max(abs(float(we_sender)), 1e-12))
        out.append(InvariantResult(
            "encounter_frequency_convention", worst, 0.0, worst, worst < 1e-9,
            f"omega_e = omega - k U cos(beta) at U={case['speed_kn']} kn, "
            f"beta={heading} deg; rel. dev over {len(case['omega'])} frequencies"))

    return out

# --------------------------------------------------------------------------- #
# sampling adequacy: pointwise vs spectral-response fidelity
# --------------------------------------------------------------------------- #

def jonswap_omega(omega, hs: float = 2.0, tp: float = 8.0, gamma: float = 3.3,
                  g: float = G):
    """One-sided JONSWAP wave spectrum ``S_zeta(omega)`` [m^2 s]."""
    omega = np.asarray(omega, dtype=float)
    wp = 2.0 * np.pi / float(tp)
    sigma = np.where(omega <= wp, 0.07, 0.09)
    r = np.exp(-((omega - wp) ** 2) / (2.0 * sigma ** 2 * wp ** 2))
    alpha = 0.076 * (g / (wp * np.sqrt(hs))) ** 0.4
    return alpha * g ** 2 / omega ** 5 * np.exp(-1.25 * (wp / omega) ** 4) * gamma ** r


def table_fidelity(transfer: RAOTransfer, reference: dict,
                   hs: float = 2.0, tp: float = 8.0) -> dict:
    """How much information is lost by *tabulating* the RAO?

    Two complementary numbers, because a single one is misleading:

    ``pointwise_max_rel_dev``
        Worst relative error of the PCHIP interpolant away from nodes.  It is
        dominated by the antiresonance (where the RAO crosses a zero, so a
        relative error blows up without any physical consequence) and is
        therefore reported but **not** used as a criterion.
    ``spectral_sigma_rel_dev``
        Relative error of the heave significant amplitude
        ``sigma = sqrt(int |RAO|^2 S_zeta domega)`` in a JONSWAP(Hs, Tp) sea,
        which is what a fatigue or operability calculation actually consumes.
        This is the quantity the interface adequacy criterion is written on.
    """
    dense = reference["reference_dense"]
    w_dense = np.asarray(dense["omega"], dtype=float)
    ref_h = np.asarray(dense["heave_rao"], dtype=float)
    ref_p = np.asarray(dense["pitch_rao"], dtype=float)
    band = (w_dense >= 0.15) & (w_dense <= 2.5)
    s_wave = jonswap_omega(w_dense[band], hs=hs, tp=tp)

    w_band = w_dense[band]
    s_band = s_wave / 2.0 ** 0  # one-sided PSD, m^2 s

    def sigma(rao_band: np.ndarray) -> float:
        return float(np.sqrt(np.trapezoid(rao_band ** 2 * s_band, w_band)))

    tab_h = pchip_interp(transfer.omega, transfer.heave_rao, w_band)
    tab_p = pchip_interp(transfer.omega, transfer.pitch_rao, w_band)
    sig_ref_h, sig_tab_h = sigma(ref_h[band]), sigma(tab_h)
    sig_ref_p, sig_tab_p = sigma(ref_p[band]), sigma(tab_p)

    # pointwise, restricted to where the RAO is not near a node
    ref_h_band = ref_h[band]
    mask = np.abs(ref_h_band) > 1e-2 * np.max(np.abs(ref_h_band))
    point = (float(np.max(np.abs(tab_h[mask] - ref_h_band[mask])
                          / np.abs(ref_h_band[mask]))) if mask.any() else 0.0)

    return {
        "sea_state": {"Hs_m": hs, "Tp_s": tp, "gamma": 3.3},
        "heave_sigma_m": {"reference": sig_ref_h, "table": sig_tab_h,
                          "rel_dev": abs(sig_tab_h - sig_ref_h) / sig_ref_h},
        "pitch_sigma_rad": {"reference": sig_ref_p, "table": sig_tab_p,
                            "rel_dev": abs(sig_tab_p - sig_ref_p) / sig_ref_p},
        "pointwise_max_rel_dev": point,
        "n_table_points": int(transfer.omega.size),
        "reference_points": int(w_dense.size),
    }
