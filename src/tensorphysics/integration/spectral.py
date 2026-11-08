"""``stress-psd-v1`` receiver: spectral fatigue recomputed independently.

The sender (ship platform, ``backend.modules.spectral_fatigue``) publishes a
one-sided hot-spot stress PSD and the S-N law it was assessed with.  The
receiver (this module) rebuilds the whole chain with TensorPhysics' own
implementation (:mod:`tensorphysics.fatigue.spectral`):

    S(f) -> m0..m4 -> alpha1, alpha2, nu0, nu_p -> range PDF -> D

so the comparison tests the *contract* (are the moments, the cycle rates and
therefore the Miner damage reproduced?) rather than a shared code path.  The
two implementations integrate the same PSD with the trapezoidal rule on the
same grid, so the remaining difference is the range-distribution closure and
the moment bookkeeping - which is exactly what a fatigue interface must agree
on.

Damage is reported both as a rate (per second) and as the accumulated Miner
damage over the contract's ``duration_s``; the sender reports the accumulated
value.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from ..fatigue.spectral import (
    SpectralMoments,
    benasciutti_tovo_damage_rate,
    dirlik_damage_rate,
    dirlik_parameters,
    moments_from_dict,
    rayleigh_damage_rate,
    spectral_moments,
)
from .interchange import StressSpectrumTransfer

__all__ = [
    "DamageComparison",
    "compare_damage",
    "damage_by_method",
    "moments_of",
]


def moments_of(transfer: StressSpectrumTransfer) -> SpectralMoments:
    """Spectral moments ``m0..m4`` (Hz convention) of the transmitted PSD."""
    table = spectral_moments(transfer.freq_hz, transfer.psd)
    return moments_from_dict(table, f_lo=float(transfer.freq_hz[0]),
                             f_hi=float(transfer.freq_hz[-1]))


def damage_by_method(transfer: StressSpectrumTransfer) -> dict:
    """Damage rates (1/s) for the three closures plus the accumulated damage."""
    mom = moments_of(transfer)
    dirl = dirlik_parameters(mom)
    rates = {
        "dirlik": dirlik_damage_rate(mom, transfer.sn_C, transfer.sn_m),
        "rayleigh": rayleigh_damage_rate(mom, transfer.sn_C, transfer.sn_m),
        "tovo_benasciutti": benasciutti_tovo_damage_rate(mom, transfer.sn_C,
                                                        transfer.sn_m),
    }
    return {
        "moments": {"m0": mom.m0, "m1": mom.m1, "m2": mom.m2,
                    "m3": mom.m3, "m4": mom.m4},
        "descriptors": {
            "rms": mom.sigma, "nu0_hz": mom.nu0, "nu_p_hz": mom.nup,
            "alpha1": mom.alpha1, "alpha2": mom.alpha2,
            "bandwidth_eps": mom.bandwidth_eps,
        },
        "dirlik_coefficients": {"D1": dirl.D1, "D2": dirl.D2, "D3": dirl.D3,
                                "Q": dirl.Q, "R": dirl.R},
        "damage_rate_per_s": rates,
        "damage_accumulated": {k: v * transfer.duration_s for k, v in rates.items()},
        "duration_s": transfer.duration_s,
    }


@dataclass(frozen=True)
class DamageComparison:
    """Sender vs receiver Miner damage on the same transmitted spectrum."""

    method: str
    damage_sender: float
    damage_receiver: float
    rel_dev: float

    def as_dict(self) -> dict:
        return {
            "method": self.method,
            "damage_sender": self.damage_sender,
            "damage_receiver": self.damage_receiver,
            "rel_dev": self.rel_dev,
        }


def compare_damage(transfer: StressSpectrumTransfer,
                   reference: Mapping[str, float]) -> dict:
    """Compare against the sender's ``dirlik_damage`` / ``narrowband_damage``.

    ``reference`` is the sender's response block; the Dirlik value is the one
    the C3 criterion is evaluated on (it is the closure both sides implement
    in closed form), the narrow-band value is reported as a second witness.
    """
    mine = damage_by_method(transfer)
    out = {}
    pairs = (("dirlik", "dirlik_damage", "dirlik"),
             ("rayleigh", "narrowband_damage", "narrowband"))
    for key, ref_key, label in pairs:
        if ref_key not in reference:
            continue
        sender = float(reference[ref_key])
        receiver = mine["damage_accumulated"][key]
        out[label] = DamageComparison(
            method=label, damage_sender=sender, damage_receiver=receiver,
            rel_dev=abs(receiver - sender) / max(abs(sender), 1e-300),
        ).as_dict()
    out["receiver_moments"] = mine["moments"]
    out["receiver_descriptors"] = mine["descriptors"]
    out["receiver_dirlik_coefficients"] = mine["dirlik_coefficients"]
    out["duration_s"] = mine["duration_s"]
    out["max_rel_dev"] = max((v["rel_dev"] for v in out.values()
                              if isinstance(v, dict) and "rel_dev" in v), default=0.0)
    return out