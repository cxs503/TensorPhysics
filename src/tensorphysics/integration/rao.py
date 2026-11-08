"""``rao-v1`` receiver: response reconstruction and physics gates.

Three things are checked here, and they answer different questions.

1. **Transport + convention fidelity (CONTRACT).**  Given only the payload's
   declared register (``m/m`` for heave, ``rad/(k a)`` for pitch/roll,
   ``rad/s`` frequencies) the receiver must rebuild the motion amplitudes the
   sender reported.  Any drift in normalisation, degrees-vs-radians or
   ``omega`` vs ``omega_e`` shows up as an O(1) error.  This is the machine
   criterion of the P27W3-C2 acceptance test.

2. **Model-independent invariants (REAL physics).**  In the long-wave limit
   ``kL -> 0`` a freely floating body follows the wave surface, so
   ``RAO_heave -> 1`` (Newman 1977, *Marine Hydrodynamics* §6.10).  The
   invariant does not depend on which strip theory the sender used, so it is a
   genuine acceptance gate on the transmitted curve rather than a self-check.

3. **Sectional added-mass cross-check (REAL, model form).**  TensorPhysics owns
   a 2-D boundary-element solver for the infinite-frequency sectional added
   mass (``phi = 0`` on the free surface, exact log-kernel panels, verified
   against the closed forms ``rho pi a^2`` for a submerged circle and a flat
   plate).  Comparing it with the sender's sectional ``A33`` isolates the
   *model-form* error of the sender's empirical frequency factor ``f_a(nu)``
   from any geometry error.  It is reported as a diagnostic, **not** as the C2
   criterion: two different hydrodynamic closures cannot be expected to agree
   to 1 % in general.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from ..hull_hydroelastic.sections import Section2D, heave_added_mass, superellipse_section
from .interchange import ContractViolation, RAOTransfer

__all__ = [
    "ResponseComparison",
    "SectionalComparison",
    "compare_responses",
    "long_wave_limit",
    "pchip_interp",
    "reconstruct_response",
    "sectional_added_mass_crosscheck",
]

G = 9.80665


def pchip_interp(x: Sequence[float], y: Sequence[float],
                 xq: Sequence[float]) -> np.ndarray:
    """Monotone cubic (Fritsch-Carlson) interpolation, no SciPy dependency.

    RAO curves are smooth but not monotone, and linear interpolation on a
    40-point log grid biases the resonance peak by a few percent.  The
    Fritsch-Carlson limiter keeps the interpolant monotone on monotone
    segments, which is exactly the property a *transfer function* table should
    have.
    """
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    if xa.size != ya.size or xa.size < 2:
        raise ValueError("pchip needs matching arrays of length >= 2")
    if np.any(np.diff(xa) <= 0.0):
        raise ValueError("x must be strictly increasing")

    h = np.diff(xa)
    delta = np.diff(ya) / h
    d = np.zeros_like(ya)
    # interior slopes: harmonic mean of neighbouring secants where they agree in sign
    for k in range(1, xa.size - 1):
        if delta[k - 1] * delta[k] > 0.0:
            w1 = 2.0 * h[k] + h[k - 1]
            w2 = h[k] + 2.0 * h[k - 1]
            d[k] = (w1 + w2) / (w1 / delta[k - 1] + w2 / delta[k])
    # endpoint slopes (one-sided three-point, limited)
    d[0] = _endpoint_slope(h[0], h[1], delta[0], delta[1]) if xa.size > 2 else delta[0]
    d[-1] = _endpoint_slope(h[-1], h[-2], delta[-1], delta[-2]) if xa.size > 2 else delta[-1]

    xq_arr = np.atleast_1d(np.asarray(xq, dtype=float))
    out = np.empty_like(xq_arr)
    for i, xv in enumerate(xq_arr):
        if xv <= xa[0]:
            out[i] = ya[0] + d[0] * (xv - xa[0])
        elif xv >= xa[-1]:
            out[i] = ya[-1] + d[-1] * (xv - xa[-1])
        else:
            k = int(np.searchsorted(xa, xv) - 1)
            t = (xv - xa[k]) / h[k]
            h00 = (1 + 2 * t) * (1 - t) ** 2
            h10 = t * (1 - t) ** 2
            h01 = t * t * (3 - 2 * t)
            h11 = t * t * (t - 1)
            out[i] = (h00 * ya[k] + h10 * h[k] * d[k]
                      + h01 * ya[k + 1] + h11 * h[k] * d[k + 1])
    return out


def _endpoint_slope(h1: float, h2: float, d1: float, d2: float) -> float:
    slope = ((2.0 * h1 + h2) * d1 - h1 * d2) / (h1 + h2)
    if slope * d1 <= 0.0:
        return 0.0
    if d1 * d2 <= 0.0 and abs(slope) > abs(3.0 * d1):
        return 3.0 * d1
    return slope


@dataclass(frozen=True)
class ResponseComparison:
    """Per-frequency reconstruction check of one contract frequency point."""

    omega: float
    heave_m_receiver: float
    pitch_deg_receiver: float
    roll_deg_receiver: float
    heave_m_sender: float
    pitch_deg_sender: float
    roll_deg_sender: float
    rel_dev_heave: float
    rel_dev_pitch: float
    rel_dev_roll: float

    @property
    def max_rel_dev(self) -> float:
        return max(self.rel_dev_heave, self.rel_dev_pitch, self.rel_dev_roll)

    def as_dict(self) -> dict:
        return {
            "omega_rad_s": self.omega,
            "heave_m": {"receiver": self.heave_m_receiver,
                        "sender": self.heave_m_sender,
                        "rel_dev": self.rel_dev_heave},
            "pitch_deg": {"receiver": self.pitch_deg_receiver,
                          "sender": self.pitch_deg_sender,
                          "rel_dev": self.rel_dev_pitch},
            "roll_deg": {"receiver": self.roll_deg_receiver,
                         "sender": self.roll_deg_sender,
                         "rel_dev": self.rel_dev_roll},
            "max_rel_dev": self.max_rel_dev,
        }


def reconstruct_response(transfer: RAOTransfer, omega: float,
                         wave_amplitude_m: float = 1.0) -> dict:
    """Rebuild the physical response amplitudes from the RAO table alone.

    ``heave = RAO_heave * a``; ``pitch = RAO_pitch * k * a`` with ``k = omega^2/g``;
    ``roll`` uses the same slope normalisation as pitch (both are angles per unit
    wave slope).  Everything derives from the payload's declared register, so a
    convention change on either side is detectable.
    """
    if wave_amplitude_m <= 0.0:
        raise ValueError("wave amplitude must be positive")
    k = transfer.wave_number(omega, G)
    heave_rao = float(pchip_interp(transfer.omega, transfer.heave_rao, [omega])[0])
    pitch_rao = float(pchip_interp(transfer.omega, transfer.pitch_rao, [omega])[0])
    roll_rao = float(pchip_interp(transfer.omega, transfer.roll_rao, [omega])[0])
    slope = k * wave_amplitude_m
    return {
        "omega_rad_s": float(omega),
        "wavenumber_1_per_m": k,
        "heave_m": heave_rao * wave_amplitude_m,
        "pitch_deg": float(np.degrees(pitch_rao * slope)),
        "roll_deg": float(np.degrees(roll_rao * slope)),
        "heave_rao": heave_rao,
        "pitch_rao": pitch_rao,
        "roll_rao": roll_rao,
        "pitch_rad": pitch_rao * slope,
        "roll_rad": roll_rao * slope,
    }


def compare_responses(transfer: RAOTransfer,
                      reference: Sequence[Mapping[str, float]],
                      wave_amplitude_m: float = 1.0) -> list[ResponseComparison]:
    """Compare the receiver's reconstruction with the sender's reported motions."""
    out: list[ResponseComparison] = []
    for row in reference:
        omega = float(row["omega"])
        rec = reconstruct_response(transfer, omega, wave_amplitude_m)
        out.append(ResponseComparison(
            omega=omega,
            heave_m_receiver=rec["heave_m"],
            pitch_deg_receiver=rec["pitch_deg"],
            roll_deg_receiver=rec["roll_deg"],
            heave_m_sender=float(row["heave_m"]),
            pitch_deg_sender=float(row["pitch_deg"]),
            roll_deg_sender=float(row["roll_deg"]),
            rel_dev_heave=_rel(rec["heave_m"], float(row["heave_m"])),
            rel_dev_pitch=_rel(rec["pitch_deg"], float(row["pitch_deg"])),
            rel_dev_roll=_rel(rec["roll_deg"], float(row["roll_deg"])),
        ))
    return out


def _rel(value: float, reference: float) -> float:
    scale = max(abs(reference), 1e-30)
    return float(abs(value - reference) / scale)


def long_wave_limit(transfer: RAOTransfer) -> dict:
    """``RAO_heave -> 1`` as ``kL -> 0`` (Newman 1977); returns the probe point."""
    L = float(transfer.hull_params.get("L", float("nan")))
    k = transfer.wave_number(float(transfer.omega[0]))
    rao0 = float(transfer.heave_rao[0])
    return {
        "omega_rad_s": float(transfer.omega[0]),
        "kL": float(k * L),
        "heave_rao": rao0,
        "abs_dev_from_unity": abs(rao0 - 1.0),
        "invariant": "heave RAO -> 1 for kL -> 0",
    }


@dataclass(frozen=True)
class SectionalComparison:
    """Infinite-frequency sectional heave added mass: sender vs TensorPhysics BEM."""

    station_x_m: float
    half_beam_m: float
    draught_m: float
    sender_a33_kg_per_m: float
    bem_a33_kg_per_m: float
    sender_f_a: float
    rel_dev: float
    n_panels: int

    def as_dict(self) -> dict:
        return {
            "station_x_m": self.station_x_m,
            "half_beam_m": self.half_beam_m,
            "draught_m": self.draught_m,
            "sender_a33_kg_per_m": self.sender_a33_kg_per_m,
            "bem_a33_inf_kg_per_m": self.bem_a33_kg_per_m,
            "sender_empirical_f_a": self.sender_f_a,
            "rel_dev": self.rel_dev,
            "bem_panels": self.n_panels,
        }


def sectional_added_mass_crosscheck(sections: Mapping[str, Sequence[float]],
                                    rho: float = 1025.0,
                                    exponent: float = 40.0,
                                    n_panels: int | None = 400) -> dict:
    """Compare the sender's sectional ``A33`` with the TensorPhysics 2-D BEM.

    The sender's section is wall-sided (``area = 2 b T``), so the BEM section is
    modelled as a superellipse of exponent 40 (rectangle to 0.1 %), floating
    (``phi = 0`` on the waterline).  The comparison is made at the maximum
    ``rho A f_a`` station, i.e. the section the sender weights most.
    """
    x = np.asarray(sections["x_m"], dtype=float)
    b = np.asarray(sections["half_beam_m"], dtype=float)
    area = np.asarray(sections["area_m2"], dtype=float)
    a33 = np.asarray(sections["a33_kg_per_m"], dtype=float)
    fa = np.asarray(sections.get("f_a", a33 / (rho * area)), dtype=float)
    T = float(sections.get("draught_m", sections.get("T", 0.0)) or 0.0)
    if T <= 0.0:
        # reconstruct the draught from the wall-sided section area
        T = float(np.median(area / (2.0 * b)))

    mid = int(np.argmax(b))
    section: Section2D = superellipse_section(float(b[mid]), T, exponent=exponent)
    bem = heave_added_mass(section, rho=rho, n_panels=n_panels, free_surface=True)
    comp = SectionalComparison(
        station_x_m=float(x[mid]),
        half_beam_m=float(b[mid]),
        draught_m=T,
        sender_a33_kg_per_m=float(a33[mid]),
        bem_a33_kg_per_m=float(bem),
        sender_f_a=float(fa[mid]),
        rel_dev=_rel(float(bem), float(a33[mid])),
        n_panels=int(n_panels or 0),
    )
    geometry = {
        "max_half_beam_m": float(np.max(b)),
        "beam_attained_m": float(2.0 * np.max(b)),
        "declared_beam_m": None,
        "note": "declared B is carried in hull_params; see geometry_audit()",
    }
    return {"comparison": comp.as_dict(), "geometry": geometry}


def geometry_audit(transfer: RAOTransfer,
                   sections: Mapping[str, Sequence[float]]) -> dict:
    """Does the sender's station geometry actually attain the declared ``B``?

    A hull-identity contract that only carries ``(L, B, T, Cb, Cwp)`` is
    under-determined: the sender may rebuild the waterline with an internal
    shape function whose maximum breadth is *not* ``B``, while still matching
    ``Aw = Cwp L B`` and ``nabla = Cb L B T``.  This audit makes that visible.
    """
    B = float(transfer.hull_params["B"])
    T = float(transfer.hull_params["T"])
    Cb = float(transfer.hull_params["Cb"])
    b = np.asarray(sections["half_beam_m"], dtype=float)
    area = np.asarray(sections["area_m2"], dtype=float)
    dx = float(sections["dx_m"])
    true_mid_area = B * T * (Cb / float(transfer.hull_params.get("Cwp", Cb)))
    return {
        "declared_B_m": B,
        "max_half_beam_m": float(np.max(b)),
        "modelled_beam_m": float(2.0 * np.max(b)),
        "beam_rel_dev": float(abs(2.0 * np.max(b) - B) / B),
        "midstation_area_m2": float(area[int(np.argmax(b))]),
        "wall_sided_midstation_area_m2": float(true_mid_area),
        "midstation_area_rel_dev": float(abs(area[int(np.argmax(b))] - true_mid_area) / true_mid_area),
        "integrated_nabla_m3": float(np.sum(area) * dx),
        "declared_nabla_m3": float(transfer.hull_params.get("nabla_m3", float("nan"))),
    }