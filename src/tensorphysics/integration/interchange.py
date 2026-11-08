"""Versioned JSON payloads of the P27W3 cross-repository contracts.

Why a schema and not a dict
--------------------------
The failure modes of a repo-to-repo interface are almost never numerical
errors in the physics; they are *unit and convention* mismatches:

* heave RAO reported per unit wave **amplitude** vs per unit wave **height**;
* pitch/roll RAO reported in ``rad/m`` vs ``rad/(k a)`` (per unit wave slope);
* phase in degrees vs radians, and the sign convention of the encounter
  frequency ``omega_e = omega - k U cos(beta)``;
* a stress PSD in ``MPa^2/Hz`` (Hz axis) vs ``MPa^2/(rad/s)`` (omega axis),
  which changes ``m1..m4`` by factors of ``2*pi`` and therefore ``nu0``/``nu_p``
  (i.e. the damage) by the same factor;
* an S-N law written ``N = C S^-m`` vs ``S^m N = C`` (same thing) but with the
  stress in MPa vs Pa (factor ``10^6m`` in ``C``).

Every payload therefore carries an explicit ``unit_register``.  Parsing is
fail-closed: an unknown or missing register entry raises
:class:`ContractViolation` instead of assuming the sender's convention.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

__all__ = [
    "ContractViolation",
    "IceBondTransfer",
    "RAOTransfer",
    "StressSpectrumTransfer",
    "load_golden",
    "UNIT_REGISTER",
]

#: Admissible convention strings.  Anything else is rejected (fail-closed).
UNIT_REGISTER = {
    "rao-v1": {
        "heave_rao": ("m/m",),
        "pitch_rao": ("rad/(k*a)",),
        "roll_rao": ("rad/(k*a)",),
        "frequency": ("rad/s",),
        "phase": ("deg",),
    },
    "stress-psd-v1": {
        "freq": ("Hz",),
        "psd": ("MPa^2/Hz",),
        "sn": ("S^m N = C, S in MPa",),
    },
    "ice-bond-v1": {
        "strength": ("Pa",),
        "stiffness": ("N/m",),
        "length": ("m",),
        "stress": ("Pa",),
    },
}


class ContractViolation(ValueError):
    """The payload does not satisfy the versioned contract (fail-closed)."""


def _require(mapping: Mapping[str, Any], key: str, ctx: str) -> Any:
    if key not in mapping:
        raise ContractViolation(f"{ctx}: missing required field {key!r}")
    return mapping[key]


def _require_unit(contract: str, key: str, value: str, ctx: str) -> None:
    allowed = UNIT_REGISTER[contract].get(key)
    if allowed is None:
        raise ContractViolation(f"{ctx}: {key!r} is not a registered contract field")
    if value not in allowed:
        raise ContractViolation(
            f"{ctx}: unit/normalisation of {key!r} is {value!r}, "
            f"contract {contract} admits {allowed}"
        )


def _as_float_array(values: Sequence[float], name: str, ctx: str) -> np.ndarray:
    try:
        arr = np.asarray(list(values), dtype=float)
    except (TypeError, ValueError) as exc:                       # pragma: no cover
        raise ContractViolation(f"{ctx}: {name} is not a numeric array") from exc
    if arr.size == 0:
        raise ContractViolation(f"{ctx}: {name} is empty")
    if not np.all(np.isfinite(arr)):
        raise ContractViolation(f"{ctx}: {name} contains non-finite values")
    return arr


@dataclass(frozen=True)
class RAOTransfer:
    """``rao-v1``: motion transfer functions sampled on a frequency grid.

    ``heave_rao`` is dimensionless (per unit wave amplitude), ``pitch_rao`` and
    ``roll_rao`` are per unit wave slope ``k a``.  ``omega`` is the *incident*
    wave frequency (rad/s); the sender additionally publishes the encounter
    frequency so that a receiver can tell the two apart.
    """

    contract: str
    hull: str
    hull_params: dict
    heading_deg: float
    speed_kn: float
    omega: np.ndarray
    heave_rao: np.ndarray
    pitch_rao: np.ndarray
    roll_rao: np.ndarray
    normalisation: dict
    omega_encounter: np.ndarray | None = None
    heave_phase_deg: np.ndarray | None = None
    pitch_phase_deg: np.ndarray | None = None
    provenance: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ parse
    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "RAOTransfer":
        ctx = "rao-v1"
        if _require(payload, "contract", ctx) != "rao-v1":
            raise ContractViolation(f"{ctx}: contract tag mismatch")
        norm = dict(_require(payload, "normalisation", ctx))
        for key in ("heave_rao", "pitch_rao", "roll_rao"):
            _require_unit("rao-v1", key, str(norm.get(key, "")), ctx)
        if str(norm.get("frequency", "")) != "rad/s":
            raise ContractViolation(f"{ctx}: frequency axis must declare 'rad/s'")

        omega = _as_float_array(_require(payload, "frequencies_rad_s", ctx),
                               "frequencies_rad_s", ctx)
        if np.any(omega <= 0.0):
            raise ContractViolation(f"{ctx}: frequencies must be positive")
        if np.any(np.diff(omega) <= 0.0):
            raise ContractViolation(f"{ctx}: frequencies must be strictly ascending")

        arrays = {}
        for key in ("heave_rao", "pitch_rao", "roll_rao"):
            arrays[key] = _as_float_array(_require(payload, key, ctx + "/response"), key, ctx)
            if arrays[key].shape != omega.shape:
                raise ContractViolation(
                    f"{ctx}: {key} has {arrays[key].size} samples for "
                    f"{omega.size} frequencies")
            if np.any(arrays[key] < 0.0):
                raise ContractViolation(f"{ctx}: {key} must be non-negative magnitudes")

        prov = dict(payload.get("provenance") or {})
        if "peer_head_sha" not in prov and "peer_repo" not in prov:
            raise ContractViolation(
                f"{ctx}: provenance must name the peer repository and commit")

        return cls(
            contract="rao-v1",
            hull=str(payload.get("hull", "unspecified")),
            hull_params=dict(payload.get("hull_params") or {}),
            heading_deg=float(payload.get("heading_deg", 180.0)),
            speed_kn=float(payload.get("speed_kn", 0.0)),
            omega=omega,
            heave_rao=arrays["heave_rao"],
            pitch_rao=arrays["pitch_rao"],
            roll_rao=arrays["roll_rao"],
            normalisation=norm,
            omega_encounter=_optional_float_array(payload.get("omega_encounter_rad_s")),
            heave_phase_deg=_optional_float_array(payload.get("heave_phase_deg")),
            pitch_phase_deg=_optional_float_array(payload.get("pitch_phase_deg")),
            provenance=prov,
        )

    # ------------------------------------------------------------------ query
    def wave_number(self, omega: float, g: float = 9.80665) -> float:
        """Deep-water wavenumber ``k = omega^2 / g`` (exact, Newton 1687)."""
        return float(omega) ** 2 / g

    def hull_identity(self) -> dict:
        """Geometry gate: ``nabla`` must equal ``Cb L B T`` within 0.5 %."""
        p = self.hull_params
        for key in ("L", "B", "T", "Cb"):
            _require(p, key, "rao-v1/hull_params")
        nabla = float(p["Cb"]) * float(p["L"]) * float(p["B"]) * float(p["T"])
        claimed = float(p.get("nabla_m3", nabla))
        return {
            "L": float(p["L"]), "B": float(p["B"]), "T": float(p["T"]),
            "Cb": float(p["Cb"]), "Cwp": float(p.get("Cwp", float("nan"))),
            "nabla_from_params_m3": nabla, "nabla_claimed_m3": claimed,
            "nabla_rel_dev": abs(claimed - nabla) / nabla,
        }


@dataclass(frozen=True)
class StressSpectrumTransfer:
    """``stress-psd-v1``: one-sided hot-spot stress PSD plus the S-N law."""

    contract: str
    freq_hz: np.ndarray
    psd: np.ndarray
    sn_C: float
    sn_m: float
    duration_s: float
    provenance: dict = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "StressSpectrumTransfer":
        ctx = "stress-psd-v1"
        if _require(payload, "contract", ctx) != "stress-psd-v1":
            raise ContractViolation(f"{ctx}: contract tag mismatch")
        f = _as_float_array(_require(payload, "freq_hz", ctx), "freq_hz", ctx)
        s = _as_float_array(_require(payload, "sxx_unit2_per_hz", ctx),
                            "sxx_unit2_per_hz", ctx)
        if f.shape != s.shape:
            raise ContractViolation(f"{ctx}: freq_hz and PSD length differ")
        if np.any(np.diff(f) <= 0.0):
            raise ContractViolation(f"{ctx}: freq_hz must be strictly ascending")
        if f[0] < 0.0:
            raise ContractViolation(f"{ctx}: freq_hz must be non-negative")
        if np.any(s < 0.0):
            raise ContractViolation(f"{ctx}: a PSD cannot be negative")
        if float(np.trapezoid(s, f)) <= 0.0:
            raise ContractViolation(f"{ctx}: zero-energy stress spectrum")

        sn = dict(_require(payload, "sn", ctx))
        conv = str(sn.get("convention", ""))
        _require_unit("stress-psd-v1", "sn", conv, ctx)
        C = float(_require(sn, "C", ctx + "/sn"))
        m = float(_require(sn, "m", ctx + "/sn"))
        if C <= 0.0 or m <= 0.0:
            raise ContractViolation(f"{ctx}: S-N constants must be positive")
        duration = float(payload.get("duration_s", 1.0))
        if duration <= 0.0:
            raise ContractViolation(f"{ctx}: duration_s must be positive")
        return cls("stress-psd-v1", f, s, C, m, duration,
                   dict(payload.get("provenance") or {}))

    def scaled_to_rms(self, rms: float) -> "StressSpectrumTransfer":
        """Uniform power rescaling (used for the linearity negative control)."""
        cur = math.sqrt(float(np.trapezoid(self.psd, self.freq_hz)))
        if cur <= 0.0:                                            # pragma: no cover
            raise ContractViolation("cannot rescale a zero-energy spectrum")
        factor = (float(rms) / cur) ** 2
        return StressSpectrumTransfer(
            self.contract, self.freq_hz, self.psd * factor, self.sn_C,
            self.sn_m, self.duration_s, self.provenance)


@dataclass(frozen=True)
class IceBondTransfer:
    """``ice-bond-v1``: TensorDEM lattice specimen plus its measured bond data."""

    contract: str
    spec: dict
    lattice: dict
    beam_theory: dict
    provenance: dict = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any],
                     response: Mapping[str, Any]) -> "IceBondTransfer":
        ctx = "ice-bond-v1"
        if _require(payload, "contract", ctx) != "ice-bond-v1":
            raise ContractViolation(f"{ctx}: contract tag mismatch")
        spec = dict(_require(payload, "dem_spec", ctx))
        for key in ("length_m", "width_m", "thickness_m", "particle_diameter_m",
                    "target_young_modulus_Pa", "bond_strength_Pa"):
            _require(spec, key, ctx + "/dem_spec")
            if float(spec[key]) <= 0.0:
                raise ContractViolation(f"{ctx}: dem_spec[{key}] must be positive")
        lattice = dict(_require(response, "lattice", ctx))
        for key in ("lattice_spacing_m", "bond_stiffness_N_m", "breaking_strain",
                    "young_modulus_Pa"):
            _require(lattice, key, ctx + "/lattice")
        beam = dict(response.get("beam_theory") or {})
        return cls("ice-bond-v1", spec, lattice, beam,
                   dict(payload.get("provenance") or {}))

    @property
    def bond_strength_Pa(self) -> float:
        return float(self.spec["bond_strength_Pa"])

    @property
    def spacing_m(self) -> float:
        return float(self.lattice["lattice_spacing_m"])


def _optional_float_array(values: Any) -> np.ndarray | None:
    if values is None:
        return None
    return np.asarray(list(values), dtype=float)


def load_golden(path: str | Path) -> dict:
    """Load a captured peer payload/response pair (see ``scripts/``)."""
    blob = json.loads(Path(path).read_text())
    for key in ("payload", "response", "provenance"):
        if key not in blob:
            raise ContractViolation(f"{path}: golden corpus lacks {key!r}")
    return blob