"""Injected configuration for the ice-encounter statistics study (P233W2-N05).

Everything domain-specific -- ship beam, speed, floe number density, floe-size
spectrum, thickness, crushing-force coefficients, contact-duration model, and
the acceptance thresholds themselves -- is loaded from JSON.  This module keeps
the physics *form* (which quantities exist and how they combine) and rejects
configurations that are missing keys, so a run can never silently fall back to
a hidden constant.
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

import numpy as np

CONFIG_DIR = pathlib.Path(__file__).resolve().parent / "configs"
DEFAULT_PREREG = CONFIG_DIR / "prereg_p233w2_n05.json"


class ConfigError(ValueError):
    """Raised when an injected configuration is missing or inconsistent."""


def _require(mapping: Mapping[str, Any], key: str, ctx: str) -> Any:
    if key not in mapping:
        raise ConfigError(f"missing required key {key!r} in {ctx}")
    return mapping[key]


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict:
    """Recursively merge ``override`` onto ``base`` (override wins)."""
    out: dict[str, Any] = dict(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], Mapping) and isinstance(value, Mapping):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


# --------------------------------------------------------------------------- #
# distributions
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ScalarSpectrum:
    """Positive scalar distribution used for floe size / ice thickness."""

    distribution: str
    params: Mapping[str, float] = field(default_factory=dict)

    def sample(self, rng: np.random.Generator, size) -> np.ndarray:
        dist, p = self.distribution, self.params
        if dist == "constant":
            return np.full(int(size), float(_require(p, "value", "constant spectrum")))
        if dist == "loguniform":
            lo = float(_require(p, "min", "loguniform spectrum"))
            hi = float(_require(p, "max", "loguniform spectrum"))
            if not (0.0 < lo < hi):
                raise ConfigError(f"loguniform requires 0 < min < max (got {lo}, {hi})")
            return np.exp(rng.uniform(np.log(lo), np.log(hi), int(size)))
        if dist == "lognormal":
            med = float(_require(p, "median", "lognormal spectrum"))
            sig = float(_require(p, "sigma_log", "lognormal spectrum"))
            if med <= 0.0 or sig < 0.0:
                raise ConfigError("lognormal requires median > 0 and sigma_log >= 0")
            return med * np.exp(sig * rng.standard_normal(int(size)))
        raise ConfigError(f"unknown distribution {dist!r}")

    def scale(self) -> float:
        """A representative magnitude (the median) used only for time-step sizing."""
        dist, p = self.distribution, self.params
        if dist == "constant":
            return float(p["value"])
        if dist == "loguniform":
            return float(np.sqrt(float(p["min"]) * float(p["max"])))
        if dist == "lognormal":
            return float(p["median"])
        raise ConfigError(f"unknown distribution {dist!r}")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], ctx: str) -> "ScalarSpectrum":
        return cls(distribution=str(_require(data, "distribution", ctx)),
                   params=dict(data.get("params", {})))


# --------------------------------------------------------------------------- #
# case specification
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ClusterSpec:
    """Neyman--Scott cluster driver (negative control for Poisson encounters)."""

    mean_offspring: float
    sigma_s: float
    rate_per_s: Optional[float] = None  # None -> derived from the Poisson rate

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], ctx: str) -> "ClusterSpec":
        return cls(mean_offspring=float(_require(data, "mean_offspring", ctx)),
                   sigma_s=float(_require(data, "sigma_s", ctx)),
                   rate_per_s=None if data.get("rate_per_s") is None
                   else float(data["rate_per_s"]))


@dataclass(frozen=True)
class ForceModelSpec:
    """Single-floe peak force  F = c * d^a * h^b * exp(sigma * Z)."""

    coefficient: float
    size_exponent: float
    thickness_exponent: float
    log_sigma: float = 0.0

    def amplitude(self, rng: np.random.Generator, floe_m, thickness_m) -> np.ndarray:
        base = (self.coefficient
                * np.power(np.asarray(floe_m, dtype=float), self.size_exponent)
                * np.power(np.asarray(thickness_m, dtype=float), self.thickness_exponent))
        if self.log_sigma <= 0.0:
            return base
        return base * np.exp(self.log_sigma * rng.standard_normal(base.shape))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], ctx: str) -> "ForceModelSpec":
        return cls(
            coefficient=float(_require(data, "coefficient", ctx)),
            size_exponent=float(_require(data, "size_exponent", ctx)),
            thickness_exponent=float(_require(data, "thickness_exponent", ctx)),
            log_sigma=float(data.get("log_sigma", 0.0)),
        )


@dataclass(frozen=True)
class PulseSpec:
    """Normalised pulse shape and the contact-duration model."""

    shape: str
    rise_fraction: float
    duration_model: str
    duration_scale: float = 1.0

    def duration_s(self, floe_m, speed_mps: float) -> np.ndarray:
        if self.duration_model == "size_over_speed":
            return self.duration_scale * np.asarray(floe_m, dtype=float) / float(speed_mps)
        raise ConfigError(f"unknown duration_model {self.duration_model!r}")

    def evaluate(self, tau: np.ndarray) -> np.ndarray:
        """Normalised pulse ``g(tau)`` (unit peak) at normalised times ``tau``."""
        if self.shape != "asymmetric_triangle":
            raise ConfigError(f"unknown pulse shape {self.shape!r}")
        r = float(self.rise_fraction)
        if not (0.0 < r < 1.0):
            raise ConfigError("rise_fraction must lie strictly between 0 and 1")
        tau = np.asarray(tau, dtype=float)
        return np.where(tau < r, tau / r, (1.0 - tau) / (1.0 - r))

    def template(self, n_samples: int) -> np.ndarray:
        """Pulse sampled at midpoints of ``n_samples`` equal cells on [0, 1]."""
        if n_samples < 2:
            raise ConfigError("pulse template needs at least 2 samples")
        return self.evaluate((np.arange(n_samples) + 0.5) / n_samples)

    def integrals(self, n_samples: int = 256) -> tuple:
        """Return (integral g dtau, integral g^2 dtau) on the unit-duration pulse."""
        g = self.template(n_samples)
        return float(np.mean(g)), float(np.mean(g * g))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], ctx: str) -> "PulseSpec":
        return cls(shape=str(_require(data, "shape", ctx)),
                   rise_fraction=float(_require(data, "rise_fraction", ctx)),
                   duration_model=str(_require(data, "duration_model", ctx)),
                   duration_scale=float(data.get("duration_scale", 1.0)))


@dataclass(frozen=True)
class CaseSpec:
    """One physical operating condition of the ice-encounter process."""

    name: str
    beam_m: float
    speed_mps: float
    number_density_per_m2: float
    floe_size: ScalarSpectrum
    thickness: ScalarSpectrum
    force: ForceModelSpec
    pulse: PulseSpec
    noise_std_ratio: float = 0.0
    clustering: Optional[ClusterSpec] = None

    # -- derived ----------------------------------------------------------- #
    @property
    def encounter_rate(self) -> float:
        """Poisson encounter rate lambda = rho_A * B * V  [1/s]."""
        return self.number_density_per_m2 * self.beam_m * self.speed_mps

    @property
    def driver_rate(self) -> float:
        """Underlying cluster (or Poisson) driver rate of the point process."""
        if self.clustering is None:
            return self.encounter_rate
        if self.clustering.rate_per_s is not None:
            return self.clustering.rate_per_s
        return self.encounter_rate / self.clustering.mean_offspring

    def pulse_scale_s(self) -> float:
        """Representative contact duration, used for time-step sizing."""
        return self.pulse.duration_scale * self.floe_size.scale() / self.speed_mps

    def with_overrides(self, overrides: Mapping[str, Any]) -> "CaseSpec":
        base = self.to_dict()
        return CaseSpec.from_dict(deep_merge(base, overrides))

    # -- (de)serialisation -------------------------------------------------- #
    def to_dict(self) -> dict:
        payload = {
            "name": self.name,
            "beam_m": self.beam_m,
            "speed_mps": self.speed_mps,
            "number_density_per_m2": self.number_density_per_m2,
            "floe_size": {"distribution": self.floe_size.distribution,
                          "params": dict(self.floe_size.params)},
            "thickness": {"distribution": self.thickness.distribution,
                          "params": dict(self.thickness.params)},
            "force": {"coefficient": self.force.coefficient,
                      "size_exponent": self.force.size_exponent,
                      "thickness_exponent": self.force.thickness_exponent,
                      "log_sigma": self.force.log_sigma},
            "pulse": {"shape": self.pulse.shape,
                      "rise_fraction": self.pulse.rise_fraction,
                      "duration_model": self.pulse.duration_model,
                      "duration_scale": self.pulse.duration_scale},
            "noise_std_ratio": self.noise_std_ratio,
        }
        if self.clustering is not None:
            payload["clustering"] = {"mean_offspring": self.clustering.mean_offspring,
                                     "sigma_s": self.clustering.sigma_s,
                                     "rate_per_s": self.clustering.rate_per_s}
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CaseSpec":
        ctx = f"case {data.get('name', '<unnamed>')!r}"
        clustering = data.get("clustering")
        return cls(
            name=str(_require(data, "name", ctx)),
            beam_m=float(_require(data, "beam_m", ctx)),
            speed_mps=float(_require(data, "speed_mps", ctx)),
            number_density_per_m2=float(_require(data, "number_density_per_m2", ctx)),
            floe_size=ScalarSpectrum.from_dict(_require(data, "floe_size", ctx), ctx + ".floe_size"),
            thickness=ScalarSpectrum.from_dict(_require(data, "thickness", ctx), ctx + ".thickness"),
            force=ForceModelSpec.from_dict(_require(data, "force", ctx), ctx + ".force"),
            pulse=PulseSpec.from_dict(_require(data, "pulse", ctx), ctx + ".pulse"),
            noise_std_ratio=float(data.get("noise_std_ratio", 0.0)),
            clustering=None if clustering is None else ClusterSpec.from_dict(clustering, ctx + ".clustering"),
        )


# --------------------------------------------------------------------------- #
# manifest
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Manifest:
    """The pre-registered manifest: criteria, execution plan and case data."""

    payload: Mapping[str, Any]
    source: str = "<memory>"

    @property
    def node(self) -> str:
        return str(_require(self.payload, "node", "manifest"))

    @property
    def criteria(self) -> tuple:
        crit = _require(self.payload, "criteria", "manifest")
        if not isinstance(crit, (list, tuple)) or not crit:
            raise ConfigError("manifest 'criteria' must be a non-empty list")
        return tuple(crit)

    @property
    def execution(self) -> Mapping[str, Any]:
        return _require(self.payload, "execution", "manifest")

    @property
    def cases(self) -> Mapping[str, Any]:
        return _require(self.payload, "cases", "manifest")

    def base_case(self, name: str = "base") -> CaseSpec:
        base = _require(self.cases, name, "manifest.cases")
        return CaseSpec.from_dict(base)

    def case(self, name: str) -> CaseSpec:
        """Resolve ``base`` overridden by ``manifest.cases.overrides[name]``."""
        overrides = self.cases.get("overrides", {})
        if name == "base":
            return self.base_case("base")
        if name not in overrides:
            raise ConfigError(f"case override {name!r} not found in manifest.cases.overrides")
        return self.base_case("base").with_overrides(overrides[name])

    def ac5_ranges(self) -> Mapping[str, Any]:
        return _require(self.cases, "ac5_ranges", "manifest.cases")


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=float)


def manifest_hash(payload: Mapping[str, Any]) -> str:
    """SHA-256 of the canonical manifest, used to detect post-hoc edits."""
    import hashlib

    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def load_manifest(path: Optional[pathlib.Path] = None) -> Manifest:
    path = pathlib.Path(path) if path is not None else DEFAULT_PREREG
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:  # pragma: no cover - defensive
        raise ConfigError(f"manifest not found: {path}") from exc
    for key in ("node", "version", "criteria", "execution", "cases"):
        _require(payload, key, f"manifest {path}")
    return Manifest(payload=payload, source=str(path))


__all__ = [
    "CONFIG_DIR",
    "DEFAULT_PREREG",
    "CaseSpec",
    "ClusterSpec",
    "ConfigError",
    "ForceModelSpec",
    "Manifest",
    "PulseSpec",
    "ScalarSpectrum",
    "canonical_json",
    "deep_merge",
    "load_manifest",
    "manifest_hash",
]