"""Deterministic, hash-pinned sampling of the uncertain input space.

Two responsibilities:

1. :func:`draw_unit` -- fill the unit hypercube with a **deterministic**
   design (scrambled Sobol' sequence, Latin hypercube, or plain Monte Carlo)
   from an integer seed.
2. :func:`sample` -- map that design through a :class:`~tensorphysics.uq.sample_spec.SampleSpec`
   and pin the result with a sha256 *sample-set hash* that covers the
   specification fingerprint, the rule, the seed, the sample count and the
   actual physical point matrix.

The hash is what makes a UQ verdict auditable: solver results can only be
consumed by the very sample set that produced them (see
:func:`tensorphysics.uq.adapter.fanout_evaluate`, which refuses to merge a
shard whose hash does not match).  Identical ``(spec, rule, seed, n)`` always
produce bit-identical points, and therefore bit-identical Sobol' indices --
this is asserted by ``tests/test_uq_sampler.py``.

The choice of rule is recorded, never inferred: a Scrambled Sobol' design with
``n`` not a power of two loses part of its balance property, which is why
:func:`draw_unit` emits a warning-free design and reports the deviation from a
power of two in :attr:`SampleSet.balance_exponent`.
"""
from __future__ import annotations

import hashlib
import json
import math
import struct
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.stats import qmc

from .sample_spec import SampleSpec

__all__ = [
    "SAMPLE_HASH_VERSION",
    "RULES",
    "SampleSet",
    "draw_unit",
    "sample",
    "compute_sample_hash",
]

#: Bumped whenever the hash payload changes (invalidates stored evidence).
SAMPLE_HASH_VERSION = "tpuq-sample-v1"

#: Supported sampling rules.
RULES = ("sobol", "lhs", "mc")


def draw_unit(
    n: int,
    dim: int,
    *,
    rule: str = "sobol",
    seed: int = 20240,
) -> np.ndarray:
    """Draw ``n`` points in ``[0, 1)^dim`` with a deterministic design.

    Parameters
    ----------
    n, dim:
        Sample count and dimension (both positive integers).
    rule:
        ``"sobol"`` (scrambled Sobol', default), ``"lhs"`` (scrambled Latin
        hypercube) or ``"mc"`` (pseudo-random Monte Carlo).
    seed:
        Integer seed.  The same ``(n, dim, rule, seed)`` yields bit-identical
        points in the same NumPy/SciPy build.
    """
    if not isinstance(n, int) or n < 1:
        raise ValueError(f"n must be a positive integer, got {n!r}")
    if not isinstance(dim, int) or dim < 1:
        raise ValueError(f"dim must be a positive integer, got {dim!r}")
    if rule not in RULES:
        raise ValueError(f"unknown sampling rule {rule!r}; expected one of {RULES}")
    rng = np.random.default_rng(seed)
    if rule == "mc":
        return rng.random((n, dim))
    if rule == "sobol":
        engine = qmc.Sobol(d=dim, scramble=True, seed=rng)
    else:
        engine = qmc.LatinHypercube(d=dim, scramble=True, seed=rng)
    if rule == "sobol" and n & (n - 1) and n > 1:
        # scipy warns that the balance properties of Sobol' points need a power
        # of two; the shortfall is recorded in the SampleSet, not silenced away.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return engine.random(n)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return engine.random(n)


def _array_digest(a: np.ndarray) -> str:
    """sha256 over shape + C-order float64 bytes (endianness pinned to '<')."""
    arr = np.ascontiguousarray(a, dtype="<f8")
    h = hashlib.sha256()
    h.update(struct.pack("<II", int(arr.shape[0]), int(arr.shape[1]) if arr.ndim > 1 else 1))
    h.update(arr.tobytes())
    return h.hexdigest()


def compute_sample_hash(
    *, spec_hash: str, rule: str, seed: int, points: np.ndarray
) -> str:
    """Canonical sample-set hash (version tag + metadata + point digest)."""
    payload = json.dumps(
        {
            "version": SAMPLE_HASH_VERSION,
            "spec": spec_hash,
            "rule": rule,
            "seed": int(seed),
            "n": int(np.shape(points)[0]),
            "dim": int(np.shape(points)[1]),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    h = hashlib.sha256(payload.encode("utf-8"))
    h.update(_array_digest(np.asarray(points)).encode("ascii"))
    return h.hexdigest()


@dataclass(frozen=True, eq=False)
class SampleSet:
    """A design of ``n`` physical parameter vectors plus its provenance.

    Attributes
    ----------
    spec:
        The uncertain-input specification the points were drawn for.
    rule, seed:
        Sampling rule and seed used (part of the hash).
    unit:
        ``(n, d)`` unit-hypercube design.
    latent:
        ``(n, d)`` latent Gaussian coordinates (copula-mixed).
    points:
        ``(n, d)`` physical parameter matrix.
    hash:
        sha256 sample-set hash, see :func:`compute_sample_hash`.
    """

    spec: SampleSpec = field(repr=False, compare=False)
    rule: str
    seed: int
    unit: np.ndarray = field(repr=False)
    latent: np.ndarray = field(repr=False)
    points: np.ndarray = field(repr=False)
    hash: str = ""

    @property
    def n(self) -> int:
        return int(self.points.shape[0])

    @property
    def dim(self) -> int:
        return int(self.points.shape[1])

    @property
    def names(self) -> tuple[str, ...]:
        return self.spec.names

    @property
    def balance_exponent(self) -> int | None:
        """``log2(n)`` when ``n`` is a power of two (Sobol' balance), else ``None``."""
        n = self.n
        if n > 0 and n & (n - 1) == 0:
            return int(math.log2(n))
        return None

    def subset(self, indices: np.ndarray) -> "SampleSet":
        """A named subset (used by the shard runner); hash follows the subset."""
        idx = np.asarray(indices, dtype=int)
        pts = self.points[idx]
        return SampleSet(
            spec=self.spec,
            rule=self.rule,
            seed=self.seed,
            unit=self.unit[idx],
            latent=self.latent[idx],
            points=pts,
            hash=compute_sample_hash(
                spec_hash=self.spec.fingerprint(), rule=self.rule, seed=self.seed, points=pts
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": SAMPLE_HASH_VERSION,
            "hash": self.hash,
            "spec_hash": self.spec.fingerprint(),
            "rule": self.rule,
            "seed": int(self.seed),
            "n": self.n,
            "dim": self.dim,
            "names": list(self.names),
            "balance_exponent": self.balance_exponent,
            "point_digest": _array_digest(self.points),
        }


def sample(
    spec: SampleSpec,
    n: int,
    *,
    rule: str = "sobol",
    seed: int = 20240,
) -> SampleSet:
    """Draw ``n`` physical points for ``spec`` and pin them with a hash."""
    if not isinstance(spec, SampleSpec):
        raise TypeError(f"spec must be a SampleSpec, got {type(spec)!r}")
    unit = draw_unit(n, spec.dim, rule=rule, seed=seed)
    latent = spec.latent_from_unit(unit)
    points, _ = spec.from_latent(latent)
    digest = compute_sample_hash(
        spec_hash=spec.fingerprint(), rule=rule, seed=seed, points=points
    )
    return SampleSet(
        spec=spec,
        rule=rule,
        seed=seed,
        unit=unit,
        latent=latent,
        points=points,
        hash=digest,
    )