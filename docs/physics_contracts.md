# Physics contracts (stage 2: unified resistance chain)

`tensorphysics.physics_contracts` is the single source of truth for *what each
physics model actually computes* and *what its name is allowed to claim*.

## Why

Audit m2086 (full resistance callgraph) found 11 production kernels; 6 agreed
bitwise (ct = 0.003461889 at SUBOFF-01) while spreads reached 357% on SUBOFF,
and the composition ITTC-1957 + empirical residual was published under the
name "Holtrop-Mennen". Audit m2087 found 8 of 12 benchmark reference rows
WITHHELD-class yet used as acceptance anchors. This module makes both failure
modes unrepresentable:

1. **Naming gate** -- a contract whose `declared_method` says "Holtrop-Mennen"
   must implement the full 1984 component set (wave, bulb, transom, Ca,
   form factor); otherwise construction fails.
2. **Provenance-classified references** -- `ReferenceBand(..., provenance=
   "WITHHELD", admissible_for_judgment=True)` raises at registration.

## Registered contracts

| key | declared method | audited aliases |
|---|---|---|
| `resistance.suboff_shared_core` | ITTC-1957 friction + SUBOFF-calibrated residual fit | A, B, C, D, E, I, J |
| `resistance.holtrop1984_full` | Holtrop & Mennen (1984) full method | F, G |
| `friction.ittc1957` | ITTC-1957 correlation line | -- |
| `friction.schoenherr1932` | Schoenherr (1932) friction line | -- |
| `wetted_surface.axisymmetric_empirical` | Axisymmetric empirical S estimate | D2 |

Reference kernels live in `physics_kernels_resistance.py` (pure functions, SI
units). `wetted_surface_m2` is a **required** argument of the shared core --
there is no silent empirical fallback (audited KVLCC2 bias: S = 40958 m^2
from the fit vs 27194 m^2 published, +50.6%).

## Consumer contract (for TensorPlatform)

TP acceptance code must resolve a model by contract key or audited alias and
read `references` with `admissible_for_judgment=True` only. The
`schoenherr1932` line exists as an *independent* cross-check of ITTC-1957;
it is not an acceptance anchor.
