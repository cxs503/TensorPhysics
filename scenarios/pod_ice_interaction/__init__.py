"""Podded propulsor thrust vectoring and ice-hull load interaction.

Physical sub-models (all first-principles, no tabulated/hard-coded performance data):

* :mod:`propeller`  blade-element/momentum theory (BEMT) open-water model of a
  wide-blade Kaplan type pod propeller (thrust/torque derivatives for the pitch
  and ice-torque coupling).
* :mod:`structure`  three-dimensional frame finite element model of the pod strut
  (axial + bi-axial bending, cantilevered at the hull root) with section stress
  recovery used for the strength constraint of the thrust attainable domain.
* :mod:`contact`   Hertz point contact theory plus a three-dimensional
  hexahedral finite element contact solve (quarter-symmetric graded mesh,
  FE flexibility matrix + linear complementarity problem) used to refine the
  ice-pod housing contact pressure and to supply the von Mises field.
* :mod:`envelope`  thrust attainable domain T_max(azimuth) bounded by motor
  torque, thrust-bearing capacity and strut/hull structural strength.
* :mod:`interaction` propeller-ice interaction: rotor dynamics under a
  blade-rate ice torque spectrum, thrust/torque time histories, spectral phase
  verification against the analytic first-order lag, and the reaction of the pod
  vibration back onto the ice load.
* :mod:`api`        FastAPI router exposing
  ``POST /api/v1/calculations/pod_ice_interaction``.
"""

from __future__ import annotations

__all__ = [
    "propeller",
    "structure",
    "contact",
    "envelope",
    "interaction",
]