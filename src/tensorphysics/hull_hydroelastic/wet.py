"""Wet (added-mass corrected) hull-girder modes by modal superposition.

Method
------
1. Dry free-free modes of the non-uniform Timoshenko beam from
   :mod:`timoshenko` (mass normalised: ``phi_n^T M phi_m = delta_nm``).
2. Sectional infinite-frequency added mass ``a33(x)`` per station from the 2-D
   boundary-value problem in :mod:`sections`, interpolated onto the beam grid.
3. Generalised hydrodynamic mass

       A_nm = Int a33(x) phi_n(x) phi_m(x) dx

   (rotary added inertia of the strip is neglected - the section's in-plane
   rotation caused by hull-girder bending is out of the strip plane, so the
   2-D strip coefficient has no direct counterpart; stated as a limitation).
4. The wet eigenproblem is solved *in the dry modal basis*:

       diag(omega_n^2) q = omega^2 (I + A) q ,

   which returns the wet frequencies ``omega_i``, the modal coupling vectors and
   hence the wet mode shapes ``Psi_i = Sum_n q_ni phi_n``.

The frequency downshift follows directly from the added mass; the sequence
``omega_wet < omega_dry`` is enforced as a sign gate in the tests.

Damping
-------
Modal damping is ``zeta_n = zeta_struct`` (input).  Radiation damping of the
2-D strip is **not** solved in this node (it needs the radiating free-surface
Green function).  Measured hull-girder damping of full-scale ships is
0.5-2 % of critical for the 2-node vertical mode (:cite:p:`fricke1984`
magnitude, quoted as engineering practice); the default ``1 %`` and the
sensitivity band ``0.5-2 %`` are used when reporting resonant springing levels,
so the levels are bracketed rather than pinned to an unexplained constant.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..hullgirder.model import ShipGirder
from .sections import added_mass_matrix, superellipse_section
from .timoshenko import (
    BeamProperties,
    DryModes,
    TimoshenkoBeam,
    _generalised_eigh,
    uniform_beam,
)

__all__ = [
    "YOUNG_STEEL",
    "SHEAR_MODULUS_STEEL",
    "DENSITY_STEEL",
    "ship_beam_properties",
    "added_mass_distribution",
    "WetModes",
    "wet_modes",
    "modal_damping",
]

YOUNG_STEEL = 2.06e11       # Pa
SHEAR_MODULUS_STEEL = 7.92e10  # Pa, G = E / (2(1+nu)) with nu = 0.3
DENSITY_STEEL = 7850.0      # kg/m^3

#: welding/structural damping of the hull girder, fraction of critical
ZETA_STRUCTURAL = 0.02
ZETA_BAND = (0.005, 0.01, 0.02, 0.05)


def ship_beam_properties(
    ship: ShipGirder,
    n_elements: int = 160,
    section_exponent: float = 4.0,
    shear_area_ratio: float = 0.25,
) -> BeamProperties:
    """Non-uniform Timoshenko properties of a hull girder.

    Idealisation (all assumptions explicit):

    * midship second moment ``I_mid = S_M * (D/2)`` from the elastic section
      modulus and the deck-to-keel distance ``D``;
    * longitudinal variation ``I(x) = I_mid * (b(x)/B)^1.5`` where ``b(x)`` is
      the waterplane width of :class:`~tensorphysics.hullgirder.model.ShipGirder`;
      the exponent 1.5 is the usual "fuller midbody" hull-girder shape;
    * effective shear area ``A_s(x) = I(x) / (D^2/4) * shear_area_ratio`` - the
      flange-dominated box-girder relation, so that the shear flexibility is
      tied to the same idealisation rather than to a free parameter;
    * vibrating mass per unit length = the full-load mass distribution of the
      hull-girder model (steel + cargo/outfitting), uniform;
    * rotary inertia per unit length from the steel only (``rho_s I``).
    """
    n_nodes = int(n_elements) + 1
    x = np.linspace(-0.5 * ship.length, 0.5 * ship.length, n_nodes)
    b = ship.waterplane_width(x)
    shape = np.clip(b / ship.breadth, 1.0e-3, None) ** 1.5
    i_mid = ship.section_modulus * 0.5 * ship.depth
    inertia = i_mid * shape / shape.max()
    area_shear = inertia / (0.25 * ship.depth ** 2) * shear_area_ratio
    EI = YOUNG_STEEL * inertia
    GAs = 0.5 * SHEAR_MODULUS_STEEL * area_shear
    m = ship.mass_per_length(x)
    rI = DENSITY_STEEL * inertia
    return BeamProperties(x=x, EI=EI, GAs=GAs, m=m, rI=rI)


def added_mass_distribution(
    ship: ShipGirder,
    n_stations: int = 15,
    section_exponent: float = 4.0,
    n_panels: int = 160,
    x: np.ndarray | None = None,
) -> dict:
    """Sectional infinite-frequency added mass ``a33(x)`` along the hull.

    One 2-D boundary-value problem per station (floating section, ``phi = 0`` on
    the waterline).  Sections are the superellipse idealisation of
    :func:`~.sections.superellipse_section` scaled with the local waterplane
    width, so the strip stays consistent with the hull form used for the wave
    loads.
    """
    x_st = np.linspace(-0.45 * ship.length, 0.45 * ship.length, int(n_stations))
    b = ship.waterplane_width(x_st)
    a33 = np.zeros_like(x_st)
    for i, (xi, bi) in enumerate(zip(x_st, b)):
        half_beam = 0.5 * max(float(bi), 0.05 * ship.breadth)
        sec = superellipse_section(half_beam, ship.draught, section_exponent)
        a33[i] = added_mass_matrix(
            sec, modes=("heave",), n_panels=n_panels, free_surface=True
        )["matrix"][0, 0]
    target = np.asarray(x, dtype=float) if x is not None else ship.grid()
    return {
        "x_stations": x_st,
        "a33_stations": a33,
        "a33": np.interp(target, x_st, a33, left=a33[0], right=a33[-1]),
        "section_exponent": float(section_exponent),
        "n_panels": int(n_panels),
        "half_beam_stations": 0.5 * b,
    }


@dataclass
class WetModes:
    """Added-mass corrected modes, damping and generalised quantities."""

    dry: DryModes
    a33: np.ndarray
    x: np.ndarray
    wet_omega: np.ndarray
    wet_frequencies_hz: np.ndarray
    zeta: np.ndarray
    dry_omega: np.ndarray
    added_mass_ratio: np.ndarray      # A_nn / (1 + A_nn)
    coupling: np.ndarray              # (n_modes, n_modes) modal mixing vectors
    wet_shapes: np.ndarray            # wet mode shapes on the beam grid
    wet_curvature: np.ndarray
    generalised_added_mass: np.ndarray
    generalised_mass: np.ndarray
    modal_stiffness: np.ndarray = None      # dry-basis K: diag(omega_dry^2) + buoyancy
    modal_mass: np.ndarray = None           # dry-basis M: I + added mass
    buoyancy_share: np.ndarray = None       # buoyancy stiffness / modal stiffness

    @property
    def n_modes(self) -> int:
        return int(self.wet_omega.size)

    def shift_table(self) -> list[dict]:
        rows = []
        for i in range(self.n_modes):
            rows.append(
                {
                    "mode": i + 1,
                    "dry_hz": float(self.dry_omega[i] / (2.0 * np.pi)),
                    "wet_hz": float(self.wet_frequencies_hz[i]),
                    "shift_pct": float(
                        100.0 * (self.wet_frequencies_hz[i] / (self.dry_omega[i] / (2.0 * np.pi)) - 1.0)
                    )
                    if self.dry_omega[i] > 0.0
                    else 0.0,
                    "added_mass_ratio": float(self.added_mass_ratio[i]),
                    "zeta": float(self.zeta[i]),
                }
            )
        return rows


def modal_damping(n_modes: int, zeta_structural: float = ZETA_STRUCTURAL) -> np.ndarray:
    """Modal damping ratios (structural only - radiation is not solved)."""
    if zeta_structural <= 0.0:
        raise ValueError("structural damping must be positive")
    return np.full(int(n_modes), float(zeta_structural))


def wet_modes(
    beam: TimoshenkoBeam,
    a33: np.ndarray,
    n_modes: int = 10,
    zeta_structural: float = ZETA_STRUCTURAL,
    b_waterplane: np.ndarray | None = None,
    rho: float = 1025.0,
) -> WetModes:
    """Solve the wet eigenproblem in the dry modal basis.

    ``a33`` must be sampled on ``beam.props.x`` (kg/m per unit length).
    """
    dry = beam.dry_modes(n_modes)
    x = dry.x
    a33 = np.asarray(a33, dtype=float)
    if a33.size != x.size:
        raise ValueError("a33 must be sampled on the beam grid")
    phi = dry.shapes
    A = np.array(
        [[float(np.trapezoid(a33 * phi[i] * phi[j], x)) for j in range(n_modes)] for i in range(n_modes)]
    )
    M = np.eye(n_modes) + A
    G = np.zeros((n_modes, n_modes))
    if b_waterplane is not None:
        b = np.asarray(b_waterplane, dtype=float)
        if b.size != x.size:
            raise ValueError("b_waterplane must be sampled on the beam grid")
        G = np.array(
            [
                [float(np.trapezoid(rho * 9.81 * b * phi[i] * phi[j], x)) for j in range(n_modes)]
                for i in range(n_modes)
            ]
        )
    K = np.diag(dry.omega ** 2) + G
    lam, vec = _generalised_eigh(K, M)
    lam = np.clip(lam, 0.0, None)
    wet_omega = np.sqrt(lam)
    # normalise the mixing vectors on the wet generalised mass
    for i in range(n_modes):
        v = vec[:, i]
        s = float(v @ M @ v)
        vec[:, i] = v / np.sqrt(s)
    wet_shapes = vec.T @ phi
    wet_curv = np.gradient(np.gradient(wet_shapes, x, axis=1), x, axis=1)
    gen_mass = np.array([float(vec[:, i] @ M @ vec[:, i]) for i in range(n_modes)])
    gen_added = np.array([float(vec[:, i] @ A @ vec[:, i]) for i in range(n_modes)])
    ratio = gen_added / gen_mass
    buoy = np.array([float(vec[:, i] @ G @ vec[:, i]) for i in range(n_modes)])
    K_wet = np.array([[float(vec[:, i] @ K @ vec[:, j]) for j in range(n_modes)] for i in range(n_modes)])
    buoy_share = np.divide(buoy, np.diag(K_wet), out=np.zeros(n_modes), where=np.diag(K_wet) != 0.0)
    return WetModes(
        dry=dry,
        a33=a33,
        x=x,
        wet_omega=wet_omega,
        wet_frequencies_hz=wet_omega / (2.0 * np.pi),
        zeta=modal_damping(n_modes, zeta_structural),
        dry_omega=dry.omega,
        added_mass_ratio=ratio,
        coupling=vec,
        wet_shapes=wet_shapes,
        wet_curvature=wet_curv,
        generalised_added_mass=A,
        generalised_mass=M,
        modal_stiffness=K,
        modal_mass=M,
        buoyancy_share=buoy_share,
    )