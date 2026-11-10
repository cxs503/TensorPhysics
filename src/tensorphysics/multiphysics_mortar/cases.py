"""The three physical validation cases of the conservation campaign node.

Every case is an *interface* between two solvers, described by

* a **master** (donor) trace space -- the fine-mesh side, e.g. the flow solver, and
* a **slave** (receiver) trace space -- the coarse-mesh side, e.g. the structural /
  ice solver,

plus the material state ``(rho, u, T)`` that the donor hands over.  The meshes are
non-matching by construction (different node counts, graded vs uniform, or crossing
diagonals in the 2D chart), so no pair of cells shares an edge.

``C1`` 1D scalar transport
    Passively advected scalar (and its momentum/energy companions) across a
    ``[0, 1]`` interface: fine uniform master (17 nodes) vs graded slave (8 nodes).

``C2`` 2D FSI panel
    Wetted curved panel ``y = A sin(2 pi x / L)`` in the plane; the fluid side
    (41 uniform nodes) hands the state to the structure side (13 cosine-clustered
    nodes).  The interface measure is the arc-length Jacobian
    ``dGamma/dx = sqrt(1 + y'^2)``, so the transfer is integrated on the true
    wetted length, not on the parameter.

``C3`` 3D flow-ice patch
    Wetted hull patch of a conical bow section in the chart ``(theta, z)`` with
    surface Jacobian ``J = r sqrt(1 + r'^2)`` (m^2).  Ice side = coarse
    ``5 x 4`` triangulation, flow side = fine ``14 x 9`` triangulation with the
    opposite diagonals (crossing, genuinely non-matching).

``C5`` refinement family
    Same 3D patch but on a *curved* section, where ``J`` is not polynomial: the
    quadrature defect is then the only thing standing between the two sides'
    measures, so the residual decays with refinement and can be tracked.

Values: ``rho`` in kg/m^3, ``u`` in m/s, ``T`` in K.  ``specific_heat`` is a
caloric stand-in (``c_v ~= c_p`` for a weakly compressible liquid); the numbers are
declared per case in :data:`CASE_PARAMETERS`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .assembly import MortarTransfer, NearestNeighborTransfer, build_coupling
from .conservation import MaterialState
from .spaces import (LineTraceSpace, TriTraceSpace, chart_metric, cosine_nodes,
                     structured_triangles)

__all__ = [
    "CASE_PARAMETERS",
    "InterfaceCase",
    "fsi_2d_case",
    "flow_ice_3d_case",
    "matching_2d_case",
    "matching_line_case",
    "refined_flow_ice_case",
    "scalar_transport_1d_case",
]

CASE_PARAMETERS: dict[str, dict] = {
    "C1_1d_scalar_transport": {
        "interface": "xi in [0, 1] (advection interface)",
        "master_nodes": 17, "slave_nodes": 8,
        "rho": "1.0 + 0.35 sin(2 pi x) + 0.12 x^2  [kg/m^3]",
        "u_x": "0.80 + 0.20 cos(3 pi x)  [m/s]",
        "T": "285 + 15 cos(2 pi x) + 4 x  [K]",
        "specific_heat": 1005.0,
        "note": "scalar-transport abstraction: 1 velocity component, target is the mass flux",
    },
    "C2_2d_fsi_panel": {
        "interface": "y = 0.25 sin(2 pi x / 4), x in [0, 4] m",
        "master_nodes": 41, "slave_nodes": 13,
        "slave_grading": "cosine clustering at both ends (structure mesh)",
        "rho": "1025 (1 + 0.02 sin(3s) + 0.01 x/L)  [kg/m^3], s = 2 pi x / L",
        "u": "(0.42 sin s + 0.08, 0.16 cos 2s)  [m/s]",
        "T": "283.15 + 6 cos s - 1.5 x/L  [K]",
        "specific_heat": 3993.0,
        "note": "seawater caloric stand-in c_v ~= c_p",
    },
    "C3_3d_flow_ice_patch": {
        "interface": "conical hull patch, chart (theta, z), theta in [0, 0.9], z in [0, 6] m",
        "radius": "r(z) = 5.0 - 0.2 z  [m]  ->  J = r sqrt(1 + r'^2) affine in z",
        "master_tri": "14 x 9 (flow side, flip=True)", "slave_tri": "5 x 4 (ice side, flip=False)",
        "rho": "1025 (1 + 0.02 cos(2 pi z / L) sin(3 theta))  [kg/m^3]",
        "u": "(0.60 sin(pi z/L), 0.25 cos(4 theta), 0.08)  [m/s]",
        "T": "271.35 + 0.8 sin(3.5 theta) - 0.15 z/L  [K]",
        "specific_heat": 3993.0,
        "note": "ice-water interface at the freezing point of seawater (271.35 K, 35 psu)",
    },
}

_L_1D = 1.0
_L_FSI = 4.0
_A_FSI = 0.25
_L_ICE = 6.0
_R0_ICE = 5.0
_GAMMA_ICE = 0.2


# ---------------------------------------------------------------------------
# 1D case
# ---------------------------------------------------------------------------


def _profile_1d(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=float).ravel()
    return {
        "density": 1.0 + 0.35 * np.sin(2.0 * np.pi * x) + 0.12 * x ** 2,
        "velocity": (0.80 + 0.20 * np.cos(3.0 * np.pi * x))[:, None],
        "temperature": 285.0 + 15.0 * np.cos(2.0 * np.pi * x) + 4.0 * x,
    }


def _state_on(space, profile: Callable[[np.ndarray], dict],
              specific_heat: float) -> MaterialState:
    pts = space.nodes if isinstance(space, LineTraceSpace) else space.node_coords
    out = profile(pts)
    return MaterialState(density=out["density"], velocity=out["velocity"],
                         temperature=out["temperature"],
                         specific_heat=specific_heat)


# ---------------------------------------------------------------------------
# 2D FSI case
# ---------------------------------------------------------------------------


def _fsi_y(x: np.ndarray) -> np.ndarray:
    return _A_FSI * np.sin(2.0 * np.pi * np.asarray(x, float) / _L_FSI)


def _fsi_dy(x: np.ndarray) -> np.ndarray:
    return _A_FSI * 2.0 * np.pi / _L_FSI * np.cos(2.0 * np.pi * np.asarray(x, float) / _L_FSI)


def _fsi_metric(x: np.ndarray) -> np.ndarray:
    """Arc-length Jacobian of the wetted panel ``dGamma/dx``."""
    return np.sqrt(1.0 + _fsi_dy(x) ** 2)


def fsi_panel_curve(x: np.ndarray) -> np.ndarray:
    """Panel geometry ``(x, y)`` -- exported for the figures."""
    x = np.asarray(x, dtype=float)
    return np.column_stack([x, _fsi_y(x)])


def _profile_fsi(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=float).ravel()
    s = 2.0 * np.pi * x / _L_FSI
    rho = 1025.0 * (1.0 + 0.02 * np.sin(3.0 * s) + 0.01 * x / _L_FSI)
    u = np.column_stack([0.42 * np.sin(s) + 0.08, 0.16 * np.cos(2.0 * s)])
    T = 283.15 + 6.0 * np.cos(s) - 1.5 * x / _L_FSI
    return {"density": rho, "velocity": u, "temperature": T}


# ---------------------------------------------------------------------------
# 3D flow-ice case
# ---------------------------------------------------------------------------


def conical_hull_metric(r0: float = _R0_ICE, gamma: float = _GAMMA_ICE
                        ) -> Callable[[np.ndarray], np.ndarray]:
    return chart_metric(lambda v: r0 - gamma * np.asarray(v, float),
                        lambda v: np.full_like(np.asarray(v, float), -gamma))


def curved_hull_metric(r0: float = _R0_ICE, gamma: float = _GAMMA_ICE,
                       beta: float = 0.06) -> Callable[[np.ndarray], np.ndarray]:
    """Curved section ``r = r0 - gamma z + beta z^2`` -> non-polynomial Jacobian."""
    return chart_metric(lambda v: r0 - gamma * np.asarray(v, float)
                        + beta * np.asarray(v, float) ** 2,
                        lambda v: -gamma + 2.0 * beta * np.asarray(v, float))


def _profile_ice(pts: np.ndarray) -> dict:
    pts = np.asarray(pts, dtype=float).reshape(-1, 2)
    theta, z = pts[:, 0], pts[:, 1]
    rho = 1025.0 * (1.0 + 0.02 * np.cos(2.0 * np.pi * z / _L_ICE) * np.sin(3.0 * theta))
    u = np.column_stack([0.60 * np.sin(np.pi * z / _L_ICE),
                         0.25 * np.cos(4.0 * theta),
                         np.full_like(theta, 0.08)])
    T = 271.35 + 0.8 * np.sin(3.5 * theta) - 0.15 * z / _L_ICE
    return {"density": rho, "velocity": u, "temperature": T}


# ---------------------------------------------------------------------------
# case containers
# ---------------------------------------------------------------------------


@dataclass
class InterfaceCase:
    """One master -> slave interface exchange with its diagnostics."""

    key: str
    title: str
    master_space: object
    slave_space: object
    master_state: MaterialState
    slave_state: MaterialState
    parameters: dict = field(default_factory=dict)

    def transfer(self, subdivisions: int = 1) -> MortarTransfer:
        return MortarTransfer(self.master_space, self.slave_space,
                              subdivisions=subdivisions, label=self.key)

    def baseline(self) -> NearestNeighborTransfer:
        return NearestNeighborTransfer(self.master_space, self.slave_space)

    def assembly(self):
        return build_coupling(self.master_space, self.slave_space, label=self.key)

    def geometry(self) -> dict:
        """Chart coordinates of both meshes (for the residual maps)."""
        out = {}
        for name, space in (("master", self.master_space), ("slave", self.slave_space)):
            if isinstance(space, LineTraceSpace):
                out[name] = {"nodes": space.nodes.tolist()}
            else:
                out[name] = {"node_coords": space.node_coords.tolist(),
                             "triangles": space.node_ids.tolist(),
                             "tri_xy": space.triangles.tolist()}
        return out


def scalar_transport_1d_case(master_nodes: int = 17, slave_nodes: int = 8
                             ) -> InterfaceCase:
    master = LineTraceSpace(np.linspace(0.0, _L_1D, master_nodes + 1), label="master-1d")
    slave = LineTraceSpace(cosine_nodes(slave_nodes, 0.0, _L_1D), label="slave-1d")
    return InterfaceCase(
        key="C1_1d_scalar_transport",
        title="1D scalar transport across a non-matching interface",
        master_space=master, slave_space=slave,
        master_state=_state_on(master, _profile_1d, 1005.0),
        slave_state=_state_on(slave, _profile_1d, 1005.0),
        parameters=CASE_PARAMETERS["C1_1d_scalar_transport"],
    )


def fsi_2d_case(master_nodes: int = 40, slave_nodes: int = 12) -> InterfaceCase:
    master = LineTraceSpace(np.linspace(0.0, _L_FSI, master_nodes + 1),
                            metric=_fsi_metric, label="master-fluid-2d")
    slave = LineTraceSpace(cosine_nodes(slave_nodes, 0.0, _L_FSI),
                           metric=_fsi_metric, label="slave-structure-2d")
    return InterfaceCase(
        key="C2_2d_fsi_panel",
        title="2D FSI wetted-panel coupling (curved interface, arc-length measure)",
        master_space=master, slave_space=slave,
        master_state=_state_on(master, _profile_fsi, 3993.0),
        slave_state=_state_on(slave, _profile_fsi, 3993.0),
        parameters=CASE_PARAMETERS["C2_2d_fsi_panel"],
    )


def flow_ice_3d_case(master_tri: tuple[int, int] = (14, 9),
                     slave_tri: tuple[int, int] = (5, 4)) -> InterfaceCase:
    metric = conical_hull_metric()
    master = TriTraceSpace(structured_triangles(*master_tri, (0.0, 0.0), (0.9, _L_ICE),
                                                flip=True),
                           metric=metric, label="master-flow-3d")
    slave = TriTraceSpace(structured_triangles(*slave_tri, (0.0, 0.0), (0.9, _L_ICE),
                                               flip=False),
                          metric=metric, label="slave-ice-3d")
    return InterfaceCase(
        key="C3_3d_flow_ice_patch",
        title="3D flow-ice coupling on a conical hull patch",
        master_space=master, slave_space=slave,
        master_state=_state_on(master, _profile_ice, 3993.0),
        slave_state=_state_on(slave, _profile_ice, 3993.0),
        parameters=CASE_PARAMETERS["C3_3d_flow_ice_patch"],
    )


def refined_flow_ice_case(nu_master: int, nu_slave: int,
                          lo: tuple[float, float] = (0.0, 0.0),
                          hi: tuple[float, float] = (0.9, _L_ICE)
                          ) -> InterfaceCase:
    """Curved-section family used by the refinement (C5) study.

    ``nu_master``/``nu_slave`` are the *theta* cell counts; the axial count keeps
    the aspect ratio (``max(2, round(0.65 * nu))``).
    """
    metric = curved_hull_metric()
    nv_m = max(2, int(round(0.65 * nu_master)))
    nv_s = max(2, int(round(0.65 * nu_slave)))
    master = TriTraceSpace(structured_triangles(nu_master, nv_m, lo, hi, flip=True),
                           metric=metric, label=f"master-curved[{nu_master}x{nv_m}]")
    slave = TriTraceSpace(structured_triangles(nu_slave, nv_s, lo, hi, flip=False),
                          metric=metric, label=f"slave-curved[{nu_slave}x{nv_s}]")
    return InterfaceCase(
        key=f"C5_refined_{nu_slave}",
        title="3D flow-ice patch on a curved section (quadrature-limited family)",
        master_space=master, slave_space=slave,
        master_state=_state_on(master, _profile_ice, 3993.0),
        slave_state=_state_on(slave, _profile_ice, 3993.0),
        parameters={"nu_master": nu_master, "nu_slave": nu_slave,
                    "nv_master": nv_m, "nv_slave": nv_s,
                    "radius": "r(z) = 5.0 - 0.2 z + 0.06 z^2  [m] (non-polynomial J)"},
    )


def matching_line_case(n_nodes: int = 9) -> InterfaceCase:
    """Negative control C4 (1D): identical meshes on both sides."""
    nodes = cosine_nodes(n_nodes, 0.0, _L_1D)
    space = LineTraceSpace(nodes, label="matching-1d")
    rng = np.random.default_rng(20261110)
    state = MaterialState(
        density=1.0 + rng.standard_normal(space.n),
        velocity=rng.standard_normal((space.n, 2)),
        temperature=280.0 + 10.0 * rng.random(space.n),
        specific_heat=1005.0)
    return InterfaceCase(key="C4_identity_1d", title="matching 1D meshes -> identity",
                         master_space=space, slave_space=LineTraceSpace(nodes, label="matching-1d"),
                         master_state=state, slave_state=state,
                         parameters={"nodes": n_nodes, "data": "seeded Gaussian nodal values"})


def matching_2d_case(nu: int = 6, nv: int = 5) -> InterfaceCase:
    """Negative control C4 (2D): identical triangulations on both sides."""
    tris = structured_triangles(nu, nv, (0.0, 0.0), (0.9, _L_ICE), flip=False)
    metric = conical_hull_metric()
    space = TriTraceSpace(tris, metric=metric, label=f"matching-2d[{nu}x{nv}]")
    rng = np.random.default_rng(20261111)
    state = MaterialState(
        density=1000.0 + rng.standard_normal(space.n),
        velocity=rng.standard_normal((space.n, 3)),
        temperature=271.0 + 2.0 * rng.random(space.n),
        specific_heat=3993.0)
    return InterfaceCase(key="C4_identity_2d", title="matching 2D patches -> identity",
                         master_space=space,
                         slave_space=TriTraceSpace(tris, metric=metric, label="matching-2d"),
                         master_state=state, slave_state=state,
                         parameters={"tri": f"{nu}x{nv}", "data": "seeded Gaussian nodal values"})