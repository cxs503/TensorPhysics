"""Ice-class hull side-shell strip: 15 deg roll load case and its analytic statics.

Physical problem
----------------
The transverse hull frame of an ice-going ship is loaded on the ice belt of the side
shell by the ice floe while the ship is rolling.  The roll is a *finite* rotation of the
section (the campaign gate is ``roll > 15 deg``), so the load application points and the
lever arms of the loads move with the section even though the ice force itself is fixed
in the world frame (its direction is set by the ship/ice relative motion, not by the
ship's attitude).  The structural response is therefore a large-rotation / small-strain
problem: elastic strain stays at hull-steel level (O(1e-3)), while the section (and the
co-rotated element frames) rotate by tens of degrees.

Idealisation
------------
The side shell is represented by a one-dimensional *strip* (frame spacing wide) clamped
at the double bottom and running up the side shell to the ice belt.  Each element
carries the smeared section properties of the plating plus the attached longitudinal
stiffener, so the strip behaves as an equivalent beam whose bending axis is longitudinal
(``x``), i.e. bending happens inside the transverse plane -- exactly the plane in which
the roll takes place.  This is the standard "transverse frame strip" idealisation used
for ice-belt framing checks; it is statically determinate, which is deliberate: the
internal forces are then given by global equilibrium alone, so the analytic reference
below is exact and independent of the discretisation.

Analytic reference (exact, statically determinate)
--------------------------------------------------
For a clamped cantilever with world-fixed point loads applied at the belt node, the
section resultants at the root are

    N   = t . sum(F_k)                     (t = current strip tangent, unit)
    M   = (r_belt - r_root) x sum(F_k)      (planar cross product, out-of-plane axis)
    sigma = |N| / A + |M| * c / I

evaluated on the *rigidly rolled* geometry (i.e. the finite-rotation update of the
section, no elastic deformation).  Because the elastic deformation of the strip is
small (sub-percent of the height), using the rolled-but-undeformed geometry as reference
introduces only a fraction of a percent of inconsistency; that residue is reported
separately in the validation artefact.

Counter-evidence baseline
-------------------------
:func:`frozen_geometry_root_stress` returns what a small-rotation ("frozen geometry")
analysis produces for the same loads: the load directions/positions of the upright
ship.  At 15 deg roll the lever arm of the section weight is completely lost in that
baseline, which is why the finite-rotation update matters for ice-belt stress checks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .corotational_beam import CorotationalBeamModel, Section
from .finite_rotation import (
    Rotation2D,
    angular_velocity_field,
    roll_transform,
)

__all__ = [
    "HULL_SECTION",
    "HULL_GEOMETRY",
    "build_side_shell_strip",
    "analytic_root_stress_rolled",
    "frozen_geometry_root_stress",
    "IceClassHullRollCase",
    "roll_energy_conservation_case",
    "pure_moment_arc_case",
]

GRAVITY = 9.80665

#: Equivalent section of one frame spacing of ice-belt side shell: 30 mm plating over
#: 0.8 m spacing plus a 400 mm T-stiffener.  A = 0.034 m^2, I = 8.0e-4 m^4 about the
#: longitudinal (bending) axis, extreme fibre 0.20 m from the neutral axis.
HULL_SECTION = Section(
    area=0.034,
    inertia=8.0e-4,
    young_modulus=2.06e11,
    fibre_distance=0.20,
    density=7850.0,
)

HULL_GEOMETRY = {
    "belt_offset_y_m": 10.0,   # half beam at the ice belt (distance from centreline)
    "side_height_m": 8.0,      # double bottom top -> ice belt height
    "n_elements": 8,
}


def build_side_shell_strip(n_elements: int = HULL_GEOMETRY["n_elements"],
                           side_height: float = HULL_GEOMETRY["side_height_m"],
                           belt_offset_y: float = HULL_GEOMETRY["belt_offset_y_m"],
                           section: Section = HULL_SECTION,
                           added_node_mass: dict[int, float] | None = None
                           ) -> tuple[np.ndarray, np.ndarray, CorotationalBeamModel]:
    """Build the upright side-shell strip and its co-rotational beam model.

    Nodes run from the clamped root ``(belt_offset_y, 0)`` upward; node ``0`` is fully
    clamped (its three DOFs are eliminated in every solve).  ``added_node_mass`` lets
    the caller hang non-structural (hull) mass on a node, used by the transient case.
    """
    if n_elements < 1:
        raise ValueError("n_elements must be >= 1")
    z = np.linspace(0.0, side_height, n_elements + 1)
    nodes = np.column_stack([np.full_like(z, float(belt_offset_y)), z])
    elements = np.array([[i, i + 1] for i in range(n_elements)], dtype=int)
    model = CorotationalBeamModel(
        reference_nodes=nodes,
        elements=elements,
        section=section,
        node_mass=_lumped_mass(nodes, elements, section, added_node_mass),
        fixed_dofs=(0, 1, 2),
    )
    return nodes, elements, model


def _lumped_mass(nodes, elements, section, added_node_mass) -> np.ndarray:
    mass = np.zeros(len(nodes))
    lengths = np.linalg.norm(np.diff(nodes, axis=0), axis=1)
    for e, (i, j) in enumerate(elements):
        half = 0.5 * section.mass_per_length * lengths[e]
        mass[i] += half
        mass[j] += half
    if added_node_mass:
        for idx, value in added_node_mass.items():
            mass[int(idx)] += float(value)
    return mass


def _rolled(nodes: np.ndarray, phi: float) -> np.ndarray:
    return Rotation2D(float(phi)).apply(nodes)


def _net_load(ice_force_n: float, attached_mass_kg: float) -> np.ndarray:
    """World-fixed resultant acting at the belt node: ice thrust toward port + weight."""
    return np.array([-float(ice_force_n), -float(attached_mass_kg) * GRAVITY])


def analytic_root_stress_rolled(phi: float, ice_force_n: float,
                                attached_mass_kg: float,
                                section: Section = HULL_SECTION,
                                nodes_upright: np.ndarray | None = None) -> dict:
    """Exact root stress of the rolled strip (rigid rolled geometry, global statics)."""
    if nodes_upright is None:
        nodes_upright, _, _ = build_side_shell_strip(section=section)
    rolled = _rolled(nodes_upright, phi)
    r_root = rolled[0]
    r_belt = rolled[-1]
    arm = r_belt - r_root
    length = float(np.linalg.norm(arm))
    tangent = arm / length
    force = _net_load(ice_force_n, attached_mass_kg)
    n_axial = float(force @ tangent)
    moment = float(arm[0] * force[1] - arm[1] * force[0])
    sigma = abs(n_axial) / section.area + abs(moment) / section.section_modulus
    return {
        "roll_deg": math.degrees(phi),
        "axial_force_n": n_axial,
        "root_moment_n_m": moment,
        "sigma_axial_pa": abs(n_axial) / section.area,
        "sigma_bending_pa": abs(moment) / section.section_modulus,
        "sigma_pa": sigma,
        "lever_arm_z_m": float(arm[1]),
    }


def frozen_geometry_root_stress(ice_force_n: float, attached_mass_kg: float,
                                section: Section = HULL_SECTION,
                                nodes_upright: np.ndarray | None = None) -> dict:
    """Small-rotation ("frozen geometry") baseline for the same load case.

    The loads are applied in the upright configuration and the finite rotation of the
    section is ignored -- this is what a geometrically linear hull analysis returns.
    """
    if nodes_upright is None:
        nodes_upright, _, _ = build_side_shell_strip(section=section)
    arm = nodes_upright[-1] - nodes_upright[0]
    force = _net_load(ice_force_n, attached_mass_kg)
    tangent = arm / float(np.linalg.norm(arm))
    n_axial = float(force @ tangent)
    moment = float(arm[0] * force[1] - arm[1] * force[0])
    sigma = abs(n_axial) / section.area + abs(moment) / section.section_modulus
    return {
        "roll_deg": 0.0,
        "axial_force_n": n_axial,
        "root_moment_n_m": moment,
        "sigma_pa": sigma,
    }


def _deformed_statics(nodes_deformed: np.ndarray, ice_force_n: float,
                      attached_mass_kg: float, section: Section) -> dict:
    """Closed-form root stress evaluated on the *current* (deformed) strip geometry."""
    r_root = nodes_deformed[0]
    arm = nodes_deformed[-1] - r_root
    tangent = arm / float(np.linalg.norm(arm))
    force = _net_load(ice_force_n, attached_mass_kg)
    n_axial = float(force @ tangent)
    moment = float(arm[0] * force[1] - arm[1] * force[0])
    sigma = abs(n_axial) / section.area + abs(moment) / section.section_modulus
    return {"axial_force_n": n_axial, "root_moment_n_m": moment, "sigma_pa": sigma}


@dataclass
class IceClassHullRollCase:
    """The campaign case: ice-belt side-shell strip at a prescribed finite roll."""

    roll_deg: float = 15.0
    ice_force_kN: float = 50.0
    attached_mass_t: float = 20.0
    n_elements: int = HULL_GEOMETRY["n_elements"]
    section: Section = HULL_SECTION

    def __post_init__(self) -> None:
        self.nodes_upright, self.elements, _ = build_side_shell_strip(
            n_elements=self.n_elements, section=self.section)

    # ------------------------------------------------------------------ helpers
    @property
    def phi(self) -> float:
        return math.radians(self.roll_deg)

    def model(self) -> CorotationalBeamModel:
        """Model whose *reference* configuration is the rigidly rolled section.

        The finite-rotation update is therefore applied to the geometry, and the
        nodal rotation DOFs measure the elastic rotation on top of the roll -- a rigid
        roll produces exactly zero strain energy (objective update).
        """
        rolled = _rolled(self.nodes_upright, self.phi)
        return CorotationalBeamModel(
            reference_nodes=rolled,
            elements=self.elements,
            section=self.section,
            fixed_dofs=(0, 1, 2),
        )

    def external_force(self, model: CorotationalBeamModel) -> np.ndarray:
        belt = model.n_nodes - 1
        f = np.zeros(model.n_dof)
        load = _net_load(self.ice_force_kN * 1e3, self.attached_mass_t * 1e3)
        f[3 * belt + 0] = load[0]
        f[3 * belt + 1] = load[1]
        return f

    # -------------------------------------------------------------------- solve
    def solve(self, load_steps: int = 4) -> dict:
        model = self.model()
        f_ext = self.external_force(model)
        out = model.solve_static(f_ext, x0=model.reference_nodes,
                                 theta0=np.zeros(model.n_nodes),
                                 load_steps=load_steps)
        fe_stress = model.max_beam_stress(out["x"], out["theta"])
        analytic = analytic_root_stress_rolled(
            self.phi, self.ice_force_kN * 1e3, self.attached_mass_t * 1e3,
            section=self.section, nodes_upright=self.nodes_upright)
        frozen = frozen_geometry_root_stress(
            self.ice_force_kN * 1e3, self.attached_mass_t * 1e3,
            section=self.section, nodes_upright=self.nodes_upright)
        sigma_fe = fe_stress["sigma_max_pa"]
        sigma_ref = analytic["sigma_pa"]
        error = abs(sigma_fe - sigma_ref) / abs(sigma_ref)
        # equilibrium closure: the same closed form evaluated on the *deformed* strip
        # (this must reproduce the beam resultants to solver accuracy; it isolates the
        #  pure statics identity from the sub-percent elastic arm change)
        deformed = _deformed_statics(out["x"], self.ice_force_kN * 1e3,
                                     self.attached_mass_t * 1e3, self.section)
        closure = abs(sigma_fe - deformed["sigma_pa"]) / abs(deformed["sigma_pa"])
        frozen_error = abs(frozen["sigma_pa"] - sigma_ref) / abs(sigma_ref)
        # objectivity: rigid roll of the rolled reference must store no energy
        objectivity = model.strain_energy(model.reference_nodes,
                                          np.zeros(model.n_nodes))
        return {
            "roll_deg": self.roll_deg,
            "ice_force_kN": self.ice_force_kN,
            "attached_mass_t": self.attached_mass_t,
            "n_elements": self.n_elements,
            "sigma_fe_pa": sigma_fe,
            "sigma_fe_axial_pa": fe_stress["axial_pa"],
            "sigma_fe_bending_pa": fe_stress["bending_pa"],
            "sigma_analytic_pa": sigma_ref,
            "stress_error_ratio": error,
            "sigma_frozen_geometry_pa": frozen["sigma_pa"],
            "frozen_geometry_error_ratio": frozen_error,
            "analytic": analytic,
            "frozen": frozen,
            "root_moment_fe_n_m": float(abs(fe_stress["section_forces"][0, 1])),
            "root_moment_analytic_n_m": abs(analytic["root_moment_n_m"]),
            "sigma_deformed_statics_pa": deformed["sigma_pa"],
            "equilibrium_closure_ratio": closure,
            "rigid_roll_strain_energy_j": float(objectivity),
            "newton_iterations": out["iterations"],
            "newton_residual": out["residual"],
            "model": model,
            "solution": out,
        }

    def stress_vs_roll(self, angles_deg, load_steps: int = 4) -> dict:
        """Sweep the prescribed roll angle: FE, exact statics and frozen baseline."""
        fe, ref, frozen = [], [], []
        for ang in angles_deg:
            case = IceClassHullRollCase(roll_deg=float(ang),
                                        ice_force_kN=self.ice_force_kN,
                                        attached_mass_t=self.attached_mass_t,
                                        n_elements=self.n_elements,
                                        section=self.section)
            res = case.solve(load_steps=load_steps)
            fe.append(res["sigma_fe_pa"] / 1e6)
            ref.append(res["sigma_analytic_pa"] / 1e6)
            frozen.append(res["sigma_frozen_geometry_pa"] / 1e6)
        return {"roll_deg": np.asarray(angles_deg, dtype=float),
                "fe_mpa": np.asarray(fe),
                "analytic_mpa": np.asarray(ref),
                "frozen_mpa": np.asarray(frozen)}


def roll_energy_conservation_case(initial_roll_deg: float = 20.0,
                                  attached_mass_t: float = 20.0,
                                  n_elements: int = 2,
                                  roll_spring_n_m_per_rad: float = 6.0e6,
                                  dt: float = 1.5e-4,
                                  duration: float = 3.0,
                                  sample_every: int = 20,
                                  section: Section = HULL_SECTION) -> dict:
    """Conservative large-rotation transient used for the energy ledger.

    Roll dynamics of the section about its double-bottom root.  The root rotation is
    carried by a generalised (metacentric) roll spring ``k`` -- the stiffness that the
    intact ship provides against heel -- while the side-shell strip itself is elastic and
    co-rotational.  The initial condition is a *pure rigid roll* of ``initial_roll_deg``
    (> 15 deg): rolled nodal positions with nodal rotations equal to the roll angle, so
    the elastic strain energy of that state is exactly zero.  From there the section
    swings as a torsion pendulum with the strip elasticity riding on top.

    All generalised forces derive from potentials (strain energy of the co-rotational
    elements, ``1/2 k theta^2`` of the roll spring, world-fixed weight), so the leapfrog
    integration is symplectic; the relative drift of
    ``T + U_strain + V_spring + V_gravity`` is the gate
    ``large_rotation_energy_conservation``.

    Idealisation limits: dry structure (no hydrostatic added mass/damping, no ice
    contact), and the roll spring lumped at the root node.
    """
    nodes, elements, _ = build_side_shell_strip(n_elements=n_elements,
                                                section=section)
    belt = len(nodes) - 1
    mass = _lumped_mass(nodes, elements, section, {belt: attached_mass_t * 1e3})
    phi0 = math.radians(initial_roll_deg)
    rolled = _rolled(nodes, phi0)
    model = CorotationalBeamModel(reference_nodes=nodes, elements=elements,
                                  section=section, node_mass=mass,
                                  fixed_dofs=(0, 1),
                                  dof_springs=((2, float(roll_spring_n_m_per_rad)),))
    theta0 = np.full(model.n_nodes, phi0)
    strain0 = model.strain_energy(rolled, theta0)
    v0 = np.zeros_like(rolled)
    w0 = np.zeros(model.n_nodes)
    n_steps = int(round(duration / dt))
    history = model.transient(v0=v0, w0=w0, dt=dt, n_steps=n_steps,
                              x0=rolled, theta0=theta0,
                              gravity=GRAVITY, sample_every=sample_every)
    history.update({
        "initial_strain_energy_j": float(strain0),
        "roll_spring_n_m_per_rad": float(roll_spring_n_m_per_rad),
        "gate_ratio": history["energy_drift_ratio"],
        "initial_roll_deg": initial_roll_deg,
        "dt": dt,
        "n_steps": n_steps,
        "attached_mass_t": attached_mass_t,
        "n_elements": n_elements,
        "model": model,
    })
    return history


def pure_moment_arc_case(rotation_deg: float = 45.0,
                         length: float = 60.0, n_elements: int = 12,
                         fibre_distance: float = 0.05,
                         section: Section | None = None,
                         load_steps: int = 10) -> dict:
    """Pure end-moment bending to a large rotation, checked against the exact arc.

    For a cantilever with a pure end moment the exact elastica is a circular arc of
    curvature ``kappa = M / (E I)`` at *any* rotation, so the exact tip position and the
    total rotation are available in closed form.  This verifies that the co-rotational
    update integrates a genuine finite-rotation deformation (not a small-rotation
    approximation); the specimen is deliberately slender so the stress stays elastic.
    """
    if section is None:
        section = Section(area=0.02, inertia=2.0e-5, young_modulus=2.06e11,
                          fibre_distance=fibre_distance, density=7850.0)
    theta_target = math.radians(float(rotation_deg))
    # sign chosen so that the arc bends toward +y in the (y, z) transverse plane
    end_moment_n_m = -section.ei * theta_target / length
    nodes = np.column_stack([np.zeros(n_elements + 1),
                             np.linspace(0.0, length, n_elements + 1)])
    elements = np.array([[i, i + 1] for i in range(n_elements)], dtype=int)
    model = CorotationalBeamModel(reference_nodes=nodes, elements=elements,
                                  section=section, fixed_dofs=(0, 1, 2))
    f_ext = np.zeros(model.n_dof)
    f_ext[3 * n_elements + 2] = end_moment_n_m
    out = model.solve_static(f_ext, x0=nodes, theta0=np.zeros(model.n_nodes),
                             load_steps=load_steps)
    tip = out["x"][-1]
    kappa = abs(end_moment_n_m) / section.ei
    theta_exact = kappa * length
    radius = 1.0 / kappa
    # exact elastica: circular arc starting along +z, tangent rotating to angle theta
    tip_exact = np.array([radius * (1.0 - math.cos(theta_exact)),
                          radius * math.sin(theta_exact)])
    # small-rotation beam theory (same sign convention)
    tip_linear = np.array([kappa * length ** 2 / 2.0, length])
    err = float(np.linalg.norm(tip - tip_exact) / np.linalg.norm(tip_exact))
    err_linear = float(np.linalg.norm(tip_linear - tip_exact)
                       / np.linalg.norm(tip_exact))
    stress = model.max_beam_stress(out["x"], out["theta"])
    return {
        "end_moment_n_m": end_moment_n_m,
        "kappa_per_m": kappa,
        "rotation_deg": math.degrees(theta_exact),
        "tip_fe_m": tip,
        "tip_exact_m": tip_exact,
        "tip_linear_m": tip_linear,
        "tip_error_ratio_fe": err,
        "tip_error_ratio_linear": err_linear,
        "sigma_max_pa": stress["sigma_max_pa"],
        "newton_iterations": out["iterations"],
        "newton_residual": out["residual"],
        "model": model,
        "solution": out,
    }