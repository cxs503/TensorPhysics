"""Ice-class drivetrain torsional chain with multiphysics time-step coordination.

Public surface
--------------
``gear``      periodic mesh stiffness, gear pair, parametric (Mathieu) oscillator
``drivetrain`` lumped inertia/spring chain (propeller - shaft - gear - shaft - motor)
``motor``     PMSM with back-EMF feedback coupled to the chain
``excitation`` ice-milling torque spectrum
``timestep``  stability limits, synchronized/sub-/super-cycling coordinator,
              interface work residual
``iacs``      Polar Class torsional screening and Miner fatigue life
``theory``    closed-form references used by the acceptance tests
``acceptance``C1..C6 computations producing the machine-checkable report dict
"""

from .gear import (GearPair, MeshStiffness, ParametricMeshOscillator,
                   mathieu_growth_rate, mathieu_instability_halfwidth)
from .drivetrain import (DrivetrainChain, InertiaNode, ShaftSegment,
                         polar_section_modulus, standard_chain,
                         uniform_segment_stiffness)
from .motor import PMSM, motor_power_balance, simulate_motor, simulate_motor_drivetrain
from .excitation import MODEL_ASSUMPTIONS, IceMillingTorque
from .timestep import (AdaptiveTimeStep, CoordinationPlan, CoarseTorqueReconstruction,
                       SolverRate, TimeStepCoordinator, cfl_step, circuit_step,
                       exchange_step, exchange_work_residual, mesh_step,
                       structural_accuracy_step, structural_explicit_step)
from .iacs import (POLAR_CLASS_TORSION_LIMITS, allowable_torsional_stress,
                   exposure_life, fatigue_limit, iacs_torsion_check, miner_damage,
                   sn_life)

__all__ = [
    "GearPair", "MeshStiffness", "ParametricMeshOscillator",
    "mathieu_growth_rate", "mathieu_instability_halfwidth",
    "DrivetrainChain", "InertiaNode", "ShaftSegment", "polar_section_modulus",
    "standard_chain", "uniform_segment_stiffness",
    "PMSM", "motor_power_balance", "simulate_motor", "simulate_motor_drivetrain",
    "MODEL_ASSUMPTIONS", "IceMillingTorque",
    "AdaptiveTimeStep", "CoordinationPlan", "CoarseTorqueReconstruction",
    "SolverRate", "TimeStepCoordinator", "cfl_step", "circuit_step",
    "exchange_step", "exchange_work_residual", "mesh_step", "structural_accuracy_step",
    "structural_explicit_step",
    "POLAR_CLASS_TORSION_LIMITS", "allowable_torsional_stress", "exposure_life",
    "fatigue_limit",
    "iacs_torsion_check", "miner_damage", "sn_life",
]