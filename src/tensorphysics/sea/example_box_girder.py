"""End-to-end SEA example: simple box girder under ice load PSD.

This example demonstrates the complete SEA workflow:
1. Define subsystems (plates, beams, cavity)
2. Compute modal densities
3. Compute loss factors (ILF, CLF)
4. Compute input power from ice load PSD
5. Solve SEA equations
6. Compute response quantities (velocity level, sound power, SPL)

The box girder represents a simplified ship hull cross-section:
- Bottom plate (ice contact)
- Side plates (2)
- Deck plate
- Stiffeners (beams)
- Internal cavity (cargo hold)
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .subsystems import PlateSubsystem, BeamSubsystem, CavitySubsystem
from .modal_density import plate_modal_density, beam_modal_density, cavity_modal_density
from .loss_factors import (
    internal_loss_factor, coupling_loss_factor_plates, radiation_loss_factor,
)
from .power_input import ice_load_psd, input_power_from_psd
from .sea_solver import SEASubsystem, SEASystem
from .response import (
    vibration_velocity_level, sound_power_level, sound_pressure_level,
)


@dataclass
class BoxGirderSEA:
    """Complete SEA model of a box girder under ice loading."""

    # Geometry
    bottom_length: float = 10.0      # m
    bottom_width: float = 8.0        # m
    side_height: float = 4.0         # m
    deck_length: float = 10.0        # m
    deck_width: float = 8.0          # m

    # Material (steel)
    young_modulus: float = 2.1e11    # Pa
    density: float = 7850.0          # kg/m³
    poisson_ratio: float = 0.3
    plate_thickness: float = 0.012   # m (12 mm)

    # Damping
    material_loss: float = 0.005
    structural_loss: float = 0.003

    # Ice load
    ice_psd_level: float = 1e6       # N²/Hz (white noise reference)
    reference_force: float = 1e6     # N

    # Fluid (air inside)
    sound_speed: float = 343.0       # m/s
    air_density: float = 1.225       # kg/m³

    def build_subsystems(self) -> list[SEASubsystem]:
        """Build all SEA subsystems."""
        # Bottom plate (ice contact)
        bottom = PlateSubsystem(
            name="bottom_plate",
            length=self.bottom_length,
            width=self.bottom_width,
            thickness=self.plate_thickness,
            young_modulus=self.young_modulus,
            density=self.density,
            poisson_ratio=self.poisson_ratio,
            loss_factor=self.material_loss,
        )

        # Side plates
        side1 = PlateSubsystem(
            name="side_plate_1",
            length=self.side_height,
            width=self.bottom_length,
            thickness=self.plate_thickness,
            young_modulus=self.young_modulus,
            density=self.density,
            poisson_ratio=self.poisson_ratio,
            loss_factor=self.material_loss,
        )

        side2 = PlateSubsystem(
            name="side_plate_2",
            length=self.side_height,
            width=self.bottom_length,
            thickness=self.plate_thickness,
            young_modulus=self.young_modulus,
            density=self.density,
            poisson_ratio=self.poisson_ratio,
            loss_factor=self.material_loss,
        )

        # Deck plate
        deck = PlateSubsystem(
            name="deck_plate",
            length=self.deck_length,
            width=self.deck_width,
            thickness=self.plate_thickness,
            young_modulus=self.young_modulus,
            density=self.density,
            poisson_ratio=self.poisson_ratio,
            loss_factor=self.material_loss,
        )

        # Internal cavity
        cavity = CavitySubsystem(
            name="cargo_hold",
            volume=self.bottom_length * self.bottom_width * self.side_height,
            sound_speed=self.sound_speed,
            density=self.air_density,
        )

        # Create SEA subsystems
        sea_subs = []
        for plate in [bottom, side1, side2, deck]:
            n = plate_modal_density(plate, 1000.0)  # reference frequency
            mass = plate.density * plate.thickness * plate.area
            sea_sub = SEASubsystem(
                subsystem=plate,
                modal_density=n,
                internal_loss_factor=internal_loss_factor(
                    material_loss=plate.loss_factor,
                    structural_loss=self.structural_loss,
                ),
            )
            sea_sub._mass = mass  # store for response calculation
            sea_subs.append(sea_sub)

        # Cavity
        n_cav = cavity_modal_density(cavity, 1000.0)
        sea_cav = SEASubsystem(
            subsystem=cavity,
            modal_density=n_cav,
            internal_loss_factor=cavity.loss_factor,
        )
        sea_subs.append(sea_cav)

        return sea_subs

    def build_coupling(self, sea_subs: list[SEASubsystem]) -> dict[tuple[int, int], float]:
        """Build coupling loss factor matrix."""
        coupling = {}
        freq = 1000.0  # reference frequency

        # Plate-to-plate couplings (adjacent plates share edges)
        # bottom (0) <-> side1 (1)
        eta = coupling_loss_factor_plates(
            sea_subs[0].subsystem, sea_subs[1].subsystem, freq,
            joint_length=self.bottom_length,
        )
        coupling[(0, 1)] = eta
        coupling[(1, 0)] = eta  # reciprocity

        # bottom (0) <-> side2 (2)
        eta = coupling_loss_factor_plates(
            sea_subs[0].subsystem, sea_subs[2].subsystem, freq,
            joint_length=self.bottom_length,
        )
        coupling[(0, 2)] = eta
        coupling[(2, 0)] = eta

        # side1 (1) <-> deck (3)
        eta = coupling_loss_factor_plates(
            sea_subs[1].subsystem, sea_subs[3].subsystem, freq,
            joint_length=self.bottom_length,
        )
        coupling[(1, 3)] = eta
        coupling[(3, 1)] = eta

        # side2 (2) <-> deck (3)
        eta = coupling_loss_factor_plates(
            sea_subs[2].subsystem, sea_subs[3].subsystem, freq,
            joint_length=self.bottom_length,
        )
        coupling[(2, 3)] = eta
        coupling[(3, 2)] = eta

        # side1 (1) <-> side2 (2) - through end bulkheads (simplified)
        coupling[(1, 2)] = 0.001
        coupling[(2, 1)] = 0.001

        # Plate-to-cavity couplings (structural-acoustic)
        # All plates couple to cavity
        for i in range(4):
            coupling[(i, 4)] = 0.0005  # structural to acoustic
            coupling[(4, i)] = 0.0005  # acoustic to structural (reciprocity)

        return coupling

    def set_input_power(
        self,
        sea_subs: list[SEASubsystem],
        frequency: float,
        psd_type: str = "white",
    ) -> None:
        """Set input power from ice load PSD."""
        # Ice load acts primarily on bottom plate
        psd = ice_load_psd(
            frequency,
            psd_type=psd_type,
            psd_level=self.ice_psd_level,
            reference_force=self.reference_force,
        )

        bottom_plate = sea_subs[0].subsystem
        pin = input_power_from_psd(bottom_plate, frequency, psd)
        sea_subs[0].input_power = pin

        # Small fraction of energy to side plates (ice splash/impact)
        for i in [1, 2]:
            sea_subs[i].input_power = pin * 0.05

        # No direct input to deck or cavity
        sea_subs[3].input_power = 0.0
        sea_subs[4].input_power = 0.0

    def run_analysis(
        self,
        frequency_bands: list[float] | None = None,
        psd_type: str = "white",
    ) -> dict:
        """Run complete SEA analysis.

        Returns:
            Dict with frequencies, energies, velocity levels, sound power, SPL.
        """
        if frequency_bands is None:
            # Standard 1/3 octave bands from 500 Hz to 5000 Hz
            frequency_bands = [
                500, 630, 800, 1000, 1250, 1600, 2000,
                2500, 3150, 4000, 5000,
            ]

        sea_subs = self.build_subsystems()
        coupling = self.build_coupling(sea_subs)

        results = {
            "frequencies": frequency_bands,
            "energies": [],
            "velocity_levels": [],
            "sound_power_levels": [],
            "spl": [],
        }

        for f in frequency_bands:
            # Set input power
            self.set_input_power(sea_subs, f, psd_type)

            # Update modal densities at this frequency
            for i, sub in enumerate(sea_subs):
                if isinstance(sub.subsystem, PlateSubsystem):
                    sub.modal_density = plate_modal_density(sub.subsystem, f)
                elif isinstance(sub.subsystem, CavitySubsystem):
                    sub.modal_density = cavity_modal_density(sub.subsystem, f)

            # Solve
            system = SEASystem(
                subsystems=sea_subs,
                coupling_matrix=coupling,
                frequency_bands=[f],
            )
            energies = system.solve_at_frequency(f)

            # Compute responses
            vel_levels = []
            sw_levels = []
            spl_vals = []

            for i, sub in enumerate(sea_subs):
                E = energies[i]

                if isinstance(sub.subsystem, PlateSubsystem):
                    mass = sub._mass
                    Lv = vibration_velocity_level(E, mass, f)
                    vel_levels.append(Lv)

                    # Radiation loss factor
                    eta_rad = radiation_loss_factor(
                        f, sub.subsystem.density, sub.subsystem.thickness,
                        sound_speed=self.sound_speed,
                        rho_fluid=self.air_density,
                    )
                    Lw = sound_power_level(E, f, eta_rad)
                    sw_levels.append(Lw)
                elif isinstance(sub.subsystem, CavitySubsystem):
                    cav = sub.subsystem
                    Lp = sound_pressure_level(E, cav.volume, cav.sound_speed, cav.density)
                    spl_vals.append(Lp)
                    vel_levels.append(None)
                    sw_levels.append(None)

            results["energies"].append(energies)
            results["velocity_levels"].append(vel_levels)
            results["sound_power_levels"].append(sw_levels)
            results["spl"].append(spl_vals)

        return results
