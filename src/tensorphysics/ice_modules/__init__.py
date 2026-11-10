"""Ice modules test surface: 8 categories × 5 conditions = 40 verification points.

Modules
-------
- level_ice: Lindqvist/Keinonen/Riska level-ice resistance
- local_pressure: IACS UR I pressure-area law
- ice_plate: Hetenyi elastic foundation plate
- broken_ice: Kotovirta 2009 broken-ice field
- ice_ridge: Riska 2011 ice ridge loads
- ice_abrasion: Su 2010 ice abrasion
- ice_adhesion: Mason 2015 ice adhesion
- ice_accumulation: Bredmoes 2018 ice accumulation
"""

from .level_ice import LevelIceModel, run_level_ice_suite
from .local_pressure import LocalPressureModel, run_local_pressure_suite
from .ice_plate import IcePlateModel, run_ice_plate_suite
from .broken_ice import BrokenIceModel, run_broken_ice_suite
from .ice_ridge import IceRidgeModel, run_ice_ridge_suite
from .ice_abrasion import IceAbrasionModel, run_ice_abrasion_suite
from .ice_adhesion import IceAdhesionModel, run_ice_adhesion_suite
from .ice_accumulation import IceAccumulationModel, run_ice_accumulation_suite

ALL_MODULES = {
    "level_ice": run_level_ice_suite,
    "local_pressure": run_local_pressure_suite,
    "ice_plate": run_ice_plate_suite,
    "broken_ice": run_broken_ice_suite,
    "ice_ridge": run_ice_ridge_suite,
    "ice_abrasion": run_ice_abrasion_suite,
    "ice_adhesion": run_ice_adhesion_suite,
    "ice_accumulation": run_ice_accumulation_suite,
}


def run_all_suites():
    """Run all 8 suites, return dict of module_name -> list[dict]."""
    results = {}
    for name, runner in ALL_MODULES.items():
        results[name] = runner()
    return results
