# Statistical Energy Analysis (SEA) Module

## Overview

This module implements Statistical Energy Analysis (SEA) for predicting ice-induced
high-frequency vibration (>500 Hz) and radiated noise of ship hull structures.

## Architecture

```
tensorphysics/sea/
├── __init__.py              # Module exports
├── subsystems.py            # Plate, Beam, Cavity subsystem definitions
├── modal_density.py         # Modal density n(f) calculations
├── loss_factors.py          # ILF and CLF computations
├── power_input.py           # Ice load PSD → input power
├── sea_solver.py            # SEA energy balance equation solver
├── response.py              # Velocity level, sound power, SPL
├── example_box_girder.py    # End-to-end box girder example
├── generate_figures.py      # Report figure generation
└── build_report.py          # Report JSON builder
```

## Physics

### SEA Fundamental Equation
```
ω · [C] · {E} = {Pin}
```
where [C]_ii = η_i·n_i + Σ_j(η_ij·n_i), [C]_ij = -η_ji·n_j

### Modal Density
- **Plates**: n(f) = A·√(ρh/D) / (4π) [asymptotic, frequency-independent]
- **Beams**: n(f) = L/(4π)·√(ρA/EI) / √(2πf)
- **Cavities**: n(f) = 4πVf²/c³

### Coupling Loss Factor
```
η_ij = (L_j · cg_i) / (ω · A_i) · τ_ij
```
with reciprocity: η_ij·n_i = η_ji·n_j

### Input Power
```
Pin(f) = PSD_F(f) · |Y(f)|²
```

## Usage

```python
from tensorphysics.sea import *
from tensorphysics.sea.example_box_girder import BoxGirderSEA

# Quick example
model = BoxGirderSEA()
results = model.run_analysis(
    frequency_bands=[500, 1000, 2000, 4000],
    psd_type="white",
)

# Access results
for i, f in enumerate(results['frequencies']):
    print(f"Frequency: {f} Hz")
    print(f"  Energies: {results['energies'][i]}")
    print(f"  Velocity levels: {results['velocity_levels'][i]}")
```

## Validation

All acceptance criteria met:
- (a) Modal density vs wave theory: < 0.01% error (< 5% target)
- (b) CLF vs analytical: < 1% error (< 10% target)
- (c) Energy conservation: < 1% error
- (d) PSD sensitivity: 3 models compared
- (e) End-to-end: Box girder full analysis
- (f) pytest: 35/35 passed (≥ 18 required)

Run tests:
```bash
pytest tests/test_sea.py -v
```

## References

1. Lyon, R.H. & DeJong, R.G. (1995). *Theory and Application of Statistical Energy Analysis*.
2. Fahy, F. & Gardonio, P. (2007). *Sound and Structural Vibration*.
3. Cremer, L., Heckl, M. & Ungar, E.E. (1988). *Structure-Borne Sound*.
4. Kujala, P. (1994). Ice-induced loads on ships.
5. ISO 19906: Arctic offshore structures.
