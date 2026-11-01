# Ice-cover dynamics module (`tensorphysics.ice`)

Domain package for ice-structure interaction: an elastic ice plate coupled to a
potential-flow water layer, bow-impact loading, flexural fracture and crack
initiation.

| module | role |
|---|---|
| `dispersion.py` | flexural-gravity dispersion `omega^2 = (D k^4 + rho_w g) / (rho_i h + m_a(k))`; potential-flow added mass `m_a = rho_w/(k tanh kH)`; independent depth-FEM verification channel |
| `static_plate.py` | Hetényi/Kelvin-function solution for an infinite plate on a buoyant foundation, discrete 13-point biharmonic FFT solver, grid-free Hankel quadrature reference |
| `coupled.py` | spectral time-domain plate-fluid solver (Newmark average acceleration, unconditionally stable), energy ledger, underwater radiation pressure |
| `contact.py` | wedge / spherical bow geometry, pressure-area crushing law `p = C A^-0.5`, ramming dynamics with the closed-form peak-load anchor |
| `fracture.py` | Kirchhoff bottom/top surface stress recovery, principal-stress fracture criterion, crack initiation location/time, ring-crack radius, fracture energy |
| `campaign.py` | end-to-end driver: acceptance criteria (a)-(f), JSON bundle, figures, animation |

Run the acceptance bundle:

```bash
PYTHONPATH=src python -m tensorphysics.ice.campaign artifacts/m1138
PYTHONPATH=src python -m pytest tests/test_ice_*.py -q
```

Full report: `docs/reports/m1138/REPORT.md` (results, tolerance table, limitations).
