"""Domain-level physics modules built on the TensorPhysics contracts.

Each subpackage is a self-contained reduced-order physics kernel that can be
driven by the platform's research runner (provenance, manifests, acceptance
gates) without touching the specialist solvers TensorLBM/FVM/FEM/DEM.
"""