"""Optional adapters for specialist Tensor solvers.

Import adapters from this package explicitly; the orchestration core never
imports solver packages as a side effect.
"""

from .tensordem import TensorDEMAdapter

__all__ = ["TensorDEMAdapter"]
