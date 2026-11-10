"""Metrology of scientific contributions (ontology, evidence fields, unit counting).

See :mod:`tensorphysics.metrology.contribution` for the facade over the contribution meter
implemented in ``tools/contribution_meter``, and :mod:`tensorphysics.metrology.criteria`
for the error-criterion metrology (metric semantics, three-type degeneracy classification,
adaptive criterion family, historical replay).
"""
from .contribution import ContributionMeter, ONTOLOGY_PATH

__all__ = ["ContributionMeter", "ONTOLOGY_PATH"]