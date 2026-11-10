"""Voyage-accumulated ice-induced fatigue damage (single window -> whole life).

Public API lives in :mod:`tensorphysics.voyage_fatigue.voyage_fatigue`; this
package re-exports it so that ``from tensorphysics.voyage_fatigue import
build_reference_case`` works.
"""

from .voyage_fatigue import *  # noqa: F401,F403
from .voyage_fatigue import __all__ as _all

__all__ = list(_all)
