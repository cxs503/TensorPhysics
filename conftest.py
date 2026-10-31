"""Pytest configuration for the TensorPhysics worktree.

The package uses a ``src/`` layout, but a ``tensorphysics`` build installed into
``site-packages`` by another checkout shadows it on ``sys.path``.  Prepending
this worktree's ``src/`` directory makes ``python -m pytest tests`` always test
the code in the worktree under test.
"""

from __future__ import annotations

import pathlib
import sys

_SRC = pathlib.Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))