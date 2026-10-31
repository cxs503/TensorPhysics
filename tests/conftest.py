"""Test bootstrap: always import TensorPhysics from this working tree.

Several worktrees of this repository can be installed editable in the same
environment, in which case ``import tensorphysics`` would resolve to whichever
copy was installed last.  Putting this tree's ``src`` first on ``sys.path``
guarantees the tests exercise the code under test.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))