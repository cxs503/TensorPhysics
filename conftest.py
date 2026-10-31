"""Pytest bootstrap: import this worktree's ``src/`` layout.

The project uses a ``src/`` layout (``src/tensorphysics``).  When several checkouts
are present in the same environment, an editable install of another checkout can win
the import race; inserting this file's ``src`` directory in front of ``sys.path``
guarantees that the tests always exercise the code of the worktree they belong to.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))