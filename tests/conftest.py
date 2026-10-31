"""Make the tests run against *this* worktree's source tree.

The environment may expose an editable install of ``tensorphysics`` that points
at a different checkout; prepending ``src`` guarantees that the code under test
is the code in this working tree.
"""

import pathlib
import sys

_SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))