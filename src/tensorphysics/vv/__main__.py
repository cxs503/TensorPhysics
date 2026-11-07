"""``python -m tensorphysics.vv`` entry point (delegates to the gate CLI)."""
from __future__ import annotations

import sys

from .cli import main

if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())