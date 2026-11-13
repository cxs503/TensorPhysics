"""``python -m tensorphysics.ice_pressure_area [--check|--seal|--json|--strict]``."""

import sys

from .library import main

if __name__ == "__main__":
    sys.exit(main())
