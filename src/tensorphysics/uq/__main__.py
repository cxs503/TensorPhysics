"""``python -m tensorphysics.uq`` -> the uncertainty-budget CLI."""
from .cli import main

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())