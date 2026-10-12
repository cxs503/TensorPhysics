from __future__ import annotations
import json
from pathlib import Path
import sys
import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
sys.path.insert(0, str(EXAMPLES))
from research_campaign_dem3d_load_sensitivity import run_campaign


def test_rejects_invalid_campaign_inputs(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        run_campaign(tmp_path, tmp_path, (0.02,), 24)
    with pytest.raises(ValueError):
        run_campaign(tmp_path, tmp_path, (0.02, 0.02), 24)
    with pytest.raises(ValueError):
        run_campaign(tmp_path, tmp_path, (0.02, float("nan")), 24)
    with pytest.raises(ValueError):
        run_campaign(tmp_path, tmp_path, (0.01, 0.02), 0)
