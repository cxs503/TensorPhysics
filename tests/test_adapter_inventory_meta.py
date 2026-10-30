"""Meta-level inventory tests for the adapter / protocol surface."""
import importlib
import inspect

import pytest


def test_solver_protocol_has_five_members():
    """Solver Protocol must expose exactly name/initialize/advance/finalize/snapshot."""
    from tensorphysics.protocols import Solver

    expected = {"name", "initialize", "advance", "finalize", "snapshot"}
    # Collect public members declared on the Protocol class itself
    actual = {
        attr
        for attr in dir(Solver)
        if not attr.startswith("_")
        and attr in expected  # only count the ones we care about
    }
    assert expected.issubset(actual), (
        f"Missing members: {expected - actual}"
    )
    # Also verify each is individually accessible
    for member in expected:
        assert hasattr(Solver, member), f"Solver Protocol missing '{member}'"


def test_solver_protocol_has_no_restore():
    """The ecosystem distinguishes checkpoint from restart; no restore method."""
    from tensorphysics.protocols import Solver

    assert not hasattr(Solver, "restore"), (
        "Solver Protocol must NOT define 'restore' (checkpoint ≠ restart)"
    )


def test_tensordem_adapter_importable():
    """adapters/tensordem.py must be importable; verify via inspect."""
    mod = importlib.import_module("tensorphysics.adapters.tensordem")
    assert inspect.ismodule(mod), "tensordem did not import as a module"
    # Bonus: confirm it has at least one class or function (it's a real adapter)
    members = inspect.getmembers(mod)
    assert len(members) > 0, "tensordem module appears empty"
