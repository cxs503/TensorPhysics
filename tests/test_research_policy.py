import pytest

from tensorphysics.research_contracts import ResearchContractError, ValidationReport
from tensorphysics.research_policy import decide_next_action


REQUIRED = ("finite_values", "time_monotonic", "mass_balance")


def report(status=None, checks=None):
    if checks is None:
        checks = tuple({"name": name, "passed": True} for name in REQUIRED)
    checks = tuple(checks)
    if status is None:
        # ValidationReport forbids a "passed" report that contains a failed
        # check, so the status follows the checks unless a test overrides it.
        status = "passed" if all(check["passed"] for check in checks) else "failed"
    return ValidationReport("run-001", status, checks, "test report")


def decide(r, **kwargs):
    defaults = dict(required_checks=REQUIRED, runs_completed=1, max_runs=3)
    defaults.update(kwargs)
    return decide_next_action(r, **defaults)


def test_passed_gates_allow_next_bounded_run():
    result = decide(report())
    assert result.action == "run_next_bounded_experiment"
    assert result.passed_required_checks == REQUIRED


def test_budget_limit_accepts_validated_result_without_extra_run():
    result = decide(report(), runs_completed=3, max_runs=3)
    assert result.action == "accept_validated_result"


def test_missing_required_check_fails_closed():
    result = decide(report(checks=({"name": "finite_values", "passed": True},)))
    assert result.action == "stop_and_repair"
    assert set(result.missing_required_checks) == {"time_monotonic", "mass_balance"}


def test_failed_required_check_never_promotes_result():
    checks = (
        {"name": "finite_values", "passed": True},
        {"name": "time_monotonic", "passed": False},
        {"name": "mass_balance", "passed": True},
    )
    result = decide(report(checks=checks))
    assert result.action == "stop_and_repair"
    assert result.failed_checks == ("time_monotonic",)


def test_nonpassed_report_cannot_be_accepted():
    result = decide(report(status="incomplete"))
    assert result.action == "stop_and_repair"


def test_novel_failure_escalates_instead_of_blind_retry():
    checks = ({"name": name, "passed": False} for name in REQUIRED)
    result = decide(report(checks=tuple(checks)), ambiguous_or_novel_failure=True)
    assert result.action == "escalate_to_stronger_reasoner"


def test_failed_gates_at_budget_stop_without_retry():
    checks = ({"name": name, "passed": False} for name in REQUIRED)
    result = decide(report(checks=tuple(checks)), runs_completed=3, max_runs=3)
    assert result.action == "stop_budget_exhausted"


@pytest.mark.parametrize("runs,max_runs", [(-1, 2), (0, 0), (True, 2), (1, True)])
def test_invalid_budgets_are_rejected(runs, max_runs):
    with pytest.raises(ResearchContractError):
        decide(report(), runs_completed=runs, max_runs=max_runs)


def test_empty_required_checks_are_rejected():
    with pytest.raises(ResearchContractError, match="required_checks"):
        decide_next_action(report(), required_checks=(), runs_completed=1, max_runs=3)


def test_a_passed_report_cannot_contain_a_failed_check():
    """The contract rejects the combination directly, not just via the policy.

    Guards the helper above: a report that claims "passed" while carrying a
    failed check must never be constructible.
    """
    checks = tuple({"name": name, "passed": False} for name in REQUIRED)
    with pytest.raises(ResearchContractError, match="cannot contain failed checks"):
        ValidationReport("run-001", "passed", checks, "test report")
