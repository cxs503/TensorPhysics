import pytest

from tensorphysics.research_contracts import (
    ExperimentPlan, ResearchContractError, ResearchGoal, ResearchMemoryRecord, ValidationReport,
)


def test_research_goal_and_plan_round_trip_as_versioned_json_data():
    goal = ResearchGoal(
        goal_id="ice-dem-goal-001",
        objective="Run a bounded DEM smoke experiment and verify output integrity.",
        acceptance_criteria=("solver exits successfully", "history and trajectory are finite"),
        constraints={"device": "cpu"},
        budget={"max_runs": 1, "max_steps": 20},
    )
    plan = ExperimentPlan(
        plan_id="ice-dem-plan-001", goal_id=goal.goal_id, solver="TensorDEM",
        parameters={"nx": 7, "ny": 3, "steps": 20},
        validation_checks=("history_finite", "trajectory_finite"),
    )
    assert goal.to_dict()["schema_version"] == 1
    assert plan.to_dict()["goal_id"] == goal.goal_id


@pytest.mark.parametrize("bad", [0, -1, True])
def test_plan_rejects_invalid_run_budget(bad):
    with pytest.raises(ResearchContractError, match="max_runs"):
        ExperimentPlan("p", "g", "TensorDEM", {}, ("finite",), max_runs=bad)


def test_validation_report_cannot_pass_if_any_check_failed():
    with pytest.raises(ResearchContractError, match="cannot contain failed"):
        ValidationReport("r", "passed", ({"name": "finite", "passed": True},
                                         {"name": "checksum", "passed": False}), "bad")


def test_contracts_reject_nan_and_unknown_versions():
    with pytest.raises(ResearchContractError, match="finite JSON"):
        ExperimentPlan("p", "g", "TensorDEM", {"dt": float("nan")}, ("finite",))
    with pytest.raises(ResearchContractError, match="schema_version"):
        ResearchGoal("g", "objective", ("criterion",), schema_version=2)


def test_memory_record_requires_traceable_evidence():
    record = ResearchMemoryRecord("g", "r", "finite output", ("file:///run/history.csv",), "sweep")
    assert record.to_dict()["evidence_uris"][0].endswith("history.csv")
