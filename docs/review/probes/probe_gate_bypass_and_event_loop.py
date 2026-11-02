import sys, json
sys.path.insert(0, "src")
from tensorphysics import physics_contracts as pc
from tensorphysics.research_runtime import AgentResult, ResearchTaskStore, TaskStatus
from tensorphysics.research_runner import ResearchTaskRunner

out = {}
# 1) untyped references: does registration validate element types?
try:
    c = pc.PhysicsModelContract(
        key="probe.dictref", declared_method="un-anchored",
        inputs=(pc.InputSpec("x", "-"),),
        components=(pc.MethodComponent("r", "f", "p"),),
        reference_kernel="x.y",
        references=({"case_id": "X", "quantity": "ct", "value": 0.99,
                     "provenance": "WITHHELD", "admissible_for_judgment": True},))
    out["dict_reference_accepted"] = True
    out["dict_ref_provenance"] = c.references[0]["provenance"]
except Exception as e:
    out["dict_reference_accepted"] = f"{type(e).__name__}: {e}"

# 2) empty-string provenance / tolerance via dict bypass
try:
    c2 = pc.PhysicsModelContract(key="probe.dictref2", declared_method="x",
        inputs=(pc.InputSpec("x", "-"),), components=(pc.MethodComponent("r", "f", "p"),),
        reference_kernel="x.y",
        references=(type("R", (), {"case_id": "X", "provenance": "WITHHELD",
                                   "admissible_for_judgment": True, "value": 1.0})(),))
    out["duck_reference_accepted"] = True
except Exception as e:
    out["duck_reference_accepted"] = f"{type(e).__name__}: {e}"

# 3) alias collision with a contract key (shadowing)
try:
    pc.register(pc.PhysicsModelContract(key="probe.alias", declared_method="x",
        inputs=(pc.InputSpec("x", "-"),), components=(pc.MethodComponent("r", "f", "p"),),
        reference_kernel="x.y", aliases=("friction.ittc1957",)))
    out["alias_shadows_key"] = pc.get_contract("friction.ittc1957").key
except Exception as e:
    out["alias_shadows_key"] = f"{type(e).__name__}: {e}"

# 4) platform_registration event status pair
class A:
    name = "a"
    def execute(self, r): return AgentResult(outcome="completed", summary="ok")
class V:
    def validate(self, r, s): return {"status": "passed"}
class R:
    def register(self, m): return {"run_id": m["run_id"]}
with ResearchTaskStore(":memory:") as store:
    t = store.create_task("task-e1", goal_id="g", plan={})
    ResearchTaskRunner(store, A(), V(), platform_registry=R()).run_once(
        "task-e1", goal={"g": 1}, workspace="/tmp")
    ev = [(e.event_type, e.from_status.value if e.from_status else None, e.to_status.value)
          for e in store.list_events("task-e1")]
    out["events"] = ev
    out["terminal_pair"] = [x for x in ev if x[0] == "platform_registration"]
# 5) terminal CANCELLED written outside the runner: any platform record?
with ResearchTaskStore(":memory:") as store:
    t = store.create_task("task-c1", goal_id="g", plan={})
    store.transition("task-c1", expected_status=TaskStatus.QUEUED,
                     new_status=TaskStatus.CANCELLED, event_type="unsafe_workspace_rejected")
    types = [e.event_type for e in store.list_events("task-c1")]
    out["cancelled_terminal"] = {"status": store.get_task("task-c1").status.value,
                                 "events": types,
                                 "platform_event_present": "platform_registration" in types}
print(json.dumps(out, indent=2, default=str))
