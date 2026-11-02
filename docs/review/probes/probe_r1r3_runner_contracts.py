import json, sys
sys.path.insert(0, "src")
from tensorphysics import physics_contracts as pc
from tensorphysics.platform_reporting import register_run_outcome
from tensorphysics.research_runtime import AgentResult, ResearchTaskStore, TaskStatus
from tensorphysics.research_runner import ResearchTaskRunner

class FakeAgent:
    name = "fake-agent"
    def execute(self, request):
        return AgentResult(outcome="completed", summary="ok", artifact_uris=("file:///h.csv",))

class PassValidator:
    def validate(self, request, result):
        return {"status": "passed", "checked": ["ct"]}

class BoomRegistry:
    def register(self, manifest):
        raise RuntimeError("boom outside OSError/ValueError hierarchy")

def p_succeeded():
    with ResearchTaskStore(":memory:") as store:
        t = store.create_task("task-p1", goal_id="g", plan={"steps": []}, max_attempts=2)
        r = ResearchTaskRunner(store, FakeAgent(), PassValidator(), platform_registry=BoomRegistry())
        raised = None
        try:
            r.run_once(t.task_id, goal={"g": 1}, workspace="/tmp")
        except Exception as exc:
            raised = f"{type(exc).__name__}: {exc}"
        final = store.get_task(t.task_id)
        ev = [(e.event_type, e.from_status.value if e.from_status else None, e.to_status.value)
              for e in store.list_events(t.task_id)]
        return {"run_once_raised": raised, "task_status_after": final.status.value,
                "events": ev, "platform_event_present": any(x[0]=="platform_registration" for x in ev)}

def p_never_raises():
    from tensorphysics.platform_registry import PlatformRegistryError
    out = {}
    for label, exc in (("ValueError", ValueError("x")), ("PlatformRegistryError", PlatformRegistryError("x")),
                       ("RuntimeError", RuntimeError("x")), ("OSError", OSError("x")), ("KeyError", KeyError("x"))):
        class R:
            def register(self, manifest, _e=exc):
                raise _e
        try:
            out[label] = register_run_outcome(run_id="r1", status="succeeded",
                                              validation_status="passed", manifest={}, registry=R())["outcome"]
        except Exception as up:
            out[label] = f"RAISED {type(up).__name__}"
    return out

def p_recover():
    with ResearchTaskStore(":memory:") as store:
        t = store.create_task("task-p2", goal_id="g", plan={}, max_attempts=1)
        store.transition(t.task_id, expected_status=TaskStatus.QUEUED, new_status=TaskStatus.CLAIMED, event_type="worker_claimed")
        store.transition(t.task_id, expected_status=TaskStatus.CLAIMED, new_status=TaskStatus.RUNNING, event_type="agent_started")
        store.recover_interrupted()
        types = [e.event_type for e in store.list_events(t.task_id)]
        return {"terminal_status": store.get_task(t.task_id).status.value, "events": types,
                "platform_event_present": "platform_registration" in types}

def p_gate():
    core = pc.get_contract("resistance.suboff_shared_core")
    out = {}
    for label in ("Holtrop & Mennen (1984) full method", "Holtrop-Mennen simplified",
                  "Holtrop 1984 wave-resistance method", "Holtrop (1984) method"):
        try:
            pc.PhysicsModelContract(key="probe.bypass", declared_method=label, inputs=core.inputs,
                                    components=core.components, reference_kernel="x.y")
            out[label] = "ACCEPTED"
        except pc.PhysicsContractError as e:
            out[label] = "REJECTED"
    return out

def p_anchor_gaps():
    out = {}
    for key, c in pc.all_contracts().items():
        out[key] = {"n_refs": len(c.references),
                    "n_admissible": len([r for r in c.references if r.admissible_for_judgment]),
                    "ref_fields": sorted(c.references[0].__dataclass_fields__) if c.references else []}
    try:
        pc.PhysicsModelContract(key="probe.no_anchor", declared_method="un-anchored method",
                                inputs=(pc.InputSpec("x", "-"),),
                                components=(pc.MethodComponent("r", "f", "p"),), reference_kernel="x.y")
        out["can_register_without_anchor"] = "YES"
    except pc.PhysicsContractError as e:
        out["can_register_without_anchor"] = f"NO ({e})"
    return out

def p_suboff():
    import math
    from tensorphysics.physics_kernels_resistance import (axisymmetric_wetted_surface,
        shared_core_resistance, ittc_1957_cf, schoenherr_1932_cf)
    L, B, cb, V, rho, nu = 4.356, 0.508, 0.542, 3.0, 998.2, 1.004e-6
    S = axisymmetric_wetted_surface(L, B, cb)
    r = shared_core_resistance(L, B, cb, V, rho, nu, S)
    c = pc.get_contract("resistance.suboff_shared_core")
    d = [x for x in c.references if x.provenance == "DERIVED"][0]
    e = [x for x in c.references if x.provenance == "EXPERIMENTAL"][0]
    vol = 0.7008
    return {"s_wet_fit": S, "re": r.reynolds, "fn": r.froude, "cf": r.cf, "cr": r.cr, "ct": r.ct,
            "derived": d.value, "derived_tol": d.tolerance_rel, "exp": e.value, "exp_tol": e.tolerance_rel,
            "rel_err_vs_derived": abs(r.ct-d.value)/d.value, "rel_err_vs_exp": abs(r.ct-e.value)/e.value,
            "cb_used": cb, "cb_vol_box": vol/(L*B*B), "cb_vol_cylbox": vol/((math.pi/4)*B*B*L),
            "cf_schoenherr": schoenherr_1932_cf(r.reynolds), "rt_n": r.rt_n}

print(json.dumps({"succeeded_path": p_succeeded(), "never_raises": p_never_raises(),
                  "recover": p_recover(), "gate": p_gate(), "anchor_gaps": p_anchor_gaps(),
                  "suboff": p_suboff()}, indent=2, default=str))
