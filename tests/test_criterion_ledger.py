"""Tests for the criterion falsifiability ledger (P96-N9).

Covers the machine-checkable acceptance criteria that are verifiable inside
this repo (the PAPER-TRACE cluster runs C1/C2/C4/C7 are recorded in
``artifacts/criterion_ledger_m5618/`` and the campaign report):

* C3 -- >= 4 gates enrolled, >= 90% of criteria instantiable;
* C4 -- M1 kill curve non-decreasing in eps (synthetic gate + KPI check);
* C5 -- tautological criterion: kill rate 0/5, marked UNFALSIFIABLE;
* C6 -- PR gate: new criterion without a killer test is rejected;
* C8 -- zero domain semantics in the package logic.

All fixtures are synthetic (no domain vocabulary): a "gate" here is a
generic numeric-band checker, a "criterion" a generic check function.
"""
from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorphysics.campaign.criterion_ledger import (  # noqa: E402
    Criterion,
    GateSpec,
    Ledger,
    aggregate,
    build_default_ledger,
    check_new_criterion,
    kill_rate,
    ledger_kpi,
    monotonic_kill_curve,
    probe_instantiable,
    run_gate,
    shard_slice,
    weakest_mutants,
)
from tensorphysics.campaign.criterion_ledger import mutants as M  # noqa: E402

# ---------------------------------------------------------------------------
# synthetic gate fixtures (synthetic names on purpose: zero domain semantics)
# ---------------------------------------------------------------------------

TAUT_GATE = textwrap.dedent("""
    def check(value, reference):
        # tautology: the reference is derived from the value itself
        derived = value
        diff = value - derived
        return abs(diff) <= abs(1e-9) and diff <= 1e-9
""")

TAUT_TEST = textwrap.dedent("""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from synth_taut_gate import check

    def test_tautology_passes():
        assert check(3.14, 3.14) is True
""")

REAL_GATE = textwrap.dedent("""
    TOL_REL = 0.01

    def check(value, reference):
        return abs(value - reference) <= TOL_REL * abs(reference)
""")

KILLER_TEST = textwrap.dedent("""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from synth_real_gate import check

    def test_killer():
        assert check(1.0, 1.0) is True
        assert check(1.05, 1.0) is False
""")

TAUT_KILLER_TEST = textwrap.dedent("""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from synth_real_gate import check

    def test_taut_killer():
        assert isinstance(check(1.0, 1.0), bool)
""")

BAND_GATE = textwrap.dedent("""
    TOL = 0.02

    def check(value, reference):
        return abs(value - reference) <= TOL * abs(reference)
""")

BAND_TEST = textwrap.dedent("""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from synth_band_gate import check

    def test_band():
        assert check(1.0, 1.0) is True
        assert check(1.01, 1.0) is True
        assert check(1.022, 1.0) is False
""")


def _make_gate_tree(tmp_path: Path, gate_src: str, gate_name: str,
                    test_src: str) -> tuple[Path, GateSpec]:
    """Lay out module + test under a common root; return (root, GateSpec)."""
    stem = Path(gate_name).stem
    root = tmp_path / "synth"
    root.mkdir(parents=True, exist_ok=True)
    (root / gate_name).write_text(gate_src, encoding="utf-8")
    (root / "tests").mkdir(exist_ok=True)
    (root / "tests" / f"test_{stem}.py").write_text(test_src, encoding="utf-8")
    spec = GateSpec(
        gate_id="SYNTH-" + stem.upper(),
        title="synthetic gate",
        module=Path(gate_name),
        tests=(Path("tests") / f"test_{stem}.py",),
        mutant_classes=(M.M1, M.M2, M.M3, M.M4, M.M5),
    )
    return root, spec


# ---------------------------------------------------------------------------
# unit: sharding / aggregation / KPI
# ---------------------------------------------------------------------------

def test_shard_slice_disjoint_complete():
    items = list(range(17))
    seen = []
    for i in range(5):
        seen.extend(shard_slice(items, i, 5))
    assert seen == items


def test_shard_slice_rejects_bad_ids():
    with pytest.raises(ValueError):
        shard_slice([1, 2, 3], 5, 2)
    with pytest.raises(ValueError):
        shard_slice([1, 2, 3], 0, 0)


def test_aggregate_merges_shards(tmp_path):
    for i, (tot, kil) in enumerate(((3, 1), (4, 2))):
        d = {"gate": "G", "shard_id": i, "n_shards": 2, "baseline_green": True,
             "total": tot, "killed": kil,
             "mutants": [{"name": f"m{i}_{j}", "op": "M1_numeric_perturb",
                          "line": j, "killed": j < kil, "snippet": "",
                          "patch": ""} for j in range(tot)]}
        (tmp_path / f"shard_{i}.json").write_text(json.dumps(d))
    res = aggregate(tmp_path)
    assert res["total"] == 7 and res["killed"] == 3
    assert res["kill_rate"] == pytest.approx(3 / 7, abs=1e-4)
    assert len(res["survivors"]) == 4


def test_kpi_falsifiability_marks():
    assert kill_rate({"total": 5, "killed": 0}) == 0.0
    from tensorphysics.campaign.criterion_ledger.kpi import falsifiability
    assert falsifiability({"total": 5, "killed": 0})["mark"] == "UNFALSIFIABLE"
    assert falsifiability({"total": 5, "killed": 1})["mark"] == "FALSIFIABLE"


def test_monotonic_kill_curve_kpi():
    r = monotonic_kill_curve({0.01: 2, 0.03: 3, 0.15: 5})
    assert r["monotonic"] is True
    r2 = monotonic_kill_curve({0.01: 4, 0.03: 3, 0.15: 5})
    assert r2["monotonic"] is False


def test_weakest_mutants_orders_survivors():
    res = {"mutants": [
        {"name": "a", "op": "M2_unit_mismatch", "line": 10, "killed": False},
        {"name": "b", "op": "M1_numeric_perturb", "line": 20, "killed": False},
        {"name": "c", "op": "M1_numeric_perturb", "line": 5, "killed": True},
    ]}
    w = weakest_mutants(res, n=10)
    assert [x["name"] for x in w] == ["b", "a"]


# ---------------------------------------------------------------------------
# unit: M1..M5 rewrites
# ---------------------------------------------------------------------------

def test_m1_m4_roundtrip_on_band_gate():
    src = BAND_GATE
    m1 = M.enumerate_m1(src)
    assert len(m1) == 3 * 1  # one numeric site (TOL), three eps levels
    out = M.apply_mutant(src, m1[0])
    assert "0.0202" in out or "2.02e-05" in out or "0.020200000000000002" in out
    m4 = M.enumerate_m4(src)
    out4 = M.apply_mutant(src, m4[0])
    assert "-0.02" in out4


def test_m5_selfref_rewrite():
    src = "def f(x):\n    return x <= 0.5\n"
    m5 = M.enumerate_m5(src)
    assert len(m5) == 1
    out = M.apply_mutant(src, m5[0])
    assert "x <= x" in out


def test_m5_literal_left_side():
    src = "def f(x):\n    return 0.5 <= x\n"
    m5 = M.enumerate_m5(src)
    assert len(m5) == 1
    out = M.apply_mutant(src, m5[0])
    assert "x <= x" in out


# ---------------------------------------------------------------------------
# C3 -- ledger: >= 4 gates, >= 90% criteria instantiable
# ---------------------------------------------------------------------------

def test_c3_ledger_gates_and_instantiability():
    led = build_default_ledger(ROOT, tp_root=None)
    s = led.summary()
    assert s["gates"] >= 4, f"need >= 4 gates, got {s['gates']}"
    # in-repo gates: all criteria instantiable (the upstream gate's
    # criteria are WITHHELD without its tree -- honest, not a pass)
    in_repo = {k: v for k, v in s["per_criterion"].items()
               if not k.startswith("PAPER-TRACE/")}
    assert len(in_repo) >= 4 * 3  # >= 4 gates with criteria
    assert all(v["instantiable"] for v in in_repo.values()), in_repo
    # with the upstream tree provided, the full ledger is 100%
    import os
    tp = os.environ.get("CRITERION_LEDGER_TP_ROOT")
    if tp and Path(tp).is_dir():
        led2 = build_default_ledger(ROOT, tp_root=Path(tp))
        s2 = led2.summary()
        assert s2["gates"] >= 5
        assert s2["instantiability_rate"] == 1.0, s2


# ---------------------------------------------------------------------------
# C5 -- negative control: tautological criterion, 0/5, UNFALSIFIABLE
# ---------------------------------------------------------------------------

def test_c5_tautology_kill_rate_zero_of_five(tmp_path):
    root, spec = _make_gate_tree(tmp_path, TAUT_GATE, "synth_taut_gate.py",
                                TAUT_TEST)
    res = run_gate(spec, root, jobs=2, timeout=120, per_class=1)
    assert res["total"] == 5, res  # exactly one mutant per class
    assert res["killed"] == 0, res["mutants"]
    assert kill_rate(res) == 0.0
    from tensorphysics.campaign.criterion_ledger.kpi import falsifiability
    assert falsifiability(res)["mark"] == "UNFALSIFIABLE"


# ---------------------------------------------------------------------------
# C4 -- M1 monotonic kill curve (synthetic band gate)
# ---------------------------------------------------------------------------

def _m1_killed_count(root, spec, eps, tmp_path):
    res = run_gate(spec, root, jobs=2, timeout=120, classes=(M.M1,),
                    m1_eps=eps, out_dir=tmp_path / f"eps_{eps}")
    return res["killed"]


def test_c4_m1_curve_non_decreasing(tmp_path):
    root, spec = _make_gate_tree(tmp_path, BAND_GATE, "synth_band_gate.py",
                                BAND_TEST)
    k1 = _m1_killed_count(root, spec, 0.01, tmp_path / "a")
    k3 = _m1_killed_count(root, spec, 0.03, tmp_path / "b")
    k15 = _m1_killed_count(root, spec, 0.15, tmp_path / "c")
    curve = monotonic_kill_curve({0.01: k1, 0.03: k3, 0.15: k15})
    assert curve["monotonic"] is True, (k1, k3, k15)
    assert k15 > k1, (k1, k3, k15)  # the curve must actually move


# ---------------------------------------------------------------------------
# C6 -- PR gate: no killer test -> reject
# ---------------------------------------------------------------------------

def test_c6_new_criterion_with_killer_accepted(tmp_path):
    root = tmp_path / "pr"
    (root / "tests").mkdir(parents=True)
    killer = root / "tests" / "test_killer.py"
    killer.write_text(KILLER_TEST, encoding="utf-8")
    r = check_new_criterion(REAL_GATE, "synth_real_gate.py", [killer], root,
                           classes=(M.M1, M.M2, M.M3, M.M4, M.M5), jobs=2)
    assert r["accepted"] is True, r
    assert r["killed"] >= 1


def test_c6_new_criterion_without_killer_rejected(tmp_path):
    root = tmp_path / "pr"
    root.mkdir(parents=True)
    r = check_new_criterion(REAL_GATE, "synth_real_gate.py", [], root,
                           classes=(M.M1, M.M2, M.M3, M.M4, M.M5))
    assert r["accepted"] is False
    assert "no killer test" in r["reason"]


def test_c6_tautological_killer_rejected(tmp_path):
    root = tmp_path / "pr"
    (root / "tests").mkdir(parents=True)
    taut = root / "tests" / "test_taut.py"
    taut.write_text(TAUT_KILLER_TEST, encoding="utf-8")
    r = check_new_criterion(REAL_GATE, "synth_real_gate.py", [taut], root,
                           classes=(M.M1, M.M2, M.M3, M.M4, M.M5), jobs=2)
    assert r["accepted"] is False
    assert r["killed"] == 0


# ---------------------------------------------------------------------------
# C8 -- zero domain semantics in the package logic
# ---------------------------------------------------------------------------

#: domain vocabulary that must NOT appear in the ledger package's logic.
#: (fixture data for the scan, not logic -- same pattern as the P34 gate's
#: declarative lexicon table.)
DOMAIN_TOKENS = (
    "阻力", "推进", "耐波", "操纵", "稳性", "砰击", "波浪弯矩", "螺旋桨",
    "hull", "propeller", "slamming", "springing", "whipping", "seakeeping",
    "resistance", "roll damping", "added resistance", "ice", "ship",
    "manuscript", "paper",
)


def test_c8_zero_domain_semantics_in_package():
    pkg = ROOT / "src" / "tensorphysics" / "campaign" / "criterion_ledger"
    # gates.py is the declarative registry (data): it legitimately names
    # gate ids / upstream paths -- same exemption as the P34 spec table.
    data_modules = {"gates.py"}
    import re
    offenders = []
    for f in sorted(pkg.glob("*.py")):
        if f.name in data_modules:
            continue
        text = f.read_text(encoding="utf-8").lower()
        for tok in DOMAIN_TOKENS:
            if re.search(r"\b" + re.escape(tok.lower()) + r"\b", text):
                offenders.append(f"{f.name}:{tok}")
    assert not offenders, offenders


# ---------------------------------------------------------------------------
# adapter unit
# ---------------------------------------------------------------------------

def test_probe_instantiable_rejects_non_callable():
    c = Criterion(id="X", gate="G", callable=None,
                   expected_verdict="PASS", evidence_ref="e")
    ok, reason = probe_instantiable(c)
    assert ok is False and "not callable" in reason


def test_ledger_rejects_unknown_gate():
    led = Ledger()
    with pytest.raises(KeyError):
        led.register_criterion(Criterion(id="C1", gate="NOPE",
                                         callable=lambda: None,
                                         expected_verdict="PASS",
                                         evidence_ref="e"))
