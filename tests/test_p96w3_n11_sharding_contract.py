"""P96W3-N11 — cluster shard execution: determinism + fail-closed acceptance suite.

Machine-checkable criteria (all numbers written to
``artifacts/p96w3_n11/acceptance.json`` at session end — zero hand-copy):

C1  swap-compatibility: 64 parameter points, n=1/2/4/8, shards executed as
    *real subprocesses* through the ``{i}/{n}`` command template; aggregated
    results bit-identical (float.hex comparison, NaN-safe) under shard-order
    and key-order permutation; every parameter point has count == 1.
C2  computable float bound: 1-shard vs 8-shard sum-sensitive total relative
    deviation <= 1e-12 and measured error <= K*eps*n*Sum|x| with K calibrated
    by measurement and hardcoded; Kahan-compensated <= 1e-15.  Negative
    control: an ill-conditioned corpus where naive *fails* the 1e-12 bar.
C3  fail-closed: delete 3/8 artifacts (killed shards) -> raise, zero partial
    output; retry of the killed shards reproduces the reference bit-identically.
C4  conflict: two shards writing the same key -> raise.  Kill-test: a
    last-write-wins aggregator must be killed by the same assertions.
C5  manifest integrity: three tampers (git sha / segment boundary / result
    hash) on real on-disk artifacts -> all red.
C6  boundary: 100 random parameter spaces (non-linear index, float endpoints),
    mapping coverage 100/100, overlap 0, balance <= 1.
C7  regression: shard=1 path bit-identical to direct in-process execution.
"""

from __future__ import annotations

import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pytest

from tensorphysics.sharding import AggregationError, FloatAudit, ShardPlan, canonical_key
from tensorphysics.sharding_cluster import (
    ARTIFACT_SCHEMA,
    FloatContract,
    ShardChannelError,
    aggregate_from_dir,
    artifact_path,
    build_plan,
    canonical_digest,
    keys_digest,
    load_plan,
    plan_specs,
    read_artifact,
    render_shard_cmd,
    run_shard,
    shard_commands,
    synthetic_scan_energy,
    verify_artifact,
    write_artifact,
    write_plan,
)

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
ARTIFACT_DIR = REPO / "artifacts" / "p96w3_n11"
MEASUREMENTS: Dict[str, Any] = {}

CMD_TPL = (
    f"{sys.executable} -m tensorphysics.sharding_cluster run-shard "
    "--plan {plan} --i {i} --n {n} --out {out}"
)


# ---------------------------------------------------------------------------
# Fixtures / harness
# ---------------------------------------------------------------------------

def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except Exception:  # pragma: no cover - degraded environments
        return "unknown-sha"


GIT_SHA = _git_sha()
TOOL_VERSION = "tensorphysics-0.1.0"


def scan_points(n: int = 64) -> List[Dict[str, float]]:
    """64 parameter points with non-linear index and float endpoints.

    ``x`` is log-spaced over 1e-3..1e3 (float endpoints, 6 decades);
    ``y`` uses a non-linear exponent so index order != value order.
    """
    pts = []
    for i in range(n):
        t = i / (n - 1)
        pts.append({"x": 10.0 ** (6.0 * t - 3.0), "y": 1e-7 + math.pi * (t ** 1.7)})
    return pts


def point_values(points: Sequence[Dict[str, float]]) -> Dict[str, float]:
    """Sum-sensitive scalar per point (synthetic energy fixture)."""
    values = {}
    for p in points:
        ck = canonical_key(p)
        values[ck] = float(synthetic_scan_energy(ck)["energy"])
    return values


def keysets_by_shard(plan: Dict[str, Any]) -> List[List[str]]:
    return [list(s.param_keys) for s in plan_specs(plan)]


def run_scan(
    stage_dir: Path,
    n_shards: int,
    points: Sequence[Dict[str, float]],
    *,
    use_subprocess: bool = True,
) -> Dict[str, Any]:
    """Build a plan, execute every shard, aggregate fail-closed.

    ``use_subprocess=True`` launches each shard through the rendered
    ``{i}/{n}`` command (the real cluster code path); ``False`` calls
    :func:`tensorphysics.sharding_cluster.run_shard` in-process with the same
    plan/artifact contract (identical results, ~50x cheaper in CI).
    """
    stage_dir.mkdir(parents=True, exist_ok=True)
    plan = build_plan(
        points,
        n_shards,
        CMD_TPL,
        tool_version=TOOL_VERSION,
        git_sha=GIT_SHA,
        one_based=True,
    )
    plan_path = write_plan(stage_dir / "plan.json", plan)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["PYTHONDONTWRITEBYTECODE"] = "1"  # no __pycache__ races between shard processes
    t0 = time.time()
    if use_subprocess:
        for cmd in shard_commands(plan, plan_path, stage_dir / "out"):
            proc = subprocess.run(cmd, shell=True, env=env, capture_output=True, text=True,
                                  timeout=300)
            assert proc.returncode == 0, f"shard command failed: {cmd}\n{proc.stderr[-800:]}"
    else:
        for i in range(n_shards):
            run_shard(plan_path, i, stage_dir / "out")
    elapsed = time.time() - t0
    merged = aggregate_from_dir(stage_dir / "out", load_plan(plan_path))
    return {
        "plan": plan,
        "plan_path": plan_path,
        "out_dir": stage_dir / "out",
        "merged": merged,
        "elapsed_s": elapsed,
        "cmds": shard_commands(plan, plan_path, stage_dir / "out"),
    }


def bits(value: Any) -> Any:
    """Bit-exact, NaN-safe comparable form (floats -> float.hex)."""
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return float.hex(value)
    if isinstance(value, dict):
        return tuple(sorted((k, bits(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(bits(v) for v in value)
    return value


def assert_bit_identical(a: Dict[str, Any], b: Dict[str, Any], label: str = "") -> None:
    assert set(a) == set(b), f"{label}: key sets differ ({len(a)} vs {len(b)})"
    for k in a:
        assert bits(a[k]) == bits(b[k]), f"{label}: value differs at {k}: {a[k]} vs {b[k]}"


def assert_contract(
    merged: Dict[str, Any],
    expected_keys: Sequence[str],
    n_expected: int,
    contributions: Sequence[Dict[str, Any]] | None = None,
) -> None:
    """Contract check used both positively and as a kill-test for bad aggregators.

    Counts are derived from the *shard contributions* (not from the merged dict),
    which is what makes a last-write-wins aggregator detectable: LWW keeps the
    key set intact while silently double-counting one key.
    """
    expected = {canonical_key(k) for k in expected_keys}
    assert set(merged) == expected, (
        f"coverage violated: missing={len(expected - set(merged))}, "
        f"extra={len(set(merged) - expected)}"
    )
    counts: Dict[str, int] = {}
    if contributions is None:
        contributions = [merged]
    for shard_results in contributions:
        for k in shard_results:
            counts[k] = counts.get(k, 0) + 1
    assert set(counts) == expected
    assert all(c == 1 for c in counts.values()), (
        f"count-per-key != 1: {sorted((k, c) for k, c in counts.items() if c != 1)[:3]}"
    )
    assert len(merged) == n_expected


def _write_raw_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True))


def _rehash(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Recompute the adversary-visible hashes (simulates a *consistent* tamper)."""
    payload["keys_digest"] = keys_digest(payload["results"].keys())
    payload["n_written"] = len(payload["results"])
    payload["results_hash"] = canonical_digest(payload["results"])
    payload["envelope_hash"] = canonical_digest(
        {k: v for k, v in payload.items() if k != "envelope_hash"}
    )
    return payload


@pytest.fixture(scope="session", autouse=True)
def _dump_measurements(request):
    """Write the measured criteria.

    Only a fully green session may overwrite ``acceptance.json`` (the artifact the
    report and figures are generated from); a red session writes
    ``acceptance_partial.json`` instead, so a failed run can never turn into
    "evidence" for the report.
    """
    yield
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    passed = getattr(request.session, "testsfailed", 1) == 0
    name = "acceptance.json" if passed else "acceptance_partial.json"
    (ARTIFACT_DIR / name).write_text(
        json.dumps(MEASUREMENTS, indent=2, sort_keys=True, default=str)
    )


@pytest.fixture(scope="session")
def reference_scan(tmp_path_factory) -> Dict[str, Any]:
    """Reference execution: n=1 (direct/one-shard path)."""
    return run_scan(tmp_path_factory.mktemp("ref_n1"), 1, scan_points(64))


# ---------------------------------------------------------------------------
# C1 — swap-compatibility
# ---------------------------------------------------------------------------

class TestC1SwapCompatibility:
    def test_n_1_2_4_8_bit_identical(self, tmp_path, reference_scan):
        ref = reference_scan["merged"]
        per_n = {"1": {"keys": len(ref), "elapsed_s": round(reference_scan["elapsed_s"], 3)}}
        plans = {1: reference_scan["plan"]}
        for n in (2, 4, 8):
            run = run_scan(tmp_path / f"n{n}", n, scan_points(64))
            assert_bit_identical(ref, run["merged"], f"n={n}")
            per_n[str(n)] = {
                "keys": len(run["merged"]),
                "elapsed_s": round(run["elapsed_s"], 3),
                "shard_sizes": [len(s.param_keys) for s in plan_specs(run["plan"])],
            }
            plans[n] = run["plan"]
        MEASUREMENTS["C1"] = {
            "n_points": len(ref),
            "shard_counts": [1, 2, 4, 8],
            "bit_identical_all": True,
            "per_n": per_n,
            "balance": {str(n): max(len(s.param_keys) for s in plan_specs(p)) -
                        min(len(s.param_keys) for s in plan_specs(p)) for n, p in plans.items()},
        }

    def test_key_order_and_shard_order_permutation(self, reference_scan, tmp_path):
        run = run_scan(tmp_path / "perm", 8, scan_points(64), use_subprocess=False)
        out_dir = run["out_dir"]
        plan = run["plan"]

        docs = [read_artifact(artifact_path(out_dir, i, 8)) for i in range(8)]

        def agg(order: List[Dict[str, Any]]) -> Dict[str, Any]:
            # re-use the fail-closed verifier + merges in the given order
            from tensorphysics.sharding import Aggregator, Manifest

            triples = []
            for doc in order:
                i = int(doc["shard_index"])
                results = verify_artifact(doc, plan=plan, shard_index=i)
                m = Manifest(i, 8, tuple(sorted(results)), plan["tool_version"],
                             plan["git_sha"], "")
                m.result_hash = m.compute_result_hash(results)
                triples.append((i, results, m))
            all_keys = [k for _, r, _ in triples for k in r]
            return Aggregator(8, all_keys).aggregate(triples)

        baseline = agg(docs)
        rng = random.Random(7)
        perms_checked = 0
        for _ in range(24):
            perm = list(range(8))
            rng.shuffle(perm)
            shuffled = [docs[i] for i in perm]
            assert_bit_identical(baseline, agg(shuffled), f"shard order {perm}")
            perms_checked += 1
        # key-order permutation inside dicts must not change anything either
        for doc in docs:
            items = list(doc["results"].items())
            random.Random(11).shuffle(items)
            doc["results"] = dict(items)
            verify_artifact(doc, plan=plan, shard_index=int(doc["shard_index"]))
        MEASUREMENTS.setdefault("C1", {})
        MEASUREMENTS["C1"]["shard_order_permutations_checked"] = perms_checked
        MEASUREMENTS["C1"]["key_order_permutation_ok"] = True

    def test_count_exactly_one_per_point(self, reference_scan, tmp_path):
        run = run_scan(tmp_path / "count", 4, scan_points(64), use_subprocess=False)
        counts: Dict[str, int] = {}
        for i in range(4):
            doc = read_artifact(artifact_path(run["out_dir"], i, 4))
            for k in doc["results"]:
                counts[k] = counts.get(k, 0) + 1
        expected = {canonical_key(p) for p in scan_points(64)}
        assert set(counts) == expected
        assert set(counts.values()) == {1}
        assert_contract(run["merged"], expected, 64)
        MEASUREMENTS.setdefault("C1", {})
        MEASUREMENTS["C1"]["count_one_per_point"] = {"n_points": len(counts),
                                                     "max_count": max(counts.values())}


# ---------------------------------------------------------------------------
# C2 — computable float bound
# ---------------------------------------------------------------------------

class TestC2FloatBound:
    """Sum-sensitive quantities: total energy (well conditioned) and signed
    work (cancelling) — the second one makes the error bound bite."""

    @staticmethod
    def _signed(values: Dict[str, float]) -> Dict[str, float]:
        """Cancelling variant: alternate the sign, keeping the same magnitudes."""
        out = {}
        for i, k in enumerate(sorted(values)):
            out[k] = values[k] if i % 2 == 0 else -values[k]
        return out

    def test_shard_count_invariance_of_sum_sensitive_totals(self, tmp_path):
        points = scan_points(64)
        runs, keysets, merged_totals = {}, {}, {}
        for n in (1, 2, 4, 8):
            run = run_scan(tmp_path / f"f{n}", n, points, use_subprocess=False)
            runs[n] = run
            keysets[n] = keysets_by_shard(run["plan"])
            # the aggregated product itself (bit-identical by C1)
            merged_totals[n] = math.fsum(v["energy"] for v in run["merged"].values())

        ref_vals = {k: float(v["energy"]) for k, v in runs[1]["merged"].items()}
        signed_vals = self._signed(ref_vals)

        rep = FloatContract.shard_count_invariance(ref_vals, keysets, reference_n=1)
        rep_signed = FloatContract.shard_count_invariance(signed_vals, keysets, reference_n=1)

        def max_devs(rep_):
            per_n = rep_["per_n"]
            return (
                max(v["naive_rel_dev_vs_ref"] for v in per_n.values()),
                max(v["kahan_rel_dev_vs_ref"] for v in per_n.values()),
            )

        naive_max, kahan_max = max_devs(rep)
        naive_max_s, kahan_max_s = max_devs(rep_signed)

        # the aggregated product is invariant to the shard count
        ref_tot = merged_totals[1]
        denom = abs(ref_tot) if ref_tot else 1.0
        merged_dev = max(abs(t - ref_tot) / denom for t in merged_totals.values())
        assert merged_dev <= 1e-12, f"aggregated total drifts with n: {merged_dev:.3e}"

        assert naive_max <= 1e-12, f"naive rel dev {naive_max:.3e} > 1e-12"
        assert naive_max_s <= 1e-12, f"naive (signed) rel dev {naive_max_s:.3e} > 1e-12"
        assert kahan_max <= 1e-15, f"kahan rel dev {kahan_max:.3e} > 1e-15"
        assert kahan_max_s <= 1e-15, f"kahan (signed) rel dev {kahan_max_s:.3e} > 1e-15"

        for label, rep_ in (("energy", rep), ("signed", rep_signed)):
            abs_sum = rep_["sum_abs"]
            for n, v in rep_["per_n"].items():
                assert v["naive_abs_err"] <= FloatContract.bound_naive(64 + int(n), abs_sum), label
                assert v["kahan_abs_err"] <= FloatContract.bound_kahan(abs_sum), label

        MEASUREMENTS["C2"] = {
            "points": 64,
            "merged_total_rel_dev_max": merged_dev,
            "energy": {
                "sum_abs": rep["sum_abs"],
                "conditioning_ratio": rep["sum_abs"] / abs(ref_tot),
                "naive_rel_dev_max": naive_max,
                "kahan_rel_dev_max": kahan_max,
                "per_n": rep["per_n"],
            },
            "signed": {
                "sum_abs": rep_signed["sum_abs"],
                "total": rep_signed["totals"][1]["exact"],
                "conditioning_ratio": rep_signed["sum_abs"] / max(
                    abs(rep_signed["totals"][1]["exact"]), 1e-300),
                "naive_rel_dev_max": naive_max_s,
                "kahan_rel_dev_max": kahan_max_s,
                "per_n": rep_signed["per_n"],
            },
            "naive_threshold": 1e-12,
            "kahan_threshold": 1e-15,
            "K_naive": FloatContract.K_NAIVE,
            "K_kahan": FloatContract.K_KAHAN,
        }

    def test_conditioning_crossover_sweep(self):
        """Quantify the conditioning ratio R = Sum|x|/|Sum x| at which the
        1e-12 (naive) and 1e-15 (Kahan) bars are crossed.

        Construction: 64 terms of magnitude m cancelling to a residual r, so
        R = 2m/|r|; measured relative error = |sum - fsum|/|fsum|.
        """
        n = 64
        rows = []
        for R_exp in range(0, 13):
            R = 10.0 ** R_exp
            # non-representable mantissas so partial sums actually round
            base = random.Random(1000 + R_exp)
            vals = [base.uniform(0.5, 1.5) for _ in range(n - 1)]
            s_pos = math.fsum(vals)
            vals.append(-s_pos + 2.0 * s_pos / R)  # cancel down to residual ~2s/R
            exact = math.fsum(vals)
            naive = FloatAudit.naive_sum(vals)
            kahan = FloatAudit.kahan_sum(vals)
            abs_sum = float(sum(abs(v) for v in vals))
            rows.append({
                "R_target": R,
                "R_actual": abs_sum / abs(exact) if exact else float("inf"),
                "naive_rel_err": abs(naive - exact) / abs(exact),
                "kahan_rel_err": abs(kahan - exact) / abs(exact),
                "naive_within_bound": abs(naive - exact) <= FloatContract.bound_naive(n, abs_sum),
                "kahan_within_bound": abs(kahan - exact) <= FloatContract.bound_kahan(abs_sum),
            })

        def crossing(key: str, thr: float):
            for row in rows:
                if row[key] > thr:
                    return row["R_actual"]
            return None

        naive_cross = crossing("naive_rel_err", 1e-12)
        kahan_cross = crossing("kahan_rel_err", 1e-15)
        assert all(r["naive_within_bound"] for r in rows)
        assert all(r["kahan_within_bound"] for r in rows)
        # the sweep must actually exhibit both regimes (below and above the bar)
        assert any(r["naive_rel_err"] <= 1e-12 for r in rows)
        assert naive_cross is not None, "naive never crossed 1e-12 in the sweep"
        MEASUREMENTS["C2_conditioning"] = {
            "rows": rows,
            "naive_crossover_R": naive_cross,
            "kahan_crossover_R": kahan_cross,
            "n": n,
        }

    def test_K_calibration_measured_and_hardcoded(self):
        rng = random.Random(2026)
        corpora = []
        for _ in range(200):
            n = rng.choice([8, 16, 32, 64, 128])
            mag = 10.0 ** rng.uniform(-6, 6)
            vals = [rng.choice([-1.0, 1.0]) * mag * rng.random() for _ in range(n)]
            corpora.append(vals)
        calib = FloatContract.calibrate(corpora, label="calibration")

        rng2 = random.Random(4242)
        holdout = []
        for _ in range(200):
            n = rng2.choice([8, 16, 32, 64, 128])
            mag = 10.0 ** rng2.uniform(-6, 6)
            holdout.append([rng2.choice([-1.0, 1.0]) * mag * rng2.random() for _ in range(n)])
        valid = FloatContract.calibrate(holdout, label="validation")

        assert calib["naive_bound_holds"], calib
        assert valid["naive_bound_holds"], valid
        assert calib["kahan_bound_holds"] and valid["kahan_bound_holds"]
        assert FloatContract.K_NAIVE == 1.0  # hardcoded, not tuned post-hoc
        MEASUREMENTS["C2_K"] = {**calib, **{f"validation_{k}": v for k, v in valid.items()
                                           if k.startswith(("naive_max", "kahan_max", "n_"))}}

    def test_negative_control_illconditioned_naive_breaks_threshold(self):
        """Control: with catastrophic cancellation naive *fails* the 1e-12 bar."""
        # 128 terms of magnitude 1e8 cancelling to a residual of order 1
        vals = [1e8, 1.0] + [-1e8 / 127.0] * 127
        exact = math.fsum(vals)
        naive = FloatAudit.naive_sum(vals)
        kahan = FloatAudit.kahan_sum(vals)
        naive_rel = abs(naive - exact) / abs(exact)
        kahan_rel = abs(kahan - exact) / abs(exact)
        abs_sum = float(sum(abs(v) for v in vals))
        within = abs(naive - exact) <= FloatContract.bound_naive(len(vals), abs_sum)
        MEASUREMENTS["C2_negative_control"] = {
            "n": len(vals),
            "naive_rel_err": naive_rel,
            "kahan_rel_err": kahan_rel,
            "naive_within_bound": within,
            "naive_passes_1e12": naive_rel <= 1e-12,
            "kahan_passes_1e15": kahan_rel <= 1e-15,
        }
        # the naive path is genuinely unsafe for this conditioning -> the 1e-12
        # criterion has teeth (it is not satisfied by accident)
        assert naive_rel > 1e-12
        assert within  # ... while still respecting the (loose) worst-case bound


# ---------------------------------------------------------------------------
# C3 — fail-closed on killed shards
# ---------------------------------------------------------------------------

class TestC3FailClosed:
    def test_kill_three_of_eight_raises_and_retry_matches(self, tmp_path, reference_scan):
        run = run_scan(tmp_path / "kill", 8, scan_points(64), use_subprocess=False)
        out_dir, plan_path, plan = run["out_dir"], run["plan_path"], run["plan"]
        killed = [1, 4, 6]
        backups = {}
        for i in killed:
            p = artifact_path(out_dir, i, 8)
            backups[i] = p.read_text()
            p.unlink()

        with pytest.raises(ShardChannelError):
            aggregate_from_dir(out_dir, load_plan(plan_path))
        # zero partial output: the stage dir is still missing the killed shards
        present = sorted(p.name for p in out_dir.glob("shard_*.json"))
        assert len(present) == 5

        # retry only the killed shards (what the scheduler does)
        env = dict(os.environ)
        env["PYTHONPATH"] = str(SRC) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        retry_cmds = shard_commands(plan, plan_path, out_dir)
        for i in killed:
            proc = subprocess.run(retry_cmds[i], shell=True, env=env, capture_output=True,
                                  text=True, timeout=300)
            assert proc.returncode == 0, proc.stderr[-500:]
        MEASUREMENTS.setdefault("C3", {})["retry_via_command_channel"] = True

        merged = aggregate_from_dir(out_dir, load_plan(plan_path))
        assert_bit_identical(reference_scan["merged"], merged, "after retry")
        MEASUREMENTS["C3"] = {
            "shards": 8,
            "killed": killed,
            "raised_on_partial": True,
            "partial_outputs_emitted": 0,
            "artifacts_before_retry": len(present),
            "retry_bit_identical": True,
        }

    def test_random_kill_chaos_never_partial(self, tmp_path, reference_scan):
        """Kill 3/8 shard *processes* at random delays.

        Atomic write means a killed shard either left no artifact (-> raise) or a
        complete one (-> bit-identical result).  What must never happen is a
        partial/mismatched mapping being returned.
        """
        run = run_scan(tmp_path / "chaos", 8, scan_points(64), use_subprocess=False)
        plan, plan_path, out_dir = run["plan"], run["plan_path"], run["out_dir"]
        victims = [0, 3, 7]
        env = dict(os.environ)
        env["PYTHONPATH"] = str(SRC) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        cmds = shard_commands(plan, plan_path, out_dir)

        killed_before_write = 0
        for i in victims:
            artifact_path(out_dir, i, 8).unlink()
            proc = subprocess.Popen(cmds[i], shell=True, env=env,
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(0.02 + 0.03 * i)
            proc.kill()
            proc.wait(timeout=30)
            if not artifact_path(out_dir, i, 8).exists():
                killed_before_write += 1

        try:
            merged = aggregate_from_dir(out_dir, plan)
        except ShardChannelError:
            merged = None

        if merged is None:
            # fail-closed: retry exactly the missing shards, then aggregate again
            for i in victims:
                subprocess.run(cmds[i], shell=True, env=env, capture_output=True,
                               text=True, timeout=300, check=True)
            merged = aggregate_from_dir(out_dir, plan)
        assert_bit_identical(reference_scan["merged"], merged, "after chaos kill/retry")

        MEASUREMENTS.setdefault("C3", {})["chaos_kill"] = {
            "shards_killed": victims,
            "killed_before_artifact_write": killed_before_write,
            "partial_result_emitted": False,
            "recovered_bit_identical": True,
        }

    def test_half_written_file_raises(self, tmp_path):
        run = run_scan(tmp_path / "half", 4, scan_points(64), use_subprocess=False)
        p = artifact_path(run["out_dir"], 2, 4)
        raw = p.read_text()
        p.write_text(raw[: int(len(raw) * 0.6)])  # truncated mid-JSON
        with pytest.raises(ShardChannelError):
            aggregate_from_dir(run["out_dir"], run["plan"])

        p.write_text(raw)
        doc = json.loads(raw)
        doc["status"] = "running"  # complete JSON, incomplete shard
        _write_raw_json(p, _rehash(doc))
        with pytest.raises(ShardChannelError):
            aggregate_from_dir(run["out_dir"], run["plan"])

        p.write_text("")  # zero-length file (flush never happened)
        with pytest.raises(ShardChannelError):
            aggregate_from_dir(run["out_dir"], run["plan"])
        MEASUREMENTS["C3_half_written"] = {
            "truncated_raises": True, "incomplete_status_raises": True, "empty_raises": True
        }

    def test_writer_refuses_foreign_key_set(self, tmp_path):
        run = run_scan(tmp_path / "guard", 4, scan_points(64), use_subprocess=False)
        plan = run["plan"]
        spec = plan_specs(plan)[0]
        bad = {k: {"energy": 1.0, "kpi": 1.0} for k in spec.param_keys[1:]}
        with pytest.raises(ShardChannelError):
            write_artifact(run["out_dir"], plan=plan, shard_index=0, results=bad)

    def test_missing_plan_raises(self, tmp_path):
        with pytest.raises(ShardChannelError):
            load_plan(tmp_path / "nope.json")
        p = tmp_path / "plan.json"
        p.write_text("{}")
        with pytest.raises(ShardChannelError):
            load_plan(p)


# ---------------------------------------------------------------------------
# C4 — conflict detection + last-write-wins kill test
# ---------------------------------------------------------------------------

class LastWriteWinsAggregator:
    """Deliberately wrong aggregator (negative control)."""

    @staticmethod
    def aggregate(shard_results: List[Tuple[int, Dict[str, Any]]]) -> Dict[str, Any]:
        merged: Dict[str, Any] = {}
        for _, results in shard_results:
            merged.update(results)  # silently overwrites — no conflict check
        return merged


class TestC4Conflict:
    def test_duplicate_key_across_shards_raises(self, tmp_path):
        run = run_scan(tmp_path / "conf", 4, scan_points(64), use_subprocess=False)
        out_dir, plan = run["out_dir"], run["plan"]

        # a buggy shard 0 reports one of shard 1's keys (self-consistent hashes)
        victims = [s.param_keys[0] for s in plan_specs(plan)]
        p0 = artifact_path(out_dir, 0, 4)
        doc = read_artifact(p0)
        doc["results"][victims[1]] = {"energy": -1.0, "kpi": -1.0}
        _write_raw_json(p0, _rehash(doc))
        with pytest.raises(ShardChannelError):
            aggregate_from_dir(out_dir, plan)

        MEASUREMENTS["C4"] = {"duplicate_key_raises": True, "n_shards": 4}

    def test_boundary_swap_between_shards_raises(self, tmp_path):
        run = run_scan(tmp_path / "swapb", 4, scan_points(64), use_subprocess=False)
        out_dir, plan = run["out_dir"], run["plan"]
        specs = plan_specs(plan)
        p0, p1 = artifact_path(out_dir, 0, 4), artifact_path(out_dir, 1, 4)
        d0, d1 = read_artifact(p0), read_artifact(p1)
        # move one key from shard 0 to shard 1 (segment boundary violation)
        moved = specs[0].param_keys[0]
        d1["results"][moved] = d0["results"][moved]
        del d0["results"][moved]
        _write_raw_json(p0, _rehash(d0))
        _write_raw_json(p1, _rehash(d1))
        with pytest.raises(ShardChannelError):
            aggregate_from_dir(out_dir, plan)
        MEASUREMENTS["C4"]["boundary_swap_raises"] = True

    def test_kill_test_last_write_wins_must_die(self, tmp_path):
        """Negative control: LWW silently loses/overwrites -> contract assert kills it."""
        run = run_scan(tmp_path / "lww", 4, scan_points(64), use_subprocess=False)
        plan = run["plan"]
        out_dir = run["out_dir"]
        expected = [k for s in plan_specs(plan) for k in s.param_keys]

        # healthy data: LWW happens to agree, so the *conflict* case is the killer
        docs = [read_artifact(artifact_path(out_dir, i, 4)) for i in range(4)]
        lww_ok = LastWriteWinsAggregator.aggregate(
            [(int(d["shard_index"]), d["results"]) for d in docs]
        )
        assert_contract(lww_ok, expected, 64)  # agrees when there is no conflict

        # inject a duplicate key with a wrong value between two shards
        docs[1]["results"][expected[0]] = {"energy": -999.0, "kpi": -999.0}
        lww_bad = LastWriteWinsAggregator.aggregate(
            [(int(d["shard_index"]), d["results"]) for d in docs]
        )
        # LWW silently accepts the conflicting write ...
        assert lww_bad[expected[0]]["energy"] == -999.0
        # ... so the contract assertions must reject it (kill test)
        with pytest.raises(AssertionError):
            assert_contract(
                lww_bad, expected, 64,
                contributions=[d["results"] for d in docs],
            )

        MEASUREMENTS["C4"]["lww_killed_by_contract"] = True
        MEASUREMENTS["C4"]["lww_silently_overwrote"] = True


# ---------------------------------------------------------------------------
# C5 — manifest integrity (three real-file tampers)
# ---------------------------------------------------------------------------

class TestC5ManifestIntegrity:
    def test_three_tampers_detected(self, tmp_path):
        run = run_scan(tmp_path / "tamper", 4, scan_points(64), use_subprocess=False)
        out_dir, plan = run["out_dir"], run["plan"]
        target = artifact_path(out_dir, 1, 4)
        pristine = target.read_text()
        results = {}

        # (a) git sha tamper, envelope re-hashed consistently
        doc = json.loads(pristine)
        doc["git_sha"] = "0" * 40
        _write_raw_json(target, _rehash(doc))
        with pytest.raises(ShardChannelError):
            aggregate_from_dir(out_dir, plan)
        results["git_sha"] = "red"

        # (b) segment boundary tamper: keys_digest claims another shard's segment
        #     (envelope re-hashed so only the *boundary* claim is wrong)
        doc = json.loads(pristine)
        doc["keys_digest"] = plan_specs(plan)[2].keys_digest
        doc["envelope_hash"] = canonical_digest(
            {k: v for k, v in doc.items() if k != "envelope_hash"}
        )
        _write_raw_json(target, doc)
        with pytest.raises(ShardChannelError):
            aggregate_from_dir(out_dir, plan)
        results["segment_boundary"] = "red"

        # (c) result payload tamper without updating results_hash
        doc = json.loads(pristine)
        first = next(iter(doc["results"]))
        doc["results"][first] = {"energy": 12345.0, "kpi": 0.0}
        doc["envelope_hash"] = canonical_digest(
            {k: v for k, v in doc.items() if k != "envelope_hash"}
        )
        _write_raw_json(target, doc)
        with pytest.raises(ShardChannelError):
            aggregate_from_dir(out_dir, plan)
        results["result_hash"] = "red"

        # restore -> green again (the check is not simply "always red")
        target.write_text(pristine)
        merged = aggregate_from_dir(out_dir, plan)
        assert len(merged) == 64
        results["restored"] = "green"
        MEASUREMENTS["C5"] = {"n_tampers": 3, "results": results}


# ---------------------------------------------------------------------------
# C6 — boundary coverage / no overlap on 100 random spaces
# ---------------------------------------------------------------------------

class TestC6Boundary:
    @staticmethod
    def _random_space(rng: random.Random, k: int) -> List[Dict[str, Any]]:
        style = rng.choice(["log", "pow", "mixed", "endpoints"])
        pts = []
        for i in range(k):
            t = i / max(k - 1, 1)
            if style == "log":
                x = 10.0 ** rng.uniform(-9, 9)
            elif style == "pow":
                x = (i ** 1.6) * rng.uniform(0.5, 1.5)
            elif style == "mixed":
                x = rng.uniform(-1e6, 1e6)
            else:
                x = rng.choice([0.0, -0.0, 1e-9, 1e9, np.finfo(np.float64).max / 1e6])
            y = float((t ** rng.choice([0.5, 1.3, 2.7])) * rng.choice([1.0, -1.0]))
            pts.append({"x": float(x), "y": y, "i": i})
        return pts

    def test_100_random_spaces(self):
        rng = random.Random(97531)
        trials = []
        total = 0
        covered = 0
        overlaps = 0
        imbalance_max = 0
        determinism_ok = 0
        for trial in range(100):
            k = rng.choice([1, 3, 7, 16, 33, 64])
            n = rng.choice([1, 2, 3, 5, 8])
            pts = self._random_space(rng, k)
            plan = build_plan(pts, n, "run {i}/{n}", tool_version=TOOL_VERSION,
                              git_sha=GIT_SHA)
            got = [key for s in plan_specs(plan) for key in s.param_keys]
            expected = [canonical_key(p) for p in pts]
            total += 1
            if set(got) == set(expected):
                covered += 1
            if len(got) != len(set(got)):
                overlaps += 1
            sizes = [len(s.param_keys) for s in plan_specs(plan)]
            imbalance_max = max(imbalance_max, max(sizes) - min(sizes))
            plan2 = build_plan(pts, n, "run {i}/{n}", tool_version=TOOL_VERSION,
                               git_sha=GIT_SHA)
            if plan2["plan_digest"] == plan["plan_digest"]:
                determinism_ok += 1
            # every key in exactly one shard (count check)
            assert sorted(got) == sorted(expected)
            trials.append({
                "k": k,
                "n_shards": n,
                "coverage_ok": set(got) == set(expected),
                "overlap": len(got) != len(set(got)),
                "imbalance": max(sizes) - min(sizes),
                "deterministic": plan2["plan_digest"] == plan["plan_digest"],
            })

        assert covered == 100
        assert overlaps == 0
        assert imbalance_max <= 1
        assert determinism_ok == 100
        MEASUREMENTS["C6"] = {
            "spaces": total,
            "coverage": f"{covered}/{total}",
            "overlaps": overlaps,
            "max_imbalance": imbalance_max,
            "determinism": f"{determinism_ok}/{total}",
            "trials": trials,
        }


# ---------------------------------------------------------------------------
# C7 — regression: n=1 == direct in-process
# ---------------------------------------------------------------------------

class TestC7Regression:
    def test_shard1_equals_direct_execution(self, tmp_path, reference_scan):
        points = scan_points(64)
        direct = {canonical_key(p): synthetic_scan_energy(canonical_key(p)) for p in points}
        assert_bit_identical(direct, reference_scan["merged"], "n=1 vs direct")
        MEASUREMENTS["C7"] = {
            "shard1_vs_direct_bit_identical": True,
            "n_points": len(direct),
        }

    def test_render_and_plan_guardrails(self, tmp_path):
        with pytest.raises(ShardChannelError):
            render_shard_cmd("no placeholders here", 0, 1)
        with pytest.raises(ShardChannelError):
            render_shard_cmd("run --i {i} --n {n}", 5, 4)
        with pytest.raises(ShardChannelError):
            build_plan([], 2, "run {i}/{n}", tool_version=TOOL_VERSION, git_sha=GIT_SHA)
        with pytest.raises(ShardChannelError):
            build_plan(scan_points(8), 2, "run {i}/{n}", tool_version=TOOL_VERSION, git_sha="")
        MEASUREMENTS["C7"]["guardrails_ok"] = True