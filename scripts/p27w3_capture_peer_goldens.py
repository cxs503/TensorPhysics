#!/usr/bin/env python3
"""Capture the *peer side* of the P27W3 cross-repo interface contracts.

Three contracts are pinned here, each with a peer implementation that lives
outside TensorPhysics:

===============  =========================================  ==========================
contract         peer side                                  pinned artifact
===============  =========================================  ==========================
``rao-v1``       ship-performance-platform                   backend/api/seakeeping_physics.py
                 ``SeakeepingPhysics`` STF strip theory      @ origin/main
``stress-psd-v1`` ship-performance-platform (P27W3-N04)      backend/physics/spectral_fatigue_core.py
                 ``spectral_fatigue_core``                   @ mission worktree / commit
``ice-bond-v1``  TensorDEM                                  src/tensordem/ice_beam_bending.py
                 ``IceBeamSpec`` lattice bond model         @ main
===============  =========================================  ==========================

It also freezes the maturity singles (``evidence_level`` upgrade records) of the
two maturity nodes N02 / N04 into ``artifacts/p27w3_contracts/maturity/`` so the
integration gate can score C5 without touching the peers again.

The script never writes into the peer repositories: sources are materialised
with ``git show <sha>:<path>`` (or read from a mission worktree for an
uncommitted deliverable) into a scratch directory, every payload and response is
hashed, and the result is a JSON golden corpus under
``artifacts/p27w3_contracts/peer_goldens/``.  The pytest suite consumes only
that corpus, so the contract tests stay hermetic.

Usage::

    python scripts/p27w3_capture_peer_goldens.py
    python scripts/p27w3_capture_peer_goldens.py --ship-root ... --dem-root ... \
        --n02-worktree ... --n04-worktree ...
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

OUT_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "p27w3_contracts"

# --------------------------------------------------------------------------- #
# canonical case definitions (shared by both sides of every contract)
# --------------------------------------------------------------------------- #

#: Canonical Wigley hull: wall-sided parabolic waterline, Cwp = Cb = 2/3.
WIGLEY = {
    "L": 100.0, "B": 10.0, "T": 5.0, "Cb": 2.0 / 3.0, "Cwp": 2.0 / 3.0,
    "rho": 1025.0, "GM": 1.0, "KG": 4.5,
}
WIGLEY["nabla"] = WIGLEY["Cb"] * WIGLEY["L"] * WIGLEY["B"] * WIGLEY["T"]
WIGLEY["displacement_t"] = WIGLEY["rho"] * WIGLEY["nabla"] / 1000.0

#: Contract frequency points (rad/s, regular waves, deep water).
CASE_OMEGA = (0.60, 1.00, 1.60)
#: Long-wave probe: kL = 0.0041 (Newman 1977: heave RAO -> 1 as kL -> 0).
OMEGA_LONG_WAVE = 0.02
#: High-frequency probe for the sectional added-mass comparison.
OMEGA_HI = 20.0
#: Wave amplitude used for the amplitude convention check.
WAVE_AMPLITUDE = 1.0

#: Spectral fatigue payload: bimodal stress PSD (wide band, exercises the
#: Dirlik moment closure) with a DNV-style single-slope S-N curve.
STRESS_PSD = {
    "freq_hz": np.linspace(0.02, 2.0, 199).tolist(),
    "f1": 0.12, "s1": 35.0, "w1": 0.045,
    "f2": 0.85, "s2": 9.0, "w2": 0.09,
}
SN = {"C": 1.0e12, "m": 3.0}   # S^m N = C, stress in MPa
DURATION_S = 3600.0

#: DEM lattice ice-beam specimen (bond strength mapping source).
DEM_BEAM = {
    "length_m": 2.0, "width_m": 0.5, "thickness_m": 0.05,
    "particle_diameter_m": 0.0125, "density_kg_m3": 917.0,
    "target_young_modulus_Pa": 5.0e9, "bond_strength_Pa": 1.5e6,
    "shear_strength_multiplier": 1.0,
}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_json(obj) -> str:
    return sha256_text(json.dumps(obj, sort_keys=True, separators=(",", ":")))


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True,
                          capture_output=True, text=True).stdout.strip()


def materialise(repo: Path, sha: str, path: str, dest: Path) -> Path:
    """``git show <sha>:<path>`` into ``dest`` (peer repo left untouched)."""
    blob = subprocess.run(["git", "-C", str(repo), "show", f"{sha}:{path}"],
                          check=True, capture_output=True).stdout
    dest.write_bytes(blob)
    return dest


def load_module(name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(name, file_path)
    if spec is None or spec.loader is None:      # pragma: no cover
        raise RuntimeError(f"cannot load {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def provenance(repo: Path, sha: str, files, extra=None) -> dict:
    return {
        "peer_repo": repo.name,
        "peer_head_sha": sha,
        "peer_commit_subject": git(repo, "log", "-1", "--format=%s", sha),
        "captured_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "files": files,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "extra": extra or {},
    }


# --------------------------------------------------------------------------- #
# contract 1: rao-v1
# --------------------------------------------------------------------------- #

def capture_rao(ship_root: Path, scratch: Path) -> dict:
    sha = git(ship_root, "rev-parse", "origin/main")
    rel = "backend/api/seakeeping_physics.py"
    src = materialise(ship_root, sha, rel, scratch / "seakeeping_physics.py")
    seakeep = load_module("p27w3_peer_seakeeping", src)

    ship = {
        "L": WIGLEY["L"], "B": WIGLEY["B"], "T": WIGLEY["T"],
        "Cb": WIGLEY["Cb"], "Cwp": WIGLEY["Cwp"],
        "displacement": WIGLEY["displacement_t"], "rho": WIGLEY["rho"],
        "GM": WIGLEY["GM"], "KG": WIGLEY["KG"], "speed": 0.0,
    }
    # Three tables, all evaluated by the peer solver:
    #   * ``coarse``   - the peer's own default 40-point log grid (today's baseline),
    #   * ``contract`` - the grid this contract recommends (log range + resonance
    #                    band refinement),
    #   * ``dense``    - 1200 points, the reference for the sampling error.
    grid_coarse = sorted(set([float(w) for w in np.geomspace(0.05, 3.0, 40)]
                             + list(CASE_OMEGA) + [OMEGA_LONG_WAVE]))
    band = [float(w) for w in np.arange(0.72, 1.36 + 1e-12, 0.02)]
    grid_contract = sorted(set([float(w) for w in np.geomspace(0.04, 3.0, 121)]
                               + band + list(CASE_OMEGA) + [OMEGA_LONG_WAVE]))
    grid_dense = [float(w) for w in np.geomspace(0.02, 3.0, 1200)]
    curves_coarse = seakeep.rao_curves(ship, heading_deg=180.0, speed_kn=0.0,
                                       freqs=grid_coarse)
    curves = seakeep.rao_curves(ship, heading_deg=180.0, speed_kn=0.0,
                                freqs=grid_contract)
    curves_dense = seakeep.rao_curves(ship, heading_deg=180.0, speed_kn=0.0,
                                      freqs=grid_dense)

    amplitudes = []
    for omega in CASE_OMEGA:
        m = seakeep.motion_amplitudes(ship, omega, 180.0, 0.0,
                                      wave_amplitude=WAVE_AMPLITUDE)
        amplitudes.append({
            "omega": omega, "heave_m": m["heave"], "pitch_deg": m["pitch"],
            "roll_deg": m["roll"], "omega_e": m["omega_e_abs"],
            "heave_rao": m["heave_rao"], "pitch_rao": m["pitch_rao"],
            "roll_rao": m["roll_rao"],
            "pitch_phase_deg": m["pitch_phase_deg"],
            "heave_phase_deg": m["heave_phase_deg"],
        })

    # off-grid probes: is a tabulated RAO able to carry the resonance?
    off_grid = [0.73, 1.13, 1.27, 1.45, 2.05]
    probes = []
    for w in off_grid:
        r = seakeep.solve_rao(ship, w, 180.0, 0.0, 1.0)
        probes.append({"omega": w, "heave_rao": r["heave_rao"],
                       "pitch_rao": r["pitch_rao"], "roll_rao": r["roll_rao"]})

    # forward-speed case: exercises the encounter-frequency convention
    speed_kn = 5.0
    speed_curves = seakeep.rao_curves(ship, heading_deg=180.0, speed_kn=speed_kn,
                                      freqs=list(CASE_OMEGA))

    # sectional 2-D coefficients at the infinite-frequency probe
    geo = seakeep.sectional_geometry(WIGLEY["L"], WIGLEY["B"], WIGLEY["T"],
                                     WIGLEY["Cb"], WIGLEY["Cwp"])
    co = seakeep.section_coefficients(OMEGA_HI, geo["b"], geo["area"],
                                      WIGLEY["T"], WIGLEY["rho"])
    sections = {
        "omega": OMEGA_HI,
        "nu": OMEGA_HI ** 2 * WIGLEY["T"] / seakeep.G,
        "x_m": [float(v) for v in geo["x"]],
        "half_beam_m": [float(v) for v in geo["b"]],
        "area_m2": [float(v) for v in geo["area"]],
        "a33_kg_per_m": [float(v) for v in co["a33"]],
        "b33_kg_per_m_s": [float(v) for v in co["b33"]],
        "f_a": [float(v) for v in co["a33"] / (WIGLEY["rho"] * geo["area"])],
        "nu_per_station": float(co["nu"]),
        "dx_m": float(geo["dx"]),
        "Aw_m2": float(geo["Aw"]),
        "nabla_m3": float(geo["nabla"]),
    }

    payload = {
        "contract": "rao-v1", "hull": "wigley-parabolic-wall-sided",
        "hull_params": {"L": WIGLEY["L"], "B": WIGLEY["B"], "T": WIGLEY["T"],
                        "Cb": WIGLEY["Cb"], "Cwp": WIGLEY["Cwp"],
                        "nabla_m3": WIGLEY["nabla"],
                        "displacement_t": WIGLEY["displacement_t"],
                        "rho": WIGLEY["rho"], "GM": WIGLEY["GM"]},
        "heading_deg": 180.0, "speed_kn": 0.0,
        "normalisation": {"heave_rao": "m/m", "pitch_rao": "rad/(k*a)",
                          "roll_rao": "rad/(k*a)", "frequency": "rad/s",
                          "phase": "deg"},
        "normalisation_notes": {
            "heave_rao": "m per unit wave amplitude a",
            "pitch_rao": "rad per unit wave slope k*a",
            "roll_rao": "rad per unit wave slope k*a",
        },
        "frequencies_rad_s": curves["frequencies"],
    }
    return {
        "payload": payload,
        "payload_sha256": sha256_json(payload),
        "response": {
            "heave_rao": curves["heave_rao"], "pitch_rao": curves["pitch_rao"],
            "roll_rao": curves["roll_rao"],
            "encounter_frequencies": curves["encounter_frequencies"],
            "heave_phase_deg": curves["heave_phase_deg"],
            "pitch_phase_deg": curves["pitch_phase_deg"],
            "motion_amplitudes": amplitudes,
            "wave_amplitude_m": WAVE_AMPLITUDE,
            "coarse_table": {"omega": curves_coarse["frequencies"],
                             "heave_rao": curves_coarse["heave_rao"],
                             "pitch_rao": curves_coarse["pitch_rao"],
                             "n_points": len(curves_coarse["frequencies"])},
            "reference_dense": {"omega": curves_dense["frequencies"],
                                "heave_rao": curves_dense["heave_rao"],
                                "pitch_rao": curves_dense["pitch_rao"]},
            "off_grid_probes": probes,
            "speed_case": {"speed_kn": speed_kn, "heading_deg": 180.0,
                           "omega": speed_curves["frequencies"],
                           "omega_encounter": speed_curves["encounter_frequencies"],
                           "heave_rao": speed_curves["heave_rao"],
                           "pitch_rao": speed_curves["pitch_rao"]},
            "sections": sections,
        },
        "provenance": provenance(ship_root, sha, {rel: sha256_text(src.read_text())},
                                 {"functions": ["rao_curves", "motion_amplitudes",
                                                "solve_rao", "section_coefficients"],
                                  "heading_deg": 180.0}),
    }


# --------------------------------------------------------------------------- #
# contract 2: stress-psd-v1
# --------------------------------------------------------------------------- #

def stress_psd_payload() -> tuple[dict, np.ndarray]:
    f = np.asarray(STRESS_PSD["freq_hz"], dtype=float)
    s = (STRESS_PSD["s1"] ** 2) * np.exp(-0.5 * ((f - STRESS_PSD["f1"])
                                                 / STRESS_PSD["w1"]) ** 2) \
        + (STRESS_PSD["s2"] ** 2) * np.exp(-0.5 * ((f - STRESS_PSD["f2"])
                                                   / STRESS_PSD["w2"]) ** 2)
    payload = {
        "contract": "stress-psd-v1", "source": "TensorFEM hot-spot stress PSD",
        "freq_hz": [float(v) for v in f],
        "sxx_unit2_per_hz": [float(v) for v in s],
        "sn": {"C": SN["C"], "m": SN["m"], "convention": "S^m N = C, S in MPa"},
        "duration_s": DURATION_S,
        "cycle_count_convention": {
            "dirlik": "n_T = nu_p * duration_s, nu_p = (1/2pi) sqrt(m4/m2)",
            "rayleigh": "n_T = nu_0 * duration_s, nu_0 = (1/2pi) sqrt(m2/m0)",
            "moments": "m_n = int omega^n S_omega domega (omega axis, rad/s)",
        },
        "notes": "bimodal Gaussian: f1=0.12 Hz sigma1=35 MPa, f2=0.85 Hz sigma2=9 MPa",
    }
    return payload, s


def capture_spectral(ship_root: Path, n04_worktree: Path | None, scratch: Path) -> dict:
    payload, s = stress_psd_payload()
    f = np.asarray(payload["freq_hz"], dtype=float)
    omega = 2.0 * np.pi * f
    s_omega = np.asarray(s, dtype=float) / (2.0 * np.pi)

    provenance_files: dict[str, str] = {}
    primary_source: dict = {}
    core = None
    if n04_worktree is not None and (n04_worktree / "backend/physics/spectral_fatigue_core.py").exists():
        core_path = n04_worktree / "backend/physics/spectral_fatigue_core.py"
        core = load_module("p27w3_peer_n04_spectral", core_path)
        provenance_files["backend/physics/spectral_fatigue_core.py"] = sha256_text(
            core_path.read_text())
        try:
            sha = git(n04_worktree, "rev-parse", "HEAD")
        except subprocess.SubprocessError:                     # pragma: no cover
            sha = "0" * 40
        primary_source = {
            "kind": "mission_worktree",
            "mission": "P27W3-N04 (6730)",
            "worktree": str(n04_worktree),
            "file": "backend/physics/spectral_fatigue_core.py",
            "committed": False,
            "note": "deliverable of mission 6730, captured from its worktree; "
                    "re-run the capture after the branch is merged to re-pin to a commit",
        }
    if core is None:                                           # pragma: no cover
        raise SystemExit("spectral_fatigue_core.py not found; pass --n04-worktree")

    mom = core.spectral_moments(omega, s_omega)
    m0, m1, m2, m4 = mom["m0"], mom["m1"], mom["m2"], mom["m4"]
    nu_p = math.sqrt(m4 / m2) / (2.0 * math.pi)
    nu_0 = math.sqrt(m2 / m0) / (2.0 * math.pi)
    n_t_dirlik = nu_p * DURATION_S
    n_t_ray = nu_0 * DURATION_S
    dirlik = float(core.dirlik_damage(m0, m1, m2, m4, SN["m"], SN["C"], n_t_dirlik))
    rayleigh = float(core.rayleigh_damage(m0, SN["m"], SN["C"], n_t_ray))

    # secondary witness: the P27W2 module (agent/m6659), committed on a branch
    secondary = {"available": False}
    sec_sha = None
    try:
        sec_sha = git(ship_root, "rev-parse", "agent/m6659")
        pkg_dir = scratch / "spectral_fatigue"
        pkg_dir.mkdir(parents=True, exist_ok=True)
        listing = git(ship_root, "ls-tree", "-r", "--name-only", sec_sha,
                      "backend/modules/spectral_fatigue").split()
        sec_files = {}
        for rel in listing:
            if rel.endswith(".py"):
                dest = pkg_dir / Path(rel).name
                materialise(ship_root, sec_sha, rel, dest)
                sec_files[rel] = sha256_text(dest.read_text())
        sys.path.insert(0, str(scratch))
        sf = load_module("spectral_fatigue", pkg_dir / "__init__.py")
        spectrum = sf.StressSpectrum.from_f(f, s)
        bw = sf.bandwidth_from_moments(spectrum.moments(4))
        sn = sf.SNCurve(m1=SN["m"], a1=SN["C"])
        secondary = {
            "available": True,
            "commit": sec_sha,
            "files": sec_files,
            "moments_omega": {f"m{i}": float(v) for i, v in enumerate(spectrum.moments(4))},
            "dirlik_damage": float(sf.dirlik_damage(bw, sn, DURATION_S)),
            "narrowband_damage": float(sf.narrowband_damage(bw, sn, DURATION_S)),
            "bandwidth": bw.as_dict(),
        }
    except (subprocess.SubprocessError, KeyError) as exc:      # pragma: no cover
        secondary = {"available": False, "error": str(exc)}

    s_probe = np.linspace(0.0, 12.0 * math.sqrt(m0), 401)
    pdf_probe = np.asarray(core.dirlik_pdf(s_probe, m0, m1, m2, m4), dtype=float)
    response = {
        "moments_omega": {"m0": m0, "m1": m1, "m2": m2, "m4": m4},
        "dirlik_parameters": {k: float(v) for k, v in
                              core.dirlik_parameters(m0, m1, m2, m4).items()},
        "dirlik_pdf_probe": {"S_MPa": [float(v) for v in s_probe],
                             "pdf": [float(v) for v in pdf_probe],
                             "integral": float(np.trapezoid(pdf_probe, s_probe))},
        "cycle_rates_hz": {"nu_p": nu_p, "nu_0": nu_0},
        "n_cycles": {"dirlik": n_t_dirlik, "rayleigh": n_t_ray},
        "dirlik_damage": dirlik,
        "narrowband_damage": rayleigh,
        "duration_s": DURATION_S,
        "sn": {"m1": SN["m"], "a1": SN["C"]},
        "spectrum": {"m0_MPa2": m0, "rms_MPa": math.sqrt(m0)},
        "secondary_witness_p27w2": secondary,
    }
    prov = {
        "peer_repo": ship_root.name,
        "peer_head_sha": sec_sha or "0" * 40,
        "peer_commit_subject": "P27W3-N04 deliverable (mission worktree, uncommitted) "
                               "+ P27W2 reference module @ agent/m6659",
        "captured_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "files": provenance_files,
        "extra": {"primary": primary_source,
                  "secondary": "backend/modules/spectral_fatigue @ agent/m6659 "
                               "(P27W2-N04, committed reference implementation)"},
    }
    return {"payload": payload, "payload_sha256": sha256_json(payload),
            "response": response, "provenance": prov}


# --------------------------------------------------------------------------- #
# contract 3: ice-bond-v1
# --------------------------------------------------------------------------- #

def capture_ice_bond(dem_root: Path) -> dict:
    sha = git(dem_root, "rev-parse", "HEAD")
    sys.path.insert(0, str(dem_root / "src"))
    from tensordem import ice_beam_bending as ibb  # noqa: E402

    spec = ibb.IceBeamSpec(**DEM_BEAM)
    props = ibb.lattice_elastic_properties(spec)
    theory = ibb.beam_theory(spec).as_dict()
    payload = {"contract": "ice-bond-v1",
               "dem_spec": dict(spec.as_dict() if hasattr(spec, "as_dict")
                                else spec.__dict__)}
    return {
        "payload": payload,
        "payload_sha256": sha256_json(payload),
        "response": {
            "lattice": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                        for k, v in props.items()},
            "beam_theory": theory,
        },
        "provenance": provenance(
            dem_root, sha,
            {"src/tensordem/ice_beam_bending.py":
             sha256_text((dem_root / "src/tensordem/ice_beam_bending.py").read_text())},
            {"functions": ["IceBeamSpec", "lattice_elastic_properties", "beam_theory"]}),
    }


# --------------------------------------------------------------------------- #
# maturity singles (C5 evidence)
# --------------------------------------------------------------------------- #

MATURITY_SOURCES = {
    "P27W3-N02": {
        "mission": 6728,
        "worktree": "/root/tp-agent/work/m6728",
        "candidates": ["docs/*rao*verification*.json", "docs/*seakeeping*report*.json",
                       "artifacts/*/maturity*.json", "docs/*verification*.json"],
    },
    "P27W3-N04": {
        "mission": 6730,
        "worktree": "/root/tp-agent/work/m6730",
        "candidates": ["docs/fatigue_verification_report.json"],
    },
}


def capture_maturity(out_dir: Path, overrides: dict[str, str] | None = None) -> dict:
    """Freeze the two maturity singles into the golden corpus."""
    records = {}
    for ext, src in MATURITY_SOURCES.items():
        worktree = Path((overrides or {}).get(ext, src["worktree"]))
        found = []
        for pattern in src["candidates"]:
            for path in sorted(worktree.glob(pattern)):
                try:
                    blob = json.loads(path.read_text())
                except json.JSONDecodeError:                   # pragma: no cover
                    continue
                if "evidence_level" not in blob:
                    continue
                found.append({
                    "path": str(path.relative_to(worktree)),
                    "sha256": sha256_text(path.read_text()),
                    "evidence_level": blob.get("evidence_level"),
                    "mission_ref": blob.get("mission_id"),
                    "title": blob.get("title"),
                    "n_fields": _count_fields(blob),
                    "fields": sorted(_field_names(blob)),
                    "criteria_pass": _criteria_pass(blob),
                })
        records[ext] = {
            "mission_id": src["mission"],
            "worktree": str(worktree),
            "checked_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "required_fields": 3 if ext == "P27W3-N02" else 5,
            "records": found,
            "n_records": len(found),
            "n_fields": max((r["n_fields"] for r in found), default=0),
        }
        records[ext]["complete"] = records[ext]["n_fields"] >= records[ext]["required_fields"]
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "maturity_records.json"
    path.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")
    return records


def _field_names(blob) -> set[str]:
    """Structured evidence fields of a maturity single (top level + criteria)."""
    names = {k for k in blob if k in {"evidence_level", "mission_id", "title",
                                      "date", "all_pass", "artifacts"}}
    crit = blob.get("criteria")
    if isinstance(crit, dict):
        names |= {f"criteria.{k}" for k in crit}
    return names


def _count_fields(blob) -> int:
    return len(_field_names(blob))


def _criteria_pass(blob) -> dict:
    crit = blob.get("criteria")
    if not isinstance(crit, dict):
        return {}
    return {k: bool(v.get("pass")) for k, v in crit.items() if isinstance(v, dict)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ship-root", default="/root/tensor-ecosystem/ship-performance-platform")
    ap.add_argument("--dem-root", default="/root/tensor-ecosystem/TensorDEM")
    ap.add_argument("--n04-worktree", default="/root/tp-agent/work/m6730")
    ap.add_argument("--n02-worktree", default="/root/tp-agent/work/m6728")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()
    out = Path(args.out_dir)
    gold = out / "peer_goldens"
    gold.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        scratch = Path(tmp)
        rao = capture_rao(Path(args.ship_root), scratch)
        psd = capture_spectral(Path(args.ship_root), Path(args.n04_worktree), scratch)
    bond = capture_ice_bond(Path(args.dem_root))
    maturity = capture_maturity(out / "maturity",
                                {"P27W3-N02": args.n02_worktree,
                                 "P27W3-N04": args.n04_worktree})

    for name, blob in (("rao_wigley_v1.json", rao),
                       ("stress_psd_v1.json", psd),
                       ("ice_bond_v1.json", bond)):
        path = gold / name
        path.write_text(json.dumps(blob, indent=2, sort_keys=True) + "\n")
        print(f"wrote {path}  sha256={sha256_text(path.read_text())}")
    for ext, rec in maturity.items():
        print(f"maturity {ext}: {rec['n_records']} record(s), "
              f"max {rec['n_fields']} field(s), complete={rec['complete']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())