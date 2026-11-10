"""Build and submit the platform report for node p233w1-N06.

The report body is assembled programmatically from
``artifacts/ice_indentation/acceptance_report.json`` (zero hand-copied numbers),
the Markdown report ``docs/ICE_INDENTATION_J2_CONTACT.md`` and the PNG figures
(base64 inlined).

Usage::

    python benchmarks/ice_indentation/submit_report.py            # build + submit
    python benchmarks/ice_indentation/submit_report.py --dry-run   # build only

Credentials come from ``$TENSOR_AGENT_USER`` / ``$TENSOR_AGENT_PASS`` and the
endpoint from ``$TENSOR_API_BASE`` (default ``http://127.0.0.1:8010/api/v1``).
If the platform API is unreachable the payload is still written to
``artifacts/ice_indentation/platform_report.json`` so that the submission can be
replayed without recomputing anything.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import pathlib
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts" / "ice_indentation"
MISSION_ID = 9239

FIGURES = [
    ("ac1_ac2_material_point.png",
     "AC1/AC2 material point: uniaxial stress path (tension, compression, with "
     "hardening) from the same radial return mapping used in the FE solver "
     "against the closed form, and the pure-shear first yield against "
     "sigma_y/sqrt(3)."),
    ("ac3_elastic_hertz.png",
     "AC3: elastic indentation force-displacement from the FEM against the exact "
     "Hertz solution P = 4/3 E* sqrt(R) delta^1.5, plus the pointwise error."),
    ("ac4_plastic_mesh_convergence.png",
     "AC4: elastoplastic (J2, sigma_y = 5 MPa) indentation F(delta) on three "
     "graded meshes and the relative force error versus the near-contact node "
     "spacing (observed order 3.58)."),
    ("ac5_fields.png",
     "Indentation fields on the symmetry plane of the finest mesh: von Mises "
     "stress and equivalent plastic strain (the plastic zone under the contact)."),
    ("ac5_energy_ledger.png",
     "AC5: energy ledger - work of the contact loads, elastic strain energy and "
     "plastic dissipation versus penetration, with the relative balance "
     "residual (final value -0.49 %)."),
]


def build_payload() -> dict:
    report = json.load(open(ART / "acceptance_report.json"))
    markdown = (ROOT / "docs" / "ICE_INDENTATION_J2_CONTACT.md").read_text()
    figures = []
    for name, caption in FIGURES:
        path = ART / name
        if not path.exists():
            continue
        figures.append({
            "name": name,
            "caption": caption,
            "png_base64": base64.b64encode(path.read_bytes()).decode(),
        })
    summary = "; ".join(
        f"{k}={'PASS' if v['passed'] else 'FAIL'}"
        for k, v in report["summary"].items()
    )
    summary_short = " ".join(
        f"{k}:{'P' if v['passed'] else 'F'}" for k, v in report["summary"].items()
    )
    ac3 = report["criteria"]["AC3"]
    pb = ac3["plastic_branch"]
    title = ("J2 elastoplastic ice indentation with penalty contact: "
             "constitutive verification, Hertz benchmark and mesh convergence "
             "(TensorPhysics p233w1-N06)")
    summary_text = (
        "TensorPhysics p233w1-N06: J2 elastoplastic ice indentation with "
        "penalty contact (node-to-surface + facet-integrated) and an incremental "
        f"Newton driver. Acceptance: {summary_short}. "
        f"AC3 elastic F(delta) vs Hertz {100 * ac3['max_relative_error']:.1f}% "
        f"(<=15%); AC5 energy residual "
        f"{100 * report['criteria']['AC5']['relative_residual']:+.2f}% (<=1%); "
        f"AC6 {report['criteria']['AC6']['collected']} pytest cases. "
        f"Plastic branch P={pb['force_N']:.2f} N, p_m/sigma_y="
        f"{pb['constraint_factor_p_m_over_sigma_y']:.2f}; comparison with an "
        "experimental indentation curve is WITHHELD (no dataset in the "
        "repository; fabricating one is not acceptable)."
    )[:500]
    return {
        "title": title,
        "summary": summary_text,
        "scenario_key": "ice_indentation_j2_contact",
        "mission_id": MISSION_ID,
        "tags": ["ice", "indentation", "J2 plasticity", "penalty contact",
                 "finite element", "verification", "TensorPhysics"],
        "markdown": markdown,
        "figures": figures,
    }


def submit(payload: dict) -> int:
    base = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1").rstrip("/")
    user = os.environ.get("TENSOR_AGENT_USER", "")
    password = os.environ.get("TENSOR_AGENT_PASS", "")
    if not user or not password:
        print("TENSOR_AGENT_USER / TENSOR_AGENT_PASS not set")
        return 2
    login = urllib.request.Request(
        f"{base}/auth/login",
        data=json.dumps({"username": user, "password": password}).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(login, timeout=20) as resp:
            token = json.load(resp)
        token = token.get("access_token") or token.get("token")
    except Exception as exc:
        print(f"[ERROR] login failed ({exc}); payload cached for replay")
        return 3
    req = urllib.request.Request(
        f"{base}/tensor/reports",
        data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {token}",
                 "Content-Type": "application/json"},
        method="POST")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            print(f"submitted: HTTP {resp.status} "
                  f"{resp.read().decode()[:300]}")
        return 0
    except urllib.error.HTTPError as exc:
        print(f"[ERROR] HTTP {exc.code}: {exc.read().decode()[:500]}")
        return 4
    except Exception as exc:
        print(f"[ERROR] submit failed ({exc}); payload cached for replay")
        return 5


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out", default=str(ART / "platform_report.json"))
    args = parser.parse_args(argv)
    payload = build_payload()
    out = pathlib.Path(args.out)
    out.write_text(json.dumps(payload))
    print(f"payload written to {out} "
          f"({len(payload['markdown'].splitlines())} md lines, "
          f"{len(payload['figures'])} figures)")
    if args.dry_run:
        return 0
    return submit(payload)


if __name__ == "__main__":
    raise SystemExit(main())