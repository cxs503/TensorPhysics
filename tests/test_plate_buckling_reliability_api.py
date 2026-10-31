"""Tests of the ``POST /calculations/plate-buckling-reliability`` endpoint.

The endpoint is tested at three levels: the framework-neutral handler
(``handle_request``), payload validation, and a real HTTP round trip on the
standard-library server.  A synthetic capacity surface is supplied in the
payload so the tests do not need the packaged FE calibration and run in
milliseconds; one test exercises the packaged calibration when it is present.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

import numpy as np
import pytest

from tensorphysics.plate_buckling_reliability import (
    ENDPOINT,
    UltimateStrengthSurrogate,
    faulkner_reference,
    handle_calculation,
    handle_request,
    load_default_surrogate,
    make_server,
)
from tensorphysics.plate_buckling_reliability.surrogate import (
    default_surrogate_path,
)

SIGMA_Y = 313.0e6
E_MOD = 2.06e11
CAL_BETAS = (1.0, 1.5, 2.0, 2.5, 3.0, 3.5)


@pytest.fixture(scope="module")
def calibration() -> dict:
    points = [{"beta": b, "sigma_u_over_sigma_y": float(faulkner_reference(b))}
              for b in CAL_BETAS]
    return UltimateStrengthSurrogate.fit(points, source="synthetic-faulkner").to_dict()


@pytest.fixture(scope="module")
def payload(calibration) -> dict:
    return {
        "panel": {"a": 3.0, "b": 1.0, "t_mm": 13.6, "sigma_y_MPa": 313.0,
                  "E_GPa": 206.0, "nu": 0.3, "mesh": [8, 4], "bc": "SSSS",
                  "imperfection_level": "average", "sigma_r_ratio": 0.0,
                  "sigma_global_MPa": 95.0, "psi_membrane": 1.5},
        "variables": {"p_ice_mean": 2.5, "p_ice_cov": 0.28, "fy_mean": 313.0,
                      "fy_cov": 0.08, "dt_mean": 2.0, "dt_sd": 0.5,
                      "k_mean": 1.0, "k_cov": 0.10,
                      "load_model_correlation": 0.25},
        "options": {"run_fem": False, "target_beta": 4.0},
        "calibration": calibration,
    }


# --------------------------------------------------------------------------- #
# payload validation
# --------------------------------------------------------------------------- #
def test_rejects_non_object_payload():
    status, body = handle_calculation(["not", "an", "object"])
    assert status == 400
    assert "error" in body


def test_rejects_bad_geometry(payload):
    bad = json.loads(json.dumps(payload))
    bad["panel"]["a"] = 1.1          # aspect ratio below the verified envelope
    status, body = handle_calculation(bad)
    assert status == 400
    assert "aspect ratio" in body["error"]


def test_rejects_unknown_option(payload):
    bad = json.loads(json.dumps(payload))
    bad["options"]["run_fem_forever"] = True
    status, body = handle_calculation(bad)
    assert status == 400
    assert "unknown option" in body["error"]


def test_rejects_negative_sample_count(payload):
    bad = json.loads(json.dumps(payload))
    bad["options"]["monte_carlo_samples"] = -5
    status, body = handle_calculation(bad)
    assert status == 400


def test_rejects_invalid_json_body():
    status, body = handle_request("POST", ENDPOINT, b"{not json")
    assert status == 400
    assert "invalid JSON" in body["error"]


# --------------------------------------------------------------------------- #
# routing
# --------------------------------------------------------------------------- #
def test_unknown_path_returns_404():
    status, body = handle_request("POST", "/calculations/unknown", {})
    assert status == 404
    assert body["endpoint"] == ENDPOINT


def test_get_on_endpoint_returns_405(payload):
    status, _ = handle_request("GET", ENDPOINT, None)
    assert status == 405


def test_health_endpoint():
    status, body = handle_request("GET", "/health", None)
    assert status == 200
    assert body["status"] == "ok"


# --------------------------------------------------------------------------- #
# successful assessment
# --------------------------------------------------------------------------- #
def test_success_payload_schema(payload):
    status, body = handle_calculation(payload)
    assert status == 200
    for key in ("sigma_ultimate_MPa", "beta", "pf", "beta_sorm", "pf_sorm",
                "beta_system", "pf_system", "design_point", "target_thickness",
                "deterministic", "probabilistic", "panel", "geometry_check",
                "endpoint", "units"):
        assert key in body
    assert body["endpoint"] == ENDPOINT
    assert body["units"]["stress"] == "MPa"
    assert body["sigma_ultimate_MPa"] > 0.0
    assert 0.0 < body["pf"] < 1.0
    assert body["beta"] > 0.0


def test_success_reports_surrogate_provenance(payload):
    status, body = handle_calculation(payload)
    assert status == 200
    surrogate = body["probabilistic"]["surrogate"]
    assert surrogate["source"] == "synthetic-faulkner"
    assert len(surrogate["coefficients"]) == 3
    assert surrogate["accuracy"]["max_rel_error"] < 1e-9


def test_target_thickness_present_and_near_design(payload):
    status, body = handle_calculation(payload)
    assert status == 200
    target = body["target_thickness"]
    assert "t_required_mm" in target
    assert abs(target["beta_achieved"] - 4.0) < 2e-3


def test_target_beta_can_be_disabled(payload):
    quiet = json.loads(json.dumps(payload))
    quiet["options"]["target_beta"] = None
    status, body = handle_calculation(quiet)
    assert status == 200
    assert body["target_thickness"] is None


def test_monte_carlo_option(payload):
    with_mc = json.loads(json.dumps(payload))
    with_mc["options"]["monte_carlo_samples"] = 5000
    status, body = handle_calculation(with_mc)
    assert status == 200
    mc = body["probabilistic"]["monte_carlo"]["ultimate"]
    assert mc["samples"] == 5000
    assert "system_monte_carlo" in body["probabilistic"]


def test_packaged_calibration_is_used_when_payload_omits_it(payload):
    if not default_surrogate_path().exists():
        pytest.skip("packaged FE calibration not generated in this checkout")
    default_payload = json.loads(json.dumps(payload))
    default_payload.pop("calibration")
    status, body = handle_calculation(default_payload)
    assert status == 200
    assert body["probabilistic"]["surrogate"]["source"] == "calibrated"


def test_run_fem_flag_executes_the_arc_length_solver(payload):
    """The deterministic branch really runs the FE trace (one short trace)."""
    live = json.loads(json.dumps(payload))
    live["panel"]["t_mm"] = 19.49
    live["options"]["run_fem"] = True
    live["options"]["target_beta"] = None
    status, body = handle_calculation(live)
    assert status == 200
    assert body["deterministic"]["capacity_source"].startswith("arc-length nonlinear FE")
    assert body["deterministic"]["ultimate"]["n_steps"] > 0
    assert abs(body["deterministic"]["ultimate"]["sigma_u_over_sigma_y"]
               / float(faulkner_reference(body["deterministic"]["ultimate"]["beta"]))
               - 1.0) < 0.10


# --------------------------------------------------------------------------- #
# real HTTP round trip
# --------------------------------------------------------------------------- #
def test_http_roundtrip(payload):
    server = make_server("127.0.0.1", 0)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}{ENDPOINT}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(request, timeout=60) as response:
            assert response.status == 200
            body = json.loads(response.read().decode("utf-8"))
        assert body["beta"] > 0.0
        assert body["endpoint"] == ENDPOINT

        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health",
                                    timeout=30) as health:
            assert json.loads(health.read().decode("utf-8"))["status"] == "ok"

        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/nope", timeout=30)
        assert exc.value.code == 404
    finally:
        server.shutdown()
        server.server_close()