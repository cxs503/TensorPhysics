"""HTTP surface of the unified buckling/reliability assessment.

Two layers, deliberately separated so the endpoint is testable without a web
framework (the platform's API gateway can mount either one):

``handle_calculation(payload) -> (status, body)``
    Framework-neutral request handler: payload validation, geometry gate, then
    the unified assessment.  Returns the HTTP status code and the JSON body.

``POST /calculations/plate-buckling-reliability``
    Reference binding exposed by :func:`serve` on a standard-library
    ``ThreadingHTTPServer`` (no third-party dependency), plus
    :func:`make_router` which returns a FastAPI ``APIRouter`` when FastAPI is
    installed.  ``GET /health`` reports liveness.

Payload (all fields optional; defaults reproduce the W4-N03 verification case)::

    {
      "panel": {"a": 3.0, "b": 1.0, "t_mm": 19.49, "sigma_y_MPa": 313.0,
                "E_GPa": 206.0, "nu": 0.3, "mesh": [8, 4], "bc": "SSSS",
                "imperfection_level": "average", "sigma_r_ratio": 0.0,
                "sigma_global_MPa": 95.0, "psi_membrane": 1.5},
      "variables": {"p_ice_mean": 2.5, "p_ice_cov": 0.28, "fy_mean": 313.0,
                    "fy_cov": 0.08, "dt_mean": 2.0, "dt_sd": 0.5,
                    "k_mean": 1.0, "k_cov": 0.10,
                    "load_model_correlation": 0.25},
      "options": {"run_fem": false, "monte_carlo_samples": 0,
                  "system_monte_carlo_samples": 0, "target_beta": 4.0},
      "calibration": {...}      # optional FE calibration override
    }

Response: ``sigma_ultimate_MPa``, ``beta``, ``pf`` (FORM), ``beta_sorm``,
``pf_sorm``, ``beta_system``/``pf_system`` (Ditlevsen bounds), the design point
in u- and x-space, the target-thickness inverse design, the deterministic block
and the surrogate provenance.
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, Tuple

from .panel import GeometryGateError
from .unified import UnifiedBucklingReliability

__all__ = ["ENDPOINT", "handle_calculation", "handle_request", "make_server",
           "serve", "make_router", "CURRENT_OPTIONS"]

ENDPOINT = "/calculations/plate-buckling-reliability"

#: Options that the handler understands (unknown keys are ignored).
_OPTION_KEYS = ("run_fem", "monte_carlo_samples", "system_monte_carlo_samples",
                "target_beta", "deflection_limit_ratio", "ds0", "max_steps")


def handle_calculation(payload: Dict[str, object]) -> Tuple[int, Dict[str, object]]:
    """Validate a payload and run the unified assessment (framework-neutral)."""
    if not isinstance(payload, dict):
        return 400, {"error": "payload must be a JSON object"}
    try:
        panel_block = payload.get("panel") or payload.get("geometry") or {}
        if not isinstance(panel_block, dict):
            return 400, {"error": "panel block must be an object"}
        options = payload.get("options") or {}
        if not isinstance(options, dict):
            return 400, {"error": "options block must be an object"}
        unknown = sorted(set(options) - set(_OPTION_KEYS))
        if unknown:
            return 400, {"error": f"unknown option(s): {unknown}"}

        model = UnifiedBucklingReliability.from_payload(payload)
        run_fem = bool(options.get("run_fem", False))
        samples = int(options.get("monte_carlo_samples", 0))
        system_samples = int(options.get("system_monte_carlo_samples", samples))
        if samples < 0 or system_samples < 0:
            return 400, {"error": "Monte Carlo sample counts must be non-negative"}
        target = options.get("target_beta", 4.0)
        target = None if target is None else float(target)

        result = model.assess(run_fem=run_fem, samples=samples,
                              system_samples=system_samples, target_beta=target)
    except (GeometryGateError, ValueError, KeyError) as exc:
        return 400, {"error": str(exc), "type": type(exc).__name__}
    except Exception as exc:                                   # pragma: no cover
        return 500, {"error": str(exc), "type": type(exc).__name__}

    body = {
        "endpoint": ENDPOINT,
        "model": "unified deterministic FEM + FORM/SORM plate buckling reliability",
        "units": {"stress": "MPa", "thickness": "mm", "beta": "-", "pf": "-"},
        "sigma_ultimate_MPa": result["sigma_ultimate_MPa"],
        "sigma_ultimate_over_sigma_y": result["deterministic"]["sigma_ultimate_over_sigma_y"],
        "beta": result["beta"],
        "pf": result["pf"],
        "beta_sorm": result["beta_sorm"],
        "pf_sorm": result["pf_sorm"],
        "beta_system": result["beta_system"],
        "pf_system": result["pf_system"],
        "design_point": result["design_point"],
        "target_thickness": result.get("target_thickness"),
        "deterministic": result["deterministic"],
        "probabilistic": result["probabilistic"],
        "panel": result["panel"],
        "geometry_check": result["geometry_check"],
    }
    return 200, body


def handle_request(method: str, path: str, body: bytes | Dict[str, object] | None) -> Tuple[int, Dict[str, object]]:
    """Route one HTTP request (used by the server and by the tests)."""
    route = path.split("?", 1)[0].rstrip("/") or "/"
    if route in ("/health", "/healthz"):
        return 200, {"status": "ok", "service": "tensorphysics-plate-buckling-reliability"}
    if route != ENDPOINT:
        return 404, {"error": f"unknown path {path!r}", "endpoint": ENDPOINT}
    if method.upper() != "POST":
        return 405, {"error": "use POST", "endpoint": ENDPOINT}
    if body is None or body == b"":
        payload: Dict[str, object] = {}
    elif isinstance(body, dict):
        payload = body
    else:
        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            return 400, {"error": f"invalid JSON body: {exc}"}
    return handle_calculation(payload)


def _make_handler():
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _respond(self, status: int, payload: Dict[str, object]) -> None:
            data = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):                                     # noqa: N802
            self._respond(*handle_request("GET", self.path, None))

        def do_POST(self):                                    # noqa: N802
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length) if length else b""
            self._respond(*handle_request("POST", self.path, body))

        def log_message(self, fmt, *args):                    # pragma: no cover
            pass

    return Handler


def make_server(host: str = "127.0.0.1", port: int = 8011) -> ThreadingHTTPServer:
    """Create (but do not start) the reference HTTP server."""
    return ThreadingHTTPServer((host, int(port)), _make_handler())


def serve(host: str = "127.0.0.1", port: int = 8011) -> None:  # pragma: no cover
    """Serve the endpoint until interrupted."""
    server = make_server(host, port)
    print(f"plate-buckling-reliability API on http://{host}:{port}{ENDPOINT}")
    try:
        server.serve_forever()
    finally:
        server.server_close()


def make_router():                                            # pragma: no cover
    """Return a FastAPI ``APIRouter`` when FastAPI is installed, else ``None``."""
    try:
        from fastapi import APIRouter, Body
        from fastapi.responses import JSONResponse
    except Exception:
        return None
    router = APIRouter()

    @router.get("/health")
    def health():                                             # noqa: D401
        return {"status": "ok", "service": "tensorphysics-plate-buckling-reliability"}

    @router.post(ENDPOINT)
    def calculate(payload: dict = Body(default={})):
        status, body = handle_calculation(payload)
        return JSONResponse(status_code=status, content=body)

    return router


if __name__ == "__main__":                                    # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8011)
    args = parser.parse_args()
    serve(args.host, args.port)