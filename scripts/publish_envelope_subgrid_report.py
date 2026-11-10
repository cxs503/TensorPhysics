#!/usr/bin/env python3
"""Post the N09 envelope report payload to TensorPlatform.

Credentials come from the agent environment (``/root/tp-agent/agent.env`` or the
current environment); nothing is stored in the repository and no number is
retyped here -- the payload is the JSON written by
``scripts/generate_envelope_subgrid_report.py``.

Usage::

    python scripts/publish_envelope_subgrid_report.py --payload /tmp/report_m9733.json
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import urllib.error
import urllib.request

ENV_PATH = pathlib.Path("/root/tp-agent/agent.env")


def load_env(path: pathlib.Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    for key in ("TENSOR_API_BASE", "TENSOR_AGENT_USER", "TENSOR_AGENT_PASS"):
        if key in os.environ:
            values.setdefault(key, os.environ[key])
    return values


def post_json(url: str, payload: dict, token: str | None = None) -> tuple[int, str]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, method="POST")
    request.add_header("Content-Type", "application/json")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            return response.status, response.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode("utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", default="/tmp/report_m9733.json")
    parser.add_argument("--env", default=str(ENV_PATH))
    args = parser.parse_args(argv)

    env = load_env(pathlib.Path(args.env))
    missing = [k for k in ("TENSOR_API_BASE", "TENSOR_AGENT_USER", "TENSOR_AGENT_PASS") if k not in env]
    if missing:
        print(f"missing credentials: {missing}", file=sys.stderr)
        return 2
    base = env["TENSOR_API_BASE"].rstrip("/")
    status, body = post_json(
        f"{base}/auth/login",
        {"username": env["TENSOR_AGENT_USER"], "password": env["TENSOR_AGENT_PASS"]},
    )
    if status >= 300:
        print(f"login failed ({status}): {body[:300]}", file=sys.stderr)
        return 1
    token = json.loads(body).get("access_token") or json.loads(body).get("token")

    payload = json.loads(pathlib.Path(args.payload).read_text())
    status, body = post_json(f"{base}/tensor/reports", payload, token)
    print(f"POST /tensor/reports -> {status}")
    print(body[:600])
    return 0 if status < 300 else 1


if __name__ == "__main__":
    raise SystemExit(main())