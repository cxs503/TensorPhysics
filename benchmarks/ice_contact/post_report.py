#!/usr/bin/env python3
"""Post the mission report payload to TensorPlatform using env credentials.

Developer helper (not part of the model): reads credentials from the agent
environment file and posts ``/tmp/report.json`` to ``/tensor/reports``.  No
credential is stored in this file or in the repository.
"""
import json
import urllib.error
import urllib.request

ENV_PATH = "/root/tp-agent/agent.env"


def load_env(path: str) -> dict:
    values = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    return values


def post_json(url: str, payload: dict, token: str | None = None) -> tuple[int, str]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, method="POST")
    request.add_header("Content-Type", "application/json")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            return response.status, response.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode("utf-8")


def main() -> None:
    env = load_env(ENV_PATH)
    base = env["TENSOR_API_BASE"].rstrip("/")
    status, body = post_json(
        f"{base}/auth/login",
        {"username": env["TENSOR_AGENT_USER"], "password": env["TENSOR_AGENT_PASS"]},
    )
    print("login", status)
    if status >= 300:
        print(body[:400])
        return
    decoded = json.loads(body)
    token = decoded.get("access_token") or decoded.get("token")
    payload = json.loads(open("/tmp/report.json", encoding="utf-8").read())
    status, body = post_json(f"{base}/tensor/reports", payload, token)
    print("report", status)
    print(body[:800])


if __name__ == "__main__":
    main()