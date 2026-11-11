"""Read-only knowledge/negative-knowledge/OSS-solution probe for mission m14678 (auth via env creds)."""
import json
import os
import sys
import urllib.parse
import urllib.request

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")


def login():
    body = json.dumps({
        "username": os.environ["TENSOR_AGENT_USER"],
        "password": os.environ["TENSOR_AGENT_PASS"],
    }).encode()
    req = urllib.request.Request(BASE + "/auth/login", data=body,
                                headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.loads(r.read().decode())
    return data.get("access_token") or data.get("token") or data


def get(path, params=None, token=None):
    url = BASE + path
    if params:
        url += "?" + urllib.parse.urlencode(params)
    hdr = {"Authorization": f"Bearer {token}"} if token else {}
    req = urllib.request.Request(url, headers=hdr)
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read().decode())
    except Exception as exc:  # noqa: BLE001
        try:
            body = exc.read().decode()[:300]  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            body = ""
        return {"__error__": f"{type(exc).__name__}: {exc}", "body": body}


def main():
    tok = login()
    if not isinstance(tok, str):
        print(json.dumps({"login": tok}, indent=2)[:2000])
        return 1
    probes = {
        "kb_search_field_exchange": ("/tensor/knowledge/search", {"q": "field exchange", "limit": 10}),
        "kb_search_zero_copy": ("/tensor/knowledge/search", {"q": "shared memory zero copy", "limit": 10}),
        "kb_search_domain": ("/tensor/knowledge/search", {"q": "TensorPhysics", "limit": 10}),
        "neg_kb": ("/tensor/negative-knowledge/search", {"q": "TensorPhysics", "limit": 20}),
        "neg_kb_field": ("/tensor/negative-knowledge/search", {"q": "field exchange", "limit": 20}),
        "oss_gap": ("/tensor/knowledge/search", {"q": "GAP-CROSS-FIELD", "limit": 10}),
    }
    out = {}
    for key, (path, params) in probes.items():
        out[key] = get(path, params, tok)
    print(json.dumps(out, indent=2, ensure_ascii=False)[:16000])
    return 0


if __name__ == "__main__":
    sys.exit(main())