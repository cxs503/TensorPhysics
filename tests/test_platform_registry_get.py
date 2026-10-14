import json
import pytest

from tensorphysics.platform_registry import PlatformRegistryError, TensorPlatformRunRegistry


class FakeResponse:
    def __init__(self, body):
        self.body = body
    def __enter__(self):
        return self
    def __exit__(self, *args):
        return False
    def read(self):
        return json.dumps(self.body).encode("utf-8")


def test_get_sends_auth_and_organization_scope(monkeypatch):
    seen = {}
    def fake_urlopen(request, timeout):
        seen["url"] = request.full_url
        seen["headers"] = dict(request.header_items())
        seen["method"] = request.get_method()
        seen["timeout"] = timeout
        return FakeResponse({
            "run_id": "dem-run-01", "organization_id": 9,
            "manifest": {"run_id": "dem-run-01", "status": "succeeded"},
        })
    monkeypatch.setattr("tensorphysics.platform_registry.urlopen", fake_urlopen)
    registry = TensorPlatformRunRegistry("https://platform.example", "secret", 9, timeout=4)
    result = registry.get("dem-run-01")
    assert result["run_id"] == "dem-run-01"
    assert seen["url"] == "https://platform.example/api/v1/tensorphysics/runs/dem-run-01?organization_id=9"
    assert seen["method"] == "GET"
    assert seen["headers"]["Authorization"] == "Bearer secret"
    assert seen["timeout"] == 4


def test_get_rejects_mismatched_manifest_identity(monkeypatch):
    def fake_urlopen(request, timeout):
        return FakeResponse({"run_id": "dem-run-01", "manifest": {"run_id": "other"}})
    monkeypatch.setattr("tensorphysics.platform_registry.urlopen", fake_urlopen)
    registry = TensorPlatformRunRegistry("https://platform.example", "secret", 9)
    with pytest.raises(PlatformRegistryError, match="expected manifest"):
        registry.get("dem-run-01")
