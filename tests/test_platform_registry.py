import json
from urllib.request import Request

import pytest

from tensorphysics.platform_registry import (
    PlatformRegistryError,
    TensorPlatformRunRegistry,
)


class FakeResponse:
    def __init__(self, body):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps(self.body).encode("utf-8")


def test_register_posts_compact_manifest_and_auth(monkeypatch):
    seen = {}

    def fake_urlopen(request, timeout):
        seen["url"] = request.full_url
        seen["headers"] = dict(request.header_items())
        seen["timeout"] = timeout
        seen["payload"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse({"run_id": "dem-run-01", "created": True, "manifest_sha256": "a" * 64})

    monkeypatch.setattr("tensorphysics.platform_registry.urlopen", fake_urlopen)
    registry = TensorPlatformRunRegistry("https://platform.example/", "secret", 12, timeout=4)
    result = registry.register({
        "run_id": "dem-run-01",
        "status": "succeeded",
        "validation_status": "not_run",
        "outputs": [{"uri": "s3://bucket/result.csv", "media_type": "text/csv"}],
    })

    assert result["created"] is True
    assert seen["url"] == "https://platform.example/api/v1/tensorphysics/runs"
    assert seen["headers"]["Authorization"] == "Bearer secret"
    assert seen["headers"]["Idempotency-key"] == "dem-run-01"
    assert seen["timeout"] == 4
    assert seen["payload"]["organization_id"] == 12
    assert "s3://bucket/result.csv" == seen["payload"]["manifest"]["outputs"][0]["uri"]


def test_registry_requires_absolute_url_and_positive_organization():
    with pytest.raises(ValueError):
        TensorPlatformRunRegistry("platform.local", "token", 1)
    with pytest.raises(ValueError):
        TensorPlatformRunRegistry("https://platform.local", "token", 0)


def test_http_conflict_is_reported_without_hiding_status(monkeypatch):
    from urllib.error import HTTPError
    import io

    def fake_urlopen(request, timeout):
        raise HTTPError(request.full_url, 409, "Conflict", {}, io.BytesIO(b"duplicate run"))

    monkeypatch.setattr("tensorphysics.platform_registry.urlopen", fake_urlopen)
    registry = TensorPlatformRunRegistry("https://platform.local", "token", 1)
    with pytest.raises(PlatformRegistryError) as error:
        registry.register({"run_id": "r", "status": "succeeded", "validation_status": "not_run"})
    assert error.value.status_code == 409
