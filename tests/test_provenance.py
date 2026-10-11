import json

import pytest

from tensorphysics import ArtifactRef, ManifestError, RunManifest, RunStatus, ValidationStatus


def make_manifest(**overrides):
    config = {"solver": "TensorDEM", "dt": 0.001, "steps": 100}
    values = dict(
        run_id="run-0001",
        status=RunStatus.SUCCEEDED,
        created_at="2026-10-11T10:00:00Z",
        config=config,
        config_sha256=RunManifest.digest_config(config),
        code_revisions={"cxs503/TensorDEM": "abc123"},
        environment={"python": "3.12", "device": "cpu"},
        resources={"cpu_threads": 1},
        random_seeds={"dem": 7},
        inputs=[ArtifactRef(uri="file:///inputs/model.json", media_type="application/json")],
        outputs=[ArtifactRef(uri="file:///outputs/probes.csv", media_type="text/csv",
                             checksum_sha256="a" * 64, size_bytes=2048)],
        validation_status=ValidationStatus.PASSED,
    )
    values.update(overrides)
    return RunManifest(**values)


def test_config_digest_is_stable_across_mapping_key_order():
    assert RunManifest.digest_config({"a": 1, "b": 2}) == RunManifest.digest_config({"b": 2, "a": 1})


def test_manifest_serializes_status_and_artifacts_as_json():
    manifest = make_manifest()
    payload = json.loads(manifest.to_json())
    assert payload["run_id"] == "run-0001"
    assert payload["status"] == "succeeded"
    assert payload["validation_status"] == "passed"
    assert payload["outputs"][0]["checksum_sha256"] == "a" * 64
    assert "tensor_payload" not in payload


def test_manifest_detects_config_mutation():
    manifest = make_manifest()
    manifest.config = {"solver": "TensorDEM", "dt": 0.002, "steps": 100}
    with pytest.raises(ManifestError, match="digest mismatch"):
        manifest.verify_config_digest()


@pytest.mark.parametrize("checksum", ["xyz", "A" * 64, "a" * 63])
def test_artifact_ref_rejects_invalid_checksum(checksum):
    with pytest.raises(ManifestError, match="checksum_sha256"):
        ArtifactRef(uri="artifact://results/state", media_type="application/octet-stream",
                    checksum_sha256=checksum)


def test_artifact_ref_rejects_negative_size():
    with pytest.raises(ManifestError, match="size_bytes"):
        ArtifactRef(uri="artifact://results/state", media_type="application/octet-stream",
                    size_bytes=-1)


def test_manifest_rejects_untraceable_code_revisions():
    config = {"dt": 0.1}
    with pytest.raises(ManifestError, match="code_revisions"):
        make_manifest(code_revisions={})


def test_manifest_does_not_allow_validation_pass_before_run_succeeds():
    with pytest.raises(ManifestError, match="cannot pass"):
        make_manifest(status=RunStatus.RUNNING, validation_status=ValidationStatus.PASSED)


def test_manifest_rejects_non_json_config():
    with pytest.raises(ManifestError, match="canonical JSON"):
        RunManifest.digest_config({"tensor": object()})


def test_manifest_rejects_non_finite_config_values():
    with pytest.raises(ManifestError, match="canonical JSON"):
        RunManifest.digest_config({"dt": float("nan")})
