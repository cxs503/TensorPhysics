from pathlib import Path

import pytest

from tensorphysics.artifact_store import ArtifactStoreError, LocalArtifactStore


def test_put_file_copies_bytes_and_returns_verified_reference(tmp_path: Path):
    source = tmp_path / "history.csv"
    source.write_bytes(b"time,force\n0,1.25\n1,2.5\n")
    store = LocalArtifactStore(tmp_path / "store")

    ref = store.put_file(source, "run-001/history.csv", media_type="text/csv")

    stored = tmp_path / "store" / "run-001" / "history.csv"
    assert stored.read_bytes() == source.read_bytes()
    assert ref.uri == stored.resolve().as_uri()
    assert ref.size_bytes == source.stat().st_size
    assert ref.checksum_sha256 is not None
    assert store.verify(ref) is True
    assert ref.metadata["verified"] is True


def test_put_file_is_idempotent_for_same_bytes_and_rejects_conflicts(tmp_path: Path):
    source = tmp_path / "source.bin"
    source.write_bytes(b"first")
    store = LocalArtifactStore(tmp_path / "store")

    first = store.put_file(source, "run-1/output.bin", media_type="application/octet-stream")
    repeated = store.put_file(source, "run-1/output.bin", media_type="application/octet-stream")
    assert repeated.checksum_sha256 == first.checksum_sha256

    source.write_bytes(b"different")
    with pytest.raises(ArtifactStoreError, match="different content"):
        store.put_file(source, "run-1/output.bin", media_type="application/octet-stream")


def test_verify_detects_modified_artifact(tmp_path: Path):
    source = tmp_path / "input.csv"
    source.write_text("a,b\n1,2\n", encoding="utf-8")
    store = LocalArtifactStore(tmp_path / "store")
    ref = store.put_file(source, "run-1/input.csv", media_type="text/csv")
    Path(ref.uri.removeprefix("file://")).write_text("tampered", encoding="utf-8")

    assert store.verify(ref) is False


@pytest.mark.parametrize("key", ["", "/absolute/file", "../escape", "run/../../escape", "run\\escape"])
def test_rejects_unsafe_keys(tmp_path: Path, key: str):
    source = tmp_path / "input.bin"
    source.write_bytes(b"data")
    store = LocalArtifactStore(tmp_path / "store")

    with pytest.raises(ValueError, match="artifact key"):
        store.put_file(source, key, media_type="application/octet-stream")


def test_rejects_directory_as_source(tmp_path: Path):
    store = LocalArtifactStore(tmp_path / "store")
    with pytest.raises(ValueError, match="regular file"):
        store.put_file(tmp_path, "run-1/directory", media_type="application/octet-stream")
