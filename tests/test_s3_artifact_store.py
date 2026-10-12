from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from tensorphysics.artifact_store import ArtifactStoreError
from tensorphysics.provenance import ArtifactRef
from tensorphysics.s3_artifact_store import S3ArtifactStore


class FakeS3:
    def __init__(self):
        self.objects = {}

    def head_object(self, *, Bucket, Key):
        if (Bucket, Key) not in self.objects:
            error = RuntimeError("not found")
            error.response = {"Error": {"Code": "404"}, "ResponseMetadata": {"HTTPStatusCode": 404}}
            raise error
        data, content_type, metadata = self.objects[(Bucket, Key)]
        return {"ContentLength": len(data), "ContentType": content_type, "Metadata": metadata}

    def put_object(self, *, Bucket, Key, Body, ContentLength, ContentType, Metadata, IfNoneMatch):
        assert IfNoneMatch == "*"
        if (Bucket, Key) in self.objects:
            error = RuntimeError("precondition failed")
            error.response = {"Error": {"Code": "PreconditionFailed"}, "ResponseMetadata": {"HTTPStatusCode": 412}}
            raise error
        data = Body.read()
        assert len(data) == ContentLength
        self.objects[(Bucket, Key)] = (data, ContentType, Metadata)


def test_s3_store_put_verify_and_idempotent(tmp_path: Path):
    source = tmp_path / "history file.csv"
    source.write_bytes(b"time,force\n0,1.0\n")
    client = FakeS3()
    store = S3ArtifactStore("engineering-artifacts", prefix="runs/team-a", client=client)
    ref = store.put_file(source, "run-1/history.csv", media_type="text/csv")
    assert ref.uri == "s3://engineering-artifacts/runs/team-a/run-1/history.csv"
    assert ref.checksum_sha256 == hashlib.sha256(source.read_bytes()).hexdigest()
    assert ref.size_bytes == source.stat().st_size
    assert store.verify(ref) is True
    assert store.put_file(source, "run-1/history.csv", media_type="text/csv").checksum_sha256 == ref.checksum_sha256
    assert len(client.objects) == 1


def test_s3_store_rejects_conflicting_content(tmp_path: Path):
    source = tmp_path / "output.bin"
    source.write_bytes(b"first")
    store = S3ArtifactStore("bucket", client=FakeS3())
    store.put_file(source, "run/output.bin", media_type="application/octet-stream")
    source.write_bytes(b"different")
    with pytest.raises(ArtifactStoreError, match="different content"):
        store.put_file(source, "run/output.bin", media_type="application/octet-stream")


def test_s3_store_verify_detects_bad_checksum_and_wrong_bucket(tmp_path: Path):
    source = tmp_path / "output.bin"
    source.write_bytes(b"payload")
    store = S3ArtifactStore("bucket", prefix="safe", client=FakeS3())
    ref = store.put_file(source, "run/output.bin", media_type="application/octet-stream")
    wrong = ArtifactRef(uri=ref.uri, media_type=ref.media_type, checksum_sha256="0" * 64, size_bytes=ref.size_bytes)
    assert store.verify(wrong) is False
    with pytest.raises(ValueError, match="configured bucket"):
        store.verify(ArtifactRef(uri="s3://other/safe/run/output.bin", media_type="application/octet-stream"))


def test_s3_store_enforces_prefix_and_safe_keys():
    store = S3ArtifactStore("bucket", prefix="safe", client=FakeS3())
    with pytest.raises(ValueError, match="outside this store"):
        store.verify(ArtifactRef(uri="s3://bucket/other/file.bin", media_type="application/octet-stream"))
    with pytest.raises(ValueError, match="artifact key"):
        store._key("../escape")
