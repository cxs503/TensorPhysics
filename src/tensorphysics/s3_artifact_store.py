"""S3-compatible artifact storage with checksum verification.

Install the optional s3 extra to use boto3. Credentials are resolved through
boto3's standard provider chain and are never written to run manifests.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote, urlsplit

from .artifact_store import ArtifactStoreError, sha256_file, _validate_key
from .provenance import ArtifactRef


class S3ArtifactStore:
    """Persist immutable artifacts in an S3 or S3-compatible bucket."""

    def __init__(
        self, bucket: str, *, prefix: str = "", region_name: str | None = None,
        endpoint_url: str | None = None, client: Any | None = None,
    ) -> None:
        if not isinstance(bucket, str) or not bucket.strip() or "/" in bucket:
            raise ValueError("bucket must be a non-empty bucket name without slashes")
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        if self.prefix:
            _validate_key(self.prefix)
        if client is None:
            try:
                import boto3
            except ImportError as exc:
                raise RuntimeError(
                    "S3 storage requires boto3; install TensorPhysics with the s3 extra"
                ) from exc
            client = boto3.client("s3", region_name=region_name, endpoint_url=endpoint_url)
        self.client = client

    def _key(self, key: str) -> str:
        relative = str(_validate_key(key))
        return f"{self.prefix}/{relative}" if self.prefix else relative

    def _reference(self, key: str, media_type: str, checksum: str, size: int) -> ArtifactRef:
        return ArtifactRef(
            uri=f"s3://{self.bucket}/{quote(key, safe='/')}",
            media_type=media_type, checksum_sha256=checksum, size_bytes=size,
            metadata={"storage": "s3", "verified": True},
        )

    def _head(self, key: str) -> dict[str, Any] | None:
        try:
            return self.client.head_object(Bucket=self.bucket, Key=key)
        except Exception as exc:
            response = getattr(exc, "response", {})
            error = response.get("Error", {}) if isinstance(response, dict) else {}
            code = str(error.get("Code", ""))
            status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
            if code in {"404", "NoSuchKey", "NotFound"} or status == 404:
                return None
            raise ArtifactStoreError(f"could not inspect s3://{self.bucket}/{key}: {exc}") from exc

    @staticmethod
    def _matches(head: dict[str, Any], checksum: str, size: int) -> bool:
        metadata = {str(k).lower(): str(v) for k, v in head.get("Metadata", {}).items()}
        return int(head.get("ContentLength", -1)) == size and metadata.get("sha256") == checksum

    def put_file(self, source: str | Path, key: str, *, media_type: str) -> ArtifactRef:
        source_path = Path(source).expanduser().resolve(strict=True)
        if not source_path.is_file():
            raise ValueError(f"artifact source is not a regular file: {source_path}")
        if not isinstance(media_type, str) or not media_type.strip():
            raise ValueError("media_type must be a non-empty string")
        object_key = self._key(key)
        checksum, size = sha256_file(source_path), source_path.stat().st_size
        existing = self._head(object_key)
        if existing is not None:
            if self._matches(existing, checksum, size):
                return self._reference(object_key, media_type, checksum, size)
            raise ArtifactStoreError(
                f"artifact key already exists with different content: s3://{self.bucket}/{object_key}"
            )
        try:
            with source_path.open("rb") as stream:
                self.client.put_object(
                    Bucket=self.bucket, Key=object_key, Body=stream, ContentLength=size,
                    ContentType=media_type, Metadata={"sha256": checksum}, IfNoneMatch="*",
                )
        except Exception as exc:
            # A concurrent identical writer is idempotent; conflicting bytes are not.
            current = self._head(object_key)
            if current is not None and self._matches(current, checksum, size):
                return self._reference(object_key, media_type, checksum, size)
            raise ArtifactStoreError(
                f"could not store s3://{self.bucket}/{object_key}: {exc}"
            ) from exc
        ref = self._reference(object_key, media_type, checksum, size)
        if not self.verify(ref):
            raise ArtifactStoreError(f"stored object failed checksum/size verification: {ref.uri}")
        return ref

    def verify(self, artifact: ArtifactRef) -> bool:
        """Verify remote object existence, byte count, and stored SHA-256 metadata."""
        parsed = urlsplit(artifact.uri)
        if parsed.scheme != "s3" or parsed.netloc != self.bucket:
            raise ValueError("S3ArtifactStore can only verify references in its configured bucket")
        object_key = unquote(parsed.path.lstrip("/"))
        if not object_key:
            raise ValueError("artifact URI must identify an object key")
        if self.prefix and not object_key.startswith(self.prefix + "/"):
            raise ValueError("artifact URI is outside this store's configured prefix")
        head = self._head(object_key)
        if head is None:
            return False
        if artifact.size_bytes is not None and int(head.get("ContentLength", -1)) != artifact.size_bytes:
            return False
        if artifact.checksum_sha256 is not None:
            metadata = {str(k).lower(): str(v) for k, v in head.get("Metadata", {}).items()}
            if metadata.get("sha256") != artifact.checksum_sha256:
                return False
        return True
