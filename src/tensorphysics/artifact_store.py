"""Artifact storage adapters with checksum-verified local persistence.

The local adapter is a reference implementation for end-to-end workflows. It
copies bytes atomically, verifies the destination checksum, and returns a
metadata-only ArtifactRef. Object-store adapters can implement the same protocol.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from pathlib import Path, PurePosixPath
from typing import Protocol, runtime_checkable
from urllib.parse import unquote, urlsplit

from .provenance import ArtifactRef


class ArtifactStoreError(RuntimeError):
    """Artifact could not be stored or verified."""


@runtime_checkable
class ArtifactStore(Protocol):
    """Minimal storage contract used by solver workflows."""

    def put_file(self, source: str | Path, key: str, *, media_type: str) -> ArtifactRef:
        """Persist a local file and return a verified reference."""
        ...

    def verify(self, artifact: ArtifactRef) -> bool:
        """Return true only if the referenced object's size and checksum match."""
        ...


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_key(key: str) -> PurePosixPath:
    if not isinstance(key, str) or not key.strip():
        raise ValueError("artifact key must be a non-empty relative path")
    if "\\" in key or "\x00" in key:
        raise ValueError("artifact key must use POSIX separators and contain no NUL")
    path = PurePosixPath(key)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise ValueError("artifact key must not be absolute or traverse parent directories")
    if not path.parts:
        raise ValueError("artifact key must identify a file")
    return path


class LocalArtifactStore:
    """Atomic, checksum-verifying artifact store backed by a local directory."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _destination(self, key: str) -> Path:
        relative = _validate_key(key)
        destination = (self.root / Path(*relative.parts)).resolve()
        if destination != self.root and self.root not in destination.parents:
            raise ValueError("artifact key escapes the storage root")
        return destination

    def put_file(self, source: str | Path, key: str, *, media_type: str) -> ArtifactRef:
        source_path = Path(source).expanduser().resolve(strict=True)
        if not source_path.is_file():
            raise ValueError(f"artifact source is not a regular file: {source_path}")
        if not isinstance(media_type, str) or not media_type.strip():
            raise ValueError("media_type must be a non-empty string")
        destination = self._destination(key)
        destination.parent.mkdir(parents=True, exist_ok=True)
        source_digest = sha256_file(source_path)
        source_size = source_path.stat().st_size

        # Idempotent repeat: existing identical content is safe; conflicting bytes
        # never silently overwrite a previously registered artifact key.
        if destination.exists():
            if destination.is_file() and destination.stat().st_size == source_size and sha256_file(destination) == source_digest:
                return self._reference(destination, media_type, source_digest, source_size)
            raise ArtifactStoreError(f"artifact key already exists with different content: {key}")

        fd, temporary_name = tempfile.mkstemp(prefix=".artifact-", dir=destination.parent)
        os.close(fd)
        temporary = Path(temporary_name)
        try:
            shutil.copyfile(source_path, temporary)
            with temporary.open("rb") as stream:
                os.fsync(stream.fileno())
            copied_digest = sha256_file(temporary)
            copied_size = temporary.stat().st_size
            if copied_digest != source_digest or copied_size != source_size:
                raise ArtifactStoreError(f"artifact verification failed after copy: {key}")
            # Hard-link creation is atomic and fails rather than replacing a race winner.
            try:
                os.link(temporary, destination)
            except FileExistsError:
                if destination.is_file() and destination.stat().st_size == source_size and sha256_file(destination) == source_digest:
                    pass
                else:
                    raise ArtifactStoreError(f"artifact key concurrently created with different content: {key}")
            return self._reference(destination, media_type, source_digest, source_size)
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass

    def verify(self, artifact: ArtifactRef) -> bool:
        """Verify a store-owned file URI, byte count, and SHA-256."""
        parsed = urlsplit(artifact.uri)
        if parsed.scheme != "file" or parsed.netloc not in ("", "localhost"):
            raise ValueError("LocalArtifactStore can only verify local file:// artifact references")
        path = Path(unquote(parsed.path)).resolve()
        if path != self.root and self.root not in path.parents:
            raise ValueError("artifact URI is outside this store's root")
        if not path.is_file():
            return False
        if artifact.size_bytes is not None and path.stat().st_size != artifact.size_bytes:
            return False
        if artifact.checksum_sha256 is not None and sha256_file(path) != artifact.checksum_sha256:
            return False
        return True

    @staticmethod
    def _reference(path: Path, media_type: str, checksum: str, size: int) -> ArtifactRef:
        return ArtifactRef(
            uri=path.resolve().as_uri(),
            media_type=media_type,
            checksum_sha256=checksum,
            size_bytes=size,
            metadata={"storage": "local", "verified": True},
        )
