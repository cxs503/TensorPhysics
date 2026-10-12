# Artifact storage and run registration

TensorPhysics provides two explicit storage backends:

- `LocalArtifactStore` for local development and tests. Its references use
  `file://` URIs and are not portable across machines.
- `S3ArtifactStore` for S3 and S3-compatible object stores. Install the optional
  dependency with `python -m pip install -e '.[s3]'`.

## S3 smoke workflow

Configure credentials through the normal boto3 provider chain (environment,
shared credentials/config, workload identity, or instance role). Do not put
credentials in code, command-line arguments, manifests, or repository secrets
unless a CI integration test explicitly needs them. For a compatible endpoint,
set `TENSORPHYSICS_S3_ENDPOINT_URL`; set `AWS_REGION` as appropriate.

```bash
export TENSORPHYSICS_S3_BUCKET=engineering-artifacts
export TENSORPHYSICS_S3_PREFIX=tensorphysics-runs
python examples/end_to_end_s3_artifact_store.py --run-id demo-2026-001
```

The example produces a small CSV, uploads it and a JSON manifest, and checks
remote object size and SHA-256 metadata with HEAD requests. It is an integration
smoke test, not a solver benchmark or physical validation. The sample CSV should
be replaced by a real solver output in an application adapter.

To also register the manifest with TensorPlatform, set
`TENSORPLATFORM_URL`, `TENSORPLATFORM_API_TOKEN`, and
`TENSORPLATFORM_ORGANIZATION_ID`, then pass `--register-platform`. Registration
sends compact manifest metadata and S3 artifact references only; it does not
upload bytes to TensorPlatform. The API token needs the platform's
`datasets:upload` permission for the target organization.

## Integrity and operational notes

- Object keys are immutable by convention: a repeated identical write is
  idempotent; different content at the same key is rejected.
- The client records SHA-256 in object metadata and checks it with the byte
  count. HEAD metadata verification is not a full re-download of object bytes;
  storage policy, versioning, retention and server-side encryption should be
  configured in the object store.
- The CI suite uses a fake S3 client; it does not prove compatibility with a
  live AWS or third-party endpoint. Test against the intended service before
  production use.
- Keep manifests compact. Store field arrays, trajectories and other large
  payloads as artifacts and reference them by URI.
