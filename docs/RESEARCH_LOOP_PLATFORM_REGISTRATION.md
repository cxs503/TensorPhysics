# S3-backed TensorDEM research loop and TensorPlatform registration

This is the executable integration path between the existing TensorPhysics research loop and TensorPlatform's run registry. It adds no separate Harness repository.

## What the command does

1. Invokes the existing bounded TensorDEM demo (examples/research_loop_dem.py).
2. Independently checks solver exit status, finite history values, trajectory readability, finite positions and monotonic time.
3. Uploads the actual solver history, trajectory (when produced), and validation report to an S3-compatible artifact store.
4. Computes SHA-256 and byte size from each local artifact, verifies the stored object metadata, and constructs the TensorPlatform manifest-v1 contract.
5. Registers the compact manifest using the tenant-scoped API and retrieves it back to verify exact manifest equality.
6. Preserves failed runs and validation failures as failed/unvalidated records instead of treating process completion as scientific acceptance.

The S3 store uses the standard boto3 credential provider chain. Tokens and credentials are supplied through environment variables; none are written into source code or the manifest.

## Configuration

Install the optional dependencies and TensorDEM in the same Python environment:

```bash
python -m pip install -e '.[s3,test]'
python -m pip install 'git+https://github.com/cxs503/TensorDEM.git'
```

Set the S3 and TensorPlatform configuration:

```bash
export TENSORPHYSICS_S3_BUCKET=engineering-artifacts
export TENSORPHYSICS_S3_PREFIX=tensorphysics-runs
export AWS_REGION=ap-southeast-1
export TENSORPLATFORM_URL=https://your-platform.example
export TENSORPLATFORM_API_TOKEN=...
export TENSORPLATFORM_ORGANIZATION_ID=7
```

Use an API token authorized for datasets:upload and datasets:read in the target organization. For S3-compatible services, set TENSORPHYSICS_S3_ENDPOINT_URL.

Run:

```bash
python examples/research_loop_dem_platform.py \
  --run-id dem-loop-2026-001 --nx 7 --ny 3 --steps 20
```

The run is written under outputs/research-loop-dem-platform/, including demo_result.json, validation_report.json, run_manifest.json, research_memory.json, and platform_registration_result.json.

## Acceptance boundary

- Local unit tests can validate manifest mapping, request construction, checksum metadata, and retrieval behavior without network credentials.
- Running this command against a real S3 service and deployed TensorPlatform is an integration test; it requires the deployment, identity, and credentials above.
- S3 HEAD metadata checks verify object presence, declared size and stored SHA-256 metadata. They are not a full re-download and recomputation of remote object bytes.
- A validation_status=passed here means the deterministic output-integrity checks passed. It is not a claim of physical accuracy, benchmark agreement, convergence, fracture-law validity, or icebreaker resistance accuracy.


## Bounded platform-registered campaign with lineage

For a two-run, fail-closed integration campaign:

```bash
python examples/research_campaign_dem_platform.py \
  --campaign-id ice-dem-smoke-001 --nx 7 --ny 3 --steps 20
```

The campaign registers the baseline run first. Only if its deterministic integrity gate passes **and** TensorPlatform read-back matches does the controller launch the second run (`nx + 1`). The second manifest records the first run ID in `parent_run_ids`. Any failed validation or failed read-back stops the campaign immediately. The report is written to `outputs/research-campaign-dem-platform/<campaign-id>/campaign_report.json`.

This two-point variation exists to test execution, lineage, artifact persistence, and fail-closed control. It is explicitly **not** a meaningful physical sensitivity study and must not be used to claim an optimal particle count or icebreaking result.
