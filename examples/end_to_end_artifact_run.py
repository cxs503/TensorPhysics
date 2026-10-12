"""Run a deterministic toy workflow and write a provenance manifest.

This is an orchestration/artifact smoke example, not a physical solver benchmark.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from tensorphysics import (
    ArtifactRef,
    RunManifest,
    RunStatus,
    Simulation,
    SimulationConfig,
    SolverState,
    ValidationStatus,
    TensorPlatformRunRegistry,
)


@dataclass
class ToySolver:
    """Tiny deterministic solver used to exercise orchestration only."""

    name: str = "toy"
    value: float = 0.0

    def initialize(self, context) -> None:
        self.value = 0.0

    def advance(self, context) -> None:
        self.value += context.dt

    def finalize(self, context) -> None:
        pass

    def snapshot(self, context) -> SolverState:
        return SolverState(
            self.name,
            context.step,
            context.time,
            {"accumulated_dt": self.value},
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_example(output_dir: Path, *, artifact_uri: str | None = None) -> Path:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = {"solver": "toy", "dt": 0.1, "end_time": 1.0, "purpose": "orchestration-smoke"}
    solver = ToySolver()
    simulation = Simulation(
        SimulationConfig(dt=config["dt"], end_time=config["end_time"]),
        [solver],
    )
    try:
        simulation.run()
        if simulation.state.value != "finished":
            raise RuntimeError(f"toy simulation did not finish: {simulation.state.value}")

        output_path = output_dir / "toy_history.csv"
        snapshot = simulation.snapshots()[solver.name]
        with output_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["step", "time", "accumulated_dt"])
            writer.writerow([
                snapshot.step,
                f"{snapshot.time:.12g}",
                f"{snapshot.payload.get('accumulated_dt', 0.0):.12g}",
            ])

        artifact = ArtifactRef(
            uri=artifact_uri or output_path.as_uri(),
            media_type="text/csv",
            checksum_sha256=sha256_file(output_path),
            size_bytes=output_path.stat().st_size,
            schema_name="tensorphysics.toy_history",
            schema_version=1,
            metadata={"units": {"time": "s", "accumulated_dt": "s"}},
        )
        manifest = RunManifest(
            run_id="toy-smoke-001",
            status=RunStatus.SUCCEEDED,
            created_at=datetime.now(timezone.utc).isoformat(),
            config=config,
            config_sha256=RunManifest.digest_config(config),
            code_revisions={"cxs503/TensorPhysics": "working-tree"},
            environment={"python": platform.python_version(), "platform": platform.platform()},
            resources={"execution": "local", "device": "cpu"},
            outputs=[artifact],
            # Successful execution is not a physics-validation claim.
            validation_status=ValidationStatus.NOT_RUN,
        )
        manifest_path = output_dir / "run_manifest.json"
        manifest_path.write_text(manifest.to_json() + "\n", encoding="utf-8")
        return manifest_path
    finally:
        simulation.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/end_to_end_smoke"))
    parser.add_argument(
        "--register-platform",
        action="store_true",
        help="register the manifest after generating the artifact",
    )
    parser.add_argument(
        "--artifact-uri",
        help="shared URI of the already-uploaded artifact; required for remote registration",
    )
    args = parser.parse_args()
    if args.register_platform and not args.artifact_uri:
        parser.error("--register-platform requires --artifact-uri pointing to the same artifact in shared storage")
    if args.register_platform and args.artifact_uri.startswith("file://"):
        parser.error("--artifact-uri must be accessible to the platform, not a local file:// URI")
    manifest_path = run_example(args.output, artifact_uri=args.artifact_uri)
    print(f"manifest={manifest_path}")
    print(f"artifact={manifest_path.parent / 'toy_history.csv'}")
    if args.register_platform:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        result = TensorPlatformRunRegistry.from_env().register(manifest)
        print(f"platform_registration={json.dumps(result, sort_keys=True)}")


if __name__ == "__main__":
    main()
