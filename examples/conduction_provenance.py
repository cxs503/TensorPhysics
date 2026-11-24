"""Report the revision and exact source bytes used by the conduction examples."""
import hashlib
from pathlib import Path
import subprocess


def provenance(filename):
    path = Path(filename).resolve()
    commit = subprocess.check_output(
        ["git", "-C", str(path.parent), "rev-parse", "HEAD"], text=True
    ).strip()
    return {"commit": commit, "file": path.name,
            "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
