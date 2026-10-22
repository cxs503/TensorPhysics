# Persistent TensorPhysics research worker

## Why this exists

The task ledger and single-attempt runner were already present, but they did not
by themselves create a process that continuously polls the ledger. This worker
is a separate long-running process: it survives the end of an interactive
Claude Code/Codex terminal session, records task transitions in SQLite, retries
within a bounded attempt budget, recovers interrupted in-flight tasks on startup,
and only records `succeeded` when an independent validator returns a valid
passing report.

It is **not** an Agent model, solver, or scientific validation algorithm. A
configured Agent CLI and a domain-appropriate validator must be supplied by the
deployment. The worker intentionally fails closed when either command is absent
or the validator does not provide structured evidence.

## Runtime contract

- Persistent task state: SQLite database on a durable local filesystem.
- Workspaces: one directory per task below the configured workspace root.
- Agent process: configured as a JSON argv array; no shell expansion is used.
- Validation: a separate command receives a JSON payload on stdin and must
  emit a JSON object on stdout, with `status` equal to `passed`, `failed`, or
  `incomplete`, and a non-empty `checks` list.
- Retry: task attempts are bounded by `max_attempts`; failed validation is not
  marked successful. Human-input tasks remain in `waiting_input`.
- Recovery: tasks left in `claimed`, `running`, or `verifying` after a worker
  interruption are moved to `retry_wait` or `failed` according to their budget.
- Idempotency: `task_id` is the primary key. Re-submitting an existing ID is
  rejected rather than silently duplicating work.
- Isolation: argv is executed without a shell, but this is not an OS sandbox.
  Use a dedicated service account, CLI-native permission restrictions, and
  preferably a container/job scheduler for untrusted agent work.

## Install on a Linux deployment host

Install this repository into a dedicated virtual environment on the host where
the selected Agent CLI is installed and authenticated. This avoids pretending
that a CLI installed in an interactive shell automatically exists inside a
container.

```bash
sudo useradd --system --create-home --shell /usr/sbin/nologin tensorphysics
sudo mkdir -p /opt/tensorphysics /var/lib/tensorphysics/workspaces /etc/tensorphysics
sudo chown -R tensorphysics:tensorphysics /opt/tensorphysics /var/lib/tensorphysics
# Clone this repository into /opt/tensorphysics as the service deployment user.
cd /opt/tensorphysics
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install .
```

Copy `deploy/systemd/tensorphysics-worker.service` to
`/etc/systemd/system/tensorphysics-worker.service`. Configure
`/etc/tensorphysics/worker.env` with the command arrays and deployment-specific
settings. Do not put secrets into Git.

Example shape (replace both commands with commands that really exist on the
host; the validator must be your actual independent validator):

```dotenv
TENSORPHYSICS_AGENT_NAME=claude-code
TENSORPHYSICS_AGENT_ARGV_JSON=["claude","-p","{prompt}"]
TENSORPHYSICS_AGENT_INHERIT_ENV=true
TENSORPHYSICS_VALIDATOR_ARGV_JSON=["/opt/tensorphysics/.venv/bin/python","/opt/tensorphysics-validator/validate.py"]
TENSORPHYSICS_POLL_SECONDS=5
TENSORPHYSICS_TASK_TIMEOUT_SECONDS=3600
TENSORPHYSICS_VALIDATOR_TIMEOUT_SECONDS=300
TENSORPHYSICS_ALLOWED_TOOLS=read_file,run_tests
LOG_LEVEL=INFO
```

The Agent CLI's own non-interactive/authentication/permission behavior varies
by vendor and version. Verify the exact command and credentials under the
service account before enabling the service. The allowed-tools field is context
for the Agent, **not** an enforcement boundary; configure native CLI permissions
and operating-system isolation separately.

Enable the service:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now tensorphysics-worker
sudo systemctl status tensorphysics-worker
sudo journalctl -u tensorphysics-worker -f
```

The systemd unit uses automatic restart. The SQLite database and workspace root
must remain on persistent storage and be backed up according to local policy.

## Submit a task

Write a plan JSON file. It must include either a `goal` object or a non-empty
`objective`, because the worker does not invent a research objective.

```json
{
  "objective": "Run the selected TensorDEM benchmark and report its evidence",
  "solver": "TensorDEM",
  "parameters": {"case": "replace-with-a-real-case"},
  "acceptance_criteria": ["replace-with-domain-specific-criteria"],
  "validation_checks": ["replace-with-independent-checks"]
}
```

Save as `plan.json`, then enqueue it using the same Python environment and
database path as the service:

```bash
sudo -u tensorphysics /opt/tensorphysics/.venv/bin/python -m tensorphysics.worker submit \
  --database /var/lib/tensorphysics/research.sqlite3 \
  --task-id dem-benchmark-001 --goal-id ice-demo --plan /path/to/plan.json \
  --max-attempts 3
```

Inspect status and the append-only event history:

```bash
sudo -u tensorphysics /opt/tensorphysics/.venv/bin/python -m tensorphysics.worker status \
  --database /var/lib/tensorphysics/research.sqlite3 --task-id dem-benchmark-001
```

## Validator interface

The configured validator receives one JSON object on stdin:

- `schema_version`, `task_id`, `goal`, and `plan`
- `agent_result.outcome`, `summary`, `artifact_uris`, and `metadata`

It must independently inspect actual output artifacts, execute applicable
integrity/physics/benchmark checks, and print one JSON report. A passing report
must contain a non-empty `checks` list, each check an object with
`{"name": "...", "passed": true}`. The adapter rejects a passing report if a
check is missing or failed. Do not use an Agent-authored claim or a fixture-only
validator as scientific evidence. The adapter can check report structure; it
cannot prove that a validator's algorithm is scientifically adequate.

## Smoke test

Run the unit tests and CLI help before enabling the daemon:

```bash
python -m pip install -e ".[test]"
python -m pytest tests/test_worker.py tests/test_research_runtime.py
python -m tensorphysics.worker --help
python -m tensorphysics.worker worker --help
```

These tests verify worker contracts and failure behavior; they do not run a
production Agent or establish scientific correctness. After deployment, verify
the actual service process, queue state, task event history, logs, and a real
solver benchmark independently.
