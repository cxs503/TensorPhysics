"""Safe-by-default subprocess adapter for local Agent CLIs.

Commands are configured as argv arrays and are never interpreted by a shell.
This adapter is a process boundary, not an OS sandbox; use a container or job
scheduler for untrusted work and resource isolation.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import time
from typing import Mapping, Sequence

from .research_runtime import AgentRequest, AgentResult


_MAX_CAPTURE_CHARS = 12_000


def _clip(value: str, limit: int = _MAX_CAPTURE_CHARS) -> str:
    if len(value) <= limit:
        return value
    return value[:limit] + "\n...[output truncated]"


class SubprocessAgentBackend:
    """Run an explicitly configured CLI command without shell expansion.

    Supported argv placeholders are `{prompt}`, `{task_id}`, `{workspace}`,
    and `{timeout_seconds}`. Each argv element is formatted independently.
    The generated prompt includes the goal, plan, and allowed-tool intent. A CLI
    adapter must still configure the underlying product's own permission mode.

    By default the process inherits the current environment so authenticated
    local CLIs work. Set `inherit_env=False` and pass a minimal `env` mapping
    for tighter credential exposure.
    """

    def __init__(
        self,
        name: str,
        argv: Sequence[str],
        *,
        env: Mapping[str, str] | None = None,
        inherit_env: bool = True,
        artifact_dir: str | Path | None = None,
        max_output_chars: int = _MAX_CAPTURE_CHARS,
    ) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("name must be a non-empty string")
        if not argv or any(not isinstance(arg, str) or not arg for arg in argv):
            raise ValueError("argv must be a non-empty sequence of non-empty strings")
        if not isinstance(inherit_env, bool):
            raise ValueError("inherit_env must be a boolean")
        if not isinstance(max_output_chars, int) or isinstance(max_output_chars, bool) or max_output_chars < 256:
            raise ValueError("max_output_chars must be an integer >= 256")
        if env is not None and any(
            not isinstance(k, str) or not k or not isinstance(v, str)
            for k, v in env.items()
        ):
            raise ValueError("env must map non-empty string keys to string values")
        self._name = name.strip()
        self._argv = tuple(argv)
        self._env = dict(env or {})
        self._inherit_env = inherit_env
        self._artifact_dir = Path(artifact_dir).expanduser() if artifact_dir is not None else None
        self._max_output_chars = max_output_chars

    @property
    def name(self) -> str:
        return self._name

    def execute(self, request: AgentRequest) -> AgentResult:
        if request.timeout_seconds <= 0:
            raise ValueError("request.timeout_seconds must be positive")
        workspace = Path(request.workspace).expanduser().resolve()
        if not workspace.is_dir():
            return AgentResult(
                outcome="failed",
                summary=f"Workspace is not an existing directory: {workspace}",
                metadata={"timed_out": False, "shell": False},
            )

        prompt = json.dumps(
            {
                "task_id": request.task_id,
                "goal": dict(request.goal),
                "plan": dict(request.plan),
                "allowed_tools": list(request.allowed_tools),
                "instruction": (
                    "Work only in the provided workspace. Return a concise summary "
                    "of changes, commands run, test outcomes, and artifact paths. "
                    "Do not claim scientific validity without independent evidence."
                ),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        replacements = {
            "prompt": prompt,
            "task_id": request.task_id,
            "workspace": str(workspace),
            "timeout_seconds": str(request.timeout_seconds),
        }
        try:
            argv = [part.format_map(replacements) for part in self._argv]
        except (KeyError, ValueError) as exc:
            raise ValueError(f"invalid argv placeholder: {exc}") from exc

        env = dict(os.environ) if self._inherit_env else {}
        env.update(self._env)
        started = time.monotonic()
        try:
            completed = subprocess.run(
                argv,
                cwd=str(workspace),
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=request.timeout_seconds,
                check=False,
                shell=False,
            )
            elapsed = round(time.monotonic() - started, 3)
            stdout = completed.stdout
            stderr = completed.stderr
            artifact_uris = self._persist_logs(request.task_id, stdout, stderr)
            metadata = {
                "returncode": completed.returncode,
                "elapsed_seconds": elapsed,
                "stdout": _clip(stdout, self._max_output_chars),
                "stderr": _clip(stderr, self._max_output_chars),
                "timed_out": False,
                "shell": False,
            }
            if completed.returncode == 0:
                summary = stdout.strip() or "Agent CLI exited successfully without stdout."
                return AgentResult(
                    outcome="completed",
                    summary=_clip(summary, self._max_output_chars),
                    artifact_uris=artifact_uris,
                    metadata=metadata,
                )
            summary = stderr.strip() or stdout.strip() or f"Agent CLI exited with code {completed.returncode}."
            return AgentResult(
                outcome="failed",
                summary=_clip(summary, self._max_output_chars),
                artifact_uris=artifact_uris,
                metadata=metadata,
            )
        except subprocess.TimeoutExpired as exc:
            elapsed = round(time.monotonic() - started, 3)
            stdout = self._decode(exc.stdout)
            stderr = self._decode(exc.stderr)
            artifact_uris = self._persist_logs(request.task_id, stdout, stderr)
            return AgentResult(
                outcome="failed",
                summary=f"Agent CLI timed out after {request.timeout_seconds:g} seconds.",
                artifact_uris=artifact_uris,
                metadata={
                    "elapsed_seconds": elapsed,
                    "stdout": _clip(stdout, self._max_output_chars),
                    "stderr": _clip(stderr, self._max_output_chars),
                    "timed_out": True,
                    "shell": False,
                },
            )
        except OSError as exc:
            return AgentResult(
                outcome="failed",
                summary=f"Could not start Agent CLI ({type(exc).__name__}): {exc}",
                metadata={"timed_out": False, "shell": False},
            )

    def _persist_logs(self, task_id: str, stdout: str, stderr: str) -> tuple[str, ...]:
        if self._artifact_dir is None:
            return ()
        safe_id = re.sub(r"[^A-Za-z0-9_.-]", "_", task_id)[:100] or "task"
        directory = self._artifact_dir / safe_id
        directory.mkdir(parents=True, exist_ok=True)
        paths = []
        for filename, content in (("stdout.log", stdout), ("stderr.log", stderr)):
            path = directory / filename
            path.write_text(content, encoding="utf-8")
            paths.append(path.resolve().as_uri())
        return tuple(paths)

    @staticmethod
    def _decode(value: str | bytes | None) -> str:
        if value is None:
            return ""
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return value
