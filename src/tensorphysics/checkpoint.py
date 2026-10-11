"""Versioned, atomic JSON checkpoints for solver-neutral state payloads."""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

from .config import SimulationConfig
from .protocols import SolverState


class CheckpointError(RuntimeError):
    """A checkpoint cannot be written or does not match the supported schema."""


@dataclass(frozen=True)
class Checkpoint:
    """A complete orchestration envelope plus solver-owned payloads."""

    step: int
    time: float
    config: SimulationConfig
    solvers: Mapping[str, SolverState]
    schema_version: int = 1


class CheckpointStore:
    """Read/write checkpoints atomically without depending on solver backends.

    Solver payloads must be JSON-compatible. Arrays/tensors should be written by
    solver-specific serializers and referenced from a small JSON payload rather
    than silently converted by the orchestration layer.
    """

    SCHEMA_VERSION = 1

    @classmethod
    def save(cls, path: str | os.PathLike[str], *, config: SimulationConfig,
             step: int, time: float, states: Mapping[str, SolverState]) -> Path:
        destination = Path(path)
        if step < 0 or time < 0:
            raise ValueError("step and time must be non-negative")
        if not isinstance(states, Mapping):
            raise TypeError("states must be a mapping of solver name to SolverState")
        serialized_states: dict[str, dict[str, Any]] = {}
        for name, state in states.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("solver state keys must be non-empty strings")
            if not isinstance(state, SolverState):
                raise TypeError(f"state for {name!r} must be a SolverState")
            if name != state.solver_name:
                raise ValueError(f"state key {name!r} does not match solver name {state.solver_name!r}")
            if state.step != step:
                raise ValueError(f"solver {name!r} snapshot step does not match checkpoint step")
            if abs(state.time - time) > config.time_tolerance:
                raise ValueError(f"solver {name!r} snapshot time does not match checkpoint time")
            serialized_states[name] = {
                "solver_name": state.solver_name,
                "step": state.step,
                "time": state.time,
                "schema_version": state.schema_version,
                "payload": dict(state.payload),
            }
        document = {
            "schema_version": cls.SCHEMA_VERSION,
            "step": step,
            "time": time,
            "config": {
                "dt": config.dt,
                "end_time": config.end_time,
                "max_steps": config.max_steps,
                "time_tolerance": config.time_tolerance,
            },
            "solvers": serialized_states,
        }
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary_name = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=destination.parent,
                prefix=f".{destination.name}.", suffix=".tmp", delete=False,
            ) as stream:
                temporary_name = stream.name
                json.dump(document, stream, ensure_ascii=False, sort_keys=True, allow_nan=False)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_name, destination)
            temporary_name = None
        except (TypeError, ValueError, OSError) as exc:
            raise CheckpointError(f"failed to write checkpoint {destination}: {exc}") from exc
        finally:
            if temporary_name is not None:
                try:
                    os.unlink(temporary_name)
                except OSError:
                    pass
        return destination

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> Checkpoint:
        source = Path(path)
        try:
            with source.open("r", encoding="utf-8") as stream:
                document = json.load(stream)
        except (OSError, json.JSONDecodeError) as exc:
            raise CheckpointError(f"failed to read checkpoint {source}: {exc}") from exc
        if not isinstance(document, dict):
            raise CheckpointError("checkpoint root must be a JSON object")
        if document.get("schema_version") != cls.SCHEMA_VERSION:
            raise CheckpointError(
                f"unsupported checkpoint schema version: {document.get('schema_version')!r}"
            )
        try:
            raw_config = document["config"]
            config = SimulationConfig(
                dt=raw_config["dt"], end_time=raw_config["end_time"],
                max_steps=raw_config["max_steps"],
                time_tolerance=raw_config["time_tolerance"],
            )
            step, time = document["step"], document["time"]
            if not isinstance(step, int) or isinstance(step, bool) or step < 0:
                raise ValueError("step must be a non-negative integer")
            if not isinstance(time, (int, float)) or isinstance(time, bool) or time < 0:
                raise ValueError("time must be a non-negative number")
            raw_solvers = document["solvers"]
            if not isinstance(raw_solvers, dict):
                raise ValueError("solvers must be a JSON object")
            states: dict[str, SolverState] = {}
            for name, raw in raw_solvers.items():
                if not isinstance(raw, dict):
                    raise ValueError(f"solver state {name!r} must be a JSON object")
                state = SolverState(
                    solver_name=raw["solver_name"], step=raw["step"], time=raw["time"],
                    payload=raw["payload"], schema_version=raw["schema_version"],
                )
                if name != state.solver_name:
                    raise ValueError(f"solver key {name!r} does not match state name")
                if state.step != step or abs(state.time - time) > config.time_tolerance:
                    raise ValueError(f"solver state {name!r} time does not match checkpoint clock")
                if not isinstance(state.payload, dict):
                    raise ValueError(f"solver payload for {name!r} must be a JSON object")
                states[name] = state
        except (KeyError, TypeError, ValueError) as exc:
            raise CheckpointError(f"invalid checkpoint structure: {exc}") from exc
        return Checkpoint(step=step, time=float(time), config=config, solvers=states)
