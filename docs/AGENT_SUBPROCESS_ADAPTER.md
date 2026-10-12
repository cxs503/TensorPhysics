# Subprocess Agent adapter

`SubprocessAgentBackend` is the first executable bridge from the research runner to
local CLI-based Agent engines. It is deliberately generic: Codex, Claude Code, or another
tool can be configured by providing the exact command-line argument vector appropriate
for the installed CLI version. The adapter never invokes a shell.

## Example

```python
from tensorphysics.agent_subprocess import SubprocessAgentBackend

agent = SubprocessAgentBackend(
    name="local-agent",
    argv=["my-agent-cli", "--prompt", "{prompt}", "--workspace", "{workspace}"],
    # Inherit auth environment by default. For production, prefer a minimal allowlist.
    inherit_env=False,
    env={"AGENT_API_KEY": "..."},
    artifact_dir="artifacts/agent-logs",
)
```

Supported per-argument placeholders: `{prompt}`, `{task_id}`, `{workspace}`,
and `{timeout_seconds}`. The prompt contains the task goal, plan, allowed-tool intent,
and workspace instructions as JSON. Since `shell=False`, shell metacharacters in values
are passed as data rather than expanded by a shell.

## Result semantics

- Exit code 0: `outcome="completed"`; independent validation is still required.
- Non-zero exit, launch error, or timeout: `outcome="failed"`.
- Standard output/error are bounded in returned metadata. If `artifact_dir` is set,
  full stdout and stderr are saved as per-task log artifacts.
- The process runs with its working directory set to the existing task workspace.

## Security boundary

This adapter does **not** provide an OS sandbox. The `allowed_tools` field expresses
task intent to the Agent; it is not an operating-system enforcement mechanism. A production
black-box research factory must run the process inside a disposable container, VM, or
scheduler-managed job with filesystem, network, CPU, memory, GPU, and wall-time limits.
Use `inherit_env=False` with a minimal credential environment where feasible. Never
put real secrets in source code, plans, prompts, or committed examples.
